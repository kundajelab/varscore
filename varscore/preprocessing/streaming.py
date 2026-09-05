"""Bounded-memory ingestion for occurrence-preserving variant preprocessing.

The immutable source VCF remains the round-trip source of truth. This module emits
one occurrence row per ALT, including unsupported alleles, and a separate canonical
relation for scoreable alleles. TSV input uses the same contract with one occurrence
per row.
"""

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyfaidx
import pysam

import varscore.core.io as io_utils
from varscore.core.logging import get_logger


logger = get_logger(__name__)

CANONICALIZER_VERSION = "lava-vcf-v1"
GVCF_UNSUPPORTED_ERROR_CODE = "UNSUPPORTED_GVCF"

OCCURRENCE_COLUMNS = [
    "record_ordinal",
    "alt_index",
    "original_chrom",
    "original_pos",
    "original_ref",
    "original_alt",
    "source_variant_id",
    "canonical_variant_id",
    "status",
    "error_code",
    "canonicalizer_version",
]
CANONICAL_COLUMNS = [
    "canonical_variant_id",
    "genome",
    "chr",
    "pos",
    "ref",
    "alt",
]
INVALID_COLUMNS = OCCURRENCE_COLUMNS + ["error_message"]

OCCURRENCE_ARROW_SCHEMA = pa.schema(
    [
        ("record_ordinal", pa.int64()),
        ("alt_index", pa.int64()),
        ("original_chrom", pa.string()),
        ("original_pos", pa.int64()),
        ("original_ref", pa.string()),
        ("original_alt", pa.string()),
        ("source_variant_id", pa.string()),
        ("canonical_variant_id", pa.string()),
        ("status", pa.string()),
        ("error_code", pa.string()),
        ("canonicalizer_version", pa.string()),
    ]
)
CANONICAL_ARROW_SCHEMA = pa.schema(
    [
        ("canonical_variant_id", pa.string()),
        ("genome", pa.string()),
        ("chr", pa.string()),
        ("pos", pa.int64()),
        ("ref", pa.string()),
        ("alt", pa.string()),
    ]
)
INVALID_ARROW_SCHEMA = pa.schema(
    list(OCCURRENCE_ARROW_SCHEMA) + [("error_message", pa.string())]
)

_GZIP_SUFFIXES = (".gz", ".bgz")
_VCF_SUFFIXES = (".vcf", ".gvcf")
_GVCF_BLOCK_HEADER = re.compile(r"^##GVCFBlock", re.MULTILINE | re.IGNORECASE)
_NON_REF_HEADER = re.compile(r"^##ALT=<ID=NON_REF(?:,|>)", re.MULTILINE | re.IGNORECASE)


class VariantIngestError(ValueError):
    """A stable, user-actionable failure raised before an ingest is published."""

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


@dataclass(frozen=True)
class ShardMetadata:
    """Integrity metadata for one atomically published Parquet shard."""

    uri: str
    row_count: int
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class OccurrenceBatch:
    """One bounded occurrence frame plus the cumulative source-record count."""

    rows: pd.DataFrame
    records_seen: int


@dataclass(frozen=True)
class StreamingIngestResult:
    """Counts and published artifacts from a completed streaming ingest."""

    source_format: str
    input_size_bytes: int
    input_sha256: str
    header_sha256: Optional[str]
    record_count: int
    alt_occurrence_count: int
    valid_occurrence_count: int
    invalid_occurrence_count: int
    unsupported_occurrence_count: int
    unique_canonical_variants: int
    duplicate_canonical_occurrences: int
    occurrence_shards: Sequence[ShardMetadata]
    canonical_shards: Sequence[ShardMetadata]
    invalid_shards: Sequence[ShardMetadata]


def resolve_input_format(path: str, fmt: str = "auto") -> str:
    """Resolve an explicit or extension-derived input format."""
    if fmt in {"vcf", "tsv"}:
        return fmt
    if fmt != "auto":
        raise VariantIngestError(
            "UNKNOWN_FORMAT", "Expected input format auto, tsv, or vcf."
        )

    base = path
    if base.lower().endswith(_GZIP_SUFFIXES):
        base = base.rsplit(".", 1)[0]
    if base.lower().endswith(".bcf"):
        raise VariantIngestError(
            "UNSUPPORTED_BCF", "BCF is unsupported; convert it to .vcf.gz first."
        )
    if base.lower().endswith(_VCF_SUFFIXES):
        return "vcf"
    return "tsv"


def inspect_vcf_header(path: str) -> str:
    """Parse and validate a VCF header before any output sink is allocated."""
    if _is_gvcf_extension(path):
        raise VariantIngestError(
            GVCF_UNSUPPORTED_ERROR_CODE, "gVCF input is not supported (file extension)."
        )
    try:
        with pysam.VariantFile(path) as source:
            header_text = str(source.header)
    except (OSError, ValueError) as exc:
        raise VariantIngestError(
            "MALFORMED_VCF", "The VCF header could not be parsed by htslib."
        ) from exc

    if _GVCF_BLOCK_HEADER.search(header_text):
        raise VariantIngestError(
            GVCF_UNSUPPORTED_ERROR_CODE,
            "gVCF input is not supported (GVCFBlock declaration found).",
        )
    if _NON_REF_HEADER.search(header_text):
        raise VariantIngestError(
            GVCF_UNSUPPORTED_ERROR_CODE,
            "gVCF input is not supported (NON_REF ALT declaration found).",
        )
    return header_text


def iter_input_occurrence_batches(
    path: str,
    fmt: str = "auto",
    batch_rows: int = 250_000,
) -> Iterator[OccurrenceBatch]:
    """Yield occurrence batches without retaining rows from earlier batches."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    source_format = resolve_input_format(path, fmt)
    if source_format == "vcf":
        yield from _iter_vcf_occurrence_batches(path, batch_rows)
    else:
        yield from _iter_tsv_occurrence_batches(path, batch_rows)


def stream_validate_variants(
    variants_loc: str,
    genome_loc: str,
    valid_out_path: str,
    invalid_out_path: str,
    occurrence_out_dir: str,
    canonical_out_dir: str,
    invalid_parquet_out_dir: str,
    manifest_out_path: str,
    header_out_path: Optional[str] = None,
    width: int = 2114,
    batch_size: int = 250_000,
    fmt: str = "auto",
    reference_build: str = "GRCh38",
) -> StreamingIngestResult:
    """Validate batches and atomically publish occurrence and canonical shards."""
    source_format = resolve_input_format(variants_loc, fmt)
    header_text = inspect_vcf_header(variants_loc) if source_format == "vcf" else None
    input_size_bytes = Path(variants_loc).stat().st_size
    input_sha256 = _sha256_file(Path(variants_loc))
    run_fingerprint = _sha256_text(
        ":".join(
            [
                input_sha256,
                CANONICALIZER_VERSION,
                reference_build,
                str(width),
                str(batch_size),
            ]
        )
    )

    output_paths = [
        valid_out_path,
        invalid_out_path,
        occurrence_out_dir,
        canonical_out_dir,
        invalid_parquet_out_dir,
        manifest_out_path,
    ]
    if header_out_path:
        output_paths.append(header_out_path)
    for output_path in output_paths:
        parent = (
            Path(output_path)
            if output_path.endswith(os.sep)
            else Path(output_path).parent
        )
        parent.mkdir(parents=True, exist_ok=True)
    Path(occurrence_out_dir).mkdir(parents=True, exist_ok=True)
    Path(canonical_out_dir).mkdir(parents=True, exist_ok=True)
    Path(invalid_parquet_out_dir).mkdir(parents=True, exist_ok=True)

    if header_text is not None and header_out_path:
        _atomic_write_text(Path(header_out_path), header_text)

    occurrence_shards: List[ShardMetadata] = []
    canonical_shards: List[ShardMetadata] = []
    invalid_shards: List[ShardMetadata] = []
    record_count = 0
    alt_occurrence_count = 0
    valid_occurrence_count = 0
    invalid_occurrence_count = 0
    unsupported_occurrence_count = 0
    first_invalid_batch = True

    Path(valid_out_path).write_text("")
    Path(invalid_out_path).write_text("")

    with pyfaidx.Fasta(genome_loc) as genome:
        for shard_index, occurrence_batch in enumerate(
            iter_input_occurrence_batches(variants_loc, source_format, batch_size)
        ):
            record_count = max(record_count, occurrence_batch.records_seen)
            if occurrence_batch.rows.empty:
                continue
            validated, valid_rows, invalid_rows = validate_occurrence_batch(
                occurrence_batch.rows,
                genome,
                reference_build=reference_build,
                width=width,
            )
            alt_occurrence_count += len(validated)
            valid_occurrence_count += len(valid_rows)
            unsupported_occurrence_count += int(
                (validated["status"] == "UNSUPPORTED").sum()
            )
            invalid_occurrence_count += int(
                validated["status"].isin(["REF_MISMATCH", "FAILED"]).sum()
            )

            occurrence_shards.append(
                _write_parquet_shard(
                    validated[OCCURRENCE_COLUMNS],
                    occurrence_out_dir,
                    shard_index,
                    OCCURRENCE_ARROW_SCHEMA,
                    run_fingerprint,
                )
            )
            if not valid_rows.empty:
                canonical = valid_rows[CANONICAL_COLUMNS].drop_duplicates(
                    "canonical_variant_id"
                )
                canonical_shards.append(
                    _write_parquet_shard(
                        canonical,
                        canonical_out_dir,
                        shard_index,
                        CANONICAL_ARROW_SCHEMA,
                        run_fingerprint,
                    )
                )
                valid_rows[["chr", "pos", "ref", "alt", "source_variant_id"]].to_csv(
                    valid_out_path,
                    sep="\t",
                    index=False,
                    header=False,
                    mode="a",
                )
            if not invalid_rows.empty:
                invalid_shards.append(
                    _write_parquet_shard(
                        invalid_rows[INVALID_COLUMNS],
                        invalid_parquet_out_dir,
                        shard_index,
                        INVALID_ARROW_SCHEMA,
                        run_fingerprint,
                    )
                )
                invalid_rows[INVALID_COLUMNS].to_csv(
                    invalid_out_path,
                    sep="\t",
                    index=False,
                    header=first_invalid_batch,
                    mode="a",
                )
                first_invalid_batch = False

    _remove_unreferenced_shards(occurrence_out_dir, occurrence_shards)
    _remove_unreferenced_shards(canonical_out_dir, canonical_shards)
    _remove_unreferenced_shards(invalid_parquet_out_dir, invalid_shards)
    unique_canonical_variants = _count_unique_canonical(canonical_out_dir)
    result = StreamingIngestResult(
        source_format=source_format,
        input_size_bytes=input_size_bytes,
        input_sha256=input_sha256,
        header_sha256=_sha256_text(header_text) if header_text is not None else None,
        record_count=record_count,
        alt_occurrence_count=alt_occurrence_count,
        valid_occurrence_count=valid_occurrence_count,
        invalid_occurrence_count=invalid_occurrence_count,
        unsupported_occurrence_count=unsupported_occurrence_count,
        unique_canonical_variants=unique_canonical_variants,
        duplicate_canonical_occurrences=max(
            0, valid_occurrence_count - unique_canonical_variants
        ),
        occurrence_shards=tuple(occurrence_shards),
        canonical_shards=tuple(canonical_shards),
        invalid_shards=tuple(invalid_shards),
    )
    manifest = {
        "canonicalizer_version": CANONICALIZER_VERSION,
        "reference_build": reference_build,
        **asdict(result),
    }
    _atomic_write_text(
        Path(manifest_out_path), json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return result


def validate_occurrence_batch(
    occurrences: pd.DataFrame,
    genome: pyfaidx.Fasta,
    reference_build: str = "GRCh38",
    width: int = 2114,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate scoreable rows while retaining every occurrence and its outcome."""
    validated = occurrences.copy()
    valid: List[Dict[str, object]] = []
    error_messages: Dict[int, str] = {}
    for row_index, row in validated.iterrows():
        if row["status"] == "UNSUPPORTED":
            continue
        result = io_utils.validate_variant(
            str(row["original_chrom"]),
            int(row["original_pos"]) - 1,
            str(row["original_ref"]),
            str(row["original_alt"]),
            genome,
            width,
        )
        if result.is_valid:
            chrom = str(result.chro)
            canonical_id = ":".join(
                [
                    chrom,
                    str(int(row["original_pos"])),
                    str(row["original_ref"]),
                    str(row["original_alt"]),
                ]
            )
            validated.at[row_index, "canonical_variant_id"] = canonical_id
            validated.at[row_index, "status"] = "PENDING"
            valid.append(
                {
                    "canonical_variant_id": canonical_id,
                    "genome": reference_build,
                    "chr": chrom,
                    "pos": int(row["original_pos"]),
                    "ref": str(row["original_ref"]),
                    "alt": str(row["original_alt"]),
                    "source_variant_id": row["source_variant_id"],
                }
            )
            continue

        error_code = (
            result.error_reason.name
            if result.error_reason is not None
            else "VALIDATION_FAILED"
        )
        validated.at[row_index, "status"] = (
            "REF_MISMATCH" if error_code == "REF_MISMATCH" else "FAILED"
        )
        validated.at[row_index, "error_code"] = error_code
        error_messages[row_index] = result.error_message or error_code

    valid_frame = pd.DataFrame(valid)
    if valid_frame.empty:
        valid_frame = pd.DataFrame(columns=CANONICAL_COLUMNS + ["source_variant_id"])
    invalid_frame = validated[
        validated["status"].isin(["UNSUPPORTED", "REF_MISMATCH", "FAILED"])
    ].copy()
    invalid_frame["error_message"] = [
        error_messages.get(index, str(row["error_code"]))
        for index, row in invalid_frame.iterrows()
    ]
    return validated, valid_frame, invalid_frame


def _iter_vcf_occurrence_batches(
    path: str, batch_rows: int
) -> Iterator[OccurrenceBatch]:
    inspect_vcf_header(path)
    rows: List[Tuple[object, ...]] = []
    previous_coordinate: Optional[Tuple[int, int]] = None
    record_count = 0
    last_yield_record_count = 0
    try:
        with pysam.VariantFile(path) as source:
            for record_ordinal, record in enumerate(source):
                _reject_gvcf_record(record, record_ordinal)
                coordinate = (record.rid, record.pos)
                if previous_coordinate is not None and coordinate < previous_coordinate:
                    raise VariantIngestError(
                        "UNSORTED_VCF",
                        "VCF records must be coordinate sorted for augmented VCF indexing "
                        + "(record {} at {}:{}).".format(
                            record_ordinal, record.contig, record.pos
                        ),
                    )
                previous_coordinate = coordinate
                record_count = record_ordinal + 1
                for alt_index, alt in enumerate(record.alts or (), start=1):
                    error_code = _unsupported_alt_error(str(alt), record.ref)
                    rows.append(
                        (
                            record_ordinal,
                            alt_index,
                            record.contig,
                            record.pos,
                            record.ref,
                            str(alt),
                            record.id,
                            None,
                            "UNSUPPORTED" if error_code else "PENDING",
                            error_code,
                            CANONICALIZER_VERSION,
                        )
                    )
                    if len(rows) >= batch_rows:
                        yield OccurrenceBatch(
                            pd.DataFrame.from_records(rows, columns=OCCURRENCE_COLUMNS),
                            record_count,
                        )
                        last_yield_record_count = record_count
                        rows = []
    except VariantIngestError:
        raise
    except (OSError, ValueError) as exc:
        raise VariantIngestError(
            "MALFORMED_VCF", "The VCF record stream could not be parsed by htslib."
        ) from exc
    if rows:
        yield OccurrenceBatch(
            pd.DataFrame.from_records(rows, columns=OCCURRENCE_COLUMNS),
            record_count,
        )
    elif record_count > last_yield_record_count:
        yield OccurrenceBatch(pd.DataFrame(columns=OCCURRENCE_COLUMNS), record_count)
    elif record_count == 0:
        logger.warning("Input VCF contains no data records.")


def _iter_tsv_occurrence_batches(
    path: str, batch_rows: int
) -> Iterator[OccurrenceBatch]:
    rows: List[Tuple[object, ...]] = []
    opener = __import__("gzip").open if path.lower().endswith(_GZIP_SUFFIXES) else open
    with opener(path, "rt", newline="") as source:
        reader = csv.reader(source, delimiter="\t")
        for record_ordinal, fields in enumerate(reader):
            if len(fields) not in {4, 5}:
                raise VariantIngestError(
                    "MALFORMED_TSV",
                    "TSV row {} has {} columns; expected 4 or 5.".format(
                        record_ordinal + 1, len(fields)
                    ),
                )
            chrom, position, ref, alt = fields[:4]
            try:
                parsed_position = int(position)
            except ValueError as exc:
                raise VariantIngestError(
                    "MALFORMED_TSV",
                    "TSV row {} has a non-integer position.".format(record_ordinal + 1),
                ) from exc
            error_code = _unsupported_alt_error(alt, ref)
            rows.append(
                (
                    record_ordinal,
                    1,
                    chrom,
                    parsed_position,
                    ref,
                    alt,
                    fields[4] or None if len(fields) == 5 else None,
                    None,
                    "UNSUPPORTED" if error_code else "PENDING",
                    error_code,
                    CANONICALIZER_VERSION,
                )
            )
            if len(rows) >= batch_rows:
                yield OccurrenceBatch(
                    pd.DataFrame.from_records(rows, columns=OCCURRENCE_COLUMNS),
                    record_ordinal + 1,
                )
                rows = []
    if rows:
        yield OccurrenceBatch(
            pd.DataFrame.from_records(rows, columns=OCCURRENCE_COLUMNS),
            record_ordinal + 1,
        )


def _reject_gvcf_record(record: pysam.VariantRecord, record_ordinal: int) -> None:
    alts = tuple(str(alt) for alt in (record.alts or ()))
    if any(alt in {"<NON_REF>", "<*>"} for alt in alts):
        raise VariantIngestError(
            GVCF_UNSUPPORTED_ERROR_CODE,
            "gVCF input is not supported (reference-block ALT at record {}, {}:{}).".format(
                record_ordinal,
                record.contig,
                record.pos,
            ),
        )
    if not alts and record.stop > record.pos:
        raise VariantIngestError(
            GVCF_UNSUPPORTED_ERROR_CODE,
            "gVCF input is not supported (reference block at record {}, {}:{}).".format(
                record_ordinal,
                record.contig,
                record.pos,
            ),
        )


def _unsupported_alt_error(alt: str, ref: str) -> Optional[str]:
    if alt == "*":
        return "SPANNING_DELETION"
    if alt.startswith("<") and alt.endswith(">"):
        return "SYMBOLIC_ALT"
    if "[" in alt or "]" in alt:
        return "BREAKEND"
    if alt == ref:
        return "REF_EQUALS_ALT"
    if alt in {"", "."}:
        return "MISSING_ALT"
    return None


def _write_parquet_shard(
    frame: pd.DataFrame,
    directory: str,
    shard_index: int,
    schema: pa.Schema,
    run_fingerprint: str,
) -> ShardMetadata:
    destination = Path(directory) / "part-{:08d}.parquet".format(shard_index)
    success_marker = destination.with_suffix(".parquet.success.json")
    existing = _verified_existing_shard(
        destination, success_marker, len(frame), run_fingerprint
    )
    if existing is not None:
        return existing

    temporary = destination.with_suffix(".parquet.tmp")
    table = pa.Table.from_pandas(frame, schema=schema, preserve_index=False, safe=True)
    pq.write_table(
        table, temporary, compression="zstd", row_group_size=max(1, len(frame))
    )
    os.replace(str(temporary), str(destination))
    metadata = ShardMetadata(
        uri=destination.name,
        row_count=len(frame),
        size_bytes=destination.stat().st_size,
        sha256=_sha256_file(destination),
    )
    marker = {**asdict(metadata), "run_fingerprint": run_fingerprint}
    _atomic_write_text(success_marker, json.dumps(marker, sort_keys=True))
    return metadata


def _verified_existing_shard(
    destination: Path,
    success_marker: Path,
    expected_rows: int,
    run_fingerprint: str,
) -> Optional[ShardMetadata]:
    if not destination.exists() or not success_marker.exists():
        return None
    try:
        marker = json.loads(success_marker.read_text())
        metadata = ShardMetadata(
            uri=str(marker["uri"]),
            row_count=int(marker["row_count"]),
            size_bytes=int(marker["size_bytes"]),
            sha256=str(marker["sha256"]),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if (
        marker.get("run_fingerprint") != run_fingerprint
        or metadata.uri != destination.name
        or metadata.row_count != expected_rows
        or metadata.size_bytes != destination.stat().st_size
        or metadata.sha256 != _sha256_file(destination)
    ):
        return None
    return metadata


def _remove_unreferenced_shards(
    directory: str, expected: Sequence[ShardMetadata]
) -> None:
    expected_paths = {Path(directory) / metadata.uri for metadata in expected}
    for shard in Path(directory).glob("part-*.parquet"):
        if shard in expected_paths:
            continue
        shard.unlink()
        shard.with_suffix(".parquet.success.json").unlink(missing_ok=True)


def _count_unique_canonical(canonical_out_dir: str) -> int:
    shards = sorted(Path(canonical_out_dir).glob("part-*.parquet"))
    if not shards:
        return 0
    relation = str(Path(canonical_out_dir) / "part-*.parquet").replace("'", "''")
    with duckdb.connect() as connection:
        row = connection.execute(
            "SELECT COUNT(DISTINCT canonical_variant_id) FROM read_parquet('{}')".format(
                relation
            )
        ).fetchone()
    return int(row[0])


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_text(destination: Path, content: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(content)
    os.replace(str(temporary), str(destination))


def _is_gvcf_extension(path: str) -> bool:
    base = path.lower()
    if base.endswith(_GZIP_SUFFIXES):
        base = base.rsplit(".", 1)[0]
    return base.endswith(".gvcf")
