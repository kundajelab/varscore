"""Stream an immutable source VCF into an occurrence-aligned augmented VCF."""

import argparse
import hashlib
import json
import math
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pysam


@dataclass(frozen=True)
class InfoField:
    """One versioned LAVA INFO field."""

    name: str
    number: str
    value_type: str
    description: str


INFO_FIELDS = (
    InfoField(
        "LAVA1_STATUS",
        "A",
        "String",
        "LAVA result status per ALT: SCORED, CACHED, UNSUPPORTED, REF_MISMATCH, FAILED, or NO_RESULT",
    ),
    InfoField(
        "LAVA1_ERROR",
        "A",
        "String",
        "Machine-readable LAVA error code per ALT, or missing",
    ),
    InfoField(
        "LAVA1_PRIORITIZED",
        "A",
        "Integer",
        "LAVA prioritization decision (1 or 0) per ALT",
    ),
    InfoField(
        "LAVA1_MAX_ABS_LOGFC",
        "A",
        "Float",
        "Largest absolute ChromBPNet log fold-change per ALT",
    ),
    InfoField(
        "LAVA1_TOP_MODEL",
        "A",
        "String",
        "Short LAVA model identifier for the headline score per ALT",
    ),
    InfoField("LAVA1_REGION", "A", "String", "Most-severe LAVA region label per ALT"),
    InfoField("LAVA1_NEAREST_GENE", "A", "String", "Nearest gene symbol per ALT"),
    InfoField(
        "LAVA1_CCRE",
        "A",
        "String",
        "Candidate cis-regulatory element accession per ALT",
    ),
    InfoField(
        "LAVA1_GNOMAD_AF", "A", "Float", "Global gnomAD allele frequency per ALT"
    ),
    InfoField("LAVA1_CADD_PHRED", "A", "Float", "CADD PHRED score per ALT"),
    InfoField(
        "LAVA1_AM_SCORE", "A", "Float", "AlphaMissense pathogenicity score per ALT"
    ),
    InfoField("LAVA1_AM_CLASS", "A", "String", "AlphaMissense class per ALT"),
    InfoField("LAVA1_REVEL", "A", "Float", "REVEL score per ALT"),
    InfoField(
        "LAVA1_MODEL_SCORE",
        ".",
        "String",
        "ALT|model|logFC|JSD|active-allele-quantile|in-peak|prioritized entries",
    ),
)


class AugmentationError(ValueError):
    """A deterministic input/result alignment failure."""


def write_augmented_vcf(
    input_vcf: str,
    bucket_dir: str,
    models_json: str,
    ingest_manifest: str,
    output_vcf: str,
    output_manifest: str,
    results_parquet_uri: str,
    include_model_scores: bool = False,
) -> dict:
    """Write BGZF VCF + CSI and atomically publish a checksummed manifest."""
    output_path = Path(output_vcf)
    if output_path.suffixes[-2:] != [".vcf", ".gz"]:
        raise AugmentationError("Output path must end in .vcf.gz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(output_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    ingest = json.loads(Path(ingest_manifest).read_text())
    source_sha256 = _sha256(Path(input_vcf))
    if source_sha256 != ingest["input_sha256"]:
        raise AugmentationError("Input VCF checksum does not match the ingest manifest")
    models = _load_models(Path(models_json))
    row_iter = iter_bucket_rows(
        Path(bucket_dir), _writer_columns(models, include_model_scores)
    )
    next_row = next(row_iter, None)
    records_written = 0
    alts_written = 0

    token = uuid.uuid4().hex
    temp_output = output_path.with_name(f".{output_path.stem}.{token}.vcf.gz")
    temp_index = Path(f"{temp_output}.csi")
    try:
        with pysam.VariantFile(input_vcf) as source:
            header = source.header.copy()
            _ensure_headers(header, models)
            with pysam.VariantFile(str(temp_output), "wz", header=header) as sink:
                for record_ordinal, record in enumerate(source):
                    alt_count = len(record.alts or ())
                    annotations = []
                    for alt_index in range(1, alt_count + 1):
                        if next_row is None:
                            raise AugmentationError(
                                "Occurrence results ended before record {} ALT {}".format(
                                    record_ordinal, alt_index
                                )
                            )
                        actual = (
                            int(next_row["record_ordinal"]),
                            int(next_row["alt_index"]),
                        )
                        expected = (record_ordinal, alt_index)
                        if actual != expected:
                            raise AugmentationError(
                                "Occurrence alignment mismatch: expected {} but found {}".format(
                                    expected, actual
                                )
                            )
                        annotations.append(
                            _row_to_annotations(
                                next_row, alt_index, models, include_model_scores
                            )
                        )
                        next_row = next(row_iter, None)
                    record.translate(header)
                    _set_info(record, annotations)
                    sink.write(record)
                    records_written += 1
                    alts_written += alt_count

        if next_row is not None:
            raise AugmentationError(
                "Occurrence results continue after the final VCF record at ({}, {})".format(
                    next_row["record_ordinal"], next_row["alt_index"]
                )
            )
        if records_written != int(ingest["record_count"]):
            raise AugmentationError(
                "Record count mismatch: wrote {} but ingest recorded {}".format(
                    records_written, ingest["record_count"]
                )
            )
        if alts_written != int(ingest["alt_occurrence_count"]):
            raise AugmentationError(
                "ALT count mismatch: wrote {} but ingest recorded {}".format(
                    alts_written, ingest["alt_occurrence_count"]
                )
            )

        pysam.tabix_index(str(temp_output), preset="vcf", force=True, csi=True)
        _validate_indexed_output(temp_output, records_written, alts_written)
        os.replace(temp_output, output_path)
        os.replace(temp_index, Path(f"{output_path}.csi"))
    finally:
        temp_output.unlink(missing_ok=True)
        temp_index.unlink(missing_ok=True)

    result = {
        "schema_version": "lava-augmented-vcf-v1",
        "source_input_sha256": source_sha256,
        "record_count": records_written,
        "alt_occurrence_count": alts_written,
        "augmented_vcf": _file_metadata(output_path),
        "csi_index": _file_metadata(Path(f"{output_path}.csi")),
        "results_parquet": {"uri": results_parquet_uri},
        "include_model_scores": include_model_scores,
        "models": models,
    }
    _atomic_write(manifest_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def iter_bucket_rows(
    bucket_root: Path, requested_columns: Sequence[str]
) -> Iterator[dict]:
    """Yield rows in strict occurrence order while retaining only one bucket."""
    buckets = []
    for path in bucket_root.glob("bucket=*"):
        match = re.fullmatch(r"bucket=(\d+)", path.name)
        if path.is_dir() and match:
            buckets.append((int(match.group(1)), path))
    for _bucket_id, bucket in sorted(buckets):
        parts = sorted(bucket.glob("part-*.parquet"))
        if not parts:
            raise AugmentationError(
                f"Augmentation bucket has no Parquet parts: {bucket}"
            )
        tables = []
        for part in parts:
            parquet = pq.ParquetFile(part)
            columns = [
                name for name in requested_columns if name in parquet.schema.names
            ]
            tables.append(parquet.read(columns=columns))
        table = (
            pa.concat_tables(tables, promote_options="default")
            if len(tables) > 1
            else tables[0]
        )
        order = pc.sort_indices(
            table,
            sort_keys=[("record_ordinal", "ascending"), ("alt_index", "ascending")],
        )
        sorted_table = table.take(order)
        for batch in sorted_table.to_batches(max_chunksize=1024):
            for row in batch.to_pylist():
                yield {key: _none_if_missing(value) for key, value in row.items()}


def _writer_columns(
    models: Sequence[Mapping[str, str]], include_model_scores: bool
) -> List[str]:
    columns = [
        "record_ordinal",
        "alt_index",
        "status",
        "error_code",
        "result_variant_id",
        "prioritized",
        "most_active_celltype",
        "region_type",
        "nearest_genes",
        "ccre",
        "af_global",
        "cadd_phred",
        "am_pathogenicity",
        "am_class",
        "revel_score",
    ]
    score_names = (
        (
            "logfc",
            "jsd",
            "active_allele_quantile",
            "in_peak",
            "prioritized",
        )
        if include_model_scores
        else ("logfc",)
    )
    columns.extend(
        f"{model['column_prefix']}_{score_name}"
        for model in models
        for score_name in score_names
    )
    return columns


def _load_models(path: Path) -> List[dict]:
    raw = json.loads(path.read_text())
    models = raw["models"] if isinstance(raw, dict) else raw
    result = []
    for index, model in enumerate(models, start=1):
        result.append(
            {
                "short_id": model.get("short_id") or f"M{index:04d}",
                "model_id": str(model["model_id"]),
                "name": str(model["name"]),
                "column_prefix": str(model["column_prefix"]),
            }
        )
    return result


def _ensure_headers(
    header: pysam.VariantHeader, models: Sequence[Mapping[str, str]]
) -> None:
    for field in INFO_FIELDS:
        existing = header.info.get(field.name)
        if existing is not None:
            if (
                str(existing.number) != field.number
                or existing.type != field.value_type
            ):
                raise AugmentationError(
                    "Input INFO/{} conflicts with LAVA contract: Number={},Type={}".format(
                        field.name, existing.number, existing.type
                    )
                )
            continue
        header.add_meta(
            "INFO",
            items=[
                ("ID", field.name),
                ("Number", field.number),
                ("Type", field.value_type),
                ("Description", field.description),
            ],
        )
    existing_models = {
        str(values["ID"]): values
        for record in header.records
        if record.key == "LAVA_MODEL"
        for values in [dict(record.items())]
        if "ID" in values
    }
    for model in models:
        existing = existing_models.get(model["short_id"])
        if existing is not None:
            if (
                _header_value(existing.get("UUID")) != model["model_id"]
                or _header_value(existing.get("Name")) != model["name"]
            ):
                raise AugmentationError(
                    "Input LAVA_MODEL/{} conflicts with model manifest".format(
                        model["short_id"]
                    )
                )
            continue
        header.add_meta(
            "LAVA_MODEL",
            items=[
                ("ID", model["short_id"]),
                ("UUID", model["model_id"]),
                ("Name", model["name"]),
            ],
        )


def _header_value(value: object) -> str:
    """Normalize htslib's quoted structured-header values."""
    text = str(value)
    return (
        text[1:-1]
        if len(text) >= 2 and text.startswith('"') and text.endswith('"')
        else text
    )


def _row_to_annotations(
    row: Mapping[str, object],
    alt_index: int,
    models: Sequence[Mapping[str, str]],
    include_model_scores: bool,
) -> Dict[str, object]:
    status = str(row.get("status") or "FAILED")
    has_result = row.get("result_variant_id") is not None
    if status == "PENDING":
        status = "SCORED" if has_result else "NO_RESULT"
    result = {
        "LAVA1_STATUS": status,
        "LAVA1_ERROR": row.get("error_code"),
    }
    if not has_result:
        return result

    logfcs = [_as_float(row.get(f"{model['column_prefix']}_logfc")) for model in models]
    finite_logfcs = [abs(value) for value in logfcs if value is not None]
    top_name = row.get("most_active_celltype")
    short_by_name = {
        name: model["short_id"]
        for model in models
        for name in (model["name"], model["column_prefix"])
    }
    result.update(
        {
            "LAVA1_PRIORITIZED": 1 if bool(row.get("prioritized")) else 0,
            "LAVA1_MAX_ABS_LOGFC": max(finite_logfcs) if finite_logfcs else None,
            "LAVA1_TOP_MODEL": short_by_name.get(str(top_name))
            if top_name is not None
            else None,
            "LAVA1_REGION": row.get("region_type"),
            "LAVA1_NEAREST_GENE": _first_gene(row.get("nearest_genes")),
            "LAVA1_CCRE": row.get("ccre"),
            "LAVA1_GNOMAD_AF": _as_float(row.get("af_global")),
            "LAVA1_CADD_PHRED": _as_float(row.get("cadd_phred")),
            "LAVA1_AM_SCORE": _as_float(row.get("am_pathogenicity")),
            "LAVA1_AM_CLASS": row.get("am_class"),
            "LAVA1_REVEL": _as_float(row.get("revel_score")),
        }
    )
    if include_model_scores:
        entries = []
        for model in models:
            prefix = model["column_prefix"]
            logfc = _as_float(row.get(f"{prefix}_logfc"))
            jsd = _as_float(row.get(f"{prefix}_jsd"))
            quantile = _as_float(row.get(f"{prefix}_active_allele_quantile"))
            in_peak = row.get(f"{prefix}_in_peak")
            prioritized = row.get(f"{prefix}_prioritized")
            if all(
                value is None for value in (logfc, jsd, quantile, in_peak, prioritized)
            ):
                continue
            entries.append(
                "|".join(
                    [
                        str(alt_index),
                        model["short_id"],
                        _format_value(logfc),
                        _format_value(jsd),
                        _format_value(quantile),
                        _format_value(in_peak),
                        _format_value(prioritized),
                    ]
                )
            )
        result["LAVA1_MODEL_SCORE"] = entries
    return result


def _set_info(
    record: pysam.VariantRecord, annotations: Sequence[Mapping[str, object]]
) -> None:
    for field in INFO_FIELDS:
        if field.name in record.info:
            del record.info[field.name]
        if field.number == "A":
            values = tuple(annotation.get(field.name) for annotation in annotations)
            if values and any(value is not None for value in values):
                record.info[field.name] = values
            continue
        values = tuple(
            value
            for annotation in annotations
            for value in annotation.get(field.name, [])
        )
        if values:
            record.info[field.name] = values


def _first_gene(value: object) -> Optional[str]:
    if value is None:
        return None
    try:
        parsed = json.loads(value) if isinstance(value, str) else value
    except json.JSONDecodeError:
        return str(value)
    if not isinstance(parsed, list) or not parsed:
        return str(parsed) if parsed else None
    first = parsed[0]
    if isinstance(first, dict):
        for key in ("gene_name", "symbol", "name", "gene_id"):
            if first.get(key):
                return str(first[key])
    return str(first)


def _none_if_missing(value: object) -> object:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _as_float(value: object) -> Optional[float]:
    value = _none_if_missing(value)
    return float(value) if value is not None else None


def _format_value(value: object) -> str:
    value = _none_if_missing(value)
    if value is None:
        return "."
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _validate_indexed_output(
    path: Path, expected_records: int, expected_alts: int
) -> None:
    records = 0
    alts = 0
    with pysam.VariantFile(str(path)) as output:
        for record in output:
            records += 1
            alts += len(record.alts or ())
    if (records, alts) != (expected_records, expected_alts):
        raise AugmentationError(
            "Indexed output validation mismatch: expected ({}, {}) found ({}, {})".format(
                expected_records, expected_alts, records, alts
            )
        )


def _file_metadata(path: Path) -> dict:
    return {
        "uri": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, text: str) -> None:
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(text)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write an occurrence-aligned augmented VCF"
    )
    parser.add_argument("--input-vcf", required=True)
    parser.add_argument("--bucket-dir", required=True)
    parser.add_argument("--models-json", required=True)
    parser.add_argument("--ingest-manifest", required=True)
    parser.add_argument("--output-vcf", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--results-parquet-uri", required=True)
    parser.add_argument("--include-model-scores", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    write_augmented_vcf(
        args.input_vcf,
        args.bucket_dir,
        args.models_json,
        args.ingest_manifest,
        args.output_vcf,
        args.output_manifest,
        args.results_parquet_uri,
        include_model_scores=args.include_model_scores,
    )


if __name__ == "__main__":
    main()
