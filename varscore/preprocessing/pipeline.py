"""This module handles variant preprocessing functional flows.

It takes in a variants TSV file and a genome FASTA file, creates valid/invalid
variant TSV files, then routes the valid variants into per-region-category TSV
files (coding, splice, promoter, intronic, ...) for downstream scoring.
"""

import argparse
from pathlib import Path

from varscore.preprocessing.region_filter import filter_variants_by_region
from varscore.preprocessing.streaming import stream_validate_variants

from varscore.core.logging import get_logger

logger = get_logger(__name__)


def preprocess_variants(
    variants_loc: str,
    genome_loc: str,
    valid_out_path: str,
    invalid_out_path: str,
    region_out_dir: str,
    categories=None,
    fmt: str = "auto",
    occurrence_out_dir: str = None,
    canonical_out_dir: str = None,
    invalid_parquet_out_dir: str = None,
    manifest_out_path: str = None,
    header_out_path: str = None,
    batch_size: int = 250000,
) -> None:
    """Preprocess variants through validation and region filtering.

    Args:
        variants_loc: Path to input variants file (TSV or VCF/gVCF).
        genome_loc: Path to genome FASTA file.
        valid_out_path: Output path for valid variants.
        invalid_out_path: Output path for invalid variants.
        region_out_dir: Directory for the per-region-category variant TSVs.
        categories: Optional subset of region categories to emit (default: all).
        fmt: Input format, one of "auto", "tsv", "vcf" (default: auto-detect).
        occurrence_out_dir: Directory for occurrence Parquet shards.
        canonical_out_dir: Directory for valid canonical Parquet shards.
        invalid_parquet_out_dir: Directory for invalid canonical Parquet shards.
        manifest_out_path: Path for the checksummed ingest manifest.
        header_out_path: Path for the parsed source VCF header.
        batch_size: Maximum occurrence rows retained by one ingest batch.
    """
    logger.info("Starting variant preprocessing.")

    artifact_root = Path(valid_out_path).parent / "ingest"
    occurrence_out_dir = occurrence_out_dir or str(artifact_root / "occurrences")
    canonical_out_dir = canonical_out_dir or str(artifact_root / "canonical" / "valid")
    invalid_parquet_out_dir = invalid_parquet_out_dir or str(
        artifact_root / "canonical" / "invalid"
    )
    manifest_out_path = manifest_out_path or str(artifact_root / "manifest.json")
    header_out_path = header_out_path or str(artifact_root / "header.vcf")

    result = stream_validate_variants(
        variants_loc,
        genome_loc,
        valid_out_path,
        invalid_out_path,
        occurrence_out_dir,
        canonical_out_dir,
        invalid_parquet_out_dir,
        manifest_out_path,
        header_out_path,
        batch_size=batch_size,
        fmt=fmt,
    )

    logger.info(
        "Validation complete: %d valid, %d invalid, %d unsupported occurrences; "
        "%d unique canonical variants.",
        result.valid_occurrence_count,
        result.invalid_occurrence_count,
        result.unsupported_occurrence_count,
        result.unique_canonical_variants,
    )

    if result.valid_occurrence_count:
        logger.info(
            "Filtering %d valid occurrences by region.", result.valid_occurrence_count
        )
        counts = filter_variants_by_region(
            valid_out_path,
            region_out_dir,
            categories,
            batch_size,
        )
        logger.info(
            "Region filtering complete: %s.",
            ", ".join(f"{cat}={n}" for cat, n in counts.items()),
        )
    else:
        logger.warning("No valid variants to filter by region.")


def main():
    args = _parse_args()
    categories = (
        [c.strip() for c in args.categories.split(",")] if args.categories else None
    )
    preprocess_variants(
        args.input,
        args.genome,
        args.valid_out,
        args.invalid_out,
        args.region_out_dir,
        categories,
        args.format,
        args.occurrence_out_dir,
        args.canonical_out_dir,
        args.invalid_parquet_out_dir,
        args.manifest_out_path,
        args.header_out_path,
        args.batch_size,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess variants through validation and region filtering."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input variants file: canonical TSV (chr, pos, ref, alt[, variant_id]) "
        "or VCF/gVCF (.vcf, .vcf.gz).",
    )
    parser.add_argument(
        "-g",
        "--genome",
        required=True,
        help="Genome FASTA file for validation.",
    )
    parser.add_argument(
        "-o",
        "--valid-out",
        dest="valid_out",
        required=True,
        help="Output path for valid variants.",
    )
    parser.add_argument(
        "--invalid-out",
        dest="invalid_out",
        required=True,
        help="Output path for invalid variants.",
    )
    parser.add_argument(
        "--region-out-dir",
        dest="region_out_dir",
        required=True,
        help="Directory for per-region-category variant TSVs (coding, splice, ...).",
    )
    parser.add_argument(
        "--categories",
        help="Comma-separated subset of region categories to emit (default: all).",
    )
    parser.add_argument(
        "-f",
        "--format",
        default="auto",
        choices=["auto", "tsv", "vcf"],
        help="Input format (default: auto, detected by extension).",
    )
    parser.add_argument(
        "--occurrence-out-dir", help="Directory for occurrence Parquet shards."
    )
    parser.add_argument(
        "--canonical-out-dir", help="Directory for valid canonical Parquet shards."
    )
    parser.add_argument(
        "--invalid-parquet-out-dir", help="Directory for invalid Parquet shards."
    )
    parser.add_argument(
        "--manifest-out",
        dest="manifest_out_path",
        help="Path for the ingest manifest JSON.",
    )
    parser.add_argument(
        "--header-out",
        dest="header_out_path",
        help="Path for the parsed input VCF header.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=250000,
        help="Maximum occurrence rows held by an ingest batch (default: 250000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
