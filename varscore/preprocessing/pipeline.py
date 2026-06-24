"""This module handles variant preprocessing functional flows.

It takes in a variants TSV file and a genome FASTA file, creates valid/invalid
variant TSV files, then routes the valid variants into per-region-category TSV
files (coding, splice, promoter, intronic, ...) for downstream scoring.
"""

import argparse
from varscore.preprocessing.region_filter import filter_variants_by_region
from varscore.preprocessing.validate import validate_variants

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
    """
    logger.info("Starting variant preprocessing.")

    valid_df, invalid_df = validate_variants(
        variants_loc,
        genome_loc,
        valid_out_path,
        invalid_out_path,
        fmt=fmt,
    )

    logger.info(
        "Validation complete: %d valid, %d invalid variants.",
        len(valid_df),
        len(invalid_df),
    )

    if not valid_df.empty:
        logger.info("Filtering %d valid variants by region.", len(valid_df))
        counts = filter_variants_by_region(
            valid_out_path,
            region_out_dir,
            categories,
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
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess variants through validation and region filtering."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input variants file: canonical TSV (chr, pos, ref, alt[, variant_id]) "
        "or VCF/gVCF (.vcf, .vcf.gz).",
    )
    parser.add_argument(
        "-g", "--genome", required=True,
        help="Genome FASTA file for validation.",
    )
    parser.add_argument(
        "-o", "--valid-out", dest="valid_out", required=True,
        help="Output path for valid variants.",
    )
    parser.add_argument(
        "--invalid-out", dest="invalid_out", required=True,
        help="Output path for invalid variants.",
    )
    parser.add_argument(
        "--region-out-dir", dest="region_out_dir", required=True,
        help="Directory for per-region-category variant TSVs (coding, splice, ...).",
    )
    parser.add_argument(
        "--categories",
        help="Comma-separated subset of region categories to emit (default: all).",
    )
    parser.add_argument(
        "-f", "--format", default="auto", choices=["auto", "tsv", "vcf"],
        help="Input format (default: auto, detected by extension).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
