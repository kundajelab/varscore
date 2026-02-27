"""This module handles variant preprocessing functional flows.

It takes in a variants TSV file and a genome FASTA file and creates valid, invalid, coding, and non-coding
variant TSV files.
"""

import argparse
from varscore.variant_region_filter import filter_variants_by_region
from varscore.validate_variants import validate_variants

from varscore.utils.logging_config import get_logger

logger = get_logger(__name__)


def preprocess_variants(
    variants_loc: str,
    genome_loc: str,
    valid_out_path: str,
    invalid_out_path: str,
    coding_out_path: str,
    noncoding_out_path: str,
) -> None:
    """Preprocess variants through validation and region filtering.

    Args:
        variants_loc: Path to input variants TSV file.
        genome_loc: Path to genome FASTA file.
        valid_out_path: Output path for valid variants.
        invalid_out_path: Output path for invalid variants.
        coding_out_path: Output path for coding (exonic) variants.
        noncoding_out_path: Output path for non-coding variants.
    """
    logger.info("Starting variant preprocessing.")

    valid_df, invalid_df = validate_variants(
        variants_loc,
        genome_loc,
        valid_out_path,
        invalid_out_path,
    )

    logger.info(
        "Validation complete: %d valid, %d invalid variants.",
        len(valid_df),
        len(invalid_df),
    )

    if not valid_df.empty:
        logger.info("Filtering %d valid variants by region.", len(valid_df))
        coding_df, noncoding_df = filter_variants_by_region(
            valid_out_path,
            coding_out_path,
            noncoding_out_path,
        )
        logger.info(
            "Region filtering complete: %d coding, %d non-coding variants.",
            len(coding_df),
            len(noncoding_df),
        )
    else:
        logger.warning("No valid variants to filter by region.")


def main():
    args = _parse_args()
    preprocess_variants(
        args.input,
        args.genome,
        args.valid_out,
        args.invalid_out,
        args.coding_out,
        args.noncoding_out,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess variants through validation and region filtering."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input variants TSV file (chr, pos, ref, alt columns).",
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
        "--coding-out", dest="coding_out", required=True,
        help="Output path for coding (exonic) variants.",
    )
    parser.add_argument(
        "--noncoding-out", dest="noncoding_out", required=True,
        help="Output path for non-coding variants.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
