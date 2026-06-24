import argparse
import logging
import os
from datetime import datetime

import pandas as pd

import varscore.scoring.alphamissense.lookup as alphamissense_utils
from varscore.core.logging import get_logger, log_timing, setup_logging

# Initialize logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)

# Mirrors io_utils.VARIANT_SCHEMA; redefined here to keep this lightweight
# lookup module free of the model/genome import chain pulled in by io_utils.
VARIANT_SCHEMA = ["chr", "pos", "ref", "alt", "variant_id"]


##################
# CORE FUNCTIONS #
##################


def score_variants_alphamissense(
    variants_loc: str,
    out_path: str,
    alphamissense_dir: str = alphamissense_utils.ALPHAMISSENSE_DIR,
) -> None:
    """Annotate variants with AlphaMissense scores.

    Looks up each variant in the chr-partitioned AlphaMissense Parquet dataset
    (via DuckDB) and writes a TSV with the original variant columns plus the
    AlphaMissense columns appended. Variants with no AlphaMissense entry keep
    empty score columns; a variant that maps to multiple transcripts produces
    one output row per matching transcript.

    Args:
        variants_loc: Path to input variants TSV (no header; columns chr, pos,
            ref, alt[, variant_id]).
        out_path: Path to save the annotated TSV.
        alphamissense_dir: Directory holding the Hive-partitioned Parquet dataset.
    """
    start_time = datetime.now()
    logger.info("Starting AlphaMissense scoring")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"AlphaMissense dir: {alphamissense_dir}")
    logger.info(f"Output path: {out_path}")

    logger.info("Loading variants...")
    variant_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    total_variants = len(variant_df)
    logger.info(f"Loaded {total_variants} variants")

    logger.info("Looking up AlphaMissense scores...")
    result_df = alphamissense_utils.lookup_alphamissense(variant_df, alphamissense_dir)

    # Match statistics
    matched_mask = result_df["am_pathogenicity"].notna()
    n_matched_rows = int(matched_mask.sum())
    n_matched_variants = int(
        result_df.loc[matched_mask, ["chr", "pos", "ref", "alt"]]
        .drop_duplicates()
        .shape[0]
    )
    logger.info(
        f"Matched {n_matched_variants}/{total_variants} variants "
        f"({n_matched_rows} total matched rows; "
        f"{len(result_df)} output rows)"
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result_df.to_csv(out_path, sep="\t", index=False)
    logger.info(f"Results saved to {out_path}")

    log_timing(logger, "AlphaMissense scoring", start_time)


########
# MAIN #
########


def main():
    args = _parse_args()
    score_variants_alphamissense(
        args.variants_loc,
        args.out_path,
        args.alphamissense_dir,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate variants with AlphaMissense pathogenicity scores."
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True, help="Location of the variants file."
    )
    parser.add_argument(
        "-o", "--out_path", required=True, help="Location to save the results."
    )
    parser.add_argument(
        "-d", "--alphamissense_dir",
        default=alphamissense_utils.ALPHAMISSENSE_DIR,
        help="Directory with the chr-partitioned AlphaMissense Parquet dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()


"""
python -m varscore.scoring.alphamissense.score -v variants.tsv -o alphamissense_scores.tsv
"""
