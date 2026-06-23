import argparse
import logging
import os
from datetime import datetime

import pandas as pd

import varscore.utils.spliceai_utils as spliceai_utils
from varscore.utils.logging_config import get_logger, log_timing, setup_logging

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


def score_variants_spliceai(
    variants_loc: str,
    out_path: str,
    spliceai_dir: str = spliceai_utils.SPLICEAI_DIR,
) -> None:
    """Annotate variants with precomputed SpliceAI delta scores.

    Looks up each variant in the chr-partitioned SpliceAI Parquet dataset (via
    DuckDB) and writes a TSV with the original variant columns plus the SpliceAI
    columns appended. Variants with no SpliceAI entry keep empty score columns.

    Args:
        variants_loc: Path to input variants TSV (no header; columns chr, pos,
            ref, alt[, variant_id]).
        out_path: Path to save the annotated TSV.
        spliceai_dir: Directory holding the Hive-partitioned Parquet dataset.
    """
    start_time = datetime.now()
    logger.info("Starting SpliceAI scoring")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"SpliceAI dir: {spliceai_dir}")
    logger.info(f"Output path: {out_path}")

    logger.info("Loading variants...")
    variant_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    total_variants = len(variant_df)
    logger.info(f"Loaded {total_variants} variants")

    logger.info("Looking up SpliceAI scores...")
    result_df = spliceai_utils.lookup_spliceai(variant_df, spliceai_dir)

    # Match statistics
    matched_mask = result_df["ds_max"].notna()
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

    log_timing(logger, "SpliceAI scoring", start_time)


########
# MAIN #
########


def main():
    args = _parse_args()
    score_variants_spliceai(
        args.variants_loc,
        args.out_path,
        args.spliceai_dir,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate variants with precomputed SpliceAI delta scores."
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True, help="Location of the variants file."
    )
    parser.add_argument(
        "-o", "--out_path", required=True, help="Location to save the results."
    )
    parser.add_argument(
        "-d", "--spliceai_dir",
        default=spliceai_utils.SPLICEAI_DIR,
        help="Directory with the chr-partitioned SpliceAI Parquet dataset.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()


"""
python -m varscore.spliceai_scoring -v variants.tsv -o spliceai_scores.tsv
"""
