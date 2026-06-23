"""SpliceAI lookups against the chr-partitioned Parquet dataset.

The dataset is produced by `varscore.scripts.construct_spliceai_parquet` from the
Illumina precomputed SpliceAI score VCFs (SNV + indel) as a Hive-partitioned
(CHROM) Parquet directory, and is queried with DuckDB so that only the partitions
for the chromosomes present in the query are scanned.
"""

import os

import duckdb
import pandas as pd

from varscore.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default location of the chr-partitioned Parquet dataset.
SPLICEAI_DIR = "./varscore/data/spliceai"

# SpliceAI columns brought over on a match (CHROM/POS/REF/ALT are the join keys
# and are not duplicated into the output). DS_* are delta scores (0-1) for
# acceptor/donor gain/loss; DP_* are the delta positions; ds_max is the maximum
# of the four delta scores (the value usually thresholded for prioritization).
SPLICEAI_COLUMNS = [
    "spliceai_symbol",
    "ds_ag",
    "ds_al",
    "ds_dg",
    "ds_dl",
    "dp_ag",
    "dp_al",
    "dp_dg",
    "dp_dl",
    "ds_max",
]


def _normalize_chr(series: pd.Series) -> pd.Series:
    """Ensure chromosome values are prefixed with 'chr' to match the dataset."""
    chrom = series.astype(str)
    return chrom.where(chrom.str.startswith("chr"), "chr" + chrom)


def lookup_spliceai(
    variant_df: pd.DataFrame,
    spliceai_dir: str = SPLICEAI_DIR,
) -> pd.DataFrame:
    """Left-join variants against SpliceAI scores on (chr, pos, ref, alt).

    Every input row is preserved (variants with no SpliceAI entry get null score
    columns). The precomputed scores carry one annotation per variant (the gene
    it falls in), so unlike AlphaMissense a matched variant expands to a single
    output row; only overlapping-gene loci with multiple precomputed records
    would produce more than one row.

    Args:
        variant_df: DataFrame with at least 'chr', 'pos', 'ref', 'alt' columns.
        spliceai_dir: Directory holding the Hive-partitioned Parquet dataset.

    Returns:
        The input DataFrame's columns (in order) with the SpliceAI columns in
        SPLICEAI_COLUMNS appended.
    """
    if not os.path.isdir(spliceai_dir):
        raise FileNotFoundError(
            f"SpliceAI Parquet directory not found at {spliceai_dir}. "
            "Generate it first with: "
            "python -m varscore.scripts.construct_spliceai_parquet"
        )

    glob = os.path.join(spliceai_dir, "**", "*.parquet")

    # Add helper columns: normalized chr for matching and a stable row id so the
    # output keeps the input order (and groups multi-record matches together).
    query_df = variant_df.copy()
    query_df["_chr"] = _normalize_chr(query_df["chr"])
    query_df["_row"] = range(len(query_df))

    # Restrict the Parquet scan to the chromosomes we actually need so DuckDB
    # prunes to those partition files instead of scanning the whole genome.
    chroms = sorted(query_df["_chr"].unique())
    chrom_list = ", ".join(f"'{c}'" for c in chroms)

    sa_cols_select = ", ".join(f"sa.{c}" for c in SPLICEAI_COLUMNS)

    con = duckdb.connect()
    con.register("variants", query_df)
    result = con.sql(
        f"""
        SELECT v.* EXCLUDE (_chr, _row), {sa_cols_select}
        FROM variants v
        LEFT JOIN (
            SELECT * FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE CHROM IN ({chrom_list})
        ) sa
          ON sa.CHROM = v._chr
         AND sa.POS = v.pos
         AND sa.REF = v.ref
         AND sa.ALT = v.alt
        ORDER BY v._row
        """
    ).df()
    con.close()

    return result
