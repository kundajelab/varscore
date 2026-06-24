"""AlphaMissense lookups against the chr-partitioned Parquet dataset.

The dataset is produced by `varscore.scripts.construct_alphamissense_parquet`
as a Hive-partitioned (CHROM) Parquet directory and is queried with DuckDB so
that only the partitions for the chromosomes present in the query are scanned.
"""

import os

import duckdb
import pandas as pd

from varscore.core.logging import get_logger

logger = get_logger(__name__)

# Default location of the chr-partitioned Parquet dataset.
ALPHAMISSENSE_DIR = "./varscore/data/alphamissense"

# AlphaMissense columns brought over on a match (CHROM/POS/REF/ALT are the join
# keys and are not duplicated into the output).
AM_COLUMNS = [
    "genome",
    "uniprot_id",
    "transcript_id",
    "protein_variant",
    "am_pathogenicity",
    "am_class",
]


def _normalize_chr(series: pd.Series) -> pd.Series:
    """Ensure chromosome values are prefixed with 'chr' to match the dataset."""
    chrom = series.astype(str)
    return chrom.where(chrom.str.startswith("chr"), "chr" + chrom)


def lookup_alphamissense(
    variant_df: pd.DataFrame,
    alphamissense_dir: str = ALPHAMISSENSE_DIR,
) -> pd.DataFrame:
    """Left-join variants against AlphaMissense scores on (chr, pos, ref, alt).

    Every input row is preserved (variants with no AlphaMissense entry get null
    score columns). Because a genomic variant can map to multiple transcripts,
    a variant may match more than one AlphaMissense row and thus expand to
    multiple output rows.

    Args:
        variant_df: DataFrame with at least 'chr', 'pos', 'ref', 'alt' columns.
        alphamissense_dir: Directory holding the Hive-partitioned Parquet dataset.

    Returns:
        The input DataFrame's columns (in order) with the AlphaMissense columns
        in AM_COLUMNS appended.
    """
    if not os.path.isdir(alphamissense_dir):
        raise FileNotFoundError(
            f"AlphaMissense Parquet directory not found at {alphamissense_dir}. "
            "Generate it first with: "
            "python -m varscore.scripts.construct_alphamissense_parquet"
        )

    glob = os.path.join(alphamissense_dir, "**", "*.parquet")

    # Add helper columns: normalized chr for matching and a stable row id so the
    # output keeps the input order (and groups multi-transcript matches together).
    query_df = variant_df.copy()
    query_df["_chr"] = _normalize_chr(query_df["chr"])
    query_df["_row"] = range(len(query_df))

    # Restrict the Parquet scan to the chromosomes we actually need so DuckDB
    # prunes to those partition files instead of scanning the whole genome.
    chroms = sorted(query_df["_chr"].unique())
    chrom_list = ", ".join(f"'{c}'" for c in chroms)

    am_cols_select = ", ".join(f"am.{c}" for c in AM_COLUMNS)

    con = duckdb.connect()
    con.register("variants", query_df)
    result = con.sql(
        f"""
        SELECT v.* EXCLUDE (_chr, _row), {am_cols_select}
        FROM variants v
        LEFT JOIN (
            SELECT * FROM read_parquet('{glob}', hive_partitioning=true)
            WHERE CHROM IN ({chrom_list})
        ) am
          ON am.CHROM = v._chr
         AND am.POS = v.pos
         AND am.REF = v.ref
         AND am.ALT = v.alt
        ORDER BY v._row
        """
    ).df()
    con.close()

    return result
