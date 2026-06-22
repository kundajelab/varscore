"""Convert the AlphaMissense hg38 TSV into per-chromosome Parquet partitions.

The raw file (`AlphaMissense_hg38.tsv.gz`, ~71M rows) is rewritten into a
Hive-partitioned Parquet dataset keyed on chromosome:

    varscore/data/alphamissense/CHROM=chr1/data_0.parquet
    varscore/data/alphamissense/CHROM=chr2/data_0.parquet
    ...

This lets DuckDB prune to a single chromosome file at query time, e.g.:

    SELECT am_pathogenicity, am_class
    FROM read_parquet('varscore/data/alphamissense/**/*.parquet', hive_partitioning=true)
    WHERE CHROM='chr1' AND POS=69094 AND REF='G' AND ALT='T';

Rows are sorted by POS within each partition so DuckDB can skip row groups
on positional lookups.

Usage:
    python -m varscore.scripts.construct_alphamissense_parquet
    python -m varscore.scripts.construct_alphamissense_parquet \
        -i varscore/data/raw/AlphaMissense_hg38.tsv.gz \
        -o varscore/data/alphamissense
"""

import argparse
from typing import Optional

import duckdb

# The TSV ships with 3 license/comment lines, then a header row whose first
# column is "#CHROM". We skip the comment lines and supply explicit column
# types so the output schema is stable (POS stays a 32-bit int).
COLUMNS = {
    "CHROM": "VARCHAR",
    "POS": "INTEGER",
    "REF": "VARCHAR",
    "ALT": "VARCHAR",
    "genome": "VARCHAR",
    "uniprot_id": "VARCHAR",
    "transcript_id": "VARCHAR",
    "protein_variant": "VARCHAR",
    "am_pathogenicity": "DOUBLE",
    "am_class": "VARCHAR",
}


def construct(in_path: str, out_path: str, threads: Optional[int] = None) -> None:
    con = duckdb.connect()
    if threads:
        con.execute(f"PRAGMA threads={threads}")

    columns_sql = ", ".join(f"'{name}': '{dtype}'" for name, dtype in COLUMNS.items())

    con.execute(
        f"""
        COPY (
            SELECT * FROM read_csv(
                '{in_path}',
                delim='\t',
                skip=3,
                header=true,
                columns={{{columns_sql}}}
            )
            ORDER BY CHROM, POS
        ) TO '{out_path}' (
            FORMAT PARQUET,
            PARTITION_BY (CHROM),
            OVERWRITE_OR_IGNORE,
            COMPRESSION ZSTD
        )
        """
    )
    print(f"Wrote per-chromosome Parquet partitions to {out_path}/")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Partition AlphaMissense hg38 TSV into per-chromosome Parquet files."
    )
    parser.add_argument(
        "-i", "--in_path",
        default="varscore/data/raw/AlphaMissense_hg38.tsv.gz",
        help="Path to AlphaMissense_hg38.tsv.gz",
    )
    parser.add_argument(
        "-o", "--out_path",
        default="varscore/data/alphamissense",
        help="Output directory for the Hive-partitioned Parquet dataset",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=None,
        help="DuckDB thread count (default: DuckDB's automatic choice)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    construct(args.in_path, args.out_path, args.threads)
