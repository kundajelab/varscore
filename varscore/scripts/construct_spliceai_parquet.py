"""Convert the precomputed SpliceAI score VCFs into per-chromosome Parquet.

The Illumina precomputed scores ship as bgzipped VCFs (one for SNVs, one for
indels) whose INFO column holds a single SpliceAI annotation per record:

    SpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL

Both files are parsed, the SpliceAI fields are split out of INFO, and the result
is rewritten into a Hive-partitioned Parquet dataset keyed on chromosome:

    varscore/data/spliceai/CHROM=chr1/data_0.parquet
    varscore/data/spliceai/CHROM=chr2/data_0.parquet
    ...

Chromosomes are normalized to a 'chr' prefix so the partition keys match the
AlphaMissense dataset and the shared lookup path. Rows are sorted by POS within
each partition so DuckDB can skip row groups on positional lookups, e.g.:

    SELECT ds_max
    FROM read_parquet('varscore/data/spliceai/**/*.parquet', hive_partitioning=true)
    WHERE CHROM='chr1' AND POS=69091 AND REF='A' AND ALT='C';

Usage:
    python -m varscore.scripts.construct_spliceai_parquet
    python -m varscore.scripts.construct_spliceai_parquet \
        -i varscore/data/raw/spliceai_scores.raw.snv.hg38.vcf.gz \
           varscore/data/raw/spliceai_scores.raw.indel.hg38.vcf.gz \
        -o varscore/data/spliceai
"""

import argparse
from typing import List, Optional

import duckdb

# The eight standard VCF columns. Header/comment lines (## and the #CHROM header)
# all start with '#' and are dropped via the read_csv `comment` option, so we
# supply the column names/types explicitly rather than auto-detecting a header.
VCF_COLUMNS = {
    "CHROM": "VARCHAR",
    "POS": "INTEGER",
    "ID": "VARCHAR",
    "REF": "VARCHAR",
    "ALT": "VARCHAR",
    "QUAL": "VARCHAR",
    "FILTER": "VARCHAR",
    "INFO": "VARCHAR",
}

DEFAULT_INPUTS = [
    "varscore/data/raw/spliceai_scores.raw.snv.hg38.vcf.gz",
    "varscore/data/raw/spliceai_scores.raw.indel.hg38.vcf.gz",
]


def construct(in_paths: List[str], out_path: str, threads: Optional[int] = None) -> None:
    con = duckdb.connect()
    if threads:
        con.execute(f"PRAGMA threads={threads}")

    columns_sql = ", ".join(f"'{name}': '{dtype}'" for name, dtype in VCF_COLUMNS.items())
    files_sql = "[" + ", ".join(f"'{p}'" for p in in_paths) + "]"

    # `sa` is the SpliceAI payload (everything after "SpliceAI="); its pipe-
    # delimited fields are ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL.
    con.execute(
        f"""
        COPY (
            SELECT
                CASE WHEN CHROM LIKE 'chr%' THEN CHROM ELSE 'chr' || CHROM END AS CHROM,
                POS,
                REF,
                ALT,
                split_part(sa, '|', 2)                      AS spliceai_symbol,
                TRY_CAST(split_part(sa, '|', 3) AS FLOAT)   AS ds_ag,
                TRY_CAST(split_part(sa, '|', 4) AS FLOAT)   AS ds_al,
                TRY_CAST(split_part(sa, '|', 5) AS FLOAT)   AS ds_dg,
                TRY_CAST(split_part(sa, '|', 6) AS FLOAT)   AS ds_dl,
                TRY_CAST(split_part(sa, '|', 7) AS INTEGER) AS dp_ag,
                TRY_CAST(split_part(sa, '|', 8) AS INTEGER) AS dp_al,
                TRY_CAST(split_part(sa, '|', 9) AS INTEGER) AS dp_dg,
                TRY_CAST(split_part(sa, '|', 10) AS INTEGER) AS dp_dl,
                GREATEST(
                    TRY_CAST(split_part(sa, '|', 3) AS FLOAT),
                    TRY_CAST(split_part(sa, '|', 4) AS FLOAT),
                    TRY_CAST(split_part(sa, '|', 5) AS FLOAT),
                    TRY_CAST(split_part(sa, '|', 6) AS FLOAT)
                ) AS ds_max
            FROM (
                SELECT
                    CHROM, POS, REF, ALT,
                    regexp_extract(INFO, 'SpliceAI=(.*)', 1) AS sa
                FROM read_csv(
                    {files_sql},
                    delim='\t',
                    header=false,
                    comment='#',
                    auto_detect=false,
                    columns={{{columns_sql}}}
                )
            )
            WHERE sa != ''
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
        description="Partition precomputed SpliceAI VCFs into per-chromosome Parquet files."
    )
    parser.add_argument(
        "-i", "--in_paths",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Paths to the precomputed SpliceAI VCF(s) (SNV and/or indel).",
    )
    parser.add_argument(
        "-o", "--out_path",
        default="varscore/data/spliceai",
        help="Output directory for the Hive-partitioned Parquet dataset",
    )
    parser.add_argument(
        "-t", "--threads", type=int, default=None,
        help="DuckDB thread count (default: DuckDB's automatic choice)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    construct(args.in_paths, args.out_path, args.threads)
