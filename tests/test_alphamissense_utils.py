import duckdb
import pandas as pd
import pytest

from varscore.utils.alphamissense_utils import AM_COLUMNS, lookup_alphamissense


def _build_dataset(tmp_path):
    """Write a tiny chr-partitioned AlphaMissense Parquet dataset for testing.

    chr1:200 G>C maps to three transcripts (a multi-transcript variant).
    """
    df = pd.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1", "chr1", "chr2"],
            "POS": [100, 200, 200, 200, 300],
            "REF": ["A", "G", "G", "G", "C"],
            "ALT": ["T", "C", "C", "C", "T"],
            "genome": ["hg38"] * 5,
            "uniprot_id": ["U1", "U2a", "U2b", "U2c", "U3"],
            "transcript_id": ["T1", "T2a", "T2b", "T2c", "T3"],
            "protein_variant": ["p1", "p2", "p2", "p2", "p3"],
            "am_pathogenicity": [0.1, 0.2, 0.2, 0.2, 0.9],
            "am_class": ["likely_benign"] * 4 + ["likely_pathogenic"],
        }
    )
    out_dir = tmp_path / "alphamissense"
    con = duckdb.connect()
    con.register("t", df)
    con.execute(
        f"COPY t TO '{out_dir}' (FORMAT PARQUET, PARTITION_BY (CHROM), OVERWRITE_OR_IGNORE)"
    )
    con.close()
    return str(out_dir)


def test_single_match(tmp_path):
    am_dir = _build_dataset(tmp_path)
    variants = pd.DataFrame(
        {"chr": ["chr1"], "pos": [100], "ref": ["A"], "alt": ["T"]}
    )
    result = lookup_alphamissense(variants, am_dir)

    assert len(result) == 1
    assert result.iloc[0]["am_pathogenicity"] == 0.1
    assert result.iloc[0]["am_class"] == "likely_benign"
    assert result.iloc[0]["transcript_id"] == "T1"
    # All AlphaMissense columns appended after the input columns.
    assert list(result.columns) == ["chr", "pos", "ref", "alt"] + AM_COLUMNS


def test_multi_transcript_expands_rows(tmp_path):
    am_dir = _build_dataset(tmp_path)
    variants = pd.DataFrame(
        {"chr": ["chr1"], "pos": [200], "ref": ["G"], "alt": ["C"]}
    )
    result = lookup_alphamissense(variants, am_dir)

    assert len(result) == 3
    assert set(result["transcript_id"]) == {"T2a", "T2b", "T2c"}
    assert (result["am_pathogenicity"] == 0.2).all()


def test_no_match_is_preserved_with_nulls(tmp_path):
    am_dir = _build_dataset(tmp_path)
    variants = pd.DataFrame(
        {"chr": ["chr1"], "pos": [999], "ref": ["A"], "alt": ["A"]}
    )
    result = lookup_alphamissense(variants, am_dir)

    assert len(result) == 1
    assert pd.isna(result.iloc[0]["am_pathogenicity"])
    assert pd.isna(result.iloc[0]["am_class"])


def test_chr_prefix_normalization(tmp_path):
    am_dir = _build_dataset(tmp_path)
    # Input chromosome without the 'chr' prefix should still match.
    variants = pd.DataFrame({"chr": ["1"], "pos": [100], "ref": ["A"], "alt": ["T"]})
    result = lookup_alphamissense(variants, am_dir)

    assert len(result) == 1
    assert result.iloc[0]["am_pathogenicity"] == 0.1
    # Original input chr value is preserved (not rewritten to 'chr1').
    assert result.iloc[0]["chr"] == "1"


def test_input_order_and_extra_columns_preserved(tmp_path):
    am_dir = _build_dataset(tmp_path)
    variants = pd.DataFrame(
        {
            "chr": ["chr2", "chr1", "chr1"],
            "pos": [300, 999, 100],
            "ref": ["C", "A", "A"],
            "alt": ["T", "A", "T"],
            "variant_id": ["v_chr2", "v_nomatch", "v_chr1"],
        }
    )
    result = lookup_alphamissense(variants, am_dir)

    # 1 (chr2) + 1 (no match) + 1 (chr1) = 3 rows, in input order.
    assert list(result["variant_id"]) == ["v_chr2", "v_nomatch", "v_chr1"]
    assert "variant_id" in result.columns
    assert result.iloc[0]["am_class"] == "likely_pathogenic"
    assert pd.isna(result.iloc[1]["am_pathogenicity"])
    assert result.iloc[2]["am_pathogenicity"] == 0.1


def test_missing_dir_raises(tmp_path):
    variants = pd.DataFrame(
        {"chr": ["chr1"], "pos": [100], "ref": ["A"], "alt": ["T"]}
    )
    with pytest.raises(FileNotFoundError):
        lookup_alphamissense(variants, str(tmp_path / "does_not_exist"))
