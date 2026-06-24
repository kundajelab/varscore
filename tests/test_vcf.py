"""Tests for VCF/gVCF parsing and format dispatch (varscore.core.io).

Fully self-contained: inline VCF strings written to tmp_path, pure text ->
DataFrame, so no genome / parquet / monkeypatching is needed.
"""
import gzip

import pandas as pd
import pytest

import varscore.core.io as vcf

# A small VCF exercising the cases we care about. Tab-separated per spec.
_HEADER = "\n".join(
    [
        "##fileformat=VCFv4.2",
        "##contig=<ID=chr1>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
    ]
)

_BODY_ROWS = [
    "chr1\t100\trs1\tA\tT\t.\t.\t.",          # plain SNV with ID
    "chr1\t200\t.\tC\tG,T\t.\t.\t.",          # multiallelic, missing ID
    "chr1\t300\t.\tG\t<NON_REF>\t.\t.\t.",    # gVCF reference block -> dropped
    "chr1\t400\t.\tA\tG,<*>\t.\t.\t.",        # one concrete + one symbolic
    "chr1\t500\t.\tAT\tA,*\t.\t.\t.",         # indel + spanning-deletion star
    "1\t600\trs2\tC\tG\t.\t.\t.",             # bare chr passthrough
]

SAMPLE_VCF = _HEADER + "\n" + "\n".join(_BODY_ROWS) + "\n"


@pytest.fixture
def vcf_path(tmp_path):
    p = tmp_path / "sample.vcf"
    p.write_text(SAMPLE_VCF)
    return str(p)


@pytest.fixture
def vcf_gz_path(tmp_path):
    p = tmp_path / "sample.vcf.gz"
    with gzip.open(p, "wt") as fh:
        fh.write(SAMPLE_VCF)
    return str(p)


def _row(df, chrom, pos, alt):
    sel = df[(df["chr"] == chrom) & (df["pos"] == pos) & (df["alt"] == alt)]
    assert len(sel) == 1, f"expected exactly one row for {chrom}:{pos} {alt}"
    return sel.iloc[0]


def test_columns_and_plain_snv(vcf_path):
    df = vcf.load_variants_vcf(vcf_path)
    assert list(df.columns) == vcf.VARIANT_SCHEMA
    row = _row(df, "chr1", 100, "T")
    assert row["ref"] == "A"
    assert row["variant_id"] == "rs1"


def test_multiallelic_split(vcf_path):
    df = vcf.load_variants_vcf(vcf_path)
    alts = set(df[(df["chr"] == "chr1") & (df["pos"] == 200)]["alt"])
    assert alts == {"G", "T"}
    # missing ID (.) -> NaN for both split rows
    assert df[(df["pos"] == 200)]["variant_id"].isna().all()


def test_symbolic_non_ref_dropped(vcf_path, caplog):
    with caplog.at_level("INFO"):
        df = vcf.load_variants_vcf(vcf_path)
    # the <NON_REF>-only site emits nothing
    assert df[(df["pos"] == 300)].empty
    assert "dropped" in caplog.text.lower()


def test_mixed_concrete_and_symbolic(vcf_path):
    df = vcf.load_variants_vcf(vcf_path)
    pos400 = df[(df["pos"] == 400)]
    assert set(pos400["alt"]) == {"G"}  # <*> dropped, G survives


def test_indel_and_spanning_deletion_star(vcf_path):
    df = vcf.load_variants_vcf(vcf_path)
    pos500 = df[(df["pos"] == 500)]
    assert set(pos500["alt"]) == {"A"}  # the AT->A deletion; '*' dropped
    assert _row(df, "chr1", 500, "A")["ref"] == "AT"


def test_bare_chr_passthrough(vcf_path):
    df = vcf.load_variants_vcf(vcf_path)
    # parser does NOT normalize chr; that is validate's job downstream
    row = _row(df, "1", 600, "G")
    assert row["variant_id"] == "rs2"


def test_gzip_matches_plaintext(vcf_path, vcf_gz_path):
    plain = vcf.load_variants_vcf(vcf_path)
    gzipped = vcf.load_variants_vcf(vcf_gz_path)
    pd.testing.assert_frame_equal(plain, gzipped)


def test_read_variants_dispatch_vcf(vcf_path, vcf_gz_path):
    assert len(vcf.read_variants(vcf_path)) == len(vcf.load_variants_vcf(vcf_path))
    assert len(vcf.read_variants(vcf_gz_path)) == len(vcf.load_variants_vcf(vcf_gz_path))


def test_read_variants_dispatch_tsv(tmp_path):
    # a 5-col canonical TSV should route to the headerless-TSV loader
    p = tmp_path / "variants.tsv"
    p.write_text("chr1\t100\tA\tT\tv1\nchr1\t200\tC\tG\tv2\n")
    df = vcf.read_variants(str(p))
    assert list(df.columns) == vcf.VARIANT_SCHEMA
    assert df.iloc[0]["variant_id"] == "v1"


def test_read_variants_bcf_rejected(tmp_path):
    with pytest.raises(ValueError, match="BCF"):
        vcf.read_variants(str(tmp_path / "data.bcf"))


def test_symbolic_only_returns_empty_with_schema(tmp_path):
    p = tmp_path / "blocks.vcf"
    p.write_text(_HEADER + "\nchr1\t10\t.\tG\t<NON_REF>\t.\t.\t.\n")
    df = vcf.load_variants_vcf(str(p))
    assert df.empty
    assert list(df.columns) == vcf.VARIANT_SCHEMA


def test_malformed_line_raises(tmp_path):
    p = tmp_path / "bad.vcf"
    p.write_text(_HEADER + "\nchr1\t10\trs1\tA\n")  # only 4 fields
    with pytest.raises(ValueError, match="malformed"):
        vcf.load_variants_vcf(str(p))
