"""Tests that preprocessing.validate preserves a custom variant_id end-to-end.

A custom id (from a TSV 5th column or a VCF ID) must survive validation: the
written valid-variants file is the canonical 5-col schema, carrying the id when
present and a blank 5th field when absent.
"""
import pandas as pd
import pytest

import varscore.core.io as io_utils
from varscore.preprocessing.validate import validate_variants

# 60 bp reference; ref alleles are derived from it so they always match.
_SEQ = "ACGT" * 15
_OTHER = {"A": "C", "C": "A", "G": "T", "T": "G"}


@pytest.fixture
def genome(tmp_path):
    p = tmp_path / "genome.fa"
    p.write_text(f">chr1\n{_SEQ}\n")
    return str(p)


def _ref_alt(pos1):  # pos1 is 1-based
    ref = _SEQ[pos1 - 1]
    return ref, _OTHER[ref]


@pytest.fixture
def variants_tsv(tmp_path):
    r30, a30 = _ref_alt(30)
    r40, a40 = _ref_alt(40)
    # row 1 carries a custom id; row 2 has none (blank 5th field)
    df = pd.DataFrame(
        [
            ("chr1", 30, r30, a30, "rs_custom"),
            ("chr1", 40, r40, a40, None),
        ]
    )
    p = tmp_path / "variants.tsv"
    df.to_csv(p, sep="\t", index=False, header=False)
    return str(p)


def test_validate_preserves_variant_id(genome, variants_tsv, tmp_path):
    valid_out = tmp_path / "valid.tsv"
    invalid_out = tmp_path / "invalid.tsv"

    valid_df, invalid_df = validate_variants(
        variants_tsv, genome, str(valid_out), str(invalid_out), width=20
    )

    # both variants are valid (refs were derived from the genome)
    assert len(valid_df) == 2
    assert invalid_df.empty

    # the written valid file round-trips to the 5-col canonical schema
    out = io_utils.load_variants(str(valid_out))
    assert list(out.columns) == io_utils.VARIANT_SCHEMA

    by_pos = {int(r["pos"]): r for _, r in out.iterrows()}
    assert by_pos[30]["variant_id"] == "rs_custom"  # custom id preserved
    assert pd.isna(by_pos[40]["variant_id"])  # absent id stays blank
