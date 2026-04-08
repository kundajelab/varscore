import pytest
from varscore.trio_annotation.run import (
    VariantAnnotationInput,
    normalize_variant,
    normalize_from_input,
    trio_annotation,
)


# ---------------------------------------------------------------------------
# normalize_variant
# ---------------------------------------------------------------------------

class TestNormalizeVariant:
    def test_snv_unchanged(self):
        assert normalize_variant("chr1", 100, "A", "T") == ("chr1", 100, "A", "T")

    def test_common_suffix_trimmed(self):
        # AGT -> AT  (common suffix G removed)
        assert normalize_variant("chr1", 100, "AGT", "AT") == ("chr1", 100, "AG", "A")

    def test_common_prefix_trimmed_adjusts_pos(self):
        # AAT → AGT: suffix T removed → AA/AG; then prefix A removed → A/G, pos+1
        assert normalize_variant("chr1", 100, "AAT", "AGT") == ("chr1", 101, "A", "G")

    def test_prefix_and_suffix_trimmed(self):
        # AATG → AGTG: suffix G removed → AAT/AGT; suffix T removed → AA/AG;
        # then prefix A removed → A/G, pos+1
        assert normalize_variant("chr1", 100, "AATG", "AGTG") == ("chr1", 101, "A", "G")

    def test_single_base_not_trimmed(self):
        # ref len == 1, nothing to trim even if bases match
        assert normalize_variant("chr1", 100, "A", "A") == ("chr1", 100, "A", "A")

    def test_insertion(self):
        # ref='A', alt='ACGT' → no common suffix/prefix to trim
        assert normalize_variant("chr1", 200, "A", "ACGT") == ("chr1", 200, "A", "ACGT")

    def test_deletion(self):
        assert normalize_variant("chr1", 200, "ACGT", "A") == ("chr1", 200, "ACGT", "A")


# ---------------------------------------------------------------------------
# normalize_from_input
# ---------------------------------------------------------------------------

class TestNormalizeFromInput:
    def test_delegates_to_normalize_variant(self):
        v = VariantAnnotationInput(chr="chr2", pos=500, ref="AGT", alt="AT")
        assert normalize_from_input(v) == normalize_variant("chr2", 500, "AGT", "AT")


# ---------------------------------------------------------------------------
# trio_annotation  — inheritance classification
# ---------------------------------------------------------------------------

def _make(chrom, pos, ref, alt):
    return VariantAnnotationInput(chr=chrom, pos=pos, ref=ref, alt=alt)


class TestTrioAnnotation:
    """Test inheritance classification across all four outcomes."""

    def test_de_novo(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "C")
        father = _make("chr1", 100, "A", "G")
        assert trio_annotation(child, mother, father) == "De_Novo"

    def test_maternal_only(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "T")
        father = _make("chr1", 100, "A", "G")
        assert trio_annotation(child, mother, father) == "M"

    def test_paternal_only(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "G")
        father = _make("chr1", 100, "A", "T")
        assert trio_annotation(child, mother, father) == "F"

    def test_both_parents(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "T")
        father = _make("chr1", 100, "A", "T")
        assert trio_annotation(child, mother, father) == "Both"

    def test_normalization_required_still_matches(self):
        """Child and parent carry equivalent but unnormalized indel representations."""
        # Child: ref='AGT' alt='AT'  normalizes to ref='AG', alt='A' at pos 100
        child = _make("chr1", 100, "AGT", "AT")
        # Parent carries the already-trimmed form — after normalization both equal
        mother = _make("chr1", 100, "AG", "A")
        father = _make("chr1", 200, "C", "G")
        assert trio_annotation(child, mother, father) == "M"

    def test_different_chrom_is_de_novo(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr2", 100, "A", "T")   # different chromosome
        father = _make("chr3", 100, "A", "T")
        assert trio_annotation(child, mother, father) == "De_Novo"

    def test_different_pos_is_de_novo(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 101, "A", "T")
        father = _make("chr1", 102, "A", "T")
        assert trio_annotation(child, mother, father) == "De_Novo"

    @pytest.mark.parametrize("chrom", ["chr1", "chr10", "chr20", "chrX", "chrY"])
    def test_de_novo_various_chroms(self, chrom):
        child = _make(chrom, 500, "G", "C")
        mother = _make(chrom, 500, "G", "A")
        father = _make(chrom, 500, "G", "T")
        assert trio_annotation(child, mother, father) == "De_Novo"

    @pytest.mark.parametrize("chrom", ["chr1", "chr10", "chr20", "chrX", "chrY"])
    def test_maternal_various_chroms(self, chrom):
        child = _make(chrom, 500, "G", "C")
        mother = _make(chrom, 500, "G", "C")
        father = _make(chrom, 500, "G", "T")
        assert trio_annotation(child, mother, father) == "M"
