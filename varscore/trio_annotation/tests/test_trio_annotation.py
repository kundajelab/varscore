import pytest
from varscore.trio_annotation.run import (
    normalize_variant,
    normalize_from_input,
    trio_annotation,
    VariantAnnotationInput,
)


def _make(chrom, pos, ref, alt):
    return VariantAnnotationInput(chr=chrom, pos=pos, ref=ref, alt=alt)


def _classify(child, maternal, paternal):
    """Run trio_annotation with list normalisation."""
    if not isinstance(maternal, list):
        maternal = [maternal]
    if not isinstance(paternal, list):
        paternal = [paternal]
    return trio_annotation(child, maternal, paternal)


# ---------------------------------------------------------------------------
# normalize_variant  (pure function — no mocking needed)
# ---------------------------------------------------------------------------

class TestNormalizeVariant:
    def test_snv_unchanged(self):
        assert normalize_variant("chr1", 100, "A", "T") == ("chr1", 100, "A", "T")

    def test_common_suffix_trimmed(self):
        # AGT -> AT  (common suffix G removed)
        assert normalize_variant("chr1", 100, "AGT", "AT") == ("chr1", 100, "AG", "A")

    def test_common_prefix_trimmed_adjusts_pos(self):
        # AAT -> AGT: suffix T removed -> AA/AG; then prefix A removed -> A/G, pos+1
        assert normalize_variant("chr1", 100, "AAT", "AGT") == ("chr1", 101, "A", "G")

    def test_prefix_and_suffix_trimmed(self):
        # AATG -> AGTG: suffix G removed -> AAT/AGT; suffix T removed -> AA/AG;
        # then prefix A removed -> A/G, pos+1
        assert normalize_variant("chr1", 100, "AATG", "AGTG") == ("chr1", 101, "A", "G")

    def test_single_base_not_trimmed(self):
        assert normalize_variant("chr1", 100, "A", "A") == ("chr1", 100, "A", "A")

    def test_insertion(self):
        assert normalize_variant("chr1", 200, "A", "ACGT") == ("chr1", 200, "A", "ACGT")

    def test_deletion(self):
        assert normalize_variant("chr1", 200, "ACGT", "A") == ("chr1", 200, "ACGT", "A")


# ---------------------------------------------------------------------------
# normalize_from_input  (pure function — no mocking needed)
# ---------------------------------------------------------------------------

class TestNormalizeFromInput:
    def test_delegates_to_normalize_variant(self):
        v = VariantAnnotationInput(chr="chr2", pos=500, ref="AGT", alt="AT")
        assert normalize_from_input(v) == normalize_variant("chr2", 500, "AGT", "AT")


# ---------------------------------------------------------------------------
# trio_annotation — inheritance classification
# ---------------------------------------------------------------------------

class TestTrioAnnotation:
    def test_de_novo(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "C")
        father = _make("chr1", 100, "A", "G")
        assert _classify(child, mother, father) == "De_Novo"

    def test_maternal_only(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "T")
        father = _make("chr1", 100, "A", "G")
        assert _classify(child, mother, father) == "M"

    def test_paternal_only(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "G")
        father = _make("chr1", 100, "A", "T")
        assert _classify(child, mother, father) == "F"

    def test_both_parents(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 100, "A", "T")
        father = _make("chr1", 100, "A", "T")
        assert _classify(child, mother, father) == "Both"

    def test_normalization_required_still_matches(self):
        """Child and parent carry equivalent but unnormalized indel representations."""
        child = _make("chr1", 100, "AGT", "AT")   # normalizes to (chr1, 100, AG, A)
        mother = _make("chr1", 100, "AG", "A")     # already normalized form
        father = _make("chr1", 200, "C", "G")
        assert _classify(child, mother, father) == "M"

    def test_different_chrom_is_de_novo(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr2", 100, "A", "T")
        father = _make("chr3", 100, "A", "T")
        assert _classify(child, mother, father) == "De_Novo"

    def test_different_pos_is_de_novo(self):
        child = _make("chr1", 100, "A", "T")
        mother = _make("chr1", 101, "A", "T")
        father = _make("chr1", 102, "A", "T")
        assert _classify(child, mother, father) == "De_Novo"

    @pytest.mark.parametrize("chrom", ["chr1", "chr10", "chr20", "chrX", "chrY"])
    def test_de_novo_various_chroms(self, chrom):
        child = _make(chrom, 500, "G", "C")
        mother = _make(chrom, 500, "G", "A")
        father = _make(chrom, 500, "G", "T")
        assert _classify(child, mother, father) == "De_Novo"

    @pytest.mark.parametrize("chrom", ["chr1", "chr10", "chr20", "chrX", "chrY"])
    def test_maternal_various_chroms(self, chrom):
        child = _make(chrom, 500, "G", "C")
        mother = _make(chrom, 500, "G", "C")
        father = _make(chrom, 500, "G", "T")
        assert _classify(child, mother, father) == "M"

    def test_multiple_children_classified_individually(self):
        """Each child variant is classified independently against the same parent sets."""
        mother = [_make("chr1", 100, "A", "T")]
        father = [_make("chr1", 200, "C", "G")]

        child1 = _make("chr1", 100, "A", "T")   # maternal only
        child2 = _make("chr1", 200, "C", "G")   # paternal only
        child3 = _make("chr1", 300, "G", "A")   # de novo

        assert _classify(child1, mother, father) == "M"
        assert _classify(child2, mother, father) == "F"
        assert _classify(child3, mother, father) == "De_Novo"
