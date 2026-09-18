import pandas as pd
import pyfaidx
import pytest

from varscore.annotation.inheritance import (
    _trim,
    left_align,
    normalized_atoms,
    build_parent_lookup,
    classify_presence,
    classify_genotyped,
    annotate_inheritance,
)
from varscore.core.io import (
    load_variants_vcf_genotyped,
    _parse_gt_indices,
    _zygosity_wrt_alt,
)

VARIANT_SCHEMA = ["chr", "pos", "ref", "alt", "variant_id"]


def _df(rows):
    """rows: list of (chr, pos, ref, alt[, variant_id])."""
    norm = [r if len(r) == 5 else (*r, None) for r in rows]
    return pd.DataFrame(norm, columns=VARIANT_SCHEMA)


@pytest.fixture
def genome(tmp_path):
    """Tiny reference with an A-run so a 1-bp deletion has multiple representations."""
    fa = tmp_path / "ref.fa"
    fa.write_text(">chr1\nTAAAAG\n")  # pos 1..6: T A A A A G
    return pyfaidx.Fasta(str(fa))


# ---------------------------------------------------------------------------
# normalization primitives
# ---------------------------------------------------------------------------

class TestTrim:
    def test_snv_unchanged(self):
        assert _trim(100, "A", "T") == (100, "A", "T")

    def test_common_suffix_trimmed(self):
        assert _trim(100, "AGT", "AT") == (100, "AG", "A")

    def test_common_prefix_adjusts_pos(self):
        assert _trim(100, "AAT", "AGT") == (101, "A", "G")


class TestLeftAlign:
    def test_snv_is_noop(self, genome):
        assert left_align("chr1", 2, "A", "C", genome) == (2, "A", "C")

    def test_deletion_shifts_left_to_canonical(self, genome):
        # "AA">"A" anchored at pos4 and "TA">"T" anchored at pos1 are the same
        # 1-bp deletion in the A-run; both must canonicalize identically.
        assert left_align("chr1", 4, "AA", "A", genome) == (1, "TA", "T")
        assert left_align("chr1", 1, "TA", "T", genome) == (1, "TA", "T")

    def test_missing_chrom_falls_back_to_trim(self, genome):
        assert left_align("chrZ", 4, "AA", "A", genome) == (4, "AA", "A")


class TestNormalizedAtoms:
    def test_chr_prefix_added(self):
        assert normalized_atoms("1", 100, "A", "T") == [("chr1", 100, "A", "T")]

    def test_mnp_atomized_by_default(self):
        assert set(normalized_atoms("chr1", 1, "TA", "CG")) == {
            ("chr1", 1, "T", "C"),
            ("chr1", 2, "A", "G"),
        }

    def test_mnp_kept_whole_when_not_atomized(self):
        assert normalized_atoms("chr1", 1, "TA", "CG", atomize=False) == [
            ("chr1", 1, "TA", "CG")
        ]

    def test_mnp_collapses_to_snv_when_only_one_diff(self):
        # TAT>CAT differs only at position 0.
        assert normalized_atoms("chr1", 1, "TAT", "CAT") == [("chr1", 1, "T", "C")]


# ---------------------------------------------------------------------------
# presence/absence classification
# ---------------------------------------------------------------------------

class TestClassifyPresence:
    def test_four_labels(self):
        child = _df([
            ("chr1", 100, "A", "T"),   # M
            ("chr1", 200, "C", "G"),   # F
            ("chr1", 300, "T", "C"),   # Both
            ("chr1", 400, "G", "A"),   # De_Novo
        ])
        mother = _df([("chr1", 100, "A", "T"), ("chr1", 300, "T", "C")])
        father = _df([("chr1", 200, "C", "G"), ("chr1", 300, "T", "C")])
        out = classify_presence(child, mother, father)
        assert list(out["inheritance"]) == ["M", "F", "Both", "De_Novo"]
        assert list(out["inheritance_mode"]) == ["presence"] * 4

    def test_no_parents_is_all_de_novo(self):
        child = _df([("chr1", 100, "A", "T")])
        out = classify_presence(child, None, None)
        assert list(out["inheritance"]) == ["De_Novo"]

    def test_chr_naming_normalized(self):
        child = _df([("1", 100, "A", "T")])
        mother = _df([("chr1", 100, "A", "T")])
        out = classify_presence(child, mother, None)
        assert out["inheritance"].iloc[0] == "M"

    def test_variant_id_preserved(self):
        child = _df([("chr1", 100, "A", "T", "myrs")])
        out = classify_presence(child, None, None)
        assert out["variant_id"].iloc[0] == "myrs"
        assert "inheritance" in out.columns

    def test_indel_left_align_match(self, genome):
        # Same A-run deletion written differently in child vs mother.
        child = _df([("chr1", 4, "AA", "A")])
        mother = _df([("chr1", 1, "TA", "T")])
        with_genome = classify_presence(child, mother, None, genome=genome)
        without = classify_presence(child, mother, None, genome=None)
        assert with_genome["inheritance"].iloc[0] == "M"
        assert without["inheritance"].iloc[0] == "De_Novo"

    def test_mnp_subset_matches_when_atomized(self):
        # Child SNV equals one position of the mother's MNP.
        child = _df([("chr1", 2, "A", "G")])
        mother = _df([("chr1", 1, "TA", "CG")])
        assert classify_presence(child, mother, None)["inheritance"].iloc[0] == "M"

    def test_mnp_subset_de_novo_when_not_atomized(self):
        child = _df([("chr1", 2, "A", "G")])
        mother = _df([("chr1", 1, "TA", "CG")])
        out = classify_presence(child, mother, None, atomize=False)
        assert out["inheritance"].iloc[0] == "De_Novo"

    def test_partial_mnp_overlap_flagged(self):
        # Child MNP shares only one atom with the mother -> not inherited, flagged.
        child = _df([("chr1", 1, "TA", "CG")])
        mother = _df([("chr1", 1, "T", "C")])
        out = classify_presence(child, mother, None)
        assert out["inheritance"].iloc[0] == "De_Novo"
        assert out["inheritance_flag"].iloc[0] == "partial_mnp_overlap"


# ---------------------------------------------------------------------------
# genotype helpers
# ---------------------------------------------------------------------------

class TestGenotypeHelpers:
    def test_parse_gt_indices(self):
        assert _parse_gt_indices("0/1") == [0, 1]
        assert _parse_gt_indices("1|2") == [1, 2]
        assert _parse_gt_indices("./.") is None
        assert _parse_gt_indices("0/.") is None  # partial missing -> no call

    def test_zygosity_wrt_alt(self):
        assert _zygosity_wrt_alt([0, 1], 1) == "het"
        assert _zygosity_wrt_alt([1, 1], 1) == "hom_alt"
        assert _zygosity_wrt_alt([0, 0], 1) == "hom_ref"
        assert _zygosity_wrt_alt([0, 2], 1) == "hom_ref"  # carries a different alt
        assert _zygosity_wrt_alt([1, 2], 2) == "het"
        assert _zygosity_wrt_alt(None, 1) == "no_call"


# ---------------------------------------------------------------------------
# genotyped VCF reader + classification
# ---------------------------------------------------------------------------

VCF = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tCHILD\tMOM\tDAD
chr1\t100\t.\tA\tT\t.\t.\t.\tGT:DP:GQ\t0/1:30:99\t0/0:30:99\t0/0:30:99
chr1\t200\t.\tC\tG\t.\t.\t.\tGT:DP:GQ\t0/1:30:99\t0/1:30:99\t0/0:30:99
chr1\t300\t.\tG\tA\t.\t.\t.\tGT:DP:GQ\t0/1:30:99\t0/0:30:99\t0/1:30:99
chr1\t400\t.\tT\tC\t.\t.\t.\tGT:DP:GQ\t0/1:30:99\t0/1:30:99\t0/1:30:99
chr1\t500\t.\tA\tG\t.\t.\t.\tGT:DP:GQ\t0/1:30:99\t./.:.:.\t0/0:30:99
chr1\t600\t.\tC\tT\t.\t.\t.\tGT:DP:GQ\t1/1:30:99\t0/1:30:99\t0/0:30:99
chr1\t700\t.\tA\tC\t.\t.\t.\tGT:DP:GQ\t0/0:30:99\t0/1:30:99\t0/0:30:99
chr2\t50\trs9\tA\tG,T\t.\t.\t.\tGT:DP:GQ\t1/2:20:50\t0/1:20:50\t0/2:20:50
"""


@pytest.fixture
def trio_vcf(tmp_path):
    p = tmp_path / "trio.vcf"
    p.write_text(VCF)
    return str(p)


class TestGenotypedReader:
    def test_multiallelic_split_and_zygosity(self, trio_vcf):
        df = load_variants_vcf_genotyped(trio_vcf, samples=["CHILD", "MOM", "DAD"])
        chr2 = df[df["chr"] == "chr2"]
        assert set(chr2["alt"]) == {"G", "T"}
        g = chr2[chr2["alt"] == "G"].iloc[0]
        assert g["ZYG__CHILD"] == "het" and g["ZYG__DAD"] == "hom_ref"
        assert g["variant_id"] == "rs9"
        t = chr2[chr2["alt"] == "T"].iloc[0]
        assert t["ZYG__DAD"] == "het" and t["ZYG__MOM"] == "hom_ref"

    def test_missing_sample_raises(self, trio_vcf):
        with pytest.raises(ValueError):
            load_variants_vcf_genotyped(trio_vcf, samples=["NOPE"])


class TestClassifyGenotyped:
    def _run(self, trio_vcf):
        df = load_variants_vcf_genotyped(trio_vcf, samples=["CHILD", "MOM", "DAD"])
        return classify_genotyped(df, "CHILD", "MOM", "DAD")

    def test_labels(self, trio_vcf):
        out = self._run(trio_vcf).set_index("pos")
        assert out.loc[100, "inheritance"] == "De_Novo"
        assert out.loc[200, "inheritance"] == "M"
        assert out.loc[300, "inheritance"] == "F"
        assert out.loc[400, "inheritance"] == "Both"
        assert out.loc[500, "inheritance"] == "Uncertain"  # mother no-call

    def test_noncarrier_child_excluded(self, trio_vcf):
        out = self._run(trio_vcf)
        assert 700 not in set(out["pos"])

    def test_mendelian_violation_flag(self, trio_vcf):
        out = self._run(trio_vcf).set_index("pos")
        assert out.loc[600, "child_zygosity"] == "hom_alt"
        assert out.loc[600, "inheritance_flag"] == "mendelian_violation"

    def test_uncertain_requires_confident_ref(self, tmp_path):
        # Both parents hom_ref but low GQ -> not confident -> Uncertain, not De_Novo.
        vcf = tmp_path / "lowgq.vcf"
        vcf.write_text(
            "##fileformat=VCFv4.2\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tCHILD\tMOM\tDAD\n"
            "chr1\t100\t.\tA\tT\t.\t.\t.\tGT:DP:GQ\t0/1:30:99\t0/0:30:5\t0/0:30:5\n"
        )
        df = load_variants_vcf_genotyped(str(vcf), samples=["CHILD", "MOM", "DAD"])
        out = classify_genotyped(df, "CHILD", "MOM", "DAD")
        assert out["inheritance"].iloc[0] == "Uncertain"


# ---------------------------------------------------------------------------
# end-to-end orchestration
# ---------------------------------------------------------------------------

class TestAnnotateInheritance:
    def test_presence_end_to_end_preserves_id(self, tmp_path):
        child = tmp_path / "child.tsv"
        child.write_text("chr1\t100\tA\tT\tv1\nchr1\t200\tC\tG\tv2\n")
        mother = tmp_path / "mom.tsv"
        mother.write_text("chr1\t100\tA\tT\t\n")
        out = tmp_path / "out.tsv"
        annotate_inheritance(str(child), str(out), mother_loc=str(mother))
        res = pd.read_csv(out, sep="\t")
        assert list(res["inheritance"]) == ["M", "De_Novo"]
        assert list(res["variant_id"]) == ["v1", "v2"]

    def test_genotype_end_to_end(self, trio_vcf, tmp_path):
        out = tmp_path / "out.tsv"
        annotate_inheritance(
            str(trio_vcf), str(out),
            child_sample="CHILD", mother_sample="MOM", father_sample="DAD",
        )
        res = pd.read_csv(out, sep="\t")
        assert set(res["inheritance_mode"]) == {"genotype"}
        assert (res["inheritance"] == "De_Novo").any()
