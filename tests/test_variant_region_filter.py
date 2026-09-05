"""Tests for region-category routing (varscore.preprocessing.region_filter).

region_annotations_batch is monkeypatched so these don't depend on the built
region_annotations.parquet — we feed controlled label sets and assert each
variant lands in exactly the right (overlapping) set of category files.
"""

import pandas as pd
import pytest

import varscore.annotation.regions as region_utils
import varscore.preprocessing.region_filter as vrf

# variant_id -> region labels the (mocked) annotator should return for it.
LABELS_BY_VARIANT = {
    "v_cds": ["splice_site", "cds", "splice_region", "exonic"],  # CDS exon edge
    "v_prom": ["promoter"],
    "v_intron": ["intronic"],
    "v_5utr": ["five_prime_utr", "exonic"],
    "v_inter": ["intergenic"],
}

VARIANTS = [
    ("chr1", 100, "A", "T", "v_cds"),
    ("chr1", 200, "C", "G", "v_prom"),
    ("chr1", 300, "G", "A", "v_intron"),
    ("chr1", 400, "T", "C", "v_5utr"),
    ("chr1", 500, "A", "G", "v_inter"),
]
# pos -> variant_id, so the mock can map a queried interval back to its labels.
_POS_TO_ID = {pos: vid for _, pos, _, _, vid in VARIANTS}


def _ann(labels):
    return region_utils.RegionAnnotation(
        labels=labels,
        gene_ids=["G1"] if labels != ["intergenic"] else [],
        primary=labels[0],
    )


@pytest.fixture
def variants_tsv(tmp_path):
    df = pd.DataFrame(VARIANTS)
    path = tmp_path / "variants.tsv"
    df.to_csv(path, sep="\t", index=False, header=False)
    return str(path)


@pytest.fixture(autouse=True)
def mock_annotator(monkeypatch):
    def fake_batch(chroms, starts, ends):
        return [_ann(LABELS_BY_VARIANT[_POS_TO_ID[int(p)]]) for p in starts]

    monkeypatch.setattr(region_utils, "region_annotations_batch", fake_batch)


def _ids_in(out_dir, category):
    path = out_dir / f"{category}.tsv"
    if path.stat().st_size == 0:
        return set()
    df = pd.read_csv(path, sep="\t", header=None)
    return set(df.iloc[:, 4])  # variant_id is the 5th column


def test_overlapping_routing(variants_tsv, tmp_path):
    out_dir = tmp_path / "regions"
    counts = vrf.filter_variants_by_region(variants_tsv, str(out_dir), batch_size=2)

    expected = {
        "coding": {"v_cds"},
        "exonic": {"v_cds", "v_5utr"},
        "five_prime_utr": {"v_5utr"},
        "three_prime_utr": set(),
        "splice": {"v_cds"},
        "splice_site": {"v_cds"},
        "splice_region": {"v_cds"},
        "intronic": {"v_intron"},
        "promoter": {"v_prom"},
        "noncoding_gene": set(),
        "genic": {"v_cds", "v_intron", "v_5utr"},  # promoter & intergenic excluded
    }
    for cat, ids in expected.items():
        assert _ids_in(out_dir, cat) == ids, cat
        assert counts[cat] == len(ids), cat


def test_intergenic_off_by_default(variants_tsv, tmp_path):
    out_dir = tmp_path / "regions"
    vrf.filter_variants_by_region(variants_tsv, str(out_dir))
    assert not (out_dir / "intergenic.tsv").exists()


def test_intergenic_when_requested(variants_tsv, tmp_path):
    out_dir = tmp_path / "regions"
    counts = vrf.filter_variants_by_region(
        variants_tsv, str(out_dir), categories=["intergenic", "genic"]
    )
    assert _ids_in(out_dir, "intergenic") == {"v_inter"}
    assert counts["intergenic"] == 1
    # only the two requested categories are written
    assert not (out_dir / "coding.tsv").exists()


def test_unknown_category_raises(variants_tsv, tmp_path):
    with pytest.raises(ValueError):
        vrf.filter_variants_by_region(
            variants_tsv, str(tmp_path / "r"), categories=["bogus"]
        )
