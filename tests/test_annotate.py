"""Tests that variant_id is threaded through the annotation utility.

The data-backed lookups (region model, nearest genes, cCREs, allele freqs) are
stubbed so annotation runs without the built artifacts — we only care that a
custom variant_id flows from input to the annotated record.
"""
import json

import pytest

import varscore.annotation.maf as maf
import varscore.annotation.regions as region_utils
from varscore.annotation.annotate import (
    AnnotatedVariant,
    VariantAnnotationInput,
    annotate_variant,
    annotate_variants_file,
)


@pytest.fixture(autouse=True)
def stub_annotation_sources(monkeypatch):
    monkeypatch.setattr(
        region_utils, "region_annotations",
        lambda *a, **k: region_utils.RegionAnnotation(
            labels=["intergenic"], gene_ids=[], primary="intergenic"
        ),
    )
    monkeypatch.setattr(region_utils, "nearest_genes", lambda *a, **k: ([], False))
    monkeypatch.setattr(region_utils, "ccre_overlap", lambda *a, **k: None)
    monkeypatch.setattr(maf, "get_ot_variant", lambda *a, **k: None)


def test_annotate_variant_carries_custom_id():
    av = annotate_variant(
        VariantAnnotationInput(
            chr="chr1", pos=100, ref="A", alt="T", variant_id="rs_custom"
        )
    )
    assert isinstance(av, AnnotatedVariant)
    assert av.variant_id == "rs_custom"


def test_annotate_variant_id_defaults_to_none():
    av = annotate_variant(VariantAnnotationInput(chr="chr1", pos=100, ref="A", alt="T"))
    assert av.variant_id is None


def test_annotate_file_preserves_variant_id(tmp_path):
    tsv = tmp_path / "variants.tsv"
    # row 1 carries a custom id; row 2 has none (blank 5th field)
    tsv.write_text("chr1\t100\tA\tT\trs_custom\nchr1\t200\tC\tG\t\n")
    out = tmp_path / "annotated.jsonl"

    annotate_variants_file(str(tsv), str(out))

    records = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["variant_id"] == "rs_custom"
    assert records[1]["variant_id"] is None
