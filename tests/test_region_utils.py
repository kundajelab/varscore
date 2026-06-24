"""Unit tests for region classification (varscore.annotation.regions).

These build a tiny in-memory interval index so they don't depend on the full
region_annotations.parquet build.
"""
import pandas as pd
import pytest

import varscore.annotation.regions as ru


def _index(rows):
    """Build a _RegionIndex from (chrom, start, end, feature, biotype, gene_id) tuples."""
    df = pd.DataFrame(
        rows, columns=["chrom", "start", "end", "feature", "biotype", "gene_id"]
    )
    return ru._RegionIndex(df)


def test_labels_for_feature_mapping():
    assert ru._labels_for_feature("exon", "protein_coding") == ("exonic",)
    assert ru._labels_for_feature("exon", "lncRNA") == ("noncoding_gene",)
    assert ru._labels_for_feature("splice_donor", "protein_coding") == (
        "splice_site", "splice_region",
    )
    assert ru._labels_for_feature("splice_acceptor", "x") == ("splice_site", "splice_region")
    assert ru._labels_for_feature("intron", "protein_coding") == ("intronic",)
    assert ru._labels_for_feature("cds", "protein_coding") == ("cds",)
    assert ru._labels_for_feature("promoter", "miRNA") == ("promoter",)


def test_inclusive_boundaries():
    # interval [10, 20] inclusive; NCLS is half-open internally.
    idx = _index([("chr1", 10, 20, "cds", "protein_coding", "G1")])
    assert idx.annotate(["chr1"], [10], [10])[0].primary == "cds"  # left edge
    assert idx.annotate(["chr1"], [20], [20])[0].primary == "cds"  # right edge
    assert idx.annotate(["chr1"], [9], [9])[0].primary == "intergenic"
    assert idx.annotate(["chr1"], [21], [21])[0].primary == "intergenic"


def test_multilabel_and_severity():
    # A position that is simultaneously CDS and a splice region (exon-edge case).
    idx = _index([
        ("chr1", 100, 200, "cds", "protein_coding", "G1"),
        ("chr1", 100, 200, "exon", "protein_coding", "G1"),
        ("chr1", 195, 205, "splice_region", "protein_coding", "G1"),
    ])
    ann = idx.annotate(["chr1"], [198], [198])[0]
    assert set(ann.labels) == {"cds", "exonic", "splice_region"}
    # severity: splice_site > cds > splice_region, so cds wins here
    assert ann.primary == "cds"
    assert ann.is_coding is True
    assert ann.gene_ids == ["G1"]


def test_splice_site_outranks_cds():
    idx = _index([
        ("chr1", 100, 200, "cds", "protein_coding", "G1"),
        ("chr1", 198, 199, "splice_donor", "protein_coding", "G1"),
    ])
    ann = idx.annotate(["chr1"], [198], [198])[0]
    assert "splice_site" in ann.labels and "cds" in ann.labels
    assert ann.primary == "splice_site"


def test_multi_gene_overlap():
    idx = _index([
        ("chr1", 1, 100, "exon", "protein_coding", "G1"),
        ("chr1", 50, 150, "exon", "lncRNA", "G2"),
    ])
    ann = idx.annotate(["chr1"], [75], [75])[0]
    assert set(ann.labels) == {"exonic", "noncoding_gene"}
    assert ann.gene_ids == ["G1", "G2"]


def test_batch_alignment_and_intergenic():
    idx = _index([
        ("chr1", 10, 20, "cds", "protein_coding", "G1"),
        ("chr2", 10, 20, "intron", "protein_coding", "G2"),
    ])
    res = idx.annotate(["chr1", "chr9", "chr2"], [15, 15, 15], [15, 15, 15])
    assert [a.primary for a in res] == ["cds", "intergenic", "intronic"]
    # unknown chromosome -> intergenic, not an error
    assert res[1].labels == ["intergenic"]
    assert res[1].gene_ids == []


def test_in_promoter_independent_of_primary():
    # TSS-proximal variant: in the promoter window AND the first exon. `primary`
    # collapses to the more-severe "exonic", but in_promoter must still be True
    # (this is what the prioritization filter gates on, not region_type).
    idx = _index([
        ("chr1", 100, 300, "promoter", "protein_coding", "G1"),
        ("chr1", 250, 400, "exon", "protein_coding", "G1"),
    ])
    ann = idx.annotate(["chr1"], [275], [275])[0]
    assert set(ann.labels) == {"promoter", "exonic"}
    assert ann.primary == "exonic"      # severity collapse hides promoter
    assert ann.in_promoter is True      # membership flag does not
    # purely upstream position: promoter only
    upstream = idx.annotate(["chr1"], [150], [150])[0]
    assert upstream.primary == "promoter" and upstream.in_promoter is True


def test_indel_span_overlap():
    # A deletion spanning into an exon should be caught via its [start, end] span.
    idx = _index([("chr1", 100, 200, "cds", "protein_coding", "G1")])
    ann = idx.annotate(["chr1"], [98], [105])[0]  # ref length 8, overlaps exon
    assert ann.is_coding is True
