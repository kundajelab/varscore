from functools import lru_cache
from typing import List, Tuple
from intervaltree import IntervalTree
from ncls import NCLS
import numpy as np
import pandas as pd
from pydantic import BaseModel

import os
import pickle


############
# DNA Tree #
############
def loadDNATree(save_loc):
    with open(save_loc, "rb") as f:
        chro_trees = pickle.load(f)
    dnatree = DNATree()
    dnatree.chro_trees = chro_trees
    return dnatree


class DNATree:
    """An object for storing and looking up genetic regions.

    A DNATree object is built on top of an interval tree and is intended to
      maintain a collection of genetic regions and serve as an interface by
      which one can efficiently compute overlaps of those regions. The object
      can be loaded and saved using loadDNATree() and DNATree.save(). Regions
      can be added using the DNATree.add() function and specifying the
      chromosome of the region and the start and end locations of the region.
      Similarly, regions can be looked up using the DNATree.overlap() function
      using the same arguments.

      IMPORTANT: Regions are assumed to be endpoint inclusive as: [start, end].
        If you want a region to represent a single position for instance, the
        region should be specified as [pos, pos].

    Attributes:
        chro_trees: A dictionary of interval trees mapping a chromosome name to
          the interval tree representing regions on that chromosome.
    """

    def __init__(self):
        self.chro_trees = dict()

    def add(self, chro, start, end, data=""):
        assert data is not None
        if chro in self.chro_trees:
            self.chro_trees[chro][start : end + 1] = data
        else:
            self.chro_trees[chro] = IntervalTree()
            self.chro_trees[chro][start : end + 1] = data

    def overlap(self, chro, start, end):
        if chro not in self.chro_trees:
            return None
        chro_overlap = self.chro_trees[chro][start : end + 1]
        if len(chro_overlap) == 0:
            return None
        return [(chro, i.begin, i.end - 1, i.data) for i in chro_overlap]

    def save(self, name):
        savename = name if name.endswith(".dnatree") else f"{name}.dnatree"
        with open(savename, "wb") as f:
            pickle.dump(self.chro_trees, f)


###############
# REGION TYPE #
###############
# Region classification is a vectorized interval-overlap join against a flat
# annotation table (varscore/data/region_annotations.parquet, built by
# scripts/construct_region_annotations.py). See docs/region_classification.md.
#
# A variant can carry MULTIPLE region labels at once (e.g. a coding exon base
# that is also a splice site). `region_annotations()` returns the full set;
# `region_type()` is a back-compat shim returning the single most-severe label.

# Final labels, ordered most -> least severe for the single-label collapse.
REGION_LABELS_BY_SEVERITY = [
    "splice_site",
    "cds",
    "splice_region",
    "five_prime_utr",
    "three_prime_utr",
    "exonic",
    "noncoding_gene",
    "intronic",
    "promoter",
    "intergenic",
]
_CODING_LABEL = "cds"
_PROMOTER_LABEL = "promoter"

# Map raw annotation-table feature types -> emitted region label(s).
def _labels_for_feature(feature, biotype):
    if feature == "exon":
        return ("exonic",) if biotype == "protein_coding" else ("noncoding_gene",)
    if feature in ("splice_donor", "splice_acceptor"):
        return ("splice_site", "splice_region")
    if feature == "intron":
        return ("intronic",)
    return (feature,)  # cds / five_prime_utr / three_prime_utr / splice_region / promoter


class RegionAnnotation(BaseModel):
    labels: List[str]      # all overlapped region labels, sorted by severity
    gene_ids: List[str]    # gene ids the variant overlaps (multi-gene aware)
    primary: str           # most-severe single label (display / back-compat)

    @property
    def is_coding(self) -> bool:
        return _CODING_LABEL in self.labels

    @property
    def in_promoter(self) -> bool:
        return _PROMOTER_LABEL in self.labels


class _RegionIndex:
    """Per-chromosome NCLS index over the region-annotation interval table.

    Intervals in the table are 1-based, endpoint-inclusive; NCLS uses half-open
    [start, end), so ends are stored (and queried) as end + 1.
    """

    def __init__(self, table: pd.DataFrame):
        self._ncls = {}
        self._feature = {}
        self._biotype = {}
        self._gene_id = {}
        for chrom, g in table.groupby("chrom", sort=False):
            g = g.reset_index(drop=True)
            starts = g["start"].to_numpy(np.int64)
            ends = g["end"].to_numpy(np.int64) + 1  # inclusive -> half-open
            self._ncls[chrom] = NCLS(starts, ends, np.arange(len(g), dtype=np.int64))
            self._feature[chrom] = g["feature"].to_numpy()
            self._biotype[chrom] = g["biotype"].to_numpy()
            self._gene_id[chrom] = g["gene_id"].to_numpy()

    def _annotation_from_hits(self, chrom, target_ids) -> RegionAnnotation:
        feats = self._feature[chrom][target_ids]
        biots = self._biotype[chrom][target_ids]
        genes = self._gene_id[chrom][target_ids]
        labels = set()
        for f, b in zip(feats, biots):
            labels.update(_labels_for_feature(f, b))
        ordered = [l for l in REGION_LABELS_BY_SEVERITY if l in labels]
        gene_ids = sorted({g for g in genes if isinstance(g, str)})
        return RegionAnnotation(
            labels=ordered or ["intergenic"],
            gene_ids=gene_ids,
            primary=ordered[0] if ordered else "intergenic",
        )

    def annotate(self, chroms, starts, ends) -> List[RegionAnnotation]:
        """Vectorized lookup; returns a RegionAnnotation per input interval."""
        chroms = np.asarray(chroms, dtype=object)
        starts = np.asarray(starts, dtype=np.int64)
        ends = np.asarray(ends, dtype=np.int64) + 1  # inclusive -> half-open
        n = len(chroms)
        intergenic = RegionAnnotation(labels=["intergenic"], gene_ids=[], primary="intergenic")
        results = [intergenic] * n
        order = np.arange(n)
        for chrom in pd.unique(chroms):
            ncls = self._ncls.get(chrom)
            if ncls is None:
                continue
            sel = order[chroms == chrom]
            q_idx, t_idx = ncls.all_overlaps_both(
                starts[sel], ends[sel], np.arange(len(sel), dtype=np.int64)
            )
            if len(q_idx) == 0:
                continue
            # Group target rows by query position, then build one annotation each.
            sort = np.argsort(q_idx, kind="stable")
            q_sorted, t_sorted = q_idx[sort], t_idx[sort]
            bounds = np.searchsorted(q_sorted, np.unique(q_sorted))
            for k, q in enumerate(np.unique(q_sorted)):
                lo = bounds[k]
                hi = bounds[k + 1] if k + 1 < len(bounds) else len(q_sorted)
                results[sel[q]] = self._annotation_from_hits(chrom, t_sorted[lo:hi])
        return results


@lru_cache(maxsize=1)
def _get_region_index():
    print("Loading region annotations ...")
    path = os.path.join(
        os.path.dirname(__file__), "..", "data", "region_annotations.parquet"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            "region_annotations.parquet not found. Build it with "
            "scripts/download_ensembl_gff.sh + scripts/construct_region_annotations.py "
            "(see docs/region_classification.md)."
        )
    table = pd.read_parquet(
        path, columns=["chrom", "start", "end", "feature", "biotype", "gene_id"]
    )
    for col in ("feature", "biotype", "gene_id"):
        table[col] = table[col].astype("category")  # dedupe strings to cut memory
    return _RegionIndex(table)


@lru_cache(maxsize=1)
def _get_ccre_dnatree():
    print("Loading CCRE DNATree ...")
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "ccres.dnatree")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CCRE DNATree file not found. Please see the README for instructions on how to construct the DNATree.")
    return loadDNATree(filepath)

def region_annotations(chro, start, end) -> RegionAnnotation:
    """Full multi-label region annotation for a single interval (1-based, incl)."""
    return _get_region_index().annotate([chro], [start], [end])[0]


def region_annotations_batch(chroms, starts, ends) -> List[RegionAnnotation]:
    """Vectorized region annotation for many intervals; aligned to input order."""
    return _get_region_index().annotate(chroms, starts, ends)


def region_type(chro, start, end) -> str:
    """Back-compat single-label classification (most-severe label)."""
    return region_annotations(chro, start, end).primary


################
# NEAREST GENE #
################
@lru_cache(maxsize=1)
def _get_genes_by_chro():
    print("Loading gene df ...")
    gene_df_loc = os.path.join(os.path.dirname(__file__), "..", "data", "gene_df.tsv")
    gene_df = pd.read_csv(gene_df_loc, sep="\t")
    gene_df = gene_df[gene_df["gene_type"] == "protein_coding"]
    return {chro: gene_df[gene_df["chro"] == chro] for chro in set(gene_df["chro"])}

class GeneAnnotation(BaseModel):
    gene_name: str
    gene_id: str
    distance: int
    type: str

def nearest_genes(chro, pos, num_genes=5) -> Tuple[List[GeneAnnotation], bool]:
    genes_chro = _get_genes_by_chro()[chro].copy()
    strand_sign = 1 * (genes_chro["strand"] == "+") - 1 * (genes_chro["strand"] == "-")

    start_dist = (pos - genes_chro["start"]) * strand_sign
    end_dist = (pos - genes_chro["end"]) * strand_sign
    var_in_gene = 1 * (start_dist * end_dist <= 0)
    genes_chro["dist"] = np.minimum(np.abs(start_dist), np.abs(end_dist)) * (
        1 - var_in_gene
    )
    genes_chro["signed_dist"] = genes_chro["dist"] * strand_sign * np.sign(start_dist)

    genes_chro = genes_chro.sort_values(by="dist", ascending=True)
    genes_chro = genes_chro.reset_index(drop=True)

    nearest_genes = [
        GeneAnnotation(
            gene_name=genes_chro.loc[i, "gene"],
            gene_id=genes_chro.loc[i, "gene_id"],
            distance=genes_chro.loc[i, "signed_dist"],
            type=genes_chro.loc[i, "gene_type"],
        )
        for i in range(num_genes)
    ]
    gene_within_100kb = genes_chro.loc[0, "dist"] <= 100000
    return nearest_genes, gene_within_100kb


class CCRE(BaseModel):
    accession: str
    group: str


def ccre_overlap(chr, start, end):
    """
    Calculates variant overlap with cCREs, using a local DNATree if available, or querying the Factorbook API if not.
    """
    element = _get_ccre_dnatree().overlap(chr, start, end)
    if element is None:
        return None
    data = element[0][3]
    return CCRE(
        accession=data[0],
        group=data[1]
    )

if __name__ == "__main__":
    print(ccre_overlap("chr1", 58046520, 58046530))
