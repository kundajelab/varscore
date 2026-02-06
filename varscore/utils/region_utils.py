from functools import lru_cache
from typing import List, Tuple
from intervaltree import IntervalTree
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
# Lazy-loaded DNATrees (loaded on first use)

@lru_cache(maxsize=1)
def _get_promoter_dnatree():
    print("Loading Promoter DNATree ...")
    return loadDNATree(
        os.path.join(
            os.path.dirname(__file__), "..", "data", "promoters_proteincoding.dnatree"
        )
    )


@lru_cache(maxsize=1)
def _get_gene_dnatree():
    print("Loading Gene DNATree ...")
    return loadDNATree(
        os.path.join(os.path.dirname(__file__), "..", "data", "genes_proteincoding.dnatree")
    )


@lru_cache(maxsize=1)
def _get_exon_dnatree():
    print("Loading Exon DNATree ...")
    return loadDNATree(
        os.path.join(os.path.dirname(__file__), "..", "data", "exons_proteincoding.dnatree")
    )


@lru_cache(maxsize=1)
def _get_ccre_dnatree():
    print("Loading CCRE DNATree ...")
    filepath = os.path.join(os.path.dirname(__file__), "..", "data", "ccres.dnatree")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"CCRE DNATree file not found. Please see the README for instructions on how to construct the DNATree.")
    return loadDNATree(filepath)

def region_type(chro, start, end):
    if _get_promoter_dnatree().overlap(chro, start, end) is not None:
        return "promoter"
    elif _get_exon_dnatree().overlap(chro, start, end) is not None:
        return "exonic"
    elif _get_gene_dnatree().overlap(chro, start, end) is not None:
        return "intronic"
    return "intergenic"


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
