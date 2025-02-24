import logging
from intervaltree import IntervalTree
import numpy as np
import pandas as pd
import requests
from pydantic import BaseModel

import os
import time
import asyncio
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
PROMOTER_DNATREE = loadDNATree(
    os.path.join(
        os.path.dirname(__file__), "..", "data", "promoters_proteincoding.dnatree"
    )
)
GENE_DNATREE = loadDNATree(
    os.path.join(os.path.dirname(__file__), "..", "data", "genes_proteincoding.dnatree")
)
EXON_DNATREE = loadDNATree(
    os.path.join(os.path.dirname(__file__), "..", "data", "exons_proteincoding.dnatree")
)
CCRE_DNATREE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ccres.dnatree")
CCRE_DNATREE = loadDNATree(CCRE_DNATREE_PATH) if os.path.exists(CCRE_DNATREE_PATH) else None


def region_type(chro, start, end):
    if PROMOTER_DNATREE.overlap(chro, start, end) is not None:
        return "promoter"
    elif EXON_DNATREE.overlap(chro, start, end) is not None:
        return "exonic"
    elif GENE_DNATREE.overlap(chro, start, end) is not None:
        return "intronic"
    return "intergenic"


################
# NEAREST GENE #
################
def __load_genes_by_chro():
    gene_df = pd.read_csv(GENE_DF_LOC, sep="\t")
    gene_df = gene_df[gene_df["gene_type"] == "protein_coding"]
    return {chro: gene_df[gene_df["chro"] == chro] for chro in set(gene_df["chro"])}


GENE_DF_LOC = os.path.join(os.path.dirname(__file__), "..", "data", "gene_df.tsv")
GENES_BY_CHRO = __load_genes_by_chro()


def nearest_genes(chro, pos, num_genes=5):
    genes_chro = GENES_BY_CHRO[chro].copy()
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
        [
            genes_chro.loc[i, "gene"],
            genes_chro.loc[i, "gene_id"],
            genes_chro.loc[i, "signed_dist"],
            genes_chro.loc[i, "gene_type"],
        ]
        for i in range(num_genes)
    ]
    gene_within_100kb = genes_chro.loc[0, "dist"] <= 100000
    return nearest_genes, gene_within_100kb


################
# CCRE Overlap #
################

query = """
query GetCCREs(
  $coordinates: [GenomicRangeInput!]
  $assembly: String!
) {
  cCREQuery(
    coordinates: $coordinates
    assembly: $assembly
  ) {
    accession
    group
  }
}
"""
screen_base_url = "https://factorbook.api.wenglab.org/graphql"
session = requests.Session()
last_request_time = 0
RATE_LIMIT_SECONDS = 0.5


class CCRE(BaseModel):
    accession: str
    group: str


def ccre_overlap(chr, start, end):
    """
    Calculates variant overlap with cCREs, using a local DNATree if available, or querying the Factorbook API if not.
    """
    if CCRE_DNATREE:
        element = CCRE_DNATREE.overlap(chr, start, end)
        if element is None:
            return None
        data = element[0][3]
        return CCRE(
            accession=data[0],
            group=data[1]
        )
    logger = logging.getLogger(__name__)
    logger.warning("WARNING: CCRE DNATree not found. Querying Factorbook API. Please execute `varscore.scripts.construct_ccre_dnatree` to construct the DNATree.")
    global last_request_time
    now = time.time()

    # Ensure we wait if the last request was too recent
    time_since_last = now - last_request_time
    if time_since_last < RATE_LIMIT_SECONDS:
        time.sleep(RATE_LIMIT_SECONDS - time_since_last)

    coordinates = [{"chromosome": chr, "start": start, "end": end}]
    variables = {"coordinates": coordinates, "assembly": "GRCh38"}
    response = requests.post(
        screen_base_url, json={"query": query, "variables": variables}
    )
    last_request_time = time.time()  # Update last request timestamp
    ccre_data = response.json()["data"]["cCREQuery"]
    if len(ccre_data) == 0:
        return None
    return CCRE(**ccre_data[0])

if __name__ == "__main__":
    print(ccre_overlap("chr1", 58046520, 58046530))
