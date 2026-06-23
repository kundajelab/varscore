"""Build the region-annotation interval table from an Ensembl GFF3.

Parses the Ensembl gene-model GFF3 into a flat, strand-aware interval table and
writes it as parquet. Each row is one labelled genomic interval (1-based,
endpoint-inclusive) carrying its feature type, gene, and biotype. Introns and
splice sites are *derived* from transcript structure here so that downstream
classification is a pure interval-overlap join (see region_utils.region_annotations).

Feature types emitted (the `feature` column):
    cds, five_prime_utr, three_prime_utr, exon,
    intron, splice_donor, splice_acceptor, splice_region, promoter

Usage:
    python -m varscore.scripts.construct_region_annotations \
        -i varscore/data/raw/Homo_sapiens.GRCh38.116.gff3.gz \
        -o varscore/data/region_annotations.parquet
"""
import argparse
import gzip
import os
import re

import numpy as np
import pandas as pd

# --- windows (base pairs); see docs/region_classification.md -----------------
SPLICE_SITE_BP = 2        # canonical donor/acceptor (intronic) — VEP definition
SPLICE_REGION_INTRON_BP = 8
SPLICE_REGION_EXON_BP = 3
PROMOTER_UPSTREAM = 2000
PROMOTER_DOWNSTREAM = 200

# Standard Ensembl seqids -> UCSC-style chrom names used elsewhere in varscore.
_STD_CHROMS = {str(i): f"chr{i}" for i in range(1, 23)}
_STD_CHROMS.update({"X": "chrX", "Y": "chrY", "MT": "chrM"})

_ATTR_RE = {
    "ID": re.compile(r"(?:^|;)ID=([^;]+)"),
    "Parent": re.compile(r"(?:^|;)Parent=([^;]+)"),
    "biotype": re.compile(r"(?:^|;)biotype=([^;]+)"),
    "Name": re.compile(r"(?:^|;)Name=([^;]+)"),
}


def _read_gff3(gff_path: str) -> pd.DataFrame:
    """Load a GFF3 into a DataFrame, keeping only standard chromosomes."""
    opener = gzip.open if gff_path.endswith(".gz") else open
    cols = ["seqid", "source", "feature", "start", "end",
            "score", "strand", "phase", "attrs"]
    with opener(gff_path, "rt") as fh:
        df = pd.read_csv(
            fh, sep="\t", comment="#", header=None, names=cols,
            dtype={"seqid": str, "start": np.int64, "end": np.int64,
                   "strand": str, "feature": str, "attrs": str},
            usecols=["seqid", "feature", "start", "end", "strand", "attrs"],
        )
    df = df[df["seqid"].isin(_STD_CHROMS)].copy()
    df["chrom"] = df["seqid"].map(_STD_CHROMS)
    for key, rx in _ATTR_RE.items():
        df[key] = df["attrs"].str.extract(rx, expand=False)
    return df.drop(columns=["attrs", "seqid"])


def _strip(prefix_series: pd.Series) -> pd.Series:
    """Strip the `type:` prefix Ensembl puts on ID/Parent (e.g. gene:ENSG...)."""
    return prefix_series.str.split(":", n=1).str[-1]


def _build_lookups(df: pd.DataFrame):
    """Return (tx2gene, gene2biotype, gene2name) mappings keyed by stable id.

    Biotype is taken from the *gene* (not the transcript): NMD / retained-intron
    transcripts belong to protein-coding genes, so the gene biotype is what
    distinguishes coding loci from genuinely non-coding genes (lncRNA, miRNA...).
    """
    genes = df[df["feature"].str.endswith("gene", na=False)].copy()
    genes["gene_id"] = _strip(genes["ID"])
    gene2name = dict(zip(genes["gene_id"], genes["Name"].fillna(genes["gene_id"])))
    gene2biotype = dict(zip(genes["gene_id"], genes["biotype"]))

    tx = df[df["ID"].str.startswith("transcript:", na=False)].copy()
    tx["tx_id"] = _strip(tx["ID"])
    tx["gene_id"] = _strip(tx["Parent"])
    tx2gene = dict(zip(tx["tx_id"], tx["gene_id"]))
    return tx2gene, gene2biotype, gene2name


def _annotate_tx_children(df: pd.DataFrame, features, tx2gene, gene2biotype, gene2name):
    """Slice rows whose Parent is a transcript and attach gene/biotype columns."""
    sub = df[df["feature"].isin(features) & df["Parent"].notna()].copy()
    sub["tx_id"] = _strip(sub["Parent"])
    sub = sub[sub["tx_id"].isin(tx2gene)]
    sub["gene_id"] = sub["tx_id"].map(tx2gene)
    sub["biotype"] = sub["gene_id"].map(gene2biotype)
    sub["gene_name"] = sub["gene_id"].map(gene2name)
    return sub


def _emit(chrom, start, end, strand, feature, gene_id, gene_name, biotype):
    """Assemble a feature-table fragment as a DataFrame (vectorized inputs)."""
    return pd.DataFrame({
        "chrom": chrom, "start": start, "end": end, "strand": strand,
        "feature": feature, "gene_id": gene_id, "gene_name": gene_name,
        "biotype": biotype,
    })


def _derive_introns_and_splices(exons: pd.DataFrame) -> pd.DataFrame:
    """Derive intron and splice-site intervals from per-transcript exon structure."""
    ex = exons.sort_values(["tx_id", "start"]).reset_index(drop=True)
    # Next exon's start within the same transcript -> defines the intron gap.
    nxt_start = ex.groupby("tx_id")["start"].shift(-1)
    nxt_tx = ex["tx_id"].shift(-1)
    has_intron = (ex["tx_id"] == nxt_tx) & (nxt_start > ex["end"] + 1)

    g = ex[has_intron].copy()
    a = (g["end"] + 1).to_numpy()              # intron 5'-most genomic base
    b = (nxt_start[has_intron] - 1).to_numpy()  # intron 3'-most genomic base
    plus = (g["strand"] == "+").to_numpy()
    chrom = g["chrom"].to_numpy()
    strand = g["strand"].to_numpy()
    gid, gname, bt = g["gene_id"].to_numpy(), g["gene_name"].to_numpy(), g["biotype"].to_numpy()

    frames = [_emit(chrom, a, b, strand, "intron", gid, gname, bt)]

    # Canonical 2bp donor/acceptor sites at each intron end (strand decides which).
    a_site_s, a_site_e = a, np.minimum(a + SPLICE_SITE_BP - 1, b)
    b_site_s, b_site_e = np.maximum(b - SPLICE_SITE_BP + 1, a), b
    donor_s = np.where(plus, a_site_s, b_site_s)
    donor_e = np.where(plus, a_site_e, b_site_e)
    accept_s = np.where(plus, b_site_s, a_site_s)
    accept_e = np.where(plus, b_site_e, a_site_e)
    frames.append(_emit(chrom, donor_s, donor_e, strand, "splice_donor", gid, gname, bt))
    frames.append(_emit(chrom, accept_s, accept_e, strand, "splice_acceptor", gid, gname, bt))

    # Broader splice region: 3bp into exon + 8bp into intron around both ends.
    left_s = (g["end"] - SPLICE_REGION_EXON_BP + 1).to_numpy()
    left_e = np.minimum(a + SPLICE_REGION_INTRON_BP - 1, b)
    right_s = np.maximum(b - SPLICE_REGION_INTRON_BP + 1, a)
    right_e = (nxt_start[has_intron] + SPLICE_REGION_EXON_BP - 1).to_numpy()
    frames.append(_emit(chrom, left_s, left_e, strand, "splice_region", gid, gname, bt))
    frames.append(_emit(chrom, right_s, right_e, strand, "splice_region", gid, gname, bt))
    return pd.concat(frames, ignore_index=True)


def _derive_promoters(df: pd.DataFrame, gene2name) -> pd.DataFrame:
    """Promoter = TSS-upstream window per gene (strand-aware)."""
    genes = df[df["feature"].str.endswith("gene", na=False)].copy()
    genes["gene_id"] = _strip(genes["ID"])
    plus = (genes["strand"] == "+").to_numpy()
    tss = np.where(plus, genes["start"].to_numpy(), genes["end"].to_numpy())
    start = np.where(plus, tss - PROMOTER_UPSTREAM, tss - PROMOTER_DOWNSTREAM)
    end = np.where(plus, tss + PROMOTER_DOWNSTREAM, tss + PROMOTER_UPSTREAM)
    start = np.maximum(start, 1)
    return _emit(
        genes["chrom"].to_numpy(), start, end, genes["strand"].to_numpy(),
        "promoter", genes["gene_id"].to_numpy(),
        genes["gene_id"].map(gene2name).to_numpy(), genes["biotype"].to_numpy(),
    )


def construct_region_annotations(gff_path: str, out_path: str) -> pd.DataFrame:
    print(f"Reading GFF3: {gff_path}")
    df = _read_gff3(gff_path)
    tx2gene, gene2biotype, gene2name = _build_lookups(df)

    print("Assembling exon / CDS / UTR features ...")
    exons = _annotate_tx_children(df, ["exon"], tx2gene, gene2biotype, gene2name)
    cds = _annotate_tx_children(df, ["CDS"], tx2gene, gene2biotype, gene2name)
    futr = _annotate_tx_children(df, ["five_prime_UTR"], tx2gene, gene2biotype, gene2name)
    tutr = _annotate_tx_children(df, ["three_prime_UTR"], tx2gene, gene2biotype, gene2name)

    base_cols = ["chrom", "start", "end", "strand", "feature",
                 "gene_id", "gene_name", "biotype"]
    parts = [
        _emit(exons["chrom"], exons["start"], exons["end"], exons["strand"],
              "exon", exons["gene_id"], exons["gene_name"], exons["biotype"]),
        _emit(cds["chrom"], cds["start"], cds["end"], cds["strand"],
              "cds", cds["gene_id"], cds["gene_name"], cds["biotype"]),
        _emit(futr["chrom"], futr["start"], futr["end"], futr["strand"],
              "five_prime_utr", futr["gene_id"], futr["gene_name"], futr["biotype"]),
        _emit(tutr["chrom"], tutr["start"], tutr["end"], tutr["strand"],
              "three_prime_utr", tutr["gene_id"], tutr["gene_name"], tutr["biotype"]),
    ]

    print("Deriving introns + splice sites ...")
    parts.append(_derive_introns_and_splices(exons))
    print("Deriving promoters ...")
    parts.append(_derive_promoters(df, gene2name))

    table = pd.concat([p[base_cols] for p in parts], ignore_index=True)
    table = table.dropna(subset=["chrom", "start", "end"])
    table["start"] = table["start"].astype(np.int64)
    table["end"] = table["end"].astype(np.int64)
    table = table[table["end"] >= table["start"]].reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    table.to_parquet(out_path, index=False)
    print(f"Wrote {len(table):,} intervals -> {out_path}")
    print(table["feature"].value_counts().to_string())
    return table


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--gff", required=True, help="Ensembl GFF3 (.gff3/.gff3.gz)")
    p.add_argument("-o", "--out", default="varscore/data/region_annotations.parquet")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    construct_region_annotations(args.gff, args.out)
