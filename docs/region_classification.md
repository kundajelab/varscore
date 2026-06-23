# Region Classification

varscore classifies each variant into genomic **region categories** (CDS, UTR,
splice site/region, intron, promoter, non-coding-gene body, intergenic) by pure
**interval overlap** against an Ensembl gene model. The labels are used to route
variants to the right scorer — coding variants to
[AlphaMissense](alphamissense.md), splice-region variants to
[SpliceAI](spliceai.md) — and for reporting.

## Design

- **Interval overlap only — no FASTA, no codon math.** The hard part of variant
  consequence (codon-level missense/stop calling, splice-disruption prediction)
  is already owned by the dedicated scorers (AlphaMissense, SpliceAI). The region
  classifier only needs to *categorize and route*, and every category here is
  derivable from interval overlap against a rich enough annotation.
- **Multi-label.** A variant can legitimately be several things at once (a coding
  base that is also a canonical splice site; exonic in one transcript and
  intronic in another). The classifier returns the full set of labels, not a
  single collapsed string.
- **Strand-aware.** Donor vs acceptor splice sites and 5′ vs 3′ UTR are resolved
  using transcript strand.

## Setup

1. Download the Ensembl gene-model GFF3 (default release 116, GRCh38):
```bash
./varscore/scripts/download_ensembl_gff.sh          # ENSEMBL_RELEASE=116 by default
```

2. Parse it into a flat interval table (parquet):
```bash
uv run python -m varscore.scripts.construct_region_annotations \
    -i varscore/data/raw/Homo_sapiens.GRCh38.116.gff3.gz \
    -o varscore/data/region_annotations.parquet
```

The output is a flat interval list — one row per labelled genomic interval
(`chrom, start, end, strand, feature, gene_id, gene_name, biotype`), 1-based and
endpoint-inclusive. `exon`, `cds`, `five_prime_utr`, and `three_prime_utr` come
straight from the GFF3; `intron`, `splice_donor`, `splice_acceptor`,
`splice_region`, and `promoter` are **derived** from per-transcript exon
structure so that lookup is a single overlap join. The table (~32M intervals) is
gitignored.

Window definitions (constants in
[construct_region_annotations.py](../varscore/scripts/construct_region_annotations.py)):

| Feature | Window |
|---|---|
| `splice_donor` / `splice_acceptor` | 2 bp intronic (canonical, VEP definition) |
| `splice_region` | 8 bp intronic + 3 bp exonic around each exon boundary |
| `promoter` | 2000 bp upstream / 200 bp downstream of the TSS |

## Labels

Returned labels, ordered most → least severe (the order used for the single-label
collapse):

```
splice_site, cds, splice_region, five_prime_utr, three_prime_utr,
exonic, noncoding_gene, intronic, promoter, intergenic
```

- **`cds`** drives coding routing — a variant is *coding* iff it overlaps a CDS
  (`RegionAnnotation.is_coding`), and those go to AlphaMissense.
- An `exon` in a protein-coding gene → `exonic`; an `exon` in a non-coding gene
  (lncRNA, miRNA, …) → `noncoding_gene`, so non-coding loci are no longer
  mislabelled as `intergenic`.
- `intergenic` is the empty case (no overlap) and is excluded from both the
  coding and non-coding scorer outputs.

## Python API

```python
from varscore.utils.region_utils import (
    region_annotations,        # single interval -> RegionAnnotation
    region_annotations_batch,  # vectorized over many intervals (preferred at scale)
    region_type,               # back-compat shim -> single most-severe label (str)
)

ann = region_annotations("chr1", 69094, 69094)
ann.labels      # e.g. ["cds", "exonic"]  (severity-ordered)
ann.gene_ids    # overlapping gene IDs (multi-gene aware)
ann.primary     # "cds"  -> most-severe single label
ann.is_coding   # True   -> overlaps a CDS

# Annotate a whole variant set in one vectorized pass:
anns = region_annotations_batch(df["chr"], df["pos"], df["pos"])
```

`region_type()` is retained only for back-compatibility (returns `.primary`);
new code should use `region_annotations[_batch]` to get the full label set.

## Implementation notes

- **Index:** a per-chromosome [NCLS](https://github.com/pyranges/ncls)
  (C-backed nested containment list) queried with a vectorized
  `all_overlaps_both` join — annotate the whole DataFrame at once instead of an
  `iterrows` loop. NCLS is the core that `pyranges` wraps; it's used directly
  here because `pyranges` 1.x needs Python ≥3.10 (this project is on 3.9 +
  pandas 2).
- **Storage:** parquet, not a pickled tree — the NCLS index rebuilds cheaply from
  sorted arrays on load, and parquet is inspectable, compressed, and consistent
  with the AlphaMissense/SpliceAI datasets.
- **cCRE / regulatory:** still annotated on the existing `DNATree` path
  (`ccre_overlap`); folding the Ensembl Regulatory Build into a `regulatory`
  label is a planned follow-up.
