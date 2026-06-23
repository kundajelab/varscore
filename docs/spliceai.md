# SpliceAI

[SpliceAI](https://github.com/Illumina/SpliceAI) predicts splice-altering
variants. Rather than running the (TensorFlow, GPU-bound) model, varscore uses
Illumina's **precomputed** delta scores for all possible SNVs and small indels,
stored as a chr-partitioned Parquet dataset for fast local lookups via DuckDB,
and exposes a scoring entrypoint to annotate variant files — mirroring the
[AlphaMissense](alphamissense.md) setup.

The precomputed scores are released under **CC BY-NC 4.0** (academic /
non-commercial use only) and cover all substitutions, 1nt insertions, and 1–4nt
deletions within genes, against an older GENCODE annotation.

## Setup

1. Download the hg38 raw SpliceAI score VCFs (SNV + indel). These live on
   Illumina BaseSpace behind a free login, so the script is a guided wrapper
   around the BaseSpace CLI rather than a one-shot download:
```bash
./varscore/scripts/download_spliceai.sh
```
   Follow its instructions to land these files in `varscore/data/raw/`:
   `spliceai_scores.raw.snv.hg38.vcf.gz` and
   `spliceai_scores.raw.indel.hg38.vcf.gz`.

2. Parse the VCFs into a chr-partitioned Parquet dataset (default output:
   `varscore/data/spliceai/CHROM=chr*/`):
```bash
uv run python -m varscore.scripts.construct_spliceai_parquet
```

The SpliceAI INFO field
(`SpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL`) is
split into columns, chromosomes are normalized to a `chr` prefix, and rows are
sorted by position within each Hive partition, so a lookup only scans the
partitions for the chromosomes it needs.

Pass `-i` to point at different/extra VCFs (e.g. only SNVs, or hg19) and
`-o` to write the dataset elsewhere; whatever location you use is the one you
point scoring at with `-d/--spliceai_dir`:
```bash
uv run python -m varscore.scripts.construct_spliceai_parquet \
    -i varscore/data/raw/spliceai_scores.raw.snv.hg38.vcf.gz \
       varscore/data/raw/spliceai_scores.raw.indel.hg38.vcf.gz \
    -o /data/spliceai
```

## Scoring variants

Annotate a TSV of variants (no header; columns `chr, pos, ref, alt[, variant_id]`)
with SpliceAI scores:

```bash
uv run python -m varscore.spliceai_scoring -v variants.tsv -o spliceai_scores.tsv
```

The output TSV has the original variant columns plus the SpliceAI columns
appended for each match:
- `spliceai_symbol` — gene the prediction is for
- `ds_ag, ds_al, ds_dg, ds_dl` — delta scores (0–1) for acceptor gain/loss and
  donor gain/loss
- `dp_ag, dp_al, dp_dg, dp_dl` — delta positions (offset to the affected site)
- `ds_max` — max of the four delta scores; the value usually thresholded for
  prioritization (e.g. ≥0.2 / 0.5 / 0.8)

Notes:
- Variants with no SpliceAI entry are kept with empty score columns.
- Each precomputed record carries a single gene annotation, so matches are
  typically 1:1; overlapping-gene loci with multiple records can produce more
  than one output row.
- Use `-d/--spliceai_dir` to point at a non-default Parquet directory.

## Python API

The lookup is also available as a function for use on an in-memory DataFrame:

```python
import pandas as pd
from varscore.utils.spliceai_utils import lookup_spliceai

variants = pd.DataFrame({"chr": ["chr1"], "pos": [69091], "ref": ["A"], "alt": ["C"]})
scored = lookup_spliceai(variants)
```
