# AlphaMissense

[AlphaMissense](https://www.science.org/doi/10.1126/science.adg7492) provides
precomputed missense pathogenicity scores from DeepMind. varscore stores them as
a chr-partitioned Parquet dataset for fast local lookups via DuckDB, and exposes
a scoring entrypoint to annotate variant files.

## Setup

1. Download the AlphaMissense hg38 score file (~640MB) from the public GCS bucket:
```bash
./varscore/scripts/download_alphamissense.sh
```

2. Rewrite it into a chr-partitioned Parquet dataset (default output:
   `varscore/data/alphamissense/CHROM=chr*/`):
```bash
uv run python -m varscore.scripts.construct_alphamissense_parquet
```

The dataset is Hive-partitioned on chromosome and sorted by position within each
partition, so a lookup only scans the partitions for the chromosomes it needs.

To write the dataset somewhere else, pass `-o/--out_path`. Whatever location you
use here is the same one you point scoring at with `-d/--alphamissense_dir` (see
below):
```bash
uv run python -m varscore.scripts.construct_alphamissense_parquet -o /data/alphamissense
```

## Scoring variants

Annotate a TSV of variants (no header; columns `chr, pos, ref, alt[, variant_id]`)
with AlphaMissense scores:

```bash
uv run python -m varscore.alphamissense_scoring -v variants.tsv -o alphamissense_scores.tsv
```

The output TSV has the original variant columns plus the AlphaMissense columns
(`genome, uniprot_id, transcript_id, protein_variant, am_pathogenicity, am_class`)
appended for each match.

Notes:
- Variants with no AlphaMissense entry are kept with empty score columns.
- A variant that maps to multiple transcripts (overlapping genes) produces one
  output row per matching transcript, so the output is not necessarily 1:1 with
  the input.
- Use `-d/--alphamissense_dir` to point at a non-default Parquet directory.

## Python API

The lookup is also available as a function for use on an in-memory DataFrame:

```python
import pandas as pd
from varscore.utils.alphamissense_utils import lookup_alphamissense

variants = pd.DataFrame({"chr": ["chr1"], "pos": [69094], "ref": ["G"], "alt": ["T"]})
scored = lookup_alphamissense(variants)
```
