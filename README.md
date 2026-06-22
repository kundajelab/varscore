# VariantScoringFunctions
Core functions for variant scoring

## Installation
Make sure you have `uv` installed. See [here](https://docs.astral.sh/uv/) for installation instructions.

### Sync Dependencies

```bash
uv sync
```

### Setup

These annotations rely on large files of bulk reference data. These need to be constructed first.

#### CCREs

1. Download the CCRE bed file:
```bash
./varscore/scripts/download_ccres.sh
```

2. Run the following script to construct the DNATree from the CCRE bed file:
```bash
uv run python -m varscore.scripts.construct_ccre_dnatree
```

#### Variants

(Specifically, minor allele frequencies for variants)

1. Download the OpenTargets variant files
```bash
./varscore/scripts/download_variants.sh
```

2. Run the following script to construct the variants dataframe from the OpenTargets variant files:
```bash
uv run python -m varscore.scripts.construct_variants_df
```


## Documentation

- [AlphaMissense](docs/alphamissense.md) — setup and variant scoring


## Building Docker

Build:

```bash
docker build -t kundajelab/varscore:dev -f Dockerfile .
```

Sanity check to make sure it works (PLEASE DO BEFORE PUSHING):

```bash
docker run --rm kundajelab/varscore:dev varscore.ingest_model
``` 

Push:

```bash
docker push kundajelab/varscore:dev
```
