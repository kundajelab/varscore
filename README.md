# VariantScoringFunctions
Core functions for variant scoring

## Installing the library

The core library (variant annotation, region classification, AlphaMissense /
parquet lookups, prioritization) is TensorFlow-free and installs on macOS arm64
and modern Python:

```bash
pip install varscore
```

Heavy / platform-constrained functionality lives behind extras:

```bash
pip install "varscore[model]"         # ChromBPNet model scoring + SHAP (legacy TensorFlow stack; Python < 3.10)
pip install "varscore[conservation]"  # CADD / PhyloP conservation lookups (pysam, pyBigWig)
```

Bulk reference data is **not** bundled — build it with the
`varscore/scripts/download_*` + `construct_*` pairs (see below and the
per-dataset docs). For environments that can't satisfy the model extra (e.g.
Kubernetes), use the published [Docker image](docs/docker.md) instead, which
bundles the TensorFlow stack and the region-annotation data.

## Development setup
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

- [Docker](docs/docker.md) — building/running the published image and its data
- [Region classification](docs/region_classification.md) — region labels, setup, and scorer routing
- [AlphaMissense](docs/alphamissense.md) — setup and variant scoring

## Docker

A published image bundles the TensorFlow / ChromBPNet stack and the
region-annotation data so the full pipeline runs in environments that can't
install the pinned TF versions (k8s, macOS arm64, modern Python). See
[docs/docker.md](docs/docker.md) for build, run, and runtime-mount details.

```bash
docker build -t kundajelab/varscore:dev -f Dockerfile .
docker run --rm kundajelab/varscore:dev varscore.variant_region_filter --help  # sanity check before pushing
docker push kundajelab/varscore:dev
```
