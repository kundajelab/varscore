# VariantScoringFunctions
Core functions for variant scoring

## Installing the library

The core library (variant annotation, region classification, AlphaMissense /
parquet lookups, prioritization) is TensorFlow-free and installs on macOS arm64
and modern Python:

```bash
pip install varscore
```

Conservation functionality lives behind an extra. The legacy `model` extra remains temporarily for direct
ChromBPNet installation compatibility, but production ML environments are independently locked under
`images/`:

```bash
pip install "varscore[model]"         # ChromBPNet model scoring + SHAP (legacy TensorFlow stack; Python < 3.10)
pip install "varscore[conservation]"  # CADD / PhyloP conservation lookups (pysam, pyBigWig)
```

Bulk reference data is **not** bundled — build it with the
`varscore/scripts/download_*` + `construct_*` pairs (see below and the
per-dataset docs). For model execution, use the independently locked
[ChromBPNet and Cherimoya images](docs/docker.md).

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

- [Model images](docs/docker.md) — independently locked ChromBPNet and Cherimoya runtimes
- [Region classification](docs/region_classification.md) — region labels, setup, and scorer routing
- [AlphaMissense](docs/alphamissense.md) — setup and variant scoring

## Model images

Each image has its own project and lockfile so incompatible ML stacks never share one dependency resolution.
See [docs/docker.md](docs/docker.md) for build, run, and runtime-mount details.

```bash
docker build -f images/chrombpnet/Dockerfile -t kundajelab/varscore:dev .
docker build -f images/cherimoya/Dockerfile -t kundajelab/cherimoya:dev .
docker run --rm kundajelab/varscore:dev varscore.preprocessing.region_filter --help  # sanity check before pushing
```
