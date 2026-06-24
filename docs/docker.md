# Docker

varscore publishes a Docker image so the full pipeline — including the legacy
TensorFlow / ChromBPNet model scoring — runs in environments that can't install
those pinned versions directly (Kubernetes, macOS arm64, modern Python). For a
pure-Python, TensorFlow-free install instead, see the library
[install instructions](../README.md#installing-the-library).

## What's in the image

- **Base:** `kundajelab/chrombpnet:latest`, which provides the TensorFlow /
  ChromBPNet stack.
- **varscore** installed with the `[model]` extra, so its dependencies resolve
  against the TF-compatible pins (notably `numpy<1.24`) rather than upgrading
  numpy out from under TensorFlow 2.8.
- **Baked reference data:** the **region-annotation interval table**
  (`varscore/data/region_annotations.parquet`), built at image-build time from
  the Ensembl GFF3, plus the tracked gene table (`gene_df.tsv`). This means
  region classification / filtering works with no host data:
  - `python -m varscore.preprocessing.region_filter`
  - `python -m varscore.preprocessing.pipeline`

### Not baked in (mounted at runtime)

These are too large, license-restricted, or job-specific to bake into the image.
Mount them and point the relevant entrypoint at the mount path:

| Dataset | Needed by | How to provide |
|---|---|---|
| Model weights (`.h5`) + peak distribution (`.npy`) | `scoring`, `model_predictions`, `ingest_model` | mount + pass via CLI args |
| Genome FASTA | `scoring`, `validate_variants` | mount + `-g/--genome_loc` |
| AlphaMissense parquet | `alphamissense_scoring` | mount at `varscore/data/alphamissense/` or pass `-d` ([docs](alphamissense.md)) |
| cCRE `ccres.dnatree`, MAF `variants.pkl.gz` | full `annotations` flow | mount at `varscore/data/` |
| Conservation (CADD ~300GB, phyloP bigwigs) | `conservation_utils` | mount at `varscore/data/` |

## Build

```bash
docker build -t kundajelab/varscore:dev -f Dockerfile .
```

Pin the Ensembl gene model with a build arg (defaults to release 116):

```bash
docker build --build-arg ENSEMBL_RELEASE=116 -t kundajelab/varscore:dev .
```

The build downloads the Ensembl GFF3 and constructs the region parquet inside
the image, so it needs network access at build time.

## Run

The entrypoint is `python -m`, so pass a varscore module plus its arguments:

```bash
# Region-classify / route a variant file (uses the baked region data — no mounts)
docker run --rm -v "$PWD:/work" kundajelab/varscore:dev \
    varscore.preprocessing.region_filter -v /work/variants.tsv -o /work/regions/

# Model scoring (mount the model, genome, and peak distribution)
docker run --rm --gpus all \
    -v /data/models:/models -v /data/genome:/genome -v "$PWD:/work" \
    kundajelab/varscore:dev \
    varscore.scoring.chrombpnet.score \
        -m /models/chrombpnet_nobias.h5 \
        -p /models/fold_0_peak_distribution.npy \
        -g /genome/hg38.fa \
        -v /work/variants.tsv \
        -o /work/results.tsv
```

Sanity check after building (do this before pushing):

```bash
docker run --rm kundajelab/varscore:dev varscore.preprocessing.region_filter --help
```

## Push

```bash
docker push kundajelab/varscore:dev
```
