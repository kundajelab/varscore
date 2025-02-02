# VariantScoringFunctions
Core functions for variant scoring

## Installation
Make sure you have `uv` installed. See [here](https://docs.astral.sh/uv/) for installation instructions.

### Sync Dependencies

```bash
uv sync
```

## Building Docker

Build:

```bash
docker build -t kundajelab/varscore:dev -f Dockerfile.lib .
```

Sanity check to make sure it works (PLEASE DO BEFORE PUSHING):

```bash
docker run --rm kundajelab/varscore:dev varscore.ingest_model
``` 

Push:

```bash
docker push kundajelab/varscore:dev
```