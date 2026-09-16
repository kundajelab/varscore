# Cherimoya image project

This Python 3.12 / PyTorch image supplies the `cherimoya-score` command targeted by Lava's
`CherimoyaPlugin`. The CLI reads varscore's canonical headerless variant TSV and exposes two operations:

- `cherimoya-score score`: score one weights fold and write a headered scalar-score TSV.
- `cherimoya-score summarize`: validate and average a complete set of fold outputs.

```bash
uv lock --project images/cherimoya --python 3.12
docker build -f images/cherimoya/Dockerfile -t kundajelab/cherimoya:dev .
docker push kundajelab/cherimoya:dev
```

The image does not install or import `lava-core`; its public boundary is the CLI, file schemas, and image
reference emitted by the Lava plugin.
