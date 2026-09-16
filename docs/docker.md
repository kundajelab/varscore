# Model images

varscore contains independently locked model-runtime projects under `images/`. The shared `varscore` wheel
stays free of TensorFlow and PyTorch; each image owns the interpreter and ML stack required by its model.

| Image project | Python/runtime | Published image | Command contract |
|---|---|---|---|
| `images/chrombpnet` | Python 3.9, TensorFlow 2.8 | `kundajelab/varscore` | `python -m varscore.scoring.chrombpnet...` |
| `images/cherimoya` | Python 3.12, PyTorch | `kundajelab/cherimoya` | `cherimoya-score score|summarize` |

The projects have separate `pyproject.toml` and `uv.lock` files and intentionally do not share one uv
workspace resolution. Both install the shared varscore package from the repository root during a build.

## Build

Use the repository root as the Docker build context:

```bash
docker build -f images/chrombpnet/Dockerfile -t kundajelab/varscore:dev .
docker build -f images/cherimoya/Dockerfile -t kundajelab/cherimoya:dev .
```

Pin the ChromBPNet image's Ensembl gene model with a build argument (default 116):

```bash
docker build -f images/chrombpnet/Dockerfile \
    --build-arg ENSEMBL_RELEASE=116 \
    -t kundajelab/varscore:dev .
```

Regenerate a lock only from its project directory:

```bash
uv lock --project images/chrombpnet --python 3.9
uv lock --project images/cherimoya --python 3.12
```

## ChromBPNet runtime

The ChromBPNet image retains the existing `python -m` entrypoint and bakes the region-annotation interval
table. Large model weights, genome FASTA, AlphaMissense data, and conservation tracks remain runtime mounts.

```bash
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

## Cherimoya runtime

`cherimoya-score` reads varscore's canonical headerless variant TSV
(`chr, pos, ref, alt[, variant_id]`). A score task downloads one fold's weights from Hugging Face and writes
a headered `variant_id, counts_log2fc, profile_l1` TSV. The summarize task validates that every fold contains
the same variants before averaging it.

```bash
docker run --rm --gpus all -v /data:/mnt/volume kundajelab/cherimoya:dev \
    cherimoya-score score \
        --hf-repo programmable-genomics/CATv1 \
        --hf-filename models/example/cherimoya.fold_0.torch \
        --hf-revision main \
        --in-window 2114 \
        -g /mnt/volume/genome.fa \
        -v /mnt/volume/variants.tsv \
        -o /mnt/volume/fold_0.tsv
```

The runtime has no dependency on `lava-core`. Lava refers to it only through an image reference, argv, and
file contracts, so the same image can be driven by local Docker, Modal, Kubernetes, or another orchestrator.

## Publish

Use immutable release tags and record their digests in model registrations. Mutable `:dev` tags are only for
local iteration; some execution fabrics cache an image snapshot by reference string.
