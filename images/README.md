# Model runtime images

Each directory is an independent Python project with its own interpreter constraint, dependency graph, and
`uv.lock`. They intentionally are not members of one shared uv workspace: ChromBPNet is pinned to the
Python 3.9 / TensorFlow 2.8 ecosystem, while Cherimoya uses Python 3.12 / PyTorch.

Both projects depend on the lightweight varscore library through a local path during development and image
builds. Shared variant I/O and scientific helpers belong in `varscore/`; framework orchestration belongs in
`lava-core`; model-specific ML dependencies belong here.

Build from the repository root so each Dockerfile can copy the shared package:

```bash
docker build -f images/chrombpnet/Dockerfile -t kundajelab/varscore:dev .
docker build -f images/cherimoya/Dockerfile -t kundajelab/cherimoya:dev .
```

Use immutable tags or digests for production registrations. A mutable `:dev` tag is only for local testing.
