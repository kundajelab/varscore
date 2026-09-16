# ChromBPNet image project

This is the legacy Python 3.9 / TensorFlow 2.8 runtime for the existing
`kundajelab/varscore` image. Its lockfile is intentionally independent from the varscore library and the
other model images.

```bash
uv lock --project images/chrombpnet --python 3.9
docker build -f images/chrombpnet/Dockerfile -t kundajelab/varscore:dev .
```

The root `varscore[model]` extra remains temporarily for installation compatibility, but this project is the
canonical image dependency definition.
