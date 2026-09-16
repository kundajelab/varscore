# syntax=docker/dockerfile:1

# Build from the varscore repository root:
#   docker build -f images/cherimoya/Dockerfile -t kundajelab/cherimoya:dev .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.5.16 /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git samtools build-essential zlib1g-dev libbz2-dev liblzma-dev libcurl4-openssl-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/varscore
COPY pyproject.toml ./
COPY varscore ./varscore
COPY images/cherimoya/pyproject.toml images/cherimoya/uv.lock ./images/cherimoya/

WORKDIR /src/varscore/images/cherimoya
RUN uv sync --frozen --no-dev --no-editable
ENV PATH="/src/varscore/images/cherimoya/.venv/bin:${PATH}"
ENV HF_HOME=/root/.cache/huggingface

WORKDIR /mnt/volume
