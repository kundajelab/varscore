# syntax=docker/dockerfile:1

# Build from the varscore repository root:
#   docker build -f images/chrombpnet/Dockerfile -t kundajelab/varscore:dev .
FROM kundajelab/chrombpnet:latest

COPY --from=ghcr.io/astral-sh/uv:0.5.16 /uv /bin/uv

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libpq-dev curl wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/varscore
COPY pyproject.toml ./
COPY varscore ./varscore
COPY images/chrombpnet/pyproject.toml images/chrombpnet/uv.lock ./images/chrombpnet/

WORKDIR /src/varscore/images/chrombpnet
RUN uv sync --frozen --no-dev --no-editable
ENV PATH="/src/varscore/images/chrombpnet/.venv/bin:${PATH}"

# Bake the region-annotation interval table into the image. Pin the source release with
# --build-arg ENSEMBL_RELEASE=<release>.
ARG ENSEMBL_RELEASE=116
ENV ENSEMBL_RELEASE=${ENSEMBL_RELEASE}
WORKDIR /src/varscore
RUN bash varscore/scripts/download_ensembl_gff.sh \
    && python varscore/scripts/construct_region_annotations.py \
        -i "varscore/data/raw/Homo_sapiens.GRCh38.${ENSEMBL_RELEASE}.gff3.gz" \
        -o varscore/data/region_annotations.parquet \
    && rm -rf varscore/data/raw

WORKDIR /mnt/volume
ENTRYPOINT ["python", "-m"]
