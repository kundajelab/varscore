# USAGE:
# docker build -t kundajelab/varscore:dev -f Dockerfile .
# docker run --rm kundajelab/varscore:dev varscore.scoring.chrombpnet.ingest ...
#
# Model-scoring image. The TensorFlow/ChromBPNet stack comes from the
# kundajelab/chrombpnet base image; we install the `[model]` extra so varscore's
# deps resolve against the TF-compatible pins (notably numpy<1.24) instead of
# letting the unbounded core numpy upgrade out from under TensorFlow 2.8.
# ------------------------------------------------

# Use the official Python image as the base
FROM kundajelab/chrombpnet:latest

RUN apt update \
    && apt install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        wget \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy the requirements files  
COPY pyproject.toml .

# Create a minimal package structure just for dependency installation
RUN mkdir varscore && touch varscore/__init__.py

# Install dependencies (model profile: TF-compatible pins on top of the base TF)
RUN pip install ".[model]"

# Copy the application files
COPY . .

# Bake the region-annotation interval table into the image so region
# classification / filtering (varscore.preprocessing.region_filter,
# varscore.preprocessing.pipeline) works out of the box with no host data.
# Pin the gene model via ENSEMBL_RELEASE for reproducible builds. The other
# datasets (AlphaMissense, conservation, MAF, model weights, genome) are
# mounted at runtime — see docs/docker.md.
# construct_region_annotations.py only needs numpy/pandas/pyarrow, so it runs as
# a plain script (no varscore import). The raw GFF3 is removed afterwards to
# keep the layer small.
ARG ENSEMBL_RELEASE=116
ENV ENSEMBL_RELEASE=${ENSEMBL_RELEASE}
RUN bash varscore/scripts/download_ensembl_gff.sh \
    && python varscore/scripts/construct_region_annotations.py \
        -i "varscore/data/raw/Homo_sapiens.GRCh38.${ENSEMBL_RELEASE}.gff3.gz" \
        -o varscore/data/region_annotations.parquet \
    && rm -rf varscore/data/raw

ENTRYPOINT [ "python", "-m" ]