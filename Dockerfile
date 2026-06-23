# USAGE:
# docker build -t kundajelab/varscore:dev -f Dockerfile .
# docker run --rm kundajelab/varscore:dev varscore.ingest_model ...
#
# Model-scoring image. The TensorFlow/ChromBPNet stack comes from the
# kundajelab/chrombpnet base image, so `pip install .` only needs to add the
# lightweight varscore core deps on top (no `[model]` extra required here).
# ------------------------------------------------

# Use the official Python image as the base
FROM kundajelab/chrombpnet:latest

RUN apt update \
    && apt install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /code

# Copy the requirements files  
COPY pyproject.toml .

# Create a minimal package structure just for dependency installation
RUN mkdir varscore && touch varscore/__init__.py

# Install dependencies
RUN pip install .

# Copy the application files
COPY . .

ENTRYPOINT [ "python", "-m" ] 