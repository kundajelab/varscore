# USAGE:
# docker build -t varscore-dev -f Dockerfile.lib .
# docker run --rm $IMAGE_ID src.varscore.ingest_model
#
# We use `uv` as a build tool only in dev. Here, to keep
# the image size small, we install the dependencies 
# directly from pip.
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