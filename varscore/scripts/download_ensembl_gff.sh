#!/usr/bin/env bash
# Download the Ensembl gene-model GFF3 (GRCh38) used to build the region
# annotation interval table. See construct_region_annotations.py for the
# parquet build step, and docs/region_classification.md for context.
set -euo pipefail

RELEASE="${ENSEMBL_RELEASE:-116}"
OUT_DIR="varscore/data/raw"
OUT_FILE="${OUT_DIR}/Homo_sapiens.GRCh38.${RELEASE}.gff3.gz"
URL="https://ftp.ensembl.org/pub/release-${RELEASE}/gff3/homo_sapiens/Homo_sapiens.GRCh38.${RELEASE}.gff3.gz"

mkdir -p "${OUT_DIR}"
wget -O "${OUT_FILE}" "${URL}"
echo "Downloaded ${OUT_FILE}"
