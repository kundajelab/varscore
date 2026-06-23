#!/usr/bin/env bash
# Download the Illumina precomputed SpliceAI score VCFs (hg38, raw) used by
# construct_spliceai_parquet.py. SpliceAI scores are CC BY-NC 4.0 (academic /
# non-commercial use only).
#
# Unlike AlphaMissense, the official files are hosted on Illumina BaseSpace and
# require a (free) login, so this is an interactive flow via the BaseSpace CLI
# (`bs`) rather than a one-shot wget. Install it from:
#   https://developer.basespace.illumina.com/docs/content/documentation/cli/cli-overview
set -euo pipefail

mkdir -p varscore/data/raw

# 1. Authenticate (opens a browser link to approve; one-time per machine):
#      bs auth
#
# 2. The scores live in the public project "Predicting splicing from primary
#    sequence", folder genome_scores_v1.3. Pull the hg38 raw SNV + indel VCFs
#    (and their .tbi indexes). Replace <DATASET_ID> with the id BaseSpace shows
#    for the genome_scores_v1.3 dataset (`bs dataset list`):
#
#      bs dataset download --id <DATASET_ID> -o varscore/data/raw \
#        --extension vcf.gz,vcf.gz.tbi
#
#    Web alternative: https://basespace.illumina.com/s/otSPW8hnhaZR -> FILES ->
#    genome_scores_v1.3, and download:
#      spliceai_scores.raw.snv.hg38.vcf.gz   (+ .tbi)
#      spliceai_scores.raw.indel.hg38.vcf.gz (+ .tbi)
#
# After download the files should land at:
#   varscore/data/raw/spliceai_scores.raw.snv.hg38.vcf.gz
#   varscore/data/raw/spliceai_scores.raw.indel.hg38.vcf.gz

echo "See the comments in this script: SpliceAI scores require a BaseSpace login."
echo "Login-free alternative (SNVs + 1bp insertions only, MANE/Ensembl annotation):"
echo "  Ensembl FTP -> pub/data_files/homo_sapiens/GRCh38/variation_plugins/"
echo "  spliceai_scores.raw.snv.ensembl_mane.grch38.*.vcf.gz"
