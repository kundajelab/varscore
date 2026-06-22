mkdir -p varscore/data/raw
# AlphaMissense hg38 precomputed pathogenicity scores (DeepMind, CC BY-NC-SA 4.0)
# Public GCS bucket: gs://dm_alphamissense/AlphaMissense_hg38.tsv.gz (~640MB)
wget -O varscore/data/raw/AlphaMissense_hg38.tsv.gz https://storage.googleapis.com/dm_alphamissense/AlphaMissense_hg38.tsv.gz
