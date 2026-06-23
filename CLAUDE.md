# CLAUDE.md

Guidance for AI agents working in this repo. Covers the non-obvious things that
aren't in `README.md` (which has install/data-setup/Docker). Read this first.

## Environment & running things (read before you run anything)

- `pyproject.toml` pins `tensorflow==2.8.1` and `requires-python >=3.8,<3.10`.
  **That TF wheel does not exist for macOS arm64**, so `uv sync` / `uv run` fail
  on a Mac with a "no source distribution or wheel" error.
- Most work does **not** need TensorFlow. Modules like region classification,
  the parquet lookups, and the scoring entrypoints run fine with the project
  venv's interpreter directly, which skips the sync:
  ```bash
  .venv/bin/python -m varscore.<module> ...
  ```
- Run tests the same way — the conda/system base env is missing `ncls` (imported
  by `varscore/utils/region_utils.py`), so use the venv:
  ```bash
  .venv/bin/python -m pytest tests/ -q
  ```
- Everything is invoked as a module: `python -m varscore.<x>` /
  `python -m varscore.scripts.<x>`.

## Data artifacts are gitignored — build them before use

None of the bulk reference data is in git (see `.gitignore`: `*.parquet`,
`*.gz`, `ccres.dnatree`, etc.). Each dataset has a
`varscore/scripts/download_*.sh` + `varscore/scripts/construct_*` pair:

- `region_annotations.parquet` ← `download_ensembl_gff.sh` + `construct_region_annotations.py`
- `data/alphamissense/` ← `download_alphamissense.sh` + `construct_alphamissense_parquet.py`
- `data/spliceai/` ← `download_spliceai.sh` + `construct_spliceai_parquet.py`
- `ccres.dnatree`, variants df ← see `README.md`

Lookups raise a clear "build it first" `FileNotFoundError` if the artifact is
missing. If you change a `construct_*` script, the corresponding artifact must be
**rebuilt** — it won't update itself.

## The scorer-module pattern (mirror it for new scorers)

External scoring resources all follow the same shape; copy it for the next one:

1. `scripts/download_<x>.sh` — fetch the raw file.
2. `scripts/construct_<x>_parquet.py` — rewrite into a **chr-partitioned,
   position-sorted parquet** via DuckDB (Hive-partitioned on `CHROM`).
3. `utils/<x>_utils.py` — a `lookup_<x>(df, ...)` that left-joins variants
   against the parquet with DuckDB, pruning to the needed chromosomes.
4. `<x>_scoring.py` — a thin TSV-in / TSV-out CLI entrypoint.
5. `docs/<x>.md` — user-facing doc.

AlphaMissense and SpliceAI are the reference implementations.

## Conventions

- **Variant files:** headerless TSV, columns `chr, pos, ref, alt[, variant_id]`
  (`io_utils.VARIANT_SCHEMA`). Scoring outputs append columns to these.
- **Chromosome naming:** datasets use UCSC-style `chr1` / `chrM`; lookups
  normalize bare `1` / `MT` to the `chr` form. Caveat: a bare-chr value that
  slips past normalization classifies silently as `intergenic` — no error.
- **Region labels are multi-label + severity-ranked.** A variant carries a *set*
  of labels (`region_labels`); `region_type` is only the severity-collapsed
  headline kept for back-compat. **Consumers should test membership**
  (`"x" in region_labels`, or the `is_coding` / `in_promoter` properties on
  `RegionAnnotation`), never `region_type == "x"` — the collapse hides
  lower-severity labels (this caused a real prioritization bug). See
  `docs/region_classification.md`.
- **`ai_docs/` vs `docs/`:** `ai_docs/` is a **gitignored** agent plan/impl
  scratch workspace — never commit it or reference it from code. Put the
  polished, synthesized version in **`docs/`**, and point code/comments there.

## Module map

- `annotations.py` — `AnnotatedVariant` model + the full annotation chain (pulls
  in TF via the model path).
- `variant_region_filter.py` — routes variants into overlapping per-region
  category TSVs (coding, splice, promoter, …) for downstream scoring.
- `variant_preprocessing.py` — validate → region-filter pipeline.
- `scoring.py` / `model_predictions.py` — model scoring (TF).
- `prioritization.py` — turns a variant×model score DB into prioritized calls.
- `alphamissense_scoring.py`, `spliceai_scoring.py` — per-scorer lookup entrypoints.
- `utils/region_utils.py` — NCLS-backed region classifier + nearest-gene/cCRE.
- `scripts/` — download + construct scripts for the gitignored data artifacts.
- `tests/` — pytest; region/filter tests monkeypatch the annotator so they don't
  need the built parquet.

## Git

- PRs target `main`. End commit messages with the `Co-Authored-By` trailer.
- Don't commit gitignored data artifacts or `.DS_Store`.
