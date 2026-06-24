# Custom Variant IDs

The canonical variant table's optional 5th column — `variant_id`
(`core.io.VARIANT_SCHEMA`) — lets you attach a stable identifier to each variant
and have it travel through **every** stage (preprocessing, region routing,
scoring, prioritization, annotation), so outputs can be joined back to your input.

## Where the id comes from

- **TSV input:** the 5th column.
- **VCF/gVCF input:** the `ID` field (`.` → blank). See [vcf.md](vcf.md).
- **No id supplied:** the 5th column is left **blank** — no synthetic id is
  generated. (Provide your own ids if you need a guaranteed unique handle, e.g.
  for prioritization grouping; otherwise variants stay distinct by
  `chr/pos/ref/alt`.)

## How it flows

- `core.io.load_variants` / `read_variants` always yield the 5-col schema
  (`variant_id` is `NaN` when absent).
- `preprocessing/validate.py` writes the **full** canonical schema for valid
  variants — the id is no longer dropped on the valid path. Invalid variants keep
  the id too, alongside the error columns.
- `preprocessing/region_filter.py` routes whole rows, so the id rides into each
  per-region-category file.
- **Scorers carry it through by name, not position** — so a 5th column never
  needs stripping/reattaching and never breaks anything:
  - ChromBPNet ([score.py](../varscore/scoring/chrombpnet/score.py)) reads
    `chr/pos/ref/alt` by name, appends score columns, and writes the whole frame.
    The model only ever sees one-hot arrays, never the DataFrame.
  - Fold-averaging ([predictions.py](../varscore/scoring/chrombpnet/predictions.py))
    copies `VARIANT_SCHEMA` columns (incl. `variant_id`) from fold 0.
  - AlphaMissense ([lookup.py](../varscore/scoring/alphamissense/lookup.py))
    preserves all input columns and replicates the id across the per-transcript
    row expansion.
- `prioritization.py` uses `variant_id` as part of its pivot index key.
- `annotation/annotate.py` carries `variant_id` onto each annotated JSONL record
  (an optional field; `null` when absent).

## Tests

End-to-end preservation is covered by `tests/test_validate.py` (validate keeps the
id, blank when absent), `tests/test_variant_region_filter.py` (routing preserves
ids), `tests/test_scoring.py` + `tests/test_alphamissense_utils.py` (scorers carry
ids, incl. multi-transcript replication), and `tests/test_annotate.py`.
