# Trio Annotation Tests

Tests for `varscore/trio_annotation/run.py`, which classifies whether a child variant was inherited from the mother, the father, both parents, or is a de novo mutation.

## Running the tests

From the project root:

```bash
python3 -m pytest varscore/trio_annotation/tests/test_trio_annotation.py -v
```

---

## Module under test

Three public symbols are tested:

| Symbol | Purpose |
|---|---|
| `normalize_variant(chrom, pos, ref, alt)` | Strips shared prefix/suffix bases from ref/alt and adjusts position. Returns a canonical `(chrom, pos, ref, alt)` tuple. |
| `normalize_from_input(variant)` | Thin wrapper — accepts a `VariantAnnotationInput` Pydantic model and calls `normalize_variant`. |
| `trio_annotation(child, mother, father)` | Compares the child variant against both parents after normalization. Returns `"Both"`, `"M"`, `"F"`, or `"De_Novo"`. |

---

## Test classes

### `TestNormalizeVariant` — 7 tests

Normalization is the step that puts all variant representations into a canonical form before comparison. Without it, the same indel written two different ways would fail to match.

The function works in two passes:

1. **Suffix trimming** — while both ref and alt are longer than 1 base and share the same trailing base, strip it.
2. **Prefix trimming** — while both are longer than 1 base and share the same leading base, strip it and increment `pos` by 1.

| Test | Input `(ref, alt)` | Expected output `(ref, alt, pos_delta)` | What is being verified |
|---|---|---|---|
| `test_snv_unchanged` | `A, T` | `A, T, +0` | Single-base SNV: nothing to trim, returned as-is |
| `test_common_suffix_trimmed` | `AGT, AT` | `AG, A, +0` | Shared trailing `T` is removed |
| `test_common_prefix_trimmed_adjusts_pos` | `AAT, AGT` | `A, G, +1` | Shared trailing `T` removed first, then shared leading `A` removed and pos incremented |
| `test_prefix_and_suffix_trimmed` | `AATG, AGTG` | `A, G, +1` | Both ends trimmed iteratively until a minimal representation is reached |
| `test_single_base_not_trimmed` | `A, A` | `A, A, +0` | Guard condition: loop stops when either allele is 1 base long |
| `test_insertion` | `A, ACGT` | `A, ACGT, +0` | Insertion with no shared flanking context: unchanged |
| `test_deletion` | `ACGT, A` | `ACGT, A, +0` | Deletion with no shared flanking context: unchanged |

---

### `TestNormalizeFromInput` — 1 test

`normalize_from_input` accepts a `VariantAnnotationInput` Pydantic model (with `.chr`, `.pos`, `.ref`, `.alt` fields) and delegates to `normalize_variant`. The single test confirms the output is identical to calling `normalize_variant` directly with the same field values.

---

### `TestTrioAnnotation` — 17 tests

This class tests the inheritance classifier end-to-end. Each call compares a child variant against one mother variant and one father variant. After normalizing all three, the function checks equality and returns one of four labels.

#### Core outcome tests (4 tests)

| Test | Child alt | Mother alt | Father alt | Expected |
|---|---|---|---|---|
| `test_de_novo` | `T` | `C` | `G` | `De_Novo` — alt not present in either parent |
| `test_maternal_only` | `T` | `T` | `G` | `M` — matches mother only |
| `test_paternal_only` | `T` | `G` | `T` | `F` — matches father only |
| `test_both_parents` | `T` | `T` | `T` | `Both` — matches both |

#### Normalization edge case (1 test)

`test_normalization_required_still_matches` — the child carries `ref=AGT, alt=AT`, which normalizes to `ref=AG, alt=A`. The mother carries the already-trimmed form `ref=AG, alt=A`. Without normalization these would not compare equal. The test verifies that normalization is applied to both sides before comparison, so the result is correctly `"M"`.

#### Boundary / mismatch tests (2 tests)

| Test | What differs | Expected |
|---|---|---|
| `test_different_chrom_is_de_novo` | Child on `chr1`, parents on `chr2`/`chr3` | `De_Novo` |
| `test_different_pos_is_de_novo` | Child at pos 100, parents at 101/102 | `De_Novo` |

These verify that chromosome and position are both part of the equality check, not just the alleles.

#### Parametrized chromosome regression tests (10 tests — 2 × 5 chromosomes)

Two parametrized tests run `De_Novo` and `M` scenarios across `chr1`, `chr10`, `chr20`, `chrX`, and `chrY`.

The multi-digit chromosomes (`chr10`, `chr20`) and sex chromosomes (`chrX`, `chrY`) are explicitly included because a prior bug in the pipeline used a regex that accidentally excluded `chr10` and `chr20` during variant validation (fixed in commit `2a15b3b`). These parametrized cases serve as regression coverage — they would fail immediately if that class of bug were reintroduced.
