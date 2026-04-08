### Goal

- Perform trio analysis on a variant set (child, mother, father)

For each variant in the child:

- Determine inheritance:

    M → inherited from mother

    F → inherited from father

    Both → inherited from both parents

    De Novo → the variant is not inherited from parents.

Then annotate those variants for downstream analysis

---

## Algorithm Design

### Context

The existing codebase operates on **4-column TSV files** (`chr, pos, ref, alt`). This simplifies trio analysis compared to full VCF-based tools like VBT (Toptas et al. 2018), since diploid genotype phasing is already resolved — each row is an individual allele.

### Approach: Normalized allele set matching

The naive approach (match child variants to parent variants by position alone) can fail when the same indel is represented differently by different callers — e.g., the same deletion written at position 103 vs position 105 due to left-alignment differences. The VBT paper demonstrates this causes tens of thousands of false Mendelian violations in real data.

**Solution**: Left-normalize variants before comparison (trim common prefix/suffix, adjust position) to unify representations, then classify using set intersection.

#### Steps

1. Load child, mother, father TSVs (4 cols: chr, pos, ref, alt)
2. Normalize all variants: trim common prefix/suffix, adjust pos
3. Build mother and father lookup sets: `{(chr, pos, ref, alt)}`
4. For each child variant:
   - in mother ∩ father → **Both**
   - in mother only → **M**
   - in father only → **F**
   - in neither → **De_Novo**
5. Return annotated DataFrame with `inheritance` column; optionally write TSV

#### Normalization

```python
def normalize_variant(chrom, pos, ref, alt):
    # Trim common suffix
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    # Trim common prefix (adjust pos)
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        pos += 1
    return chrom, pos, ref, alt
```

Full left-shifting (reference FASTA lookup) is optional for a first pass and can be added if needed.

---

## Implementation Plan

### Files to create

**`tri_analytics/trio_analysis.py`**

```
- normalize_variant(chrom, pos, ref, alt) -> tuple
- load_variants(path) -> pd.DataFrame
- classify_inheritance(child_df, mother_df, father_df) -> pd.DataFrame
- run_trio_analysis(child_path, mother_path, father_path, output_path=None) -> pd.DataFrame
```

**`tri_analytics/__init__.py`** — export `run_trio_analysis`

### Reuse from existing code

- `varscore/validate_variants.py` — optionally pre-validate trio TSVs
- `varscore/annotations.py` — downstream annotation after inheritance classification
- `varscore/utils/variant_utils.py` — check for existing normalization helpers

---

## Limitations vs Full VBT

| Feature | This approach | VBT |
|---|---|---|
| SNVs | Accurate | Accurate |
| Simple indels | Accurate (with normalization) | Accurate |
| Complex overlapping indels | May miss edge cases | Handles |
| Input format | TSV (existing architecture) | VCF required |
| Python-native | Yes | No (C++ binary) |
| Diploid phasing / same-allele elimination | Not needed (TSV = individual alleles) | Required for VCF |

VBT's most complex logic (4-stage haplotype comparison, same-allele-match elimination) applies to **diploid VCF genotypes**. Since our TSVs already represent individual alleles, that complexity is unnecessary here.

---

## Implementation: `tri_anotation` walkthrough

The entry point is `tri_anotation()` in `run_tri_analytics.py`. Below is a step-by-step trace of what happens when it runs.

### Step 1 — Load inputs

Three 4-column TSVs are read into pandas DataFrames with the schema `(chr, pos, ref, alt)`:

```
child_df   ← variants_loc
mother_df  ← maturnal_var_loc
father_df  ← paternal_var_loc
```

No header is expected; columns are assigned by position.

### Step 2 — Validate child variants (optional)

If `genome_loc` is supplied, child variants are checked against the reference FASTA via `validate_variants()` (from `varscore/utils/io_utils.py`). This confirms that the `ref` allele in each row actually matches the genome at that position. Invalid variants (ref mismatch, unknown chromosome, etc.) are logged and dropped before classification proceeds.

This step is skipped if `genome_loc` is `None`.

### Step 3 — Build normalized parent lookup sets

For each parent DataFrame, every variant is passed through `normalize_variant(chrom, pos, ref, alt)`:

```
Trim common suffix:
  while ref and alt share a trailing base → remove it from both

Trim common prefix (shift position):
  while ref and alt share a leading base → remove it from both, pos += 1
```

This converts any representation of the same indel to a single canonical `(chr, pos, ref, alt)` tuple. The results are collected into two Python sets:

```
mother_set = { (chr, pos, ref, alt), ... }   # O(1) lookup
father_set = { (chr, pos, ref, alt), ... }
```

Using sets means membership tests during classification are O(1) regardless of how many parent variants there are.

### Step 4 — Classify each child variant

Each child variant is normalized by the same function, producing a canonical tuple. That tuple is then tested against both parent sets:

```
in mother ∩ father  →  Both
in mother only      →  M
in father only      →  F
in neither          →  De_Novo
```

The result is appended to an `inheritance_labels` list, which is assigned as a new column on `child_df` at the end.

### Step 5 — Write output

The annotated DataFrame (original 4 columns + `inheritance`) is written as a tab-separated file to `valid_out_path`. A summary of label counts is logged.

### Data flow diagram

```
variants_loc ──┐
               ├─► _load_variants() ──► [validate] ──► child_df
maturnal_var ──┤                                            │
               └─► _load_variants() ──► _build_normalized_set() ──► mother_set ──┐
paternal_var ──────► _load_variants() ──► _build_normalized_set() ──► father_set ─┤
                                                                                   │
                                          child_df rows ──► normalize ──► lookup ──┘
                                                                              │
                                                               inheritance label
                                                                              │
                                                          child_df + inheritance
                                                                              │
                                                              write TSV ──► valid_out_path
```

---

## Verification

1. **Unit tests with synthetic data**: known inheritance per variant → assert correct label
2. **Representation test**: same indel in two representations → normalization should unify them → same classification
3. **De novo test**: child variant absent from both parents → `De_Novo`
4. **Integration test**: run `run_trio_analysis()` end-to-end, check output TSV has `inheritance` column
