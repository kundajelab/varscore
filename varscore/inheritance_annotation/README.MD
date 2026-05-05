### Summary
Inheritance annotation is a Python module used to classify variant inheritance from parents.  

- Purpose:

Given variant files for a child, mother, and father in tab-separated TSV format, our goal is to determine the inheritance pattern of each variant observed in the child and annotate results for downstream analysis.

For each variant in the child it determines inheritance:

    M → inherited from mother
    F → inherited from father
    Both → inherited from both parents
    De Novo → the variant is not inherited from parents.



## Algorithm

For each variant in the child, the algorithm first normalizes all variants (child, mother, and father) by trimming common prefix/suffix bases from the `ref`/`alt` alleles and adjusting the position accordingly, producing a standardized `(chr, pos, ref, alt)` tuple. It then builds O(1) lookup sets from the normalized maternal and paternal variants, and classifies the child's variant by checking membership: if found in both parent sets it returns `Both`, if only in the mother `M`, if only in the father `F`, and if absent in both `De_Novo`.

```
Child variant
      │
      ▼
  normalize(chr, pos, ref, alt)
      │
      ├──── in mother_set AND father_set? ──► Both
      │
      ├──── in mother_set only? ────────────► M
      │
      ├──── in father_set only? ────────────► F
      │
      └──── in neither? ────────────────────► De_Novo
```

## Running the tests

The test suite covers three areas: `normalize_variant` is tested for correct trimming of common prefixes and suffixes across SNVs, insertions, deletions, and multi-base indels, verifying that position is adjusted when a prefix is removed. `normalize_from_input` is tested to confirm it correctly delegates to `normalize_variant` when given a `VariantAnnotationInput` object. The inheritance classification logic is tested across all four outcomes (`M`, `F`, `Both`, `De_Novo`), including cases where normalization is required for a match, mismatches on chromosome or position, parametrized checks across multiple chromosomes, and independent classification of multiple child variants against the same parent sets.

From the project root:

```bash
python3 -m pytest tests/test_inheritance_annotation.py -v
```