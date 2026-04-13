### Summary
Trio annotation is a Python module used to classify variant inheritance from parents.  

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

The test suite in `tests/test_trio_annotation.py` validates the core logic of the trio annotation module across three areas. `TestNormalizeVariant` exercises the `normalize_variant` function with a range of cases — SNVs, insertions, deletions, and variants requiring both prefix and suffix trimming — verifying that the standardized `(chr, pos, ref, alt)` tuple is produced correctly in each case. `TestNormalizeFromInput` confirms that `normalize_from_input` correctly delegates to `normalize_variant` using a `VariantAnnotationInput` object. `TestTrioAnnotation` covers the full inheritance classification logic, testing all four outcomes (`M`, `F`, `Both`, `De_Novo`) including edge cases such as variants that require normalization before matching, mismatches on chromosome or position, multiple child variants classified independently against the same parent sets, and parametrized checks across standard chromosomes (`chr1`, `chr10`, `chr20`, `chrX`, `chrY`).

From the project root:

```bash
python3 -m pytest tests/test_trio_annotation.py -v
```