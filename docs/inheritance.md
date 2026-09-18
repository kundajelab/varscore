# Trio inheritance annotation

Classify each of a child's variants by how it was inherited from the parents:

| Label      | Meaning                                                        |
|------------|---------------------------------------------------------------|
| `M`        | inherited from the mother only                                 |
| `F`        | inherited from the father only                                 |
| `Both`     | present in both parents                                        |
| `De_Novo`  | absent in both parents (a candidate de-novo mutation)         |
| `Uncertain`| genotype mode only — a parent is a no-call / low-confidence    |

This is a **separate pipeline** from the per-variant annotation chain
(`varscore.annotation.annotate`): inheritance needs cross-sample context (child +
mother + father), which the single-variant chain can't provide. Module:
[varscore/annotation/inheritance.py](../varscore/annotation/inheritance.py).

## Two modes (auto-selected)

### Genotype / zygosity-aware — recommended when you have it
Triggered when you pass `--child-sample/--mother-sample/--father-sample` against a
**single jointly-genotyped multi-sample VCF**. Genotypes (`GT`) are read directly,
so this is the **only** input that supports *true* de-novo calling: a hom-ref
parent has a confident reference call at the child's site even with no ALT record.

```bash
python -m varscore.annotation.inheritance \
  -v trio.vcf.gz -o inheritance.tsv \
  --child-sample CHILD --mother-sample MOM --father-sample DAD
```

Logic, per site where the child carries the ALT (`het` or `hom_alt`):
- `M`/`F`/`Both` by which parent(s) carry ≥1 copy of that ALT.
- `De_Novo` only when **both** parents are `hom_ref` *and* confident
  (`GQ ≥ --min-gq`, default 20; `DP ≥ --min-dp`, default 10).
- `Uncertain` when a parent is a no-call (`./.`) or below those thresholds — we
  can't distinguish a true reference from missing data, so we don't claim de novo.
- A `hom_alt` child must inherit an ALT from *both* parents; if not, the row is
  flagged `mendelian_violation` in `inheritance_flag`.

Output adds `inheritance`, `inheritance_mode=genotype`, `child_zygosity`, and
`inheritance_flag`. Multi-allelic sites are split, and zygosity is computed with
respect to each specific ALT allele.

### Presence / absence — fallback for variant lists
Used when you pass three separate child/mother/father files (`-v/-m/-p`), each a
headerless TSV (`chr, pos, ref, alt[, variant_id]`) or a VCF. A child variant is
classified by canonical set membership in the parent call sets.

```bash
python -m varscore.annotation.inheritance \
  -v child.tsv -m mother.tsv -p father.tsv -g genome.fa -o inheritance.tsv
```

> **Caveat:** here `De_Novo` means "not observed in either parent file." A parent
> with a no-call / low-coverage / filtered site at that position looks identical to
> a true reference, so this mode **over-calls de novo**. Use genotype mode for
> confident de-novo calling. Zygosity and Mendelian checks are unavailable here.

Output adds `inheritance`, `inheritance_mode=presence`, and `inheritance_flag`.
The child's original columns and `variant_id` are preserved end-to-end.

## Canonical normalization

So that equivalent representations compare equal across samples, every variant is
normalized before matching:
- **Chromosome naming** — bare `1` / `MT` are normalized to the `chr…` form.
- **Indel left-alignment** — with `-g/--genome`, indels are left-aligned against
  the reference (vt/bcftools algorithm) so a repeat-shifted indel matches no matter
  how it was written. Without a genome, indels are only trimmed (shared
  prefix/suffix), which can miss repeat-shifted indels → occasional false
  `De_Novo`. A genome is therefore **recommended** for presence/absence mode.
- **MNP atomization** (default; disable with `--no-atomize`) — a multi-nucleotide
  polymorphism such as `TA>CG` is split into per-position SNVs (`T>C`, `A>G`),
  the phase-free canonical form (cf. `bcftools norm --atomize`).

### Why atomize (and why not "merge")?
Different callers represent the same biology differently: one emits a single MNP
record `chr1:100 TA>CG`, another emits two SNV records. Naive whole-record matching
would miss that. The fix is to reduce **down** to atomic SNVs on both sides.

We do **not** try to *merge* adjacent SNVs up into MNPs: that is only correct if
the two SNVs sit on the same haplotype (cis), which is unknowable without phase
information. Merging variants that are actually on opposite chromosomes (trans)
would fabricate an MNP that doesn't exist.

With atomization on, a child variant is "inherited" from a parent iff **all** its
atoms are present in that parent. A child SNV that matches one position of a
parent's MNP therefore counts as inherited (the standard atomized comparison). A
child MNP that matches only *some* of a parent's atoms is **not** inherited and is
flagged `partial_mnp_overlap` (a likely calling/representation artifact — adjacent
bases don't recombine, so genuine partial inheritance is essentially impossible).

With `--no-atomize`, whole normalized `(chr, pos, ref, alt)` tuples are matched
instead — simplest, but an MNP-vs-split disagreement yields a false `De_Novo`.

## Caveats / not yet handled
- **Sex chromosomes & mitochondria** are treated like autosomes; the special
  inheritance of chrX/chrY/chrM (e.g. a son's chrX is maternal-only) is not modeled.
- De-novo confidence uses simple `GQ`/`DP` thresholds, not full genotype posteriors.
- Separate single-sample VCFs are treated as presence/absence (a hom-ref parent has
  no record to read), so they don't get true de-novo or zygosity output.
