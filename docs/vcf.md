# VCF / gVCF Input

varscore accepts **VCF and gVCF** as an alternative to the canonical headerless
variant TSV (`chr, pos, ref, alt[, variant_id]` — `core.io.VARIANT_SCHEMA`). VCF
support is an **entry-point adapter**: the file is parsed into the canonical
variant table, and the rest of the pipeline (validate → region filter → scorers →
prioritization) is unchanged.

Format loading lives in the IO layer: `core.io.read_variants(path, fmt="auto")`
is the format-neutral entry point, dispatching to the per-format adapters
`core.io.load_variants` (TSV) and `core.io.load_variants_vcf` (VCF/gVCF). The
parser is pure Python and adds **no new dependency** — stdlib `gzip` reads both
plain `.vcf` and bgzipped `.vcf.gz` (bgzip is a valid gzip stream).

## Usage

Standalone conversion to the canonical TSV:

```bash
python -m varscore.preprocessing.vcf -v input.vcf.gz -o variants.tsv
```

Or pass a VCF straight to validation / the full preprocessing pipeline via the
`--format` flag (defaults to `auto`, which detects by extension):

```bash
python -m varscore.preprocessing.validate -v input.vcf.gz -g genome.fa -f auto ...
python -m varscore.preprocessing.pipeline -i input.vcf.gz -g genome.fa -f auto \
    -o valid.tsv --invalid-out invalid.tsv --region-out-dir regions/
```

`-f/--format` accepts `auto`, `tsv`, or `vcf`.

## What the parser does

- **Multi-allelic split.** A site with `ALT = A,AT` becomes one canonical row per
  ALT allele.
- **Symbolic / non-variant alleles dropped.** `<NON_REF>`, `<*>`, `<DEL>` (and any
  other `<...>` form), the spanning-deletion `*`, breakend notation (`[`/`]`), and
  monomorphic `ALT == REF` rows are filtered out — these dominate **gVCF reference
  blocks** and would otherwise explode into millions of non-variant rows. The
  dropped count is logged.
- **1-based POS** is kept as-is (matches `pos`).
- **`ID`** is carried into `variant_id`; `ID == "."` becomes missing.
- **Chromosome naming passes through untouched.** Bare `1` → `chr1` normalization
  and the ACGT/genome checks happen downstream in `core/io.validate_variant`, so a
  non-ACGT or otherwise invalid allele lands in the *invalid* output with a proper
  error reason — exactly as it would from a TSV input.

## Limitations

- **No binary BCF.** Convert first: `bcftools view in.bcf -Oz -o out.vcf.gz`. A
  `.bcf` path raises a clear error.
- **INFO / FORMAT / genotypes are ignored.** A multi-sample VCF is treated as a
  list of sites × ALT alleles, not per-sample calls.
- **Whole-file in memory.** Like the TSV loader, the parser returns one DataFrame;
  `validate` then batches at 250k. Fine for typical inputs.

## Custom variant IDs (future work)

The VCF parser already captures the `ID` column into `variant_id`, and the
standalone converter writes the full 5-column canonical TSV, so IDs survive the
conversion. **End-to-end** ID preservation through the scorers is not yet wired:
`preprocessing/validate.py` writes only `chr, pos, ref, alt` for valid variants
(it drops `variant_id`). That single projection is the only blocker — the input
path here is built to be extensible to it without further redesign.
