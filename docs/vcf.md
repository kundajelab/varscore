# VCF input

The preprocessing pipeline accepts coordinate-sorted VCF and BGZF-compressed
VCF as alternatives to the canonical headerless variant TSV
(`chr, pos, ref, alt[, variant_id]`). Binary BCF and gVCF are not supported.

The production path uses htslib through `pysam` and processes a configurable
number of ALT occurrences at a time. It does not load the input file into one
DataFrame. TSV rows pass through the same occurrence contract.

## Usage

```bash
python -m varscore.preprocessing.pipeline \
    -i input.vcf.gz \
    -g genome.fa \
    -f auto \
    -o valid.tsv \
    --invalid-out invalid.tsv \
    --region-out-dir regions/ \
    --occurrence-out-dir ingest/occurrences/ \
    --canonical-out-dir ingest/canonical/valid/ \
    --invalid-parquet-out-dir ingest/canonical/invalid/ \
    --manifest-out ingest/manifest.json \
    --header-out ingest/header.vcf
```

The artifact arguments are optional for older callers. When omitted, the
pipeline writes them under an `ingest/` directory beside `valid.tsv`.

## Outputs

The compatibility outputs remain available:

- `valid.tsv`: headerless `chr, pos, ref, alt, source_variant_id` rows;
- `invalid.tsv`: diagnostic rows with stable error codes;
- bounded region-category TSVs for existing scorers.

The durable outputs are:

- occurrence Parquet shards with `(record_ordinal, alt_index)` identity and one
  row for every ALT, including unsupported alleles;
- canonical Parquet shards for scoreable alleles;
- invalid Parquet shards;
- the parsed source header;
- a manifest containing input/header digests, counts, shard integrity metadata,
  reference build, and canonicalizer version.

Each Parquet shard is atomically renamed into place and has a sibling success
marker containing its row count, byte size, and SHA-256 digest. The final
manifest is the completion marker for the ingest as a whole.

## Semantics

- A multi-allelic VCF record produces one occurrence per ALT in original order.
- Duplicate records remain duplicate occurrences. Canonical scoring identity is
  `chr:pos:ref:alt`, and downstream consumers may select it distinctly.
- VCF `ID` and an optional fifth TSV column become `source_variant_id`.
- Spanning-deletion `*`, symbolic structural variants, breakends, and
  `ALT == REF` remain in the occurrence relation with `UNSUPPORTED` status and a
  precise error code. They are not sent to scoring.
- Bare chromosomes are normalized during reference validation. POS remains
  one-based.
- INFO, FORMAT, and samples are not copied into the canonical relation. The
  immutable source VCF retains them for occurrence-aware augmented output.

## Rejections

- gVCF declarations (`##GVCFBlock*` or `##ALT=<ID=NON_REF,...>`) are rejected
  from the parsed header before output sinks are allocated.
- Undeclared gVCF input fails on the first `<NON_REF>`, legacy `<*>`, or
  `END > POS` reference block without a concrete ALT.
- Ordinary symbolic structural variants with `END` and spanning-deletion `*`
  are not misclassified as gVCF.
- Unsorted VCF records are rejected because augmented output includes a CSI
  random-access index.
- BCF is rejected with an instruction to convert it first.

The older `core.io.read_variants` and `load_variants_vcf` APIs still return a
whole DataFrame for compatibility. Large-file callers must use
`preprocessing.streaming.iter_input_occurrence_batches` or the preprocessing
pipeline.

## Custom variant IDs

The source identifier is preserved through compatibility outputs and the
occurrence relation. It is not the canonical scoring key. See
[variant_id.md](variant_id.md).
