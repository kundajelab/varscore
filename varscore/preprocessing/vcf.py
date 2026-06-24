"""VCF/gVCF input adapter for the variant preprocessing entry point.

The whole downstream pipeline (validate -> region_filter -> scorers ->
prioritization) consumes the canonical headerless variant table
(``chr, pos, ref, alt[, variant_id]`` -- ``core.io.VARIANT_SCHEMA``). VCF support
is therefore purely an *entry-point adapter*: parse VCF/gVCF into that canonical
DataFrame and nothing downstream changes.

Implementation notes:
  - Pure Python, **no new dependency**. Stdlib ``gzip`` streams both plain ``.vcf``
    and bgzipped ``.vcf.gz`` (bgzip is a valid gzip stream; same trick as
    ``annotation/maf.py``). Consequence: binary ``.bcf`` is not supported (convert
    with ``bcftools view``), and there is no tabix random access -- we scan once.
  - Multi-allelic ``ALT`` (comma-separated) is split into one canonical row per
    concrete ALT allele.
  - Symbolic / non-substitution alleles (``<NON_REF>``, ``<*>``, ``<DEL>``, ``*``,
    breakends) are dropped -- these dominate gVCF reference blocks and would
    otherwise explode into millions of non-variant rows.
  - VCF ``POS`` is 1-based, matching ``pos``. ``ID == "."`` -> missing
    ``variant_id``; any other ID is carried through (so the new input path is
    *extensible* to end-to-end ID preservation -- see ``preprocessing/validate.py``
    for the one remaining drop that gates that future work).
  - Chromosome naming is passed through untouched; ``core/io.validate_variant``
    normalizes bare ``1`` -> ``chr1`` and rejects non-ACGT alleles downstream, so
    we deliberately do not duplicate that here.

See ``docs/vcf.md`` for the user-facing reference.
"""

import argparse
import gzip
import logging

import pandas as pd

import varscore.core.io as io_utils
from varscore.core.logging import get_logger, setup_logging

# Initialize logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)

# Mirrors io_utils.VARIANT_SCHEMA; redefined here to keep parsing self-contained.
VARIANT_SCHEMA = ["chr", "pos", "ref", "alt", "variant_id"]

# Extensions that indicate gzip/bgzip compression.
_GZIP_SUFFIXES = (".gz", ".bgz")
# Extensions that indicate a VCF (after stripping any gzip suffix).
_VCF_SUFFIXES = (".vcf", ".gvcf")


##################
# CORE FUNCTIONS #
##################


def _is_symbolic_alt(alt: str, ref: str) -> bool:
    """Return True if ``alt`` is not a concrete REF->ALT base substitution.

    Drops the alleles that pepper gVCF reference blocks and structural-variant
    records and that downstream genome-aware validation could not score anyway:
    the symbolic ``<...>`` forms (``<NON_REF>``, ``<*>``, ``<DEL>``, ...), the
    spanning-deletion star / missing / empty allele, breakend notation, and the
    monomorphic ``alt == ref`` no-op some gVCF tools emit. Concrete sequences
    (including indels, and even malformed ones like lowercase/``N``) are kept and
    left for ``core/io.validate_variant`` to vet, matching the TSV input path.
    """
    if alt in {"*", ".", ""}:
        return True
    if alt.startswith("<") and alt.endswith(">"):
        return True
    if "[" in alt or "]" in alt:
        return True
    if alt == ref:
        return True
    return False


def load_variants_vcf(path: str) -> pd.DataFrame:
    """Parse a VCF/gVCF (``.vcf`` or ``.vcf.gz``) into the canonical variant table.

    Args:
        path: Path to a plain or bgzipped VCF/gVCF file.

    Returns:
        DataFrame with columns ``chr, pos, ref, alt, variant_id`` (one row per
        concrete ALT allele). The columns are always present, even when empty.
    """
    opener = gzip.open if path.endswith(_GZIP_SUFFIXES) else open

    rows = []
    n_sites = 0  # data (non-header) lines seen
    n_symbolic = 0  # symbolic / non-variant ALT alleles dropped

    with opener(path, "rt") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.startswith("#"):  # ## meta + #CHROM header
                continue
            line = line.rstrip("\n")
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 5:
                raise ValueError(
                    f"{path}:{lineno}: malformed VCF data line, expected at least "
                    f"5 tab-separated fields (CHROM POS ID REF ALT), got {len(fields)}"
                )
            chrom, pos, vid, ref, alt = fields[:5]
            n_sites += 1

            variant_id = None if vid == "." else vid
            pos = int(pos)  # VCF POS is 1-based, matching our schema

            for allele in alt.split(","):  # split multi-allelic sites
                if _is_symbolic_alt(allele, ref):
                    n_symbolic += 1
                    continue
                rows.append((chrom, pos, ref, allele, variant_id))

    logger.info(
        "Parsed %d VCF sites -> %d concrete variant rows; "
        "dropped %d symbolic/non-variant ALT alleles",
        n_sites,
        len(rows),
        n_symbolic,
    )

    return pd.DataFrame(rows, columns=VARIANT_SCHEMA)


def read_variants(path: str, fmt: str = "auto") -> pd.DataFrame:
    """Load variants into the canonical table, dispatching on file format.

    Args:
        path: Input variants file.
        fmt: One of ``"auto"``, ``"tsv"``, ``"vcf"``. ``"auto"`` routes
            ``.vcf``/``.vcf.gz`` (and ``.gvcf``) to the VCF parser and everything
            else to the headerless-TSV loader; ``.bcf`` raises a clear error.

    Returns:
        DataFrame with columns ``chr, pos, ref, alt, variant_id``.
    """
    if fmt == "vcf":
        return load_variants_vcf(path)
    if fmt == "tsv":
        return io_utils.load_variants(path)
    if fmt != "auto":
        raise ValueError(f"Unknown format {fmt!r}; expected one of auto, tsv, vcf.")

    # auto-detect by extension (strip any gzip suffix first)
    base = path
    if base.endswith(_GZIP_SUFFIXES):
        base = base.rsplit(".", 1)[0]
    if base.endswith(".bcf") or path.endswith(".bcf"):
        raise ValueError(
            f"Binary BCF is not supported ({path}); convert to VCF first, e.g. "
            "`bcftools view in.bcf -Oz -o out.vcf.gz`."
        )
    if base.endswith(_VCF_SUFFIXES):
        return load_variants_vcf(path)
    return io_utils.load_variants(path)


########
# MAIN #
########


def main():
    args = _parse_args()
    df = read_variants(args.variants_loc, args.format)
    # Write the full 5-col canonical table (variant_id included) so the converter
    # output faithfully carries any VCF IDs forward.
    df.to_csv(args.out_path, sep="\t", index=False, header=False)
    logger.info("Wrote %d variant rows to %s", len(df), args.out_path)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a VCF/gVCF into the canonical headerless variant TSV "
        "(chr, pos, ref, alt, variant_id)."
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True,
        help="Input variants file (.vcf, .vcf.gz, or canonical TSV).",
    )
    parser.add_argument(
        "-o", "--out_path", required=True,
        help="Location to save the canonical headerless variant TSV.",
    )
    parser.add_argument(
        "-f", "--format", default="auto", choices=["auto", "tsv", "vcf"],
        help="Input format (default: auto, detected by extension).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()


"""
python -m varscore.preprocessing.vcf -v input.vcf.gz -o variants.tsv
"""
