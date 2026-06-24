"""CLI to convert an input variants file into the canonical headerless TSV.

The actual format loading/dispatch lives in ``core.io`` (``read_variants`` +
the per-format adapters ``load_variants`` / ``load_variants_vcf``). This module
is just the user-facing converter entry point:

    python -m varscore.preprocessing.vcf -v input.vcf.gz -o variants.tsv

VCF/gVCF input is described in ``docs/vcf.md``. Validation and the full
preprocessing pipeline accept VCF directly via their own ``-f/--format`` flag, so
this converter is only needed when you want the canonical TSV as a standalone
artifact.
"""

import argparse
import logging

import varscore.core.io as io_utils
from varscore.core.logging import get_logger, setup_logging

# Initialize logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)


def main():
    args = _parse_args()
    df = io_utils.read_variants(args.variants_loc, args.format)
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
