"""Route variants into region-category files for downstream scoring.

Each variant is annotated with its set of region labels (via region_utils) and
written to every category file whose membership rule it satisfies. Membership is
*overlapping*, not a partition: a variant can land in several files at once (a
CDS-edge SNV is coding + exonic + splice + splice_site + splice_region + genic),
so a downstream pipeline can pick up exactly the set relevant to each scorer.

Outputs are bare, headerless variant TSVs (chr, pos, ref, alt[, variant_id]) so
each file is a drop-in input to the per-scorer entrypoints (alphamissense_scoring,
spliceai_scoring, ...).
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

import varscore.core.io as io_utils
import varscore.annotation.regions as region_utils
from varscore.core.logging import (
    get_logger,
    log_progress,
    log_timing,
    setup_logging,
)

# Initialize logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)


###############
# CATEGORIES  #
###############

# Labels that lie within the transcribed gene body. Promoter is upstream of the
# TSS (regulatory, not transcribed) and intergenic is no gene at all, so both are
# excluded from "genic".
GENE_BODY_LABELS = frozenset({
    "cds", "exonic", "five_prime_utr", "three_prime_utr",
    "intronic", "splice_site", "splice_region", "noncoding_gene",
})

# category name -> predicate over a variant's set of region labels.
# A variant is written to every category whose predicate returns True.
CATEGORY_RULES = {
    # protein-coding territory -> AlphaMissense
    "coding":          lambda L: "cds" in L,
    "exonic":          lambda L: "exonic" in L,            # protein-coding exon umbrella
    "five_prime_utr":  lambda L: "five_prime_utr" in L,
    "three_prime_utr": lambda L: "three_prime_utr" in L,
    # splice territory -> SpliceAI
    "splice":          lambda L: ("splice_site" in L) or ("splice_region" in L),
    "splice_site":     lambda L: "splice_site" in L,       # canonical +/-2bp
    "splice_region":   lambda L: "splice_region" in L,     # wider window
    # other gene / regulatory territory
    "intronic":        lambda L: "intronic" in L,
    "promoter":        lambda L: "promoter" in L,
    "noncoding_gene":  lambda L: "noncoding_gene" in L,
    # gene-body umbrella (any transcribed feature; excludes promoter/intergenic)
    "genic":           lambda L: not L.isdisjoint(GENE_BODY_LABELS),
    # variants that mapped to no annotated region ("what didn't map"); off by default
    "intergenic":      lambda L: not (L - {"intergenic"}),
}
# Note: there is deliberately no "noncoding" category. "Not coding" is ambiguous
# under multi-transcript union and is the wrong axis for chromatin models like
# ChromBPNet, which score any locus (coding included) — feed those the full
# variant set, not a region subset. Route coding/splice via the categories above.

# Emitted unless the caller restricts with --categories. intergenic is available
# on request but not written by default.
DEFAULT_CATEGORIES = [c for c in CATEGORY_RULES if c != "intergenic"]


##################
# CORE FUNCTIONS #
##################


def filter_variants_by_region(
    variants_loc: str,
    out_dir: str,
    categories: Optional[List[str]] = None,
    batch_size: int = 250000,
) -> Dict[str, int]:
    """Route variants into per-region-category TSV files under out_dir.

    Args:
        variants_loc: Path to the input variants TSV (chr, pos, ref, alt[, id]).
        out_dir: Directory to write `<category>.tsv` files into.
        categories: Which categories to emit (default: DEFAULT_CATEGORIES). See
            CATEGORY_RULES for the full vocabulary.
        batch_size: Number of variants to annotate at once to bound memory.

    Returns:
        Mapping of category -> number of variants written to that category's file.
        Files are written incrementally; an empty file is left for any requested
        category with zero matches (so the output set is predictable).
    """
    categories = categories or DEFAULT_CATEGORIES
    unknown = [c for c in categories if c not in CATEGORY_RULES]
    if unknown:
        raise ValueError(
            f"Unknown categories {unknown}. Valid: {list(CATEGORY_RULES)}"
        )

    start_time = datetime.now()
    logger.info("Starting variant region filtering")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Categories: {categories}")
    logger.info(f"Batch size: {batch_size}")

    os.makedirs(out_dir, exist_ok=True)

    logger.info("Loading variants...")
    variant_df = io_utils.load_variants(variants_loc)
    total_variants = len(variant_df)
    logger.info(f"Loaded {total_variants} variants")

    paths = {c: os.path.join(out_dir, f"{c}.tsv") for c in categories}
    counts = {c: 0 for c in categories}
    # Truncate-and-open one handle per category; rows are appended per batch so
    # we never hold the whole (overlapping) output set in memory at once.
    handles = {c: open(paths[c], "w") for c in categories}
    try:
        logger.info(f"Processing variants in batches of {batch_size}...")
        for start_idx in range(0, total_variants, batch_size):
            end_idx = min(start_idx + batch_size, total_variants)
            batch_df = variant_df.iloc[start_idx:end_idx]

            label_sets = _batch_label_sets(batch_df)
            for cat in categories:
                rule = CATEGORY_RULES[cat]
                mask = np.fromiter(
                    (rule(s) for s in label_sets), dtype=bool, count=len(label_sets)
                )
                if not mask.any():
                    continue
                selected = batch_df[mask]
                selected.to_csv(handles[cat], sep="\t", index=False, header=False)
                counts[cat] += len(selected)

            log_progress(logger, end_idx, total_variants, "Region filtering")
    finally:
        for handle in handles.values():
            handle.close()

    logger.info("Filtering summary (variants per category):")
    for cat in categories:
        logger.info(f"  {cat}: {counts[cat]} -> {paths[cat]}")
    log_timing(logger, "Variant region filtering", start_time)

    return counts


def _batch_label_sets(batch_df) -> List[set]:
    """Annotate a batch and return each variant's set of region labels."""
    if batch_df.empty:
        return []
    chroms = batch_df["chr"].to_numpy()
    starts = batch_df["pos"].astype(int).to_numpy()  # 1-based
    ends = starts + batch_df["ref"].astype(str).str.len().to_numpy() - 1

    # One vectorized overlap join for the whole batch (no per-row Python loop).
    annotations = region_utils.region_annotations_batch(chroms, starts, ends)
    return [set(a.labels) for a in annotations]


########
# MAIN #
########


def main():
    """Main function for CLI interface."""
    args = _parse_args()
    categories = (
        [c.strip() for c in args.categories.split(",")] if args.categories else None
    )

    counts = filter_variants_by_region(
        args.variants_loc,
        args.out_dir,
        categories,
        args.batch_size,
    )

    print("\nRegion Filtering Summary:")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
    print(f"Files written to: {args.out_dir}")


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Route variants into overlapping per-region-category TSV files "
            "(coding, splice, promoter, intronic, ...) for downstream scoring."
        )
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True,
        help="Location of the variants file (TSV with chr, pos, ref, alt columns).",
    )
    parser.add_argument(
        "-o", "--out_dir", required=True,
        help="Directory to write <category>.tsv files into.",
    )
    parser.add_argument(
        "--categories",
        help=(
            "Comma-separated subset of categories to emit (default: all except "
            f"intergenic). Valid: {','.join(CATEGORY_RULES)}"
        ),
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=250000,
        help="Number of variants to process at once (default: 250000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
