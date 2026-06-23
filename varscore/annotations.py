import argparse
import logging
import time
import pandas as pd
from pydantic import BaseModel

from typing import List, Optional

logger = logging.getLogger(__name__)

import varscore.utils.region_utils as region_utils
import varscore.utils.variant_utils as variant_utils


class VariantAnnotationInput(BaseModel):
    chr: str
    pos: int
    ref: str
    alt: str


class AnnotatedVariant(BaseModel):
    chr: str
    pos: int
    ref: str
    alt: str
    ref_length: int
    alt_length: int
    variant_length: int
    variant_type: str
    region_type: str            # most-severe single label (back-compat)
    region_labels: List[str]    # all overlapping region labels (multi-label)
    region_gene_ids: List[str]  # genes the variant overlaps
    is_coding: bool             # overlaps a CDS
    in_promoter: bool           # overlaps a promoter window (any gene)
    nearest_genes: List[region_utils.GeneAnnotation]
    gene_within_100kb: bool
    ccre_id: Optional[str]
    ccre_group: Optional[str]
    in_ot: bool
    af_afr: float
    af_ami: float
    af_amr: float
    af_asj: float
    af_eas: float
    af_fin: float
    af_mid: float
    af_nfe: float
    af_sas: float
    af_remaining: float
    
##################
# CORE FUNCTIONS #
##################


def annotate_variants_file(
    variants_loc: str,
    out_path: Optional[str] = None,
    batch_size: int = 2000,
) -> None:
    """Annotate variants from a TSV file and save results as JSONL.

    Reads, annotates, and writes in batches to keep memory usage constant
    regardless of input size.

    Args:
        variants_loc: Path to input TSV file (no header, columns: chr, pos, ref, alt)
        out_path: Path to save annotated variants JSONL
        batch_size: Number of variants to process per batch
    """
    if out_path is None:
        out_path = variants_loc.replace(".tsv", "_annotated.jsonl")

    # Count total lines for progress reporting
    with open(variants_loc) as f:
        total_variants = sum(1 for _ in f)
    logger.info(f"Starting annotation of {total_variants} variants from {variants_loc}")
    logger.info(f"Batch size: {batch_size} | Output: {out_path}")

    total_written = 0
    start_time = time.time()

    # Read input TSV in chunks
    chunks = pd.read_csv(
        variants_loc, sep="\t", header=None,
        names=["chr", "pos", "ref", "alt"],
        chunksize=batch_size,
    )

    for batch_num, chunk_df in enumerate(chunks, start=1):
        batch_start = time.time()

        # Convert chunk to VariantAnnotationInput objects
        variants = [
            VariantAnnotationInput(
                chr=row["chr"],
                pos=int(row["pos"]),
                ref=row["ref"],
                alt=row["alt"],
            )
            for _, row in chunk_df.iterrows()
        ]

        # Annotate batch
        annotated = annotate_variants(variants)

        # Write batch to JSONL (append mode after the first batch)
        records = [av.model_dump() for av in annotated]
        batch_df = pd.DataFrame(records)
        write_mode = "w" if batch_num == 1 else "a"
        batch_df.to_json(out_path, orient="records", lines=True, mode=write_mode)

        total_written += len(records)
        batch_elapsed = time.time() - batch_start
        overall_elapsed = time.time() - start_time
        logger.info(
            f"Batch {batch_num}: annotated {len(records)} variants in {batch_elapsed:.1f}s "
            f"({total_written}/{total_variants} total, {overall_elapsed:.1f}s elapsed)"
        )

    overall_elapsed = time.time() - start_time
    logger.info(f"Done — {total_written} variants annotated in {overall_elapsed:.1f}s -> {out_path}")


def annotate_variants(variants: List[VariantAnnotationInput]) -> List[AnnotatedVariant]:
    """Takes in a list of Variants and returns a list of AnnotatedVariants."""
    return [annotate_variant(variant) for variant in variants]


def annotate_variant(variant: VariantAnnotationInput) -> AnnotatedVariant:
    """Takes in a Variant and returns an AnnotatedVariant."""
    # Get basic data from variant
    var_chr = variant.chr
    var_pos = variant.pos
    var_ref = variant.ref
    var_alt = variant.alt
    # Compute new info
    var_ref_length = len(var_ref)
    var_alt_length = len(var_alt)
    var_variant_length = max(len(var_ref), len(var_alt))

    if var_alt_length > var_ref_length:
        var_variant_type = "insertion"
    elif var_alt_length < var_ref_length:
        var_variant_type = "deletion"
    else:
        var_variant_type = "SNV" if var_ref_length == 1 else "substitution"

    var_start = var_pos
    var_end = var_pos + var_ref_length - 1
    var_region = region_utils.region_annotations(var_chr, var_start, var_end)
    var_nearest_genes, var_gene_within_100kb = region_utils.nearest_genes(
        var_chr, var_pos, num_genes=5
    )

    # CCRE Info
    var_ccre = region_utils.ccre_overlap(var_chr, var_start, var_end)
    var_ccre_id = var_ccre.accession if var_ccre else None
    var_ccre_group = var_ccre.group if var_ccre else None
    
    # Allele Frequency Info
    ot_mafs = variant_utils.get_ot_variant(var_chr, var_pos, var_ref, var_alt)

    # Instantiate AnnotatedVariant
    return AnnotatedVariant(
        chr=var_chr,
        pos=var_pos,
        ref=var_ref,
        alt=var_alt,
        ref_length=var_ref_length,
        alt_length=var_alt_length,
        variant_length=var_variant_length,
        variant_type=var_variant_type,
        region_type=var_region.primary,
        region_labels=var_region.labels,
        region_gene_ids=var_region.gene_ids,
        is_coding=var_region.is_coding,
        in_promoter=var_region.in_promoter,
        nearest_genes=var_nearest_genes,
        gene_within_100kb=var_gene_within_100kb,
        ccre_id=var_ccre_id,
        ccre_group=var_ccre_group,
        in_ot=ot_mafs is not None,
        af_afr=ot_mafs["afr_adj"] if ot_mafs else 0.0,
        af_ami=ot_mafs["ami_adj"] if ot_mafs else 0.0,
        af_amr=ot_mafs["amr_adj"] if ot_mafs else 0.0,
        af_asj=ot_mafs["asj_adj"] if ot_mafs else 0.0,
        af_eas=ot_mafs["eas_adj"] if ot_mafs else 0.0,
        af_fin=ot_mafs["fin_adj"] if ot_mafs else 0.0,
        af_mid=ot_mafs["mid_adj"] if ot_mafs else 0.0,
        af_nfe=ot_mafs["nfe_adj"] if ot_mafs else 0.0,
        af_sas=ot_mafs["sas_adj"] if ot_mafs else 0.0,
        af_remaining=ot_mafs["remaining_adj"] if ot_mafs else 0.0,
    )


########
# MAIN #
########


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    annotate_variants_file(args.variants_loc, args.out_path, args.batch_size)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate variants with region type, nearest genes, cCREs, and allele frequencies."
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True, help="Input TSV file (no header, columns: chr, pos, ref, alt)"
    )
    parser.add_argument(
        "-o", "--out_path", required=False, help="Output JSONL file with annotations"
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=2000, help="Number of variants per batch (default: 2000)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()


"""
python -m varscore.annotations -v variants.tsv -o annotated.jsonl
"""
