import pandas as pd
from pydantic import BaseModel

from typing import List, Optional, Tuple

import varscore.utils.region_utils as region_utils
import varscore.utils.variant_utils as variant_utils
# import polars as pl
# import pybedtools
# from pybedtools import BedTool


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
    region_type: str
    nearest_genes: List[Tuple[str, str, int, str]]
    """(gene_name, gene_id, distance, type)"""
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
    var_region_type = region_utils.region_type(
        var_chr, var_start, var_end
    )
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
        region_type=var_region_type,
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


if __name__ == "__main__":
    tv = VariantAnnotationInput(chr="chr6", pos=14501369, ref="A", alt="G")
    av = annotate_variant(tv)
    print(av)
    print(annotate_variant(VariantAnnotationInput(chr="chr1", pos=109208255, ref="T", alt="A")))
