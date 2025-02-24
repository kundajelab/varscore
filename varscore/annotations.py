import pandas as pd
from pydantic import BaseModel

from typing import List, Optional, Tuple

import varscore.utils.region_utils as region_utils

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
    nearest_genes: List[Tuple[str, int, str]]
    gene_within_100kb: bool
    ccre_id: Optional[str]
    ccre_group: Optional[str]


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
    var_ccre = region_utils.ccre_overlap(var_chr, var_start, var_end)
    var_ccre_id = var_ccre.accession if var_ccre else None
    var_ccre_group = var_ccre.group if var_ccre else None
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
    )


if __name__ == "__main__":
    tv = VariantAnnotationInput(chr="chr6", pos=14501369, ref="A", alt="G")
    print(tv)
    av = annotate_variant(tv)
    print(av)
