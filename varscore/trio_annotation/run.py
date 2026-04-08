import os
import pandas as pd
from pydantic import BaseModel

import argparse
from varscore.utils.logging_config import get_logger

logger = get_logger(__name__)

TRIO_VARIANT_SCHEMA = ["chr", "pos", "ref", "alt"]


class VariantAnnotationInput(BaseModel):
    chr: str
    pos: int
    ref: str
    alt: str

def normalize_variant(chrom, pos, ref, alt):
    """
    Trims common prefix/suffix from ref/alt and adjusts position accordingly.

    Does NOT left-shift against a reference FASTA. Complex indels with multiple
    valid representations may not be matched correctly.
    """
    # Trim common suffix
    while len(ref) > 1 and len(alt) > 1 and ref[-1] == alt[-1]:
        ref, alt = ref[:-1], alt[:-1]
    # Trim common prefix (adjust pos)
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref, alt = ref[1:], alt[1:]
        pos += 1
    return chrom, pos, ref, alt


def normalize_from_input(variant) -> tuple:
    """Normalize a variant object with .chr, .pos, .ref, .alt attributes."""
    return normalize_variant(variant.chr, variant.pos, variant.ref, variant.alt)

def trio_annotation(
    variants_char: VariantAnnotationInput,
    maternal_var_char: VariantAnnotationInput,
    paternal_var_char: VariantAnnotationInput,
) -> str:
    """Classify inheritance for a single child variant against one mother and one father variant.

    Args:
        variants_char: The child variant to classify.
        maternal_var_char: The corresponding mother variant.
        paternal_var_char: The corresponding father variant.

    Returns:
        "Both" if present in both parents, "M" if maternal only,
        "F" if paternal only, or "De_Novo" if absent in both.
    """
    norm = normalize_from_input(variants_char)
    in_mother = norm == normalize_from_input(maternal_var_char)
    in_father = norm == normalize_from_input(paternal_var_char)

    if in_mother and in_father:
        return "Both"
    elif in_mother:
        return "M"
    elif in_father:
        return "F"
    else:
        return "De_Novo"


