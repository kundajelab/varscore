from pydantic import BaseModel
from typing import List, Optional


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


def normalize_from_input(variant:VariantAnnotationInput) -> tuple:
    """Normalize a variant object with .chr, .pos, .ref, .alt attributes."""
    return normalize_variant(variant.chr, variant.pos, variant.ref, variant.alt)

def trio_annotation(
    variants_char: VariantAnnotationInput,
    maternal_variants_set: Optional[set[VariantAnnotationInput]], 
    paternal_variants_set: Optional[set[VariantAnnotationInput]]
) -> str:
    """Classify inheritance for a single child variant against one mother and one father variant.

    Args:
        variants_char: The child variant to classify.
        maternal_variants: A set of mother variants.
        paternal_variants: A set of father variants.

    Returns:
        "Both" if present in both parents, "M" if maternal only,
        "F" if paternal only, or "De_Novo" if absent in both.
    """
    # Convert to normalized set for O(1) lookup
    mother_normal_set = {normalize_from_input(var) for var in maternal_variants_set}
    father_normal_set = {normalize_from_input(var) for var in paternal_variants_set}

    norm = normalize_from_input(variants_char)

    if (norm in mother_normal_set) and (norm in father_normal_set):
        return "Both"
    elif norm in mother_normal_set:
        return "M"
    elif norm in father_normal_set:
        return "F"
    else:
        return "De_Novo"