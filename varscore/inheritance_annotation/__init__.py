from typing import Optional, Set

from pydantic import BaseModel
import argparse
import logging
import pandas as pd

logger = logging.getLogger(__name__)


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

def classify(
    variants_char: VariantAnnotationInput,
    maternal_variants_norm: Optional[Set[tuple]],
    paternal_variants_norm: Optional[Set[tuple]]
) -> str:
    """Classify inheritance for a single child variant against one mother and one father variant.

    Args:
        variants_char: The child variant to classify.
        maternal_variants_norm: A pre-normalized set of mother variant tuples.
        paternal_variants_norm: A pre-normalized set of father variant tuples.

    Returns:
        "Both" if present in both parents, "M" if maternal only,
        "F" if paternal only, or "De_Novo" if absent in both.
    """
    norm = normalize_from_input(variants_char)

    in_mother = norm in maternal_variants_norm if maternal_variants_norm is not None else None
    in_father = norm in paternal_variants_norm if paternal_variants_norm is not None else None

    if in_mother and in_father:
        return "Both"
    elif in_mother:
        return "M"
    elif in_father:
        return "F"
    else:
        return "De_Novo"

def annotate(
    coding_variants_loc:str,
    out_path:str,
    maternal_variants_loc:Optional[str]=None,
    paternal_variants_loc:Optional[str]=None,
)->None:
    coding_var_df = pd.read_csv(coding_variants_loc, sep='\t', header=None, names=["chr", "pos", "ref", "alt"])

    def build_norm_set(path: str | None) -> set[tuple] | None:
        if path is None:
            return None
        df = pd.read_csv(path, sep='\t', header=None, names=["chr", "pos", "ref", "alt"])
        return {
            normalize_variant(row.chr, row.pos, row.ref, row.alt)
            for row in df.itertuples(index=False)
        }

    maternal_norm_set = build_norm_set(maternal_variants_loc)
    paternal_norm_set = build_norm_set(paternal_variants_loc)

    variants = [
        VariantAnnotationInput(chr=row.chr, pos=int(row.pos), ref=row.ref, alt=row.alt)
        for row in coding_var_df.itertuples(index=False)
    ]
    inheritance = [classify(var, maternal_norm_set, paternal_norm_set) for var in variants]
    results = coding_var_df.copy()
    results['inheritance'] = inheritance

    results.to_csv(out_path, sep='\t', index=False)
    
        

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    annotate(args.coding_variants_loc, args.out_path, args.maternal_variants_loc, args.paternal_variants_loc)

def _parse_args():
    parser = argparse.ArgumentParser(
        description="annotate variant inheritance returns M if inherited from mother, F from father, Both if from both, and De-Novo if from neither"
    )
    parser.add_argument(
        "-v", "--coding_variants_loc", required=True, help="Input TSV file (no header, columns: chr, pos, ref, alt)"
    )
    parser.add_argument(
        "-o", "--out_path", required=True, help="Output tsv file with annotations"
    )
    parser.add_argument(
        "-m", "--maternal_variants_loc", required=False, help="Input TSV file (no header, columns: chr, pos, ref, alt)"
    )
    parser.add_argument(
        "-p", "--paternal_variants_loc", required=False, help="Input TSV file (no header, columns: chr, pos, ref, alt)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    main()

"""
python -m varscore.inheritance_annotation -v coding_variants.tsv -o inheritance_annotation.tsv -m maternal_variants.tsv -p paternal_variants.tsv 
"""