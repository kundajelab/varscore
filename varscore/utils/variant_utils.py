import gzip
import pickle
from functools import lru_cache
import pandas as pd
import os


VARIANTS_DF_PATH = "./varscore/data/variants.pkl.gz"


@lru_cache(maxsize=1)
def _get_ot_variants_df():
    if not os.path.exists(VARIANTS_DF_PATH):
        raise FileNotFoundError(f"Variants dataframe path not found. You need to generate this file first. Please refer to documentation for instructions.")
    print("Loading variants dataframe ...")
    with gzip.open(VARIANTS_DF_PATH, "rb") as f:
        return pickle.load(f)


def get_ot_variant(chr: str, pos: int, ref: str, alt: str) -> pd.Series:
    """Get the OT variant for a given chromosome, position, reference, and alternative allele."""
    chr = chr.replace("chr", "")
    try:
        return _get_ot_variants_df().loc[(chr, pos, ref, alt)].to_dict()
    except KeyError:
        return None


if __name__ == "__main__":
    print(get_ot_variant("chrX", 103790599, "A", "G"))
