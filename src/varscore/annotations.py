import pandas as pd

# import polars as pl
# import pybedtools
# from pybedtools import BedTool
from pydantic import BaseModel
from typing import List, Tuple

import region_utils


'''
def add_n_closest_elements(
    variants_bed_df: pl.DataFrame,
    elements_bed_file: str,
    k_elements: int,
    element_label: str,
) -> pl.DataFrame:
    """
    Add the n closest elements to each variant in the variants_bed_df DataFrame.
    Parameters:
    variants_bed_df (pl.DataFrame): A Polars DataFrame containing variant inforrmation, represented the BED format, where the first columns being 'chrom', 'chromStart', 'chromEnd', 'name', etc.
                                    bedtools closest requires that all input files are presorted data by chromosome and then by
                                    start position (e.g., sort -k1,1 -k2,2n in.bed > in.sorted.bed for BED files).
    elements_bed_file (str): Path to a file containing all elements (such as genes) to be considered for proximity calculations.
    k_elements (int): The number of closest elements to add for each variant.
    element_label (str): A label to use for the new columns indicating the closest elements and their distances. Example: 'gene'.
    Returns:
    pl.DataFrame: A Polars DataFrame with the variant names and additional columns for the closest elements and their distances, such as
    'variant_id', 'closest_gene_1', 'closest_gene_distance_1', 'closest_gene_2', 'closest_gene_distance_2', etc.
    """
    variants_pd = variants_bed_df.to_pandas()
    variants_bt = BedTool.from_dataframe(variants_pd)
    # Take in the elements file
    elements_pl = pl.read_csv(elements_bed_file, separator="\t")
    elements_pd = elements_pl.to_pandas()
    elements_bt = BedTool.from_dataframe(elements_pd)
    # Get the n closest elements to each variant.
    closests_bed = variants_bt.closest(elements_bt, d=True, t="first", k=k_elements)
    closests_pl = pl.from_pandas(closests_bed.to_dataframe(header=None))
    new_cols = []
    if not closests_pl.is_empty():
        closests_pl = closests_pl.rename(
            {
                "5": "variant_id",
                "9": "close_element",
                closests_pl.columns[-1]: "element_distance",
            }
        )
        closests_pl = closests_pl.group_by(
            pl.col("variant_id"), maintain_order=True
        ).agg(pl.col("close_element"), pl.col("element_distance"))

        for i in range(k_elements):
            new_cols.append(
                pl.col("close_element")
                .list.get(i, null_on_oob=True)
                .alias(f"closest_{element_label}_{i+1}")
            )
            new_cols.append(
                pl.col("element_distance")
                .list.get(i, null_on_oob=True)
                .alias(f"closest_{element_label}_distance_{i+1}")
            )
        closests_pl = closests_pl.with_columns(new_cols)
        closests_pl = closests_pl.drop(["close_element", "element_distance"])
    else:
        # Make empty columns if no elements are found.
        closests_pl = pl.DataFrame(columns=["variant_id"])
        for i in range(k_elements):
            new_cols.append(pl.lit(None).alias(f"closest_{element_label}_{i+1}")),
            new_cols.append(
                pl.lit(None).alias(f"closest_{element_label}_distance_{i+1}")
            )
    return closests_pl


def add_closest_elements_in_window(
    variants_bed_df: pl.DataFrame,
    elements_bed_file: str,
    window_size: str,
    element_label: str,
) -> pd.DataFrame:
    """
    Annotates variants with the closest elements within a specified window size.
    Parameters:
    variants_bed_df (pl.DataFrame): A Polars DataFrame containing variant inforrmation, represented the BED format, where the first columns being 'chrom', 'chromStart', 'chromEnd', 'name', etc.
                                    bedtools closest requires that all input files are presorted data by chromosome and then by
                                    start position (e.g., sort -k1,1 -k2,2n in.bed > in.sorted.bed for BED files).
    elements_bed_file (str): Path to a file containing all elements (such as genes) to be considered for proximity calculations.
    window_size (str): The window size within which to search for the closest elements.
    element_label (str): The label to be used for the annotated elements.
    Returns:
    pd.DataFrame: A Pandas DataFrame with the variant names and additional columns for the closest elements within the specified window size.
                  Columns include 'variant_id' and 'element_label_within_window_{window_size}_bp'.
    """

    variants_pd = variants_bed_df.to_pandas()
    variants_bt = BedTool.from_dataframe(variants_pd)
    # Take in the elements file
    elements_pd = pd.read_table(elements_bed_file, header=None)
    elements_bt = BedTool.from_dataframe(elements_pd)

    closests_bt = variants_bt.window(elements_bt, w=window_size)
    closests_pl = pl.from_pandas(closests_bt.to_dataframe(header=None))
    result_label = f"{element_label}_within_{window_size}_bp"
    if not closests_pl.is_empty():
        closests_pl = closests_pl.rename({"5": "variant_id", "9": "close_element"})

        closests_pl = closests_pl.group_by(
            pl.col("variant_id"), maintain_order=True
        ).agg(pl.col("close_element"))
        closests_pl = closests_pl.with_columns(
            pl.col("close_element").list.join(";").alias(result_label)
        )

        closests_pl = closests_pl.drop("close_element")
    else:
        # TODO convert this to polars
        closests_pl = pl.DataFrame(
            {
                "variant_id": variant_scores["variant_id"],
                f"{element_label}_within_{window_size}_bp": [""]
                * len(variant_scores["variant_id"]),
            }
        )
    return closests_pl
'''


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
    nearest_genes: List[str]
    gene_within_100kb: bool


def annotate_variant(variant):
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

    var_region_type = region_utils.region_type(
        var_chr, var_pos, var_pos + var_ref_length - 1
    )
    var_nearest_genes, var_gene_within_100kb = region_utils.nearest_genes(
        var_chr, var_pos, num_genes=3
    )
    # Instantiate AnnotatedVariant
    var = AnnotatedVariant(
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
    )

    return var


if __name__ == "__main__":

    class TestVariant(BaseModel):
        chr: str
        pos: int
        ref: str
        alt: str

    tv = TestVariant(chr="chr7", pos=27220000, ref="A", alt="TT")
    print(tv)
    av = annotate_variant(tv)
    print(av)
