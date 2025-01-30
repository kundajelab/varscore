import numpy as np
import pandas as pd


def variant_score_db_reformat(variant_score_db: pd.DataFrame) -> pd.DataFrame:
    """Transform from a variant x model per row DB to a variant per row DF."""
    non_pivot_columns = [
        "variant_id",
        "chr",
        "pos",
        "ref",
        "alt",
        # "ref_length",
        # "alt_length",
        # "variant_length",
        # "variant_type",
        "region_type",
        # "nearest_genes",
        # "gene_within_100kb",
    ]
    pivoted_df = variant_score_db.pivot(
        index=non_pivot_columns,
        columns="model_id",
        values=["logfc", "logfc_pval", "active_allele_quantile", "in_peak"],
    )
    pivoted_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_df.columns]
    pivoted_df = pivoted_df.reset_index()
    return pivoted_df


def prioritize_variants(
    variant_score_db: pd.DataFrame,
) -> pd.DataFrame:
    """Prioritize variants given a variant score DB.

    First, transform from a variant x model per row DB to a variant per row DF
      by calling variant_score_db_reformat(). Then, add prioritization columns.
    """
    # Turn into 2D representation
    models = sorted(set(variant_score_db["model_id"]))
    variants_df = variant_score_db_reformat(variant_score_db)
    # Variant prioritization
    for m in models:
        variants_df[f"prioritized_{m}"] = (
            (np.abs(variants_df[f"logfc_{m}"]) >= 0.25)
            & (variants_df[f"active_allele_quantile_{m}"] >= 0.05)
            & (
                (variants_df["region_type"] == "promoter")
                | (variants_df[f"in_peak_{m}"] == True)
                | (
                    (variants_df[f"in_peak_{m}"] == False)
                    & (variants_df[f"logfc_{m}"] > 0)
                )
            )
        )
    variants_df["prioritized"] = variants_df[[f"prioritized_{m}" for m in models]].any(
        axis=1
    )
    # Most active celltype
    variants_LFC_list = [
        pd.Series(variants_df[f"logfc_{m}"], name=m, dtype=float) for m in models
    ]
    variants_LFC = pd.concat(variants_LFC_list, axis=1)
    variants_LFC_abs = np.abs(variants_LFC)
    most_active_celltype = variants_LFC_abs.idxmax(axis=1)
    most_active_celltype_LFC = variants_LFC.apply(
        lambda row: row[most_active_celltype[row.name]], axis=1
    )
    variants_df["most_active_celltype"] = most_active_celltype
    variants_df["most_active_celltype_logfc"] = most_active_celltype_LFC
    # TODO: Cluster
    # TODO: Reorder columns
    return variants_df


if __name__ == "__main__":
    print("hi")
    df = pd.DataFrame()
    df["variant_id"] = ["v1", "v2", "v1", "v2", "v3", "v3"]
    df["model_id"] = ["A", "A", "B", "B", "A", "B"]
    df["chr"] = [1, 1, 1, 1, 1, 1]
    df["pos"] = [100, 200, 100, 200, 300, 300]
    df["ref"] = ["A", "A", "A", "A", "A", "A"]
    df["alt"] = ["T", "G", "T", "G", "C", "C"]
    df["ref_length"] = [1, 1, 1, 1, 1, 1]
    df["alt_length"] = [1, 1, 1, 1, 1, 1]
    df["variant_length"] = [1, 1, 1, 1, 1, 1]
    df["variant_type"] = [1, 1, 1, 1, 1, 1]
    df["region_type"] = [
        "promoter",
        "intronic",
        "promoter",
        "intronic",
        "exonic",
        "exonic",
    ]
    df["nearest_genes"] = [1, 1, 1, 1, 1, 1]
    df["gene_within_100kb"] = [1, 1, 1, 1, 1, 1]
    df["logfc"] = [1, -1, 2, -2, 4, 3]
    df["logfc_pval"] = [0.001, 0.5, 0.001, 0.05, 0.001, 0.05]
    df["active_allele_quantile"] = [1, 1, 1, 1, 1, 1]
    df["in_peak"] = [True, True, False, True, True, True]
    df["jsd"] = [1, 1, 1, 1, 1, 1]
    print(df)
    print(list(df.columns))
    pivot = variant_score_db_reformat(df)
    print(pivot)
    print(list(pivot.columns))
    prio = prioritize_variants(df)
    print(prio)
    print(prio[["prioritized", "most_active_celltype", "most_active_celltype_logfc"]])
