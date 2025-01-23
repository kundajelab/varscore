import numpy as np
import pandas as pd


def variant_score_db_reformat(variant_score_db: pd.DataFrame) -> pd.DataFrame:
	"""Transform from a variant x model per row DB to a variant per row DF.
	"""
	non_pivot_columns = ["variant_id", "chr", "pos", "ref", "alt", "ref_length", "alt_length", "variant_length", "variant_type", "region_type", "nearest_genes", "gene_within_100kb"]
	pivoted_df = variant_score_db.pivot(index=non_pivot_columns, columns="model_id", values=["logfc", "logfc_pval", "active_allele_quantile", "in_peak"])
	pivoted_df.columns = [f"{col[0]_{col[1]}}" for col in pivoted_df.columns]
	pivoted_df = pivoted_df.reset_index()
	return pivoted_df


def prioritize_variants(variant_score_db: pd.DataFrame, ) -> pd.DataFrame:
	"""Prioritize variants given a variant score DB.

	First, transform from a variant x model per row DB to a variant per row DF
	  by calling variant_score_db_reformat(). Then, add prioritization columns.
	"""
	# Turn into 2D representation
	models = sorted(set(variant_score_db["model_id"]))
	variants_df = variant_score_db_reformat(variant_score_db)
	# Variant prioritization
	for m in models:
		variants_df[f"prioritized_{m}"] = (variants_df[f"logfc_pval_{m}"] <= 0.01) # LFC p-val <= 0.01
											and (variants_df[f"active_allele_quantile_{m}"] >= 0.05) # Active allele quantile >= 0.05
											and ((variants_df["region_type"] == "promoter") # In promoter
												or ((variants_df[f"in_peak_{m}"] == True) and (variants_df[f"logfc_{m}"] < 0)) # In peak with negative LFC
												or ((variants_df[f"in_peak_{m}"] == False) and (variants_df[f"logfc_{m}"] > 0))) # Out of peak with positive LFC
	variants_df["prioritized"] = variants_df[[f"prioritized_{m}" for m in models]].any(axis=1)
	# Most active celltype
	variants_LFC_list = [pd.Series(variants_df[f"logfc_{m}"]*variants_df[f"prioritized_{m}"], name=m) for m in models]
	variants_LFC = pd.concat(variants_LFC_list, axis=1)
	variants_LFC_abs = np.abs(variants_LFC)
	most_active_celltype = variants_LFC_abs.idxmax(axis=1)
	most_active_celltype_LFC = variants_LFC.apply(lambda row: row[most_active_celltype[row.name]], axis=1)
	variants_df["most_active_celltype"] = most_active_celltype
	variants_df["most_active_celltype_logfc"] = most_active_celltype_LFC
	# TODO: Cluster
	# TODO: Reorder columns
	return variants_df