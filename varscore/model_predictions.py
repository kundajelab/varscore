import pandas as pd
import numpy as np
from functools import reduce
import operator
import argparse

from typing import List

import varscore.utils.io_utils as io_utils
import varscore.utils.region_utils as region_utils


##################
# CORE FUNCTIONS #
##################


def model_predictions(
    fold_0_scores_loc: str,
    fold_1_scores_loc: str,
    fold_2_scores_loc: str,
    fold_3_scores_loc: str,
    fold_4_scores_loc: str,
    peaks_dnatree_loc: str,
    output_path: str,
) -> None:
    """Computes model-level variant scores and if variants are in model peaks.

    Loads and averages per-fold scores into model-level scores. Then, annotates
      if each variant is in a peak for that model or not.

    Args:
        fold_0_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_1_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_2_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_3_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_4_scores_loc: The path to the scores generated from fold 0 of the
          model.
        peaks_dnatree_loc: The path to the model's peaks DNATree.
        save_loc: The path to save the final model predictions.
    """
    # Average fold predictions
    average_scores = _average_fold_predictions(
        fold_0_scores_loc,
        fold_1_scores_loc,
        fold_2_scores_loc,
        fold_3_scores_loc,
        fold_4_scores_loc,
    )
    # Annotate in/out of peak
    average_scores["in_peak"] = _compute_in_peaks(average_scores, peaks_dnatree_loc)
    # Save
    average_scores.to_csv(output_path, sep="\t", index=False)


def _average_fold_predictions(
    fold_0_scores_loc: str,
    fold_1_scores_loc: str,
    fold_2_scores_loc: str,
    fold_3_scores_loc: str,
    fold_4_scores_loc: str,
) -> pd.DataFrame:
    """Load and average per fold scores."""
    fold_paths = [
        fold_0_scores_loc,
        fold_1_scores_loc,
        fold_2_scores_loc,
        fold_3_scores_loc,
        fold_4_scores_loc,
    ]
    score_dfs = [pd.read_csv(x, sep="\t") for x in fold_paths]
    # Construct average_df from columns
    average_df_columns = list()
    for col_name in score_dfs[0].columns:
        df_cols = [df[col_name] for df in score_dfs]
        # io_utils.VARIANT_SCHEME columns come from fold 0
        if col_name in io_utils.VARIANT_SCHEMA:
            average_df_columns.append(score_dfs[0][col_name])
        # Non p-value columns/scores are arithmetically averaged
        elif "pval" not in col_name:
            average_df_columns.append(sum(df_cols) / len(df_cols))
        # p-value columns/scores are geometrically averaged
        else:
            average_df_columns.append(
                reduce(operator.mul, df_cols) ** (1 / len(score_dfs))
            )
    average_df = pd.concat(average_df_columns, axis=1)
    average_df.columns = score_dfs[0].columns
    return average_df


def _compute_in_peaks(
    average_scores: pd.DataFrame, peaks_dnatree_loc: str
) -> List[bool]:
    peaks_dnatree = region_utils.loadDNATree(peaks_dnatree_loc)
    in_peak = []
    for _, row in average_scores.iterrows():
        chro, start, ref = row["chr"], row["pos"], row["ref"]
        end = start + len(ref) - 1
        in_peak.append(peaks_dnatree.overlap(chro, start, end) is not None)
    return in_peak


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Average fold-level scores to compute model-level variant scores, including whether variants are in model peaks."
    )
    parser.add_argument(
        "-f0",
        "--fold_0_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 0 of the model.",
    )
    parser.add_argument(
        "-f1",
        "--fold_1_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 1 of the model.",
    )
    parser.add_argument(
        "-f2",
        "--fold_2_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 2 of the model.",
    )
    parser.add_argument(
        "-f3",
        "--fold_3_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 3 of the model.",
    )
    parser.add_argument(
        "-f4",
        "--fold_4_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 4 of the model.",
    )
    parser.add_argument(
        "-p",
        "--peaks_dnatree_loc",
        required=True,
        type=str,
        help="The path to the model's peaks DNATree.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        help="The path to save the final model predictions.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_predictions(
        args.fold_0_scores_loc,
        args.fold_1_scores_loc,
        args.fold_2_scores_loc,
        args.fold_3_scores_loc,
        args.fold_4_scores_loc,
        args.peaks_dnatree_loc,
        args.output_path,
    )
