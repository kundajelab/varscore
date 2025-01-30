import pandas as pd

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
    model_peaks_dnatree_loc: str,
    save_loc: str
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
        model_peaks_dnatree_loc: The path to the model's peaks DNATree.
        save_loc: The path to save the final model predictions.
    """
    # Average fold predictions
    average_scores = __average_fold_predictions(fold_0_scores_loc, fold_1_scores_loc, fold_2_scores_loc, fold_3_scores_loc, fold_4_scores_loc)
    # Annotate in/out of peak
    average_scores["in_peak"] = __compute_in_peaks(average_scores, model_peaks_dnatree_loc)
    # Save
    average_scores.to_csv(save_loc, sep="\t", index=False)


def __average_fold_predictions(
    fold_0_scores_loc: str,
    fold_1_scores_loc: str,
    fold_2_scores_loc: str,
    fold_3_scores_loc: str,
    fold_4_scores_loc: str
) -> pd.DataFrame:
    """Load and average per fold scores.
    """
    fold_paths = [fold_0_scores_loc, fold_1_scores_loc, fold_2_scores_loc, fold_3_scores_loc, fold_4_scores_loc]
    score_dfs = [pd.read_csv(x, sep="\t") for x in fold_paths]
    # Construct average_df from columns
    average_df_columns = list()
    for x in score_dfs[0].columns:
        # io_utils.VARIANT_SCHEME columns come from fold 0
        if x in io_utils.VARIANT_SCHEMA:
            average_df_columns.append(score_dfs[0][x])
        # Non p-value columns/scores are arithmetically averaged
        elif "pval" not in x:
            average_df_columns.append((score_dfs[0][x] + score_dfs[1][x] + score_dfs[2][x] + score_dfs[3][x] + score_dfs[4][x])/5)
        # p-value columns/scores are geometrically averaged
        else:
            average_df_columns.append((score_dfs[0][x] * score_dfs[1][x] * score_dfs[2][x] * score_dfs[3][x] * score_dfs[4][x])**(1/5))
    average_df = pd.concat(average_df_columns, axis=1, names=scores_dfs[0].columns)
    return average_df

def __compute_in_peaks(average_scores: pd.DataFrame, peaks_dnatree_loc: str) -> List[bool]:
    peaks_dnatree = region_utils.loadDNATree(peaks_dnatree_loc)
    in_peak = []
    for _, row in average_scores.iterrows():
        chro, start, ref = row["chr"], row["pos"], row["ref"]
        end = start + len(ref) - 1
        in_peak.append(peaks_dnatree.overlap(chro, start, end) is not None)
    return in_peak