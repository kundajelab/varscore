import numpy as np

import argparse
import os


##################
# CORE FUNCTIONS #
##################
def average_interpretations(
    fold_0_dir: str,
    fold_1_dir: str,
    fold_2_dir: str,
    fold_3_dir: str,
    fold_4_dir: str,
    save_dir: str,
) -> None:
    """Compute average count-scaled profiles and shaps for ref and alt variants."""
    fold_dirs = [fold_0_dir, fold_1_dir, fold_2_dir, fold_3_dir, fold_4_dir]
    ref_logcts_locs = [os.path.join(fd, "ref_logcts.npy") for fd in fold_dirs]
    ref_logits_locs = [os.path.join(fd, "ref_logits.npy") for fd in fold_dirs]
    ref_shaps_locs = [os.path.join(fd, "ref_shaps.npy") for fd in fold_dirs]
    alt_logcts_locs = [os.path.join(fd, "alt_logcts.npy") for fd in fold_dirs]
    alt_logits_locs = [os.path.join(fd, "alt_logits.npy") for fd in fold_dirs]
    alt_shaps_locs = [os.path.join(fd, "alt_shaps.npy") for fd in fold_dirs]
    # Load data
    ref_logcts = [np.load(x) for x in ref_logcts_locs]
    ref_logits = [np.load(x) for x in ref_logits_locs]
    ref_shaps = [np.load(x) for x in ref_shaps_locs]
    alt_logcts = [np.load(x) for x in alt_logcts_locs]
    alt_logits = [np.load(x) for x in alt_logits_locs]
    alt_shaps = [np.load(x) for x in alt_shaps_locs]
    # Average
    ref_avg_logcts = np.mean(ref_logcts, axis=0)
    ref_avg_logits = np.mean(ref_logits, axis=0)
    ref_avg_shaps = np.mean(ref_shaps, axis=0)
    alt_avg_logcts = np.mean(alt_logcts, axis=0)
    alt_avg_logits = np.mean(alt_logits, axis=0)
    alt_avg_shaps = np.mean(alt_shaps, axis=0)
    # Compute count-scaled profiles
    ref_exp_logits = np.exp(ref_avg_logits)
    ref_profile = ref_exp_logits / np.sum(ref_exp_logits, axis=1, keepdims=True)
    ref_counts_profile = np.exp(ref_avg_logcts) * ref_profile
    alt_exp_logits = np.exp(alt_avg_logits)
    alt_profile = alt_exp_logits / np.sum(alt_exp_logits, axis=1, keepdims=True)
    alt_counts_profile = np.exp(alt_avg_logcts) * alt_profile
    # Save
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, "average_ref_profiles.npy"), ref_counts_profile)
    np.save(os.path.join(save_dir, "average_ref_shaps.npy"), ref_avg_shaps)
    np.save(os.path.join(save_dir, "average_alt_profiles.npy"), alt_counts_profile)
    np.save(os.path.join(save_dir, "average_alt_shaps.npy"), alt_avg_shaps)


########
# MAIN #
########
def main():
    # Call the average_interpretation function with arguments
    args = _parse_args()
    average_interpretations(
        args.fold_0_dir,
        args.fold_1_dir,
        args.fold_2_dir,
        args.fold_3_dir,
        args.fold_4_dir,
        args.save_dir,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Average interpretations across multiple folds and save the results."
    )
    parser.add_argument(
        "--fold_0_dir",
        required=True,
        help=f"The directory where fold 0 interpretations are.",
    )
    parser.add_argument(
        "--fold_1_dir",
        required=True,
        help=f"The directory where fold 1 interpretations are.",
    )
    parser.add_argument(
        "--fold_2_dir",
        required=True,
        help=f"The directory where fold 2 interpretations are.",
    )
    parser.add_argument(
        "--fold_3_dir",
        required=True,
        help=f"The directory where fold 3 interpretations are.",
    )
    parser.add_argument(
        "--fold_4_dir",
        required=True,
        help=f"The directory where fold 4 interpretations are.",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help=f"The directory to save averaged interpretations to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""
uv run python -m src.shap.average_interpretations --fold_0_dir ~/varscore_test/fold_0_interpretations --fold_1_dir ~/varscore_test/fold_1_interpretations --fold_2_dir ~/varscore_test/fold_2_interpretations --fold_3_dir ~/varscore_test/fold_3_interpretations --fold_4_dir ~/varscore_test/fold_4_interpretations --save_dir ~/varscore_test/average_interpretations
"""
