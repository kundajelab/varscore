import numpy as np

import os

import src.utils.math_utils as math_utils


##################
# CORE FUNCTIONS #
##################
def average_interpretations(
	fold_0_ref_logcts_loc: str,
	fold_0_ref_logits_loc: str,
	fold_0_ref_shaps_loc: str,
	fold_0_alt_logcts_loc: str,
	fold_0_alt_logits_loc: str,
	fold_0_alt_shaps_loc: str,

	fold_1_ref_logcts_loc: str,
	fold_1_ref_logits_loc: str,
	fold_1_ref_shaps_loc: str,
	fold_1_alt_logcts_loc: str,
	fold_1_alt_logits_loc: str,
	fold_1_alt_shaps_loc: str,

	fold_2_ref_logcts_loc: str,
	fold_2_ref_logits_loc: str,
	fold_2_ref_shaps_loc: str,
	fold_2_alt_logcts_loc: str,
	fold_2_alt_logits_loc: str,
	fold_2_alt_shaps_loc: str,

	fold_3_ref_logcts_loc: str,
	fold_3_ref_logits_loc: str,
	fold_3_ref_shaps_loc: str,
	fold_3_alt_logcts_loc: str,
	fold_3_alt_logits_loc: str,
	fold_3_alt_shaps_loc: str,

	fold_4_ref_logcts_loc: str,
	fold_4_ref_logits_loc: str,
	fold_4_ref_shaps_loc: str,
	fold_4_alt_logcts_loc: str,
	fold_4_alt_logits_loc: str,
	fold_4_alt_shaps_loc: str,

	ref_counts_profile_save_loc: str,
	ref_shap_save_loc: str,
	alt_counts_profile_save_loc: str,
	alt_shap_save_loc: str
) -> None:
	"""Compute average count-scaled profiles and shaps for ref and alt variants.
	"""
	ref_logcts_locs = [fold_0_ref_logcts_loc, fold_1_ref_logcts_loc, fold_2_ref_logcts_loc, fold_3_ref_logcts_loc, fold_4_ref_logcts_loc]
	ref_logits_locs = [fold_0_ref_logits_loc, fold_1_ref_logits_loc, fold_2_ref_logits_loc, fold_3_ref_logits_loc, fold_4_ref_logits_loc]
	ref_shaps_locs = [fold_0_ref_shaps_loc, fold_1_ref_shaps_loc, fold_2_ref_shaps_loc, fold_3_ref_shaps_loc, fold_4_ref_shaps_loc]
	alt_logcts_locs = [fold_0_alt_logcts_loc, fold_1_alt_logcts_loc, fold_2_alt_logcts_loc, fold_3_alt_logcts_loc, fold_4_alt_logcts_loc]
	alt_logits_locs = [fold_0_alt_logits_loc, fold_1_alt_logits_loc, fold_2_alt_logits_loc, fold_3_alt_logits_loc, fold_4_alt_logits_loc]
	alt_shaps_locs = [fold_0_alt_shaps_loc, fold_1_alt_shaps_loc, fold_2_alt_shaps_loc, fold_3_alt_shaps_loc, fold_4_alt_shaps_loc]
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
	ref_profile = exp_ref_logits/np.sum(exp_ref_logits, axis=1, keepdims=True)
	ref_counts_profile = np.exp(ref_avg_logcts)*ref_profile
	alt_exp_logits = np.exp(alt_avg_logits)
	alt_profile = exp_alt_logits/np.sum(exp_alt_logits, axis=1, keepdims=True)
	alt_counts_profile = np.exp(alt_avg_logcts)*alt_profile
	# Save
	np.save(ref_counts_profile_save_loc, ref_counts_profile)
	np.save(ref_shap_save_loc, ref_avg_shaps)
	np.save(alt_counts_profile_save_loc, alt_counts_profile)
	np.save(alt_shap_save_loc, alt_avg_shaps)


########
# MAIN #
########
def main(args):
    # Call the average_interpretation function with arguments
    args = _parse_args()
    average_interpretations(
        args.fold_0_ref_logcts_loc,
        args.fold_0_ref_logits_loc,
        args.fold_0_ref_shaps_loc,
        args.fold_0_alt_logcts_loc,
        args.fold_0_alt_logits_loc,
        args.fold_0_alt_shaps_loc,

        args.fold_1_ref_logcts_loc,
        args.fold_1_ref_logits_loc,
        args.fold_1_ref_shaps_loc,
        args.fold_1_alt_logcts_loc,
        args.fold_1_alt_logits_loc,
        args.fold_1_alt_shaps_loc,

        args.fold_2_ref_logcts_loc,
        args.fold_2_ref_logits_loc,
        args.fold_2_ref_shaps_loc,
        args.fold_2_alt_logcts_loc,
        args.fold_2_alt_logits_loc,
        args.fold_2_alt_shaps_loc,

        args.fold_3_ref_logcts_loc,
        args.fold_3_ref_logits_loc,
        args.fold_3_ref_shaps_loc,
        args.fold_3_alt_logcts_loc,
        args.fold_3_alt_logits_loc,
        args.fold_3_alt_shaps_loc,

        args.fold_4_ref_logcts_loc,
        args.fold_4_ref_logits_loc,
        args.fold_4_ref_shaps_loc,
        args.fold_4_alt_logcts_loc,
        args.fold_4_alt_logits_loc,
        args.fold_4_alt_shaps_loc,

        args.ref_counts_profile_save_loc,
        args.ref_shap_save_loc,
        args.alt_counts_profile_save_loc,
        args.alt_shap_save_loc
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Average interpretations across multiple folds and save the results.")
    
    # Add arguments for all the input locations (folds 0-4, ref/alt files)
    for i in range(5):
        parser.add_argument(f"--fold_{i}_ref_logcts_loc", required=True, help=f"Fold {i} reference log counts location.")
        parser.add_argument(f"--fold_{i}_ref_logits_loc", required=True, help=f"Fold {i} reference logits location.")
        parser.add_argument(f"--fold_{i}_ref_shaps_loc", required=True, help=f"Fold {i} reference SHAPs location.")
        parser.add_argument(f"--fold_{i}_alt_logcts_loc", required=True, help=f"Fold {i} alternate log counts location.")
        parser.add_argument(f"--fold_{i}_alt_logits_loc", required=True, help=f"Fold {i} alternate logits location.")
        parser.add_argument(f"--fold_{i}_alt_shaps_loc", required=True, help=f"Fold {i} alternate SHAPs location.")

    # Add arguments for the output locations
    parser.add_argument("--ref_counts_profile_save_loc", required=True, help="Location to save the reference counts profile.")
    parser.add_argument("--ref_shap_save_loc", required=True, help="Location to save the reference SHAP values.")
    parser.add_argument("--alt_counts_profile_save_loc", required=True, help="Location to save the alternate counts profile.")
    parser.add_argument("--alt_shap_save_loc", required=True, help="Location to save the alternate SHAP values.")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
