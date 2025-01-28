import numpy as np
import pandas as pd
from scipy.stats import binom

import src.utils.chrombpnet_utils as chrombpnet_utils


##################
# CORE FUNCTIONS #
##################
def interpret(
    model_loc: str,
    variants_loc: str,
    genome_loc: str,
    peaks_dist_loc: str,
    ref_logcts_save_loc: str,
    ref_logits_save_loc: str,
    ref_shaps_save_loc: str,
    alt_logcts_save_loc: str,
    alt_logits_save_loc: str,
    alt_shaps_save_loc: str
) -> None:
    """Score variants through a model.

    Generates one-hot encodings for all the variants and scores them through a
      model. Saves scores in a tsv format with each score saved as a new column
      on the original variants dataframe.

    Args:
        model_loc: The path of a model.
        variants_loc: The path of variants tsv.
        genome_loc: The path to the model's genome fasta.
        peaks_dist_loc: The path to the model's peak distribution.
        ref_logcts_save_loc: The path to save the ref logcounts.
        ref_logits_save_loc: The path to save the ref profile logits.
        ref_shaps_save_loc: The path to save the ref Shap scores.
        alt_logcts_save_loc: The path to save the alt logcounts.
        alt_logits_save_loc: The path to save the alt profile logits.
        alt_shaps_save_loc: The path to save the alt Shap scores.
    """
    # Get reference and alternate sequences
    ref_seqs, alt_seqs = io_utils.get_variant_seqs(variants_loc, genome_loc)
    # Load model
    model = chrombpnet_utils.load_chrombpnet(model_loc)
    # Make predictions
    ref_pred_logits, ref_pred_logcts = chrombpnet_utils.predict(model, ref_seqs)
    alt_pred_logits, alt_pred_logcts = chrombpnet_utils.predict(model, alt_seqs)
    # Compute count shaps
    ref_shaps = chrombpnet_utils.deepshap(model, ref_seqs)
    alt_shaps = chrombpnet_utils.deepshap(model, alt_seqs)
    # Save
    np.save(ref_logcts_save_loc, ref_pred_logcts)
    np.save(ref_logits_save_loc, ref_pred_logits)
    np.save(ref_shaps_save_loc, ref_shaps)
    np.save(alt_logcts_save_loc, alt_pred_logcts)
    np.save(alt_logits_save_loc, alt_pred_logits)
    np.save(alt_shaps_save_loc, alt_shaps)


########
# MAIN #
########
def main(args):
    # Call the interpret function with arguments
    args = _parse_args()
    interpret(
        args.model_loc,
        args.variants_loc,
        args.genome_loc,
        args.peaks_dist_loc,
        args.ref_logcts_save_loc,
        args.ref_logits_save_loc,
        args.ref_shaps_save_loc,
        args.alt_logcts_save_loc,
        args.alt_logits_save_loc,
        args.alt_shaps_save_loc
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Interpret variants using a trained model and save the results.")
    
    # Add arguments for input file locations
    parser.add_argument("--model_loc", required=True, help="Location of the model file.")
    parser.add_argument("--variants_loc", required=True, help="Location of the variants file.")
    parser.add_argument("--genome_loc", required=True, help="Location of the genome file.")
    parser.add_argument("--peaks_dist_loc", required=True, help="Location of the peaks distribution file.")
    
    # Add arguments for saving the output
    parser.add_argument("--ref_logcts_save_loc", required=True, help="Location to save the reference log counts.")
    parser.add_argument("--ref_logits_save_loc", required=True, help="Location to save the reference logits.")
    parser.add_argument("--ref_shaps_save_loc", required=True, help="Location to save the reference SHAP values.")
    parser.add_argument("--alt_logcts_save_loc", required=True, help="Location to save the alternate log counts.")
    parser.add_argument("--alt_logits_save_loc", required=True, help="Location to save the alternate logits.")
    parser.add_argument("--alt_shaps_save_loc", required=True, help="Location to save the alternate SHAP values.")
    
    return parser.parse_args()


if __name__ == "__main__":
    main()
