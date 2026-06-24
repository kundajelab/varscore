import numpy as np
import pandas as pd
from scipy.stats import binom

import argparse
import os

import varscore.scoring.chrombpnet.model as chrombpnet_utils
import varscore.core.io as io_utils


##################
# CORE FUNCTIONS #
##################
def interpret(
    model_loc: str, variants_loc: str, genome_loc: str, output_dir: str
) -> None:
    """Score variants through a model.

    Generates one-hot encodings for all the variants and scores them through a
      model. Saves scores in a tsv format with each score saved as a new column
      on the original variants dataframe.

    Args:
        model_loc: The path of a model.
        variants_loc: The path of variants tsv.
        genome_loc: The path to the model's genome fasta.
        save_dir: The directory to save all outputs to.
    """
    # Get reference and alternate sequences
    variant_df = io_utils.load_variants(variants_loc)
    ref_seqs, alt_seqs = io_utils.get_variant_seqs(variant_df, genome_loc)
    # Load model
    model = chrombpnet_utils.load_chrombpnet(model_loc)
    # Make predictions
    ref_logits, ref_logcts = chrombpnet_utils.predict(model, ref_seqs)
    alt_logits, alt_logcts = chrombpnet_utils.predict(model, alt_seqs)
    # Compute count shaps
    ref_shaps = chrombpnet_utils.deepshap(model, ref_seqs)
    alt_shaps = chrombpnet_utils.deepshap(model, alt_seqs)
    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "ref_logcts.npy"), ref_logcts)
    np.save(os.path.join(output_dir, "ref_logits.npy"), ref_logits)
    np.save(os.path.join(output_dir, "ref_shaps.npy"), ref_shaps)
    np.save(os.path.join(output_dir, "alt_logcts.npy"), alt_logcts)
    np.save(os.path.join(output_dir, "alt_logits.npy"), alt_logits)
    np.save(os.path.join(output_dir, "alt_shaps.npy"), alt_shaps)


########
# MAIN #
########


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Interpret variants using a trained model and save the results."
    )
    parser.add_argument(
        "-m", "--model_loc", required=True, help="Location of the model file."
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True, help="Location of the variants file."
    )
    parser.add_argument(
        "-g", "--genome_loc", required=True, help="Location of the genome file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="The directory to save all outputs to.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    interpret(args.model_loc, args.variants_loc, args.genome_loc, args.output_dir)

"""
create a dummy ~/varscore_test/test_variants.tsv

uv run python -m src.shap.interpretation --model_loc /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15/chrombpnet_wo_bias.h5 --variants_loc ~/varscore_test/test_variants.tsv --genome_loc /oak/stanford/groups/akundaje/soumyak/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta --save_dir ~/varscore_test/fold_0_interpretations

then copy ~/varscore_test/fold_0_interpretations for 4 more folds
"""
