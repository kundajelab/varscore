import numpy as np
import pandas as pd
from scipy.stats import binom

import src.utils.chrombpnet_utils as chrombpnet_utils


##################
# CORE FUNCTIONS #
##################
def score_variants(
    model_loc: str,
    variants_loc: str,
    genome_loc: str,
    peaks_dist_loc: str,
    save_loc: str,
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
        save_loc: The path to save the scores dataframe.
    """
    # Load variants
    variants_df = _load_variants(variants_loc)
    # Get reference and alternate sequences
    with pyfaidx.Fasta(genome_loc) as genome:
        ref_seqs, alt_seqs = _get_variant_seqs(variants_df, genome)
    # Load model
    model = chrombpnet_utils.load_chrombpnet(model_loc)
    # Make predictions
    ref_pred_logits, ref_pred_logcts = chrombpnet_utils.predict(model, ref_seqs)
    alt_pred_logits, alt_pred_logcts = chrombpnet_utils.predict(model, alt_seqs)
    # Compute scores
    peaks_dist = np.load(peaks_dist_loc)
    scores = _scores_from_preds(ref_pred_logcts, alt_pred_logcts, peaks_dist)
    # Save
    for score_name, score_vals in scores.items():
        variant_df[score_name] = score_vals
    variant_df.to_csv(save_loc, sep="\t", index=False)


def _load_variants(variants_loc: str) -> pd.DataFrame:
    """Load a variants DataFrame."""
    VARIANT_SCHEMA = ["chr", "pos", "ref", "alt"]
    variants_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    return variants_df


def _get_variant_seqs(variants_df, genome, width=2114):
    """Get one-hot encoded peak sequences from a DataFrame of peaks."""
    ref_sequences = []
    alt_sequences = []
    for _, row in regions.iterrows():
        chro, pos, ref, alt = row["chro"], int(row["pos"]) - 1, row["ref"], row["alt"]
        ref_seq = str(genome[chro][pos - width // 2 : pos + width // 2])
        assert ref_seq[width // 2 : width // 2 + len(ref)] == ref
        alt_seq = (
            ref_seq[: width // 2]
            + alt
            + str(genome[chro][pos + len(ref) : pos + width // 2 + len(ref) - len(alt)])
        )
        assert len(alt_seq) == width
        assert alt_seq[width // 2 : width // 2 + len(alt)] == alt
        ref_sequences.append(ref_seq)
        alt_sequences.append(alt_seq)
    ref_onehot = chrombpnet_utils.dna_to_one_hot(ref_sequences)
    alt_onehot = chrombpnet_utils.dna_to_one_hot(alt_sequences)
    return ref_onehot, alt_onehot


def _scores_from_preds(
    ref_logcts: np.ndarray, alt_logcts: np.ndarray, peaks_dist: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute scores based on logcount predictions."""
    scores_dict = dict()
    # Log2 Fold Change
    scores_dict["lfc"] = (alt_logcts - ref_logcts) / np.exp(2)
    # LFC p-Value
    scores_dict["lfc-pval"] = _compute_lfc_pval(ref_logcts, alt_logcts)
    # Active allele percentile
    ref_quantiles = np.searchsorted(peaks_dist, ref_logcts) / len(ref_quantiles)
    alt_quantiles = np.searchsorted(peaks_dist, alt_logcts) / len(alt_quantiles)
    scores_dict["active-allele-quantile"] = np.maximum(ref_quantiles, alt_quantiles)
    # Return
    return scores_dict


def _compute_lfc_pval(ref_logcts, alt_logcts):
    """Computes LFC p-values."""
    min_cts = np.exp(np.minimum(ref_logcts, alt_logcts))
    total_cts = np.exp(ref_logcts) + np.exp(alt_logcts)
    return [binom.cdf(min_cts[i], total_cts[i], 0.5) for i in range(len(ref_logcts))]


########
# MAIN #
########
def main(args):
    # Call the ingest_model function with arguments
    args = _parse_args()
    score_variants(
        args.model_loc,
        args.variants_loc,
        args.genome_loc,
        args.peaks_dist_loc,
        args.save_loc,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Score variants using a trained model and associated data."
    )
    parser.add_argument(
        "--model_loc", required=True, help="Location of the model file."
    )
    parser.add_argument(
        "--variants_loc", required=True, help="Location of the variants file."
    )
    parser.add_argument(
        "--genome_loc", required=True, help="Location of the genome file."
    )
    parser.add_argument(
        "--peaks_dist_loc",
        required=True,
        help="Location of the peaks distribution file.",
    )
    parser.add_argument(
        "--save_loc", required=True, help="Location to save the results."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
