import numpy as np
import pandas as pd
import pyfaidx
from scipy.stats import binom
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.models import load_model

import chrombpnet_utils


################
# ENTRY POINTS #
################
def ingest_model(model_info_dict: dict[str, str], peaks_dist_save_dir: str) -> bool:
    """Initial processing of 5 folds of a model.

    Attempts to ingest each of 5 folds. If all 5 folds ingest successfully,
      will save their peak distributions. Returns a boolean success flag.

    Args:
        model_info_dir: A dictionary that has all information about a model. In
          particular, it has information about each fold's location, the model
          name, the peak location, and the genome location.
        peaks_dist_save_dir: The directory in which peak distributions are
          saved.
    """
    # Verify that model_info_dict has the correct keys
    expected_keys = ["model_name", "peaks_loc", "genome_loc"] + [
        f"fold_{i}_loc" for i in range(5)
    ]
    for k in expected_keys:
        if key not in model_info_dict:
            raise KeyError(f"model_info_dict is missing key {k}")
    # Ingest each fold
    folds_ingested_successfully = True
    peak_dists = []
    for i in range(5):
        peak_dist_fold_i = ingest_fold(
            model_info_dict[f"fold_{i}_loc"],
            model_info_dict["peaks_loc"],
            model_info_dict["genome_loc"],
        )
        if peak_dist_fold_i is None:
            folds_ingested_successfully = False
            break
        else:
            peak_dists.append(peak_dist_fold_i)
    # Save peak distributions if folds ingested successfully
    if not folds_ingested_successfully:
        return True
    for i in range(5):
        np.save(
            f"{peaks_dist_save_dir}/{model_info_dict['model_name']}_fold_{i}.npy",
            peak_dists[i],
        )
    return True


def score_variants(
    model_loc: str, variants_loc: str, genome_loc: str, peaks_dist_loc: str
) -> dict[str, np.ndarray]:
    """Score variants through a model.

    Generates one-hot encodings for all the variants and scores them through a
      model. Returns scores as dictionaries from strings to np.ndarray. Each key
      is a type of score and each value is the score for each variant in order.

    Args:
        model_loc: The path of a model.
        variants_loc: The path of variants tsv.
        genome_loc: The path to the model's genome fasta.
        peaks_dist_loc: The path to the model's peak distribution.
    """
    # Load variants
    variants_df = load_variants(variants_loc)
    # Get reference and alternate sequences
    with pyfaidx.Fasta(genome_loc) as genome:
        ref_seqs, alt_seqs = get_variant_seqs(variants_df, genome)
    # Load model
    model = load_chrombpnet(model_loc)
    # Make predictions
    ref_pred_logits, ref_pred_logcts = predict(model, ref_seqs)
    alt_pred_logits, alt_pred_logcts = predict(model, alt_seqs)
    # Compute scores
    peaks_dist = np.load(peaks_dist_loc)
    return scores_from_preds(ref_pred_logcts, alt_pred_logcts, peaks_dist)


####################
# HELPER FUNCTIONS #
####################
def ingest_fold(model_loc: str, peaks_loc: str, genome_loc: str) -> np.ndarray | None:
    """Initial processing of a single fold of a model.

    Checks that the model is valid. Then, produces the peak distribution.
      Returns None if the model is not valid.

    Args:
        model_loc: The path to the model weights.
        peaks_loc: The path to the model peaks.
        genome_loc: The path to the genome fasta.
    """
    # Load model
    model = load_chrombpnet(model_loc)
    # Check model validity
    if not model_is_valid(model):
        return None
    # TODO: Hash model
    # Get peak distribution
    peaks_dist = get_peaks_distribution(model, peaks_loc, genome_loc)
    return peaks_dist


def load_chrombpnet(model_loc: str) -> tf.keras.Model:
    """Loads a ChromBPNet model."""
    custom_objects = {"multinomial_nll": chrombpnet_utils.multinomial_nll, "tf": tf}
    get_custom_objects().update(custom_objects)
    model = load_model(model_hdf5, compile=False)
    return model


def model_is_valid(model: tf.keras.Model) -> bool:
    """Checks that a ChromBPNet model is valid."""
    # TODO: IMPLEMENT (Not sure what to do for this, maybe check input size?)
    return True


def get_peaks_distribution(
    model: tf.keras.Model, peaks_loc: str, genome_loc: str
) -> np.ndarray:
    """Computes a 1000-dimensional distribution of peak logcounts from a model."""
    peaks_df = load_peaks(peaks_loc)
    N = len(peaks_df)
    if N < 1000:
        raise ValueError("The number of peaks must be greater than 1000.")
    peak_seqs = get_peak_seqs(peaks_df, genome_loc)
    _, peak_logcts = predict(model, peak_seqs)
    sorted_peak_logcts = np.sort(peak_logcts)
    peak_distribution = [sorted_peak_logcts[int(i * N / 1000)] for i in range(1000)]
    return peak_distribution


def load_peaks(peaks_loc: str) -> pd.DataFrame:
    """Load a peaks DataFrame, add window start/stop columns."""
    NARROWPEAK_SCHEMA = ["chro", "start", "end", "4", "5", "6", "7", "8", "9", "summit"]
    flank_size = 2114 // 2
    peaks_df = pd.read_csv(peaks_loc, sep="\t", names=NARROWPEAK_SCHEMA)
    peaks_df["summit_pos"] = peaks_df["start"] + peaks_df["summit"]
    peaks_df["window_start"] = peaks_df["summit"] - flank_size
    peaks_df["window_end"] = peaks_df["summit"] + flank_size
    return peaks_df


def get_peak_seqs(peaks_df: pd.DataFrame, genome_loc: str, width=2114) -> np.ndarray:
    """Get one-hot encoded peak sequences from a DataFrame of peaks."""
    sequences = []
    for _, row in regions.iterrows():
        chro, window_start, window_end = (
            row["chro"],
            row["window_start"],
            row["window_end"],
        )
        seq = str(genome[chro][window_start:window_stop])
        assert len(seq) == width
        sequences.append(seqs)
    onehot = chrombpnet_utils.dna_to_one_hot(sequences)
    return onehot


def load_variants(variants_loc: str) -> pd.DataFrame:
    """Load a variants DataFrame."""
    VARIANT_SCHEMA = ["chr", "pos", "ref", "alt"]
    variants_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    return variants_df


def get_variant_seqs(variants_df, genome, width=2114):
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


def predict(
    model: tf.keras.Model, seqs: np.ndarray, batch_size: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Make predictions on sequences."""
    pred_logits_batches, pred_logcts_batches = [], []
    for i in range(0, seqs.shape[0], batch_size):
        seq_batch = seqs[i : i + batch_size]
        pred_logits_i, pred_logcts_i = model.predict_on_batch(seq_batch)
        pred_logits_batches.append(pred_logits_i)
        pred_logcts_batches.append(pred_logcts_i)
    pred_logits = np.vstack(pred_logits_batches)
    pred_logcts = np.vstack(pred_logcts_batches)
    return pred_logits, pred_logcts


def scores_from_preds(
    ref_logcts: np.ndarray, alt_logcts: np.ndarray, peaks_dist: np.ndarray
) -> dict[str, np.ndarray]:
    """Compute scores based on logcount predictions."""
    scores_dict = dict()
    # Log2 Fold Change
    scores_dict["lfc"] = (alt_logcts - ref_logcts) / np.exp(2)
    # LFC p-Value
    scores_dict["lfc-pval"] = compute_lfc_pval(ref_logcts, alt_logcts)
    # Active allele percentile
    ref_quantiles = np.searchsorted(peaks_dist, ref_logcts) / len(ref_quantiles)
    alt_quantiles = np.searchsorted(peaks_dist, alt_logcts) / len(alt_quantiles)
    scores_dict["active-allele-quantile"] = np.maximum(ref_quantiles, alt_quantiles)
    # Return
    return scores_dict


def compute_lfc_pval(ref_logcts, alt_logcts):
    """Computes LFC p-values."""
    min_cts = np.exp(np.minimum(ref_logcts, alt_logcts))
    total_cts = np.exp(ref_logcts) + np.exp(alt_logcts)
    return [binom.cdf(min_cts[i], total_cts[i], 0.5) for i in range(len(ref_logcts))]
