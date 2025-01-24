import numpy as np
import pandas as pd
import pyfaidx

import chrombpnet_utils


def ingest_model(model_info_dict: dict[str, str], peaks_dist_save_dir: str) -> bool:
    """Initial processing of 5 folds of a model.

    Attempts to ingest each of 5 folds. If all 5 folds ingest successfully,
      will save their peak distributions. Returns a boolean success flag.

    Args:
        model_info_dir: A dictionary that has all information about a model. In
          particular, it has information about each fold's locaqtion, the model
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
        peak_dist_fold_i = _ingest_fold(
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
    # TODO: Save peak DNATree
    return True


def _ingest_fold(model_loc: str, peaks_loc: str, genome_loc: str) -> np.ndarray | None:
    """Initial processing of a single fold of a model.

    Checks that the model is valid. Then, produces the peak distribution.
      Returns None if the model is not valid.

    Args:
        model_loc: The path to the model weights.
        peaks_loc: The path to the model peaks.
        genome_loc: The path to the genome fasta.
    """
    # Load model
    model = chrombpnet_utils.load_chrombpnet(model_loc)
    # Check model validity
    if not chrombpnet_utils.model_is_valid(model):
        return None
    # TODO: Hash model
    # Get peak distribution
    peaks_dist = _get_peaks_distribution(model, peaks_loc, genome_loc)
    return peaks_dist


def _get_peaks_distribution(
    model: tf.keras.Model, peaks_loc: str, genome_loc: str
) -> np.ndarray:
    """Computes a 1000-dimensional distribution of peak logcounts from a model."""
    peaks_df = _load_peaks(peaks_loc)
    N = len(peaks_df)
    if N < 1000:
        raise ValueError("The number of peaks must be greater than 1000.")
    peak_seqs = _get_peak_seqs(peaks_df, genome_loc)
    _, peak_logcts = predict(model, peak_seqs)
    sorted_peak_logcts = np.sort(peak_logcts)
    peak_distribution = [sorted_peak_logcts[int(i * N / 1000)] for i in range(1000)]
    return peak_distribution


def _load_peaks(peaks_loc: str) -> pd.DataFrame:
    """Load a peaks DataFrame, add window start/stop columns."""
    NARROWPEAK_SCHEMA = ["chro", "start", "end", "4", "5", "6", "7", "8", "9", "summit"]
    flank_size = 2114 // 2
    peaks_df = pd.read_csv(peaks_loc, sep="\t", names=NARROWPEAK_SCHEMA)
    peaks_df["summit_pos"] = peaks_df["start"] + peaks_df["summit"]
    peaks_df["window_start"] = peaks_df["summit"] - flank_size
    peaks_df["window_end"] = peaks_df["summit"] + flank_size
    return peaks_df


def _get_peak_seqs(peaks_df: pd.DataFrame, genome_loc: str, width=2114) -> np.ndarray:
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
