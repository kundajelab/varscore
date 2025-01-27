import numpy as np
import pandas as pd
import pyfaidx

import src.varscore.chrombpnet_utils as chrombpnet_utils


##################
# CORE FUNCTIONS #
##################
def ingest_model(
    peaks_loc: str,
    genome_loc: str,
    fold_0_loc: str,
    fold_1_loc: str,
    fold_2_loc: str,
    fold_3_loc: str,
    fold_4_loc: str,
    output_dir: str,
) -> bool:
    """Initial processing of 5 folds of a model.

    Attempts to ingest each of 5 folds. If all 5 folds ingest successfully,
      will save their peak distributions. Returns a boolean success flag.

    Args:
        peaks_loc: The path to the peaks file.
        genome_loc: The path to the genomes files.
        fold_0_loc: The path to the fold 0 weights.
        fold_1_loc: The path to the fold 1 weights.
        fold_2_loc: The path to the fold 2 weights.
        fold_3_loc: The path to the fold 3 weights.
        fold_4_loc: The path to the fold 4 weights.
        output_dir: The directory in which peak distributions are
          saved.
    """
    # Ingest each fold
    model_folds = [fold_0_loc, fold_1_loc, fold_2_loc, fold_3_loc, fold_4_loc]
    folds_ingested_successfully = True
    peak_dists = []
    for i in range(5):
        peak_dist_fold_i = _ingest_fold(model_folds[i], peaks_loc, genome_loc)
        if peak_dist_fold_i is None:
            folds_ingested_successfully = False
            break
        else:
            peak_dists.append(peak_dist_fold_i)
    # Save peak distributions if folds ingested successfully
    if not folds_ingested_successfully:
        return False
    for i in range(5):
        np.save(
            f"{output_dir}/fold_{i}.npy",
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
    _, peak_logcts = utils_chrombpnet.predict(model, peak_seqs)
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


########
# MAIN #
########
def main(args):
    # Call the ingest_model function with arguments
    args = _parse_args()
    success = ingest_model(
        args.peaks_loc,
        args.genome_loc,
        args.fold_0_loc,
        args.fold_1_loc,
        args.fold_2_loc,
        args.fold_3_loc,
        args.fold_4_loc,
        args.output_dir,
    )
    if not success:
        raise RuntimeError("Model ingestion failed.")


def _parse_args():
    parser = argparse.ArgumentParser(description="Ingest a model and associated data.")
    parser.add_argument(
        "--peaks_loc", required=True, help="Location of the peaks file."
    )
    parser.add_argument(
        "--genome_loc", required=True, help="Location of the genome file."
    )
    parser.add_argument("--fold_0_loc", required=True, help="Location of fold 0 file.")
    parser.add_argument("--fold_1_loc", required=True, help="Location of fold 1 file.")
    parser.add_argument("--fold_2_loc", required=True, help="Location of fold 2 file.")
    parser.add_argument("--fold_3_loc", required=True, help="Location of fold 3 file.")
    parser.add_argument("--fold_4_loc", required=True, help="Location of fold 4 file.")
    parser.add_argument(
        "--output_dir", required=True, help="Directory to save peaks distribution data."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
