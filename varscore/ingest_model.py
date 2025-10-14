from typing import Union
import numpy as np
import argparse
import tensorflow as tf
import os

import varscore.utils.chrombpnet_utils as chrombpnet_utils
import varscore.utils.io_utils as io_utils
import varscore.utils.region_utils as region_utils
from varscore.utils.logging_config import get_logger

# Set up logger for this module
logger = get_logger(__name__)


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
    # Load peaks
    peaks_df = io_utils.load_peaks(peaks_loc)
    peak_seqs = io_utils.get_peak_seqs(peaks_df, genome_loc)
    # Ingest each fold
    model_folds = [fold_0_loc, fold_1_loc, fold_2_loc, fold_3_loc, fold_4_loc]
    folds_ingested_successfully = True
    peak_dists = []
    for i in range(5):
        peak_dist_fold_i = _ingest_fold(model_folds[i], peak_seqs)
        if peak_dist_fold_i is None:
            folds_ingested_successfully = False
            break
        else:
            peak_dists.append(peak_dist_fold_i)
    # Save peak distributions if folds ingested successfully
    if not folds_ingested_successfully:
        return False
    for i in range(5):
        os.makedirs(output_dir, exist_ok=True)
        filename = "fold_{i}_peak_distribution.npy".format(i=i)
        np.save(os.path.join(output_dir, filename), peak_dists[i])
    # TODO: Save peak DNATree
    # Save DNATree
    peaks_dnatree = region_utils.DNATree()
    for _, row in peaks_df.iterrows():
        peaks_dnatree.add(row["chr"], row["window_start"], row["window_end"])
    filename = "peaks.dnatree".format(i=i)
    peaks_dnatree.save(os.path.join(output_dir, filename))
    return True


def _ingest_fold(model_loc: str, peak_seqs: np.ndarray) -> Union[np.ndarray, None]:
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
    peaks_dist = _get_peaks_distribution(model, peak_seqs)
    return peaks_dist


def _get_peaks_distribution(model: tf.keras.Model, peak_seqs: np.ndarray) -> np.ndarray:
    """Computes a 1000-dimensional distribution of peak logcounts from a model."""
    N = peak_seqs.shape[0]
    if N < 1000:
        error_msg = "The number of peaks must be greater than 1000."
        logger.error(error_msg)
        raise ValueError(error_msg)
    # Forward pass on sequences
    _, peak_logcts = chrombpnet_utils.predict(model, peak_seqs)
    # Peak distribuion
    sorted_peak_logcts = np.sort(peak_logcts.squeeze())
    peak_distribution = [sorted_peak_logcts[int(i * N / 1000)] for i in range(1000)]
    return peak_distribution


########
# MAIN #
########
def _parse_args():
    parser = argparse.ArgumentParser(description="Ingest a model and associated data.")
    parser.add_argument(
        "-p", "--peaks_loc", required=True, help="Location of the peaks file."
    )
    parser.add_argument(
        "-g", "--genome_loc", required=True, help="Location of the genome file."
    )
    parser.add_argument(
        "-f0", "--fold_0_loc", required=True, help="Location of fold 0 file."
    )
    parser.add_argument(
        "-f1", "--fold_1_loc", required=True, help="Location of fold 1 file."
    )
    parser.add_argument(
        "-f2", "--fold_2_loc", required=True, help="Location of fold 2 file."
    )
    parser.add_argument(
        "-f3", "--fold_3_loc", required=True, help="Location of fold 3 file."
    )
    parser.add_argument(
        "-f4", "--fold_4_loc", required=True, help="Location of fold 4 file."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        required=True,
        help="Directory to save peaks distribution data.",
    )
    return parser.parse_args()


def main():
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


if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=2 && uv run python -m varscore.ingest_model --peaks_loc /users/riyasinh/data/test.2k.bed --genome_loc /oak/stanford/groups/akundaje/soumyak/refs/hg38/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta --fold_0_loc /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15/chrombpnet_wo_bias.h5 --fold_1_loc /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15_fold_1/chrombpnet_wo_bias.h5 --fold_2_loc /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15_fold_2/chrombpnet_wo_bias.h5 --fold_3_loc /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15_fold_3/chrombpnet_wo_bias.h5 --fold_4_loc /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15_fold_4/chrombpnet_wo_bias.h5 --output_dir ~/varscore_test/
"""
