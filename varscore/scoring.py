import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import binom
from scipy.special import softmax

import argparse
import os
import pyfaidx
import logging
from datetime import datetime

import varscore.utils.chrombpnet_utils as chrombpnet_utils
import varscore.utils.io_utils as io_utils
from varscore.utils.logging_config import get_logger, log_timing, log_progress, setup_logging

# Set up logger for this module
# Initialize logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)


##################
# CORE FUNCTIONS #
##################


def score_variants(
    model_loc: str,
    variants_loc: str,
    genome_loc: str,
    peaks_dist_loc: str,
    out_path: str,
    batch_size: int = 250000,
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
        out_path: The path to save the scores dataframe.
        batch_size: Number of variants to process at once to manage memory.
    """
    start_time = datetime.now()
    logger.info("Starting variant scoring")
    logger.info(f"Model: {model_loc}")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"Genome file: {genome_loc}")
    logger.info(f"Peaks distribution: {peaks_dist_loc}")
    logger.info(f"Output path: {out_path}")
    logger.info(f"Batch size: {batch_size}")
    
    # Load model and peak distribution once
    logger.info("Loading model...")
    model = chrombpnet_utils.load_chrombpnet(model_loc)
    logger.info("Model loaded successfully")
    
    logger.info("Loading peak distribution...")
    peaks_dist = np.load(peaks_dist_loc)
    logger.info(f"Peak distribution loaded: {len(peaks_dist)} values")
    
    # Load full variant dataframe
    logger.info("Loading variants...")
    variant_df = io_utils.load_variants(variants_loc)
    total_variants = len(variant_df)
    logger.info(f"Loaded {total_variants} variants")
    
    # Create output directory
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    logger.info(f"Output directory created")
    
    # Open genome file once for all batches
    logger.info("Opening genome file...")
    genome = pyfaidx.Fasta(genome_loc)
    
    # Process and write in batches
    logger.info(f"Processing variants in batches of {batch_size}...")
    first_batch = True
    for start_idx in range(0, total_variants, batch_size):
        batch_start = datetime.now()
        end_idx = min(start_idx + batch_size, total_variants)
        batch_df = variant_df.iloc[start_idx:end_idx].copy()
        
        # Get sequences for this batch (pass genome handle)
        logger.info(f"Batch {start_idx}-{end_idx}: Getting sequences...")
        ref_seqs, alt_seqs = io_utils.get_variant_seqs_with_genome(batch_df, genome)
        
        # Make predictions for this batch
        logger.info(f"Batch {start_idx}-{end_idx}: Making predictions...")
        ref_pred_logits, ref_pred_logcts = chrombpnet_utils.predict(model, ref_seqs)
        alt_pred_logits, alt_pred_logcts = chrombpnet_utils.predict(model, alt_seqs)
        
        # Score this batch
        logger.info(f"Batch {start_idx}-{end_idx}: Computing scores...")
        _score_variant_df(
            batch_df,
            ref_pred_logits,
            alt_pred_logits,
            ref_pred_logcts,
            alt_pred_logcts,
            peaks_dist,
        )
        
        # Write batch to file (append mode after first batch)
        logger.info(f"Batch {start_idx}-{end_idx}: Writing to file...")
        mode = 'w' if first_batch else 'a'
        header = True if first_batch else False
        batch_df.to_csv(out_path, sep="\t", index=False, mode=mode, header=header)
        first_batch = False
        
        # Log progress
        batch_duration = (datetime.now() - batch_start).total_seconds()
        logger.info(f"Batch {start_idx}-{end_idx} completed in {batch_duration:.2f}s")
        log_progress(logger, end_idx, total_variants, "Scoring")
        
        # Explicitly delete large arrays to free memory
        del ref_seqs, alt_seqs, ref_pred_logits, alt_pred_logits
        del ref_pred_logcts, alt_pred_logcts, batch_df
    
    # Close genome file
    genome.close()
    logger.info("Genome file closed")
    logger.info(f"Results saved to {out_path}")
    
    # Log timing
    log_timing(logger, "Variant scoring", start_time)


def _score_variant_df(
    variant_df,
    ref_pred_logits,
    alt_pred_logits,
    ref_pred_logcts,
    alt_pred_logcts,
    peaks_dist,
):
    variant_df["lfc"] = _compute_lfc(ref_pred_logcts, alt_pred_logcts)
    # TODO: test this more before enabling; gives NaNs somewhat often?
    # variant_df["lfc_pval"] = _compute_lfc_pval(ref_pred_logcts, alt_pred_logcts)
    variant_df["jsd"] = _compute_jsd(ref_pred_logits, alt_pred_logits)
    variant_df["active_allele_quantile"] = _compute_active_allele_quantile(
        ref_pred_logcts, alt_pred_logcts, peaks_dist
    )
    variant_df["ips"] = _compute_ips(
        variant_df["lfc"], variant_df["jsd"], variant_df["active_allele_quantile"]
    )
    return variant_df


def _compute_lfc(ref_logcts, alt_logcts):
    """Computes LFC.

    Args:
        ref_logcts: Reference allele logcounts. Shape: (N, 1).
        alt_logcts: Alternate allele logcounts. Shape: (N, 1).
    """
    return (alt_logcts - ref_logcts) / np.log(2)


def _compute_lfc_pval(ref_logcts, alt_logcts):
    """Computes LFC p-values.

    Args:
        ref_logcts: Reference allele logcounts. Shape: (N, 1).
        alt_logcts: Alternate allele logcounts. Shape: (N, 1).
    """
    min_cts = np.exp(np.minimum(ref_logcts, alt_logcts))
    total_cts = np.exp(ref_logcts) + np.exp(alt_logcts)
    return [binom.cdf(min_cts[i], total_cts[i], 0.5) for i in range(len(ref_logcts))]


def _compute_jsd(ref_logits, alt_logits):
    """Computes JSD.

    Args:
        ref_logits: Reference allele logits. Shape: (N, 1000).
        alt_logits: Alternate allele logits. Shape: (N, 1000).
    """
    ref_profile = softmax(ref_logits, axis=1)
    alt_profile = softmax(alt_logits, axis=1)
    return np.squeeze(
        [jensenshannon(x, y, base=2.0) for x, y in zip(ref_profile, alt_profile)]
    )


def _compute_active_allele_quantile(ref_logcts, alt_logcts, peaks_dist):
    """
    Computes the active allele quantile.

    Args:
        ref_logcts: Reference allele logcounts. Shape: (N, 1).
        alt_logcts: Alternate allele logcounts. Shape: (N, 1).
        peaks_dist: Peak distribution. Shape: (1000,).

    Returns:
        active_allele_quantile: Active allele quantile. Shape: (N, 1).
    """
    ref_quantiles = np.searchsorted(peaks_dist, ref_logcts) / len(peaks_dist)
    alt_quantiles = np.searchsorted(peaks_dist, alt_logcts) / len(peaks_dist)
    return np.maximum(ref_quantiles, alt_quantiles)


def _compute_ips(lfc, jsd, active_allele_quantile):
    return lfc * jsd * active_allele_quantile


########
# MAIN #
########
def main():
    # Call the score_variants function with arguments
    args = _parse_args()
    score_variants(
        args.model_loc,
        args.variants_loc,
        args.genome_loc,
        args.peaks_dist_loc,
        args.out_path,
        args.batch_size,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Score variants using a trained model and associated data."
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
        "-p",
        "--peaks_dist_loc",
        required=True,
        help="Location of the peaks distribution file.",
    )
    parser.add_argument(
        "-o", "--out_path", required=True, help="Location to save the results."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=250000,
        help="Number of variants to process at once (default: 250000)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

"""
uv run python -m varscore.scoring -m /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15/chrombpnet_wo_bias.h5 -p /users/riyasinh/projects/chrombpnet-registry/tmpfiles/models/25ff1db5-77ef-4ee0-9b05-f263870ea558/fold_2_peak_distribution.npy -g /users/riyasinh/projects/chrombpnet-registry/tmpfiles/genomes/hg38/genome.fasta -v /users/riyasinh/projects/chrombpnet-registry/tmpfiles/jobs/5af4e0fc-3643-4589-9030-aac7743f1e10/variants/fb3196d9-ca91-4053-a253-58754be8dd12.tsv -o ./results.tsv
"""
