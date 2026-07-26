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

# NOTE: varscore.scoring.chrombpnet.model pulls in the TensorFlow/SHAP stack
# (the `[model]` extra). It is imported lazily inside score_variants() so that
# importing this module — and the pure-numpy scoring math below — stays
# TensorFlow-free.
import varscore.core.io as io_utils
from varscore.core.logging import get_logger, log_timing, log_progress, setup_logging

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
    model_batch_size: int = 256,
    reuse_output: bool = True,
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
        model_batch_size: Batch size for ChromBPNet model predictions (default: 256).
        reuse_output: If True, attempt to reuse existing output and skip already scored variants.
    """
    start_time = datetime.now()
    logger.info("Starting variant scoring")
    logger.info(f"Model: {model_loc}")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"Genome file: {genome_loc}")
    logger.info(f"Peaks distribution: {peaks_dist_loc}")
    logger.info(f"Output path: {out_path}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Model batch size: {model_batch_size}")
    logger.info(f"Reuse output: {reuse_output}")
    
    import varscore.scoring.chrombpnet.model as chrombpnet_utils

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
    
    # Try to reuse existing output if requested
    existing_scores_df = None
    if reuse_output and os.path.exists(out_path):
        try:
            logger.info(f"Output file exists. Attempting to load existing scores from {out_path}")
            existing_scores_df = pd.read_csv(out_path, sep="\t")
            logger.info(f"Loaded {len(existing_scores_df)} existing scored variants")
            
            # Create a variant key for matching (chr:pos:ref:alt)
            variant_df['_variant_key'] = (
                variant_df['chr'].astype(str) + ':' + 
                variant_df['pos'].astype(str) + ':' + 
                variant_df['ref'].astype(str) + ':' + 
                variant_df['alt'].astype(str)
            )
            existing_scores_df['_variant_key'] = (
                existing_scores_df['chr'].astype(str) + ':' + 
                existing_scores_df['pos'].astype(str) + ':' + 
                existing_scores_df['ref'].astype(str) + ':' + 
                existing_scores_df['alt'].astype(str)
            )
            
            # Find variants that haven't been scored yet
            already_scored_keys = set(existing_scores_df['_variant_key'])
            variant_df['_already_scored'] = variant_df['_variant_key'].isin(already_scored_keys)
            variants_to_score = variant_df[~variant_df['_already_scored']].copy()
            variants_to_score = variants_to_score.drop(columns=['_variant_key', '_already_scored'])
            
            num_already_scored = len(variant_df) - len(variants_to_score)
            logger.info(f"Found {num_already_scored} already scored variants")
            logger.info(f"Need to score {len(variants_to_score)} remaining variants")
            
            # Update variant_df to only include variants that need scoring
            variant_df = variants_to_score
            
            # Clean up temporary column from existing scores
            existing_scores_df = existing_scores_df.drop(columns=['_variant_key'])
            
        except Exception as e:
            logger.warning(f"Failed to load existing scores: {e}")
            logger.warning("Will score all variants from scratch")
            existing_scores_df = None
            # Clean up any temporary columns
            if '_variant_key' in variant_df.columns:
                variant_df = variant_df.drop(columns=['_variant_key'])
            if '_already_scored' in variant_df.columns:
                variant_df = variant_df.drop(columns=['_already_scored'])
    
    # Update total_variants to reflect the number we actually need to score
    total_variants = len(variant_df)
    
    if total_variants == 0:
        logger.info("All variants have already been scored! Nothing to do.")
        log_timing(logger, "Variant scoring", start_time)
        return
    
    # Open genome file once for all batches
    logger.info("Opening genome file...")
    genome = pyfaidx.Fasta(genome_loc)
    
    # Process and write in batches
    logger.info(f"Processing variants in batches of {batch_size}...")
    # If we have existing scores, we'll append to the existing file
    # Otherwise, we'll create a new file
    first_batch = existing_scores_df is None
    
    for start_idx in range(0, total_variants, batch_size):
        batch_start = datetime.now()
        end_idx = min(start_idx + batch_size, total_variants)
        batch_df = variant_df.iloc[start_idx:end_idx].copy()
        
        # Get sequences for this batch (pass genome handle)
        logger.info(f"Batch {start_idx}-{end_idx}: Getting sequences...")
        ref_seqs, alt_seqs = io_utils.get_variant_seqs_with_genome(batch_df, genome)
        
        # Make predictions for this batch
        logger.info(f"Batch {start_idx}-{end_idx}: Making predictions...")
        ref_pred_logits, ref_pred_logcts = chrombpnet_utils.predict(model, ref_seqs, batch_size=model_batch_size)
        alt_pred_logits, alt_pred_logcts = chrombpnet_utils.predict(model, alt_seqs, batch_size=model_batch_size)
        
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
        args.model_batch_size,
        args.reuse_output,
    )


def build_parser() -> argparse.ArgumentParser:
    """Return this command's argument parser without consuming ``sys.argv``.

    Split out from ``_parse_args`` so callers that build this command's argv --
    notably the orchestration plugin in ``varscore.lava`` -- can validate what
    they emit against the real flag definitions instead of duplicating them.
    """
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
    parser.add_argument(
        "--model-batch-size", type=int, default=256,
        help="Batch size for ChromBPNet model predictions (default: 256)."
    )
    parser.add_argument(
        "--no-reuse-output", dest="reuse_output", action="store_false",
        help="Disable reusing existing output (default: reuse is enabled)."
    )
    parser.set_defaults(reuse_output=True)
    return parser


def _parse_args():
    return build_parser().parse_args()


if __name__ == "__main__":
    main()

"""
uv run python -m varscore.scoring.chrombpnet.score -m /oak/stanford/groups/akundaje/projects/chromatin-atlas-2022/ATAC/ENCSR474XFV/chrombpnet_model_feb15/chrombpnet_wo_bias.h5 -p /users/riyasinh/projects/chrombpnet-registry/tmpfiles/models/25ff1db5-77ef-4ee0-9b05-f263870ea558/fold_2_peak_distribution.npy -g /users/riyasinh/projects/chrombpnet-registry/tmpfiles/genomes/hg38/genome.fasta -v /users/riyasinh/projects/chrombpnet-registry/tmpfiles/jobs/5af4e0fc-3643-4589-9030-aac7743f1e10/variants/fb3196d9-ca91-4053-a253-58754be8dd12.tsv -o ./results.tsv
"""
