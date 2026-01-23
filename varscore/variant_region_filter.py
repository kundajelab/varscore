import pandas as pd
import argparse
import os
from typing import Tuple, Optional
from datetime import datetime

import varscore.utils.io_utils as io_utils
from varscore.utils.logging_config import get_logger, log_timing, log_progress, setup_logging
import varscore.utils.region_utils as region_utils
import logging

# Set up logger for this module
# Initialize logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)


##################
# CORE FUNCTIONS #
##################


def filter_variants_by_region(
    variants_loc: str,
    coding_out_path: Optional[str] = None,
    noncoding_out_path: Optional[str] = None,
    batch_size: int = 250000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter variants into coding (exonic) and non-coding (promoter + intronic) variants.

    Args:
        variants_loc: The path of variants tsv.
        coding_out_path: Optional path to save coding (exonic) variants. If None, only returns dataframes.
        noncoding_out_path: Optional path to save non-coding (promoter + intronic) variants. If None, only returns dataframes.
        batch_size: Number of variants to process at once to manage memory.

    Returns:
        Tuple of (coding_variants_df, noncoding_variants_df)
    """
    start_time = datetime.now()
    logger.info("Starting variant region filtering")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"Batch size: {batch_size}")
    
    # Load full variant dataframe
    logger.info("Loading variants...")
    variant_df = io_utils.load_variants(variants_loc)
    total_variants = len(variant_df)
    logger.info(f"Loaded {total_variants} variants")
    
    # Create output directories if saving variants
    if not coding_out_path:
        coding_out_path = variants_loc.replace(".tsv", "_coding.tsv")
    if not noncoding_out_path:
        noncoding_out_path = variants_loc.replace(".tsv", "_noncoding.tsv")
    if coding_out_path:
        os.makedirs(os.path.dirname(coding_out_path) if os.path.dirname(coding_out_path) else ".", exist_ok=True)
        logger.info(f"Coding variants will be saved to: {coding_out_path}")
    if noncoding_out_path:
        os.makedirs(os.path.dirname(noncoding_out_path) if os.path.dirname(noncoding_out_path) else ".", exist_ok=True)
        logger.info(f"Non-coding variants will be saved to: {noncoding_out_path}")
    
    # Lists to collect coding and non-coding variants
    coding_variants = []
    noncoding_variants = []
    
    # Process in batches
    logger.info(f"Processing variants in batches of {batch_size}...")
    for start_idx in range(0, total_variants, batch_size):
        end_idx = min(start_idx + batch_size, total_variants)
        batch_df = variant_df.iloc[start_idx:end_idx].copy()
        
        # Filter this batch
        batch_coding, batch_noncoding = _filter_batch_by_region(batch_df)
        logger.info(f"Found {len(batch_coding)} coding and {len(batch_noncoding)} non-coding variants in batch {start_idx} to {end_idx}")
        
        coding_variants.append(batch_coding)
        noncoding_variants.append(batch_noncoding)
        
        # Log progress
        log_progress(logger, end_idx, total_variants, "Region filtering")
    
    # Combine all batches
    logger.info("Combining batch results...")
    coding_df = pd.concat(coding_variants, ignore_index=True) if coding_variants else pd.DataFrame()
    noncoding_df = pd.concat(noncoding_variants, ignore_index=True) if noncoding_variants else pd.DataFrame()
    
    # Save coding variants if requested
    if coding_out_path and not coding_df.empty:
        logger.info(f"Saving {len(coding_df)} coding variants...")
        coding_df.to_csv(coding_out_path, sep="\t", index=False, header=False)
        logger.info(f"Coding variants saved to {coding_out_path}")
    elif coding_out_path and coding_df.empty:
        logger.warning("No coding variants to save")
    
    # Save non-coding variants if requested
    if noncoding_out_path and not noncoding_df.empty:
        logger.info(f"Saving {len(noncoding_df)} non-coding variants...")
        noncoding_df.to_csv(noncoding_out_path, sep="\t", index=False, header=False)
        logger.info(f"Non-coding variants saved to {noncoding_out_path}")
    elif noncoding_out_path and noncoding_df.empty:
        logger.info("No non-coding variants to save")
    
    # Log summary
    logger.info(f"Filtering summary: {len(coding_df)} coding, {len(noncoding_df)} non-coding variants")
    
    # Log timing
    log_timing(logger, "Variant region filtering", start_time)
    
    return coding_df, noncoding_df


def _filter_batch_by_region(batch_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter a batch of variants by region type.
    
    Args:
        batch_df: DataFrame with variant information
        
    Returns:
        Tuple of (coding_variants_df, noncoding_variants_df)
    """
    coding_variants = []
    noncoding_variants = []
    
    for idx, row in batch_df.iterrows():
        chro = row["chr"]
        pos = int(row["pos"])  # 1-based position
        ref = row["ref"]
        
        # Calculate variant region (1-based, inclusive)
        var_start = pos
        var_end = pos + len(ref) - 1
        
        # Determine region type
        region = region_utils.region_type(chro, var_start, var_end)
        
        # Classify as coding (exonic) or non-coding (promoter + intronic)
        if region == "exonic":
            coding_variants.append(row)
        else:
            noncoding_variants.append(row)
        # Note: intergenic variants are excluded (not in coding or non-coding categories)
    
    coding_df = pd.DataFrame(coding_variants) if coding_variants else pd.DataFrame()
    noncoding_df = pd.DataFrame(noncoding_variants) if noncoding_variants else pd.DataFrame()
    
    return coding_df, noncoding_df


########
# MAIN #
########


def main():
    """Main function for CLI interface."""
    args = _parse_args()
    
    coding_df, noncoding_df = filter_variants_by_region(
        args.variants_loc,
        args.coding_out_path,
        args.noncoding_out_path,
        args.batch_size,
    )
    
    # Print summary
    print(f"\nRegion Filtering Summary:")
    print(f"Coding (exonic) variants: {len(coding_df)}")
    print(f"Non-coding (promoter + intronic) variants: {len(noncoding_df)}")
    
    if not coding_df.empty and args.coding_out_path:
        print(f"Coding variants saved to: {args.coding_out_path}")
    if not noncoding_df.empty and args.noncoding_out_path:
        print(f"Non-coding variants saved to: {args.noncoding_out_path}")


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Filter variants into coding (exonic) and non-coding (promoter + intronic) categories."
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True, 
        help="Location of the variants file (TSV with chr, pos, ref, alt columns)."
    )
    parser.add_argument(
        "-c", "--coding_out_path", 
        help="Location to save coding (exonic) variants (optional)."
    )
    parser.add_argument(
        "-n", "--noncoding_out_path", 
        help="Location to save non-coding (promoter + intronic) variants (optional)."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=250000,
        help="Number of variants to process at once (default: 250000)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
