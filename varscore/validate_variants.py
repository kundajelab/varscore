import numpy as np
import pandas as pd
import pyfaidx
import argparse
import os
from typing import Tuple, List, Optional
from datetime import datetime

import varscore.utils.io_utils as io_utils
from varscore.utils.logging_config import get_logger, log_timing, log_progress, setup_logging
import logging

# Set up logger for this module
# Initialize logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging(level="INFO")

logger = get_logger(__name__)


##################
# CORE FUNCTIONS #
##################


def validate_variants(
    variants_loc: str,
    genome_loc: str,
    valid_out_path: Optional[str] = None,
    invalid_out_path: Optional[str] = None,
    width: int = 2114,
    batch_size: int = 250000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate variants and return valid and invalid variant dataframes.

    Args:
        variants_loc: The path of variants tsv.
        genome_loc: The path to the genome fasta.
        valid_out_path: Optional path to save valid variants. If None, only returns dataframes.
        invalid_out_path: Optional path to save invalid variants. If None, only returns dataframes.
        width: Sequence width for validation (default: 2114).
        batch_size: Number of variants to process at once to manage memory.

    Returns:
        Tuple of (valid_variants_df, invalid_variants_df)
    """
    start_time = datetime.now()
    logger.info("Starting variant validation")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"Genome file: {genome_loc}")
    logger.info(f"Sequence width: {width}")
    logger.info(f"Batch size: {batch_size}")
    
    # Load full variant dataframe
    logger.info("Loading variants...")
    variant_df = io_utils.load_variants(variants_loc)
    total_variants = len(variant_df)
    logger.info(f"Loaded {total_variants} variants")
    
    # Create output directories if saving variants
    if not valid_out_path:
        valid_out_path = variants_loc.replace(".tsv", "_valid.tsv")
    if not invalid_out_path:
        invalid_out_path = variants_loc.replace(".tsv", "_invalid.tsv")
    if valid_out_path:
        os.makedirs(os.path.dirname(valid_out_path), exist_ok=True)
        logger.info(f"Valid variants will be saved to: {valid_out_path}")
    if invalid_out_path:
        os.makedirs(os.path.dirname(invalid_out_path), exist_ok=True)
        logger.info(f"Invalid variants will be saved to: {invalid_out_path}")
    
    # Open genome file once for all batches
    logger.info("Opening genome file...")
    genome = pyfaidx.Fasta(genome_loc)
    
    # Lists to collect valid and invalid variants
    valid_variants = []
    invalid_variants = []
    
    # Process in batches
    logger.info(f"Processing variants in batches of {batch_size}...")
    for start_idx in range(0, total_variants, batch_size):
        end_idx = min(start_idx + batch_size, total_variants)
        batch_df = variant_df.iloc[start_idx:end_idx].copy()
        
        # Validate this batch
        batch_valid, batch_invalid = io_utils.validate_variants(batch_df, genome, width)
        logger.info(f"Found {len(batch_valid)} valid and {len(batch_invalid)} invalid variants in batch {start_idx} to {end_idx}")
        
        valid_variants.append(batch_valid)
        invalid_variants.append(batch_invalid)
        
        # Log progress
        log_progress(logger, end_idx, total_variants, "Validation")
    
    # Close genome file
    genome.close()
    logger.info("Genome file closed")
    
    # Combine all batches
    logger.info("Combining batch results...")
    valid_df = pd.concat(valid_variants, ignore_index=True) if valid_variants else pd.DataFrame()
    invalid_df = pd.concat(invalid_variants, ignore_index=True) if invalid_variants else pd.DataFrame()
    
    # Save valid variants if requested
    if valid_out_path and not valid_df.empty:
        logger.info(f"Saving {len(valid_df)} valid variants...")
        # Select only 4 columns: chr, pos, ref, alt (no variant_id)
        valid_df[["chr", "pos", "ref", "alt"]].to_csv(valid_out_path, sep="\t", index=False, header=False)
        logger.info(f"Valid variants saved to {valid_out_path}")
    elif valid_out_path and valid_df.empty:
        logger.warning("No valid variants to save")
    
    # Save invalid variants if requested
    if invalid_out_path and not invalid_df.empty:
        logger.info(f"Saving {len(invalid_df)} invalid variants...")
        invalid_df.to_csv(invalid_out_path, sep="\t", index=False)
        logger.info(f"Invalid variants saved to {invalid_out_path}")
    elif invalid_out_path and invalid_df.empty:
        logger.info("No invalid variants to save")
    
    # Log summary
    logger.info(f"Validation summary: {len(valid_df)} valid, {len(invalid_df)} invalid variants")
    
    # Log timing
    log_timing(logger, "Variant validation", start_time)
    
    return valid_df, invalid_df

########
# MAIN #
########


def main():
    """Main function for CLI interface."""
    args = _parse_args()
    
    valid_df, invalid_df = validate_variants(
        args.variants_loc,
        args.genome_loc,
        args.valid_out_path,
        args.invalid_out_path,
        args.width,
        args.batch_size,
    )
    
    # Print summary
    print(f"\nValidation Summary:")
    print(f"Valid variants: {len(valid_df)}")
    print(f"Invalid variants: {len(invalid_df)}")
    
    if not valid_df.empty and args.valid_out_path:
        print(f"Valid variants saved to: {args.valid_out_path}")
    if not invalid_df.empty and args.invalid_out_path:
        print(f"Invalid variants saved to: {args.invalid_out_path}")


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate variants and identify invalid ones that cannot be processed."
    )
    parser.add_argument(
        "-v", "--variants_loc", required=True, 
        help="Location of the variants file."
    )
    parser.add_argument(
        "-g", "--genome_loc", required=True, 
        help="Location of the genome file."
    )
    parser.add_argument(
        "-o","--valid_out_path", 
        help="Location to save valid variants (optional)."
    )
    parser.add_argument(
        "-i", "--invalid_out_path", 
        help="Location to save invalid variants (optional)."
    )
    parser.add_argument(
        "-w", "--width", type=int, default=2114,
        help="Sequence width for validation (default: 2114)."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=250000,
        help="Number of variants to process at once (default: 250000)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
