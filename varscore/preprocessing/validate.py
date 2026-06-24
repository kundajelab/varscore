import numpy as np
import pandas as pd
import pyfaidx
import argparse
import os
from typing import Tuple, List, Optional
from datetime import datetime

import varscore.core.io as io_utils
from varscore.core.logging import get_logger, log_timing, log_progress, setup_logging
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
    fmt: str = "auto",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate variants and return valid and invalid variant dataframes.

    Args:
        variants_loc: The path of variants tsv.
        genome_loc: The path to the genome fasta.
        valid_out_path: Optional path to save valid variants. If None, only returns dataframes.
        invalid_out_path: Optional path to save invalid variants. If None, only returns dataframes.
        width: Sequence width for validation (default: 2114).
        batch_size: Number of variants to process at once to manage memory.
        fmt: Input format, one of "auto", "tsv", "vcf" (default: auto-detect by
            extension). VCF/gVCF is converted to the canonical table on load.

    Returns:
        Tuple of (valid_variants_df, invalid_variants_df)
    """
    start_time = datetime.now()
    logger.info("Starting variant validation")
    logger.info(f"Variants file: {variants_loc}")
    logger.info(f"Genome file: {genome_loc}")
    logger.info(f"Sequence width: {width}")
    logger.info(f"Batch size: {batch_size}")
    
    # Load full variant dataframe (TSV or VCF/gVCF, dispatched by `fmt`)
    logger.info("Loading variants...")
    variant_df = io_utils.read_variants(variants_loc, fmt)
    total_variants = len(variant_df)
    logger.info(f"Loaded {total_variants} variants")
    
    # Create output directories if saving variants
    if not valid_out_path:
        valid_out_path = variants_loc.replace(".tsv", "_valid.tsv")
    if not invalid_out_path:
        invalid_out_path = variants_loc.replace(".tsv", "_invalid.tsv")
    if valid_out_path:
        valid_dir = os.path.dirname(valid_out_path)
        if valid_dir:
            os.makedirs(valid_dir, exist_ok=True)
        logger.info(f"Valid variants will be saved to: {valid_out_path}")
    if invalid_out_path:
        invalid_dir = os.path.dirname(invalid_out_path)
        if invalid_dir:
            os.makedirs(invalid_dir, exist_ok=True)
        logger.info(f"Invalid variants will be saved to: {invalid_out_path}")
    
    # Track if we've written the first batch (for headers on invalid file)
    first_invalid_batch = True
    
    # Clear output files before starting (since we append incrementally)
    if valid_out_path:
        open(valid_out_path, "w").close()
    if invalid_out_path:
        open(invalid_out_path, "w").close()
    
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
        
        if len(batch_invalid) > 0:
            logger.info(f"Invalid variants preview:")
            logger.info(batch_invalid.head())
        
        valid_variants.append(batch_valid)
        invalid_variants.append(batch_invalid)
        
        # Write valid variants incrementally (append mode, no headers).
        # We write the full canonical schema so a custom variant_id (from a TSV
        # 5th column or a VCF ID) is preserved end-to-end; downstream modules read
        # by column name and carry it through. When no id was supplied the 5th
        # field is simply blank (NaN -> empty). See docs/variant_id.md.
        if valid_out_path and not batch_valid.empty:
            batch_valid[io_utils.VARIANT_SCHEMA].to_csv(
                valid_out_path, sep="\t", index=False, header=False, mode="a"
            )
        
        # Write invalid variants incrementally (headers only on first batch)
        if invalid_out_path and not batch_invalid.empty:
            batch_invalid.to_csv(
                invalid_out_path, sep="\t", index=False, 
                header=first_invalid_batch, mode="a"
            )
            first_invalid_batch = False
        
        # Log progress
        log_progress(logger, end_idx, total_variants, "Validation")
    
    # Close genome file
    genome.close()
    logger.info("Genome file closed")
    
    # Combine all batches
    logger.info("Combining batch results...")
    valid_df = pd.concat(valid_variants, ignore_index=True) if valid_variants else pd.DataFrame()
    invalid_df = pd.concat(invalid_variants, ignore_index=True) if invalid_variants else pd.DataFrame()

    # Log summary
    logger.info(f"Valid variants saved to {valid_out_path}")
    logger.info(f"Invalid variants saved to {invalid_out_path}")
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
        args.format,
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
    parser.add_argument(
        "-f", "--format", default="auto", choices=["auto", "tsv", "vcf"],
        help="Input format (default: auto, detected by extension)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
