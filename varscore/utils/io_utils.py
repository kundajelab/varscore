import re
import numpy as np
import pandas as pd
import pyfaidx

from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import varscore.utils.chrombpnet_utils as chrombpnet_utils
from varscore.utils.logging_config import get_logger

# Set up logger for this module
logger = get_logger(__name__)


class ValidationErrorReason(Enum):
    """Reasons why a variant validation might fail."""
    REF_MISMATCH = "reference_mismatch"
    ALT_LENGTH_MISMATCH = "alt_length_mismatch"
    ALT_MISMATCH = "alt_mismatch"
    GENOME_ACCESS_ERROR = "genome_access_error"
    INVALID_CHROMOSOME = "invalid_chromosome"
    INVALID_REFERENCE = "invalid_reference"
    INVALID_ALTERNATE = "invalid_alternate"


@dataclass
class ValidationResult:
    """Result of validating a single variant."""
    is_valid: bool
    ref_seq: Optional[str] = None
    alt_seq: Optional[str] = None
    error_reason: Optional[ValidationErrorReason] = None
    error_message: Optional[str] = None


def get_peak_seqs(
    peaks_df: pd.DataFrame, genome_loc: str, width: int = 2114
) -> np.ndarray:
    """Get one-hot encoded peak sequences from peaks."""
    # Load sequences
    sequences = []
    with pyfaidx.Fasta(genome_loc) as genome:
        for _, row in peaks_df.iterrows():
            chro, window_start, window_end = (
                row["chr"],
                row["window_start"],
                row["window_end"],
            )
            seq = str(genome[chro][window_start:window_end])
            if len(seq) == width:
                sequences.append(seq)
    # Convert to one-hot encoding
    onehot = chrombpnet_utils.dna_to_one_hot(sequences)
    return onehot


NARROWPEAK_SCHEMA = ["chr", "start", "end", "4", "5", "6", "7", "8", "9", "summit_offset"]


def load_peaks(peaks_loc: str, width: int = 2114) -> pd.DataFrame:
    """Load a peaks DataFrame, add window start/stop columns."""
    peaks_df = pd.read_csv(peaks_loc, sep="\t", names=NARROWPEAK_SCHEMA)
    # if summit column is not present in shorter bed files, calculate it as half the length of the peak
    if pd.isna(peaks_df["summit_offset"]).all():
        peaks_df["summit_offset"] = (peaks_df["end"] - peaks_df["start"]) // 2
    flank_size = width // 2
    peaks_df["summit_pos"] = peaks_df["start"] + peaks_df["summit_offset"]
    peaks_df["window_start"] = peaks_df["summit_pos"] - flank_size
    peaks_df["window_end"] = peaks_df["summit_pos"] + flank_size
    return peaks_df


def get_variant_seqs(
    variants_df: pd.DataFrame, genome_loc: str, width: int = 2114
) -> Tuple[np.ndarray, np.ndarray]:
    """Get one-hot encoded ref/alt sequences from variants."""
    with pyfaidx.Fasta(genome_loc) as genome:
        return get_variant_seqs_with_genome(variants_df, genome, width)


def validate_variant(
    chro: str, pos: int, ref: str, alt: str, genome: pyfaidx.Fasta, width: int = 2114
) -> ValidationResult:
    """Validate a single variant.
    
    Args:
        chro: Chromosome
        pos: Position (0-based)
        ref: Reference allele
        alt: Alternate allele
        genome: Open genome file handle
        width: Sequence width
        
    Returns:
        ValidationResult with validation status and error details if invalid
    """
    try:
        
        if chro not in genome.keys() or not re.match(r'^chr([1-9]|1[0-9]|2[0-2]|X|Y)$', chro):
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.INVALID_CHROMOSOME,
                error_message=f"Chromosome {chro} not found in genome"
            )
            
        if not set(ref).issubset({'A', 'C', 'G', 'T'}):
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.INVALID_REFERENCE,
                error_message=f"Reference {ref} is not a valid DNA sequence"
            )
            
        if not set(alt).issubset({'A', 'C', 'G', 'T'}):
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.INVALID_ALTERNATE,
                error_message=f"Alternate {alt} is not a valid DNA sequence"
            )

        ref_seq = str(genome[chro][pos - width // 2 : pos + width // 2])
        
        if ref_seq[width // 2 : width // 2 + len(ref)] != ref:
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.REF_MISMATCH,
                error_message=f"Expected {ref}, got {ref_seq[width // 2 : width // 2 + len(ref)]}"
            )
        
        alt_seq = (
            ref_seq[: width // 2]
            + alt
            + str(genome[chro][pos + len(ref) : pos + width // 2 + len(ref) - len(alt)])
        )
        
        if len(alt_seq) != width:
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.ALT_LENGTH_MISMATCH,
                error_message=f"Expected length {width}, got {len(alt_seq)}"
            )
        
        if alt_seq[width // 2 : width // 2 + len(alt)] != alt:
            return ValidationResult(
                is_valid=False,
                error_reason=ValidationErrorReason.ALT_MISMATCH,
                error_message=f"Expected {alt}, got {alt_seq[width // 2 : width // 2 + len(alt)]}"
            )
        
        return ValidationResult(
            is_valid=True,
            ref_seq=ref_seq,
            alt_seq=alt_seq
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            error_reason=ValidationErrorReason.GENOME_ACCESS_ERROR,
            error_message=str(e)
        )


def get_variant_seqs_with_genome(
    variants_df: pd.DataFrame, genome: pyfaidx.Fasta, width: int = 2114
) -> Tuple[np.ndarray, np.ndarray]:
    """Get one-hot encoded ref/alt sequences from variants using pre-opened genome.
    
    This function raises AssertionError for invalid variants (legacy behavior).
    """
    ref_sequences = []
    alt_sequences = []
    
    for _, row in variants_df.iterrows():
        chro, pos, ref, alt = (
            row["chr"],
            int(row["pos"]) - 1,
            row["ref"],
            row["alt"],
        )
        
        result = validate_variant(chro, pos, ref, alt, genome, width)
        
        if not result.is_valid:
            full_error = f"Failed to get sequence for variant {chro}:{pos}:{ref}:{alt} : {result.error_reason.value} - {result.error_message}"
            logger.error(full_error)
            raise AssertionError(full_error)
        
        ref_sequences.append(result.ref_seq)
        alt_sequences.append(result.alt_seq)
    
    # Convert to one-hot encoding
    ref_onehot = chrombpnet_utils.dna_to_one_hot(ref_sequences)
    alt_onehot = chrombpnet_utils.dna_to_one_hot(alt_sequences)
    return ref_onehot, alt_onehot


def validate_variants(
    variants_df: pd.DataFrame, genome: pyfaidx.Fasta, width: int = 2114
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Validate variants and return valid and invalid dataframes.
    
    Args:
        variants_df: DataFrame with variant information
        genome: Open genome file handle  
        width: Sequence width
        
    Returns:
        Tuple of (valid_variants_df, invalid_variants_df)
        Invalid variants have 'error_reason' and 'error_message' columns added.
    """
    valid_variants = []
    invalid_variants = []
    
    for idx, row in variants_df.iterrows():
        chro, pos, ref, alt = (
            row["chr"],
            int(row["pos"]) - 1,
            row["ref"],
            row["alt"],
        )
        
        result = validate_variant(chro, pos, ref, alt, genome, width)
        
        if result.is_valid:
            valid_variants.append(row)
        else:
            variant_row = row.copy()
            variant_row['error_reason'] = result.error_reason.value
            variant_row['error_message'] = result.error_message
            invalid_variants.append(variant_row)
    
    valid_df = pd.DataFrame(valid_variants) if valid_variants else pd.DataFrame()
    invalid_df = pd.DataFrame(invalid_variants) if invalid_variants else pd.DataFrame()
    
    return valid_df, invalid_df


VARIANT_SCHEMA = ["chr", "pos", "ref", "alt", "variant_id"]


def load_variants(variants_loc: str) -> pd.DataFrame:
    """Load a variants DataFrame."""
    variants_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    return variants_df


if __name__ == "__main__":
    genome = pyfaidx.Fasta("/users/riyasinh/projects/chrombpnet-registry/tmpfiles/genomes/hg38/genome.fasta")
    print(genome.keys())