import gzip
import re
import numpy as np
import pandas as pd
import pyfaidx

from typing import Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from varscore.core.logging import get_logger
# NOTE: chrombpnet_utils pulls in the TensorFlow stack, so it is imported lazily
# inside the (few) functions that need one-hot encoding. This keeps lightweight
# entrypoints like load_variants() usable without TensorFlow installed.

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
    chro: Optional[str] = None
    ref_seq: Optional[str] = None
    alt_seq: Optional[str] = None
    error_reason: Optional[ValidationErrorReason] = None
    error_message: Optional[str] = None


def get_peak_seqs(
    peaks_df: pd.DataFrame, genome_loc: str, width: int = 2114
) -> np.ndarray:
    """Get one-hot encoded peak sequences from peaks."""
    import varscore.scoring.chrombpnet.model as chrombpnet_utils
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
        if not str(chro).startswith("chr"):
            chro = "chr" + str(chro)
        if chro not in genome.keys() or not re.match(r'^chr([1-9]|1[0-9]|2[0-2]|X|Y)$', chro):
            return ValidationResult(
                is_valid=False,
                chro=chro,
                error_reason=ValidationErrorReason.INVALID_CHROMOSOME,
                error_message=f"Chromosome {chro} not found in genome"
            )
            
        if not set(ref).issubset({'A', 'C', 'G', 'T'}):
            return ValidationResult(
                is_valid=False,
                chro=chro,
                error_reason=ValidationErrorReason.INVALID_REFERENCE,
                error_message=f"Reference {ref} is not a valid DNA sequence"
            )
            
        if not set(alt).issubset({'A', 'C', 'G', 'T'}):
            return ValidationResult(
                is_valid=False,
                chro=chro,
                error_reason=ValidationErrorReason.INVALID_ALTERNATE,
                error_message=f"Alternate {alt} is not a valid DNA sequence"
            )

        ref_seq = str(genome[chro][pos - width // 2 : pos + width // 2])
        
        if ref_seq[width // 2 : width // 2 + len(ref)] != ref:
            return ValidationResult(
                is_valid=False,
                chro=chro,
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
                chro=chro,
                error_reason=ValidationErrorReason.ALT_LENGTH_MISMATCH,
                error_message=f"Expected length {width}, got {len(alt_seq)}"
            )
        
        if alt_seq[width // 2 : width // 2 + len(alt)] != alt:
            return ValidationResult(
                is_valid=False,
                chro=chro,
                error_reason=ValidationErrorReason.ALT_MISMATCH,
                error_message=f"Expected {alt}, got {alt_seq[width // 2 : width // 2 + len(alt)]}"
            )
        
        return ValidationResult(
            is_valid=True,
            chro=chro,
            ref_seq=ref_seq,
            alt_seq=alt_seq
        )
        
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            chro=chro,
            error_reason=ValidationErrorReason.GENOME_ACCESS_ERROR,
            error_message=str(e)
        )


def get_variant_seqs_with_genome(
    variants_df: pd.DataFrame, genome: pyfaidx.Fasta, width: int = 2114
) -> Tuple[np.ndarray, np.ndarray]:
    """Get one-hot encoded ref/alt sequences from variants using pre-opened genome.
    
    This function raises AssertionError for invalid variants (legacy behavior).
    """
    import varscore.scoring.chrombpnet.model as chrombpnet_utils
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
            variant_row = row.copy()
            variant_row['chr'] = result.chro
            valid_variants.append(variant_row)
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
    """Load a variants DataFrame from a headerless canonical TSV."""
    variants_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    return variants_df


# Extensions that indicate gzip/bgzip compression.
_GZIP_SUFFIXES = (".gz", ".bgz")
# Extensions that indicate a VCF (after stripping any gzip suffix).
_VCF_SUFFIXES = (".vcf", ".gvcf")


def _is_symbolic_alt(alt: str, ref: str) -> bool:
    """Return True if ``alt`` is not a concrete REF->ALT base substitution.

    Drops the alleles that pepper gVCF reference blocks and structural-variant
    records and that genome-aware validation could not score anyway: the symbolic
    ``<...>`` forms (``<NON_REF>``, ``<*>``, ``<DEL>``, ...), the spanning-deletion
    star / missing / empty allele, breakend notation, and the monomorphic
    ``alt == ref`` no-op some gVCF tools emit. Concrete sequences (including
    indels, and even malformed ones like lowercase/``N``) are kept and left for
    ``validate_variant`` to vet, matching the canonical-TSV input path.
    """
    if alt in {"*", ".", ""}:
        return True
    if alt.startswith("<") and alt.endswith(">"):
        return True
    if "[" in alt or "]" in alt:
        return True
    if alt == ref:
        return True
    return False


def load_variants_vcf(path: str) -> pd.DataFrame:
    """Parse a VCF/gVCF (``.vcf`` or ``.vcf.gz``) into the canonical variant table.

    Pure Python, no extra dependency: stdlib ``gzip`` streams both plain ``.vcf``
    and bgzipped ``.vcf.gz`` (bgzip is a valid gzip stream). Multi-allelic ``ALT``
    is split into one row per concrete allele; symbolic / non-variant alleles
    (see ``_is_symbolic_alt``) are dropped with a logged count so gVCF reference
    blocks don't explode into millions of non-variant rows. ``POS`` is 1-based
    (matching ``pos``); ``ID == "."`` becomes a missing ``variant_id``. Chromosome
    naming and ACGT/genome checks are deferred to ``validate_variant`` downstream.

    Args:
        path: Path to a plain or bgzipped VCF/gVCF file.

    Returns:
        DataFrame with columns ``chr, pos, ref, alt, variant_id`` (one row per
        concrete ALT allele). The columns are always present, even when empty.
    """
    opener = gzip.open if path.endswith(_GZIP_SUFFIXES) else open

    rows = []
    n_sites = 0  # data (non-header) lines seen
    n_symbolic = 0  # symbolic / non-variant ALT alleles dropped

    with opener(path, "rt") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.startswith("#"):  # ## meta + #CHROM header
                continue
            line = line.rstrip("\n")
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 5:
                raise ValueError(
                    f"{path}:{lineno}: malformed VCF data line, expected at least "
                    f"5 tab-separated fields (CHROM POS ID REF ALT), got {len(fields)}"
                )
            chrom, pos, vid, ref, alt = fields[:5]
            n_sites += 1

            variant_id = None if vid == "." else vid
            pos = int(pos)  # VCF POS is 1-based, matching our schema

            for allele in alt.split(","):  # split multi-allelic sites
                if _is_symbolic_alt(allele, ref):
                    n_symbolic += 1
                    continue
                rows.append((chrom, pos, ref, allele, variant_id))

    logger.info(
        "Parsed %d VCF sites -> %d concrete variant rows; "
        "dropped %d symbolic/non-variant ALT alleles",
        n_sites,
        len(rows),
        n_symbolic,
    )

    return pd.DataFrame(rows, columns=VARIANT_SCHEMA)


def read_variants(path: str, fmt: str = "auto") -> pd.DataFrame:
    """Load variants into the canonical table, dispatching on file format.

    This is the format-neutral entry point; ``load_variants`` (TSV) and
    ``load_variants_vcf`` (VCF/gVCF) are the per-format adapters it composes.

    Args:
        path: Input variants file.
        fmt: One of ``"auto"``, ``"tsv"``, ``"vcf"``. ``"auto"`` routes
            ``.vcf``/``.vcf.gz`` (and ``.gvcf``) to the VCF parser and everything
            else to ``load_variants``; ``.bcf`` raises a clear error.

    Returns:
        DataFrame with columns ``chr, pos, ref, alt, variant_id``.
    """
    if fmt == "vcf":
        return load_variants_vcf(path)
    if fmt == "tsv":
        return load_variants(path)
    if fmt != "auto":
        raise ValueError(f"Unknown format {fmt!r}; expected one of auto, tsv, vcf.")

    # auto-detect by extension (strip any gzip suffix first)
    base = path
    if base.endswith(_GZIP_SUFFIXES):
        base = base.rsplit(".", 1)[0]
    if base.endswith(".bcf") or path.endswith(".bcf"):
        raise ValueError(
            f"Binary BCF is not supported ({path}); convert to VCF first, e.g. "
            "`bcftools view in.bcf -Oz -o out.vcf.gz`."
        )
    if base.endswith(_VCF_SUFFIXES):
        return load_variants_vcf(path)
    return load_variants(path)


if __name__ == "__main__":
    genome = pyfaidx.Fasta("/users/riyasinh/projects/chrombpnet-registry/tmpfiles/genomes/hg38/genome.fasta")
    print(genome.keys())