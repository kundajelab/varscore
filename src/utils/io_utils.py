import pandas as pd
import pyfaidx

import src.utils.chrombpnet_utils as chrombpnet_utils


def get_peak_seqs(peaks_loc, genome_loc):
    """Get one-hot encoded peak sequences from peaks."""
    # Load peaks DataFrame
    peaks_df = load_peaks(peaks_loc)
    # Load sequences
    sequences = []
    with pyfaidx.Fasta(genome_loc) as genome:
        for _, row in peaks_df.iterrows():
            chro, window_start, window_end = (
                row["chro"],
                row["window_start"],
                row["window_end"],
            )
            seq = str(genome[chro][window_start:window_end])
            assert len(seq) == width
            sequences.append(seq)
    # Convert to one-hot encoding
    onehot = chrombpnet_utils.dna_to_one_hot(sequences)
    return onehot


def load_peaks(peaks_loc: str) -> pd.DataFrame:
    """Load a peaks DataFrame, add window start/stop columns."""
    NARROWPEAK_SCHEMA = ["chro", "start", "end", "4", "5", "6", "7", "8", "9", "summit"]
    flank_size = 2114 // 2
    peaks_df = pd.read_csv(peaks_loc, sep="\t", names=NARROWPEAK_SCHEMA)
    peaks_df["summit_pos"] = peaks_df["start"] + peaks_df["summit"]
    peaks_df["window_start"] = peaks_df["summit"] - flank_size
    peaks_df["window_end"] = peaks_df["summit"] + flank_size
    return peaks_df


def get_variant_seqs(variants_loc, genome_loc, width=2114):
    """Get one-hot encoded ref/alot sequences from variants."""
    # Load variants DataFrame
    variants_df = load_variants(variants_loc)
    # Load sequences
    ref_sequences = []
    alt_sequences = []
    with pyfaidx.Fasta(genome_loc) as genome:
        for _, row in regions.iterrows():
            chro, pos, ref, alt = row["chro"], int(row["pos"]) - 1, row["ref"], row["alt"]
            ref_seq = str(genome[chro][pos - width // 2 : pos + width // 2])
            assert ref_seq[width // 2 : width // 2 + len(ref)] == ref
            alt_seq = (
                ref_seq[: width // 2]
                + alt
                + str(genome[chro][pos + len(ref) : pos + width // 2 + len(ref) - len(alt)])
            )
            assert len(alt_seq) == width
            assert alt_seq[width // 2 : width // 2 + len(alt)] == alt
            ref_sequences.append(ref_seq)
            alt_sequences.append(alt_seq)
    # Convert to one-hot encoding
    ref_onehot = chrombpnet_utils.dna_to_one_hot(ref_sequences)
    alt_onehot = chrombpnet_utils.dna_to_one_hot(alt_sequences)
    return ref_onehot, alt_onehot


def load_variants(variants_loc: str) -> pd.DataFrame:
    """Load a variants DataFrame."""
    VARIANT_SCHEMA = ["chr", "pos", "ref", "alt"]
    variants_df = pd.read_csv(variants_loc, sep="\t", names=VARIANT_SCHEMA)
    return variants_df


