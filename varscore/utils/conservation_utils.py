

import pysam
from enum import Enum
import os
import pyBigWig

SNV_FILEPATH = "varscore/data/cadd/whole_genome_SNVs.tsv.gz"
INDEL_FILEPATH = "varscore/data/cadd/gnomad.genomes.r4.0.indel.tsv.gz"

PHYLOP_FILEPATH = "varscore/data/phylop100vertebrates.bw"

class MutationType(Enum):
    SNV = "SNV"
    INDEL = "INDEL"
    
MUTATION_TYPE_FILE_MAP = {
    MutationType.SNV: {
        "score_file": SNV_FILEPATH,
        "index_file": SNV_FILEPATH + ".tbi"
    },
    MutationType.INDEL: {
        "score_file": INDEL_FILEPATH,
        "index_file": INDEL_FILEPATH + ".tbi"
    }
}

def get_cadd_score(chrom: str, pos: int, ref: str, alt: str) -> tuple[float, float]:
    """
    Retrieve CADD scores from a bgzipped, tabix-indexed file for a given variant.
    
    Parameters:
        chrom (str): Chromosome (e.g., "1", "chr1").
        pos (int): Position of the variant (1-based).
        ref (str): Reference allele.
        alt (str): Alternate allele.
        filepath (str): Path to the compressed CADD TSV file.
    
    Returns:
        tuple: (raw_score, phred_score) as floats if found, else None.
    """
    mutation_type = MutationType.SNV if len(ref) == 1 and len(alt) == 1 else MutationType.INDEL
    score_file, index_file = MUTATION_TYPE_FILE_MAP[mutation_type]["score_file"], MUTATION_TYPE_FILE_MAP[mutation_type]["index_file"]
    
    if not os.path.exists(score_file) or not os.path.exists(index_file):
        raise FileNotFoundError(f"CADD score file or index file not found. Please see the README for instructions on how to download the file.")
    
    # Standardize chromosome format if needed (here we assume the file uses no "chr" prefix)
    if chrom.startswith("chr"):
        chrom = chrom.replace("chr", "")
    
    with pysam.TabixFile(score_file, index=index_file) as tbx:
        for line in tbx.fetch(chrom, pos-1, pos):
            # Skip header or comment lines.
            if line.startswith("#"):
                continue
            fields = line.strip().split('\t')
            # Columns: Chrom, Pos, Ref, Alt, RawScore, PHRED
            # Make sure to compare the right types.
            record_chrom, record_pos, record_ref, record_alt = fields[0], int(fields[1]), fields[2], fields[3]
            if record_chrom == chrom and record_pos == pos and record_ref == ref and record_alt == alt:
                raw_score = float(fields[4])
                phred_score = float(fields[5])
                return raw_score, phred_score
    
    # Return None if no matching record was found.
    return None
  
    
def phylop_score(chr, start, end) -> list[float]:
    """
    Calculates the phylop score for a given region.
    """
    if not os.path.exists(PHYLOP_FILEPATH):
        raise FileNotFoundError(f"PhyloP Bigwig file not found. Please see the README for instructions on how to download the file.")
    with open(PHYLOP_FILEPATH, "rb") as f:
        phylop = pyBigWig.open(PHYLOP_FILEPATH)
        score = phylop.values(chr, start, end)
    return score

# Example usage:
if __name__ == "__main__":
    result = get_cadd_score("1", 10001, "T", "A")
    print(result)
    
    # print(phylop_score("chr1", 58046520, 58046521))
