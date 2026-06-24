from varscore.annotation.regions import DNATree
import pandas as pd

def construct_ccre_dnatree(ccre_bed: str, out_path: str) -> None:
    """Constructs a DNATree from a CCRE bed file and saves it to out_path.

    Args:
        ccre_bed: The path to the CCRE bed file.
        out_path: The path to save the DNATree.
    """
    # Load CCREs
    ccre_df = pd.read_csv(ccre_bed, sep="\t", header=None)
    ccre_df.columns = ["chr", "start", "end", "dhs", "accession", "group"]
    # Construct DNATree
    ccre_dnatree = DNATree()
    for _, row in ccre_df.iterrows():
        ccre_dnatree.add(row["chr"], row["start"], row["end"], [row["accession"], row["group"]])
    # Save
    ccre_dnatree.save(out_path)
    return None
  
if __name__ == "__main__":
    import os
    BED_FILE = "varscore/data/raw/GRCh38-cCREs.bed"
    OUTPUT_FILE = "varscore/data/ccres.dnatree"
    assert os.path.exists(BED_FILE)
    construct_ccre_dnatree(BED_FILE, OUTPUT_FILE)
