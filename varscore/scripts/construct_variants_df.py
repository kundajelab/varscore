import gzip
import pickle
import pandas as pd
import pyarrow.parquet as pq
import os

def flatten_allele_frequencies(lst):
    """
    Given a list of dictionaries (each with keys 'populationName' and 'alleleFrequency'),
    returns a new dictionary mapping each populationName to its alleleFrequency.
    If the list is empty or a particular alleleFrequency is missing, that key won't appear.
    """
    if lst is None or len(lst) == 0:
        return {}
    flattened = {}
    for entry in lst:
        if isinstance(entry, dict):
            pop = entry.get("populationName")
            if pop is not None:
                freq = entry.get("alleleFrequency")
                if freq is not None:
                    freq = round(freq, 3)
                else:
                    freq = 0.0
                flattened[pop] = freq
    return flattened
  
def ot_variant_file_to_af_df(variant_file_path):
    df = pq.read_table(variant_file_path)

    # get column subset
    desired_columns = ["chromosome", "position", "referenceAllele", "alternateAllele", "alleleFrequencies"]
    df_subset = df.select(desired_columns)

    # Convert the pyarrow table to a pandas DataFrame for easier processing.
    df_p = df_subset.to_pandas()

    # Apply the flattening function to the 'alleleFrequencies' column and convert results to separate columns.
    allele_df = df_p["alleleFrequencies"].apply(flatten_allele_frequencies).apply(pd.Series)

    # Drop the original 'alleleFrequencies' column and join with the new flattened columns.
    df_p = pd.concat([df_p.drop(columns=["alleleFrequencies"]), allele_df], axis=1)

    return df_p

if __name__ == "__main__":
    variants_dir = "./varscore/data/raw/variants"
    out_path = "./varscore/data/variants.pkl.gz"
    dfs = []
    for file in os.listdir(variants_dir):
        if file.endswith(".parquet"):
            print(f"Processing {file}...")
            input_path = os.path.join(variants_dir, file)
            df = ot_variant_file_to_af_df(input_path).fillna(0.0)
            dfs.append(df)
    df = pd.concat(dfs)
    # Create multi-index for faster lookup
    df = df.set_index(["chromosome", "position", "referenceAllele", "alternateAllele"]).sort_index()
    # Save to gzipped pickle
    with gzip.open(out_path, "wb") as f:
        pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
