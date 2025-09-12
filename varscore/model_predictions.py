import pandas as pd
import numpy as np
from functools import reduce
import operator
import argparse

from typing import List

import varscore.utils.io_utils as io_utils
import varscore.utils.region_utils as region_utils


##################
# CORE FUNCTIONS #
##################


def model_predictions(
    fold_0_scores_loc: str,
    fold_1_scores_loc: str,
    fold_2_scores_loc: str,
    fold_3_scores_loc: str,
    fold_4_scores_loc: str,
    peaks_dnatree_loc: str,
    output_path: str,
    batch_size: int = 250000,
) -> None:
    """Computes model-level variant scores and if variants are in model peaks.

    Loads and averages per-fold scores into model-level scores. Then, annotates
      if each variant is in a peak for that model or not.

    Args:
        fold_0_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_1_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_2_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_3_scores_loc: The path to the scores generated from fold 0 of the
          model.
        fold_4_scores_loc: The path to the scores generated from fold 0 of the
          model.
        peaks_dnatree_loc: The path to the model's peaks DNATree.
        output_path: The path to save the final model predictions.
        batch_size: Number of rows to process at once to manage memory.
    """
    fold_paths = [
        fold_0_scores_loc,
        fold_1_scores_loc,
        fold_2_scores_loc,
        fold_3_scores_loc,
        fold_4_scores_loc,
    ]
    
    # Load peaks DNATree once
    peaks_dnatree = region_utils.loadDNATree(peaks_dnatree_loc)
    
    # Get column names from first file
    first_df = pd.read_csv(fold_paths[0], sep="\t", nrows=0)
    columns = first_df.columns
    
    # Open all files as iterators
    fold_iterators = [
        pd.read_csv(path, sep="\t", chunksize=batch_size) 
        for path in fold_paths
    ]
    
    first_batch = True
    batch_num = 0
    
    # Process chunks from all files together
    try:
        while True:
            # Read next chunk from each fold
            chunks = []
            for iterator in fold_iterators:
                try:
                    chunk = next(iterator)
                    chunks.append(chunk)
                except StopIteration:
                    # If any iterator is exhausted, we're done
                    return
            
            # All chunks should have the same length
            if len(set(len(chunk) for chunk in chunks)) != 1:
                raise ValueError("Fold files have different numbers of rows")
            
            # Average this batch
            averaged_batch = _average_batch(chunks, columns)
            
            # Compute in_peak for this batch
            averaged_batch["in_peak"] = _compute_in_peaks_batch(
                averaged_batch, peaks_dnatree
            )
            
            # Write batch to file
            mode = 'w' if first_batch else 'a'
            header = first_batch
            averaged_batch.to_csv(
                output_path, sep="\t", index=False, mode=mode, header=header
            )
            first_batch = False
            
            batch_num += 1
            print(f"Processed batch {batch_num} ({len(averaged_batch)} variants)")
            
            # Explicitly free memory before next iteration
            del averaged_batch
            del chunks
            
    finally:
        # Clean up file handles
        for iterator in fold_iterators:
            iterator.close()


def _average_batch(chunks: List[pd.DataFrame], columns: pd.Index) -> pd.DataFrame:
    """Average a batch of chunks from all folds."""
    # Construct average_df from columns
    average_df_columns = []
    
    for col_name in columns:
        df_cols = [chunk[col_name] for chunk in chunks]
        
        # io_utils.VARIANT_SCHEMA columns come from fold 0
        if col_name in io_utils.VARIANT_SCHEMA:
            average_df_columns.append(chunks[0][col_name])
        # Non p-value columns/scores are arithmetically averaged
        elif "pval" not in col_name:
            average_df_columns.append(sum(df_cols) / len(df_cols))
        # p-value columns/scores are geometrically averaged
        else:
            average_df_columns.append(
                reduce(operator.mul, df_cols) ** (1 / len(chunks))
            )
    
    average_df = pd.concat(average_df_columns, axis=1)
    average_df.columns = columns
    return average_df


def _compute_in_peaks_batch(
    batch_df: pd.DataFrame, peaks_dnatree
) -> List[bool]:
    """Compute in_peak for a batch of variants."""
    in_peak = []
    for _, row in batch_df.iterrows():
        chro, start, ref = row["chr"], row["pos"], row["ref"]
        end = start + len(ref) - 1
        in_peak.append(peaks_dnatree.overlap(chro, start, end) is not None)
    return in_peak


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Average fold-level scores to compute model-level variant scores, including whether variants are in model peaks."
    )
    parser.add_argument(
        "-f0",
        "--fold_0_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 0 of the model.",
    )
    parser.add_argument(
        "-f1",
        "--fold_1_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 1 of the model.",
    )
    parser.add_argument(
        "-f2",
        "--fold_2_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 2 of the model.",
    )
    parser.add_argument(
        "-f3",
        "--fold_3_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 3 of the model.",
    )
    parser.add_argument(
        "-f4",
        "--fold_4_scores_loc",
        required=True,
        type=str,
        help="The path to the scores generated from fold 4 of the model.",
    )
    parser.add_argument(
        "-p",
        "--peaks_dnatree_loc",
        required=True,
        type=str,
        help="The path to the model's peaks DNATree.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=True,
        type=str,
        help="The path to save the final model predictions.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=250000,
        help="Number of rows to process at once (default: 250000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    model_predictions(
        args.fold_0_scores_loc,
        args.fold_1_scores_loc,
        args.fold_2_scores_loc,
        args.fold_3_scores_loc,
        args.fold_4_scores_loc,
        args.peaks_dnatree_loc,
        args.output_path,
        args.batch_size,
    )
