import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import argparse
import base64
import io
import multiprocessing
import os

import varscore.utils.io_utils as io_utils
import varscore.utils.plot_utils as plot_utils


#################
# CORE FUNCTION #
#################
def prepare_variant_plotting(
    variants_loc: str, plotting_data_dir: str, out_path: str
) -> None:
    """Plot variants."""
    variants_df = io_utils.load_variants(variants_loc)
    ref_counts_profile = np.load(
        os.path.join(plotting_data_dir, "average_ref_profiles.npy")
    )
    ref_shap_contributions = np.load(
        os.path.join(plotting_data_dir, "average_ref_shap_contributions.npy")
    )
    ref_shap_sequences = []
    with open(os.path.join(plotting_data_dir, "average_ref_shap_sequences.txt"), "r") as f:
        for line in f:
            ref_shap_sequences.append(line.strip())
    ref_hits = pd.read_csv(
        os.path.join(plotting_data_dir, "ref_hits", "hits.tsv"), sep="\t"
    )
    alt_counts_profile = np.load(
        os.path.join(plotting_data_dir, "average_alt_profiles.npy")
    )
    alt_shap_contributions = np.load(
        os.path.join(plotting_data_dir, "average_alt_shap_contributions.npy")
    )
    alt_shap_sequences = []
    with open(os.path.join(plotting_data_dir, "average_alt_shap_sequences.txt"), "r") as f:
        for line in f:
            alt_shap_sequences.append(line.strip())
    alt_hits = pd.read_csv(
        os.path.join(plotting_data_dir, "alt_hits", "hits.tsv"), sep="\t"
    )
    # Turn arrays to strings
    ref_counts_profile_strs = [np.array2string(row, separator=",") for row in ref_counts_profile]
    ref_shap_contributions_strs = [np.array2string(row, separator=",") for row in ref_shap_contributions]
    alt_counts_profile_strs = [np.array2string(row, separator=",") for row in alt_counts_profile]
    alt_shap_contributions_strs = [np.array2string(row, separator=",") for row in alt_shap_contributions]
    # Track motifs that overlap variants
    ref_motifs = []
    alt_motifs = []
    for index, row in variants_df.iterrows():
        pos = 2114 // 2
        ref = row["ref"]
        alt = row["alt"]
        ref_length = len(ref)
        alt_length = len(alt)
        # title = f"{row['chr']}@{row['pos']}:{ref}->{alt}"
        # Ref motif analysis
        ref_hits_i = []
        ref_motifs_i = []
        for _, row in ref_hits[
            (ref_hits["peak_id"] == index)
            & (ref_hits["hit_coefficient"] >= 10)
        ].iterrows():
            ref_hits_i.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    # Get only relevant part of the motif name
                    # e.g. "pos_1_patterns.ATF3#1_232" -> "ATF3#1"
                    "motif_name": row["motif_name"]
                    .split("patterns.")[1]
                    .rsplit("_", 1)[0],
                }
            )
            if (
                ((pos <= row["start"]) and (row["start"] <= pos + ref_length - 1))
                or ((pos <= row["end"]) and (row["end"] <= pos + ref_length - 1))
                or ((row["start"] <= pos) and (pos + ref_length - 1 <= row["end"]))
            ):
                ref_motifs_i.append(
                    row["motif_name"].split("patterns.")[1].rsplit("_", 1)[0]
                )
        ref_motifs.append(ref_motifs_i)
        # Alt motif analysis
        alt_hits_i = []
        alt_motifs_i = []
        for _, row in alt_hits[
            (alt_hits["peak_id"] == index)
            & (alt_hits["hit_coefficient"] >= 10)
        ].iterrows():
            alt_hits_i.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "motif_name": row["motif_name"]
                    .split("patterns.")[1]
                    .rsplit("_", 1)[0],
                }
            )
            if (
                ((pos <= row["start"]) and (row["start"] <= pos + ref_length - 1))
                or ((pos <= row["end"]) and (row["end"] <= pos + ref_length - 1))
                or ((row["start"] <= pos) and (pos + ref_length - 1 <= row["end"]))
            ):
                alt_motifs_i.append(
                    row["motif_name"].split("patterns.")[1].rsplit("_", 1)[0]
                )
        alt_motifs.append(alt_motifs_i)
    # Save
    variants_df["ref_profile"] = ref_counts_profile_strs
    variants_df["ref_shap_contributions"] = ref_shap_contributions_strs
    variants_df["ref_shap_sequence"] = ref_shap_sequences
    variants_df["ref_motifs"] = [",".join(motifs) for motifs in ref_motifs]
    variants_df["alt_profile"] = alt_counts_profile_strs
    variants_df["alt_shap_contributions"] = alt_shap_contributions_strs
    variants_df["alt_shap_sequence"] = alt_shap_sequences
    variants_df["alt_motifs"] = [",".join(motifs) for motifs in alt_motifs]    
    variants_df.to_csv(out_path, sep="\t", index=False)


def parser():
    parser = argparse.ArgumentParser(description="Prepare variant plotting.")
    parser.add_argument(
        "-v",
        "--variants_loc",
        required=True,
        help="The path to the variants TSV file.",
    )
    parser.add_argument(
        "-p",
        "--plotting_data_dir",
        required=True,
        help="The directory where the average interpretation data is.",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        required=True,
        help="The path to save the TSV with images.",
    )
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    prepare_variant_plotting(
        args.variants_loc, args.plotting_data_dir, args.out_path
    )
    # print("hi")
    # prepare_variant_plotting(
    #     "/users/riyasinh/projects/varscore/plot_dir/variants.csv",
    #     "/users/riyasinh/projects/varscore/plot_dir",
    #     "/users/riyasinh/projects/varscore/plot_dir/variants_with_plots.tsv",
    # )
    # print("done")
