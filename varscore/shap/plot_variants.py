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
def plot_variants(
    variants_loc: str, plotting_data_dir: str, out_path: str, num_cpus: int = 4
) -> None:
    """Plot variants."""
    variants_df = io_utils.load_variants(variants_loc)
    ref_counts_profile = np.load(
        os.path.join(plotting_data_dir, "average_ref_profiles.npy")
    )
    ref_shaps = np.load(os.path.join(plotting_data_dir, "average_ref_shaps.npy"))
    alt_counts_profile = np.load(
        os.path.join(plotting_data_dir, "average_alt_profiles.npy")
    )
    alt_shaps = np.load(os.path.join(plotting_data_dir, "average_alt_shaps.npy"))
    ref_hits = pd.read_csv(os.path.join(plotting_data_dir, "ref_hits.tsv"), sep="\t")
    alt_hits = pd.read_csv(os.path.join(plotting_data_dir, "alt_hits.tsv"), sep="\t")
    # Prepare plotting
    payloads = []
    for index, row in variants_df.iterrows():
        ref_profile = ref_counts_profile[index]
        alt_profile = alt_counts_profile[index]
        ref_shap = ref_shaps[index]
        alt_shap = alt_shaps[index]
        ref = row["ref"]
        alt = row["alt"]
        ref_length = len(ref)
        alt_length = len(alt)
        # title = f"{row['chr']}@{row['pos']}:{ref}->{alt}"
        ref_hits_i = []
        for _, row in ref_hits[ref_hits["peak_id"] == index].iterrows():
            ref_hits_i.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "motif_name": row["motif_name"],
                }
            )
        alt_hits_i = []
        for _, row in alt_hits[alt_hits["peak_id"] == index].iterrows():
            alt_hits_i.append(
                {
                    "start": row["start"],
                    "end": row["end"],
                    "motif_name": row["motif_name"],
                }
            )
        payloads.append(
            (
                ref_profile,
                alt_profile,
                ref_shap,
                alt_shap,
                ref_hits_i,
                alt_hits_i,
                ref_length,
                alt_length,
                ref,
                alt,
                800
            )
        )
    # Plot
    with multiprocessing.Pool(processes=num_cpus) as p:
        plot_strings = p.starmap(_plot_variant_to_utf8, payloads)
    # Save
    variants_df["plot"] = plot_strings
    variants_df.to_csv(out_path, sep="\t", index=False)


def _plot_variant_to_utf8(
    allele1_pred,
    allele2_pred,
    allele1_shap,
    allele2_shap,
    allele1_hits,
    allele2_hits,
    allele1_length,
    allele2_length,
    allele1_label,
    allele2_label,
    window_size
):
    fig = plot_utils.variant_plot(
        allele1_pred,
        allele2_pred,
        allele1_shap,
        allele2_shap,
        allele1_hits,
        allele2_hits,
        allele1_length,
        allele2_length,
        allele1_label,
        allele2_label,
        window_size
    )
    # Encode image in UTF-8
    buf = io.BytesIO()
    fig.savefig(buf, format="svg")
    plt.close(fig)
    buf.seek(0)
    utf8_plot = base64.b64encode(buf.read()).decode("utf-8")
    return utf8_plot


def parser():
    parser = argparse.ArgumentParser(description="Plot variants.")
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
    parser.add_argument(
        "-n",
        "--num_cpus",
        default=4,
        type=int,
        help="The number of CPUs to use.",
    )
    return parser


if __name__ == "__main__":
    args = parser().parse_args()
    plot_variants(args.variants_loc, args.plotting_data_dir, args.out_path, args.num_cpus)
    print("hi")
    plot_variants(
        "/users/salil512/varscore_test/test_variants.tsv",
        "/users/salil512/varscore_test/average_interpretations",
        "/users/salil512/varscore_test/tsv_with_imgs.tsv",
        4,
    )
    print("done")
