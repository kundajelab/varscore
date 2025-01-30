import logomaker
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import base64
import io
import multiprocessing
import os

import varscore.utils.io_utils as io_utils


#################
# CORE FUNCTION #
#################
def plot_variants(
    variants_loc: str, plotting_data_dir: str, save_loc: str, num_cpus: int = 4
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
        title = f"{row['chr']}@{row['pos']}:{ref}->{alt}"
        payloads.append(
            (
                ref_profile,
                alt_profile,
                ref_shap,
                alt_shap,
                ref_length,
                alt_length,
                ref,
                alt,
                300,
                title,
            )
        )
    # Plot
    with multiprocessing.Pool(processes=num_cpus) as p:
        plot_strings = p.starmap(_plot_variant_to_utf8, payloads)
    # Save
    variants_df["plot"] = plot_strings
    variants_df.to_csv(save_loc, sep="\t", index=False)


def _plot_variant_and_save(
    allele1_pred,
    allele2_pred,
    allele1_shap,
    allele2_shap,
    allele1_length,
    allele2_length,
    allele1_label,
    allele2_label,
    window_size,
    title,
    save_loc,
):
    fig = plot_variant(
        allele1_pred,
        allele2_pred,
        allele1_shap,
        allele2_shap,
        allele1_length,
        allele2_length,
        allele1_label,
        allele2_label,
        window_size,
        title,
    )
    plt.savefig(save_loc, bbox_inches="tight")
    plt.close()


def _plot_variant_to_utf8(
    allele1_pred,
    allele2_pred,
    allele1_shap,
    allele2_shap,
    allele1_length,
    allele2_length,
    allele1_label,
    allele2_label,
    window_size,
    title,
):
    fig = plot_variant(
        allele1_pred,
        allele2_pred,
        allele1_shap,
        allele2_shap,
        allele1_length,
        allele2_length,
        allele1_label,
        allele2_label,
        window_size,
        title,
    )
    # Encode image in UTF-8
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    utf8_plot = base64.b64encode(buf.read()).decode("utf-8")
    return utf8_plot


######################
# PLOTTING UTILITIES #
######################
def plot_variant(
    allele1_pred,
    allele2_pred,
    allele1_shap,
    allele2_shap,
    allele1_length,
    allele2_length,
    allele1_label,
    allele2_label,
    window_size,
    title,
):
    fig, axs = plt.subplots(3, 1, figsize=(20, 8), dpi=400)
    # PLOT PROFILE
    _plot_profile(
        allele1_pred,
        allele2_pred,
        allele1_length,
        allele2_length,
        allele1_label,
        allele2_label,
        window_size,
        axs[0],
    )
    # PLOT SHAP
    _plot_shap(
        allele1_shap,
        allele2_shap,
        allele1_length,
        allele2_length,
        allele1_label,
        allele2_label,
        window_size,
        axs[1],
        axs[2],
    )
    # PLOT
    plt.suptitle(title, fontsize=24)
    plt.subplots_adjust(hspace=0.3, top=0.785)
    fig.set_facecolor("white")
    return fig
    plt.savefig(save_loc, bbox_inches="tight")
    plt.close()


def _plot_profile(
    allele1_pred,
    allele2_pred,
    allele1_length,
    allele2_length,
    allele1_label,
    allele2_label,
    window_size,
    ax0,
):
    total_length = 1000
    C = total_length // 2
    F = window_size // 2
    if allele1_length < allele2_length:
        # INSERTION
        allele1_pred_plots = []
        allele1_pred_plots.append(
            (list(range(-F, allele1_length)), allele1_pred[C - F : C + allele1_length])
        )
        allele1_pred_plots.append(
            (
                list(range(allele2_length, F + allele2_length)),
                allele1_pred[C + allele1_length : C + F + allele1_length],
            )
        )
        allele2_pred_plots = [
            (
                list(range(-F, F + allele2_length)),
                allele2_pred[C - F : C + F + allele2_length],
            )
        ]
        vlines = [-0.5, allele2_length - 0.5]
    elif allele1_length > allele2_length:
        # DELETION
        allele1_pred_plots = [
            (
                list(range(-F, F + allele1_length)),
                allele1_pred[C - F : C + F + allele1_length],
            )
        ]
        allele2_pred_plots = []
        allele2_pred_plots.append(
            (list(range(-F, allele2_length)), allele2_pred[C - F : C + allele2_length])
        )
        allele2_pred_plots.append(
            (
                list(range(allele1_length, F + allele1_length)),
                allele2_pred[C + allele2_length : C + F + allele2_length],
            )
        )
        vlines = [-0.5, allele1_length - 0.5]
    else:
        # SUBSTITUTION
        allele1_pred_plots = [
            (
                list(range(-F, F + allele1_length)),
                allele1_pred[C - F : C + F + allele1_length],
            )
        ]
        allele2_pred_plots = [
            (
                list(range(-F, F + allele2_length)),
                allele2_pred[C - F : C + F + allele2_length],
            )
        ]
        vlines = [-0.5, +allele1_length - 0.5]
    _plotter_profile(
        allele1_pred_plots,
        allele2_pred_plots,
        vlines,
        ax0,
        allele1_label,
        allele2_label,
    )


def _plot_shap(
    allele1_shap,
    allele2_shap,
    allele1_length,
    allele2_length,
    allele1_label,
    allele2_label,
    window_size,
    ax1,
    ax2,
):
    total_length = 2114
    C = total_length // 2
    F = window_size // 2
    if allele1_length < allele2_length:
        # INSERTION
        allele1_shap_plot = np.concatenate(
            [
                allele1_shap[C - F : C + allele1_length],
                np.zeros((allele2_length - allele1_length, 4)),
                allele1_shap[C + allele1_length : C + F + allele1_length],
            ]
        )
        allele2_shap_plot = allele2_shap[C - F : C + F + allele2_length]
        vlines = [-0.5, allele2_length - 0.5]
    elif allele1_length > allele2_length:
        # DELETION
        allele1_shap_plot = allele1_shap[C - F : C + F + allele1_length]
        allele2_shap_plot = np.concatenate(
            [
                allele2_shap[C - F : C + allele2_length],
                np.zeros((allele1_length - allele2_length, 4)),
                allele2_shap[C + allele2_length : C + F + allele2_length],
            ]
        )
        vlines = [-0.5, allele1_length - 0.5]
    else:
        # SUBSTITUTION
        allele1_shap_plot = allele1_shap[C - F : C + F + allele1_length]
        allele2_shap_plot = allele2_shap[C - F : C + F + allele2_length]
        vlines = [-0.5, allele1_length - 0.5]
    assert allele1_shap_plot.shape == allele2_shap_plot.shape
    _plotter_shap(
        allele1_shap_plot,
        allele2_shap_plot,
        vlines,
        -F,
        ax1,
        ax2,
        allele1_label,
        allele2_label,
    )


def _plotter_profile(
    allele1_profiles, allele2_profiles, vlines, ax0, allele1_label, allele2_label
):
    xmins, xmaxs = [], []
    for i, (x, allele1_profile) in enumerate(allele1_profiles):
        ax0.plot(
            x,
            allele1_profile,
            label=(f"ref ({allele1_label})" if i == 0 else ""),
            color="C0",
        )
        xmins.append(min(x))
        xmaxs.append(max(x))
    for i, (x, allele2_profile) in enumerate(allele2_profiles):
        ax0.plot(
            x,
            allele2_profile,
            label=(f"alt ({allele2_label})" if i == 0 else ""),
            color="C1",
        )
        xmins.append(min(x))
        xmaxs.append(max(x))
    for v in vlines:
        ax0.axvline(v, color="black", ls="--", linewidth=1)
    xmin, xmax = min(xmins), max(xmaxs)
    ax0.set_xlim(xmin, xmax)
    ax0.set_xticks(np.arange(xmin + (50 - xmin % 50) % 50, xmax + 1, 50))
    ax0.legend(prop={"size": 18}, loc="upper right")


def _plotter_shap(
    allele1_shap, allele2_shap, vlines, xmin, ax1, ax2, allele1_label, allele2_label
):
    active_allele = "ref" if np.sum(allele1_shap) > np.sum(allele2_shap) else "alt"
    df1 = pd.DataFrame(allele1_shap, columns=["A", "C", "G", "T"])
    df1.index += xmin
    df2 = pd.DataFrame(allele2_shap, columns=["A", "C", "G", "T"])
    df2.index += xmin
    logomaker.Logo(df1, ax=ax1)
    logomaker.Logo(df2, ax=ax2)
    for v in vlines:
        ax1.axvline(v, color="k", linestyle="--", linewidth=1)
        ax2.axvline(v, color="k", linestyle="--", linewidth=1)
    ymax = 1.1 * max(
        np.max(np.maximum(allele1_shap, 0)), np.max(np.maximum(allele2_shap, 0))
    )
    ymin = 1.1 * min(
        np.min(np.minimum(allele1_shap, 0)), np.min(np.minimum(allele2_shap, 0))
    )
    ax1.set_ylim(bottom=ymin, top=ymax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    plt.text(
        0.988,
        0.903,
        f"ref ({allele1_label})",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax1.transAxes,
        size=18,
        color="black",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgrey"),
    )
    plt.text(
        0.988,
        0.903,
        f"alt ({allele2_label})",
        verticalalignment="top",
        horizontalalignment="right",
        transform=ax2.transAxes,
        size=18,
        color="black",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgrey"),
    )


if __name__ == "__main__":
    print("hi")
    plot_variants(
        "/users/salil512/varscore_test/test_variants.tsv",
        "/users/salil512/varscore_test/average_interpretations",
        "/users/salil512/varscore_test/tsv_with_imgs.tsv",
        4,
    )
    print("done")
