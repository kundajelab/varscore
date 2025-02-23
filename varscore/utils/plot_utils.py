import logomaker
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import Any, Dict, List


######################
# PLOTTING UTILITIES #
######################
def variant_plot(
    allele1_pred: np.ndarray,
    allele2_pred: np.ndarray,
    allele1_shap: np.ndarray,
    allele2_shap: np.ndarray,
    allele1_hits: List[Dict[str, Any]],
    allele2_hits: List[Dict[str, Any]],
    allele1_length: int,
    allele2_length: int,
    allele1_label: str,
    allele2_label: str,
    window_size: int = 1000,
    title: str = None,
    figsize: tuple = (20, 8),
    dpi: int = 400,
):
    fig, axs = plt.subplots(3, 1, figsize=figsize, dpi=dpi)
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
        allele1_hits,
        allele2_hits,
        allele1_length,
        allele2_length,
        allele1_label,
        allele2_label,
        window_size,
        axs[1],
        axs[2],
    )
    # PLOT
    if title is not None:
        plt.suptitle(title, fontsize=24)
    plt.subplots_adjust(hspace=0.3, top=0.785)
    fig.set_facecolor("white")
    return fig


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
    allele1_hits,
    allele2_hits,
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

        allele1_plotted_hits = []
        allele2_plotted_hits = []
        for hit in allele1_hits:
            if hit["start"] >= (C - F) and hit["end"] < (C + F + allele1_length):
                hit_start = (
                    hit["start"]
                    - C
                    + (
                        allele1_length - allele1_length
                        if hit["start"] >= C + allele1_length
                        else 0
                    )
                )  # Forward over allele gap
                hit_end = (
                    hit["end"]
                    - C
                    + (
                        allele1_length - allele1_length
                        if hit["end"] >= C + allele1_length
                        else 0
                    )
                )  # Forward over allele gap
                allele1_plotted_hits.append((hit_start, hit_end, hit["motif_name"]))
        for hit in allele2_hits:
            if hit["start"] >= (C - F) and hit["end"] < (C + F + allele2_length):
                allele2_plotted_hits.append(
                    (hit["start"] - C, hit["end"] - C, hit["motif_name"])
                )  # In an insertion, allele2 hits are normal

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

        allele1_plotted_hits = []
        allele2_plotted_hits = []
        for hit in allele1_hits:
            if hit["start"] >= (C - F) and hit["end"] < (C + F + allele1_length):
                print(
                    hit["start"],
                    hit["end"],
                    hit["motif_name"],
                    hit["start"] - C,
                    hit["end"] - C,
                )
                allele1_plotted_hits.append(
                    (hit["start"] - C, hit["end"] - C, hit["motif_name"])
                )  # In a deletion, allele1 hits are normal
        for hit in allele2_hits:
            if hit["start"] >= (C - F) and hit["end"] < (C + F + allele2_length):
                hit_start = (
                    hit["start"]
                    - C
                    + (
                        allele1_length - allele2_length
                        if hit["start"] >= C + allele2_length
                        else 0
                    )
                )  # Forward over allele gap
                hit_end = (
                    hit["end"]
                    - C
                    + (
                        allele1_length - allele2_length
                        if hit["end"] >= C + allele2_length
                        else 0
                    )
                )  # Forward over allele gap
                print(
                    hit["start"],
                    hit["end"],
                    hit_start,
                    hit_end,
                    hit["motif_name"],
                    hit["start"] - C,
                    hit["end"] - C,
                )
                allele2_plotted_hits.append((hit_start, hit_end, hit["motif_name"]))

        vlines = [-0.5, allele1_length - 0.5]
    else:
        # SUBSTITUTION
        allele1_shap_plot = allele1_shap[C - F : C + F + allele1_length]
        allele2_shap_plot = allele2_shap[C - F : C + F + allele2_length]

        allele1_plotted_hits = []
        allele2_plotted_hits = []
        for hit in allele1_hits:
            if hit["start"] >= (C - F) and hit["end"] < (C + F + allele1_length):
                # print(hit["start"], hit["end"], hit["motif_name"], hit["start"] - C, hit["end"] - C)
                allele1_plotted_hits.append(
                    (hit["start"] - C, hit["end"] - C, hit["motif_name"])
                )
        for hit in allele2_hits:
            if hit["start"] >= (C - F) and hit["end"] < (C + F + allele2_length):
                # print(hit["start"], hit["end"], hit["motif_name"], hit["start"] - C, hit["end"] - C)
                allele2_plotted_hits.append(
                    (hit["start"] - C, hit["end"] - C, hit["motif_name"])
                )

        vlines = [-0.5, allele1_length - 0.5]
    assert allele1_shap_plot.shape == allele2_shap_plot.shape
    _plotter_shap(
        allele1_shap_plot,
        allele2_shap_plot,
        allele1_plotted_hits,
        allele2_plotted_hits,
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
        ax0.axvline(v, color="k", ls="--", linewidth=0.25)
    xmin, xmax = min(xmins), max(xmaxs)
    ax0.set_xlim(xmin, xmax)
    ax0.set_xticks(np.arange(xmin + (50 - xmin % 50) % 50, xmax + 1, 50))
    ax0.legend(prop={"size": 18}, loc="upper right")


def _plotter_shap(
    allele1_shap,
    allele2_shap,
    allele1_hits,
    allele2_hits,
    vlines,
    xmin,
    ax1,
    ax2,
    allele1_label,
    allele2_label,
):
    df1 = pd.DataFrame(allele1_shap, columns=["A", "C", "G", "T"])
    df1.index += xmin
    df2 = pd.DataFrame(allele2_shap, columns=["A", "C", "G", "T"])
    df2.index += xmin
    logomaker.Logo(df1, ax=ax1)
    logomaker.Logo(df2, ax=ax2)
    for v in vlines:
        ax1.axvline(v, color="k", linestyle="--", linewidth=0.25)
        ax2.axvline(v, color="k", linestyle="--", linewidth=0.25)
    ymax = 1.1 * max(
        np.max(np.maximum(allele1_shap, 0)), np.max(np.maximum(allele2_shap, 0))
    )
    ymin = 1.1 * min(
        np.min(np.minimum(allele1_shap, 0)), np.min(np.minimum(allele2_shap, 0))
    )
    ax1.set_ylim(bottom=ymin, top=ymax)
    ax2.set_ylim(bottom=ymin, top=ymax)
    for i, hit in enumerate(allele1_hits):
        ax1.add_patch(
            plt.Rectangle(
                (hit[0], ymin), hit[1] - hit[0], ymax - ymin, color="blue", alpha=0.2
            )
        )
        ax1.text(
            hit[0] + (hit[1] - hit[0]) / 2,
            ymax + 0.15 * ymax * (i % 2),  # Alternate height for hits
            hit[2],
            verticalalignment="top",
            horizontalalignment="center",
            size=2,
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgrey"),
        )
    for i, hit in enumerate(allele2_hits):
        ax2.add_patch(
            plt.Rectangle(
                (hit[0], ymin), hit[1] - hit[0], ymax - ymin, color="blue", alpha=0.2
            )
        )
        ax2.text(
            hit[0] + (hit[1] - hit[0]) / 2,
            ymax + 0.15 * ymax * (i % 2),  # Alternate height for hits
            hit[2],
            verticalalignment="top",
            horizontalalignment="center",
            size=2,
            color="black",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="lightgrey"),
        )
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
