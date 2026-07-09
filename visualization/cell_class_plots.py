"""Sorted vs ground-truth comparisons and anatomy/probe figures.

Comparing sorted spikes to ground-truth spikes visualizes information loss
due to simulated Neuropixels recording degradation and sorting errors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.constants import CELL_CLASS_ORDER, FIGURE_DPI, MAX_LINE_POINTS, REGION_ORDER
from visualization.load_outputs import SimulationOutputs, downsample_series
from visualization.neural_plots import _population_activity


def plot_sorted_vs_ground_truth_spike_counts(data: SimulationOutputs, output_dir: Path) -> None:
    gt_counts, sorted_counts, labels = [], [], []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        gt_counts.append(data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0])
        sorted_counts.append(data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0])
        labels.append(ct)

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, gt_counts, width, label="Ground truth", color="steelblue")
    ax.bar(x + width / 2, sorted_counts, width, label="Sorted", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Total spikes")
    ax.set_title("Sorted vs ground-truth spike counts by cell class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sorted_vs_ground_truth_spike_counts.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_sorted_vs_ground_truth_population_activity(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> None:
    n = len(data.cell_class_order)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ct in zip(axes, data.cell_class_order):
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"].tolist()
        t_gt, pop_gt = _population_activity(data.spikes_gt, uids, data.session_duration_s, bin_size)
        t_s, pop_s = _population_activity(data.spikes_sorted, uids, data.session_duration_s, bin_size)
        t_ds, y_gt = downsample_series(t_gt, pop_gt, MAX_LINE_POINTS)
        _, y_s = downsample_series(t_s, pop_s, MAX_LINE_POINTS)
        ax.plot(t_ds, y_gt, label="Ground truth", linewidth=0.7, color="steelblue")
        ax.plot(t_ds, y_s, label="Sorted", linewidth=0.7, color="coral", alpha=0.8)
        ax.set_ylabel(f"{ct}\n(Hz)")
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Population activity: sorted vs ground truth", y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "sorted_vs_ground_truth_population_activity.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_sorting_loss_by_cell_class(data: SimulationOutputs, output_dir: Path) -> None:
    losses, labels = [], []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        gt_n = data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0]
        sorted_n = data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0]
        loss = 1.0 - sorted_n / gt_n if gt_n > 0 else np.nan
        losses.append(loss)
        labels.append(ct)

    colors = ["coral" if l >= 0 else "mediumpurple" for l in losses]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, losses, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Sorting loss (1 − sorted / ground truth)")
    ax.set_title("Sorting loss by cell class\n(negative = apparent contamination)")
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{loss:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "sorting_loss_by_cell_class.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_probe_region_geometry(data: SimulationOutputs, output_dir: Path) -> None:
    if data.anatomy is None:
        raise FileNotFoundError(
            f"anatomy_regions.csv not found in {data.input_dir}; required for probe geometry plot."
        )
    anatomy = data.anatomy
    region_colors = {"CA1": "#4C72B0", "CA2": "#55A868", "CA3": "#C44E52", "DG": "#8172B2"}

    fig, ax = plt.subplots(figsize=(6, 10))
    for _, row in anatomy.iterrows():
        z0 = float(row["depth_start_um"])
        z1 = float(row["depth_end_um"])
        region = str(row["region"])
        layer = str(row.get("layer", ""))
        color = region_colors.get(region, "gray")
        ax.barh(
            (z0 + z1) / 2, width=1.0, height=z1 - z0,
            color=color, alpha=0.6, edgecolor="black", linewidth=0.5,
        )
        label = f"{region}\n{layer}\nch {row.get('channels', '')}"
        ax.text(0.5, (z0 + z1) / 2, label, ha="center", va="center", fontsize=7)

    ax.set_ylim(anatomy["depth_end_um"].max(), 0)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_ylabel("Probe depth (µm)")
    ax.set_title("Simulated Neuropixels probe geometry through hippocampus")
    fig.tight_layout()
    fig.savefig(output_dir / "probe_region_geometry.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_unit_depth_by_cell_class(data: SimulationOutputs, output_dir: Path) -> None:
    if "depth_um" not in data.units.columns:
        raise ValueError("units.csv has no depth_um or channel column for depth plot.")

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CELL_CLASS_ORDER)))
    for color, ct in zip(colors, data.cell_class_order):
        subset = data.units[data.units["cell_type"] == ct]
        ax.scatter(
            subset["depth_um"], np.random.default_rng(42).normal(0, 0.05, len(subset)),
            s=12, alpha=0.6, label=ct, color=color,
        )
    ax.set_xlabel("Unit depth (µm)")
    ax.set_yticks([])
    ax.set_title("Unit depth by cell class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "unit_depth_by_cell_class.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_unit_count_by_region_and_cell_class(data: SimulationOutputs, output_dir: Path) -> None:
    counts = (
        data.units.groupby(["region", "cell_type"]).size()
        .unstack(fill_value=0)
        .reindex(index=[r for r in REGION_ORDER if r in data.units["region"].unique()],
                 columns=[c for c in CELL_CLASS_ORDER if c in data.units["cell_type"].unique()],
                 fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(counts))
    colors = plt.cm.tab10(np.linspace(0, 1, len(counts.columns)))
    for color, ct in zip(colors, counts.columns):
        ax.bar(counts.index, counts[ct], bottom=bottom, label=ct, color=color, edgecolor="white")
        bottom += counts[ct].to_numpy()

    ax.set_ylabel("Number of units")
    ax.set_xlabel("Region")
    ax.set_title("Unit count by region and cell class")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "unit_count_by_region_and_cell_class.png", dpi=FIGURE_DPI)
    plt.close(fig)


def generate_cell_class_plots(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> None:
    plot_sorted_vs_ground_truth_spike_counts(data, output_dir)
    plot_sorted_vs_ground_truth_population_activity(data, output_dir, rate_bin_size)
    plot_sorting_loss_by_cell_class(data, output_dir)
    plot_probe_region_geometry(data, output_dir)
    plot_unit_depth_by_cell_class(data, output_dir)
    plot_unit_count_by_region_and_cell_class(data, output_dir)
