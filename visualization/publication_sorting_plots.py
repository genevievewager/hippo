"""Publication multi-panel sorting / capture figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from visualization.constants import FIGURE_DPI
from visualization.load_outputs import SimulationOutputs
from visualization.publication_style import (
    apply_publication_theme,
    panel_label,
    save_pub_figure,
)

apply_publication_theme()

_LEGACY_SORTING_STEMS = (
    "sorted_vs_ground_truth_spike_counts",
    "sorted_vs_ground_truth_population_activity",
    "sorting_loss_by_cell_class",
    "probe_region_geometry",
    "unit_depth_by_cell_class",
    "unit_count_by_region_and_cell_class",
)


def plot_fig_sorting_summary(data: SimulationOutputs, output_dir: Path) -> Path:
    """One page: GT vs sorted spike counts and sorting loss by cell class."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.5, 5.8))
    gs = GridSpec(1, 2, figure=fig, wspace=0.32)

    # A — spike counts
    ax_a = fig.add_subplot(gs[0, 0])
    labels, gt_counts, sorted_counts = [], [], []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        gt_counts.append(int(data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0]))
        sorted_counts.append(
            int(data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0])
        )
        labels.append(ct)
    x = np.arange(len(labels))
    width = 0.38
    ax_a.bar(x - width / 2, gt_counts, width, label="Ground truth", color="steelblue")
    ax_a.bar(x + width / 2, sorted_counts, width, label="Sorted", color="coral")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax_a.set_ylabel("Total spikes")
    # Keep GT/sorted legend clear of bars (top-left; counts peak mid/right).
    ax_a.legend(fontsize=11, frameon=False, loc="upper left")
    ymax = max(max(gt_counts + [0]), max(sorted_counts + [0]))
    ax_a.set_ylim(0, ymax * 1.18)
    panel_label(ax_a, "A")

    # B — sorting loss
    ax_b = fig.add_subplot(gs[0, 1])
    losses = []
    for gt_n, so_n in zip(gt_counts, sorted_counts):
        losses.append(1.0 - so_n / gt_n if gt_n > 0 else np.nan)
    colors = ["coral" if (isinstance(l, float) and l >= 0) else "mediumpurple" for l in losses]
    ax_b.bar(labels, losses, color=colors, edgecolor="white")
    ax_b.axhline(0, color="black", lw=0.8)
    ax_b.set_ylabel("Sorting loss (1 − sorted/GT)")
    ax_b.tick_params(axis="x", rotation=20, labelsize=8)
    ymax = max([l for l in losses if isinstance(l, float) and not np.isnan(l)] + [0.0])
    ymin = min([l for l in losses if isinstance(l, float) and not np.isnan(l)] + [0.0])
    span = max(ymax - ymin, 0.1)
    ax_b.set_ylim(ymin - 0.08 * span, ymax + 0.22 * span)
    for i, loss in enumerate(losses):
        if loss is None or (isinstance(loss, float) and np.isnan(loss)):
            continue
        ax_b.text(i, loss, f"{loss:.2f}", ha="center", va="bottom", fontsize=8)
    panel_label(ax_b, "B")

    path = save_pub_figure(
        fig, output_dir / "fig_sorting_summary.png", dpi=FIGURE_DPI,
        rect=(0.09, 0.12, 0.97, 0.94),
    )
    for stem in _LEGACY_SORTING_STEMS:
        (output_dir / f"{stem}.png").unlink(missing_ok=True)
    return path


def generate_publication_sorting_plots(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> Path:
    """Write compact sorting summary (``rate_bin_size`` kept for API compat)."""
    del rate_bin_size
    return plot_fig_sorting_summary(data, output_dir)
