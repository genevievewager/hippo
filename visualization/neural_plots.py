"""Ground-truth spike, population-rate, and raster visualizations.

Ground-truth Poisson spike trains are generated from RatInABox firing rates
and provide the true neural activity before Neuropixels recording degradation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from visualization.constants import CELL_CLASS_ORDER, FIGURE_DPI, MAX_LINE_POINTS
from visualization.load_outputs import (
    SimulationOutputs,
    downsample_series,
    sort_units_by_class_and_rate,
    sort_units_by_rate_model,
)


def _bin_spike_rates(
    spike_times: np.ndarray,
    t_edges: np.ndarray,
) -> np.ndarray:
    counts, _ = np.histogram(spike_times, bins=t_edges)
    dt = t_edges[1] - t_edges[0]
    rates = counts / dt
    return gaussian_filter1d(rates.astype(float), sigma=1.0)


def _population_activity(
    spikes: pd.DataFrame,
    unit_ids: list[int],
    session_duration: float,
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    t_edges = np.arange(0, session_duration + bin_size, bin_size)
    t_centers = t_edges[:-1] + bin_size / 2
    subset = spikes[spikes["unit_id"].isin(unit_ids)]
    if subset.empty:
        return t_centers, np.zeros(len(t_centers))
    counts, _ = np.histogram(subset["time"], bins=t_edges)
    pop_rate = counts / (bin_size * max(len(unit_ids), 1))
    return t_centers, pop_rate


def plot_population_rates_over_time(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> None:
    t_edges = np.arange(0, data.session_duration_s + bin_size, bin_size)
    t_centers = t_edges[:-1] + bin_size / 2

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CELL_CLASS_ORDER)))

    for color, ct in zip(colors, CELL_CLASS_ORDER):
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"].tolist()
        if not uids:
            continue
        subset = data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)]
        if subset.empty:
            continue
        counts, _ = np.histogram(subset["time"], bins=t_edges)
        pop_rate = counts / (bin_size * len(uids))
        pop_rate = gaussian_filter1d(pop_rate.astype(float), sigma=1.0)
        t_ds, y_ds = downsample_series(t_centers, pop_rate, MAX_LINE_POINTS)
        ax.plot(t_ds, y_ds, label=ct, color=color, linewidth=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population firing rate (Hz)")
    ax.set_title("Population firing rate by cell class (ground truth)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "population_rates_over_time.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_cell_class_rate_distributions(data: SimulationOutputs, output_dir: Path) -> None:
    n = len(data.cell_class_order)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ct in zip(axes, data.cell_class_order):
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        rates = data.unit_mean_rates_gt.reindex(uids).dropna().to_numpy()
        ax.hist(rates, bins=25, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_xlabel("Mean rate (Hz)")
        ax.set_title(ct)
    axes[0].set_ylabel("Unit count")
    fig.suptitle("Mean firing rate distributions by cell class", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "cell_class_rate_distributions.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_example_units(data: SimulationOutputs, output_dir: Path) -> None:
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        rates = data.unit_mean_rates_gt.reindex(uids).dropna()
        if rates.empty:
            continue
        sorted_uids = rates.sort_values()
        picks = {
            "low": int(sorted_uids.index[0]),
            "median": int(sorted_uids.index[len(sorted_uids) // 2]),
            "high": int(sorted_uids.index[-1]),
        }
        fig, axes = plt.subplots(3, 3, figsize=(14, 10))
        for col, (label, uid) in enumerate(picks.items()):
            unit_spikes = data.spikes_gt.loc[data.spikes_gt["unit_id"] == uid, "time"].to_numpy()
            bin_size = 0.250
            t_edges = np.arange(0, data.session_duration_s + bin_size, bin_size)
            rate = _bin_spike_rates(unit_spikes, t_edges)
            t_centers = t_edges[:-1] + bin_size / 2

            ax = axes[0, col]
            ax.plot(data.behavior["x"], data.behavior["y"], color="lightgray", linewidth=0.4)
            if len(unit_spikes):
                spike_idx = np.searchsorted(data.behavior["time"], unit_spikes)
                spike_idx = np.clip(spike_idx, 0, len(data.behavior) - 1)
                ax.scatter(data.behavior["x"].iloc[spike_idx], data.behavior["y"].iloc[spike_idx],
                           s=3, c="red", alpha=0.4)
            ax.set_title(f"{label}-rate (unit {uid})")
            ax.set_aspect("equal", adjustable="box")

            ax = axes[1, col]
            if len(unit_spikes):
                ax.eventplot(unit_spikes, colors="black", linewidths=0.4)
            ax.set_xlim(0, data.session_duration_s)
            ax.set_title("Spike train")

            ax = axes[2, col]
            ax.plot(t_centers, rate, color="darkgreen", linewidth=0.7)
            ax.set_xlabel("Time (s)")
            ax.set_title("Firing rate")

        fig.suptitle(f"Example units — {ct}", y=1.01)
        fig.tight_layout()
        fig.savefig(output_dir / f"example_units_{ct}.png", dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)


def _plot_raster(
    spikes: pd.DataFrame,
    sorted_units: pd.DataFrame,
    title: str,
    output_path: Path,
    class_boundaries: bool = False,
) -> None:
    unit_list = sorted_units["unit_id"].tolist()
    n_units = len(unit_list)
    if n_units == 0:
        return

    spike_groups = spikes.groupby("unit_id")["time"].apply(list).to_dict()
    height = max(4, min(20, n_units * 0.04))

    fig, ax = plt.subplots(figsize=(14, height))
    y_positions = np.arange(n_units)

    for y, uid in enumerate(unit_list):
        times = spike_groups.get(uid, [])
        if times:
            ax.scatter(times, np.full(len(times), y), s=0.3, c="black", marker="|", linewidths=0.2)

    if class_boundaries:
        prev_class = None
        for y, (_, row) in enumerate(sorted_units.iterrows()):
            if prev_class is not None and row["cell_type"] != prev_class:
                ax.axhline(y - 0.5, color="red", linewidth=0.6, alpha=0.5)
            prev_class = row["cell_type"]

        yticks, ylabels = [], []
        for ct in CELL_CLASS_ORDER:
            mask = sorted_units["cell_type"] == ct
            if not mask.any():
                continue
            ys = y_positions[mask.to_numpy()]
            yticks.append(float(np.mean(ys)))
            ylabels.append(ct)
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
    else:
        ax.set_yticks([])

    ax.set_xlim(0, sorted_units.attrs.get("session_duration", spikes["time"].max()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Units")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(fig)


def plot_ground_truth_rasters(data: SimulationOutputs, output_dir: Path) -> None:
    sorted_all = sort_units_by_class_and_rate(data.units, data.unit_mean_rates_gt)
    sorted_all.attrs["session_duration"] = data.session_duration_s

    _plot_raster(
        data.spikes_gt, sorted_all,
        "Ground-truth Poisson spike trains by cell class",
        output_dir / "ground_truth_spike_raster_all_cell_classes.png",
        class_boundaries=True,
    )

    for ct in data.cell_class_order:
        class_units = sorted_all[sorted_all["cell_type"] == ct]
        n = len(class_units)
        _plot_raster(
            data.spikes_gt, class_units,
            f"Ground-truth spike raster — {ct} (n={n} units)",
            output_dir / f"ground_truth_spike_raster_{ct}.png",
            class_boundaries=False,
        )

    sorted_eq = sort_units_by_rate_model(data.units, data.unit_mean_rates_gt)
    sorted_eq.attrs["session_duration"] = data.session_duration_s
    _plot_raster(
        data.spikes_gt, sorted_eq,
        "Ground-truth Poisson spike trains by rate model",
        output_dir / "ground_truth_spike_raster_by_rate_model.png",
        class_boundaries=False,
    )


def plot_ground_truth_spike_summaries(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> None:
    # Spike counts by cell class
    counts = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        n_spikes = data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0]
        counts.append(n_spikes)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data.cell_class_order, counts, color="steelblue", edgecolor="white")
    ax.set_ylabel("Total spikes")
    ax.set_title("Ground-truth spike counts by cell class")
    fig.tight_layout()
    fig.savefig(output_dir / "ground_truth_spike_counts_by_cell_class.png", dpi=FIGURE_DPI)
    plt.close(fig)

    # Mean rate by cell class
    mean_rates = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        mean_rates.append(float(data.unit_mean_rates_gt.reindex(uids).mean()))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data.cell_class_order, mean_rates, color="darkorange", edgecolor="white")
    ax.set_ylabel("Mean firing rate (Hz)")
    ax.set_title("Mean firing rate by cell class")
    fig.tight_layout()
    fig.savefig(output_dir / "ground_truth_mean_rate_by_cell_class.png", dpi=FIGURE_DPI)
    plt.close(fig)

    # Rate distribution panels
    n = len(data.cell_class_order)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
    if n == 1:
        axes = [axes]
    for ax, ct in zip(axes, data.cell_class_order):
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        rates = data.unit_mean_rates_gt.reindex(uids).dropna().to_numpy()
        ax.hist(rates, bins=25, color="teal", edgecolor="white")
        ax.set_title(ct)
        ax.set_xlabel("Mean rate (Hz)")
    axes[0].set_ylabel("Unit count")
    fig.suptitle("Ground-truth rate distributions by cell class", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "ground_truth_rate_distribution_by_cell_class.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    # Population activity over time
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(CELL_CLASS_ORDER)))
    for color, ct in zip(colors, data.cell_class_order):
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"].tolist()
        t_centers, pop = _population_activity(
            data.spikes_gt, uids, data.session_duration_s, bin_size,
        )
        t_ds, y_ds = downsample_series(t_centers, pop, MAX_LINE_POINTS)
        ax.plot(t_ds, y_ds, label=ct, color=color, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Population rate (Hz)")
    ax.set_title(f"Ground-truth population activity ({bin_size * 1000:.0f} ms bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "ground_truth_population_activity_by_cell_class.png", dpi=FIGURE_DPI)
    plt.close(fig)


def generate_neural_plots(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> None:
    """Generate compressed publication-style neural population figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove older single-panel neural PNGs so the folder stays publication-clean.
    legacy_prefixes = (
        "population_activity_by_",
        "population_rates_over_time",
        "population_rate_heatmap",
        "circuit_population_activity",
        "mean_rate_by_",
        "cell_class_rate_distributions",
        "example_units_",
        "ground_truth_",
        "rate_equation_",
    )
    for png in output_dir.glob("*.png"):
        if png.name.startswith("fig_"):
            continue
        if any(png.name.startswith(p) for p in legacy_prefixes):
            png.unlink(missing_ok=True)

    from visualization.population_activity_plots import generate_population_activity_plots

    generate_population_activity_plots(data, output_dir, rate_bin_size=rate_bin_size)
