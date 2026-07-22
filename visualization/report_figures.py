"""Combined report figure and summary CSV tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.constants import CELL_CLASS_ORDER, FIGURE_DPI, MAX_LINE_POINTS
from visualization.load_outputs import SimulationOutputs, downsample_series
from visualization.neural_plots import _population_activity, _bin_spike_rates


def build_summary_tables(data: SimulationOutputs, output_dir: Path) -> None:
    """Save CSV summary tables for the visualization report."""
    cell_rows = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        rates = data.unit_mean_rates_gt.reindex(uids).dropna()
        gt_n = data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0]
        sorted_n = data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0]
        cell_rows.append({
            "cell_type": ct,
            "n_units": int(len(uids)),
            "total_ground_truth_spikes": int(gt_n),
            "total_sorted_spikes": int(sorted_n),
            "mean_rate_hz": float(rates.mean()) if len(rates) else 0.0,
            "median_rate_hz": float(rates.median()) if len(rates) else 0.0,
            "std_rate_hz": float(rates.std()) if len(rates) else 0.0,
        })
    pd.DataFrame(cell_rows).to_csv(output_dir / "figure_summary_cell_classes.csv", index=False)

    beh = data.behavior
    x_min, x_max, y_min, y_max = data.bounds
    behavior_row = {
        "session_duration_s": data.session_duration_s,
        "mean_speed_cm_s": float(beh["speed"].mean()),
        "median_speed_cm_s": float(beh["speed"].median()),
        "max_speed_cm_s": float(beh["speed"].max()),
        "mean_distance_to_wall_cm": float(beh["distance_to_wall"].mean()),
        "arena_x_min": x_min,
        "arena_x_max": x_max,
        "arena_y_min": y_min,
        "arena_y_max": y_max,
    }
    pd.DataFrame([behavior_row]).to_csv(output_dir / "figure_summary_behavior.csv", index=False)

    sorting_rows = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        gt_n = data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0]
        sorted_n = data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0]
        loss = 1.0 - sorted_n / gt_n if gt_n > 0 else np.nan
        sorting_rows.append({
            "cell_type": ct,
            "ground_truth_spike_count": int(gt_n),
            "sorted_spike_count": int(sorted_n),
            "apparent_sorting_loss": float(loss),
        })
    pd.DataFrame(sorting_rows).to_csv(output_dir / "figure_summary_sorting.csv", index=False)


def plot_simulation_report_summary(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> None:
    """Multi-panel summary figure for course reports."""
    beh = data.behavior
    x_min, x_max, y_min, y_max = data.bounds

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # A: trajectory
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.plot(beh["x"], beh["y"], linewidth=0.4, color="steelblue")
    ax_a.scatter(beh["x"].iloc[0], beh["y"].iloc[0], c="green", s=20)
    ax_a.scatter(beh["x"].iloc[-1], beh["y"].iloc[-1], c="red", s=20)
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_ylim(y_min, y_max)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_title("A. Trajectory")
    ax_a.set_xlabel("x (cm)")
    ax_a.set_ylabel("y (cm)")

    # B: occupancy
    ax_b = fig.add_subplot(gs[0, 1])
    H, _, _ = np.histogram2d(beh["x"], beh["y"], bins=30, range=[[x_min, x_max], [y_min, y_max]])
    ax_b.imshow(H.T, origin="lower", aspect="auto", extent=[x_min, x_max, y_min, y_max], cmap="hot")
    ax_b.set_title("B. Occupancy")
    ax_b.set_xlabel("x (cm)")
    ax_b.set_ylabel("y (cm)")

    # C: speed over time
    ax_c = fig.add_subplot(gs[0, 2])
    t, speed = beh["time"].to_numpy(), beh["speed"].to_numpy()
    t_ds, speed_ds = downsample_series(t, speed, 2000)
    ax_c.plot(t_ds, speed_ds, linewidth=0.5, color="darkorange")
    ax_c.set_title("C. Speed")
    ax_c.set_xlabel("Time (s)")
    ax_c.set_ylabel("cm/s")

    # D: population rate by cell class
    ax_d = fig.add_subplot(gs[1, 0])
    colors = plt.cm.tab10(np.linspace(0, 1, len(CELL_CLASS_ORDER)))
    for color, ct in zip(colors, data.cell_class_order):
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"].tolist()
        t_c, pop = _population_activity(data.spikes_gt, uids, data.session_duration_s, rate_bin_size)
        t_ds, y_ds = downsample_series(t_c, pop, 2000)
        ax_d.plot(t_ds, y_ds, label=ct, color=color, linewidth=0.6)
    ax_d.set_title("D. Population rate")
    ax_d.set_xlabel("Time (s)")
    ax_d.set_ylabel("Hz")
    ax_d.legend(fontsize=6)

    # E: compact raster (subsample units)
    ax_e = fig.add_subplot(gs[1, 1:])
    max_raster_units = 80
    sampled = data.units.sample(n=min(max_raster_units, len(data.units)), random_state=42)
    sampled = sampled.sort_values("cell_type")
    spike_groups = data.spikes_gt.groupby("unit_id")["time"].apply(list).to_dict()
    for y, uid in enumerate(sampled["unit_id"]):
        times = spike_groups.get(uid, [])
        if times:
            ax_e.scatter(times, np.full(len(times), y), s=0.2, c="black", marker="|")
    ax_e.set_title(f"E. Ground-truth raster (subset, n={len(sampled)})")
    ax_e.set_xlabel("Time (s)")
    ax_e.set_yticks([])
    ax_e.set_xlim(0, data.session_duration_s)

    # F: mean rate by cell class
    ax_f = fig.add_subplot(gs[2, 0])
    mean_rates = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        mean_rates.append(float(data.unit_mean_rates_gt.reindex(uids).mean()))
    ax_f.bar(data.cell_class_order, mean_rates, color="teal", edgecolor="white")
    ax_f.set_title("F. Mean rate by class")
    ax_f.set_ylabel("Hz")
    ax_f.tick_params(axis="x", rotation=20)

    # G: probe geometry
    ax_g = fig.add_subplot(gs[2, 1:])
    if data.anatomy is not None:
        region_colors = {"CA1": "#4C72B0", "CA2": "#55A868", "CA3": "#C44E52", "DG": "#8172B2"}
        for _, row in data.anatomy.iterrows():
            z0, z1 = float(row["depth_start_um"]), float(row["depth_end_um"])
            region = str(row["region"])
            ax_g.barh((z0 + z1) / 2, width=1, height=z1 - z0,
                      color=region_colors.get(region, "gray"), alpha=0.6, edgecolor="black", linewidth=0.3)
            ax_g.text(0.5, (z0 + z1) / 2, f"{region}/{row.get('layer', '')}", ha="center", va="center", fontsize=6)
        ax_g.set_ylim(data.anatomy["depth_end_um"].max(), 0)
        ax_g.set_xticks([])
        ax_g.set_ylabel("Depth (µm)")
    ax_g.set_title("G. Probe geometry")

    fig.suptitle("Hippocampal Neuropixels simulation — report summary", fontsize=14, y=1.01)
    fig.savefig(output_dir / "simulation_report_summary.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def generate_report_figures(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    build_summary_tables(data, output_dir)
    plot_simulation_report_summary(data, output_dir, rate_bin_size)
