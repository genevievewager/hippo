"""Behavioral trajectory and occupancy figures.

RatInABox generates the simulated locomotor trajectory used to drive
RatInABox neural populations through spatial and proprioceptive state.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from visualization.constants import FIGURE_DPI, MAX_LINE_POINTS
from visualization.load_outputs import SimulationOutputs, downsample_series


def _overlay_arena_features(ax, data: SimulationOutputs) -> None:
    x_min, x_max, y_min, y_max = data.bounds
    ax.plot([x_min, x_max, x_max, x_min, x_min],
            [y_min, y_min, y_max, y_max, y_min], "k--", linewidth=0.8, alpha=0.5)

    for key, style in [
        ("reward_zones", {"facecolor": "gold", "alpha": 0.2}),
        ("walls", {"edgecolor": "gray", "facecolor": "none", "linewidth": 1.5}),
        ("barriers", {"edgecolor": "brown", "facecolor": "none", "linewidth": 1.2}),
    ]:
        zones = data.summary.get(key)
        if not zones:
            continue
        for zone in zones:
            if isinstance(zone, dict) and all(k in zone for k in ("x", "y", "w", "h")):
                rect = plt.Rectangle((zone["x"], zone["y"]), zone["w"], zone["h"], **style)
                ax.add_patch(rect)

    cues = data.summary.get("cue_locations") or data.summary.get("landmarks")
    if cues:
        for cue in cues:
            if isinstance(cue, dict) and "x" in cue and "y" in cue:
                ax.scatter(cue["x"], cue["y"], marker="*", s=80, c="red", zorder=5)


def plot_behavior_trajectory_xy(data: SimulationOutputs, output_dir: Path) -> None:
    beh = data.behavior
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(beh["x"], beh["y"], linewidth=0.6, color="steelblue", alpha=0.8)
    ax.scatter(beh["x"].iloc[0], beh["y"].iloc[0], c="green", s=40, label="Start", zorder=4)
    ax.scatter(beh["x"].iloc[-1], beh["y"].iloc[-1], c="red", s=40, label="End", zorder=4)
    _overlay_arena_features(ax, data)
    ax.set_xlabel("x position (cm)")
    ax.set_ylabel("y position (cm)")
    ax.set_title("Open-field trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_trajectory_xy.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_behavior_trajectory_time_colored(data: SimulationOutputs, output_dir: Path) -> None:
    beh = data.behavior
    x, y = beh["x"].to_numpy(), beh["y"].to_numpy()
    t = beh["time"].to_numpy()
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())

    fig, ax = plt.subplots(figsize=(6, 6))
    lc = LineCollection(segments, cmap="viridis", norm=norm, linewidths=0.8)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    _overlay_arena_features(ax, data)
    ax.set_xlabel("x position (cm)")
    ax.set_ylabel("y position (cm)")
    ax.set_title("Open-field trajectory (colored by time)")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(lc, ax=ax)
    cbar.set_label("Time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_trajectory_time_colored.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_behavior_speed_over_time(data: SimulationOutputs, output_dir: Path) -> None:
    t = data.behavior["time"].to_numpy()
    speed = data.behavior["speed"].to_numpy()
    t_ds, speed_ds = downsample_series(t, speed, MAX_LINE_POINTS)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_ds, speed_ds, linewidth=0.7, color="darkorange")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (cm/s)")
    ax.set_title("Locomotor speed over time")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_speed_over_time.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_behavior_head_direction_over_time(data: SimulationOutputs, output_dir: Path) -> None:
    t = data.behavior["time"].to_numpy()
    hd = data.behavior["head_direction"].to_numpy()
    t_ds, hd_ds = downsample_series(t, hd, MAX_LINE_POINTS)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t_ds, hd_ds, linewidth=0.7, color="purple")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Head direction (rad)")
    ax.set_title("Head direction over time")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_head_direction_over_time.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_behavior_occupancy_heatmap(data: SimulationOutputs, output_dir: Path, n_bins: int = 40) -> None:
    beh = data.behavior
    x_min, x_max, y_min, y_max = data.bounds
    H, xedges, yedges = np.histogram2d(
        beh["x"], beh["y"], bins=n_bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    occupancy = H * data.behavior_dt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        occupancy.T, origin="lower", aspect="auto",
        extent=[x_min, x_max, y_min, y_max], cmap="hot",
    )
    ax.set_xlabel("x position (cm)")
    ax.set_ylabel("y position (cm)")
    ax.set_title("Occupancy heatmap")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Occupancy time (s)")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_occupancy_heatmap.png", dpi=FIGURE_DPI)
    plt.close(fig)


def plot_behavior_speed_map(data: SimulationOutputs, output_dir: Path, n_bins: int = 40) -> None:
    beh = data.behavior
    x_min, x_max, y_min, y_max = data.bounds
    speed_sum, _, _ = np.histogram2d(
        beh["x"], beh["y"], bins=n_bins,
        range=[[x_min, x_max], [y_min, y_max]],
        weights=beh["speed"],
    )
    counts, _, _ = np.histogram2d(
        beh["x"], beh["y"], bins=n_bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    with np.errstate(invalid="ignore"):
        speed_map = np.divide(speed_sum, counts, where=counts > 0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        speed_map.T, origin="lower", aspect="auto",
        extent=[x_min, x_max, y_min, y_max], cmap="plasma",
    )
    ax.set_xlabel("x position (cm)")
    ax.set_ylabel("y position (cm)")
    ax.set_title("Average speed spatial map")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean speed (cm/s)")
    fig.tight_layout()
    fig.savefig(output_dir / "behavior_speed_map.png", dpi=FIGURE_DPI)
    plt.close(fig)


def generate_behavior_plots(data: SimulationOutputs, output_dir: Path) -> None:
    """Generate compressed publication-style behavioral figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    from visualization.publication_behavior_plots import (
        plot_fig_behavior_dynamics,
        _cleanup_legacy_pngs,
    )

    plot_fig_behavior_dynamics(data, output_dir)
    _cleanup_legacy_pngs(output_dir)

