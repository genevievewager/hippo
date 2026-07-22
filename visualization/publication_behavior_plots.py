"""Publication multi-panel behavioral / feature figures (seaborn).

Compresses the former single-panel behavior/ and features/ sprawl into a
small set of ``fig_*`` panels matching neural publication style.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from visualization.behavior_plots import _overlay_arena_features
from visualization.constants import CELL_CLASS_ORDER, FIGURE_DPI, MAX_LINE_POINTS
from visualization.load_outputs import SimulationOutputs, downsample_series
from visualization.publication_style import (
    figure_legend_below,
    legend_below,
    panel_label,
    save_pub_figure,
)

sns.set_theme(style="ticks", context="paper", font_scale=1.0)


def _cleanup_legacy_pngs(folder: Path) -> None:
    folder = Path(folder)
    if not folder.exists():
        return
    for png in folder.glob("*.png"):
        if png.name.startswith("fig_"):
            continue
        png.unlink(missing_ok=True)


def _occupancy(data: SimulationOutputs, n_bins: int = 40) -> tuple[np.ndarray, list[float]]:
    beh = data.behavior
    x_min, x_max, y_min, y_max = data.bounds
    H, _, _ = np.histogram2d(
        beh["x"], beh["y"], bins=n_bins,
        range=[[x_min, x_max], [y_min, y_max]],
    )
    return H * data.behavior_dt, [x_min, x_max, y_min, y_max]


def _speed_map(data: SimulationOutputs, n_bins: int = 40) -> tuple[np.ndarray, list[float]]:
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
    with np.errstate(invalid="ignore", divide="ignore"):
        smap = np.full_like(speed_sum, np.nan, dtype=float)
        np.divide(speed_sum, counts, out=smap, where=counts > 0)
    return smap, [x_min, x_max, y_min, y_max]


def _draw_trajectory_xy(ax, data: SimulationOutputs) -> list:
    """Draw trajectory; return legend handles (caller places legend outside)."""
    beh = data.behavior
    t = beh["time"].to_numpy()
    x = beh["x"].to_numpy()
    y = beh["y"].to_numpy()
    if len(t) > MAX_LINE_POINTS:
        idx = np.linspace(0, len(t) - 1, MAX_LINE_POINTS).astype(int)
        x, y = x[idx], y[idx]
    ax.plot(x, y, lw=0.55, color="steelblue", alpha=0.85)
    ax.scatter(beh["x"].iloc[0], beh["y"].iloc[0], c="#2ca02c", s=28, zorder=4)
    ax.scatter(beh["x"].iloc[-1], beh["y"].iloc[-1], c="#d62728", s=28, zorder=4)
    _overlay_arena_features(ax, data)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="box")
    sns.despine(ax=ax)
    return [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c", markersize=7, label="Start"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728", markersize=7, label="End"),
    ]


def _draw_trajectory_time(ax, data: SimulationOutputs) -> None:
    beh = data.behavior
    x, y = beh["x"].to_numpy(), beh["y"].to_numpy()
    t = beh["time"].to_numpy()
    if len(t) > MAX_LINE_POINTS:
        idx = np.linspace(0, len(t) - 1, MAX_LINE_POINTS).astype(int)
        x, y, t = x[idx], y[idx], t[idx]
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t.min(), t.max())
    lc = LineCollection(segments, cmap="viridis", norm=norm, linewidths=0.7)
    lc.set_array(t[:-1])
    ax.add_collection(lc)
    ax.autoscale()
    _overlay_arena_features(ax, data)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Time (s)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    sns.despine(ax=ax)


def plot_fig_behavior_overview(
    data: SimulationOutputs,
    output_dir: Path,
    n_bins: int = 40,
) -> Path:
    """Trajectory + occupancy + speed map (publication 2×2)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.0, 7.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    handles = _draw_trajectory_xy(ax_a, data)
    # Annotate start/end beside points (no legend over trajectory)
    beh = data.behavior
    ax_a.annotate("Start", (beh["x"].iloc[0], beh["y"].iloc[0]),
                  textcoords="offset points", xytext=(6, 6), fontsize=7, color="#2ca02c")
    ax_a.annotate("End", (beh["x"].iloc[-1], beh["y"].iloc[-1]),
                  textcoords="offset points", xytext=(6, -10), fontsize=7, color="#d62728")
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    _draw_trajectory_time(ax_b, data)
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    occ, extent = _occupancy(data, n_bins=n_bins)
    im = ax_c.imshow(occ.T, origin="lower", aspect="equal", extent=extent, cmap="hot")
    ax_c.set_xlabel("x (cm)")
    ax_c.set_ylabel("y (cm)")
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Time (s)", fontsize=8)
    sns.despine(ax=ax_c)
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    smap, extent = _speed_map(data, n_bins=n_bins)
    im = ax_d.imshow(smap.T, origin="lower", aspect="equal", extent=extent, cmap="plasma")
    ax_d.set_xlabel("x (cm)")
    ax_d.set_ylabel("y (cm)")
    cbar = plt.colorbar(im, ax=ax_d, fraction=0.046, pad=0.04)
    cbar.set_label("cm/s", fontsize=8)
    sns.despine(ax=ax_d)
    panel_label(ax_d, "D")

    dur = float(data.session_duration_s) if hasattr(data, "session_duration_s") else float(data.behavior["time"].iloc[-1])
    return save_pub_figure(
        fig, output_dir / "fig_behavior_overview.png", dpi=FIGURE_DPI,
        rect=(0.02, 0.08, 0.98, 0.94),
    )


def plot_fig_behavior_dynamics(data: SimulationOutputs, output_dir: Path) -> Path:
    """Speed, heading, wall distance over time + feature distributions."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    beh = data.behavior
    t = beh["time"].to_numpy()

    # No twinx: twin y-labels collide with the neighboring panel.
    fig = plt.figure(figsize=(11.0, 7.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    t_ds, y_ds = downsample_series(t, beh["speed"].to_numpy(), MAX_LINE_POINTS)
    ax_a.plot(t_ds, y_ds, lw=0.7, color="#e07a2f")
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Speed (cm/s)")
    sns.despine(ax=ax_a)
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    t_ds, y_ds = downsample_series(t, beh["head_direction"].to_numpy(), MAX_LINE_POINTS)
    ax_b.plot(t_ds, y_ds, lw=0.7, color="#6b4c9a")
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Head direction (rad)")
    sns.despine(ax=ax_b)
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    if "distance_to_wall" in beh.columns:
        t_ds, y_ds = downsample_series(t, beh["distance_to_wall"].to_numpy(), MAX_LINE_POINTS)
        ax_c.plot(t_ds, y_ds, lw=0.7, color="#2a9d8f")
        ax_c.set_ylabel("Distance to wall (cm)")
    elif "acceleration" in beh.columns:
        t_ds, y_ds = downsample_series(t, beh["acceleration"].to_numpy(), MAX_LINE_POINTS)
        ax_c.plot(t_ds, y_ds, lw=0.7, color="#264653")
        ax_c.set_ylabel("Acceleration (cm/s²)")
    else:
        ax_c.text(0.5, 0.5, "No wall/accel columns", ha="center", va="center", transform=ax_c.transAxes)
    ax_c.set_xlabel("Time (s)")
    sns.despine(ax=ax_c)
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    dist_cols = [
        ("speed", "Speed"),
        ("head_direction", "HD"),
        ("distance_to_wall", "Wall dist"),
        ("acceleration", "Accel"),
    ]
    zrows = []
    for col, label in dist_cols:
        if col not in beh.columns:
            continue
        vals = beh[col].to_numpy(dtype=float)
        std = float(np.std(vals))
        if std < 1e-12:
            continue
        z = (vals - float(np.mean(vals))) / std
        zrows.append(pd.DataFrame({"z": z, "feature": label}))
    if zrows:
        long = pd.concat(zrows, ignore_index=True)
        sns.violinplot(
            data=long, x="feature", y="z", hue="feature",
            palette="deep", legend=False, cut=0, inner="quartile",
            density_norm="width", ax=ax_d,
        )
        ax_d.set_xlabel("")
        ax_d.set_ylabel("z-scored value")
        ax_d.tick_params(axis="x", labelsize=8)
        sns.despine(ax=ax_d)
    else:
        ax_d.axis("off")
    panel_label(ax_d, "D")

    return save_pub_figure(fig, output_dir / "fig_behavior_dynamics.png", dpi=FIGURE_DPI)


def plot_fig_behavior_features(data: SimulationOutputs, output_dir: Path) -> Path:
    """Key behavioral covariates over time (compact multi-trace panel)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    beh = data.behavior
    t = beh["time"].to_numpy()

    panels = [
        ("x", "x (cm)"),
        ("y", "y (cm)"),
        ("speed", "Speed (cm/s)"),
        ("head_direction", "HD (rad)"),
        ("distance_to_wall", "Wall dist (cm)"),
        ("acceleration", "Accel (cm/s²)"),
        ("theta_phase", "Theta phase"),
        ("ripple", "Ripple"),
    ]
    available = [(c, lab) for c, lab in panels if c in beh.columns][:6]

    fig = plt.figure(figsize=(11.0, 7.6))
    n = max(len(available), 1)
    nrows = int(np.ceil(n / 2))
    gs = GridSpec(nrows, 2, figure=fig, hspace=0.48, wspace=0.35)
    labels = "ABCDEF"

    for i, (col, lab) in enumerate(available):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        t_ds, y_ds = downsample_series(t, beh[col].to_numpy(), MAX_LINE_POINTS)
        ax.plot(t_ds, y_ds, lw=0.65, color=sns.color_palette("deep")[i % 6])
        ax.set_ylabel(lab, fontsize=8)
        if i >= n - 2:
            ax.set_xlabel("Time (s)")
        sns.despine(ax=ax)
        panel_label(ax, labels[i])

    for j in range(len(available), nrows * 2):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        ax.axis("off")

    return save_pub_figure(fig, output_dir / "fig_behavior_features.png", dpi=FIGURE_DPI)


def plot_fig_neural_drivers(data: SimulationOutputs, output_dir: Path) -> Path | None:
    """Compress neural driver features into a 2×2 publication panel."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from visualization.feature_plots import _compute_class_driver_averages
    except Exception:
        return None

    class_drivers = _compute_class_driver_averages(data)
    present = [ct for ct in CELL_CLASS_ORDER if ct in class_drivers]
    if not present:
        return None

    t = data.behavior["time"].to_numpy()
    driver_keys = ["place", "hd", "speed", "boundary"]
    driver_labels = {"place": "Place", "hd": "HD", "speed": "Speed", "boundary": "Boundary"}
    palette = sns.color_palette("deep", n_colors=max(len(present), 1))
    ct_color = {ct: c for ct, c in zip(present, palette)}

    fig = plt.figure(figsize=(11.0, 8.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)

    legend_handles = []
    for ax_idx, key in enumerate(driver_keys[:3]):
        ax = fig.add_subplot(gs[ax_idx // 2, ax_idx % 2])
        for ct in present:
            y = class_drivers[ct][key]
            t_ds, y_ds = downsample_series(t, y, min(MAX_LINE_POINTS, 1500))
            line, = ax.plot(t_ds, y_ds, lw=0.8, color=ct_color[ct], label=ct, alpha=0.9)
            if ax_idx == 0:
                legend_handles.append(line)
        ax.set_ylabel(f"{driver_labels[key]} drive")
        ax.set_xlabel("Time (s)")
        sns.despine(ax=ax)
        panel_label(ax, "ABC"[ax_idx])

    ax_d = fig.add_subplot(gs[1, 1])
    mat = np.zeros((len(present), len(driver_keys)))
    for i, ct in enumerate(present):
        for j, key in enumerate(driver_keys):
            mat[i, j] = float(np.mean(np.abs(class_drivers[ct][key])))
    sns.heatmap(
        mat, ax=ax_d, annot=True, fmt=".2f", cmap="viridis",
        xticklabels=[driver_labels[k] for k in driver_keys],
        yticklabels=present,
        cbar_kws={"label": "Mean |drive|", "shrink": 0.85},
    )
    ax_d.tick_params(axis="y", labelsize=7, pad=2)
    ax_d.tick_params(axis="x", labelsize=8)
    panel_label(ax_d, "D")

    if legend_handles:
        figure_legend_below(
            fig,
            legend_handles,
            [h.get_label() for h in legend_handles],
            ncol=min(5, len(legend_handles)),
            fontsize=7,
            y=0.01,
        )

    return save_pub_figure(
        fig, output_dir / "fig_neural_drivers.png", dpi=FIGURE_DPI,
        rect=(0.02, 0.08, 0.98, 0.94),
    )


def generate_publication_behavior_plots(
    data: SimulationOutputs,
    behavior_dir: Path,
    features_dir: Path | None = None,
) -> list[Path]:
    """Write publication behavior (+ optional features) figures and clean legacy."""
    behavior_dir = Path(behavior_dir)
    behavior_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for fn in (plot_fig_behavior_overview, plot_fig_behavior_dynamics):
        written.append(fn(data, behavior_dir))

    _cleanup_legacy_pngs(behavior_dir)

    if features_dir is not None:
        features_dir = Path(features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)
        written.append(plot_fig_behavior_features(data, features_dir))
        nd = plot_fig_neural_drivers(data, features_dir)
        if nd is not None:
            written.append(nd)
        _cleanup_legacy_pngs(features_dir)

    return written
