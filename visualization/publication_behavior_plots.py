"""Publication multi-panel behavioral / feature figures (seaborn).

Compresses the former single-panel behavior/ and features/ sprawl into a
small set of ``fig_*`` panels matching neural publication style.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

from visualization.behavior_plots import _overlay_arena_features
from visualization.constants import CELL_CLASS_ORDER, FIGURE_DPI, MAX_LINE_POINTS, cell_class_colors
from visualization.load_outputs import SimulationOutputs, downsample_series
from visualization.publication_style import (
    apply_publication_theme,
    enable_open_axes,
    figure_legend_below,
    panel_label,
    save_pub_figure,
    style_figure_axes,
    strip_box_frames,
)

apply_publication_theme()

# Locomotor covariate trace hierarchy (panels D–I): one hue, stepped saturation.
_BEHAVIOR_TRACE_HUE = 0.52  # teal (~187°)
_BEHAVIOR_TRACE_TIERS: dict[str, tuple[float, float]] = {
    # Primary — position (darkest)
    "x": (0.72, 0.22),
    "y": (0.72, 0.22),
    # Secondary — first derivatives / kinematic covariates (middle)
    "speed": (0.52, 0.40),
    "head_direction": (0.52, 0.40),
    "distance_to_wall": (0.52, 0.40),
    # Tertiary — acceleration (lightest)
    "acceleration": (0.28, 0.68),
}
_BEHAVIOR_TRACE_DEFAULT_TIER = (0.48, 0.44)


def behavior_covariate_trace_color(column: str) -> tuple[float, float, float]:
    """RGB for a covariate time-series: location darkest → accel lightest."""
    sat, light = _BEHAVIOR_TRACE_TIERS.get(column, _BEHAVIOR_TRACE_DEFAULT_TIER)
    return colorsys.hls_to_rgb(_BEHAVIOR_TRACE_HUE, light, sat)


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


def _draw_trajectory_colored(
    ax,
    data: SimulationOutputs,
    *,
    color_col: str = "time",
    cbar_label: str = "Time (s)",
    cmap: str = "viridis",
    mark_endpoints: bool = False,
    cax=None,
    equal_aspect: bool = True,
    draw_cbar: bool = True,
):
    """Path segments colored by a behavioral column (default: elapsed time).

    Returns the LineCollection mappable (for attaching a colorbar externally).
    """
    beh = data.behavior
    x, y = beh["x"].to_numpy(), beh["y"].to_numpy()
    c = beh[color_col].to_numpy(dtype=float)
    if len(c) > MAX_LINE_POINTS:
        idx = np.linspace(0, len(c) - 1, MAX_LINE_POINTS).astype(int)
        x, y, c = x[idx], y[idx], c[idx]
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    c_seg = c[:-1]
    finite = np.isfinite(c_seg)
    if finite.any():
        norm = plt.Normalize(float(np.nanmin(c_seg)), float(np.nanmax(c_seg)))
    else:
        norm = plt.Normalize(0.0, 1.0)
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=0.7)
    lc.set_array(c_seg)
    ax.add_collection(lc)
    ax.autoscale()
    if mark_endpoints:
        ax.scatter(beh["x"].iloc[0], beh["y"].iloc[0], c="#2ca02c", s=28, zorder=4)
        ax.scatter(beh["x"].iloc[-1], beh["y"].iloc[-1], c="#d62728", s=28, zorder=4)
        ax.annotate(
            "Start", (beh["x"].iloc[0], beh["y"].iloc[0]),
            textcoords="offset points", xytext=(6, 6), fontsize=8, color="#2ca02c",
        )
        ax.annotate(
            "End", (beh["x"].iloc[-1], beh["y"].iloc[-1]),
            textcoords="offset points", xytext=(6, -10), fontsize=8, color="#d62728",
        )
    _overlay_arena_features(ax, data)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    x_min, x_max, y_min, y_max = data.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")
    if draw_cbar:
        if cax is not None:
            cbar = ax.figure.colorbar(lc, cax=cax)
        else:
            cbar = plt.colorbar(lc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=10)
        cbar.ax.tick_params(labelsize=8)
    sns.despine(ax=ax)
    return lc


def plot_fig_behavior_dynamics(
    data: SimulationOutputs,
    output_dir: Path,
    n_bins: int = 40,
) -> Path:
    """Merged behavior page: spatial overview (A–C) + covariate traces (D–I).

    Top row (former overview A–C): time-colored trajectory, speed map, occupancy.
    Below (former ``fig_behavior_features`` A–F): x, y, speed, heading, wall
    distance, acceleration over time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    beh = data.behavior
    t = beh["time"].to_numpy()

    cov_panels = [
        ("x", "x (cm)"),
        ("y", "y (cm)"),
        ("speed", "Speed (cm/s)"),
        ("head_direction", "HD (rad)"),
        ("distance_to_wall", "Wall dist (cm)"),
        ("acceleration", "Accel (cm/s²)"),
    ]
    available = [(c, lab) for c, lab in cov_panels if c in beh.columns][:6]
    n_cov = max(len(available), 1)
    cov_rows = int(np.ceil(n_cov / 2))

    # Larger page: A–C are big squares; D–I keep mutual scaling (1.25 each).
    # Top row height ≈ one-third figure width so squares fill their slots.
    fig_w = 14.0
    fig_h = 5.5 + 2.9 * cov_rows
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = GridSpec(
        1 + cov_rows, 1, figure=fig,
        height_ratios=[1.85] + [1.25] * cov_rows,
        hspace=0.28,
        left=0.07,
        right=0.98,
        top=0.98,
        bottom=0.045,
    )
    gs_top = outer[0].subgridspec(1, 3, wspace=0.18)
    gs_cov = outer[1:].subgridspec(cov_rows, 2, hspace=0.28, wspace=0.16)

    def _label(ax, letter: str) -> None:
        # Outside the axes, top-left corner (not over data).
        panel_label(ax, letter, x=0.0, y=1.10, ha="left", va="bottom")

    def _square_with_cbar(ax, mappable, cbar_label: str):
        """Attach a colorbar; square sizing is applied later when packing A–C."""
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label(cbar_label, fontsize=13, labelpad=4)
        cbar.ax.tick_params(labelsize=11, length=4, width=1.0)
        return cbar

    def _style_spatial(ax) -> None:
        """Larger labels/ticks for the enlarged A–C squares."""
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        ax.tick_params(axis="both", labelsize=12, length=5, width=1.1)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(12)

    # A — trajectory colored by time (square: arena is square)
    ax_a = fig.add_subplot(gs_top[0, 0])
    lc_a = _draw_trajectory_colored(
        ax_a, data, mark_endpoints=True, equal_aspect=True, draw_cbar=False,
    )
    cbar_a = _square_with_cbar(ax_a, lc_a, "s")
    _label(ax_a, "A")

    # B — speed over the maze
    ax_b = fig.add_subplot(gs_top[0, 1])
    smap, extent = _speed_map(data, n_bins=n_bins)
    im_b = ax_b.imshow(smap.T, origin="lower", aspect="equal", extent=extent, cmap="plasma")
    ax_b.set_xlabel("x (cm)")
    ax_b.set_ylabel("")
    ax_b.tick_params(labelleft=False)
    cbar_b = _square_with_cbar(ax_b, im_b, "cm/s")
    sns.despine(ax=ax_b)
    _label(ax_b, "B")

    # C — occupancy in the maze
    ax_c = fig.add_subplot(gs_top[0, 2])
    occ, extent = _occupancy(data, n_bins=n_bins)
    im_c = ax_c.imshow(occ.T, origin="lower", aspect="equal", extent=extent, cmap="hot")
    ax_c.set_xlabel("x (cm)")
    ax_c.set_ylabel("")
    ax_c.tick_params(labelleft=False)
    cbar_c = _square_with_cbar(ax_c, im_c, "s")
    sns.despine(ax=ax_c)
    _label(ax_c, "C")

    # D–I — behavioral covariates (former fig_behavior_features A–F)
    letters = "DEFGHI"
    cov_axes: list = []
    for i, (col, lab) in enumerate(available):
        ax = fig.add_subplot(gs_cov[i // 2, i % 2])
        cov_axes.append(ax)
        t_ds, y_ds = downsample_series(t, beh[col].to_numpy(), MAX_LINE_POINTS)
        ax.plot(t_ds, y_ds, lw=0.75, color=behavior_covariate_trace_color(col))
        ax.set_ylabel(lab, fontsize=10)
        on_bottom = i >= n_cov - 2
        if on_bottom:
            ax.set_xlabel("Time (s)")
        sns.despine(ax=ax)
        _label(ax, letters[i])

    for j in range(len(available), cov_rows * 2):
        ax = fig.add_subplot(gs_cov[j // 2, j % 2])
        ax.axis("off")

    # Keep GridSpec gutters (avoid save_pub_figure subplots_adjust widening them).
    path = Path(output_dir) / "fig_behavior_dynamics.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    enable_open_axes()
    style_figure_axes(fig)
    strip_box_frames(fig)
    # Re-apply after style_figure_axes (it resets tick visibility / sizes).
    ax_b.tick_params(labelleft=False)
    ax_c.tick_params(labelleft=False)
    for ax in (ax_a, ax_b, ax_c):
        _style_spatial(ax)
    for cbar, cbar_lab in ((cbar_a, "s"), (cbar_b, "cm/s"), (cbar_c, "s")):
        cbar.set_label(cbar_lab, fontsize=13, labelpad=4)
        cbar.ax.tick_params(labelsize=11, length=4, width=1.0)
        cbar.ax.yaxis.label.set_size(13)
    for i, ax in enumerate(cov_axes):
        if i < n_cov - 2:
            ax.tick_params(labelbottom=False)

    # Pack A–C as equal squares spanning the same left/right edges as D–I.
    fig.canvas.draw()
    top_cells = [gs_top[0, i].get_position(fig) for i in range(3)]
    band_y0 = min(c.y0 for c in top_cells)
    band_y1 = max(c.y1 for c in top_cells)
    band_h = band_y1 - band_y0

    # Match outer edges of the lower D–I block (plot boxes).
    left_cov = [cov_axes[i] for i in range(0, len(cov_axes), 2)]
    right_cov = [cov_axes[i] for i in range(1, len(cov_axes), 2)]
    left = min(ax.get_position().x0 for ax in left_cov)
    right = max(ax.get_position().x1 for ax in right_cov)

    spatial = [ax_a, ax_b, ax_c]

    def _cbar_for(ax):
        bb = ax.get_position()
        best = None
        best_dx = 1e9
        for other in fig.axes:
            if other is ax:
                continue
            ob = other.get_position()
            if abs(ob.height - bb.height) < 0.08 and ob.x0 >= bb.x1 - 0.02:
                dx = ob.x0 - bb.x1
                if -0.01 <= dx < best_dx and ob.width < bb.width * 0.5:
                    best_dx = dx
                    best = other
        return best

    pairs = [(ax, _cbar_for(ax)) for ax in spatial]
    n = 3
    cbar_frac = 0.048
    pad = 0.006
    width_budget = right - left
    # Solve for side and equal inter-group gaps so:
    #   A.x0 == left  and  C.cbar.x1 == right
    # n*(side + pad + side*cbar_frac) + (n-1)*gap = width_budget
    # Prefer max square that fits the top band; leftover width → equal gaps.
    side = min(
        band_h * 0.90,  # leave room for x labels under squares
        width_budget / (n * (1.0 + cbar_frac) + 0.02),
    )
    unit = side * (1.0 + cbar_frac) + pad
    gap = max(0.012, (width_budget - n * unit) / (n - 1))
    # If gaps went negative, shrink side to fit with a minimum gap.
    min_gap = 0.012
    if gap < min_gap:
        gap = min_gap
        side = (width_budget - (n - 1) * gap - n * pad) / (n * (1.0 + cbar_frac))
        unit = side * (1.0 + cbar_frac) + pad

    y0 = band_y1 - side - 0.025
    y0 = max(y0, band_y0)
    x = left
    for ax, cax in pairs:
        ax.set_position([x, y0, side, side])
        ax.set_aspect("equal", adjustable="datalim")
        cx = x + side + pad
        if cax is not None:
            cax.set_position([cx, y0, side * cbar_frac, side])
        x = cx + side * cbar_frac + gap

    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    (output_dir / "fig_behavior_overview.png").unlink(missing_ok=True)
    (output_dir / "fig_behavior_features.png").unlink(missing_ok=True)
    return path


def plot_fig_behavior_overview(
    data: SimulationOutputs,
    output_dir: Path,
    n_bins: int = 40,
) -> Path:
    """Deprecated: absorbed into ``fig_behavior_dynamics``."""
    return plot_fig_behavior_dynamics(data, output_dir, n_bins=n_bins)


def plot_fig_behavior_features(data: SimulationOutputs, output_dir: Path) -> Path:
    """Deprecated: covariate traces now live under ``fig_behavior_dynamics``."""
    return plot_fig_behavior_dynamics(data, output_dir)


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
    ct_color = cell_class_colors(present)

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
            y=0.01,
            title="Cell class (A–C)",
        )

    return save_pub_figure(
        fig, output_dir / "fig_neural_drivers.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.14, 0.96, 0.94),
    )


def generate_publication_behavior_plots(
    data: SimulationOutputs,
    behavior_dir: Path,
    features_dir: Path | None = None,
) -> list[Path]:
    """Write merged behavior figure (+ neural drivers) and clean legacy."""
    behavior_dir = Path(behavior_dir)
    behavior_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Spatial A–C + covariate D–I live in the behavior folder.
    written.append(plot_fig_behavior_dynamics(data, behavior_dir))
    _cleanup_legacy_pngs(behavior_dir)

    # Neural drivers stay with covariates (same conceptual section).
    out_features = Path(features_dir) if features_dir is not None else behavior_dir
    out_features.mkdir(parents=True, exist_ok=True)
    # Drop standalone covariate page if a previous run left it in features/.
    (out_features / "fig_behavior_features.png").unlink(missing_ok=True)
    (out_features / "fig_behavior_dynamics.png").unlink(missing_ok=True)
    nd = plot_fig_neural_drivers(data, out_features)
    if nd is not None:
        written.append(nd)
    _cleanup_legacy_pngs(out_features)

    return written
