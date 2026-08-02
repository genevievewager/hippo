"""Ground-truth spike, population-rate, and raster visualizations.

Ground-truth Poisson spike trains are generated from RatInABox firing rates
and provide the true neural activity before Neuropixels recording degradation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from visualization.constants import (
    CELL_CLASS_ORDER,
    FIGURE_DPI,
    LOCAL_INT_CELL_TYPES,
    MAX_LINE_POINTS,
    cell_class_colors,
)
from visualization.load_outputs import (
    SimulationOutputs,
    downsample_series,
    sort_units_by_class_and_rate,
    sort_units_by_rate_model,
)
from visualization.publication_style import (
    enable_open_axes,
    strip_box_frames,
    style_figure_axes,
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
    colors = cell_class_colors(CELL_CLASS_ORDER)

    for ct in CELL_CLASS_ORDER:
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
        ax.plot(t_ds, y_ds, label=ct, color=colors[ct], linewidth=0.8)

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


# Primary programmed covariate per cell class (RatInABox / synthetic backend).
_PRIMARY_TUNING: dict[str, str] = {
    "CA1_pyr": "place",
    "CA2_pyr": "place",
    "CA3_pyr": "place",
    "DG_granule": "place",
    "MEC_grid": "place",
    "MEC_hd": "head_direction",
    "MEC_speed": "speed",
    "Sub_bvc": "boundary",
    "INT_CA1": "theta",
    "INT_CA2": "theta",
    "INT_CA3": "theta",
    "INT_DG": "theta",
    "INT_SUB": "theta",
    "interneuron": "theta",
    "CA1_int": "theta",  # legacy
}

_TUNING_TITLES: dict[str, str] = {
    "place": "Spatial rate map",
    "head_direction": "HD tuning",
    "speed": "Speed tuning",
    "boundary": "Wall-distance tuning",
    "theta": "Theta-phase tuning",
}


def _unit_row_index(data: SimulationOutputs, unit_id: int) -> int:
    idx = np.flatnonzero(data.units["unit_id"].to_numpy() == int(unit_id))
    if len(idx) == 0:
        raise KeyError(f"unit_id {unit_id} not in units table")
    return int(idx[0])


def _unit_rate_trace(data: SimulationOutputs, unit_id: int) -> np.ndarray:
    """Per-timestep rate (Hz) for one unit, aligned to behavior samples."""
    if data.has_ground_truth_rates:
        assert data.rates_hz is not None
        row = _unit_row_index(data, unit_id)
        n = len(data.behavior)
        rate = np.asarray(data.rates_hz[row], dtype=float)
        if len(rate) == n:
            return rate
        if len(rate) > n:
            return rate[:n]
        out = np.zeros(n, dtype=float)
        out[: len(rate)] = rate
        return out

    # Spike-derived fallback on the behavior time grid.
    spikes = data.spikes_gt.loc[data.spikes_gt["unit_id"] == unit_id, "time"].to_numpy()
    t = data.behavior["time"].to_numpy()
    dt = float(data.behavior_dt) if data.behavior_dt > 0 else 0.05
    edges = np.concatenate([t - 0.5 * dt, [t[-1] + 0.5 * dt]])
    counts, _ = np.histogram(spikes, bins=edges)
    return gaussian_filter1d(counts.astype(float) / dt, sigma=1.0)


def _mean_rate_1d(
    feature: np.ndarray,
    rate: np.ndarray,
    *,
    n_bins: int,
    feat_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy-normalized mean rate vs a 1-D behavioral feature."""
    finite = np.isfinite(feature) & np.isfinite(rate)
    feature = feature[finite]
    rate = rate[finite]
    if feature.size == 0:
        centers = np.linspace(0.0, 1.0, n_bins)
        return centers, np.full(n_bins, np.nan)
    lo, hi = feat_range if feat_range is not None else (float(np.min(feature)), float(np.max(feature)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, 1.0
    edges = np.linspace(lo, hi, n_bins + 1)
    weighted, _ = np.histogram(feature, bins=edges, weights=rate)
    occ, _ = np.histogram(feature, bins=edges)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.full(n_bins, np.nan, dtype=float)
        np.divide(weighted, occ, out=means, where=occ > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means


def _spatial_rate_map(
    x: np.ndarray,
    y: np.ndarray,
    rate: np.ndarray,
    bounds: tuple[float, float, float, float],
    n_bins: int = 40,
    smooth_sigma: float = 1.75,
) -> tuple[np.ndarray, list[float]]:
    """Occupancy-normalized spatial rate map with Gaussian spillover.

    Spikes/rate and occupancy are smoothed separately, then divided so
    unvisited bins inherit rate from nearby visited bins (no NaN holes).
    """
    x_min, x_max, y_min, y_max = bounds
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(rate)
    x, y, rate = x[finite], y[finite], rate[finite]
    rate_sum, _, _ = np.histogram2d(
        x, y, bins=n_bins, range=[[x_min, x_max], [y_min, y_max]], weights=rate,
    )
    occ, _, _ = np.histogram2d(
        x, y, bins=n_bins, range=[[x_min, x_max], [y_min, y_max]],
    )
    if smooth_sigma and smooth_sigma > 0:
        rate_sum = gaussian_filter(rate_sum.astype(float), sigma=float(smooth_sigma), mode="nearest")
        occ = gaussian_filter(occ.astype(float), sigma=float(smooth_sigma), mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        rmap = np.zeros_like(rate_sum, dtype=float)
        np.divide(rate_sum, occ, out=rmap, where=occ > 1e-6)
    return rmap, [x_min, x_max, y_min, y_max]


def _smooth_spatial_rate_map(
    rmap: np.ndarray,
    *,
    sigma: float = 1.75,
    min_weight: float = 1e-3,
) -> np.ndarray:
    """Nan-aware Gaussian smooth of an already-binned rate map."""
    visited = np.isfinite(rmap)
    if not visited.any():
        return np.zeros_like(rmap, dtype=float)
    values = np.nan_to_num(rmap, nan=0.0)
    num = gaussian_filter(values, sigma=sigma, mode="nearest")
    den = gaussian_filter(visited.astype(float), sigma=sigma, mode="nearest")
    out = np.zeros_like(num, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        np.divide(num, den, out=out, where=den > min_weight)
    return out


_FEATURE_AXIS: dict[str, tuple[str, str]] = {
    "place": ("Distance to field (cm)", "Preferred dist (cm)"),
    "head_direction": ("Head direction (rad)", "Preferred HD (rad)"),
    "speed": ("Speed (cm/s)", "Peak rate (Hz)"),
    "boundary": ("Distance to wall (cm)", "Preferred wall dist (cm)"),
    "theta": ("Theta phase (rad)", "Preferred phase (rad)"),
}

_MAX_MOSAIC = 16


def _class_unit_ids_by_rate(data: SimulationOutputs, cell_type: str) -> list[int]:
    # The 3×3 INT_CA1 panel represents all local INT pools.
    if cell_type in ("INT_CA1", "interneuron", "CA1_int"):
        mask = data.units["cell_type"].astype(str).isin(LOCAL_INT_CELL_TYPES)
    else:
        mask = data.units["cell_type"] == cell_type
    uids = data.units.loc[mask, "unit_id"]
    rates = data.unit_mean_rates_gt.reindex(uids).dropna().sort_values(ascending=False)
    return [int(u) for u in rates.index.tolist()]


def _place_field_center(data: SimulationOutputs, unit_id: int) -> tuple[float, float]:
    """Preferred place location from metadata, else peak of the spatial rate map."""
    row = data.units.loc[data.units["unit_id"] == unit_id]
    if not row.empty:
        r = row.iloc[0]
        if "place_x" in data.units.columns and "place_y" in data.units.columns:
            cx, cy = r.get("place_x"), r.get("place_y")
            if np.isfinite(cx) and np.isfinite(cy):
                return float(cx), float(cy)
    beh = data.behavior
    rmap, extent = _spatial_rate_map(
        beh["x"].to_numpy(), beh["y"].to_numpy(),
        _unit_rate_trace(data, unit_id), data.bounds, n_bins=32,
    )
    cx, cy, _ = _field_centroid(rmap, extent)
    if not np.isfinite(cx):
        x_min, x_max, y_min, y_max = data.bounds
        return 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
    return cx, cy


def _feature_vector(
    data: SimulationOutputs, kind: str, unit_id: int | None = None,
) -> tuple[np.ndarray, tuple[float, float] | None, int]:
    beh = data.behavior
    if kind == "place":
        if unit_id is None:
            return np.array([]), None, 25
        cx, cy = _place_field_center(data, unit_id)
        dist = np.sqrt((beh["x"].to_numpy() - cx) ** 2 + (beh["y"].to_numpy() - cy) ** 2)
        x_min, x_max, y_min, y_max = data.bounds
        diag = float(np.hypot(x_max - x_min, y_max - y_min))
        hi = min(float(np.nanpercentile(dist, 99)) if np.isfinite(dist).any() else diag, diag)
        return dist, (0.0, max(hi, 1.0)), 25
    if kind == "head_direction":
        # Behavior HD is atan2-style [-π, π]; wrap to [0, 2π) for binning.
        hd = np.mod(beh["head_direction"].to_numpy().astype(float), 2 * np.pi)
        return hd, (0.0, 2 * np.pi), 36
    if kind == "speed":
        spd = beh["speed"].to_numpy()
        hi = float(np.nanpercentile(spd, 99)) if np.isfinite(spd).any() else 40.0
        return spd, (0.0, max(hi, 1.0)), 25
    if kind == "boundary":
        if "distance_to_wall" not in beh.columns:
            return np.array([]), None, 25
        dist = beh["distance_to_wall"].to_numpy()
        hi = float(np.nanpercentile(dist, 99)) if np.isfinite(dist).any() else 50.0
        return dist, (0.0, max(hi, 1.0)), 25
    if kind == "theta":
        if "theta_phase" not in beh.columns:
            return np.array([]), None, 36
        return beh["theta_phase"].to_numpy(), (0.0, 2 * np.pi), 36
    raise ValueError(f"Not a 1-D tuning kind: {kind}")


def _unit_tuning_1d(
    data: SimulationOutputs, unit_id: int, kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    feat, feat_range, n_bins = _feature_vector(data, kind, unit_id=unit_id)
    if feat.size == 0 or feat_range is None:
        return np.array([]), np.array([])
    rate = _unit_rate_trace(data, unit_id)
    return _mean_rate_1d(feat, rate, n_bins=n_bins, feat_range=feat_range)


def _preferred_from_curve(centers: np.ndarray, means: np.ndarray, kind: str) -> float:
    if centers.size == 0 or not np.isfinite(means).any():
        return float("nan")
    if kind == "speed":
        # Linear speed cells: amplitude, not a preferred speed peak.
        return float(np.nanmax(means))
    return float(centers[int(np.nanargmax(means))])


def _field_centroid(rmap: np.ndarray, extent: list[float]) -> tuple[float, float, float]:
    if not np.isfinite(rmap).any():
        return float("nan"), float("nan"), float("nan")
    ix, iy = np.unravel_index(np.nanargmax(rmap), rmap.shape)
    x_min, x_max, y_min, y_max = extent
    nx, ny = rmap.shape
    xc = x_min + (ix + 0.5) * (x_max - x_min) / nx
    yc = y_min + (iy + 0.5) * (y_max - y_min) / ny
    return float(xc), float(yc), float(np.nanmax(rmap))


def _bare_image_axes(ax) -> None:
    """No ticks, no spines, no closed box around an image panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _open_spines(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _finalize_tuning_figure(fig, path: Path) -> Path:
    style_figure_axes(fig)
    strip_box_frames(fig)
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>" or "outline" in ax.spines:
            for spine in ax.spines.values():
                spine.set_visible(False)
            continue
        has_image = any(type(c).__name__ == "AxesImage" for c in ax.get_children())
        no_tick_labels = (
            not any(t.get_text() for t in ax.get_xticklabels())
            and not any(t.get_text() for t in ax.get_yticklabels())
        )
        if has_image and no_tick_labels:
            _bare_image_axes(ax)
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    return path


# Figure 8 panel formats: MEC_grid keeps the 3×3 rate-map mosaic; others are heatmaps.
_FEATURED_TUNING_PANELS: dict[str, str] = {
    "MEC_grid": "place_mosaic",
    "MEC_hd": "tuning_heatmap",
    "MEC_speed": "tuning_heatmap",
    "Sub_bvc": "tuning_heatmap",
}

# Alias of classes with explicit figure-8 panel formats (tests / older call sites).
_FEATURED_CELL_CLASSES: frozenset[str] = frozenset(_FEATURED_TUNING_PANELS)

_PANEL_CLASS_LABELS: dict[str, str] = {
    "CA1_pyr": "CA1 pyramidal",
    "INT_CA1": "Interneuron (local)",
    "INT_CA2": "CA2 interneuron",
    "INT_CA3": "CA3 interneuron",
    "INT_DG": "DG interneuron",
    "INT_SUB": "Sub interneuron",
    "interneuron": "Interneuron (local)",  # legacy
    "CA1_int": "Interneuron (local)",  # legacy
    "CA2_pyr": "CA2 pyramidal",
    "CA3_pyr": "CA3 pyramidal",
    "DG_granule": "DG granule",
    "Sub_bvc": "Subiculum BVC",
    "MEC_grid": "MEC grid",
    "MEC_hd": "MEC head direction",
    "MEC_speed": "MEC speed",
}


def _panel_class_title(cell_type: str) -> str:
    """Short panel title: region + cell type only."""
    return _PANEL_CLASS_LABELS.get(cell_type, cell_type.replace("_", " "))


def _draw_placeholder(ax, cell_type: str, reason: str = "representation TBD") -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    _bare_image_axes(ax)
    ax.set_facecolor("#f4f5f7")
    ax.text(
        0.5, 0.58, _panel_class_title(cell_type), transform=ax.transAxes,
        ha="center", va="center", fontsize=11, fontweight="bold", color="#3d4450",
    )
    ax.text(
        0.5, 0.38, reason, transform=ax.transAxes,
        ha="center", va="center", fontsize=8, color="#7a8494",
    )


def _draw_place_mosaic_panel(fig, ax, data: SimulationOutputs, unit_ids: list[int], cell_type: str) -> None:
    """Compact rate-map mosaic (best option for place/grid classes)."""
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    beh = data.behavior
    x = beh["x"].to_numpy()
    y = beh["y"].to_numpy()
    show = unit_ids[: min(9, len(unit_ids))]
    maps = {
        uid: _spatial_rate_map(x, y, _unit_rate_trace(data, uid), data.bounds, n_bins=24)
        for uid in show
    }
    vmax = max((float(np.nanmax(rm)) for rm, _ in maps.values()), default=1.0)
    vmax = max(vmax, 1e-6)

    ss = ax.get_subplotspec()
    # Keep the outer axes only as a title anchor (no frame).
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.patch.set_alpha(0.0)
    ax.set_title(_panel_class_title(cell_type), fontsize=10, pad=8)

    n = len(show)
    ncols = 3 if n >= 3 else max(n, 1)
    nrows = int(np.ceil(n / ncols))
    inner = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=ss, wspace=0.05, hspace=0.12)
    mosaic_axes = []
    last_im = None
    for i, uid in enumerate(show):
        a = fig.add_subplot(inner[i // ncols, i % ncols])
        mosaic_axes.append(a)
        rmap, extent = maps[uid]
        last_im = a.imshow(
            rmap.T, origin="lower", aspect="equal", extent=extent,
            cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest",
        )
        _bare_image_axes(a)
    for j in range(n, nrows * ncols):
        a = fig.add_subplot(inner[j // ncols, j % ncols])
        a.axis("off")
    if last_im is not None and mosaic_axes:
        cbar = fig.colorbar(last_im, ax=mosaic_axes, fraction=0.04, pad=0.02)
        cbar.set_label("Hz", fontsize=7)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=6)


def _draw_tuning_heatmap_panel(
    fig, ax, data: SimulationOutputs, unit_ids: list[int], cell_type: str, kind: str,
) -> None:
    """Units × feature rate heatmap, sorted by preferred feature value."""
    curves: list[np.ndarray] = []
    prefs: list[float] = []
    centers_ref: np.ndarray | None = None
    for uid in unit_ids:
        centers, means = _unit_tuning_1d(data, uid, kind)
        if centers.size == 0 or not np.isfinite(means).any():
            continue
        centers_ref = centers
        curves.append(means)
        prefs.append(_preferred_from_curve(centers, means, kind))
    if not curves or centers_ref is None:
        _draw_placeholder(ax, cell_type, "no tuning samples")
        return

    order = np.argsort(np.nan_to_num(prefs, nan=np.inf))
    heat = np.vstack(curves)[order]
    xlabel, _ = _FEATURE_AXIS[kind]
    if kind in ("head_direction", "theta"):
        x0, x1 = 0.0, 2 * np.pi
    else:
        x0, x1 = float(centers_ref[0]), float(centers_ref[-1])
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))
    im = ax.imshow(
        heat, aspect="auto", origin="lower", cmap=cmap,
        extent=[x0, x1, 0, len(order)],
        interpolation="nearest",
    )
    ax.set_title(_panel_class_title(cell_type), fontsize=10, pad=4)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Units (sorted)", fontsize=8)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=2))
    ax.tick_params(labelsize=7)
    if kind in ("head_direction", "theta"):
        ax.set_xlim(x0, x1)
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels(["0", "π", "2π"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Hz", fontsize=7)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(labelsize=6)
    _open_spines(ax)


def _draw_speed_curves_panel(
    fig, ax, data: SimulationOutputs, unit_ids: list[int], cell_type: str,
) -> None:
    """All units: rate vs speed, with population mean."""
    curves: list[np.ndarray] = []
    centers_ref: np.ndarray | None = None
    for uid in unit_ids:
        centers, means = _unit_tuning_1d(data, uid, "speed")
        if centers.size == 0 or not np.isfinite(means).any():
            continue
        centers_ref = centers
        curves.append(means)
        ax.plot(centers, means, color="#9aa3ad", lw=0.8, alpha=0.45, solid_capstyle="round")
    if not curves or centers_ref is None:
        _draw_placeholder(ax, cell_type, "no speed samples")
        return
    stack = np.vstack(curves)
    count = np.sum(np.isfinite(stack), axis=0)
    filled = np.where(np.isfinite(stack), stack, 0.0)
    mean_curve = np.divide(
        filled.sum(axis=0), count, out=np.full(stack.shape[1], np.nan), where=count > 0,
    )
    if np.isfinite(mean_curve).any():
        ax.plot(centers_ref, mean_curve, color="#c0392b", lw=2.0, label="Mean", solid_capstyle="round")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(_panel_class_title(cell_type), fontsize=10, pad=4)
    ax.set_xlabel("Speed (cm/s)", fontsize=8)
    ax.set_ylabel("Rate (Hz)", fontsize=8)
    ax.tick_params(labelsize=7)
    _open_spines(ax)


def _draw_spikes_on_trajectory_panel(
    ax, data: SimulationOutputs, unit_ids: list[int], cell_type: str,
) -> None:
    """Median-rate unit spikes overlaid on the session trajectory."""
    beh = data.behavior
    t = beh["time"].to_numpy()
    x_min, x_max, y_min, y_max = data.bounds
    uid = unit_ids[len(unit_ids) // 2]
    spikes = data.spikes_gt.loc[data.spikes_gt["unit_id"] == uid, "time"].to_numpy()
    ax.plot(beh["x"], beh["y"], color="#d0d5db", lw=0.5, solid_capstyle="round")
    if len(spikes):
        spike_idx = np.clip(np.searchsorted(t, spikes), 0, len(t) - 1)
        ax.scatter(
            beh["x"].iloc[spike_idx], beh["y"].iloc[spike_idx],
            s=5, c="#c0392b", alpha=0.4, linewidths=0,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(_panel_class_title(cell_type), fontsize=10, pad=4)
    ax.set_xlabel("x (cm)", fontsize=8)
    ax.set_ylabel("y (cm)", fontsize=8)
    ax.tick_params(labelsize=7)
    _open_spines(ax)


def plot_population_tuning(data: SimulationOutputs, output_dir: Path) -> list[Path]:
    """One 3×3 page: class tuning panels (mosaic for grid; heatmaps elsewhere)."""
    enable_open_axes()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for old in output_dir.glob("population_tuning_*.png"):
        old.unlink(missing_ok=True)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.5, 12.5))
    gs = GridSpec(
        3, 3, figure=fig,
        hspace=0.42, wspace=0.32,
        left=0.06, right=0.97, top=0.93, bottom=0.05,
    )
    letters = "ABCDEFGHI"
    for idx, ct in enumerate(CELL_CLASS_ORDER):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        unit_ids = _class_unit_ids_by_rate(data, ct)
        kind = _PRIMARY_TUNING.get(ct, "place")
        panel = _FEATURED_TUNING_PANELS.get(ct, "tuning_heatmap")
        if not unit_ids:
            _draw_placeholder(ax, ct, "not in session")
        elif panel == "place_mosaic":
            _draw_place_mosaic_panel(fig, ax, data, unit_ids, ct)
        else:
            _draw_tuning_heatmap_panel(fig, ax, data, unit_ids, ct, kind)
        ax.text(
            -0.02, 1.08, letters[idx], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="right", clip_on=False,
        )

    fig.suptitle(
        "Population tuning by cell class",
        fontsize=14, fontweight="bold", y=0.98,
    )
    path = output_dir / "fig_population_tuning.png"
    return [_finalize_tuning_figure(fig, path)]


def plot_example_units(data: SimulationOutputs, output_dir: Path) -> None:
    """Backward-compatible alias: population tuning + trajectory overview."""
    plot_population_tuning(data, output_dir)
    plot_spikes_on_trajectory_by_class(data, output_dir)


def plot_spikes_on_trajectory_by_class(data: SimulationOutputs, output_dir: Path) -> Path | None:
    """Matching 3×3 page: spikes-on-trajectory for every class present in session."""
    enable_open_axes()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13.5, 12.5))
    gs = GridSpec(
        3, 3, figure=fig,
        hspace=0.40, wspace=0.30,
        left=0.06, right=0.97, top=0.93, bottom=0.05,
    )
    letters = "ABCDEFGHI"
    for idx, ct in enumerate(CELL_CLASS_ORDER):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        unit_ids = _class_unit_ids_by_rate(data, ct)
        if not unit_ids:
            _draw_placeholder(ax, ct, "not in session")
        else:
            _draw_spikes_on_trajectory_panel(ax, data, unit_ids, ct)
        ax.text(
            -0.02, 1.08, letters[idx], transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="right", clip_on=False,
        )

    fig.suptitle(
        "Spikes on trajectory by cell class",
        fontsize=14, fontweight="bold", y=0.98,
    )
    path = output_dir / "fig_spikes_on_trajectory_by_class.png"
    return _finalize_tuning_figure(fig, path)


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
    colors = cell_class_colors(data.cell_class_order)
    ax.bar(
        data.cell_class_order, counts,
        color=[colors[ct] for ct in data.cell_class_order], edgecolor="white",
    )
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
    ax.bar(
        data.cell_class_order, mean_rates,
        color=[colors[ct] for ct in data.cell_class_order], edgecolor="white",
    )
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
    colors = cell_class_colors(data.cell_class_order)
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"].tolist()
        t_centers, pop = _population_activity(
            data.spikes_gt, uids, data.session_duration_s, bin_size,
        )
        t_ds, y_ds = downsample_series(t_centers, pop, MAX_LINE_POINTS)
        ax.plot(t_ds, y_ds, label=ct, color=colors[ct], linewidth=0.8)
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
    """Generate publication neural figures, population tuning, and trajectory overview."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        if png.name.startswith("population_tuning_"):
            png.unlink(missing_ok=True)
            continue
        if any(png.name.startswith(p) for p in legacy_prefixes):
            png.unlink(missing_ok=True)

    from visualization.population_activity_plots import generate_population_activity_plots

    generate_population_activity_plots(data, output_dir, rate_bin_size=rate_bin_size)
    plot_spikes_on_trajectory_by_class(data, output_dir)
    plot_population_tuning(data, output_dir)
