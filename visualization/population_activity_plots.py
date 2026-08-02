"""Publication-style hippocampal population activity figures (seaborn).

Produces a small set of multi-panel figures instead of many single-panel PNGs.
Uses ground-truth ``rates.npy`` when available (overlays + feedforward).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d

from visualization.constants import (
    CELL_CLASS_ORDER,
    CELL_TYPE_TO_CIRCUIT_NODE,
    FIGURE_DPI,
    REGION_TO_CIRCUIT_NODE,
    cell_class_colors,
    circuit_node_colors,
    circuit_node_order_for_present,
    region_colors,
)
from visualization.load_outputs import SimulationOutputs, downsample_series
from visualization.neural_plots import _population_activity
from visualization.publication_circuit_plots import plot_fig_circuit_feedforward
from visualization.publication_style import (
    apply_publication_theme,
    legend_outside,
    panel_label,
    save_pub_figure,
)

apply_publication_theme()


def _smooth(y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if y.size == 0:
        return y
    return gaussian_filter1d(y.astype(float), sigma=sigma)


def _unit_index_map(units: pd.DataFrame) -> dict[int, int]:
    return {int(uid): i for i, uid in enumerate(units["unit_id"].to_numpy())}


def _circuit_node_for_unit(row: pd.Series) -> str:
    ct = str(row.get("cell_type", ""))
    if ct in CELL_TYPE_TO_CIRCUIT_NODE:
        return CELL_TYPE_TO_CIRCUIT_NODE[ct]
    region = str(row.get("region", ""))
    return REGION_TO_CIRCUIT_NODE.get(region, "OTHER")


def _annotate_units(data: SimulationOutputs) -> pd.DataFrame:
    units = data.units.copy()
    units["circuit_node"] = units.apply(_circuit_node_for_unit, axis=1)
    units["mean_rate_hz"] = units["unit_id"].map(data.unit_mean_rates_gt).fillna(0.0)
    return units


def _mean_rate_trace(
    data: SimulationOutputs,
    unit_ids: list[int],
    bin_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    if data.has_ground_truth_rates:
        assert data.rates_hz is not None and data.rate_times_s is not None
        idx_map = _unit_index_map(data.units)
        cols = [idx_map[int(u)] for u in unit_ids if int(u) in idx_map]
        t = data.rate_times_s
        if not cols:
            return t, np.zeros_like(t, dtype=float)
        return t, data.rates_hz[cols].mean(axis=0)
    return _population_activity(
        data.spikes_gt, unit_ids, data.session_duration_s, bin_size,
    )


def _long_traces_by_key(
    data: SimulationOutputs,
    key: str,
    order: list[str],
    bin_size: float,
    max_points: int = 1200,
    units: pd.DataFrame | None = None,
) -> pd.DataFrame:
    units_df = data.units if units is None else units
    rows = []
    present = [k for k in order if k in set(units_df[key].astype(str))]
    for label in present:
        uids = units_df.loc[units_df[key].astype(str) == label, "unit_id"].tolist()
        t, y = _mean_rate_trace(data, uids, bin_size)
        t, y = downsample_series(t, _smooth(y), max_points)
        for ti, yi in zip(t, y):
            rows.append({"time_s": float(ti), "rate_hz": float(yi), key: label})
    return pd.DataFrame(rows)


def plot_fig_population_activity(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> Path:
    """Combined cell-class, circuit-node, and regional population activity.

    (A) Mean rate traces by cell class (per-unit mean).
    (B) Mean rate traces by circuit node (per-unit mean).
    (C) Mean rate traces by anatomical region (per-unit mean).

    Unit counts live in ``fig_probe_trajectory`` panel D.
    """
    units = _annotate_units(data)
    classes = [c for c in data.cell_class_order if c in set(units["cell_type"].astype(str))]
    nodes = circuit_node_order_for_present(set(units["circuit_node"]))
    regions = [r for r in data.region_order if r in set(units["region"].astype(str))]
    class_palette = cell_class_colors(classes)
    node_color = circuit_node_colors(nodes)
    region_palette = region_colors(regions)

    fig = plt.figure(figsize=(12.2, 9.6))
    gs = GridSpec(
        3, 1, figure=fig,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.38,
    )

    # A — cell class mean-rate traces
    ax_a = fig.add_subplot(gs[0, 0])
    trace_df = _long_traces_by_key(data, "cell_type", classes, bin_size, max_points=1000)
    if not trace_df.empty:
        sns.lineplot(
            data=trace_df, x="time_s", y="rate_hz", hue="cell_type",
            hue_order=classes, palette=class_palette, lw=1.0, ax=ax_a, legend=True,
        )
        legend_outside(
            ax_a, title="Cell class", fontsize=9, ncol=1,
        )
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Mean rate (Hz)")
    sns.despine(ax=ax_a)
    panel_label(ax_a, "A")

    # B — circuit-node mean-rate traces
    ax_b = fig.add_subplot(gs[1, 0])
    node_df = _long_traces_by_key(
        data, "circuit_node", nodes, bin_size, max_points=1000, units=units,
    )
    if not node_df.empty:
        sns.lineplot(
            data=node_df, x="time_s", y="rate_hz", hue="circuit_node",
            hue_order=nodes, palette=node_color, lw=1.1, ax=ax_b, legend=True,
        )
        legend_outside(
            ax_b, title="Circuit node", fontsize=9, ncol=1,
        )
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Mean rate (Hz)")
    sns.despine(ax=ax_b)
    panel_label(ax_b, "B")

    # C — anatomical region mean-rate traces (legend matches traces)
    ax_c = fig.add_subplot(gs[2, 0])
    region_df = _long_traces_by_key(data, "region", regions, bin_size, max_points=1000)
    if not region_df.empty:
        sns.lineplot(
            data=region_df, x="time_s", y="rate_hz", hue="region",
            hue_order=regions, palette=region_palette, lw=1.1, ax=ax_c, legend=True,
        )
        legend_outside(
            ax_c, title="Region", fontsize=9, ncol=1,
        )
    ax_c.set_xlabel("Time (s)")
    ax_c.set_ylabel("Mean rate (Hz)")
    sns.despine(ax=ax_c)
    panel_label(ax_c, "C")

    path = save_pub_figure(
        fig, output_dir / "fig_population_activity.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.05, 0.82, 0.97),
    )
    # Former split stems — drop stale PNGs so the PDF has one page.
    for stem in ("fig_cell_class_population", "fig_circuit_population"):
        (Path(output_dir) / f"{stem}.png").unlink(missing_ok=True)
    return path


def plot_fig_cell_class_population(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> Path:
    """Deprecated: content lives in ``fig_population_activity``."""
    return plot_fig_population_activity(data, output_dir, bin_size)


def plot_fig_circuit_population(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> Path:
    """Deprecated: content lives in ``fig_population_activity``."""
    return plot_fig_population_activity(data, output_dir, bin_size)


def plot_fig_population_structure(
    data: SimulationOutputs, output_dir: Path, max_units: int = 180,
) -> Path:
    """Compressed ground-truth spike raster ordered by circuit node.

    Region × cell-class unit counts live in ``fig_probe_trajectory`` panel D.
    GT vs sorted spike counts live in ``fig_sorting_summary`` (not repeated here).
    """
    units = _annotate_units(data)
    nodes = circuit_node_order_for_present(set(units["circuit_node"]))

    units_r = units.copy()
    node_rank = {n: i for i, n in enumerate(nodes)}
    class_rank = {c: i for i, c in enumerate(CELL_CLASS_ORDER)}
    units_r["node_rank"] = units_r["circuit_node"].map(node_rank).fillna(99)
    units_r["class_rank"] = units_r["cell_type"].map(class_rank).fillna(99)
    units_r = units_r.sort_values(
        ["node_rank", "class_rank", "mean_rate_hz"], ascending=[True, True, False]
    )
    if len(units_r) > max_units:
        keep = []
        per = max(8, max_units // max(len(nodes), 1))
        for _, grp in units_r.groupby("circuit_node", sort=False):
            keep.append(grp.head(per))
        units_r = pd.concat(keep, ignore_index=True)

    fig, ax = plt.subplots(figsize=(14.0, 9.0))
    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.08)

    spike_groups = data.spikes_gt.groupby("unit_id")["time"].apply(list).to_dict()
    spike_rows = [
        spike_groups.get(int(uid), [])
        for uid in units_r["unit_id"].tolist()
    ]
    ax.eventplot(
        spike_rows,
        orientation="horizontal",
        colors="black",
        linewidths=0.28,
        linelengths=0.35,
        lineoffsets=np.arange(len(spike_rows)),
    )
    y = 0
    for node in nodes:
        n_here = int((units_r["circuit_node"] == node).sum())
        if n_here == 0:
            continue
        ax.axhline(y - 0.5, color="crimson", lw=0.5, alpha=0.45)
        ax.text(
            -0.02 * data.session_duration_s, y + n_here / 2, node,
            fontsize=9, va="center", ha="right",
        )
        y += n_here
    ax.set_xlim(0, data.session_duration_s)
    ax.set_ylim(len(units_r) - 0.5, -0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("")
    ax.set_yticks([])
    sns.despine(ax=ax)

    path = save_pub_figure(
        fig, output_dir / "fig_population_structure.png", dpi=FIGURE_DPI,
        pad_inches=0.25,
        adjust=False,
    )
    (Path(output_dir) / "fig_spike_raster_summary.png").unlink(missing_ok=True)
    return path


def plot_fig_spike_raster_summary(
    data: SimulationOutputs, output_dir: Path, max_units: int = 180,
) -> Path:
    """Deprecated: raster lives in ``fig_population_structure``."""
    return plot_fig_population_structure(data, output_dir, max_units=max_units)


def generate_population_activity_plots(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> list[Path]:
    """Write compressed publication figures; return output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_fig_circuit_feedforward(data, output_dir),
        plot_fig_population_activity(data, output_dir, rate_bin_size),
        plot_fig_population_structure(data, output_dir),
    ]
    for stem in (
        "fig_spike_raster_summary",
        "fig_cell_class_population",
        "fig_circuit_population",
    ):
        (output_dir / f"{stem}.png").unlink(missing_ok=True)
    return paths
