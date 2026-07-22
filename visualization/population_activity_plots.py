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
    CIRCUIT_NODE_ORDER,
    FIGURE_DPI,
    MAX_LINE_POINTS,
    REGION_ORDER,
    REGION_TO_CIRCUIT_NODE,
)
from visualization.load_outputs import SimulationOutputs, downsample_series
from visualization.neural_plots import _population_activity
from visualization.publication_style import (
    figure_legend_below,
    legend_below,
    legend_outside,
    panel_label,
    save_pub_figure,
)

sns.set_theme(style="ticks", context="paper", font_scale=1.0)


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
) -> pd.DataFrame:
    rows = []
    present = [k for k in order if k in set(data.units[key].astype(str))]
    for label in present:
        uids = data.units.loc[data.units[key].astype(str) == label, "unit_id"].tolist()
        t, y = _mean_rate_trace(data, uids, bin_size)
        t, y = downsample_series(t, _smooth(y), max_points)
        for ti, yi in zip(t, y):
            rows.append({"time_s": float(ti), "rate_hz": float(yi), key: label})
    return pd.DataFrame(rows)


def plot_fig_circuit_population(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> Path:
    """Fig: circuit population activity (publication panel)."""
    units = _annotate_units(data)
    nodes = [n for n in CIRCUIT_NODE_ORDER if n in set(units["circuit_node"])]
    palette = sns.color_palette("deep", n_colors=max(len(nodes), 1))
    node_color = {n: c for n, c in zip(nodes, palette)}

    fig = plt.figure(figsize=(12.2, 8.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.15, 1.0], hspace=0.38, wspace=0.48)

    # A — overlay (legend outside axes)
    ax_a = fig.add_subplot(gs[0, 0])
    for node in nodes:
        uids = units.loc[units["circuit_node"] == node, "unit_id"].tolist()
        t, y = _mean_rate_trace(data, uids, bin_size)
        t, y = downsample_series(t, _smooth(y), MAX_LINE_POINTS)
        ax_a.plot(t, y, label=node, color=node_color[node], lw=1.2)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Mean rate (Hz)")
    # Figure-level legend below — never overlaps traces
    handles, labels = ax_a.get_legend_handles_labels()
    if ax_a.get_legend() is not None:
        ax_a.get_legend().remove()
    sns.despine(ax=ax_a)
    panel_label(ax_a, "A")

    # B — stacked ridges (compressed)
    ax_b = fig.add_subplot(gs[0, 1])
    offset = 0.0
    yticks, ylabels = [], []
    for node in nodes:
        uids = units.loc[units["circuit_node"] == node, "unit_id"].tolist()
        t, y = _mean_rate_trace(data, uids, bin_size)
        t, y = downsample_series(t, _smooth(y), 900)
        scale = max(float(np.percentile(y, 95)), 1e-6)
        yn = y / scale
        ax_b.fill_between(t, offset, offset + yn, color=node_color[node], alpha=0.55, lw=0)
        ax_b.plot(t, offset + yn, color=node_color[node], lw=0.7)
        yticks.append(offset + 0.45)
        ylabels.append(node)
        offset += 1.15
    ax_b.set_yticks(yticks)
    ax_b.set_yticklabels(ylabels, fontsize=8)
    ax_b.set_xlabel("Time (s)")
    sns.despine(ax=ax_b, left=False)
    panel_label(ax_b, "B")

    # C — mean rate by node
    ax_c = fig.add_subplot(gs[1, 0])
    node_df = (
        units.groupby("circuit_node", as_index=False)["mean_rate_hz"]
        .mean()
    )
    node_df["circuit_node"] = pd.Categorical(node_df["circuit_node"], nodes, ordered=True)
    node_df = node_df.sort_values("circuit_node")
    sns.barplot(
        data=node_df, x="circuit_node", y="mean_rate_hz",
        hue="circuit_node", palette=node_color, legend=False, ax=ax_c,
    )
    ax_c.set_xlabel("Circuit node")
    ax_c.set_ylabel("Mean rate (Hz)")
    sns.despine(ax=ax_c)
    panel_label(ax_c, "C")

    # D — unit counts × mean rate; color by node only (style legend is too crowded)
    ax_d = fig.add_subplot(gs[1, 1])
    summary = (
        units.groupby(["circuit_node", "cell_type"], as_index=False)
        .agg(n_units=("unit_id", "count"), mean_rate_hz=("mean_rate_hz", "mean"))
    )
    summary["circuit_node"] = pd.Categorical(summary["circuit_node"], nodes, ordered=True)
    sns.scatterplot(
        data=summary, x="n_units", y="mean_rate_hz",
        hue="circuit_node", palette=node_color, s=55, ax=ax_d, legend=False,
    )
    ax_d.set_xlabel("Units")
    ax_d.set_ylabel("Mean rate (Hz)")
    sns.despine(ax=ax_d)
    panel_label(ax_d, "D")

    src = "rates.npy + feedforward" if data.has_ground_truth_rates else "binned spikes"
    if handles:
        figure_legend_below(fig, handles, labels, ncol=min(7, len(handles)), fontsize=7, y=0.02)
    return save_pub_figure(
        fig, output_dir / "fig_circuit_population.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.12, 0.98, 0.92),
    )


def plot_fig_cell_class_population(
    data: SimulationOutputs, output_dir: Path, bin_size: float = 0.250,
) -> Path:
    """Fig: cell-class / region population activity (publication panel)."""
    units = _annotate_units(data)
    classes = data.cell_class_order
    regions = data.region_order

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

    # A — cell class traces
    ax_a = fig.add_subplot(gs[0, :])
    trace_df = _long_traces_by_key(data, "cell_type", classes, bin_size, max_points=1000)
    if not trace_df.empty:
        sns.lineplot(
            data=trace_df, x="time_s", y="rate_hz", hue="cell_type",
            hue_order=classes, palette="tab10", lw=1.0, ax=ax_a,
        )
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Mean rate (Hz)")
    legend_outside(ax_a, fontsize=6, ncol=1, bbox=(1.01, 1.0))
    sns.despine(ax=ax_a)
    panel_label(ax_a, "A")

    # B — region traces
    ax_b = fig.add_subplot(gs[1, 0])
    region_df = _long_traces_by_key(data, "region", regions, bin_size, max_points=1000)
    if not region_df.empty:
        sns.lineplot(
            data=region_df, x="time_s", y="rate_hz", hue="region",
            hue_order=regions, palette="Set2", lw=1.1, ax=ax_b,
        )
    ax_b.set_xlabel("Time (s)")
    ax_b.set_ylabel("Mean rate (Hz)")
    legend_outside(ax_b, fontsize=6)
    sns.despine(ax=ax_b)
    panel_label(ax_b, "B")

    # C — violin of per-unit mean rates
    ax_c = fig.add_subplot(gs[1, 1])
    plot_df = units.copy()
    plot_df["cell_type"] = pd.Categorical(plot_df["cell_type"], classes, ordered=True)
    sns.violinplot(
        data=plot_df, x="cell_type", y="mean_rate_hz",
        hue="cell_type", palette="tab10", legend=False,
        cut=0, inner="quartile", density_norm="width", ax=ax_c,
    )
    ax_c.set_xlabel("Cell class")
    ax_c.set_ylabel("Unit mean rate (Hz)")
    ax_c.tick_params(axis="x", rotation=35, labelsize=7)
    sns.despine(ax=ax_c)
    panel_label(ax_c, "C")

    src = "rates.npy" if data.has_ground_truth_rates else "spikes"
    return save_pub_figure(
        fig, output_dir / "fig_cell_class_population.png", dpi=FIGURE_DPI,
        rect=(0.02, 0.08, 0.98, 0.94),
    )


def plot_fig_population_structure(
    data: SimulationOutputs, output_dir: Path,
) -> Path:
    """Fig: anatomy / composition / rate heatmap (publication panel)."""
    units = _annotate_units(data)

    fig = plt.figure(figsize=(10.5, 7.0))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.28)

    # A — unit counts by region × cell class
    ax_a = fig.add_subplot(gs[0, 0])
    ct = pd.crosstab(units["region"], units["cell_type"])
    ct = ct.reindex(
        index=[r for r in REGION_ORDER if r in ct.index],
        columns=[c for c in CELL_CLASS_ORDER if c in ct.columns],
    ).fillna(0)
    sns.heatmap(ct, annot=True, fmt=".0f", cmap="Blues", cbar=False, ax=ax_a)
    ax_a.set_xlabel("Cell class")
    ax_a.set_ylabel("Region")
    ax_a.tick_params(labelsize=7)
    panel_label(ax_a, "A")

    # B — mean rate by region × cell class
    ax_b = fig.add_subplot(gs[0, 1])
    pivot = units.pivot_table(
        index="region", columns="cell_type", values="mean_rate_hz", aggfunc="mean",
    )
    pivot = pivot.reindex(
        index=[r for r in REGION_ORDER if r in pivot.index],
        columns=[c for c in CELL_CLASS_ORDER if c in pivot.columns],
    )
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="mako", ax=ax_b)
    ax_b.set_xlabel("Cell class")
    ax_b.set_ylabel("Region")
    ax_b.tick_params(labelsize=7)
    panel_label(ax_b, "B")

    # C — rate heatmap (subset of units)
    ax_c = fig.add_subplot(gs[1, :])
    if data.has_ground_truth_rates:
        assert data.rates_hz is not None and data.rate_times_s is not None
        node_rank = {n: i for i, n in enumerate(CIRCUIT_NODE_ORDER)}
        class_rank = {c: i for i, c in enumerate(CELL_CLASS_ORDER)}
        units_h = _annotate_units(data)
        units_h["node_rank"] = units_h["circuit_node"].map(node_rank).fillna(99)
        units_h["class_rank"] = units_h["cell_type"].map(class_rank).fillna(99)
        units_h = units_h.sort_values(
            ["node_rank", "class_rank", "mean_rate_hz"], ascending=[True, True, False]
        )
        keep = []
        for _, grp in units_h.groupby("circuit_node", sort=False):
            keep.append(grp.head(12))
        units_h = pd.concat(keep, ignore_index=True)
        idx_map = _unit_index_map(data.units)
        rows = [idx_map[int(u)] for u in units_h["unit_id"] if int(u) in idx_map]
        mat = data.rates_hz[rows]
        t = data.rate_times_s
        if len(t) > 800:
            step = int(np.ceil(len(t) / 800))
            mat = mat[:, ::step]
            t = t[::step]
        sns.heatmap(
            mat, cmap="viridis", cbar_kws={"label": "Rate (Hz)", "shrink": 0.8},
            xticklabels=False, yticklabels=False, ax=ax_c,
        )
        ax_c.set_xlabel("Time →")
        ax_c.set_ylabel("Units (by circuit node)")
        y = 0
        for node in CIRCUIT_NODE_ORDER:
            n_here = int((units_h["circuit_node"] == node).sum())
            if n_here == 0:
                continue
            ax_c.text(2, y + n_here / 2, node, color="white", fontsize=7, va="center")
            y += n_here
            if y < len(rows):
                ax_c.axhline(y, color="white", lw=0.6, alpha=0.55)
    else:
        ax_c.text(0.5, 0.5, "rates.npy not available", ha="center", va="center")
        ax_c.set_axis_off()
    panel_label(ax_c, "C")

    return save_pub_figure(
        fig, output_dir / "fig_population_structure.png", dpi=FIGURE_DPI,
        rect=(0.10, 0.08, 0.98, 0.94),
    )


def plot_fig_spike_raster_summary(
    data: SimulationOutputs, output_dir: Path, max_units: int = 150,
) -> Path:
    """Fig: compressed spike raster + sorting comparison (publication panel)."""
    units = _annotate_units(data)
    node_rank = {n: i for i, n in enumerate(CIRCUIT_NODE_ORDER)}
    class_rank = {c: i for i, c in enumerate(CELL_CLASS_ORDER)}
    units["node_rank"] = units["circuit_node"].map(node_rank).fillna(99)
    units["class_rank"] = units["cell_type"].map(class_rank).fillna(99)
    units = units.sort_values(
        ["node_rank", "class_rank", "mean_rate_hz"], ascending=[True, True, False]
    )
    if len(units) > max_units:
        keep = []
        per = max(10, max_units // max(len(CIRCUIT_NODE_ORDER), 1))
        for _, grp in units.groupby("circuit_node", sort=False):
            keep.append(grp.head(per))
        units = pd.concat(keep, ignore_index=True)

    fig = plt.figure(figsize=(10.5, 6.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1.0], wspace=0.28)

    # A — raster
    ax_a = fig.add_subplot(gs[0, 0])
    spike_groups = data.spikes_gt.groupby("unit_id")["time"].apply(list).to_dict()
    for y, uid in enumerate(units["unit_id"].tolist()):
        times = spike_groups.get(int(uid), [])
        if times:
            ax_a.scatter(
                times, np.full(len(times), y),
                s=0.25, c="black", marker="|", linewidths=0.15,
            )
    y = 0
    for node in CIRCUIT_NODE_ORDER:
        n_here = int((units["circuit_node"] == node).sum())
        if n_here == 0:
            continue
        ax_a.axhline(y - 0.5, color="crimson", lw=0.5, alpha=0.45)
        ax_a.text(
            -0.02 * data.session_duration_s, y + n_here / 2, node,
            fontsize=7, va="center", ha="right",
        )
        y += n_here
    ax_a.set_xlim(0, data.session_duration_s)
    ax_a.set_ylim(len(units), -1)
    ax_a.set_xlabel("Time (s)")
    ax_a.set_ylabel("Units")
    ax_a.set_yticks([])
    sns.despine(ax=ax_a)
    panel_label(ax_a, "A")

    # B — GT vs sorted spike counts
    ax_b = fig.add_subplot(gs[0, 1])
    rows = []
    for ct in data.cell_class_order:
        uids = data.units.loc[data.units["cell_type"] == ct, "unit_id"]
        gt_n = int(data.spikes_gt[data.spikes_gt["unit_id"].isin(uids)].shape[0])
        so_n = int(data.spikes_sorted[data.spikes_sorted["unit_id"].isin(uids)].shape[0])
        rows.append({"cell_type": ct, "source": "ground_truth", "spikes": gt_n})
        rows.append({"cell_type": ct, "source": "sorted", "spikes": so_n})
    cmp = pd.DataFrame(rows)
    sns.barplot(
        data=cmp, x="spikes", y="cell_type", hue="source",
        palette={"ground_truth": "#4C72B0", "sorted": "#DD8452"},
        ax=ax_b,
    )
    ax_b.set_xlabel("Total spikes")
    ax_b.set_ylabel("")
    legend_outside(ax_b, fontsize=7, ncol=1, bbox=(1.02, 1.0))
    sns.despine(ax=ax_b)
    panel_label(ax_b, "B")

    return save_pub_figure(
        fig, output_dir / "fig_spike_raster_summary.png", dpi=FIGURE_DPI,
        rect=(0.02, 0.08, 0.98, 0.94),
    )


def generate_population_activity_plots(
    data: SimulationOutputs, output_dir: Path, rate_bin_size: float = 0.250,
) -> list[Path]:
    """Write compressed publication figures; return output paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_fig_circuit_population(data, output_dir, rate_bin_size),
        plot_fig_cell_class_population(data, output_dir, rate_bin_size),
        plot_fig_population_structure(data, output_dir),
        plot_fig_spike_raster_summary(data, output_dir),
    ]
    return paths
