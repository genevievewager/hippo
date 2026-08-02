"""Validation figures for imported / schematic probe trajectories."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.constants import CELL_CLASS_ORDER, FIGURE_DPI, REGION_ORDER

REGION_COLORS = {
    "CA1": "#4C72B0",
    "CA2": "#55A868",
    "CA3": "#C44E52",
    "DG": "#8172B2",
    "dentate_gyrus": "#8172B2",
    "Subiculum": "#CCB974",
    "subiculum": "#CCB974",
    "MEC": "#64B5CD",
    "entorhinal_cortex": "#64B5CD",
    "deep_entorhinal_HATA": "#2A9D8F",
    "HPF_ProS_transition": "#E9C46A",
    "visual_cortex": "#BDBDBD",
}


def _source_label(meta: dict[str, Any] | None, anatomy: pd.DataFrame) -> str:
    if meta and meta.get("anatomy_source"):
        src = str(meta["anatomy_source"])
        if meta.get("schematic_fallback_used") or meta.get("used_fallback"):
            return f"{src} (fallback)"
        return src
    if meta and meta.get("source"):
        src = str(meta["source"])
        if meta.get("used_fallback"):
            return f"{src} (fallback)"
        return src
    if "trajectory_source" in anatomy.columns and len(anatomy):
        return str(anatomy["trajectory_source"].iloc[0])
    return "unknown"


def plot_probe_trajectory_regions(
    anatomy: pd.DataFrame,
    output_path: Path,
    *,
    units: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Vertical probe depth axis with region/layer bands and channel ranges."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 11))
    if anatomy is None or anatomy.empty:
        ax.text(0.5, 0.5, "No anatomy regions", ha="center", va="center")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    z_max = float(anatomy["depth_end_um"].max())
    for _, row in anatomy.iterrows():
        z0 = float(row["depth_start_um"])
        z1 = float(row["depth_end_um"])
        region = str(row["region"])
        layer = str(row.get("layer") or row.get("layer_or_area") or "")
        included = row.get("include_in_hippocampal_simulation", True)
        if isinstance(included, str):
            included = included.lower() not in {"false", "0", "no"}
        color = REGION_COLORS.get(region, "#9E9E9E")
        alpha = 0.35 if not included else 0.65
        ax.barh(
            (z0 + z1) / 2.0,
            width=1.0,
            height=max(z1 - z0, 1.0),
            color=color,
            alpha=alpha,
            edgecolor="black",
            linewidth=0.6,
            hatch="//" if not included else None,
        )
        ch0 = row.get("channel_start", "")
        ch1 = row.get("channel_end", "")
        cell_types = str(row.get("cell_types", "") or "")
        excl = "" if included else " [excluded]"
        label = f"{region} / {layer}{excl}\nch {ch0}–{ch1}"
        if cell_types and included:
            label += f"\n{cell_types}"
        ax.text(0.5, (z0 + z1) / 2.0, label, ha="center", va="center", fontsize=6.5)

    if units is not None and len(units) and "depth_um" in units.columns:
        rng = np.random.default_rng(0)
        xs = 1.15 + rng.uniform(-0.08, 0.08, size=len(units))
        ax.scatter(xs, units["depth_um"], s=8, c="k", alpha=0.35, zorder=5)
        ax.set_xlim(0, 1.4)
    else:
        ax.set_xlim(0, 1.0)

    ax.set_ylim(z_max * 1.02, -0.02 * z_max)
    ax.set_xticks([])
    ax.set_ylabel("Probe depth (µm), dorsal → ventral")
    src = _source_label(meta, anatomy)
    ax.set_title(
        "Probe trajectory region/layer bands\n"
        f"source: {src}\n"
        "Trajectory-informed simulation geometry; approximate screenshot-derived region depths."
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    if output_path.suffix.lower() == ".png":
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_unit_count_by_imported_region(
    units: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    if units is None or units.empty or "region" not in units.columns:
        ax.text(0.5, 0.5, "No units", ha="center", va="center")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path

    region_order = list(REGION_ORDER) + [
        "visual_cortex", "HPF_ProS_transition", "subiculum",
        "dentate_gyrus", "entorhinal_cortex", "deep_entorhinal_HATA",
    ]
    present = list(units["region"].unique())
    ordered = [r for r in region_order if r in present] + [
        r for r in present if r not in region_order
    ]
    counts = (
        units.groupby(["region", "cell_type"]).size()
        .unstack(fill_value=0)
        .reindex(
            index=ordered,
            columns=[c for c in CELL_CLASS_ORDER if c in set(units["cell_type"])],
            fill_value=0,
        )
    )
    bottom = np.zeros(len(counts))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(counts.columns), 1)))
    for color, ct in zip(colors, counts.columns):
        ax.bar(counts.index.astype(str), counts[ct], bottom=bottom, label=ct, color=color)
        bottom += counts[ct].to_numpy()
    ax.set_ylabel("Number of units")
    ax.set_xlabel("Region")
    ax.set_title("Unit count by region")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_unit_count_by_cell_type(
    units: pd.DataFrame,
    output_path: Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    if units is None or units.empty or "cell_type" not in units.columns:
        ax.text(0.5, 0.5, "No units", ha="center", va="center")
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        return output_path
    order = [c for c in CELL_CLASS_ORDER if c in set(units["cell_type"])]
    extra = [c for c in units["cell_type"].unique() if c not in order]
    counts = units["cell_type"].value_counts().reindex(order + extra, fill_value=0)
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0", edgecolor="white")
    ax.set_ylabel("Number of units")
    ax.set_xlabel("Cell type")
    ax.set_title("Unit count by cell type")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_channel_region_map(
    anatomy: pd.DataFrame,
    output_path: Path,
    *,
    n_channels: int = 384,
    meta: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channel_region = np.array(["unassigned"] * n_channels, dtype=object)
    for _, row in anatomy.iterrows():
        ch0, ch1 = row.get("channel_start"), row.get("channel_end")
        if pd.isna(ch0) or pd.isna(ch1) or int(ch0) <= 0:
            continue
        for ch in range(int(ch0), int(ch1) + 1):
            if 1 <= ch <= n_channels:
                channel_region[ch - 1] = str(row["region"])

    fig, ax = plt.subplots(figsize=(10, 3))
    regions_present = []
    preferred = list(REGION_ORDER) + [
        "visual_cortex", "HPF_ProS_transition", "subiculum",
        "dentate_gyrus", "entorhinal_cortex", "deep_entorhinal_HATA", "unassigned",
    ]
    for r in preferred:
        if r in set(channel_region.tolist()):
            regions_present.append(r)
    for r in set(channel_region.tolist()):
        if r not in regions_present:
            regions_present.append(r)
    color_lookup = {r: REGION_COLORS.get(r, "#BDBDBD") for r in regions_present}
    for ch in range(n_channels):
        ax.axvspan(
            ch + 0.5, ch + 1.5,
            color=color_lookup.get(channel_region[ch], "#BDBDBD"), linewidth=0,
        )
    ax.set_xlim(0.5, n_channels + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Channel (1-based)")
    src = _source_label(meta, anatomy)
    ax.set_title(f"Channel → region map (source: {src})")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color_lookup[r]) for r in regions_present
    ]
    ax.legend(
        handles, regions_present, loc="upper right", fontsize=7,
        ncol=min(4, len(regions_present)),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_trajectory_validation_figures(
    anatomy: pd.DataFrame,
    units: pd.DataFrame,
    figures_dir: Path,
    *,
    meta: dict[str, Any] | None = None,
    n_channels: int = 384,
) -> dict[str, Path]:
    """Write compact ``figures/trajectory/fig_probe_trajectory.png``."""
    from visualization.publication_trajectory_plots import (
        generate_publication_trajectory_figures,
    )

    path = generate_publication_trajectory_figures(
        anatomy,
        units,
        figures_dir,
        n_channels=n_channels,
        meta=meta,
    )
    return {"fig_probe_trajectory": path}
