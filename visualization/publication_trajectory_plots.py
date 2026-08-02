"""Publication multi-panel probe-trajectory figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from visualization.constants import (
    CELL_CLASS_ORDER,
    FIGURE_DPI,
    REGION_ORDER,
    analysis_cell_class,
    cell_class_colors,
    cell_class_order_for_counts,
)
from visualization.publication_style import (
    apply_publication_theme,
    enable_open_axes,
    panel_label,
    strip_box_frames,
    style_figure_axes,
)

apply_publication_theme()

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

_LEGACY_TRAJECTORY_STEMS = (
    "probe_trajectory_regions",
    "probe_areas_nte_style",
    "probe_trajectory_3d",
    "channel_region_map",
    "unit_count_by_region",
    "unit_count_by_cell_type",
    "unit_count_by_imported_region",
    "unit_count_by_region_and_cell_class",
    "unit_depth_by_cell_class",
    "probe_region_geometry",
)


def _as_bool(val: Any, default: bool = True) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() not in {"false", "0", "no", "n"}


def plot_fig_probe_trajectory(
    anatomy: pd.DataFrame,
    units: pd.DataFrame | None,
    output_dir: Path,
    *,
    n_channels: int = 384,
    meta: dict[str, Any] | None = None,
) -> Path:
    """One page: probe anatomy + units captured along the trajectory.

    (A) Region bands along probe depth.
    (B) Channel index mapped to regions.
    (C) Unit positions in probe-local 3D (legend beside the panel).
    (D) Region × cell-class unit-count heatmap (cells picked up by the probe).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(17.5, 10.2))
    # Top: wide A/B + 3D + legend. Bottom: full-width count heatmap.
    outer = GridSpec(
        2, 1, figure=fig,
        height_ratios=[1.45, 1.0],
        hspace=0.32,
        left=0.06, right=0.97, top=0.94, bottom=0.12,
    )
    top = outer[0].subgridspec(
        1, 4,
        width_ratios=[1.25, 1.25, 3.4, 1.05],
        wspace=0.35,
    )

    # A — depth strip
    ax_a = fig.add_subplot(top[0, 0])
    if anatomy is None or anatomy.empty:
        ax_a.text(0.5, 0.5, "No anatomy", ha="center", va="center")
    else:
        z_max = float(anatomy["depth_end_um"].max())
        for _, row in anatomy.iterrows():
            z0 = float(row["depth_start_um"])
            z1 = float(row["depth_end_um"])
            region = str(row["region"])
            included = _as_bool(row.get("include_in_hippocampal_simulation", True))
            color = REGION_COLORS.get(region, "#9E9E9E")
            ax_a.barh(
                (z0 + z1) / 2.0,
                width=1.0,
                height=max(z1 - z0, 1.0),
                color=color,
                alpha=0.35 if not included else 0.7,
                edgecolor="black",
                linewidth=0.5,
                hatch="//" if not included else None,
            )
            layer = str(row.get("layer") or row.get("layer_or_area") or "")
            label = f"{region}"
            if layer and layer != region:
                label += f"\n{layer}"
            ax_a.text(0.5, (z0 + z1) / 2.0, label, ha="center", va="center", fontsize=9)
        ax_a.set_ylim(z_max, 0)
        ax_a.set_xlim(0, 1.0)
        ax_a.set_xticks([])
        ax_a.set_ylabel("Probe depth (µm)")
    panel_label(ax_a, "A", x=-0.08, y=1.02)

    # B — channel map (crop empty tip so labels fill the strip)
    ax_b = fig.add_subplot(top[0, 1])
    if anatomy is None or anatomy.empty or "channel_start" not in anatomy.columns:
        ax_b.text(0.5, 0.5, "No channel map", ha="center", va="center")
    else:
        ch_hi = 0
        for _, row in anatomy.iterrows():
            ch0 = row.get("channel_start")
            ch1 = row.get("channel_end")
            if pd.isna(ch0) or pd.isna(ch1):
                continue
            ch0, ch1 = int(ch0), int(ch1)
            ch_hi = max(ch_hi, ch1)
            region = str(row["region"])
            included = _as_bool(row.get("include_in_hippocampal_simulation", True))
            color = REGION_COLORS.get(region, "#9E9E9E")
            ax_b.barh(
                (ch0 + ch1) / 2.0,
                width=1.0,
                height=max(ch1 - ch0 + 1, 1),
                color=color,
                alpha=0.35 if not included else 0.7,
                edgecolor="black",
                linewidth=0.4,
            )
            ax_b.text(
                0.5, (ch0 + ch1) / 2.0, f"{region}\nch {ch0}–{ch1}",
                ha="center", va="center", fontsize=8,
            )
        ax_b.set_ylim(max(ch_hi + 2, 1), 0)
        ax_b.set_xlim(0, 1.0)
        ax_b.set_xticks([])
        ax_b.set_ylabel("Channel index")
    panel_label(ax_b, "B", x=-0.08, y=1.02)

    # C — unit positions in probe-local 3D
    ax_c = fig.add_subplot(top[0, 2], projection="3d")
    class_handles: list = []
    class_labels: list[str] = []
    if units is None or units.empty or "depth_um" not in units.columns:
        ax_c.text2D(0.5, 0.5, "No depth_um", transform=ax_c.transAxes, ha="center", va="center")
    else:
        present_classes = [
            c for c in CELL_CLASS_ORDER if c in set(units["cell_type"].astype(str))
        ]
        extra_classes = [
            c for c in units["cell_type"].astype(str).unique() if c not in present_classes
        ]
        class_order = present_classes + extra_classes
        class_palette = cell_class_colors(class_order)

        depth_mm = units["depth_um"].to_numpy(dtype=float) / 1000.0
        rng = np.random.default_rng(42)
        rad = np.abs(rng.normal(0.0, 0.07, depth_mm.size))
        ang = rng.uniform(0.0, 2.0 * np.pi, depth_mm.size)
        lat1 = rad * np.cos(ang)
        lat2 = rad * np.sin(ang)

        d_lo, d_hi = float(np.nanmin(depth_mm)), float(np.nanmax(depth_mm))
        ax_c.plot(
            [d_lo, d_hi], [0.0, 0.0], [0.0, 0.0],
            color="#9aa3ad", lw=2.6, alpha=0.7, zorder=1,
        )

        uid_to_xyz = {
            int(uid): (float(d), float(a), float(b))
            for uid, d, a, b in zip(units["unit_id"], depth_mm, lat1, lat2)
        }
        for ct in class_order:
            subset = units[units["cell_type"].astype(str) == ct]
            if subset.empty:
                continue
            pts = np.array([uid_to_xyz[int(uid)] for uid in subset["unit_id"]], dtype=float)
            sc = ax_c.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2],
                s=75, alpha=0.95, label=ct, color=class_palette[ct],
                depthshade=True, edgecolors="white", linewidths=0.5, zorder=2,
                clip_on=False,
            )
            class_handles.append(sc)
            class_labels.append(ct)

        lat_abs = np.abs(np.concatenate([lat1, lat2]))
        # Keep the same tick values as before; only expand axis limits so
        # edge markers are not clipped near Lateral₂.
        tick = float(np.round(max(1.08 * np.nanpercentile(lat_abs, 99), 0.05), 2))
        lat_lim = tick * 1.35
        xpad = 0.06 * max(d_hi - d_lo, 0.2)
        ax_c.set_xlim(d_lo - xpad, d_hi + 1.5 * xpad)
        ax_c.set_ylim(-lat_lim, lat_lim)
        ax_c.set_zlim(-lat_lim, lat_lim)
        # Pull the projection in so markers near the box faces stay fully visible.
        ax_c.set_box_aspect((3.2, 1.0, 1.0), zoom=0.98)
        ax_c.set_xlabel("Depth along probe (mm)", fontsize=10)
        # Clear default lateral titles; place readable ones outside the ticks.
        ax_c.set_ylabel("")
        ax_c.set_zlabel("")
        ax_c.xaxis.labelpad = 12
        ax_c.set_yticks([-tick, 0.0, tick])
        ax_c.set_zticks([-tick, 0.0, tick])
        ax_c.tick_params(axis="x", labelsize=8, pad=4)
        ax_c.tick_params(axis="y", labelsize=8, pad=4)
        ax_c.tick_params(axis="z", labelsize=8, pad=4)
        ax_c.view_init(elev=18, azim=-62)
        try:
            ax_c.set_proj_type("ortho")
        except Exception:
            pass
        ax_c.text2D(
            0.90, 0.10, "Lateral₁ (mm)",
            transform=ax_c.transAxes, fontsize=10,
            ha="center", va="center", rotation=-22, clip_on=False,
        )
        ax_c.text2D(
            1.14, 0.50, "Lateral₂ (mm)",
            transform=ax_c.transAxes, fontsize=10,
            ha="center", va="center", rotation=90, clip_on=False,
        )
    ax_c.text2D(
        -0.02, 1.04, "C", transform=ax_c.transAxes,
        fontsize=13, fontweight="bold", va="bottom", ha="right", clip_on=False,
    )

    # Legend for C — directly beside the 3D panel
    ax_leg = fig.add_subplot(top[0, 3])
    ax_leg.axis("off")
    if class_handles:
        ax_leg.legend(
            class_handles, class_labels,
            loc="center left",
            bbox_to_anchor=(0.0, 0.5),
            ncol=1,
            fontsize=10,
            frameon=False,
            title="Cell class",
            title_fontsize=11,
            handletextpad=0.5,
            borderaxespad=0.0,
            markerscale=0.9,
        )

    # D — region × cell-class unit counts (full width under top row)
    ax_d = fig.add_subplot(outer[1, 0])
    if units is None or units.empty or "region" not in units.columns or "cell_type" not in units.columns:
        ax_d.text(0.5, 0.5, "No units", ha="center", va="center")
    else:
        cell_classes = units["cell_type"].astype(str).map(analysis_cell_class)
        ct = pd.crosstab(units["region"], cell_classes)
        present = list(units["region"].astype(str).unique())
        ordered = [r for r in REGION_ORDER if r in present] + [
            r for r in present if r not in REGION_ORDER
        ]
        ct = ct.reindex(
            index=ordered,
            columns=cell_class_order_for_counts(ct.columns),
        ).fillna(0)
        sns.heatmap(
            ct, annot=True, fmt=".0f", cmap="Blues", cbar=True, ax=ax_d,
            cbar_kws={"label": "Units", "shrink": 0.9, "pad": 0.02},
            annot_kws={"fontsize": 10},
        )
        ax_d.set_aspect("auto")
        ax_d.tick_params(axis="x", labeltop=False, labelbottom=True, labelsize=9, pad=2)
        ax_d.tick_params(axis="y", labelsize=10, pad=2)
        for label in ax_d.get_xticklabels():
            label.set_rotation(30)
            label.set_ha("right")
            label.set_rotation_mode("anchor")
        ax_d.xaxis.set_label_position("bottom")
        ax_d.set_xlabel("Cell class", labelpad=6)
        ax_d.set_ylabel("Region")
    panel_label(ax_d, "D", x=-0.02, y=1.08)

    enable_open_axes()
    style_figure_axes(fig)
    strip_box_frames(fig)
    path = output_dir / "fig_probe_trajectory.png"
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    for stem in _LEGACY_TRAJECTORY_STEMS:
        for ext in (".png", ".pdf"):
            (output_dir / f"{stem}{ext}").unlink(missing_ok=True)
    return path


def generate_publication_trajectory_figures(
    anatomy: pd.DataFrame,
    units: pd.DataFrame | None,
    figures_dir: Path,
    *,
    n_channels: int = 384,
    meta: dict[str, Any] | None = None,
) -> Path:
    """Write ``figures/trajectory/fig_probe_trajectory.png`` and clean legacy singles."""
    from visualization.constants import FIGURE_SUBDIR_TRAJECTORY as sub

    figures_dir = Path(figures_dir)
    out = figures_dir / sub if figures_dir.name != sub else figures_dir
    return plot_fig_probe_trajectory(
        anatomy, units, out, n_channels=n_channels, meta=meta,
    )
