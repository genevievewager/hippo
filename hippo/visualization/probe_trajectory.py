"""Publication probe-trajectory visualization for lab / NTE insertions.

Produces Overleaf-ready PNG/PDF figures and the same anatomy region-depth table
used by the hippocampal Neuropixels simulation.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from hippo.anatomy.trajectory_config import (
    DEFAULT_TRAJECTORY_CONFIG,
    load_trajectory_config,
    resolve_anatomy_regions_file,
)
from hippo.anatomy.trajectory_import import (
    assign_channels_from_depth,
    load_lab_anatomy_regions_csv,
)
from hippo.visualization.nte_bridge import (
    find_nte_repository,
    import_nte_probe_areas_mat,
    matlab_available,
    nte_clone_instructions,
    try_matlab_nte_probe_areas_export,
)

FIGURE_DPI = 300

BAND_COLORS = {
    "visual_cortex": "#B0B0B0",
    "VIS": "#B0B0B0",
    "HPF_ProS_transition": "#E9C46A",
    "HPF_ProS": "#E9C46A",
    "subiculum": "#F4A261",
    "SUB": "#F4A261",
    "dentate_gyrus": "#9B5DE5",
    "DG_mo": "#9B5DE5",
    "entorhinal_cortex": "#00BBF9",
    "ENT": "#00BBF9",
    "deep_entorhinal_HATA": "#00F5D4",
    "HATA_ENT": "#00F5D4",
}

SHORT_LABELS = {
    "visual_cortex": "VIS",
    "HPF_ProS_transition": "HPF/ProS",
    "subiculum": "SUB",
    "dentate_gyrus": "DG_mo",
    "entorhinal_cortex": "ENT",
    "deep_entorhinal_HATA": "HATA/ENT",
}

REQUIRED_ANATOMY_COLUMNS = [
    "depth_start_mm",
    "depth_end_mm",
    "depth_start_um",
    "depth_end_um",
    "region",
    "acronym",
    "layer_or_area",
    "parent_structure",
    "probe_shank",
    "channel_start",
    "channel_end",
    "include_in_hippocampal_simulation",
    "candidate_cell_classes",
    "source",
    "notes",
]


def _parse_bool(val: Any, default: bool = True) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() not in {"false", "0", "no", "n"}


def _color_for_row(row: pd.Series) -> str:
    for key in (row.get("acronym"), row.get("region"), row.get("layer_or_area")):
        if key is None or (isinstance(key, float) and np.isnan(key)):
            continue
        s = str(key)
        if s in BAND_COLORS:
            return BAND_COLORS[s]
    return "#9E9E9E"


def _short_label(row: pd.Series) -> str:
    region = str(row.get("region") or "")
    acronym = str(row.get("acronym") or "")
    if region in SHORT_LABELS:
        return SHORT_LABELS[region]
    if acronym in SHORT_LABELS:
        return SHORT_LABELS[acronym]
    if acronym and acronym != "nan":
        return acronym
    return region[:10] if region else "?"


def _ensure_mm_um(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "depth_start_mm" in out.columns and "depth_start_um" not in out.columns:
        out["depth_start_um"] = out["depth_start_mm"].astype(float) * 1000.0
        out["depth_end_um"] = out["depth_end_mm"].astype(float) * 1000.0
    if "depth_start_um" in out.columns and "depth_start_mm" not in out.columns:
        out["depth_start_mm"] = out["depth_start_um"].astype(float) / 1000.0
        out["depth_end_mm"] = out["depth_end_um"].astype(float) / 1000.0
    return out


def _standardize_anatomy_table(
    df: pd.DataFrame,
    *,
    probe_config: dict[str, Any],
    source: str,
) -> pd.DataFrame:
    out = _ensure_mm_um(df)
    out = assign_channels_from_depth(out, probe_config)

    if "acronym" not in out.columns:
        out["acronym"] = out.get("layer_or_area", out.get("region", ""))
    if "layer_or_area" not in out.columns:
        out["layer_or_area"] = out.get("layer", out.get("acronym", ""))
    if "parent_structure" not in out.columns:
        out["parent_structure"] = ""
    if "probe_shank" not in out.columns:
        out["probe_shank"] = probe_config.get("selected_shank") or 1
    if "candidate_cell_classes" not in out.columns:
        out["candidate_cell_classes"] = out.get("cell_types", "")
    if "include_in_hippocampal_simulation" not in out.columns:
        out["include_in_hippocampal_simulation"] = True
    else:
        out["include_in_hippocampal_simulation"] = out["include_in_hippocampal_simulation"].map(
            lambda v: _parse_bool(v, True)
        )
    if "source" not in out.columns:
        out["source"] = source
    else:
        out["source"] = out["source"].fillna(source)
    if "notes" not in out.columns:
        out["notes"] = ""

    # Ensure required column order (+ extras).
    cols = [c for c in REQUIRED_ANATOMY_COLUMNS if c in out.columns]
    extras = [c for c in out.columns if c not in cols]
    return out[cols + extras]


def load_anatomy_for_visualization(
    trajectory_config: dict[str, Any],
    *,
    anatomy_regions_file: str | Path | None = None,
    nte_export_file: str | Path | None = None,
    prefer_explicit_anatomy_file: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prefer NTE export, else config / explicit anatomy CSV.

    When ``prefer_explicit_anatomy_file`` is True and ``anatomy_regions_file``
    is set (e.g. simulation already wrote ``anatomy_regions_used.csv``), that
    table wins so figures and simulated units stay identical.
    """
    probe = trajectory_config.get("probe") or {}
    insertion = trajectory_config.get("insertion") or {}
    meta: dict[str, Any] = {"source": "unknown", "nte_positions": {}}

    if prefer_explicit_anatomy_file and anatomy_regions_file and Path(str(anatomy_regions_file)).exists():
        raw = pd.read_csv(anatomy_regions_file)
        # Already-channelized tables from the simulation may skip re-assignment
        # if channel columns exist; still normalize mm/µm.
        raw = _ensure_mm_um(raw)
        if "channel_start" not in raw.columns or raw["channel_start"].isna().all():
            anatomy = _standardize_anatomy_table(
                raw, probe_config=probe, source="simulation_anatomy_regions_used",
            )
        else:
            anatomy = _standardize_anatomy_table(
                raw, probe_config=probe,
                source=str(raw["source"].iloc[0]) if "source" in raw.columns else "simulation_anatomy_regions_used",
            )
        meta["source"] = "simulation_anatomy_regions_used"
        meta["anatomy_regions_file"] = str(Path(str(anatomy_regions_file)).resolve())
        return anatomy, meta

    export = nte_export_file or insertion.get("trajectory_export_file")
    if export and Path(str(export)).exists():
        path = Path(str(export))
        if path.suffix.lower() == ".mat":
            raw, nte_meta = import_nte_probe_areas_mat(
                path,
                site_pitch_um=float(probe.get("site_pitch_um") or 15.0),
                n_channels=int(probe.get("n_channels") or probe.get("n_channels_recorded") or 384),
            )
            meta["source"] = "neuropixels_trajectory_explorer_export"
            meta["nte_positions"] = nte_meta.get("probe_positions_ccf") or {}
            meta["nte_export"] = str(path.resolve())
            anatomy = _standardize_anatomy_table(raw, probe_config=probe, source=meta["source"])
            return anatomy, meta
        raw = load_lab_anatomy_regions_csv(path)
        meta["source"] = "trajectory_export_table"
        meta["nte_export"] = str(path.resolve())
        anatomy = _standardize_anatomy_table(raw, probe_config=probe, source=meta["source"])
        return anatomy, meta

    regions = anatomy_regions_file or resolve_anatomy_regions_file(trajectory_config)
    if regions is None or not Path(str(regions)).exists():
        raise FileNotFoundError(
            "No NTE export or anatomy_regions_file available for visualization"
        )
    raw = load_lab_anatomy_regions_csv(regions)
    meta["source"] = "screenshot_derived_region_table"
    meta["anatomy_regions_file"] = str(Path(str(regions)).resolve())
    anatomy = _standardize_anatomy_table(
        raw, probe_config=probe, source="screenshot_derived_approximate",
    )
    return anatomy, meta


def plot_region_depth_strip(
    anatomy: pd.DataFrame,
    insertion: dict[str, Any],
    output_stem: Path,
    *,
    include_non_hippocampal_regions: bool = True,
) -> dict[str, Path]:
    """Publication region-depth strip with probe shank + channel markers."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    df = anatomy.copy()
    if not include_non_hippocampal_regions:
        # Still draw excluded bands faintly for context; hatch them.
        pass

    z0 = float(df["depth_start_mm"].min())
    z1 = float(df["depth_end_mm"].max())

    fig, ax = plt.subplots(figsize=(6.2, 9.5))
    ax.set_xlim(0, 3.2)
    ax.set_ylim(z1 * 1.03, z0 - 0.03 * (z1 - z0 + 1e-6))

    for _, row in df.iterrows():
        d0 = float(row["depth_start_mm"])
        d1 = float(row["depth_end_mm"])
        included = _parse_bool(row.get("include_in_hippocampal_simulation"), True)
        color = _color_for_row(row)
        alpha = 0.85 if included else 0.35
        rect = FancyBboxPatch(
            (0.15, d0),
            1.7,
            max(d1 - d0, 0.01),
            boxstyle="square,pad=0",
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.8,
            alpha=alpha,
            hatch="////" if not included else None,
        )
        ax.add_patch(rect)
        label = _short_label(row)
        if not included:
            label = f"{label}*"
        ax.text(
            1.0, (d0 + d1) / 2.0, label,
            ha="center", va="center", fontsize=11, fontweight="bold", color="#111111",
        )
        # Right-side depth range + optional channels.
        ch0, ch1 = row.get("channel_start"), row.get("channel_end")
        ch_txt = ""
        if ch0 is not None and ch1 is not None and not pd.isna(ch0) and int(ch0) > 0:
            ch_txt = f"\nch {int(ch0)}–{int(ch1)}"
        ax.text(
            2.05, (d0 + d1) / 2.0,
            f"{d0:.2f}–{d1:.2f} mm{ch_txt}",
            ha="left", va="center", fontsize=8, color="#333333",
        )

    # Black probe shank through center of bands.
    shank_x = 1.0
    ax.plot([shank_x, shank_x], [z0, z1], color="black", linewidth=3.5, solid_capstyle="round", zorder=5)
    # Tip marker.
    ax.plot(shank_x, z1, "v", color="black", markersize=9, zorder=6)
    ax.plot(shank_x, z0, "o", color="black", markersize=5, zorder=6)

    # White channel-site markers along shank.
    if "channel_start" in df.columns and df["channel_start"].notna().any():
        pitch_mm = None
        # Infer pitch from consecutive channels if possible.
        depths = []
        for _, row in df.iterrows():
            ch0, ch1 = row.get("channel_start"), row.get("channel_end")
            if pd.isna(ch0) or pd.isna(ch1) or int(ch0) <= 0:
                continue
            for ch in range(int(ch0), int(ch1) + 1, max(1, (int(ch1) - int(ch0)) // 12 or 1)):
                # depth ≈ (ch-1) * pitch; prefer row mid if pitch unknown.
                depths.append(float(row["depth_start_mm"]) + (ch - int(ch0)) / max(int(ch1) - int(ch0), 1) * (float(row["depth_end_mm"]) - float(row["depth_start_mm"])))
        for d in depths:
            ax.plot(shank_x, d, "o", color="white", markersize=2.4, markeredgecolor="black", markeredgewidth=0.3, zorder=7)
    else:
        for d in np.linspace(z0, z1, 40):
            ax.plot(shank_x, d, "o", color="white", markersize=2.2, markeredgecolor="black", markeredgewidth=0.3, zorder=7)

    ap = insertion.get("ap_mm_from_bregma")
    ml = insertion.get("ml_mm_from_bregma")
    dv = insertion.get("dv_mm_from_brain_surface")
    h_ang = insertion.get("horizontal_angle_deg")
    v_ang = insertion.get("vertical_angle_deg")
    dv_note = "≈ " if insertion.get("dv_uncertain") else ""
    subtitle = f"AP {ap}, ML {ml}, DV {dv_note}{dv} mm, angle {h_ang}h/{v_ang}V"

    ax.set_xticks([])
    ax.set_ylabel("Depth along probe (mm)", fontsize=11)
    ax.set_title(
        "Trajectory-informed NP2.0 recording geometry\n" + subtitle,
        fontsize=12, pad=12,
    )
    ax.text(
        0.5, -0.06,
        "Approximate screenshot-derived region depths; not histology-confirmed.\n"
        "* hatched = excluded from hippocampal decoder by default (e.g. visual cortex).",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#444444",
    )
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def plot_nte_style_probe_areas(
    anatomy: pd.DataFrame,
    output_stem: Path,
    *,
    matlab_png: Path | None = None,
) -> dict[str, Path]:
    """NTE-like probe-areas strip (Python fallback or copied MATLAB export)."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")

    if matlab_png is not None and Path(matlab_png).exists():
        from shutil import copy2
        copy2(matlab_png, png)
        # Also wrap into a PDF page.
        img = plt.imread(str(matlab_png))
        fig, ax = plt.subplots(figsize=(5, 9))
        ax.imshow(img)
        ax.axis("off")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        return {"png": png, "pdf": pdf}

    # Python NTE-style: tip_distance-like axis (deepest at bottom as tip).
    df = anatomy.copy()
    z_max = float(df["depth_end_mm"].max())
    fig, ax = plt.subplots(figsize=(4.2, 9))
    for _, row in df.iterrows():
        d0 = float(row["depth_start_mm"])
        d1 = float(row["depth_end_mm"])
        # Convert dorsal-origin depth → tip_distance-like (0 at tip).
        tip0 = z_max - d1
        tip1 = z_max - d0
        color = _color_for_row(row)
        ax.barh(
            (tip0 + tip1) / 2.0, width=1.0, height=max(tip1 - tip0, 0.01),
            left=0, color=color, edgecolor="black", linewidth=0.6, alpha=0.85,
        )
        ax.text(
            0.5, (tip0 + tip1) / 2.0, _short_label(row),
            ha="center", va="center", fontsize=9, fontweight="bold",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, z_max * 1.02)
    ax.set_xticks([])
    ax.set_ylabel("Distance from tip (mm)  [NTE-style]")
    ax.set_title("Probe areas (NTE-style)")
    ax.text(
        0.5, -0.05,
        "Python recreation of NTE probe-areas strip\n"
        "(MATLAB NTE plot unavailable or not used).",
        transform=ax.transAxes, ha="center", va="top", fontsize=7, color="#555555",
    )
    fig.tight_layout()
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def plot_probe_trajectory_3d(
    anatomy: pd.DataFrame,
    insertion: dict[str, Any],
    output_stem: Path,
    *,
    nte_positions: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Coordinate-space 3D trajectory (not a full Allen CCF mesh render)."""
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    ap = float(insertion.get("ap_mm_from_bregma") or 0.0)
    ml = float(insertion.get("ml_mm_from_bregma") or 0.0)
    dv = float(insertion.get("dv_mm_from_brain_surface") or anatomy["depth_end_mm"].max())
    h_ang = float(insertion.get("horizontal_angle_deg") or 0.0)
    v_ang = float(insertion.get("vertical_angle_deg") or 90.0)

    # Approximate linear probe path in stereotaxic mm (coordinate space only).
    # Elevation 90° = straight down; azimuth rotates AP/ML components.
    elev = np.deg2rad(v_ang)
    azim = np.deg2rad(h_ang)
    depths = np.linspace(0.0, float(anatomy["depth_end_mm"].max()), 200)
    # Direction of insertion (from surface toward tip).
    d_ap = np.cos(elev) * np.cos(azim)
    d_ml = np.cos(elev) * np.sin(azim)
    d_dv = np.sin(elev)
    # Normalize DV-dominant paths.
    norm = np.sqrt(d_ap**2 + d_ml**2 + d_dv**2) or 1.0
    d_ap, d_ml, d_dv = d_ap / norm, d_ml / norm, d_dv / norm

    xs = ap + depths * d_ap
    ys = ml + depths * d_ml
    zs = 0.0 + depths * abs(d_dv)  # depth from brain surface

    # Override endpoints with CCF positions when NTE export provides them (µm → mm).
    if nte_positions:
        try:
            xs = np.linspace(
                nte_positions["ccf_ap_start_um"] / 1000.0,
                nte_positions["ccf_ap_end_um"] / 1000.0,
                len(depths),
            )
            ys = np.linspace(
                nte_positions["ccf_ml_start_um"] / 1000.0,
                nte_positions["ccf_ml_end_um"] / 1000.0,
                len(depths),
            )
            zs = np.linspace(
                nte_positions["ccf_dv_start_um"] / 1000.0,
                nte_positions["ccf_dv_end_um"] / 1000.0,
                len(depths),
            )
        except Exception:
            pass

    fig = plt.figure(figsize=(7.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Color segments by region.
    for _, row in anatomy.iterrows():
        d0 = float(row["depth_start_mm"])
        d1 = float(row["depth_end_mm"])
        mask = (depths >= d0) & (depths <= d1)
        if not np.any(mask):
            continue
        ax.plot(
            xs[mask], ys[mask], zs[mask],
            color=_color_for_row(row), linewidth=3.0, label=_short_label(row),
        )
    ax.scatter([xs[0]], [ys[0]], [zs[0]], c="black", s=40, label="insertion")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c="red", s=40, marker="v", label="tip")
    ax.set_xlabel("AP (mm)")
    ax.set_ylabel("ML (mm)")
    ax.set_zlabel("DV depth (mm)")
    ax.set_title(
        "Probe trajectory in coordinate space\n"
        "(not a full Allen CCF mesh render)"
    )
    # Deduplicate legend entries.
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, lab in zip(handles, labels):
        seen[lab] = h
    ax.legend(seen.values(), seen.keys(), fontsize=7, loc="upper left")
    fig.tight_layout()
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": png, "pdf": pdf}


def plot_probe_trajectory(
    trajectory_config: str,
    anatomy_regions_file: str | None = None,
    nte_export_file: str | None = None,
    output_dir: str = "outputs/figures",
    include_non_hippocampal_regions: bool = False,
    use_nte_style: bool = True,
    make_3d: bool = False,
    run_root: str | Path | None = None,
    prefer_explicit_anatomy_file: bool = False,
) -> dict[str, Any]:
    """Generate probe trajectory figures + anatomy table for simulation.

    Parameters
    ----------
    trajectory_config:
        Path to YAML (e.g. ``configs/trajectories/lab_npx2_default.yaml``).
    anatomy_regions_file:
        Optional override CSV; otherwise taken from the trajectory config.
    nte_export_file:
        Optional Neuropixels Trajectory Explorer ``.mat`` / table export.
        When present, overrides the screenshot-derived region table.
    output_dir:
        Directory for PNG/PDF figures (created if needed).
    include_non_hippocampal_regions:
        If False, visual cortex remains marked excluded for decoder units
        (still drawn on the figure).
    use_nte_style:
        Also write an NTE-style probe-areas strip (Python and/or MATLAB).
    make_3d:
        Also write a coordinate-space 3D trajectory figure.
    run_root:
        Optional experiment root. When set, also writes
        ``<run_root>/trajectory/anatomy_regions_used.csv`` (and NTE CSV).

    Returns
    -------
    dict with paths and metadata.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        cfg = load_trajectory_config(trajectory_config)

    insertion = cfg.get("insertion") or {}
    probe = cfg.get("probe") or {}
    sim = cfg.get("simulation") or {}

    anatomy, load_meta = load_anatomy_for_visualization(
        cfg,
        anatomy_regions_file=anatomy_regions_file,
        nte_export_file=nte_export_file,
        prefer_explicit_anatomy_file=prefer_explicit_anatomy_file,
    )

    # Decoder exclusion policy.
    if not include_non_hippocampal_regions and bool(
        sim.get("exclude_non_hippocampal_regions_by_default", True)
    ):
        # Keep flags from table; ensure VIS-like rows stay excluded.
        mask_vis = anatomy["region"].astype(str).str.contains("visual", case=False, na=False) | (
            anatomy["acronym"].astype(str).isin(["VIS", "VISp", "VISl"])
        )
        anatomy.loc[mask_vis, "include_in_hippocampal_simulation"] = False

    figures_dir = Path(output_dir)
    # Keep PNGs under figures/trajectory/ when callers pass the figures root.
    from visualization.constants import FIGURE_SUBDIR_TRAJECTORY

    if figures_dir.name == "figures":
        figures_dir = figures_dir / FIGURE_SUBDIR_TRAJECTORY
    elif figures_dir.name != FIGURE_SUBDIR_TRAJECTORY and (figures_dir / "behavior").exists():
        # Heuristic: looks like figures root without ending in "figures".
        figures_dir = figures_dir / FIGURE_SUBDIR_TRAJECTORY
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Infer run root from output_dir / figures layout.
    if run_root is None:
        out = Path(output_dir)
        if out.name == FIGURE_SUBDIR_TRAJECTORY:
            run_root = out.parent.parent if out.parent.name == "figures" else out.parent
        elif out.name == "figures":
            run_root = out.parent
        else:
            run_root = out.parent if out.name == "figures" else out
    run_root = Path(run_root)

    from hippo.anatomy.trajectory_config import (
        resolve_cell_capture_file,
        write_trial_trajectory_bundle,
    )

    capture_src = resolve_cell_capture_file(cfg)
    bundle = write_trial_trajectory_bundle(
        run_root,
        cfg,
        anatomy,
        cell_capture_src=capture_src,
        anatomy_source=str(load_meta.get("source") or "visualization"),
    )
    traj_dir = run_root / "trajectory"
    anatomy_used_path = Path(bundle["anatomy_regions_used"])

    # Also write NTE-specific table when export was used.
    nte_csv_path = None
    if load_meta.get("source") == "neuropixels_trajectory_explorer_export":
        nte_csv_path = traj_dir / "anatomy_regions_from_nte.csv"
        anatomy.to_csv(nte_csv_path, index=False, na_rep="NA")

    strip = plot_region_depth_strip(
        anatomy,
        insertion,
        figures_dir / "probe_trajectory_regions",
        include_non_hippocampal_regions=include_non_hippocampal_regions,
    )

    outputs: dict[str, Any] = {
        "trajectory_config": str(Path(trajectory_config).resolve()),
        "active_trajectory_name": bundle.get("active_trajectory_name"),
        "active_trajectory_yaml": bundle.get("active_trajectory_yaml"),
        "anatomy_source": load_meta.get("source"),
        "anatomy_regions_used": str(anatomy_used_path.resolve()),
        "anatomy_regions_csv": str((run_root / "anatomy_regions.csv").resolve()),
        "probe_trajectory_regions_png": str(strip["png"]),
        "probe_trajectory_regions_pdf": str(strip["pdf"]),
        "visual_cortex_excluded": bool(
            (~anatomy.loc[
                anatomy["region"].astype(str).str.contains("visual", case=False, na=False),
                "include_in_hippocampal_simulation",
            ]).all()
        ) if anatomy["region"].astype(str).str.contains("visual", case=False, na=False).any() else True,
        "deployment_spike_source": (cfg.get("decoder") or {}).get("deployment_spike_source", "sorted"),
        "nte_repository": None,
        "messages": [],
    }
    if nte_csv_path is not None:
        outputs["anatomy_regions_from_nte"] = str(nte_csv_path.resolve())

    if use_nte_style:
        matlab_png = None
        nte_repo = find_nte_repository()
        outputs["nte_repository"] = str(nte_repo) if nte_repo else None
        export_path = load_meta.get("nte_export")
        if nte_repo and export_path and Path(str(export_path)).suffix.lower() == ".mat":
            if matlab_available():
                candidate = figures_dir / "_nte_matlab_probe_areas.png"
                ok = try_matlab_nte_probe_areas_export(
                    nte_repo, Path(str(export_path)), candidate,
                )
                if ok:
                    matlab_png = candidate
                else:
                    outputs["messages"].append(
                        "NTE-style MATLAB plot unavailable; using Python region-depth strip plot."
                    )
            else:
                outputs["messages"].append(
                    "NTE-style MATLAB plot unavailable; using Python region-depth strip plot."
                )
        elif not nte_repo:
            outputs["messages"].append(nte_clone_instructions())
            outputs["messages"].append(
                "NTE-style MATLAB plot unavailable; using Python region-depth strip plot."
            )
        else:
            outputs["messages"].append(
                "NTE-style MATLAB plot unavailable; using Python region-depth strip plot."
            )

        nte_style = plot_nte_style_probe_areas(
            anatomy,
            figures_dir / "probe_areas_nte_style",
            matlab_png=matlab_png,
        )
        outputs["probe_areas_nte_style_png"] = str(nte_style["png"])
        outputs["probe_areas_nte_style_pdf"] = str(nte_style["pdf"])

    if make_3d:
        traj3d = plot_probe_trajectory_3d(
            anatomy,
            insertion,
            figures_dir / "probe_trajectory_3d",
            nte_positions=load_meta.get("nte_positions") or None,
        )
        outputs["probe_trajectory_3d_png"] = str(traj3d["png"])
        outputs["probe_trajectory_3d_pdf"] = str(traj3d["pdf"])

    # Metadata sidecar next to figures / trajectory folder.
    meta_path = traj_dir / "trajectory_visualization_meta.json"
    meta_path.write_text(json.dumps({
        "insertion": insertion,
        "probe": probe,
        "load_meta": load_meta,
        "outputs": {k: v for k, v in outputs.items() if k != "messages"},
        "messages": outputs["messages"],
        "note": (
            "Ground-truth spikes are diagnostics only; "
            "deployable decoder selection uses sorted spikes."
        ),
    }, indent=2, default=str))
    outputs["visualization_meta"] = str(meta_path.resolve())

    for msg in outputs["messages"]:
        print(msg, flush=True)
    return outputs


def main(argv: list[str] | None = None) -> None:
    from hippo.anatomy.trajectory_config import (
        list_trajectory_configs,
        resolve_trajectory_config,
    )

    available = ", ".join(r["name"] for r in list_trajectory_configs()) or "(none)"
    parser = argparse.ArgumentParser(
        description="Plot lab / NTE probe trajectory through brain regions",
    )
    parser.add_argument(
        "--trajectory",
        "--trajectory-config",
        "--trajectory-name",
        dest="trajectory_config",
        type=str,
        default=None,
        help=(
            "Active insertion name or YAML path "
            f"(default: lab_npx2_default). Available: {available}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Trial output root (recommended). Figures go to <output>/figures/ and "
            "coords to <output>/trajectory/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Figures directory (legacy). Prefer --output <trial_dir>.",
    )
    parser.add_argument("--nte-export", type=Path, default=None)
    parser.add_argument("--anatomy-regions-file", type=Path, default=None)
    parser.add_argument(
        "--include-non-hippocampal-regions",
        action="store_true",
        help="Mark visual cortex (etc.) as included for decoder units",
    )
    parser.add_argument("--make-3d", action="store_true")
    parser.add_argument(
        "--use-nte-style",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--list-trajectories",
        action="store_true",
        help="List selectable trajectory configs and exit",
    )
    args = parser.parse_args(argv)

    if args.list_trajectories:
        for row in list_trajectory_configs(include_templates=True):
            mark = " (default)" if row["is_default"] else ""
            print(f"{row['name']}{mark}\t{row['path']}")
        return

    traj = resolve_trajectory_config(args.trajectory_config)
    from visualization.constants import FIGURE_SUBDIR_TRAJECTORY

    if args.output is not None:
        output_dir = Path(args.output) / "figures" / FIGURE_SUBDIR_TRAJECTORY
    elif args.output_dir is not None:
        output_dir = args.output_dir
        if output_dir.name == "figures":
            output_dir = output_dir / FIGURE_SUBDIR_TRAJECTORY
    else:
        output_dir = Path("outputs/run_001/figures") / FIGURE_SUBDIR_TRAJECTORY
        print(
            "Note: writing to outputs/run_001/figures/trajectory. "
            "Pass --output outputs/<trial> to attach coords to a trial folder.",
            flush=True,
        )

    result = plot_probe_trajectory(
        trajectory_config=str(traj),
        anatomy_regions_file=str(args.anatomy_regions_file) if args.anatomy_regions_file else None,
        nte_export_file=str(args.nte_export) if args.nte_export else None,
        output_dir=str(output_dir),
        include_non_hippocampal_regions=args.include_non_hippocampal_regions,
        use_nte_style=args.use_nte_style,
        make_3d=args.make_3d,
    )
    print(json.dumps(
        {k: v for k, v in result.items() if k != "messages"},
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
