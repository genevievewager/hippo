"""Import Neuropixels Trajectory Explorer exports into anatomy_regions tables.

Native Trajectory Explorer save format is MATLAB ``.mat`` (``probe_areas`` /
``probe_positions_ccf``). Lab workflows often re-export region-depth tables as
CSV / TSV / JSON, or author a YAML table by hand. This module accepts all of
those and normalizes to a single internal schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Canonical columns written to anatomy_regions.csv
ANATOMY_REGION_COLUMNS: list[str] = [
    "region",
    "layer",
    "acronym",
    "depth_start_um",
    "depth_end_um",
    "channel_start",
    "channel_end",
    "ccf_ap_um",
    "ccf_ml_um",
    "ccf_dv_um",
    "insertion_ap_mm",
    "insertion_ml_mm",
    "insertion_depth_um",
    "insertion_angle_ap_deg",
    "insertion_angle_ml_deg",
    "probe_type",
    "active_site_pitch_um",
    "notes",
]

# Extra columns retained for downstream viz / density summaries.
_EXTRA_COLUMNS = ["channels", "n_channels", "n_units", "cell_types", "trajectory_source"]

# Map free-text / Allen CCF labels → (region, layer, acronym).
# More specific layer patterns must come before generic region aliases.
_CCF_ALIASES: list[tuple[re.Pattern[str], tuple[str, str, str]]] = [
    (re.compile(r"oriens|stratum oriens|ca1so", re.I), ("CA1", "oriens", "CA1so")),
    (re.compile(r"radiatum|stratum radiatum|ca1sr", re.I), ("CA1", "radiatum", "CA1sr")),
    (re.compile(r"pyramidal.*(ca1|field)|ca1.*pyramid|ca1sp", re.I), ("CA1", "pyramidal", "CA1sp")),
    (re.compile(r"dentate.*granule|granule cell layer|dg-sg", re.I), ("DG", "granule", "DG-sg")),
    (re.compile(r"polymorph|hilus|dg-po", re.I), ("DG", "hilus", "DG-po")),
    (re.compile(r"layer\s*3|l3|entm3", re.I), ("MEC", "layer3", "ENTm3")),
    (re.compile(r"layer\s*2|l2|entm2", re.I), ("MEC", "layer2", "ENTm2")),
    (re.compile(r"field\s*ca2|\bca2\b", re.I), ("CA2", "pyramidal", "CA2")),
    (re.compile(r"field\s*ca3|\bca3\b", re.I), ("CA3", "pyramidal", "CA3")),
    (re.compile(r"field\s*ca1|\bca1\b", re.I), ("CA1", "pyramidal", "CA1")),
    (re.compile(r"dentate gyrus|\bdg\b", re.I), ("DG", "granule", "DG")),
    (re.compile(r"subiculum|\bsub\b", re.I), ("Subiculum", "pyramidal", "SUB")),
    (re.compile(r"entorhinal.*medial|\bmec\b|entm", re.I), ("MEC", "layer2", "ENTm")),
    (re.compile(r"entorhinal.*lateral|\blec\b|entl", re.I), ("MEC", "layer2", "ENTl")),
    (re.compile(r"entorhinal|\bec\b", re.I), ("MEC", "layer2", "ENT")),
]


def _na() -> Any:
    return pd.NA


def _normalize_region_layer(name: str, layer_hint: str | None = None) -> tuple[str, str, str]:
    text = str(name).strip()
    layer_text = (layer_hint or "").strip()
    combined = f"{text} {layer_text}".strip()
    for pattern, mapped in _CCF_ALIASES:
        if pattern.search(combined) or pattern.search(text):
            region, layer, acronym = mapped
            if layer_hint and layer_hint.strip():
                # Prefer explicit layer column when provided.
                lh = layer_hint.strip().lower()
                if lh in {"oriens", "pyramidal", "radiatum", "granule", "hilus", "layer2", "layer3"}:
                    layer = lh if lh != "layer 2" else "layer2"
                    if lh in {"layer 2", "l2"}:
                        layer = "layer2"
                    if lh in {"layer 3", "l3"}:
                        layer = "layer3"
            return region, layer, acronym
    # Fallback: use raw tokens.
    region = text.split(",")[0].strip() or "unknown"
    layer = (layer_hint or "unspecified").strip() or "unspecified"
    acronym = re.sub(r"[^A-Za-z0-9]+", "", region)[:12] or "UNK"
    return region, layer, acronym


def _channels_for_depths(
    depth_start_um: float,
    depth_end_um: float,
    *,
    n_channels: int,
    pitch_um: float,
) -> tuple[int, int, int]:
    """Return 1-based inclusive channel_start/end and n_channels in band."""
    depths = np.arange(n_channels) * float(pitch_um)
    mask = (depths >= float(depth_start_um)) & (depths < float(depth_end_um))
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return 0, 0, 0
    return int(idxs[0]) + 1, int(idxs[-1]) + 1, int(len(idxs))


def _empty_row(**overrides: Any) -> dict[str, Any]:
    row = {col: _na() for col in ANATOMY_REGION_COLUMNS}
    row.update(overrides)
    return row


def _finalize_table(
    rows: list[dict[str, Any]],
    *,
    n_channels: int = 384,
    pitch_um: float = 20.0,
    probe_type: str = "neuropixels_1.0",
    insertion: dict[str, Any] | None = None,
    trajectory_source: str = "unknown",
) -> pd.DataFrame:
    insertion = insertion or {}
    finalized: list[dict[str, Any]] = []
    for raw in rows:
        row = _empty_row()
        for col in ANATOMY_REGION_COLUMNS:
            if col in raw and raw[col] is not None and not (isinstance(raw[col], float) and np.isnan(raw[col])):
                row[col] = raw[col]

        region = row.get("region")
        layer = row.get("layer")
        if pd.isna(region) or region is None or str(region).strip() == "":
            name = raw.get("name") or raw.get("area") or raw.get("ccf_name") or "unknown"
            layer_hint = None if pd.isna(layer) else str(layer)
            region, layer, acronym = _normalize_region_layer(str(name), layer_hint)
            row["region"] = region
            row["layer"] = layer
            if pd.isna(row.get("acronym")):
                row["acronym"] = acronym
        elif pd.isna(row.get("acronym")):
            _, _, acronym = _normalize_region_layer(str(region), str(layer) if not pd.isna(layer) else None)
            row["acronym"] = acronym

        if pd.isna(row.get("depth_start_um")) or pd.isna(row.get("depth_end_um")):
            # Allow tip_distance_* (NTE: 0 at tip) → dorsal-origin depth.
            if "tip_distance_start_um" in raw and "tip_distance_end_um" in raw:
                active_len = float(n_channels) * float(pitch_um)
                tip0 = float(raw["tip_distance_start_um"])
                tip1 = float(raw["tip_distance_end_um"])
                # tip near 0 = ventral tip; convert to dorsal-origin depth.
                d0 = active_len - max(tip0, tip1)
                d1 = active_len - min(tip0, tip1)
                row["depth_start_um"] = float(min(d0, d1))
                row["depth_end_um"] = float(max(d0, d1))

        for key, dest in [
            ("insertion_ap_mm", "insertion_ap_mm"),
            ("insertion_ml_mm", "insertion_ml_mm"),
            ("insertion_depth_um", "insertion_depth_um"),
            ("insertion_angle_ap_deg", "insertion_angle_ap_deg"),
            ("insertion_angle_ml_deg", "insertion_angle_ml_deg"),
            ("angle_ap_deg", "insertion_angle_ap_deg"),
            ("angle_ml_deg", "insertion_angle_ml_deg"),
        ]:
            if key in insertion and insertion[key] is not None and pd.isna(row.get(dest)):
                row[dest] = insertion[key]

        if pd.isna(row.get("probe_type")):
            row["probe_type"] = probe_type
        if pd.isna(row.get("active_site_pitch_um")):
            row["active_site_pitch_um"] = float(pitch_um)

        z0 = row.get("depth_start_um")
        z1 = row.get("depth_end_um")
        if not pd.isna(z0) and not pd.isna(z1):
            ch0, ch1, n_ch = _channels_for_depths(
                float(z0), float(z1), n_channels=n_channels, pitch_um=float(pitch_um),
            )
            row["channel_start"] = ch0
            row["channel_end"] = ch1
            row["n_channels"] = n_ch
            row["channels"] = f"{ch0}-{ch1}" if n_ch else ""
        else:
            row["n_channels"] = 0
            row["channels"] = ""

        density = raw.get("density", np.nan)
        n_ch = int(row.get("n_channels") or 0)
        if n_ch and not (isinstance(density, float) and np.isnan(density)):
            row["n_units"] = max(1, int(float(density) * n_ch))
        else:
            row["n_units"] = raw.get("n_units", _na())

        if "cell_types" in raw and raw["cell_types"] is not None:
            row["cell_types"] = raw["cell_types"]
        row["trajectory_source"] = trajectory_source
        if pd.isna(row.get("notes")):
            row["notes"] = raw.get("notes", _na())
        finalized.append(row)

    df = pd.DataFrame(finalized)
    # Ensure column order: required first, then extras present.
    ordered = [c for c in ANATOMY_REGION_COLUMNS if c in df.columns]
    for c in _EXTRA_COLUMNS:
        if c in df.columns and c not in ordered:
            ordered.append(c)
    for c in df.columns:
        if c not in ordered:
            ordered.append(c)
    return df[ordered]


def schematic_anatomy_table(
    region_segments: list[dict] | None = None,
    *,
    n_channels: int = 384,
    pitch_um: float = 20.0,
    probe_type: str = "neuropixels_1.0",
    region_to_cell_type: dict[tuple[str, str], str] | None = None,
) -> pd.DataFrame:
    """Build the legacy schematic hippocampal anatomy table."""
    from hippo_sim.config import REGION_SEGMENTS, REGION_TO_CELL_TYPE

    segments = region_segments if region_segments is not None else list(REGION_SEGMENTS)
    mapping = region_to_cell_type or REGION_TO_CELL_TYPE
    rows = []
    for seg in segments:
        region, layer = seg["region"], seg["layer"]
        _, _, acronym = _normalize_region_layer(region, layer)
        rows.append({
            "region": region,
            "layer": layer,
            "acronym": acronym,
            "depth_start_um": seg["z_start"],
            "depth_end_um": seg["z_end"],
            "density": seg.get("density", np.nan),
            "cell_types": mapping.get((region, layer), ""),
            "notes": "schematic hippocampal probe geometry",
        })
    return _finalize_table(
        rows,
        n_channels=n_channels,
        pitch_um=pitch_um,
        probe_type=probe_type,
        trajectory_source="schematic_fallback",
    )


def anatomy_table_to_region_segments(df: pd.DataFrame) -> list[dict]:
    """Convert anatomy table rows to SimConfig.region_segments dicts.

    Rows with ``include_in_hippocampal_simulation=false`` are skipped so
    visual cortex (etc.) does not receive hippocampal decoder units by default.
    """
    segments: list[dict] = []
    for _, row in df.iterrows():
        if "include_in_hippocampal_simulation" in df.columns:
            flag = row.get("include_in_hippocampal_simulation")
            if flag is False or (
                isinstance(flag, str) and flag.strip().lower() in {"false", "0", "no", "n"}
            ):
                continue
        z0 = row.get("depth_start_um")
        z1 = row.get("depth_end_um")
        if pd.isna(z0) or pd.isna(z1):
            continue
        density = 5.0
        if "n_units" in row.index and "n_channels" in row.index and not pd.isna(row.get("n_channels")):
            n_ch = int(row["n_channels"])
            if n_ch > 0 and not pd.isna(row.get("n_units")):
                density = float(row["n_units"]) / float(n_ch)
        layer = row.get("layer")
        if layer is None or (isinstance(layer, float) and np.isnan(layer)):
            layer = row.get("layer_or_area", "unspecified")
        segments.append({
            "region": str(row["region"]),
            "layer": str(layer),
            "z_start": float(z0),
            "z_end": float(z1),
            "density": density,
        })
    return segments


def write_anatomy_regions_csv(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    # Write NA as empty for readability.
    out.to_csv(path, index=False, na_rep="NA")
    return path


def load_probe_trajectory_config(path: Path | str | None) -> dict[str, Any]:
    """Load configs/probe_trajectory.yaml (or return empty defaults)."""
    defaults: dict[str, Any] = {
        "probe": {
            "type": "neuropixels_1.0",
            "n_channels": 384,
            "site_pitch_um": 20.0,
            "active_length_um": 3840.0,
            "shank_count": 1,
        },
        "trajectory": {
            "source": "schematic",
            "export_file": None,
            "insertion_ap_mm": None,
            "insertion_ml_mm": None,
            "insertion_depth_um": None,
            "angle_ap_deg": None,
            "angle_ml_deg": None,
        },
        "simulation": {
            "use_trajectory_regions": True,
            "fallback_to_schematic_hippocampus": True,
        },
    }
    if path is None:
        return defaults
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectory config not found: {path}")
    with open(path) as f:
        loaded = yaml.safe_load(f) or {}
    for section, values in defaults.items():
        merged = dict(values)
        merged.update(loaded.get(section) or {})
        defaults[section] = merged
    return defaults


def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix in {".json", ".jsonl"}:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            if "regions" in data:
                return pd.DataFrame(data["regions"])
            if "anatomy_regions" in data:
                return pd.DataFrame(data["anatomy_regions"])
            # Single-region dict or metadata+regions.
            if "region" in data:
                return pd.DataFrame([data])
            return pd.DataFrame(data)
        return pd.DataFrame(data)
    if suffix in {".yaml", ".yml"}:
        with open(path) as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            if "regions" in data:
                return pd.DataFrame(data["regions"])
            if "anatomy_regions" in data:
                return pd.DataFrame(data["anatomy_regions"])
            return pd.DataFrame([data])
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported trajectory export format: {path.suffix}")


def _import_mat(path: Path, *, n_channels: int, pitch_um: float, probe_type: str) -> pd.DataFrame:
    """Best-effort import of Neuropixels Trajectory Explorer .mat saves."""
    from scipy.io import loadmat

    mat = loadmat(path, simplify_cells=True)
    rows: list[dict[str, Any]] = []
    probe_areas = mat.get("probe_areas")
    if probe_areas is None:
        raise ValueError(f"{path} has no probe_areas variable (NTE export expected)")

    # probe_areas may be list/array of per-probe structures.
    probes = probe_areas if isinstance(probe_areas, (list, tuple, np.ndarray)) else [probe_areas]
    for probe in np.atleast_1d(probes):
        if probe is None:
            continue
        # Expected: struct with tip_distance and Allen structure fields.
        if isinstance(probe, dict):
            tip = probe.get("tip_distance")
            names = (
                probe.get("name")
                or probe.get("acronym")
                or probe.get("safe_name")
                or probe.get("structure")
            )
            if tip is None:
                continue
            tip = np.asarray(tip).astype(float).ravel()
            if names is None:
                names = [f"area_{i}" for i in range(max(0, len(tip) - 1))]
            else:
                names = list(np.atleast_1d(names))
            # tip_distance borders: length n_borders; regions between consecutive borders.
            if len(tip) >= 2:
                for i in range(len(tip) - 1):
                    name = names[i] if i < len(names) else f"area_{i}"
                    if isinstance(name, dict):
                        name = name.get("name") or name.get("acronym") or str(name)
                    rows.append({
                        "name": str(name),
                        "tip_distance_start_um": float(tip[i]),
                        "tip_distance_end_um": float(tip[i + 1]),
                        "notes": f"imported from NTE mat: {path.name}",
                    })
            elif "acronym" in probe or "name" in probe:
                # Alternate flat table encoding.
                acronyms = np.atleast_1d(probe.get("acronym", probe.get("name")))
                starts = np.atleast_1d(probe.get("depth_start_um", tip))
                ends = np.atleast_1d(probe.get("depth_end_um", tip))
                for i, acr in enumerate(acronyms):
                    rows.append({
                        "name": str(acr),
                        "depth_start_um": float(starts[min(i, len(starts) - 1)]),
                        "depth_end_um": float(ends[min(i, len(ends) - 1)]),
                        "notes": f"imported from NTE mat: {path.name}",
                    })

    insertion: dict[str, Any] = {}
    positions = mat.get("probe_positions_ccf")
    if positions is not None:
        try:
            pos = np.asarray(positions, dtype=object)
            first = pos.flat[0] if pos.size else None
            arr = np.asarray(first, dtype=float)
            # [AP(start,end); DV(start,end); ML(start,end)]
            if arr.ndim >= 2 and arr.shape[0] >= 3:
                insertion["ccf_ap_um"] = float(arr[0, 0])
                insertion["ccf_dv_um"] = float(arr[1, 0])
                insertion["ccf_ml_um"] = float(arr[2, 0])
        except Exception:
            pass

    if not rows:
        raise ValueError(f"Could not parse region bands from NTE mat: {path}")

    # Stamp shared CCF insertion coords onto each row when available.
    for row in rows:
        for k in ("ccf_ap_um", "ccf_ml_um", "ccf_dv_um"):
            if k in insertion:
                row[k] = insertion[k]

    return _finalize_table(
        rows,
        n_channels=n_channels,
        pitch_um=pitch_um,
        probe_type=probe_type,
        insertion={
            "insertion_ap_mm": insertion.get("ccf_ap_um", _na()) / 1000.0
            if "ccf_ap_um" in insertion else None,
            "insertion_ml_mm": insertion.get("ccf_ml_um", _na()) / 1000.0
            if "ccf_ml_um" in insertion else None,
        },
        trajectory_source="neuropixels_trajectory_explorer",
    )


def _manual_trajectory_from_config(
    traj_cfg: dict[str, Any],
    probe_cfg: dict[str, Any],
) -> pd.DataFrame | None:
    """Build a single-band or multi-band table from manual YAML coordinates."""
    regions = traj_cfg.get("regions")
    if regions:
        rows = []
        for r in regions:
            rows.append(dict(r))
        return _finalize_table(
            rows,
            n_channels=int(probe_cfg.get("n_channels", 384)),
            pitch_um=float(probe_cfg.get("site_pitch_um", 20.0)),
            probe_type=str(probe_cfg.get("type", "neuropixels_1.0")),
            insertion=traj_cfg,
            trajectory_source="manual_yaml",
        )

    # If only insertion coords given without regions, return None (caller falls back).
    has_coords = any(
        traj_cfg.get(k) is not None
        for k in (
            "insertion_ap_mm",
            "insertion_ml_mm",
            "insertion_depth_um",
            "angle_ap_deg",
            "angle_ml_deg",
        )
    )
    if not has_coords:
        return None
    # Manual coords alone cannot define region bands — annotate schematic later.
    return None


def import_trajectory(
    export_file: Path | str | None = None,
    *,
    trajectory_config: Path | str | dict[str, Any] | None = None,
    fallback_schematic: bool = True,
    region_segments: list[dict] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Import a probe trajectory into the standard anatomy table.

    Parameters
    ----------
    export_file:
        Path to NTE export (``.mat``, ``.csv``, ``.tsv``, ``.json``, ``.yaml``).
    trajectory_config:
        Path to ``probe_trajectory.yaml`` or an already-loaded dict.
    fallback_schematic:
        If True and no usable export/manual regions exist, use schematic geometry.
    region_segments:
        Optional override schematic segments for fallback.

    Returns
    -------
    anatomy_df, meta
        ``meta`` includes ``source``, ``used_fallback``, ``export_file``, ``config``.
    """
    if isinstance(trajectory_config, dict):
        cfg = load_probe_trajectory_config(None)
        for section in ("probe", "trajectory", "simulation"):
            if section in trajectory_config:
                cfg[section].update(trajectory_config[section] or {})
    else:
        cfg = load_probe_trajectory_config(trajectory_config)

    probe_cfg = cfg["probe"]
    traj_cfg = cfg["trajectory"]
    sim_cfg = cfg["simulation"]
    n_channels = int(probe_cfg.get("n_channels", 384))
    pitch_um = float(probe_cfg.get("site_pitch_um", 20.0))
    probe_type = str(probe_cfg.get("type", "neuropixels_1.0"))

    export_path = export_file or traj_cfg.get("export_file")
    if export_path is not None and str(export_path).strip() and str(export_path).lower() != "null":
        export_path = Path(export_path)
    else:
        export_path = None

    use_traj = bool(sim_cfg.get("use_trajectory_regions", True))
    allow_fallback = bool(fallback_schematic and sim_cfg.get("fallback_to_schematic_hippocampus", True))

    meta: dict[str, Any] = {
        "source": "schematic_fallback",
        "used_fallback": False,
        "export_file": str(export_path) if export_path else None,
        "config": cfg,
    }

    if use_traj and export_path is not None:
        if not export_path.exists():
            if not allow_fallback:
                raise FileNotFoundError(f"Trajectory export not found: {export_path}")
            meta["used_fallback"] = True
            meta["source"] = "schematic_fallback"
            meta["warning"] = f"export missing: {export_path}"
            df = schematic_anatomy_table(
                region_segments, n_channels=n_channels, pitch_um=pitch_um, probe_type=probe_type,
            )
            return df, meta

        suffix = export_path.suffix.lower()
        if suffix == ".mat":
            df = _import_mat(
                export_path, n_channels=n_channels, pitch_um=pitch_um, probe_type=probe_type,
            )
        else:
            raw = _read_tabular(export_path)
            rows = raw.to_dict(orient="records")
            # Normalize common alternate column names.
            renamed_rows = []
            for r in rows:
                rr = dict(r)
                aliases = {
                    "z_start": "depth_start_um",
                    "z_end": "depth_end_um",
                    "depth_um_start": "depth_start_um",
                    "depth_um_end": "depth_end_um",
                    "area": "name",
                    "ccf_name": "name",
                    "structure": "name",
                    "ch_start": "channel_start",
                    "ch_end": "channel_end",
                }
                for src, dst in aliases.items():
                    if src in rr and dst not in rr:
                        rr[dst] = rr[src]
                renamed_rows.append(rr)
            df = _finalize_table(
                renamed_rows,
                n_channels=n_channels,
                pitch_um=pitch_um,
                probe_type=probe_type,
                insertion=traj_cfg,
                trajectory_source="neuropixels_trajectory_explorer",
            )
        meta["source"] = "neuropixels_trajectory_explorer"
        meta["used_fallback"] = False
        return df, meta

    if use_traj:
        manual = _manual_trajectory_from_config(traj_cfg, probe_cfg)
        if manual is not None and len(manual):
            meta["source"] = "manual_yaml"
            return manual, meta

    if not allow_fallback and export_path is None:
        raise ValueError(
            "No trajectory export/manual regions available and schematic fallback disabled"
        )

    meta["used_fallback"] = True
    meta["source"] = "schematic_fallback"
    df = schematic_anatomy_table(
        region_segments, n_channels=n_channels, pitch_um=pitch_um, probe_type=probe_type,
    )
    # Stamp insertion metadata from config when present.
    for col_src, col_dst in [
        ("insertion_ap_mm", "insertion_ap_mm"),
        ("insertion_ml_mm", "insertion_ml_mm"),
        ("insertion_depth_um", "insertion_depth_um"),
        ("angle_ap_deg", "insertion_angle_ap_deg"),
        ("angle_ml_deg", "insertion_angle_ml_deg"),
    ]:
        val = traj_cfg.get(col_src)
        if val is not None:
            df[col_dst] = val
    return df, meta


def validate_channel_assignment(df: pd.DataFrame, n_channels: int = 384) -> dict[str, Any]:
    """Check that every channel maps to at most one anatomical band."""
    ownership: dict[int, list[str]] = {ch: [] for ch in range(1, n_channels + 1)}
    for _, row in df.iterrows():
        ch0, ch1 = row.get("channel_start"), row.get("channel_end")
        if pd.isna(ch0) or pd.isna(ch1) or int(ch0) <= 0:
            continue
        label = f"{row.get('region')}:{row.get('layer')}"
        for ch in range(int(ch0), int(ch1) + 1):
            if 1 <= ch <= n_channels:
                ownership[ch].append(label)
    overlaps = {ch: labels for ch, labels in ownership.items() if len(labels) > 1}
    unassigned = [ch for ch, labels in ownership.items() if len(labels) == 0]
    return {
        "n_channels": n_channels,
        "n_overlapping_channels": len(overlaps),
        "overlaps": overlaps,
        "n_unassigned_channels": len(unassigned),
        "ok_zero_or_one": len(overlaps) == 0,
    }


def _parse_bool(val: Any, default: bool = True) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() not in {"false", "0", "no", "n"}


def load_lab_anatomy_regions_csv(
    path: Path | str,
    *,
    exclude_non_hippocampal: bool = True,
) -> pd.DataFrame:
    """Load a screenshot-/export-derived region-depth table (mm or µm)."""
    path = Path(path)
    df = pd.read_csv(path)
    # Normalize column names.
    rename = {
        "layer": "layer_or_area",
    }
    for src, dst in rename.items():
        if src in df.columns and dst not in df.columns:
            df = df.rename(columns={src: dst})

    if "depth_start_mm" in df.columns and "depth_end_mm" in df.columns:
        df["depth_start_um"] = df["depth_start_mm"].astype(float) * 1000.0
        df["depth_end_um"] = df["depth_end_mm"].astype(float) * 1000.0
    elif "depth_start_um" not in df.columns:
        raise ValueError(
            f"{path} needs depth_start_mm/depth_end_mm or depth_start_um/depth_end_um"
        )

    if "layer" not in df.columns:
        df["layer"] = df.get("layer_or_area", "unspecified")

    if "include_in_hippocampal_simulation" in df.columns:
        df["include_in_hippocampal_simulation"] = df["include_in_hippocampal_simulation"].map(
            lambda v: _parse_bool(v, True)
        )
    else:
        df["include_in_hippocampal_simulation"] = True

    if exclude_non_hippocampal:
        # Keep all rows in the written anatomy table, but mark excluded ones;
        # filtering for capture happens downstream. Visual cortex stays visible
        # on the trajectory figure.
        pass

    df["trajectory_source"] = "lab_trajectory_config"
    return df


def assign_channels_from_depth(
    anatomy_regions: pd.DataFrame,
    probe_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Convert depth bands to approximate channel ranges using site pitch.

    For NP2.0, ``site_pitch_um`` comes from config (typically 15 µm) and should
    be confirmed from the actual channel map.
    """
    import warnings

    probe_config = probe_config or {}
    pitch = float(probe_config.get("site_pitch_um") or 15.0)
    n_channels = int(
        probe_config.get("n_channels_recorded")
        or probe_config.get("n_channels")
        or 384
    )
    active_start = probe_config.get("active_channel_start")
    active_end = probe_config.get("active_channel_end")
    if probe_config.get("confirm_from_channel_map"):
        warnings.warn(
            f"Using site_pitch_um={pitch} for NP channel assignment; "
            "confirm from the actual channel map.",
            UserWarning,
            stacklevel=2,
        )

    out = anatomy_regions.copy()
    if "depth_start_um" not in out.columns and "depth_start_mm" in out.columns:
        out["depth_start_um"] = out["depth_start_mm"].astype(float) * 1000.0
        out["depth_end_um"] = out["depth_end_mm"].astype(float) * 1000.0

    ch_starts, ch_ends, n_chs, ch_labels = [], [], [], []
    for _, row in out.iterrows():
        z0 = float(row["depth_start_um"])
        z1 = float(row["depth_end_um"])
        depths = np.arange(n_channels) * pitch
        mask = (depths >= z0) & (depths < z1)
        idxs = np.where(mask)[0]
        if active_start is not None and active_end is not None:
            a0, a1 = int(active_start), int(active_end)
            idxs = idxs[(idxs + 1 >= a0) & (idxs + 1 <= a1)]
        if len(idxs) == 0:
            ch_starts.append(0)
            ch_ends.append(0)
            n_chs.append(0)
            ch_labels.append("")
        else:
            c0 = int(idxs[0]) + 1
            c1 = int(idxs[-1]) + 1
            ch_starts.append(c0)
            ch_ends.append(c1)
            n_chs.append(int(len(idxs)))
            ch_labels.append(f"{c0}-{c1}")

    out["channel_start"] = ch_starts
    out["channel_end"] = ch_ends
    out["n_channels"] = n_chs
    out["channels"] = ch_labels
    out["active_site_pitch_um"] = pitch
    if "probe_type" not in out.columns or out["probe_type"].isna().all():
        out["probe_type"] = probe_config.get("type", "NP2.0")
    return out
