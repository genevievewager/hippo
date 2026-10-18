"""Derived matched-arm configs (duration, density, equal-area arena)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.regions import LAB_PRIMARY, map_sim_region

# Exact condition-key list for the matched arm (Phase-1 decision 2).
CONDITION_KEYS: tuple[str, ...] = (
    "session_duration_s",          # = lab segment trimmed duration
    "unit_density_per_mm",         # raised so yield ≥ lab per-region counts; then subsample
    "arena_size_cm",               # equal-area square vs lab boundary polygon
    "behavior_fit",                # pending approval — speed/wall/occupancy targets only
)

REGION_DEPTH_MM: dict[str, float] = {
    # from configs/trajectories/lab_npx2_default_regions.csv
    "entorhinal_transition": 0.20,
    "hippocampal_formation_transition": 0.55,
    "entorhinal_cortex_layer6a": 0.75,
    "entorhinal_cortex_layer5": 0.30,
    "medial_entorhinal_layer3": 0.30,
    "medial_entorhinal_layer2": 0.25,
    "medial_entorhinal_layer1": 0.25,
}

MEC_BANDS = [k for k, v in {
    "entorhinal_transition": "MEC",
    "entorhinal_cortex_layer6a": "MEC",
    "entorhinal_cortex_layer5": "MEC",
    "medial_entorhinal_layer3": "MEC",
    "medial_entorhinal_layer2": "MEC",
    "medial_entorhinal_layer1": "MEC",
    "hippocampal_formation_transition": "SUB",
}.items() if v == "MEC"]

SUB_BANDS = ["hippocampal_formation_transition"]


def equal_area_arena_side_cm(width_cm: float, height_cm: float) -> float:
    """Side of a square with the same area as width×height rectangle/bbox."""
    return float(np.sqrt(max(float(width_cm), 0.0) * max(float(height_cm), 0.0)))


def _boundary_wh(cfg: dict[str, Any], room: str) -> tuple[float, float, float]:
    """Return width, height, area from boundary polygon (adapter-equivalent)."""
    from analysis.real_quadrant.adapter import _room_boundary

    _cx, _cy, width, height, _xmin, _ymin = _room_boundary(cfg, room)
    return float(width), float(height), float(width) * float(height)


def lab_segment_match_spec(
    session_dir: Path,
    room: str = "A",
    *,
    trim_start_s: float = 60.0,
    trim_end_s: float = 10.0,
    density_safety: float = 1.3,
) -> dict[str, Any]:
    """Compute matched-arm knobs for one lab room segment."""
    session_dir = Path(session_dir)
    with open(session_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    rooms = cfg["preprocessing"]["map_rooms"]["rooms"]
    if room not in rooms:
        raise KeyError(room)
    t0, t1 = [float(x) for x in rooms[room]["range"]]
    retained_s = max(0.0, (t1 - t0) - float(trim_start_s) - float(trim_end_s))
    width, height, area = _boundary_wh(cfg, room)
    arena = equal_area_arena_side_cm(width, height)

    units = pd.read_csv(
        session_dir / "clusters_dataset.csv",
        usecols=lambda c: c in ("Region", "BClabel"),
    )
    # Some sessions (e.g. 7394_18072024_2roomsB) omit BClabel; keep all rows.
    if "BClabel" in units.columns:
        units = units[units["BClabel"].astype(str) != "NON-SOMA"]
    regions = units["Region"].astype(str)
    required = {
        "MEC": int((regions == "MEC").sum()),
        "SUB": int((regions == "SUB").sum()),
    }

    # Baseline densities from default capture yaml
    base_path = Path("configs/trajectories/lab_npx2_default_cell_capture.yaml")
    base = yaml.safe_load(base_path.read_text())
    capture = copy.deepcopy(base["cell_capture"])

    def expected_for(lab_label: str, scale: float) -> float:
        bands = MEC_BANDS if lab_label == "MEC" else SUB_BANDS
        total = 0.0
        for b in bands:
            dens = float(capture[b]["unit_density_per_mm"]) * scale
            total += dens * float(REGION_DEPTH_MM[b])
        return total

    # Scale densities so E[yield] >= safety * required (Poisson mean)
    scales = {"MEC": 1.0, "SUB": 1.0}
    for lab_label in ("MEC", "SUB"):
        need = required[lab_label]
        if need <= 0:
            continue
        base_e = expected_for(lab_label, 1.0)
        if base_e <= 0:
            scales[lab_label] = float("inf")
            continue
        scales[lab_label] = max(1.0, (need * density_safety) / base_e)

    # Apply per-band: MEC bands use MEC scale; SUB band uses SUB scale
    for b in MEC_BANDS:
        capture[b]["unit_density_per_mm"] = float(
            capture[b]["unit_density_per_mm"] * scales["MEC"]
        )
    for b in SUB_BANDS:
        capture[b]["unit_density_per_mm"] = float(
            capture[b]["unit_density_per_mm"] * scales["SUB"]
        )
        # Ensure HPF band is included in decoder for Report 4 matching
        capture[b]["include"] = True

    expected = {
        "MEC": expected_for("MEC", 1.0),  # densities already scaled in capture
        "SUB": expected_for("SUB", 1.0),
    }
    # recompute expected with scaled densities already in capture
    expected = {"MEC": 0.0, "SUB": 0.0}
    for b in MEC_BANDS:
        expected["MEC"] += float(capture[b]["unit_density_per_mm"]) * REGION_DEPTH_MM[b]
    for b in SUB_BANDS:
        expected["SUB"] += float(capture[b]["unit_density_per_mm"]) * REGION_DEPTH_MM[b]

    return {
        "session": session_dir.name,
        "room": room,
        "segment_t0": t0,
        "segment_t1": t1,
        "session_duration_s": retained_s,
        "trim_start_s": trim_start_s,
        "trim_end_s": trim_end_s,
        "boundary_width_cm": width,
        "boundary_height_cm": height,
        "boundary_area_cm2": area,
        "arena_size_cm": arena,
        "shape_confound": {
            "lab": "boundary polygon",
            "sim": "equal-area square",
            "aspect_lab": width / height if height else None,
            "aspect_sim": 1.0,
        },
        "required_counts": required,
        "density_scales": scales,
        "expected_yield": expected,
        "cell_capture": capture,
        "condition_keys": list(CONDITION_KEYS),
        "behavior_fit": "pending_approval",
    }


def build_matched_capture_yaml(spec: dict[str, Any], dest: Path) -> Path:
    dest = assert_dq_path(dest, kind="capture_yaml")
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cell_capture": spec["cell_capture"],
        "recording_capture": yaml.safe_load(
            Path("configs/trajectories/lab_npx2_default_cell_capture.yaml").read_text()
        ).get("recording_capture"),
        "data_quality_match": {
            k: spec[k]
            for k in (
                "session",
                "room",
                "session_duration_s",
                "arena_size_cm",
                "required_counts",
                "expected_yield",
                "density_scales",
                "shape_confound",
            )
        },
    }
    dest.write_text(yaml.safe_dump(payload, sort_keys=False))
    return dest


def write_condition_keys(path: Path | None = None) -> Path:
    path = assert_dq_path(
        path or (DATA_QUALITY_ROOT / "condition_keys" / "matched_arm.json"),
        kind="condition_keys",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"condition_keys": list(CONDITION_KEYS)}, indent=2) + "\n")
    return path
