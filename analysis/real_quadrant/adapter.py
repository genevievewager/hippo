"""Load one room segment from HIPPO_DATA_ROOT into an analyze_source bundle.

Hard rule: never use dataset Cell_*, spike_rate_dataset.csv, or
dataset_polar.csv as decoder features (centre / non-causal). Cell_* is only
for the load-time integrity check.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from realtime.quadrant_n5_run import (
    SEGMENT_TRIM_END_S,
    SEGMENT_TRIM_START_S,
    prepared_source_bundle,
    segment_retained_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = REPO_ROOT / "outputs" / "real_quadrant" / "cache"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "outputs" / "real_quadrant" / "manifest.jsonl"

WINDOW_S = 0.250
UPDATE_DT = 0.050
CENTRE_HALF = 0.125
MIN_UNITS = 30
MIN_VALID_FRAC = 0.7

# Columns that must never be used as decoder features.
FORBIDDEN_FEATURE_FILES = (
    "spike_rate_dataset.csv",
    "dataset_polar.csv",
)


class IntegrityError(RuntimeError):
    """Centre-window rebuild did not match Cell_* exactly."""


def _require_data_root() -> Path:
    root = os.environ.get("HIPPO_DATA_ROOT", "").strip()
    if not root:
        raise EnvironmentError("HIPPO_DATA_ROOT is not set")
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"HIPPO_DATA_ROOT not a directory: {path}")
    return path


def _parse_spike_times(cell: Any) -> np.ndarray:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return np.zeros(0, dtype=float)
    if isinstance(cell, str):
        if not cell.strip():
            return np.zeros(0, dtype=float)
        return np.fromstring(cell, sep=",", dtype=float)
    return np.asarray(cell, dtype=float).ravel()


def window_count_matrix(
    spike_times: list[np.ndarray],
    grid: np.ndarray,
    lo_off: float,
    hi_off: float,
) -> np.ndarray:
    """Count spikes in [t+lo_off, t+hi_off) for each unit and grid time."""
    grid = np.asarray(grid, dtype=float)
    n_t = len(grid)
    n_u = len(spike_times)
    out = np.zeros((n_t, n_u), dtype=np.int32)
    left_edges = grid + float(lo_off)
    right_edges = grid + float(hi_off)
    for j, st in enumerate(spike_times):
        if st.size == 0:
            continue
        st = np.sort(np.asarray(st, dtype=float))
        left = np.searchsorted(st, left_edges, side="left")
        right = np.searchsorted(st, right_edges, side="left")
        out[:, j] = (right - left).astype(np.int32)
    return out


def causal_count_matrix(
    spike_times: list[np.ndarray],
    grid: np.ndarray,
    *,
    window_s: float = WINDOW_S,
) -> np.ndarray:
    """Causal features: spikes in [t-window_s, t)."""
    return window_count_matrix(spike_times, grid, -float(window_s), 0.0)


def centre_window_match_fraction(
    spike_times: list[np.ndarray],
    grid: np.ndarray,
    cell_matrix: np.ndarray,
    *,
    half_width_s: float = CENTRE_HALF,
) -> float:
    """Fraction of (time, unit) bins where rebuilt centre window equals Cell_*."""
    rebuilt = window_count_matrix(
        spike_times, grid, -float(half_width_s), float(half_width_s),
    )
    truth = np.rint(np.asarray(cell_matrix, dtype=float)).astype(np.int32)
    if rebuilt.shape != truth.shape:
        raise ValueError(
            f"shape mismatch rebuilt={rebuilt.shape} cell={truth.shape}"
        )
    return float(np.mean(rebuilt == truth))


def _file_fingerprint(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:24]


def load_units_and_spikes(
    session_dir: Path,
) -> tuple[list[int], list[np.ndarray], pd.DataFrame]:
    """Parse clusters with usecols only; drop BClabel NON-SOMA."""
    clusters_path = session_dir / "clusters_dataset.csv"
    if not clusters_path.is_file():
        raise FileNotFoundError(clusters_path)
    # One row per cell; timestamp strings are large — never load unused cols.
    usecols = ["cell", "timestamp", "BClabel", "Region"]
    hdr = pd.read_csv(clusters_path, nrows=0)
    usecols = [c for c in usecols if c in hdr.columns]
    if "cell" not in usecols or "timestamp" not in usecols:
        raise ValueError("clusters_dataset.csv must have cell and timestamp")
    ds_hdr = pd.read_csv(session_dir / "dataset.csv", nrows=0)
    cell_cols = {c for c in ds_hdr.columns if c.startswith("Cell_")}

    unit_ids: list[int] = []
    spike_times: list[np.ndarray] = []
    regions: list[str] = []
    labels: list[str] = []
    for chunk in pd.read_csv(clusters_path, usecols=usecols, chunksize=64):
        for _, row in chunk.iterrows():
            cid = int(row["cell"])
            if f"Cell_{cid}" not in cell_cols:
                continue
            bcl = str(row["BClabel"]) if "BClabel" in row.index else "GOOD"
            if bcl == "NON-SOMA":
                continue
            unit_ids.append(cid)
            spike_times.append(_parse_spike_times(row["timestamp"]))
            regions.append(str(row["Region"]) if "Region" in row.index else "unknown")
            labels.append(bcl)
    if not unit_ids:
        raise RuntimeError(f"no units after NON-SOMA filter in {session_dir.name}")
    units_df = pd.DataFrame({
        "unit_id": unit_ids,
        "region": regions,
        "cell_type": labels,
        "BClabel": labels,
    })
    return unit_ids, spike_times, units_df


def _room_boundary(
    cfg: dict[str, Any], room: str,
) -> tuple[float, float, float, float, float, float]:
    """Return centre_x, centre_y, width, height, xmin, ymin from boundary polygon."""
    mr = cfg["preprocessing"]["map_rooms"]
    idx_map = {int(k): str(v) for k, v in (mr.get("index") or {}).items()}
    # Prefer matching Room index for this label; lowercase rooms share geometry
    # with their uppercase counterpart when not in index.
    room_idx = None
    for i, lab in idx_map.items():
        if lab == room:
            room_idx = i
            break
    if room_idx is None:
        # e.g. room 'a' → use 'A' polygon
        alt = room.upper() if room != room.upper() else room
        for i, lab in idx_map.items():
            if lab == alt:
                room_idx = i
                break
    if room_idx is None:
        raise KeyError(f"no boundary Room index for room={room!r}")
    bdf = pd.DataFrame(cfg["preprocessing"]["boundary"])
    sub = bdf[bdf["Room"].astype(int) == int(room_idx)]
    if sub.empty:
        raise KeyError(f"empty boundary for room={room!r}")
    xmin, xmax = float(sub["X"].min()), float(sub["X"].max())
    ymin, ymax = float(sub["Y"].min()), float(sub["Y"].max())
    return (
        0.5 * (xmin + xmax),
        0.5 * (ymin + ymax),
        xmax - xmin,
        ymax - ymin,
        xmin,
        ymin,
    )


def _append_manifest(record: dict[str, Any], path: Path = DEFAULT_MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def build_segment_bundle(
    session_name: str,
    room: str = "A",
    *,
    data_root: Path | None = None,
    cache_root: Path | None = None,
    window_s: float = WINDOW_S,
    update_dt: float = UPDATE_DT,
    trim_start_s: float = SEGMENT_TRIM_START_S,
    trim_end_s: float = SEGMENT_TRIM_END_S,
    min_units: int = MIN_UNITS,
    min_valid_frac: float = MIN_VALID_FRAC,
) -> dict[str, Any]:
    """Build a prepared analyze_source bundle for one room segment.

    Raises IntegrityError if the centre-window rebuild ≠ Cell_* (exact).
    Raises RuntimeError for pre-registered exclusions (logged to manifest).
    """
    root = Path(data_root) if data_root is not None else _require_data_root()
    session_dir = root / session_name
    if not session_dir.is_dir():
        raise FileNotFoundError(session_dir)
    for forbidden in FORBIDDEN_FEATURE_FILES:
        # Presence is fine; using them as features is forbidden (enforced by
        # never reading them here).
        _ = (session_dir / forbidden).is_file()

    with open(session_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    rooms = cfg["preprocessing"]["map_rooms"]["rooms"]
    if room not in rooms:
        raise KeyError(f"room {room!r} not in map_rooms for session")
    t0, t1 = [float(x) for x in rooms[room]["range"]]
    cx, cy, width, height, _xmin, _ymin = _room_boundary(cfg, room)

    unit_ids, spike_times, units_df = load_units_and_spikes(session_dir)
    if len(unit_ids) < int(min_units):
        reason = {
            "session": session_name,
            "room": room,
            "status": "excluded",
            "reason": "min_units",
            "n_units": len(unit_ids),
            "min_units": int(min_units),
        }
        _append_manifest(reason)
        raise RuntimeError(
            f"excluded {session_name} room={room}: n_units={len(unit_ids)} < {min_units}"
        )

    # Behavior grid from dataset timestamps (not Cell_* as features).
    ds_cols = ["timestamp", "X", "Y", "valid", "room"] + [
        f"Cell_{u}" for u in unit_ids
    ]
    ds = pd.read_csv(session_dir / "dataset.csv", usecols=ds_cols)
    grid_all = ds["timestamp"].to_numpy(dtype=float)
    # Restrict to segment time range (inclusive start, exclusive end-ish).
    seg = (grid_all >= t0) & (grid_all < t1)
    if not seg.any():
        raise RuntimeError(f"no dataset rows in segment {room} [{t0}, {t1})")
    grid = grid_all[seg]
    y_global = np.column_stack([
        ds.loc[seg, "X"].to_numpy(dtype=float),
        ds.loc[seg, "Y"].to_numpy(dtype=float),
    ])
    y = y_global.copy()
    y[:, 0] = y[:, 0] - cx
    y[:, 1] = y[:, 1] - cy
    target_valid = ds.loc[seg, "valid"].to_numpy()
    if target_valid.dtype != bool:
        target_valid = pd.to_numeric(target_valid, errors="coerce").fillna(0).to_numpy() != 0
    # Non-finite or invalid targets are masked for fit/eval only (counts stay in X).
    target_valid = target_valid & np.isfinite(y).all(axis=1)
    y = y.astype(float, copy=True)
    y[~target_valid] = np.nan

    cell_mat = np.column_stack([
        ds.loc[seg, f"Cell_{u}"].to_numpy(dtype=float) for u in unit_ids
    ])

    # Integrity: centre window must match Cell_* exactly (not used as features).
    match_frac = centre_window_match_fraction(spike_times, grid, cell_mat)
    if match_frac < 1.0 - 1e-15:
        reason = {
            "session": session_name,
            "room": room,
            "status": "integrity_fail",
            "match_fraction": match_frac,
        }
        _append_manifest(reason)
        raise IntegrityError(
            f"centre-window integrity failed for {session_name} room={room}: "
            f"match_fraction={match_frac:.6f}"
        )

    cache_root = Path(cache_root) if cache_root is not None else DEFAULT_CACHE_ROOT
    cache_root.mkdir(parents=True, exist_ok=True)
    key_payload = {
        "clusters": _file_fingerprint(session_dir / "clusters_dataset.csv"),
        "dataset": _file_fingerprint(session_dir / "dataset.csv"),
        "unit_ids": unit_ids,
        "window_s": float(window_s),
        "update_dt": float(update_dt),
        "room": room,
        "t0": t0,
        "t1": t1,
        "kind": "causal_counts",
    }
    key = _cache_key(key_payload)
    cache_path = cache_root / f"{key}.npz"
    if cache_path.is_file():
        blob = np.load(cache_path)
        X_counts = np.asarray(blob["X_counts"], dtype=float)
        if X_counts.shape != (len(grid), len(unit_ids)):
            X_counts = causal_count_matrix(
                spike_times, grid, window_s=window_s,
            ).astype(float)
            np.savez_compressed(cache_path, X_counts=X_counts)
    else:
        X_counts = causal_count_matrix(
            spike_times, grid, window_s=window_s,
        ).astype(float)
        np.savez_compressed(cache_path, X_counts=X_counts)

    retained = segment_retained_mask(
        grid, t0, t1, trim_start_s=trim_start_s, trim_end_s=trim_end_s,
    )
    if not retained.any():
        reason = {
            "session": session_name,
            "room": room,
            "status": "excluded",
            "reason": "empty_retained",
        }
        _append_manifest(reason)
        raise RuntimeError(f"excluded {session_name} room={room}: empty retained")
    valid_frac = float(np.mean(target_valid[retained]))
    if valid_frac < float(min_valid_frac):
        reason = {
            "session": session_name,
            "room": room,
            "status": "excluded",
            "reason": "min_valid_frac",
            "valid_frac": valid_frac,
            "min_valid_frac": float(min_valid_frac),
        }
        _append_manifest(reason)
        raise RuntimeError(
            f"excluded {session_name} room={room}: valid_frac={valid_frac:.3f} "
            f"< {min_valid_frac}"
        )

    arena_cm = float(max(width, height))
    bundle = prepared_source_bundle(
        X_counts=X_counts,
        y=y,
        decode_times=grid,
        unit_ids=unit_ids,
        units_df=units_df,
        arena_cm=arena_cm,
        segment_t0=t0,
        segment_t1=t1,
        target_valid=target_valid,
        arena_width_cm=float(width),
        arena_height_cm=float(height),
        apply_segment_trims=True,
        y_is_room_local=True,
        meta={
            "session": session_name,
            "room": room,
            "integrity_match_fraction": match_frac,
            "n_units": len(unit_ids),
            "valid_frac_retained": valid_frac,
            "trim_start_s": float(trim_start_s),
            "trim_end_s": float(trim_end_s),
            "cache_key": key,
            "window_s": float(window_s),
            "update_dt": float(update_dt),
        },
    )
    _append_manifest({
        "session": session_name,
        "room": room,
        "status": "ok",
        "n_units": len(unit_ids),
        "n_times": int(len(grid)),
        "valid_frac_retained": valid_frac,
        "integrity_match_fraction": match_frac,
        "cache_key": key,
    })
    return bundle
