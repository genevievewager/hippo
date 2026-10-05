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


class ExtentError(RuntimeError):
    """Retained path extent is not comparable to the room boundary polygon."""


class RateMapStabilityError(RuntimeError):
    """Train split-half rate-map stability is below the load-time threshold."""


# Retained tracking bbox must be this fraction of the boundary width/height.
EXTENT_FRAC_LO = 0.50
EXTENT_FRAC_HI = 1.10

# Load-time place-field sanity: on the training part of a segment, split-half
# rate-map stability (8×8 bins, ≥40 samples/bin) must clear this fraction of
# scored units with Pearson corr > 0.5. Below → warn, flag, refuse decode.
RATEMAP_N_BINS = 8
RATEMAP_MIN_SAMPLES_PER_BIN = 40
RATEMAP_CORR_THRESH = 0.5
RATEMAP_MIN_FRAC_ABOVE = 0.05
TRAIN_FRAC_FOR_STABILITY = 0.80


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
    """Return centre_x, centre_y, width, height, xmin, ymin from boundary polygon.

    Boundary coordinates are centimetres in the same frame as
    ``postions_dataset.csv``.
    """
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


def split_half_rate_map_stability(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    n_bins: int = RATEMAP_N_BINS,
    min_samples: int = RATEMAP_MIN_SAMPLES_PER_BIN,
    corr_thresh: float = RATEMAP_CORR_THRESH,
) -> dict[str, float | int]:
    """Per-unit Pearson corr of rate maps between first/second half of ``mask``.

    Bins are an ``n_bins``×``n_bins`` grid over the finite extent of ``y[mask]``.
    A bin contributes only when both halves have ≥ ``min_samples`` visits.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    idx = np.where(mask)[0]
    empty = {
        "n_scored_units": 0,
        "n_stable_units": 0,
        "median_corr": float("nan"),
        "frac_gt_thresh": float("nan"),
        "corr_thresh": float(corr_thresh),
    }
    if idx.size < 2 * int(min_samples):
        return empty
    mid = idx.size // 2
    halves = [idx[:mid], idx[mid:]]
    yf = y[mask]
    yf = yf[np.isfinite(yf).all(axis=1)]
    if len(yf) < int(min_samples):
        return empty
    xedges = np.linspace(float(yf[:, 0].min()), float(yf[:, 0].max()), int(n_bins) + 1)
    yedges = np.linspace(float(yf[:, 1].min()), float(yf[:, 1].max()), int(n_bins) + 1)
    maps: list[np.ndarray] = []
    for h in halves:
        yh = y[h]
        Xh = X[h]
        finite = np.isfinite(yh).all(axis=1)
        yh, Xh = yh[finite], Xh[finite]
        if len(yh) < int(min_samples):
            return empty
        xi = np.clip(np.digitize(yh[:, 0], xedges) - 1, 0, int(n_bins) - 1)
        yi = np.clip(np.digitize(yh[:, 1], yedges) - 1, 0, int(n_bins) - 1)
        rmap = np.full((Xh.shape[1], int(n_bins), int(n_bins)), np.nan)
        for bx in range(int(n_bins)):
            for by in range(int(n_bins)):
                m = (xi == bx) & (yi == by)
                if int(m.sum()) >= int(min_samples):
                    rmap[:, bx, by] = Xh[m].mean(axis=0)
        maps.append(rmap)
    corrs: list[float] = []
    for u in range(maps[0].shape[0]):
        a = maps[0][u].ravel()
        b = maps[1][u].ravel()
        m = np.isfinite(a) & np.isfinite(b)
        if int(m.sum()) < 4:
            continue
        if float(np.std(a[m])) < 1e-12 or float(np.std(b[m])) < 1e-12:
            continue
        corrs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    if not corrs:
        return empty
    arr = np.asarray(corrs, dtype=float)
    n_stable = int(np.sum(arr > float(corr_thresh)))
    return {
        "n_scored_units": int(arr.size),
        "n_stable_units": n_stable,
        "median_corr": float(np.median(arr)),
        "frac_gt_thresh": float(n_stable / arr.size),
        "corr_thresh": float(corr_thresh),
    }


def assert_path_extent_matches_boundary(
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    boundary_width_cm: float,
    boundary_height_cm: float,
    *,
    frac_lo: float = EXTENT_FRAC_LO,
    frac_hi: float = EXTENT_FRAC_HI,
) -> dict[str, float]:
    """Require retained path bbox ≈ boundary size (catches normalized coords)."""
    x = np.asarray(x_cm, dtype=float)
    y = np.asarray(y_cm, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if not m.any():
        raise ExtentError("no finite positions for extent check")
    path_w = float(np.nanmax(x[m]) - np.nanmin(x[m]))
    path_h = float(np.nanmax(y[m]) - np.nanmin(y[m]))
    bw = float(boundary_width_cm)
    bh = float(boundary_height_cm)
    if bw <= 0 or bh <= 0:
        raise ExtentError(f"invalid boundary size width={bw} height={bh}")
    wr, hr = path_w / bw, path_h / bh
    stats = {
        "path_width_cm": path_w,
        "path_height_cm": path_h,
        "boundary_width_cm": bw,
        "boundary_height_cm": bh,
        "width_ratio": wr,
        "height_ratio": hr,
    }
    if not (frac_lo <= wr <= frac_hi and frac_lo <= hr <= frac_hi):
        raise ExtentError(
            "retained path extent not comparable to boundary polygon: "
            f"width_ratio={wr:.3f} height_ratio={hr:.3f} "
            f"(require {frac_lo:.2f}–{frac_hi:.2f} of boundary); "
            f"path=({path_w:.2f}×{path_h:.2f}) cm, "
            f"boundary=({bw:.2f}×{bh:.2f}) cm. "
            "Check that targets come from postions_dataset.csv in cm, "
            "not dataset.csv normalized X/Y."
        )
    return stats


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
    require_ratemap_stability: bool = True,
    min_ratemap_frac_above: float = RATEMAP_MIN_FRAC_ABOVE,
) -> dict[str, Any]:
    """Build a prepared analyze_source bundle for one room segment.

    Raises IntegrityError if the centre-window rebuild ≠ Cell_* (exact).
    Raises RuntimeError for pre-registered exclusions (logged to manifest).
    Raises RateMapStabilityError when train split-half rate-map stability is
    below ``min_ratemap_frac_above`` (logged + flagged; decode refused).
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

    # Neural grid + Cell_* (integrity only) from dataset.csv.
    # Targets MUST come from postions_dataset.csv (cm). dataset.csv X/Y are
    # normalized (~0–1) and must never be used as decoder targets.
    pos_path = session_dir / "postions_dataset.csv"
    if not pos_path.is_file():
        raise FileNotFoundError(pos_path)
    ds_cols = ["timestamp", "valid"] + [f"Cell_{u}" for u in unit_ids]
    ds = pd.read_csv(session_dir / "dataset.csv", usecols=ds_cols)
    pos = pd.read_csv(pos_path, usecols=["timestamp", "X", "Y", "valid", "room"])
    # Align positions onto the neural timestamp grid.
    aligned = ds[["timestamp", "valid"]].merge(
        pos.rename(columns={"valid": "pos_valid"}),
        on="timestamp",
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(ds):
        raise RuntimeError(
            f"timestamp alignment failed: dataset n={len(ds)} aligned n={len(aligned)}"
        )
    grid_all = aligned["timestamp"].to_numpy(dtype=float)
    seg = (grid_all >= t0) & (grid_all < t1)
    if not seg.any():
        raise RuntimeError(f"no dataset rows in segment {room} [{t0}, {t1})")
    grid = grid_all[seg]
    y_global = np.column_stack([
        aligned.loc[seg, "X"].to_numpy(dtype=float),
        aligned.loc[seg, "Y"].to_numpy(dtype=float),
    ])
    # Room-local cm: origin at boundary-polygon centre.
    y = y_global.astype(float, copy=True)
    y[:, 0] = y[:, 0] - cx
    y[:, 1] = y[:, 1] - cy
    ds_valid = aligned.loc[seg, "valid"].to_numpy()
    if ds_valid.dtype != bool:
        ds_valid = pd.to_numeric(ds_valid, errors="coerce").fillna(0).to_numpy() != 0
    pos_valid = aligned.loc[seg, "pos_valid"].to_numpy()
    if pos_valid.dtype != bool:
        pos_valid = pd.to_numeric(pos_valid, errors="coerce").fillna(0).to_numpy() != 0
    # Prefer positions valid flag; also require finite cm coords.
    target_valid = pos_valid & ds_valid & np.isfinite(y_global).all(axis=1)
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
    # Extent check on retained + valid path in global cm (pre room-local).
    try:
        extent_stats = assert_path_extent_matches_boundary(
            y_global[retained & target_valid, 0],
            y_global[retained & target_valid, 1],
            width,
            height,
        )
    except ExtentError as exc:
        reason = {
            "session": session_name,
            "room": room,
            "status": "excluded",
            "reason": "path_extent_mismatch",
            "detail": str(exc),
        }
        _append_manifest(reason)
        raise

    # Load-time place-field sanity on the training part of the retained segment.
    from realtime.quadrant_n5_run import causal_train_test_split

    train_mask, _test_mask = causal_train_test_split(
        grid, TRAIN_FRAC_FOR_STABILITY, gap_s=max(float(window_s), 1.0),
    )
    train_part = (
        retained & target_valid & train_mask & np.isfinite(y).all(axis=1)
    )
    ratemap = split_half_rate_map_stability(X_counts, y, train_part)
    frac = ratemap.get("frac_gt_thresh")
    frac_f = float(frac) if frac is not None and np.isfinite(frac) else 0.0
    ratemap_flagged = frac_f < float(min_ratemap_frac_above)
    if ratemap_flagged:
        warn = {
            "session": session_name,
            "room": room,
            "status": "ratemap_stability_flag",
            "frac_gt_thresh": frac_f,
            "min_frac": float(min_ratemap_frac_above),
            "median_corr": ratemap.get("median_corr"),
            "n_scored_units": ratemap.get("n_scored_units"),
            "n_stable_units": ratemap.get("n_stable_units"),
        }
        _append_manifest(warn)
        print(
            f"WARNING: rate-map stability low for {session_name} room={room}: "
            f"frac_gt_{RATEMAP_CORR_THRESH}={frac_f:.4f} "
            f"(min={min_ratemap_frac_above}); n_stable="
            f"{ratemap.get('n_stable_units')}/"
            f"{ratemap.get('n_scored_units')}",
            flush=True,
        )
        if require_ratemap_stability:
            raise RateMapStabilityError(
                f"refusing decode for {session_name} room={room}: "
                f"train split-half rate-map frac_gt_{RATEMAP_CORR_THRESH}="
                f"{frac_f:.4f} < {min_ratemap_frac_above}"
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
            "target_source": "postions_dataset.csv",
            "target_units": "cm",
            "extent": extent_stats,
            "ratemap_stability": {
                **ratemap,
                "min_frac_above": float(min_ratemap_frac_above),
                "flagged": bool(ratemap_flagged),
                "train_frac": float(TRAIN_FRAC_FOR_STABILITY),
            },
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
        "ratemap_stability_frac_gt_thresh": frac_f,
        "ratemap_stability_flagged": bool(ratemap_flagged),
    })
    return bundle
