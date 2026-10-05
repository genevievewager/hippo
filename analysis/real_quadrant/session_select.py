"""Fixed rules for choosing real-data sessions (no hand-picking)."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from analysis.real_quadrant.adapter import (
    DEFAULT_MANIFEST_PATH,
    MIN_UNITS,
    MIN_VALID_FRAC,
    RATEMAP_CORR_THRESH,
    RATEMAP_MIN_FRAC_ABOVE,
    RATEMAP_MIN_SAMPLES_PER_BIN,
    RATEMAP_N_BINS,
    TRAIN_FRAC_FOR_STABILITY,
    _room_boundary,
    assert_path_extent_matches_boundary,
    load_units_and_spikes,
    split_half_rate_map_stability,
)
from realtime.quadrant_n5_run import causal_train_test_split, segment_retained_mask

REPO_ROOT = Path(__file__).resolve().parents[2]

M1_RULE = "lexicographically first directory name among 2rooms sessions"
M2_RULE = (
    "2rooms sessions with room A that pass pre-registered exclusions "
    "(min 30 units after NON-SOMA, min 0.7 valid fraction on retained, "
    "path extent 50–110% of boundary, train split-half rate-map stability "
    f"frac>{RATEMAP_CORR_THRESH} ≥ {RATEMAP_MIN_FRAC_ABOVE} on dataset "
    "Cell_* + postions); choose the session whose unit count is closest "
    "to the median; ties → lexicographically first name"
)


def _data_root(data_root: Path | None = None) -> Path:
    if data_root is not None:
        return Path(data_root)
    root = os.environ.get("HIPPO_DATA_ROOT", "").strip()
    if not root:
        raise EnvironmentError("HIPPO_DATA_ROOT is not set")
    return Path(root)


def list_2room_sessions(data_root: Path | None = None) -> list[str]:
    root = _data_root(data_root)
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and "2rooms" in p.name
    )


def m1_lex_first_session(data_root: Path | None = None) -> str:
    sessions = list_2room_sessions(data_root)
    if not sessions:
        raise RuntimeError("no 2rooms sessions")
    return sessions[0]


def _passes_room_a_exclusions(
    session_dir: Path,
) -> tuple[bool, int | None, str | None]:
    """Return (ok, n_units, exclude_reason)."""
    try:
        with open(session_dir / "config.yaml") as f:
            cfg = yaml.safe_load(f)
        rooms = cfg["preprocessing"]["map_rooms"]["rooms"]
        if "A" not in rooms:
            return False, None, "no_room_A"
        t0, t1 = [float(x) for x in rooms["A"]["range"]]
        unit_ids, _, _ = load_units_and_spikes(session_dir)
        n_u = len(unit_ids)
        if n_u < MIN_UNITS:
            return False, n_u, "min_units"
        pos = pd.read_csv(
            session_dir / "postions_dataset.csv",
            usecols=["timestamp", "X", "Y", "valid"],
        )
        seg = (pos.timestamp >= t0) & (pos.timestamp < t1)
        g = pos.loc[seg]
        times = g.timestamp.to_numpy(dtype=float)
        retained = segment_retained_mask(times, t0, t1)
        vv = g.valid.to_numpy()
        if vv.dtype != bool:
            vv = pd.to_numeric(g.valid, errors="coerce").fillna(0).to_numpy() != 0
        x = g.X.to_numpy(dtype=float)
        y = g.Y.to_numpy(dtype=float)
        m = retained & vv & np.isfinite(x) & np.isfinite(y)
        if not retained.any() or not m.any():
            return False, n_u, "empty_retained"
        valid_frac = float(m.sum() / retained.sum())
        if valid_frac < MIN_VALID_FRAC:
            return False, n_u, "min_valid_frac"
        _cx, _cy, w, h, _xmin, _ymin = _room_boundary(cfg, "A")
        assert_path_extent_matches_boundary(x[m], y[m], w, h)
        # Cheap place-field proxy for selection: dataset Cell_* vs postions.
        # Adapter still enforces the same threshold on causal features at load.
        ds = pd.read_csv(
            session_dir / "dataset.csv",
            usecols=["timestamp"] + [f"Cell_{u}" for u in unit_ids],
        )
        joined = g[["timestamp", "X", "Y", "valid"]].merge(
            ds, on="timestamp", how="inner",
        )
        if len(joined) < 100:
            return False, n_u, "ratemap_stability"
        X_cell = np.column_stack([
            joined[f"Cell_{u}"].to_numpy(dtype=float) for u in unit_ids
        ])
        y_cm = np.column_stack([
            joined["X"].to_numpy(dtype=float),
            joined["Y"].to_numpy(dtype=float),
        ])
        t_j = joined["timestamp"].to_numpy(dtype=float)
        vv_j = joined["valid"].to_numpy()
        if vv_j.dtype != bool:
            vv_j = pd.to_numeric(joined["valid"], errors="coerce").fillna(0).to_numpy() != 0
        retained_j = segment_retained_mask(t_j, t0, t1)
        train_mask, _ = causal_train_test_split(
            t_j, TRAIN_FRAC_FOR_STABILITY, gap_s=1.0,
        )
        train_part = (
            retained_j & vv_j & train_mask & np.isfinite(y_cm).all(axis=1)
        )
        stab = split_half_rate_map_stability(
            X_cell,
            y_cm,
            train_part,
            n_bins=RATEMAP_N_BINS,
            min_samples=RATEMAP_MIN_SAMPLES_PER_BIN,
            corr_thresh=RATEMAP_CORR_THRESH,
        )
        frac = stab.get("frac_gt_thresh")
        frac_f = float(frac) if frac is not None and np.isfinite(frac) else 0.0
        if frac_f < float(RATEMAP_MIN_FRAC_ABOVE):
            return False, n_u, "ratemap_stability"
        return True, n_u, None
    except Exception as exc:
        return False, None, type(exc).__name__


def select_median_units_2room_session(
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Apply M2_RULE; return selection record (includes session name for local use)."""
    root = _data_root(data_root)
    sessions = list_2room_sessions(root)
    m1 = sessions[0] if sessions else None
    eligible: list[tuple[int, str]] = []
    excluded: list[str] = []
    for name in sessions:
        ok, n_u, reason = _passes_room_a_exclusions(root / name)
        if ok and n_u is not None:
            eligible.append((int(n_u), name))
        else:
            excluded.append(reason or "unknown")
    if not eligible:
        raise RuntimeError("no eligible 2rooms sessions under selection rule")
    units = np.asarray([n for n, _ in eligible], dtype=float)
    med = float(np.median(units))
    eligible_sorted = sorted(eligible, key=lambda x: (abs(x[0] - med), x[1]))
    n_sel, name_sel = eligible_sorted[0]
    return {
        "rule": M2_RULE,
        "m1_rule": M1_RULE,
        "m1_session": m1,
        "selected_session": name_sel,
        "selected_n_units": int(n_sel),
        "median_units": med,
        "n_2rooms": len(sessions),
        "n_eligible": len(eligible),
        "exclude_reason_counts": dict(Counter(excluded)),
        "same_as_m1_lex_first": bool(name_sel == m1),
        "unit_count_range": [int(units.min()), int(units.max())],
    }


def append_selection_manifest(
    record: dict[str, Any],
    *,
    path: Path = DEFAULT_MANIFEST_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"kind": "session_selection", **record}, default=str) + "\n")


def animal_id_from_session(session_name: str) -> str:
    """Animal ID = first underscore-separated token of the session directory name."""
    return str(session_name).split("_", 1)[0]


M3_RULE = (
    "Per animal among 2rooms sessions that pass all M2 exclusions (min units, "
    "valid frac, path extent, train split-half rate-map stability): select the "
    "2 sessions whose unit counts are closest to that animal's median eligible "
    "unit count; ties → lexicographically first session name. If an animal has "
    "fewer than 2 eligible sessions, use all eligible and record the shortfall. "
    "Room A only."
)


def select_m3_cohort(
    data_root: Path | None = None,
    *,
    n_per_animal: int = 2,
) -> dict[str, Any]:
    """Apply M3_RULE; return cohort record (local use only — never commit)."""
    root = _data_root(data_root)
    sessions = list_2room_sessions(root)
    by_animal: dict[str, list[tuple[int, str]]] = {}
    excluded_by_animal: dict[str, list[dict[str, Any]]] = {}
    for name in sessions:
        animal = animal_id_from_session(name)
        ok, n_u, reason = _passes_room_a_exclusions(root / name)
        if ok and n_u is not None:
            by_animal.setdefault(animal, []).append((int(n_u), name))
        else:
            excluded_by_animal.setdefault(animal, []).append({
                "session": name,
                "reason": reason or "unknown",
                "n_units": n_u,
            })

    selected: list[dict[str, Any]] = []
    animals_out: list[dict[str, Any]] = []
    for animal in sorted(by_animal.keys() | excluded_by_animal.keys()):
        eligible = by_animal.get(animal, [])
        excluded = excluded_by_animal.get(animal, [])
        reason_counts = dict(Counter(e["reason"] for e in excluded))
        if not eligible:
            animals_out.append({
                "animal": animal,
                "n_eligible": 0,
                "n_selected": 0,
                "n_excluded": len(excluded),
                "exclude_reason_counts": reason_counts,
                "median_units": None,
                "selected": [],
                "note": "no eligible sessions",
            })
            continue
        units = np.asarray([n for n, _ in eligible], dtype=float)
        med = float(np.median(units))
        ranked = sorted(eligible, key=lambda x: (abs(x[0] - med), x[1]))
        pick = ranked[: int(n_per_animal)]
        note = None
        if len(eligible) < int(n_per_animal):
            note = (
                f"fewer than {n_per_animal} eligible "
                f"({len(eligible)}); using all"
            )
        animal_rec = {
            "animal": animal,
            "n_eligible": len(eligible),
            "n_selected": len(pick),
            "n_excluded": len(excluded),
            "exclude_reason_counts": reason_counts,
            "median_units": med,
            "unit_count_range": [int(units.min()), int(units.max())],
            "selected": [
                {"session": name, "n_units": int(n_u)} for n_u, name in pick
            ],
            "note": note,
        }
        animals_out.append(animal_rec)
        for n_u, name in pick:
            selected.append({
                "animal": animal,
                "session": name,
                "n_units": int(n_u),
                "animal_median_units": med,
            })

    return {
        "rule": M3_RULE,
        "n_per_animal": int(n_per_animal),
        "n_2rooms": len(sessions),
        "n_animals": len(animals_out),
        "n_selected_sessions": len(selected),
        "animals": animals_out,
        "selected_sessions": selected,
    }
