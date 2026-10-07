"""Load M3 figure-contract artifacts and fixed example-selection rules."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
M3_ROOT = REPO_ROOT / "outputs" / "real_quadrant" / "m3"
ROOM_A = M3_ROOT / "room_A"

EXAMPLE_SESSION_RULE = (
    "Among M3 cohort sessions, pick the session whose n_units is closest to "
    "the cohort median n_units; ties → lexicographically first session name."
)
EXAMPLE_UNIT_RULE = (
    "On the example session, among units with ≥50 spikes on train_ok times, "
    "select the 6 with highest peak occupancy-normalized train rate "
    "(5 cm bins); ties → unit_id ascending."
)
COHORT_RULE_KEY = "rule"


def load_cohort() -> dict[str, Any]:
    return json.loads((M3_ROOT / "cohort_manifest.json").read_text())


def load_report2_contrasts() -> dict[str, Any]:
    path = M3_ROOT / "report2_contrasts.json"
    if path.is_file():
        return json.loads(path.read_text())
    raise FileNotFoundError(path)


def session_dir(session: str) -> Path:
    return ROOM_A / session


def load_session_report(session: str) -> dict[str, Any]:
    return json.loads((session_dir(session) / "session_report.json").read_text())


def load_contract(session: str) -> dict[str, Any]:
    fc = session_dir(session) / "figure_contract"
    arrays = dict(np.load(fc / "arrays.npz", allow_pickle=True))
    latents = dict(np.load(fc / "latents.npz", allow_pickle=True))
    preds = dict(np.load(fc / "predictions.npz", allow_pickle=True))
    a13 = json.loads((fc / "a13.json").read_text()) if (fc / "a13.json").is_file() else {}
    prov = (
        json.loads((fc / "provenance.json").read_text())
        if (fc / "provenance.json").is_file() else {}
    )
    return {
        "arrays": arrays,
        "latents": latents,
        "preds": preds,
        "a13": a13,
        "provenance": prov,
        "figure_contract_dir": fc,
    }


def pick_example_session(cohort: dict[str, Any] | None = None) -> dict[str, Any]:
    cohort = cohort or load_cohort()
    sels = list(cohort["selected_sessions"])
    units = np.asarray([int(s["n_units"]) for s in sels], float)
    med = float(np.median(units))
    ranked = sorted(sels, key=lambda s: (abs(int(s["n_units"]) - med), s["session"]))
    pick = dict(ranked[0])
    pick["cohort_median_units"] = med
    pick["rule"] = EXAMPLE_SESSION_RULE
    return pick


def pick_example_units(
    session: str,
    *,
    n_units: int = 6,
    min_spikes: int = 50,
) -> dict[str, Any]:
    """Train-only peak-rate ranking; loads spikes from HIPPO_DATA_ROOT (no fit)."""
    from analysis.real_quadrant.adapter import load_units_and_spikes

    contract = load_contract(session)
    arrays = contract["arrays"]
    times = np.asarray(arrays["decode_times"], float)
    y = np.asarray(arrays["y"], float)
    train_ok = np.asarray(arrays["train_ok"], bool)
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    unit_ids, spike_times, units_df = load_units_and_spikes(root / session)

    # Align unit order to latent width if needed via source_summary n_units.
    bin_cm = 5.0
    x_edges = np.arange(np.nanmin(y[:, 0]) - 1e-6, np.nanmax(y[:, 0]) + bin_cm, bin_cm)
    y_edges = np.arange(np.nanmin(y[:, 1]) - 1e-6, np.nanmax(y[:, 1]) + bin_cm, bin_cm)
    t_train = times[train_ok]
    y_train = y[train_ok]
    occ, _, _ = np.histogram2d(y_train[:, 0], y_train[:, 1], bins=[x_edges, y_edges])
    occ = occ.astype(float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05

    scored: list[tuple[float, int, int]] = []  # (-peak, unit_id, index)
    for j, (uid, st) in enumerate(zip(unit_ids, spike_times)):
        st = np.asarray(st, float)
        # spikes falling in train windows (approx: spike time in train decode coverage)
        if len(t_train) == 0:
            continue
        t0, t1 = float(t_train[0]), float(t_train[-1])
        st_tr = st[(st >= t0) & (st <= t1)]
        if st_tr.size < int(min_spikes):
            continue
        # position at nearest train decode time
        idx = np.searchsorted(t_train, st_tr, side="right") - 1
        idx = np.clip(idx, 0, len(t_train) - 1)
        counts, _, _ = np.histogram2d(
            y_train[idx, 0], y_train[idx, 1], bins=[x_edges, y_edges],
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(occ > 0, counts / (occ * dt), 0.0)
        peak = float(np.nanmax(rate)) if rate.size else 0.0
        scored.append((-peak, int(uid), int(j)))
    scored.sort()
    chosen = scored[: int(n_units)]
    return {
        "rule": EXAMPLE_UNIT_RULE,
        "session": session,
        "unit_ids": [uid for _, uid, _ in chosen],
        "unit_indices": [j for _, _, j in chosen],
        "peak_rates_hz": [-neg for neg, _, _ in chosen],
        "n_candidates": len(scored),
        "unit_ids_all": [int(u) for u in unit_ids],
        "spike_times": spike_times,
        "units_df": units_df,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "occ_train": occ,
        "dt": dt,
    }


def animal_means_from_reports(
    *,
    grid: str = "final",
) -> dict[str, dict[str, float]]:
    """Per-animal mean normalized error. grid='final' or 'grid20' for pca/lds family."""
    cohort = load_cohort()
    by_animal: dict[str, dict[str, list[float]]] = {}
    for sel in cohort["selected_sessions"]:
        report = load_session_report(sel["session"])
        by_m = {m["method"]: m for m in report["methods"]}
        animal = sel["animal"]
        by_animal.setdefault(animal, {})
        name_map = {
            "raw": "raw",
            "raw_smooth": "raw_smooth",
            "raw_lag": "raw_lag",
            "dm": "dm",
            "dm_smooth": "dm_smooth",
            "isomap": "isomap",
            "gpfa": "gpfa",
            "gpfa_causal": "gpfa_causal",
            "pca": "pca" if grid == "final" else "pca_grid20",
            "lds": "lds" if grid == "final" else "lds_grid20",
            "pca_smooth": "pca_smooth" if grid == "final" else "pca_smooth_grid20",
            "lds_smooth": "lds_smooth" if grid == "final" else "lds_smooth_grid20",
        }
        for logical, key in name_map.items():
            rec = by_m.get(key)
            if not rec or rec.get("status") not in (None, "ok"):
                continue
            ne = rec.get("normalized_error")
            if ne is None or not np.isfinite(ne):
                continue
            by_animal[animal].setdefault(logical, []).append(float(ne))
    return {
        a: {m: float(np.mean(v)) for m, v in ms.items()}
        for a, ms in by_animal.items()
    }


def extent_exclusion_count(cohort: dict[str, Any] | None = None) -> int:
    cohort = cohort or load_cohort()
    return int(sum(
        int((a.get("exclude_reason_counts") or {}).get("ExtentError", 0))
        for a in cohort.get("animals", [])
    ))
