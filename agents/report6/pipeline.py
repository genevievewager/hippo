"""M0–M2 core pipeline on cached latents; null/split/refit helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agents.report6 import PILOT_SEED
from agents.report6.detector import evaluate_detector, fit_detector
from agents.report6.io_cohort import SessionData, iter_sessions, public_cohort_table
from agents.report6.paths import DT_S, OUT_ROOT
from agents.report6.refit import circular_shift_rows, circular_shift_units, fit_transform_pca_lds
from agents.report6.target_rule import find_targets
from agents.report6.visits import binary_runs_from_labels


def _eval_mask(s: SessionData) -> np.ndarray:
    return s.test_mask & s.retained & s.valid


def _Z_for(s: SessionData, method: str) -> tuple[np.ndarray, int]:
    if method == "lds":
        return s.Z_lds, s.primary_d_lds
    if method == "pca":
        return s.Z_pca, s.primary_d_pca
    raise ValueError(method)


def run_m1_m2_on_Z(
    Z: np.ndarray,
    s: SessionData,
    *,
    method: str,
    seed: int = PILOT_SEED,
) -> dict[str, Any]:
    """Target rule + detector on a latent matrix aligned to session masks."""
    d_use = min(3, Z.shape[1])
    Z_use = np.asarray(Z[:, :d_use], dtype=float)
    tr = s.train_ok
    times = s.decode_times
    tgt = find_targets(Z_use, tr, times_s=times, dt_s=DT_S, seed=seed)
    out: dict[str, Any] = {
        "alias": s.alias,
        "animal": s.animal_alias,
        "arm": s.arm,
        "method": method,
        "d_full": int(Z.shape[1]),
        "d_target": d_use,
        "n_units": s.n_units,
        "m1": {k: v for k, v in tgt.items() if k != "labels"},
        "targets_eval": [],
    }
    if not tgt.get("ok") or tgt.get("labels") is None:
        out["status"] = "m1_failed"
        return out
    labels = tgt["labels"]
    if not tgt["targets"]:
        out["status"] = "no_qualifying_target"
        # still attach candidate summary
        return out

    for tinfo in tgt["targets"]:
        lab = int(tinfo["label"])
        in_train = binary_runs_from_labels(labels, lab)
        model = fit_detector(Z_use, in_train, tr)
        ev = evaluate_detector(
            model, Z_use, labels, lab, _eval_mask(s), times_s=times, dt_s=DT_S,
        )
        # drop heavy arrays from saved JSON
        slim = {k: v for k, v in ev.items() if k not in ("p", "binary")}
        slim["target"] = tinfo
        slim["model"] = {
            "mu": model.mu.tolist(),
            "logit_a": model.logit_a,
            "logit_b": model.logit_b,
            "enter_p": model.enter_p,
            "exit_p": model.exit_p,
        }
        out["targets_eval"].append(slim)
    out["status"] = "ok"
    # keep best target by occupancy for null summaries
    out["top_distinguishable_pass"] = bool(
        out["targets_eval"][0]["distinguishable_pass"]
    ) if out["targets_eval"] else False
    return out


def run_m0_m2_session(s: SessionData, methods: tuple[str, ...] = ("lds", "pca")) -> dict[str, Any]:
    results = {"alias": s.alias, "arm": s.arm, "methods": {}}
    for m in methods:
        Z, _ = _Z_for(s, m)
        results["methods"][m] = run_m1_m2_on_Z(Z, s, method=m)
    return results


def null_a_time_shifts(
    s: SessionData,
    method: str = "lds",
    n_shifts: int = 20,
) -> dict[str, Any]:
    Z, _ = _Z_for(s, method)
    T = Z.shape[0]
    passes = []
    for i in range(1, n_shifts + 1):
        shift = int(T * i / (n_shifts + 1))
        Zs = circular_shift_rows(Z, shift)
        r = run_m1_m2_on_Z(Zs, s, method=method, seed=PILOT_SEED + i)
        ok = (
            r.get("status") == "ok"
            and bool(r.get("top_distinguishable_pass"))
        )
        passes.append({"shift_i": i, "shift": shift, "pass": ok, "status": r.get("status")})
    n_pass = sum(1 for p in passes if p["pass"])
    return {
        "alias": s.alias,
        "method": method,
        "n_shifts": n_shifts,
        "n_pass": n_pass,
        "null_a_pass": n_pass <= 4,  # ≤ 4/20
        "shifts": passes,
    }


def null_b_surrogate(
    s: SessionData,
    method: str,
    *,
    n_surr: int = 5,
    seed: int = PILOT_SEED,
) -> dict[str, Any]:
    if s.X_counts is None:
        return {"alias": s.alias, "method": method, "error": "no_X_counts"}
    X = s.X_counts
    d = s.primary_d_lds if method == "lds" else s.primary_d_pca
    rng = np.random.default_rng(seed)
    T, n_u = X.shape
    passes = []
    for i in range(n_surr):
        shifts = rng.integers(1, T, size=n_u)
        Xs = circular_shift_units(X, shifts)
        Z = fit_transform_pca_lds(method, Xs, s.train_ok, d=d, seed=seed + i)
        r = run_m1_m2_on_Z(Z, s, method=method, seed=seed + i)
        ok = r.get("status") == "ok" and bool(r.get("top_distinguishable_pass"))
        passes.append({"i": i, "pass": ok, "status": r.get("status")})
    n_pass = sum(1 for p in passes if p["pass"])
    return {
        "alias": s.alias,
        "method": method,
        "n_surr": n_surr,
        "n_pass": n_pass,
        "null_b_pass": n_pass <= 1,
        "surrogates": passes,
    }


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=2, default=_default) + "\n")


def run_all_m0_m2() -> Path:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    save_json(public_cohort_table(), OUT_ROOT / "cohort_public.json")
    all_res = []
    for s in iter_sessions():
        print(f"[M0-M2] {s.alias} {s.arm} n_units={s.n_units} has_X={s.X_counts is not None}", flush=True)
        all_res.append(run_m0_m2_session(s))
        save_json(all_res[-1], OUT_ROOT / "m0_m2" / f"{s.alias}.json")
    save_json(all_res, OUT_ROOT / "m0_m2" / "all.json")
    return OUT_ROOT / "m0_m2" / "all.json"
