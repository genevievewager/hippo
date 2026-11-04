"""Split-population agreement (Cohen's κ) with Procrustes alignment."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import orthogonal_procrustes

from agents.report6 import PILOT_SEED
from agents.report6.detector import cohen_kappa, evaluate_detector, fit_detector
from agents.report6.io_cohort import SessionData
from agents.report6.paths import DT_S
from agents.report6.refit import fit_transform_pca_lds
from agents.report6.target_rule import find_targets
from agents.report6.visits import binary_runs_from_labels


def _procrustes_apply(Z_ref_train: np.ndarray, Z_other: np.ndarray, train_mask: np.ndarray) -> np.ndarray:
    """Align Z_other to Z_ref using train rows; apply to all rows."""
    A = np.asarray(Z_ref_train[train_mask], dtype=float)
    B = np.asarray(Z_other[train_mask], dtype=float)
    # center
    mu_a, mu_b = A.mean(0), B.mean(0)
    A0, B0 = A - mu_a, B - mu_b
    R, _ = orthogonal_procrustes(B0, A0)
    return (Z_other - mu_b) @ R + mu_a


def split_population_kappa(
    s: SessionData,
    method: str = "pca",
    *,
    seed: int = PILOT_SEED + 1,
    n_chance: int = 20,
) -> dict[str, Any]:
    if s.X_counts is None:
        return {"alias": s.alias, "error": "no_X_counts"}
    X = s.X_counts
    n_u = X.shape[1]
    if n_u < 8:
        return {"alias": s.alias, "error": "too_few_units"}
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_u)
    half = n_u // 2
    idx_a, idx_b = perm[:half], perm[half : 2 * half]
    d = s.primary_d_pca if method == "pca" else s.primary_d_lds
    # cap d by half units
    d = int(min(d, max(2, half // 2)))

    Za = fit_transform_pca_lds(method, X[:, idx_a], s.train_ok, d=d, seed=seed)
    Zb = fit_transform_pca_lds(method, X[:, idx_b], s.train_ok, d=d, seed=seed + 1)
    d_use = min(3, Za.shape[1], Zb.shape[1])
    Za, Zb = Za[:, :d_use], Zb[:, :d_use]
    Zb_al = _procrustes_apply(Za, Zb, s.train_ok)

    tgt = find_targets(Za, s.train_ok, times_s=s.decode_times, dt_s=DT_S, seed=seed)
    if not tgt.get("targets"):
        return {
            "alias": s.alias,
            "method": method,
            "status": "no_qualifying_target",
            "kappa": None,
            "pass": False,
        }
    lab = int(tgt["targets"][0]["label"])
    labels = tgt["labels"]
    in_train = binary_runs_from_labels(labels, lab)
    model = fit_detector(Za, in_train, s.train_ok)
    ev = s.test_mask & s.retained & s.valid
    ev_a = evaluate_detector(model, Za, labels, lab, ev, times_s=s.decode_times)
    # For half B: reuse same target model in aligned coords; neighbour labels from A
    ev_b = evaluate_detector(model, Zb_al, labels, lab, ev, times_s=s.decode_times)
    bin_a = ev_a["binary"][ev]
    bin_b = ev_b["binary"][ev]
    kappa = cohen_kappa(bin_a, bin_b)
    chance = []
    for i in range(n_chance):
        shuf = rng.permutation(bin_b)
        chance.append(cohen_kappa(bin_a, shuf))
    thr = float(np.quantile(chance, 0.95)) if chance else 0.0
    passed = bool(np.isfinite(kappa) and kappa >= 0.2 and kappa > thr)
    return {
        "alias": s.alias,
        "method": method,
        "status": "ok",
        "d": d_use,
        "n_units_half": int(half),
        "kappa": float(kappa) if np.isfinite(kappa) else None,
        "chance_p95": thr,
        "pass": passed,
        "distinguishable_a": ev_a["distinguishable_pass"],
    }
