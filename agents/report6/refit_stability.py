"""Bootstrap block refits + Procrustes overlap of in-target times."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import orthogonal_procrustes

from agents.report6 import PILOT_SEED
from agents.report6.detector import evaluate_detector, fit_detector
from agents.report6.io_cohort import SessionData
from agents.report6.paths import DT_S
from agents.report6.pipeline import run_m1_m2_on_Z
from agents.report6.refit import fit_transform_pca_lds
from agents.report6.target_rule import find_targets
from agents.report6.visits import binary_runs_from_labels


def _block_bootstrap_mask(train_mask: np.ndarray, *, n_blocks: int = 10, rng: np.random.Generator) -> np.ndarray:
    idx = np.where(train_mask)[0]
    if idx.size < n_blocks * 2:
        return train_mask.copy()
    blocks = np.array_split(idx, n_blocks)
    chosen = rng.choice(len(blocks), size=len(blocks), replace=True)
    sel = np.concatenate([blocks[i] for i in chosen])
    out = np.zeros_like(train_mask, dtype=bool)
    out[sel] = True
    return out


def _align(Z_ref: np.ndarray, Z_new: np.ndarray, mask: np.ndarray) -> np.ndarray:
    A = Z_ref[mask]
    B = Z_new[mask]
    mu_a, mu_b = A.mean(0), B.mean(0)
    R, _ = orthogonal_procrustes(B - mu_b, A - mu_a)
    return (Z_new - mu_b) @ R + mu_a


def bootstrap_stability(
    s: SessionData,
    method: str = "pca",
    *,
    B: int = 10,
    seed: int = PILOT_SEED + 2,
) -> dict[str, Any]:
    if s.X_counts is None:
        return {"alias": s.alias, "error": "no_X_counts"}
    d = s.primary_d_pca if method == "pca" else s.primary_d_lds
    Z0 = fit_transform_pca_lds(method, s.X_counts, s.train_ok, d=d, seed=seed)
    d_use = min(3, Z0.shape[1])
    Z0 = Z0[:, :d_use]
    base = find_targets(Z0, s.train_ok, times_s=s.decode_times, dt_s=DT_S, seed=seed)
    if not base.get("targets"):
        return {"alias": s.alias, "method": method, "status": "no_qualifying_target", "pass": False}
    lab = int(base["targets"][0]["label"])
    labels0 = base["labels"]
    in0 = binary_runs_from_labels(labels0, lab)
    model0 = fit_detector(Z0, in0, s.train_ok)
    ev = s.test_mask & s.retained & s.valid
    bin0 = model0.hysteresis(model0.predict_proba(Z0))[ev]

    rng = np.random.default_rng(seed)
    jacs = []
    for b in range(B):
        fit_m = _block_bootstrap_mask(s.train_ok, n_blocks=10, rng=rng)
        Zb = fit_transform_pca_lds(method, s.X_counts, fit_m, d=d, seed=seed + b)
        Zb = _align(Z0, Zb[:, :d_use], s.train_ok)
        # detect original target region in aligned bootstrap latents
        bin_b = model0.hysteresis(model0.predict_proba(Zb))[ev]
        inter = np.logical_and(bin0, bin_b).sum()
        union = np.logical_or(bin0, bin_b).sum()
        jacs.append(float(inter / union) if union else 1.0)

    med = float(np.median(jacs)) if jacs else float("nan")
    return {
        "alias": s.alias,
        "method": method,
        "status": "ok",
        "B": B,
        "jaccard": jacs,
        "median_jaccard": med,
        "pass": bool(med >= 0.3),
    }
