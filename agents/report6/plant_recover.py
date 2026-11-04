"""Plant-and-recover on one matched-sim session."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.report6 import PILOT_SEED
from agents.report6.detector import evaluate_detector, fit_detector
from agents.report6.io_cohort import SessionData
from agents.report6.paths import DT_S
from agents.report6.refit import fit_transform_pca_lds
from agents.report6.target_rule import find_targets
from agents.report6.visits import binary_runs_from_labels, count_visits


def plant_pattern(
    X: np.ndarray,
    train_mask: np.ndarray,
    times_s: np.ndarray,
    *,
    n_events: int = 40,
    dwell_s: float = 0.500,
    min_gap_s: float = 5.0,
    seed: int = PILOT_SEED + 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Add a recurring population pattern; return (X_planted, planted_mask)."""
    X = np.asarray(X, dtype=float).copy()
    T, n_u = X.shape
    dt = float(np.median(np.diff(times_s))) if len(times_s) > 1 else DT_S
    dwell_n = max(1, int(round(dwell_s / dt)))
    gap_n = max(1, int(round(min_gap_s / dt)))
    rng = np.random.default_rng(seed)
    # unit pattern: random signs, L2-normalised
    p = rng.normal(size=n_u)
    p = p / (np.linalg.norm(p) + 1e-12)
    # scale: 2 × median train count L2
    norms = np.linalg.norm(X[train_mask], axis=1)
    scale = 2.0 * float(np.median(norms))
    amp = scale * p

    # candidate starts with gaps
    starts = []
    t = gap_n
    while t + dwell_n < T and len(starts) < n_events:
        starts.append(t)
        t += dwell_n + gap_n
    # if too few, pack denser
    if len(starts) < n_events:
        starts = list(range(gap_n, T - dwell_n, max(gap_n, dwell_n + 1)))[:n_events]

    planted = np.zeros(T, dtype=bool)
    for st in starts:
        planted[st : st + dwell_n] = True
        X[st : st + dwell_n] = X[st : st + dwell_n] + amp
    return X, planted


def plant_and_recover(
    s: SessionData,
    method: str = "pca",
    *,
    seed: int = PILOT_SEED + 3,
) -> dict[str, Any]:
    if s.X_counts is None:
        return {"alias": s.alias, "error": "no_X_counts"}
    Xp, planted = plant_pattern(
        s.X_counts, s.train_ok, s.decode_times, seed=seed,
    )
    d = s.primary_d_pca if method == "pca" else s.primary_d_lds
    Z = fit_transform_pca_lds(method, Xp, s.train_ok, d=d, seed=seed)
    d_use = min(3, Z.shape[1])
    Z = Z[:, :d_use]
    tgt = find_targets(Z, s.train_ok, times_s=s.decode_times, dt_s=DT_S, seed=seed)
    if not tgt.get("targets"):
        return {
            "alias": s.alias,
            "method": method,
            "status": "no_qualifying_target",
            "hit_rate": 0.0,
            "fa_per_min": None,
            "pass": False,
        }
    # choose target with max overlap to planted on train
    labels = tgt["labels"]
    best_lab, best_ov = None, -1.0
    for tinfo in tgt["targets"]:
        lab = int(tinfo["label"])
        memb = labels == lab
        ov = float((memb & planted & s.train_ok).sum()) / max(1, int((planted & s.train_ok).sum()))
        if ov > best_ov:
            best_ov, best_lab = ov, lab
    assert best_lab is not None
    in_train = binary_runs_from_labels(labels, best_lab)
    model = fit_detector(Z, in_train, s.train_ok)
    ev = s.test_mask & s.retained & s.valid
    det = evaluate_detector(model, Z, labels, best_lab, ev, times_s=s.decode_times)
    binary = det["binary"]

    # hit rate on planted test windows
    planted_test = planted & ev
    # group planted runs in test
    padded = np.concatenate([[False], planted_test, [False]])
    dlt = np.diff(padded.astype(np.int8))
    ents = np.where(dlt == 1)[0]
    exs = np.where(dlt == -1)[0]
    hits = 0
    for e, x in zip(ents, exs):
        if binary[e:x].any():
            hits += 1
    n_planted_test = len(ents)
    hit_rate = float(hits / n_planted_test) if n_planted_test else 0.0

    # false alarms: detected visits on eval with no planted within 1 s
    times = s.decode_times
    vis = count_visits(binary[ev], times_s=times[ev], dt_s=DT_S)
    # rebuild visit intervals on full timeline
    padded = np.concatenate([[False], binary & ev, [False]])
    dlt = np.diff(padded.astype(np.int8))
    dents = np.where(dlt == 1)[0]
    dexs = np.where(dlt == -1)[0]
    fa = 0
    for e, x in zip(dents, dexs):
        # any planted within 1s of visit mid
        mid_t = 0.5 * (times[e] + times[min(x - 1, len(times) - 1)])
        near = planted & (np.abs(times - mid_t) <= 1.0)
        if not near.any():
            fa += 1
    dur_min = float((times[ev][-1] - times[ev][0]) / 60.0) if ev.sum() > 1 else float("nan")
    fa_rate = float(fa / dur_min) if dur_min and dur_min > 0 else float("nan")
    passed = hit_rate >= 0.5 and np.isfinite(fa_rate) and fa_rate <= 2.0
    return {
        "alias": s.alias,
        "method": method,
        "status": "ok",
        "n_planted_test": n_planted_test,
        "hits": hits,
        "hit_rate": hit_rate,
        "n_fa_visits": fa,
        "fa_per_min": fa_rate if np.isfinite(fa_rate) else None,
        "train_overlap_best": best_ov,
        "best_label": best_lab,
        "pass": bool(passed),
        "eval_visits": vis,
    }
