"""M1: density / cluster target rule (pre-registered in PILOT_PLAN.md)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from agents.report6 import PILOT_SEED
from agents.report6.visits import binary_runs_from_labels, count_visits

K_GRID = (2, 3, 4, 5, 6)
ELBOW_REL_DROP = 0.10
DEFAULT_K = 4
OCC_MIN = 0.05
VISITS_MIN = 10
MIN_OUTSIDE_S = 1.0
MAX_TARGETS = 3
TARGET_DIMS = 3


@dataclass
class Candidate:
    label: int
    occupancy: float
    n_visits: int
    mean_dwell_s: float
    median_dwell_s: float
    qualifies: bool
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_k(Z_train: np.ndarray, *, seed: int = PILOT_SEED) -> tuple[int, dict[str, Any]]:
    """Elbow on inertia; smallest k with relative drop < ELBOW_REL_DROP else DEFAULT_K."""
    X = np.asarray(Z_train, dtype=float)
    if X.shape[0] < 10:
        return 2, {"inertias": {}, "reason": "too_few_samples"}
    inertias: dict[int, float] = {}
    for k in K_GRID:
        if X.shape[0] <= k:
            break
        km = KMeans(n_clusters=k, random_state=seed, n_init=5, algorithm="lloyd")
        km.fit(X)
        inertias[k] = float(km.inertia_)
    if len(inertias) < 2:
        k = min(inertias) if inertias else 2
        return int(k), {"inertias": inertias, "reason": "short_grid"}
    ks = sorted(inertias)
    i2 = inertias[ks[0]]
    chosen = DEFAULT_K if DEFAULT_K in inertias else ks[-1]
    reason = "default"
    for prev, cur in zip(ks, ks[1:]):
        drop = (inertias[prev] - inertias[cur]) / max(i2, 1e-12)
        if drop < ELBOW_REL_DROP:
            chosen = int(cur)
            reason = f"elbow_at_{cur}_drop_{drop:.4f}"
            break
    else:
        # no early elbow: keep DEFAULT_K if available else max k
        reason = "no_elbow"
    if chosen not in inertias:
        chosen = ks[min(len(ks) - 1, 2)]
    return int(chosen), {"inertias": inertias, "reason": reason}


def _target_dims(Z: np.ndarray) -> int:
    return int(min(TARGET_DIMS, Z.shape[1]))


def find_targets(
    Z: np.ndarray,
    train_mask: np.ndarray,
    *,
    times_s: np.ndarray,
    dt_s: float = 0.050,
    seed: int = PILOT_SEED,
) -> dict[str, Any]:
    """Fit k-means on train latents; qualify candidates; return up to MAX_TARGETS."""
    Z = np.asarray(Z, dtype=float)
    d = _target_dims(Z)
    Z3 = Z[:, :d]
    train = np.asarray(train_mask, dtype=bool)
    if train.sum() < 20:
        return {
            "ok": False,
            "reason": "train_too_short",
            "k": None,
            "candidates": [],
            "targets": [],
            "labels": None,
            "scaler_mean": None,
            "scaler_scale": None,
            "centroids": None,
        }

    scaler = StandardScaler()
    Z_train = scaler.fit_transform(Z3[train])
    k, elbow = choose_k(Z_train, seed=seed)
    km = KMeans(n_clusters=int(k), random_state=seed, n_init=5, algorithm="lloyd")
    km.fit(Z_train)
    Z_all = scaler.transform(Z3)
    labels = km.predict(Z_all)

    candidates: list[Candidate] = []
    for lab in range(int(k)):
        memb = binary_runs_from_labels(labels, lab)
        # visits / occupancy on training block only
        stats = count_visits(
            memb[train],
            times_s=times_s[train],
            dt_s=dt_s,
            min_outside_s=MIN_OUTSIDE_S,
        )
        qual = (
            float(stats["occupancy"]) >= OCC_MIN
            and int(stats["n_visits"]) >= VISITS_MIN
        )
        candidates.append(
            Candidate(
                label=int(lab),
                occupancy=float(stats["occupancy"]),
                n_visits=int(stats["n_visits"]),
                mean_dwell_s=float(stats["mean_dwell_s"]),
                median_dwell_s=float(stats["median_dwell_s"]),
                qualifies=bool(qual),
            )
        )

    qualifiers = sorted(
        [c for c in candidates if c.qualifies],
        key=lambda c: c.occupancy,
        reverse=True,
    )[:MAX_TARGETS]
    for i, c in enumerate(qualifiers):
        c.rank = i

    centroids = {}
    for c in candidates:
        pts = Z3[train & (labels == c.label)]
        centroids[str(c.label)] = pts.mean(axis=0).tolist() if len(pts) else None

    return {
        "ok": True,
        "k": int(k),
        "elbow": elbow,
        "n_dims": d,
        "candidates": [c.to_dict() for c in candidates],
        "targets": [c.to_dict() for c in qualifiers],
        "labels": labels.astype(np.int16),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "centroids": centroids,
        "cluster_centers_std": km.cluster_centers_.tolist(),
    }
