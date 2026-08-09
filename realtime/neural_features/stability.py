"""Latent-space stability metrics for feature-set × manifold comparisons.

Metrics are applied where appropriate (PCA vs Isomap). All comparisons that
require a fitted transform use train-only fits on the clean condition, then
apply the frozen transform to degraded features (no refit on degraded data)
unless ``refit_on_degraded=True`` is explicitly requested for an alternate
analysis mode.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import trustworthiness as sk_trustworthiness


def explained_variance_summary(pca_explained_ratio: np.ndarray | list[float] | None) -> dict[str, Any]:
    if pca_explained_ratio is None:
        return {
            "latent_dimensionality_proxy": None,
            "explained_variance_sum": None,
        }
    r = np.asarray(pca_explained_ratio, dtype=float)
    # Participation-ratio proxy for effective dimensionality.
    if r.size == 0 or float(np.sum(r)) <= 0:
        pr = None
    else:
        pr = float((np.sum(r) ** 2) / max(np.sum(r ** 2), 1e-12))
    return {
        "latent_dimensionality_proxy": pr,
        "explained_variance_sum": float(np.sum(r)),
        "explained_variance_ratio": r.tolist(),
    }


def pairwise_distance_preservation(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    max_samples: int = 400,
    seed: int = 0,
) -> float:
    """Spearman-like Pearson correlation of pairwise distances (sampled)."""
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = X.shape[0]
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(seed)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X = X[idx]
        Z = Z[idx]
    from scipy.spatial.distance import pdist

    dx = pdist(X)
    dz = pdist(Z)
    if np.std(dx) < 1e-12 or np.std(dz) < 1e-12:
        return 0.0
    return float(np.corrcoef(dx, dz)[0, 1])


def neighborhood_preservation(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> float:
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = X.shape[0]
    if n < n_neighbors + 2:
        return float("nan")
    k = min(n_neighbors, n - 1)
    try:
        return float(sk_trustworthiness(X, Z, n_neighbors=k))
    except Exception:
        return float("nan")


def latent_trajectory_smoothness(Z: np.ndarray) -> dict[str, float]:
    """Temporal smoothness / velocity of the latent trajectory."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] < 3:
        return {
            "latent_velocity_mean": float("nan"),
            "latent_velocity_std": float("nan"),
            "latent_smoothness": float("nan"),
        }
    vel = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    acc = np.diff(vel)
    # Smoothness: inverse of mean |acceleration| (higher = smoother).
    mean_acc = float(np.mean(np.abs(acc))) if acc.size else 0.0
    smooth = 1.0 / (1.0 + mean_acc)
    return {
        "latent_velocity_mean": float(np.mean(vel)),
        "latent_velocity_std": float(np.std(vel)),
        "latent_smoothness": float(smooth),
    }


def procrustes_alignment_error(
    Z_ref: np.ndarray,
    Z_other: np.ndarray,
) -> float:
    """
    Mean squared error after optimal orthogonal Procrustes alignment.

    Both clouds are centered. Lower = more similar geometry.
    """
    A = np.asarray(Z_ref, dtype=float)
    B = np.asarray(Z_other, dtype=float)
    if A.shape != B.shape or A.shape[0] < 2:
        return float("nan")
    A = A - A.mean(axis=0, keepdims=True)
    B = B - B.mean(axis=0, keepdims=True)
    R, _ = orthogonal_procrustes(B, A)
    B_aligned = B @ R
    return float(np.mean((A - B_aligned) ** 2))


def compute_latent_stability(
    X_feat: np.ndarray,
    Z: np.ndarray,
    *,
    Z_ref: np.ndarray | None = None,
    pca_explained_ratio: np.ndarray | list[float] | None = None,
    n_neighbors: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Aggregate latent stability metrics for one (feature, manifold) state."""
    out: dict[str, Any] = {
        "feature_dimension": int(X_feat.shape[1]) if X_feat.ndim == 2 else None,
        "latent_dimension": int(Z.shape[1]) if Z.ndim == 2 else None,
    }
    out.update(explained_variance_summary(pca_explained_ratio))
    out["pairwise_distance_preservation"] = pairwise_distance_preservation(
        X_feat, Z, seed=seed,
    )
    out["neighborhood_trustworthiness"] = neighborhood_preservation(
        X_feat, Z, n_neighbors=n_neighbors,
    )
    out.update(latent_trajectory_smoothness(Z))
    if Z_ref is not None:
        out["procrustes_alignment_error"] = procrustes_alignment_error(Z_ref, Z)
    else:
        out["procrustes_alignment_error"] = None
    return out
