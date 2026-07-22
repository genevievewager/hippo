"""Geometry evaluation metrics for Isomap embeddings."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness


def trustworthiness_at_ks(
    X: np.ndarray,
    Z: np.ndarray,
    neighbor_ks: tuple[int, ...] | list[int] = (5, 10, 20),
) -> dict[str, float]:
    """Compute trustworthiness at several neighborhood sizes."""
    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = X.shape[0]
    out: dict[str, float] = {}
    for k in neighbor_ks:
        kk = int(min(max(1, k), max(1, n - 1)))
        score = float(sklearn_trustworthiness(X, Z, n_neighbors=kk))
        out[f"trustworthiness_k{kk}"] = score
    if out:
        out["trustworthiness"] = float(np.mean(list(out.values())))
    else:
        out["trustworthiness"] = float("nan")
    return out


def knn_overlap(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> float:
    """Fraction of shared k-nearest neighbors between input and embedding spaces."""
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = X.shape[0]
    k = int(min(max(1, n_neighbors), max(1, n - 1)))
    nn_x = NearestNeighbors(n_neighbors=k + 1).fit(X)
    nn_z = NearestNeighbors(n_neighbors=k + 1).fit(Z)
    idx_x = nn_x.kneighbors(return_distance=False)[:, 1:]
    idx_z = nn_z.kneighbors(return_distance=False)[:, 1:]
    overlaps = [
        len(set(idx_x[i].tolist()) & set(idx_z[i].tolist())) / float(k)
        for i in range(n)
    ]
    return float(np.mean(overlaps))


def _sample_pair_indices(
    n: int,
    n_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample unique unordered pairs (i < j)."""
    max_pairs = n * (n - 1) // 2
    n_pairs = int(min(n_pairs, max_pairs))
    if n_pairs <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    # Rejection sampling of upper-triangle pairs
    i = rng.integers(0, n, size=n_pairs * 2)
    j = rng.integers(0, n, size=n_pairs * 2)
    mask = i < j
    i, j = i[mask], j[mask]
    # Deduplicate
    pairs = np.unique(np.stack([i, j], axis=1), axis=0)
    if len(pairs) < n_pairs:
        # Fill remaining via deterministic upper triangle if needed
        need = n_pairs - len(pairs)
        extra_i, extra_j = [], []
        for a in range(n):
            for b in range(a + 1, n):
                extra_i.append(a)
                extra_j.append(b)
                if len(extra_i) >= need + len(pairs):
                    break
            if len(extra_i) >= need + len(pairs):
                break
        existing = set(map(tuple, pairs.tolist()))
        for a, b in zip(extra_i, extra_j):
            if (a, b) not in existing:
                pairs = np.vstack([pairs, [a, b]])
                if len(pairs) >= n_pairs:
                    break
    pairs = pairs[:n_pairs]
    return pairs[:, 0], pairs[:, 1]


def geodesic_distance_preservation(
    geo_dist: np.ndarray,
    Z: np.ndarray,
    *,
    n_pairs: int = 100_000,
    random_state: int = 42,
) -> dict[str, float]:
    """Correlate graph geodesic distances with embedding Euclidean distances.

    Returns residual variance ``1 - R^2(D_G, D_Z)`` and Pearson correlation.
    """
    geo_dist = np.asarray(geo_dist, dtype=float)
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    if geo_dist.shape[0] != n or geo_dist.shape[1] != n:
        raise ValueError("geo_dist must be square with side equal to n_samples")

    rng = np.random.default_rng(random_state)
    i, j = _sample_pair_indices(n, n_pairs, rng)
    if len(i) == 0:
        return {
            "geodesic_distance_correlation": float("nan"),
            "residual_variance": float("nan"),
            "n_distance_pairs": 0,
        }

    d_g = geo_dist[i, j]
    finite = np.isfinite(d_g)
    i, j, d_g = i[finite], j[finite], d_g[finite]
    if len(d_g) < 3:
        return {
            "geodesic_distance_correlation": float("nan"),
            "residual_variance": float("nan"),
            "n_distance_pairs": int(len(d_g)),
        }

    d_z = np.linalg.norm(Z[i] - Z[j], axis=1)
    if np.std(d_g) < 1e-12 or np.std(d_z) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(d_g, d_z)[0, 1])
    residual = float(1.0 - corr ** 2)
    return {
        "geodesic_distance_correlation": corr,
        "residual_variance": residual,
        "n_distance_pairs": int(len(d_g)),
    }


def continuity_proxy(
    X: np.ndarray,
    Z: np.ndarray,
    *,
    n_neighbors: int = 10,
) -> float:
    """Continuity-like score: fraction of embedding neighbors that are input neighbors.

    This is the dual of trustworthiness's local check and equals ``knn_overlap``
    when neighborhoods are compared symmetrically at the same ``k``.
    """
    return knn_overlap(X, Z, n_neighbors=n_neighbors)


def evaluate_isomap_geometry(
    X_preprocessed: np.ndarray,
    Z: np.ndarray,
    *,
    geo_dist: np.ndarray | None = None,
    trustworthiness_neighbors: tuple[int, ...] = (5, 10, 20),
    compute_residual_variance: bool = True,
    compute_geodesic_preservation: bool = True,
    sampled_distance_pairs: int = 100_000,
    random_state: int = 42,
) -> dict[str, Any]:
    """Bundle neighborhood and geodesic preservation metrics."""
    metrics: dict[str, Any] = {}
    metrics.update(
        trustworthiness_at_ks(X_preprocessed, Z, neighbor_ks=trustworthiness_neighbors)
    )
    metrics["knn_overlap_k10"] = knn_overlap(X_preprocessed, Z, n_neighbors=10)
    metrics["continuity_k10"] = continuity_proxy(X_preprocessed, Z, n_neighbors=10)

    if geo_dist is not None and (compute_residual_variance or compute_geodesic_preservation):
        geo_metrics = geodesic_distance_preservation(
            geo_dist,
            Z,
            n_pairs=sampled_distance_pairs,
            random_state=random_state,
        )
        metrics.update(geo_metrics)
    else:
        metrics.setdefault("geodesic_distance_correlation", float("nan"))
        metrics.setdefault("residual_variance", float("nan"))
        metrics.setdefault("n_distance_pairs", 0)
    return metrics


def procrustes_aligned_error(
    Z_ref: np.ndarray,
    Z_other: np.ndarray,
) -> dict[str, float]:
    """Align ``Z_other`` to ``Z_ref`` by orthogonal Procrustes and report MSE / correlation."""
    from scipy.linalg import orthogonal_procrustes

    A = np.asarray(Z_ref, dtype=float)
    B = np.asarray(Z_other, dtype=float)
    if A.shape != B.shape:
        raise ValueError("Procrustes requires matching shapes")
    A_c = A - A.mean(axis=0, keepdims=True)
    B_c = B - B.mean(axis=0, keepdims=True)
    # R minimizes ||A_c - B_c @ R||; returned scale is Trace(S), not a multiplier.
    R, _ = orthogonal_procrustes(B_c, A_c)
    B_aligned = B_c @ R
    # Optional isotropic scale for magnitude-matched comparison
    denom = float(np.sum(B_aligned ** 2))
    if denom > 1e-12:
        scale = float(np.sum(A_c * B_aligned) / denom)
        B_aligned = scale * B_aligned
    err = float(np.mean((A_c - B_aligned) ** 2))
    flat_a = A_c.ravel()
    flat_b = B_aligned.ravel()
    if np.std(flat_a) < 1e-12 or np.std(flat_b) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(flat_a, flat_b)[0, 1])
    return {
        "procrustes_mse": err,
        "procrustes_correlation": corr,
    }
