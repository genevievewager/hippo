"""Diffusion Maps + Nyström out-of-sample embedding (realtime-deployable).

Offline ``fit`` constructs a landmark diffusion operator and eigendecomposition.
Online ``transform`` / ``transform_one`` only evaluate a query-to-landmark kernel
and a precomputed Nyström projection — no eigendecomposition, graph rebuild,
or retraining.

Query-side local bandwidth adaptation does **not** update the fitted manifold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

from realtime.manifolds.base import ManifoldEncoder

DIFFUSION_NYSTROM_NAME = "diffusion_nystrom"
DEFAULT_N_LANDMARKS = 512
DEFAULT_LANDMARK_METHOD = "minibatch_kmeans"
DEFAULT_N_COMPONENTS = 10
DEFAULT_LOCAL_SCALE_K = 10
DEFAULT_ALPHA = 1.0
DEFAULT_DIFFUSION_TIME = 1
DEFAULT_EPS = 1e-8
DEFAULT_EIGENVALUE_MIN = 1e-6
LANDMARK_COUNT_BENCHMARK = (128, 256, 512, 1024, 2048)
LANDMARK_METHODS = ("random", "kmeans", "minibatch_kmeans")

# Hard realtime operation deadline (distinct from decode-window length).
DEFAULT_OPERATION_DEADLINE_MS = 25.0


def _as_contig(arr: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(arr, dtype=dtype))


def pairwise_sq_euclidean(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Vectorized squared Euclidean distances ``|a-b|^2 = |a|^2+|b|^2-2 a·b``."""
    A = np.asarray(A)
    if B is None:
        B = A
    else:
        B = np.asarray(B)
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    d2 = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _entropy(p: np.ndarray, eps: float) -> float:
    q = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = q / q.sum()
    return float(-np.sum(q * np.log(q)))


class DiffusionNystrom(ManifoldEncoder):
    """Landmark diffusion maps with Nyström extension for streaming queries.

    Parameters
    ----------
    n_landmarks :
        Size of the landmark set ``M`` (diffusion operator is ``M×M``).
    landmark_method :
        ``random``, ``kmeans``, or ``minibatch_kmeans`` (default).
    n_components :
        Number of *nontrivial* diffusion coordinates (trivial λ≈1 dropped).
    local_scale_k :
        Neighbor index for self-tuning Gaussian bandwidths.
    alpha :
        Density-normalization exponent (1.0 recovers the Laplace–Beltrami
        operator in the large-sample limit).
    diffusion_time :
        Diffusion time ``τ`` in ``z_k = λ_k^τ ψ_k``.
    eps :
        Floor for bandwidths, degrees, and kernel row sums.
    eigenvalue_min :
        Drop eigenmodes with ``|λ|`` below this threshold.
    dtype :
        Online array dtype (``float32`` recommended).
    """

    name = DIFFUSION_NYSTROM_NAME
    realtime_compatible = True
    deployment_tag = "realtime_nystrom"

    def __init__(
        self,
        *,
        n_landmarks: int = DEFAULT_N_LANDMARKS,
        landmark_method: str = DEFAULT_LANDMARK_METHOD,
        n_components: int = DEFAULT_N_COMPONENTS,
        local_scale_k: int = DEFAULT_LOCAL_SCALE_K,
        alpha: float = DEFAULT_ALPHA,
        diffusion_time: int | float = DEFAULT_DIFFUSION_TIME,
        eps: float = DEFAULT_EPS,
        eigenvalue_min: float = DEFAULT_EIGENVALUE_MIN,
        random_state: int = 42,
        transform: str = "sqrt_counts",
        standardize: bool = True,
        dtype: str | np.dtype = "float32",
        ood_distance_q: float = 0.99,
        ood_distance_factor: float = 2.0,
        ood_neff_min: float = 2.0,
    ):
        method = str(landmark_method).strip().lower()
        if method not in LANDMARK_METHODS:
            raise ValueError(
                f"landmark_method must be one of {LANDMARK_METHODS}, got {landmark_method!r}"
            )
        if transform not in ("counts", "sqrt_counts", "rates", "zscore_rates", "none"):
            raise ValueError(
                "transform must be one of: counts, sqrt_counts, rates, zscore_rates, none"
            )
        self.n_landmarks = int(n_landmarks)
        self.landmark_method = method
        self.n_components = int(n_components)
        self.local_scale_k = int(local_scale_k)
        self.alpha = float(alpha)
        self.diffusion_time = float(diffusion_time)
        self.eps = float(eps)
        self.eigenvalue_min = float(eigenvalue_min)
        self.random_state = int(random_state)
        self.transform_name = transform
        self.standardize = bool(standardize)
        self.dtype = np.dtype(dtype)
        self.ood_distance_q = float(ood_distance_q)
        self.ood_distance_factor = float(ood_distance_factor)
        self.ood_neff_min = float(ood_neff_min)

        self.scaler_: StandardScaler | None = None
        self.scale_mean_: np.ndarray | None = None
        self.scale_scale_: np.ndarray | None = None
        self.landmarks_: np.ndarray | None = None
        self.landmark_sq_norms_: np.ndarray | None = None
        self.landmark_scales_: np.ndarray | None = None
        self.q_landmarks_: np.ndarray | None = None
        self.q_landmarks_alpha_: np.ndarray | None = None
        self.degree_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvalues_all_: np.ndarray | None = None
        self.eigenvectors_right_: np.ndarray | None = None
        self.projection_matrix_: np.ndarray | None = None
        self.landmark_coordinates_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.n_landmarks_fitted_: int | None = None
        self.actual_n_components_: int | None = None
        self.n_components_dropped_: int = 0
        self.dropped_eigenvalues_: list[float] = []
        self.fit_n_samples_: int | None = None
        self.memory_bytes_: int | None = None
        self.ood_nearest_threshold_: float | None = None
        self.last_ood_: dict[str, Any] | None = None
        self.last_stage_latencies_ms_: dict[str, float] | None = None
        self._fit_count: int = 0
        self._d2_buf: np.ndarray | None = None
        self._k_buf: np.ndarray | None = None
        self._p_buf: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Preprocessing (training-time scaler is frozen at fit)
    # ------------------------------------------------------------------
    def _count_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if self.transform_name == "sqrt_counts":
            return np.sqrt(np.maximum(X, 0.0))
        if self.transform_name in ("rates", "zscore_rates"):
            return np.maximum(X, 0.0)
        return X

    def _fit_preprocess(self, X_train: np.ndarray) -> np.ndarray:
        X = self._count_transform(X_train)
        self.n_features_in_ = int(X.shape[1])
        if self.standardize or self.transform_name == "zscore_rates":
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
            self.scale_mean_ = _as_contig(self.scaler_.mean_, self.dtype)
            scale = np.asarray(self.scaler_.scale_, dtype=np.float64)
            scale = np.where(np.abs(scale) < self.eps, 1.0, scale)
            self.scale_scale_ = _as_contig(scale, self.dtype)
        else:
            self.scaler_ = None
            self.scale_mean_ = None
            self.scale_scale_ = None
        return X

    def _transform_preprocess(self, X: np.ndarray) -> np.ndarray:
        X = self._count_transform(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return X

    def _scale_one(self, x: np.ndarray) -> np.ndarray:
        """Hot-path scaling without sklearn (uses frozen mean/scale)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if self.transform_name == "sqrt_counts":
            x = np.sqrt(np.maximum(x, 0.0))
        elif self.transform_name in ("rates", "zscore_rates"):
            x = np.maximum(x, 0.0)
        if self.scale_mean_ is not None and self.scale_scale_ is not None:
            x = (x - np.asarray(self.scale_mean_, dtype=np.float64)) / np.asarray(
                self.scale_scale_, dtype=np.float64
            )
        return _as_contig(x, self.dtype)

    # ------------------------------------------------------------------
    # Landmarks
    # ------------------------------------------------------------------
    def _select_landmarks(self, X: np.ndarray) -> np.ndarray:
        n = int(X.shape[0])
        m = int(min(max(2, self.n_landmarks), n))
        rng = np.random.default_rng(self.random_state)
        if m >= n:
            return np.asarray(X, dtype=np.float64).copy()
        if self.landmark_method == "random":
            idx = rng.choice(n, size=m, replace=False)
            return np.asarray(X[idx], dtype=np.float64)
        if self.landmark_method == "kmeans":
            km = KMeans(
                n_clusters=m,
                random_state=self.random_state,
                n_init=10,
            )
            km.fit(X)
            return np.asarray(km.cluster_centers_, dtype=np.float64)
        km = MiniBatchKMeans(
            n_clusters=m,
            random_state=self.random_state,
            batch_size=min(1024, max(256, m * 2)),
            n_init=3,
        )
        km.fit(X)
        return np.asarray(km.cluster_centers_, dtype=np.float64)

    def _local_scales(self, d2: np.ndarray) -> np.ndarray:
        m = int(d2.shape[0])
        k = int(min(max(1, self.local_scale_k), max(1, m - 1)))
        d = np.sqrt(np.maximum(d2, 0.0))
        d = d.copy()
        np.fill_diagonal(d, np.inf)
        # k-th nearest *other* landmark
        scales = np.partition(d, kth=k - 1, axis=1)[:, k - 1]
        return np.maximum(scales, self.eps)

    def _query_sigma(self, d2_row: np.ndarray) -> float:
        m = int(d2_row.size)
        k = int(min(max(1, self.local_scale_k), m))
        d2k = float(np.partition(d2_row, kth=k - 1)[k - 1])
        return float(max(np.sqrt(max(d2k, 0.0)), self.eps))

    def _allocate_buffers(self, m: int) -> None:
        dt = self.dtype
        self._d2_buf = np.empty(m, dtype=dt)
        self._k_buf = np.empty(m, dtype=dt)
        self._p_buf = np.empty(m, dtype=dt)

    def _memory_bytes(self) -> int:
        total = 0
        for arr in (
            self.landmarks_,
            self.landmark_sq_norms_,
            self.landmark_scales_,
            self.q_landmarks_,
            self.q_landmarks_alpha_,
            self.degree_,
            self.eigenvalues_,
            self.eigenvectors_right_,
            self.projection_matrix_,
            self.landmark_coordinates_,
            self.scale_mean_,
            self.scale_scale_,
        ):
            if arr is not None:
                total += int(np.asarray(arr).nbytes)
        return total

    # ------------------------------------------------------------------
    # Fit (offline only)
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, labels: Any | None = None) -> DiffusionNystrom:
        self._fit_count += 1
        X_train = np.asarray(X_train)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2-D [n_times, n_features]")
        X = self._fit_preprocess(X_train)
        self.fit_n_samples_ = int(X.shape[0])

        landmarks = self._select_landmarks(X)
        m = int(landmarks.shape[0])
        self.n_landmarks_fitted_ = m

        d2 = pairwise_sq_euclidean(landmarks)
        scales = self._local_scales(d2)
        denom = scales[:, None] * scales[None, :] + self.eps
        K = np.exp(-d2 / denom)

        q = np.maximum(K.sum(axis=1), self.eps)
        q_alpha = np.power(q, self.alpha)
        K_tilde = K / (q_alpha[:, None] * q_alpha[None, :] + self.eps)

        degree = np.maximum(K_tilde.sum(axis=1), self.eps)
        d_inv_sqrt = 1.0 / np.sqrt(degree)
        A = (d_inv_sqrt[:, None] * K_tilde) * d_inv_sqrt[None, :]
        A = 0.5 * (A + A.T)

        evals, evecs = np.linalg.eigh(A)
        order = np.argsort(evals)[::-1]
        evals = np.real(evals[order])
        evecs = np.real(evecs[:, order])
        self.eigenvalues_all_ = np.asarray(evals, dtype=np.float64)

        # Drop trivial (largest, λ≈1) mode; keep subsequent stable modes.
        nontrivial = evals[1:]
        evecs_nt = evecs[:, 1:]
        keep: list[int] = []
        dropped: list[float] = []
        for i, lam in enumerate(nontrivial):
            if abs(float(lam)) < self.eigenvalue_min:
                dropped.append(float(lam))
                continue
            keep.append(i)
            if len(keep) >= self.n_components:
                break
        if not keep:
            raise RuntimeError(
                "DiffusionNystrom: no stable nontrivial eigenmodes "
                f"(eigenvalue_min={self.eigenvalue_min})"
            )
        self.n_components_dropped_ = int(max(0, self.n_components - len(keep)))
        self.dropped_eigenvalues_ = dropped
        evals_k = nontrivial[keep]
        # Right eigenvectors of P = D^{-1} K̃ : ψ = D^{-1/2} v
        psi = d_inv_sqrt[:, None] * evecs_nt[:, keep]
        # Column sign convention: make mean positive for reproducibility.
        signs = np.sign(psi.mean(axis=0))
        signs[signs == 0] = 1.0
        psi = psi * signs[None, :]

        tau = self.diffusion_time
        weights = np.power(np.abs(evals_k), tau - 1.0) * np.sign(evals_k)
        # λ^{τ-1} may be negative for odd integer powers of negative λ; use real power.
        projection = psi * weights[None, :]
        landmark_coords = psi * np.power(np.abs(evals_k), tau)[None, :] * np.sign(evals_k)[None, :]

        dt = self.dtype
        self.landmarks_ = _as_contig(landmarks, dt)
        self.landmark_sq_norms_ = _as_contig(np.einsum("ij,ij->i", landmarks, landmarks), dt)
        self.landmark_scales_ = _as_contig(scales, dt)
        self.q_landmarks_ = _as_contig(q, dt)
        self.q_landmarks_alpha_ = _as_contig(q_alpha, dt)
        self.degree_ = _as_contig(degree, dt)
        self.eigenvalues_ = _as_contig(evals_k, np.float64)
        self.eigenvectors_right_ = _as_contig(psi, dt)
        self.projection_matrix_ = _as_contig(projection, dt)
        self.landmark_coordinates_ = _as_contig(landmark_coords, dt)
        self.actual_n_components_ = int(self.eigenvalues_.shape[0])
        self._allocate_buffers(m)

        # OOD threshold from landmark nearest-neighbor distances.
        nn = np.partition(d2 + np.eye(m) * np.max(d2 + 1.0), 1, axis=1)[:, 1]
        nn = np.sqrt(np.maximum(nn, 0.0))
        self.ood_nearest_threshold_ = float(
            np.quantile(nn, self.ood_distance_q) * self.ood_distance_factor
        )
        self.memory_bytes_ = self._memory_bytes()
        self.last_ood_ = None
        self.last_stage_latencies_ms_ = None
        return self

    def fit_transform(
        self,
        X_train: np.ndarray,
        labels: Any | None = None,
    ) -> np.ndarray:
        self.fit(X_train, labels=labels)
        return self.transform(X_train)

    # ------------------------------------------------------------------
    # Transform (online)
    # ------------------------------------------------------------------
    def _require_fitted(self) -> None:
        if self.landmarks_ is None or self.projection_matrix_ is None:
            raise RuntimeError("DiffusionNystrom must be fit before transform")

    def _kernel_and_weights(
        self,
        d2: np.ndarray,
        *,
        sigma_x: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (kernel, transition weights p, sigma_x) for queries.

        ``d2`` is (n, M) or (M,).
        """
        scales = np.asarray(self.landmark_scales_, dtype=np.float64)
        q_alpha = np.asarray(self.q_landmarks_alpha_, dtype=np.float64)
        d2 = np.maximum(np.asarray(d2, dtype=np.float64), 0.0)
        squeeze = d2.ndim == 1
        if squeeze:
            d2 = d2[None, :]
        if np.isscalar(sigma_x):
            sig = np.full(d2.shape[0], float(sigma_x), dtype=np.float64)
        else:
            sig = np.asarray(sigma_x, dtype=np.float64).ravel()
        denom = sig[:, None] * scales[None, :] + self.eps
        K = np.exp(-d2 / denom)
        q_x = np.maximum(K.sum(axis=1, keepdims=True), self.eps)
        K_tilde = K / (np.power(q_x, self.alpha) * q_alpha[None, :] + self.eps)
        deg = np.maximum(K_tilde.sum(axis=1, keepdims=True), self.eps)
        p = K_tilde / deg
        if squeeze:
            return K[0], p[0], sig
        return K, p, sig

    def _query_d2_batch(self, X: np.ndarray) -> np.ndarray:
        L = np.asarray(self.landmarks_, dtype=np.float64)
        X = np.asarray(X, dtype=np.float64)
        x2 = np.einsum("ij,ij->i", X, X)[:, None]
        d2 = x2 + np.asarray(self.landmark_sq_norms_, dtype=np.float64)[None, :] - 2.0 * (X @ L.T)
        np.maximum(d2, 0.0, out=d2)
        return d2

    def _sigma_x_batch(self, d2: np.ndarray) -> np.ndarray:
        m = int(d2.shape[1])
        k = int(min(max(1, self.local_scale_k), m))
        d2k = np.partition(d2, kth=k - 1, axis=1)[:, k - 1]
        return np.maximum(np.sqrt(np.maximum(d2k, 0.0)), self.eps)

    def transform(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = np.asarray(X)
        if X.ndim == 1:
            return self.transform_one(X).reshape(1, -1)
        Xp = self._transform_preprocess(X)
        d2 = self._query_d2_batch(Xp)
        sigma_x = self._sigma_x_batch(d2)
        _, p, _ = self._kernel_and_weights(d2, sigma_x=sigma_x)
        Z = p @ np.asarray(self.projection_matrix_, dtype=np.float64)
        return np.asarray(Z, dtype=self.dtype)

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        """Streaming Nyström projection of a single feature vector."""
        self._require_fitted()
        t0 = time.perf_counter_ns()
        x_scaled = self._scale_one(x)
        t_scale = time.perf_counter_ns()

        landmarks = self.landmarks_
        assert landmarks is not None
        m = int(landmarks.shape[0])
        d2 = self._d2_buf
        k_buf = self._k_buf
        p_buf = self._p_buf
        assert d2 is not None and k_buf is not None and p_buf is not None

        x_norm = float(x_scaled @ x_scaled)
        # d2 = ||L||^2 + ||x||^2 - 2 L x
        np.dot(landmarks, x_scaled, out=d2)
        d2 *= np.float32(-2.0) if self.dtype == np.float32 else -2.0
        d2 += self.landmark_sq_norms_
        d2 += self.dtype.type(x_norm) if hasattr(self.dtype, "type") else x_norm
        np.maximum(d2, 0.0, out=d2)

        sigma_x = self._query_sigma(np.asarray(d2, dtype=np.float64))
        # p_buf holds denom = σ_x σ_j + ε, then k_buf = exp(-d2 / denom)
        np.multiply(self.landmark_scales_, self.dtype.type(sigma_x), out=p_buf)
        p_buf += self.dtype.type(self.eps)
        np.divide(d2, p_buf, out=k_buf)
        np.negative(k_buf, out=k_buf)
        np.exp(k_buf, out=k_buf)
        q_x = float(max(np.sum(k_buf, dtype=np.float64), self.eps))
        q_x_alpha = q_x ** self.alpha
        # k̃_j = k_j / (q_x^α q_j^α) stored in p_buf
        np.multiply(self.q_landmarks_alpha_, self.dtype.type(q_x_alpha), out=p_buf)
        p_buf += self.dtype.type(self.eps)
        np.divide(k_buf, p_buf, out=p_buf)
        deg = float(max(np.sum(p_buf, dtype=np.float64), self.eps))
        p_buf /= self.dtype.type(deg)
        z = np.dot(p_buf, self.projection_matrix_)
        t1 = time.perf_counter_ns()

        p64 = np.asarray(p_buf, dtype=np.float64)
        k64 = np.asarray(k_buf, dtype=np.float64)
        nearest = float(np.sqrt(float(np.min(d2))))
        ksum = float(max(k64.sum(), self.eps))
        k_norm = k64 / ksum
        entropy = _entropy(p64, self.eps)
        n_eff = float(1.0 / max(np.sum(p64 * p64), self.eps))
        max_w = float(np.max(p64))
        ood = bool(
            (self.ood_nearest_threshold_ is not None and nearest > self.ood_nearest_threshold_)
            or n_eff < self.ood_neff_min
        )
        self.last_ood_ = {
            "nearest_landmark_distance": nearest,
            "sigma_x": float(sigma_x),
            "max_kernel_weight": max_w,
            "kernel_entropy": entropy,
            "effective_n_landmarks": n_eff,
            "max_unnormalized_kernel": float(np.max(k64)),
            "ood_flag": ood,
        }
        self.last_stage_latencies_ms_ = {
            "feature_scaling_ms": (t_scale - t0) / 1e6,
            "diffusion_nystrom_transform_ms": (t1 - t_scale) / 1e6,
            "total_ms": (t1 - t0) / 1e6,
        }
        return np.asarray(z, dtype=self.dtype)

    def query_diagnostics(self, x: np.ndarray) -> dict[str, Any]:
        """Compute OOD / kernel diagnostics for one observation (also updates last_ood_)."""
        self.transform_one(x)
        return dict(self.last_ood_ or {})

    def query_diagnostics_batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        """Vectorized OOD diagnostics for a batch of observations."""
        self._require_fitted()
        Xp = self._transform_preprocess(np.asarray(X))
        d2 = self._query_d2_batch(Xp)
        sigma_x = self._sigma_x_batch(d2)
        K, p, _ = self._kernel_and_weights(d2, sigma_x=sigma_x)
        nearest = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        max_w = p.max(axis=1)
        p_clip = np.clip(p, self.eps, None)
        p_clip = p_clip / p_clip.sum(axis=1, keepdims=True)
        entropy = -np.sum(p_clip * np.log(p_clip), axis=1)
        n_eff = 1.0 / np.maximum(np.sum(p * p, axis=1), self.eps)
        thresh = self.ood_nearest_threshold_
        ood = (n_eff < self.ood_neff_min)
        if thresh is not None:
            ood = ood | (nearest > thresh)
        return {
            "nearest_landmark_distance": nearest.astype(np.float64),
            "sigma_x": np.asarray(sigma_x, dtype=np.float64),
            "max_kernel_weight": max_w.astype(np.float64),
            "kernel_entropy": entropy.astype(np.float64),
            "effective_n_landmarks": n_eff.astype(np.float64),
            "max_unnormalized_kernel": K.max(axis=1).astype(np.float64),
            "ood_flag": ood.astype(bool),
        }

    @property
    def latent_dim(self) -> int:
        if self.actual_n_components_ is None:
            raise RuntimeError("DiffusionNystrom must be fit before latent_dim")
        return int(self.actual_n_components_)

    def nystrom_landmark_consistency(self) -> dict[str, float]:
        """Compare Nyström(landmarks) to stored landmark diffusion coordinates."""
        self._require_fitted()
        assert self.landmarks_ is not None and self.landmark_coordinates_ is not None
        # Landmarks are already in the scaled feature space used at fit.
        d2 = pairwise_sq_euclidean(np.asarray(self.landmarks_, dtype=np.float64))
        sigma_x = self._sigma_x_batch(d2)
        _, p, _ = self._kernel_and_weights(d2, sigma_x=sigma_x)
        Z = p @ np.asarray(self.projection_matrix_, dtype=np.float64)
        ref = np.asarray(self.landmark_coordinates_, dtype=np.float64)
        err = Z - ref
        mse = float(np.mean(err ** 2))
        denom = np.linalg.norm(ref - ref.mean(axis=0)) + self.eps
        rel = float(np.linalg.norm(err) / denom)
        corr = []
        for k in range(ref.shape[1]):
            if np.std(ref[:, k]) < self.eps or np.std(Z[:, k]) < self.eps:
                continue
            corr.append(float(np.corrcoef(ref[:, k], Z[:, k])[0, 1]))
        return {
            "landmark_mse": mse,
            "landmark_relative_error": rel,
            "landmark_mean_correlation": float(np.mean(corr)) if corr else float("nan"),
            "n_landmarks": int(ref.shape[0]),
            "n_components": int(ref.shape[1]),
        }

    def get_metadata(self) -> dict[str, Any]:
        return {
            "manifold_type": DIFFUSION_NYSTROM_NAME,
            "manifold_method": DIFFUSION_NYSTROM_NAME,
            "manifold_grouping": None,
            "manifold_n_components": self.n_components,
            "actual_n_features": self.actual_n_components_,
            "n_landmarks": self.n_landmarks_fitted_ or self.n_landmarks,
            "n_landmarks_requested": self.n_landmarks,
            "landmark_method": self.landmark_method,
            "local_scale_k": self.local_scale_k,
            "alpha": self.alpha,
            "diffusion_time": self.diffusion_time,
            "eps": self.eps,
            "eigenvalue_min": self.eigenvalue_min,
            "n_components_dropped": self.n_components_dropped_,
            "dropped_eigenvalues": list(self.dropped_eigenvalues_),
            "eigenvalues": (
                np.asarray(self.eigenvalues_).tolist()
                if self.eigenvalues_ is not None else None
            ),
            "activity_representation": self.transform_name,
            "standardize": self.standardize,
            "dtype": str(self.dtype),
            "realtime_compatible": self.realtime_compatible,
            "supports_realtime": True,
            "deployment_tag": self.deployment_tag,
            "memory_bytes": self.memory_bytes_,
            "fit_n_samples": self.fit_n_samples_,
            "n_features_in": self.n_features_in_,
            "ood_nearest_threshold": self.ood_nearest_threshold_,
            "nystrom_consistency": (
                self.nystrom_landmark_consistency()
                if self.landmarks_ is not None else None
            ),
            "explained_variance_ratio": None,
            "groups": [{
                "group_name": "all",
                "n_units": self.n_features_in_,
                "n_components": self.actual_n_components_,
            }],
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, output_dir: Path) -> None:
        self._require_fitted()
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "class_name": "DiffusionNystrom",
            "n_landmarks": self.n_landmarks,
            "n_landmarks_fitted": self.n_landmarks_fitted_,
            "landmark_method": self.landmark_method,
            "n_components": self.n_components,
            "actual_n_components": self.actual_n_components_,
            "local_scale_k": self.local_scale_k,
            "alpha": self.alpha,
            "diffusion_time": self.diffusion_time,
            "eps": self.eps,
            "eigenvalue_min": self.eigenvalue_min,
            "random_state": self.random_state,
            "transform": self.transform_name,
            "standardize": self.standardize,
            "dtype": str(self.dtype),
            "n_features_in": self.n_features_in_,
            "fit_n_samples": self.fit_n_samples_,
            "n_components_dropped": self.n_components_dropped_,
            "dropped_eigenvalues": list(self.dropped_eigenvalues_),
            "memory_bytes": self.memory_bytes_,
            "ood_nearest_threshold": self.ood_nearest_threshold_,
            "ood_distance_q": self.ood_distance_q,
            "ood_distance_factor": self.ood_distance_factor,
            "ood_neff_min": self.ood_neff_min,
            "realtime_compatible": self.realtime_compatible,
            "deployment_tag": self.deployment_tag,
            "eigenvalues": (
                np.asarray(self.eigenvalues_).tolist()
                if self.eigenvalues_ is not None else None
            ),
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        arrays = {
            "landmarks.npy": self.landmarks_,
            "landmark_sq_norms.npy": self.landmark_sq_norms_,
            "landmark_scales.npy": self.landmark_scales_,
            "q_landmarks.npy": self.q_landmarks_,
            "q_landmarks_alpha.npy": self.q_landmarks_alpha_,
            "degree.npy": self.degree_,
            "eigenvalues.npy": self.eigenvalues_,
            "eigenvectors_right.npy": self.eigenvectors_right_,
            "projection_matrix.npy": self.projection_matrix_,
            "landmark_coordinates.npy": self.landmark_coordinates_,
        }
        if self.eigenvalues_all_ is not None:
            arrays["eigenvalues_all.npy"] = self.eigenvalues_all_
        if self.scale_mean_ is not None:
            arrays["scale_mean.npy"] = self.scale_mean_
        if self.scale_scale_ is not None:
            arrays["scale_scale.npy"] = self.scale_scale_
        for name, arr in arrays.items():
            if arr is not None:
                np.save(output_dir / name, arr)
        if self.scaler_ is not None:
            joblib.dump(self.scaler_, output_dir / "scaler.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> DiffusionNystrom:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            n_landmarks=int(meta.get("n_landmarks", DEFAULT_N_LANDMARKS)),
            landmark_method=meta.get("landmark_method", DEFAULT_LANDMARK_METHOD),
            n_components=int(meta.get("n_components", DEFAULT_N_COMPONENTS)),
            local_scale_k=int(meta.get("local_scale_k", DEFAULT_LOCAL_SCALE_K)),
            alpha=float(meta.get("alpha", DEFAULT_ALPHA)),
            diffusion_time=float(meta.get("diffusion_time", DEFAULT_DIFFUSION_TIME)),
            eps=float(meta.get("eps", DEFAULT_EPS)),
            eigenvalue_min=float(meta.get("eigenvalue_min", DEFAULT_EIGENVALUE_MIN)),
            random_state=int(meta.get("random_state", 42)),
            transform=meta.get("transform", "sqrt_counts"),
            standardize=bool(meta.get("standardize", True)),
            dtype=meta.get("dtype", "float32"),
            ood_distance_q=float(meta.get("ood_distance_q", 0.99)),
            ood_distance_factor=float(meta.get("ood_distance_factor", 2.0)),
            ood_neff_min=float(meta.get("ood_neff_min", 2.0)),
        )
        obj.n_landmarks_fitted_ = meta.get("n_landmarks_fitted")
        obj.actual_n_components_ = meta.get("actual_n_components")
        obj.n_features_in_ = meta.get("n_features_in")
        obj.fit_n_samples_ = meta.get("fit_n_samples")
        obj.n_components_dropped_ = int(meta.get("n_components_dropped", 0))
        obj.dropped_eigenvalues_ = list(meta.get("dropped_eigenvalues") or [])
        obj.memory_bytes_ = meta.get("memory_bytes")
        obj.ood_nearest_threshold_ = meta.get("ood_nearest_threshold")

        def _load_npy(name: str) -> np.ndarray | None:
            path = input_dir / name
            if path.exists():
                return np.load(path)
            return None

        obj.landmarks_ = _load_npy("landmarks.npy")
        obj.landmark_sq_norms_ = _load_npy("landmark_sq_norms.npy")
        obj.landmark_scales_ = _load_npy("landmark_scales.npy")
        obj.q_landmarks_ = _load_npy("q_landmarks.npy")
        obj.q_landmarks_alpha_ = _load_npy("q_landmarks_alpha.npy")
        obj.degree_ = _load_npy("degree.npy")
        obj.eigenvalues_ = _load_npy("eigenvalues.npy")
        obj.eigenvectors_right_ = _load_npy("eigenvectors_right.npy")
        obj.projection_matrix_ = _load_npy("projection_matrix.npy")
        obj.landmark_coordinates_ = _load_npy("landmark_coordinates.npy")
        obj.eigenvalues_all_ = _load_npy("eigenvalues_all.npy")
        obj.scale_mean_ = _load_npy("scale_mean.npy")
        obj.scale_scale_ = _load_npy("scale_scale.npy")
        scaler_path = input_dir / "scaler.joblib"
        if scaler_path.exists():
            obj.scaler_ = joblib.load(scaler_path)
        if obj.landmarks_ is not None:
            obj._allocate_buffers(int(obj.landmarks_.shape[0]))
            if obj.n_landmarks_fitted_ is None:
                obj.n_landmarks_fitted_ = int(obj.landmarks_.shape[0])
        return obj
