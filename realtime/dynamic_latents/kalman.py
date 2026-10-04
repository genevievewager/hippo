"""Numerically stable Kalman filter / RTS smoother for linear Gaussian LDS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import solve

# Full eigenvalue clip is cheap for latent covariances (k ≈ 2–10) and expensive
# for observation-space S/R (n ≈ 100+). Large matrices use symmetrize + jitter.
_EIGH_PSD_MAX_DIM = 24


@dataclass
class FilterResult:
    mu: np.ndarray  # [T, d]
    P: np.ndarray  # [T, d, d]
    mu_pred: np.ndarray  # [T, d]
    P_pred: np.ndarray  # [T, d, d]
    loglik: float


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def _ensure_psd(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    M = _symmetrize(np.asarray(M, dtype=float))
    n = int(M.shape[0])
    if n <= _EIGH_PSD_MAX_DIM:
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, eps)
        return (V * w) @ V.T
    return M + float(eps) * np.eye(n)


def _woodbury_prepare(C: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precompute ``R^{-1} C`` and ``C^T R^{-1} C`` for n ≫ k updates."""
    R = _ensure_psd(R)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
    Rinv_C = R_inv @ C
    M = C.T @ Rinv_C
    return Rinv_C, M


def _woodbury_gain(P_p: np.ndarray, C: np.ndarray, Rinv_C: np.ndarray, M: np.ndarray):
    """Kalman gain via Woodbury: ``K = P C^T (C P C^T + R)^{-1}`` without forming S."""
    d_lat = P_p.shape[0]
    try:
        P_inv = np.linalg.inv(P_p)
    except np.linalg.LinAlgError:
        P_inv = np.linalg.pinv(P_p)
    inner = P_inv + M
    try:
        Sinv_C = Rinv_C - Rinv_C @ solve(inner, M)
    except np.linalg.LinAlgError:
        Sinv_C = Rinv_C - Rinv_C @ (np.linalg.pinv(inner) @ M)
    K = P_p @ Sinv_C.T
    I = np.eye(d_lat)
    return K, I


def kalman_filter(
    X: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu0: np.ndarray,
    P0: np.ndarray,
    *,
    compute_loglik: bool = True,
    predict_first: bool = False,
) -> FilterResult:
    """Causal Kalman filter over sequence ``X`` of shape ``[T, n]``.

    ``predict_first=False`` treats ``mu0`` as the filtered belief at t=0's prior
    (no ``A`` predict), matching sequential ``step(..., is_first=True)``.
    ``predict_first=True`` applies the transition before the first update
    (warm-start continuation).
    """
    X = np.asarray(X, dtype=float)
    T, n = X.shape
    d_lat = A.shape[0]
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    Q = _ensure_psd(Q)
    R = _ensure_psd(R)
    mu0 = np.asarray(mu0, dtype=float).ravel()
    P0 = _ensure_psd(P0)
    eye_n = np.eye(n)
    use_woodbury = n > _EIGH_PSD_MAX_DIM
    Rinv_C = M = None
    if use_woodbury:
        Rinv_C, M = _woodbury_prepare(C, R)

    mu = np.zeros((T, d_lat))
    P = np.zeros((T, d_lat, d_lat))
    mu_pred = np.zeros((T, d_lat))
    P_pred = np.zeros((T, d_lat, d_lat))
    loglik = 0.0
    I = np.eye(d_lat)

    m = mu0.copy()
    P_t = P0.copy()
    for t in range(T):
        if t == 0 and not predict_first:
            m_pred = m
            P_p = P_t
        else:
            m_pred = A @ m
            P_p = _ensure_psd(A @ P_t @ A.T + Q)

        mu_pred[t] = m_pred
        P_pred[t] = P_p

        innov = X[t] - (C @ m_pred + d)
        S = None
        if use_woodbury:
            K, I = _woodbury_gain(P_p, C, Rinv_C, M)
            S = None
        else:
            S = _ensure_psd(C @ P_p @ C.T + R)
            try:
                K = solve(S + 1e-8 * eye_n, (P_p @ C.T).T).T
            except np.linalg.LinAlgError:
                K = (P_p @ C.T) @ np.linalg.pinv(S)

        m = m_pred + K @ innov
        P_t = _ensure_psd((I - K @ C) @ P_p)
        mu[t] = m
        P[t] = P_t

        if compute_loglik and S is not None:
            sign, logdet = np.linalg.slogdet(S)
            if sign <= 0:
                logdet = np.log(np.maximum(np.linalg.det(S + 1e-8 * eye_n), 1e-12))
            quad = float(innov @ solve(S + 1e-8 * eye_n, innov))
            loglik += -0.5 * (n * np.log(2.0 * np.pi) + logdet + quad)

    return FilterResult(mu=mu, P=P, mu_pred=mu_pred, P_pred=P_pred, loglik=loglik)


def kalman_filter_step(
    x_t: np.ndarray,
    mu_prev: np.ndarray,
    P_prev: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    is_first: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Single causal filter update. Returns ``(mu_t, P_t)``."""
    x_t = np.asarray(x_t, dtype=float).ravel()
    mu_prev = np.asarray(mu_prev, dtype=float).ravel()
    P_prev = _ensure_psd(P_prev)
    A = np.asarray(A, dtype=float)
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float).ravel()
    n = int(C.shape[0])
    d_lat = A.shape[0]
    I = np.eye(d_lat)
    if Q.shape[0] <= _EIGH_PSD_MAX_DIM:
        Q = _ensure_psd(Q)

    if is_first:
        m_pred = mu_prev
        P_p = P_prev
    else:
        m_pred = A @ mu_prev
        P_p = _ensure_psd(A @ P_prev @ A.T + Q)

    innov = x_t - (C @ m_pred + d)
    if n > _EIGH_PSD_MAX_DIM:
        Rinv_C, M = _woodbury_prepare(C, R)
        K, I = _woodbury_gain(P_p, C, Rinv_C, M)
    else:
        R = _ensure_psd(R)
        S = _ensure_psd(C @ P_p @ C.T + R)
        try:
            K = solve(S + 1e-8 * np.eye(n), (P_p @ C.T).T).T
        except np.linalg.LinAlgError:
            K = (P_p @ C.T) @ np.linalg.pinv(S)
    mu = m_pred + K @ innov
    P = _ensure_psd((I - K @ C) @ P_p)
    return mu, P


def rts_smooth(filt: FilterResult, A: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rauch–Tung–Striebel smoother (acausal). Returns ``(mu_s, P_s)``."""
    A = np.asarray(A, dtype=float)
    Q = _ensure_psd(Q)
    T, d_lat = filt.mu.shape
    mu_s = filt.mu.copy()
    P_s = filt.P.copy()
    for t in range(T - 2, -1, -1):
        P_p = filt.P_pred[t + 1]
        try:
            G = solve(P_p, (filt.P[t] @ A.T).T).T
        except np.linalg.LinAlgError:
            G = filt.P[t] @ A.T @ np.linalg.pinv(P_p)
        mu_s[t] = filt.mu[t] + G @ (mu_s[t + 1] - filt.mu_pred[t + 1])
        P_s[t] = _ensure_psd(filt.P[t] + G @ (P_s[t + 1] - P_p) @ G.T)
    return mu_s, P_s
