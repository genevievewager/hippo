"""Numerically stable Kalman filter / RTS smoother for linear Gaussian LDS."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import solve


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
    # Eigenvalue clip for numerical PSD.
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def kalman_filter(
    X: np.ndarray,
    A: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    mu0: np.ndarray,
    P0: np.ndarray,
) -> FilterResult:
    """Causal Kalman filter over sequence ``X`` of shape ``[T, n]``."""
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

    mu = np.zeros((T, d_lat))
    P = np.zeros((T, d_lat, d_lat))
    mu_pred = np.zeros((T, d_lat))
    P_pred = np.zeros((T, d_lat, d_lat))
    loglik = 0.0
    I = np.eye(d_lat)

    m = mu0.copy()
    P_t = P0.copy()
    for t in range(T):
        if t == 0:
            m_pred = m
            P_p = P_t
        else:
            m_pred = A @ m
            P_p = _ensure_psd(A @ P_t @ A.T + Q)

        mu_pred[t] = m_pred
        P_pred[t] = P_p

        innov = X[t] - (C @ m_pred + d)
        S = _ensure_psd(C @ P_p @ C.T + R)
        # K = P C^T S^{-1}
        try:
            K = solve(S, (P_p @ C.T).T).T
        except np.linalg.LinAlgError:
            K = (P_p @ C.T) @ np.linalg.pinv(S)

        m = m_pred + K @ innov
        P_t = _ensure_psd((I - K @ C) @ P_p)
        mu[t] = m
        P[t] = P_t

        # Gaussian log-likelihood contribution.
        sign, logdet = np.linalg.slogdet(S)
        if sign <= 0:
            logdet = np.log(np.maximum(np.linalg.det(S + 1e-8 * np.eye(n)), 1e-12))
        quad = float(innov @ solve(S + 1e-8 * np.eye(n), innov))
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
    Q = _ensure_psd(Q)
    R = _ensure_psd(R)
    d_lat = A.shape[0]
    I = np.eye(d_lat)

    if is_first:
        m_pred = mu_prev
        P_p = P_prev
    else:
        m_pred = A @ mu_prev
        P_p = _ensure_psd(A @ P_prev @ A.T + Q)

    innov = x_t - (C @ m_pred + d)
    S = _ensure_psd(C @ P_p @ C.T + R)
    try:
        K = solve(S, (P_p @ C.T).T).T
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
