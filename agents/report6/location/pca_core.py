"""PCA hard-split levels 1–3 (amendment stop point 2). Standard-library / sklearn only."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from sklearn.covariance import LedoitWolf
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

from agents.report6.target_rule import OCC_MIN, VISITS_MIN, choose_k
from agents.report6.visits import count_visits


ANGLE_STABLE_DEG = 30.0


def _sqrt(X: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.asarray(X, dtype=float), 0.0, None))


def project_u(Z: np.ndarray) -> np.ndarray:
    """Remove global-rate direction u = 1/sqrt(N) in z-space."""
    n = Z.shape[1]
    u = np.ones(n, dtype=float) / np.sqrt(n)
    return Z - np.outer(Z @ u, u)


def fit_pca_half(
    X: np.ndarray,
    mask: np.ndarray,
    *,
    d: int,
    drop_u: bool,
) -> dict[str, Any]:
    """Fit mean/sd/V on half; return transform for all bins."""
    mask = np.asarray(mask, dtype=bool)
    Xs = _sqrt(X)
    m = Xs[mask].mean(axis=0)
    s = Xs[mask].std(axis=0)
    s = np.where(s < 1e-8, 1.0, s)
    Z_all = (Xs - m) / s
    if drop_u:
        Z_all = project_u(Z_all)
    Z_fit = Z_all[mask]
    Zc = Z_fit - Z_fit.mean(axis=0)
    # economy SVD
    _u, _s, vt = np.linalg.svd(Zc, full_matrices=False)
    d_use = int(min(d, vt.shape[0], Z_fit.shape[0] - 1, Z_fit.shape[1]))
    V = vt[:d_use].T  # N x d
    C_all = Z_all @ V  # T x d
    return {
        "m": m,
        "s": s,
        "V": V,
        "d": d_use,
        "C": C_all,
        "drop_u": drop_u,
    }


def principal_angles_deg(V_a: np.ndarray, V_b: np.ndarray) -> np.ndarray:
    """Principal angles in degrees (ascending)."""
    # subspace_angles returns radians
    ang = subspace_angles(V_a, V_b)
    return np.sort(np.degrees(np.asarray(ang, dtype=float)))


def n_stable(angles_deg: np.ndarray, thr: float = ANGLE_STABLE_DEG) -> int:
    return int(np.sum(np.asarray(angles_deg) < thr))


def rigid_R(V_a: np.ndarray, V_b: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes R from SVD of V_b.T @ V_a (loadings only)."""
    M = V_b.T @ V_a
    u, _, vt = np.linalg.svd(M)
    return u @ vt  # d x d


def cloud_rho(C: np.ndarray, mask: np.ndarray) -> float:
    X = C[mask]
    X = X - X.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(X * X, axis=1))))


def coord_errors(
    C_a: np.ndarray,
    C_b: np.ndarray,
    R: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    retained: np.ndarray,
) -> dict[str, float]:
    """Rigid and affine coordinate errors / rho_A."""
    C_b_al = C_b @ R  # c~_B = R^T c_B if R maps B->A: amendment c~_B = R^T c_B
    # R from SVD(V_b.T V_a); aligned B: c~_B(x) = R.T @ c_B(x)  => C_b @ R
    rho = cloud_rho(C_a, mask_a)
    rho = max(rho, 1e-12)
    kept = np.asarray(retained, dtype=bool)
    e = np.linalg.norm(C_a - C_b_al, axis=1) / rho

    def _summ(m: np.ndarray) -> dict[str, float]:
        mm = m & kept
        if not np.any(mm):
            return {"median": float("nan"), "p90": float("nan")}
        vals = e[mm]
        return {
            "median": float(np.median(vals)),
            "p90": float(np.percentile(vals, 90)),
        }

    # affine: map C_b -> C_a on A-bins, residual on B-bins
    A = C_b[mask_a]
    B = C_a[mask_a]
    # least squares: A W ≈ B
    W, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    C_aff = C_b @ W
    e_aff = np.linalg.norm(C_a - C_aff, axis=1) / rho
    mb = mask_b & kept
    affine_b = float(np.median(e_aff[mb])) if np.any(mb) else float("nan")

    return {
        "rho_A": rho,
        "e_rigid_A": _summ(mask_a),
        "e_rigid_B": _summ(mask_b),
        "e_rigid_all": _summ(kept),
        "e_affine_B_median": affine_b,
    }


def poisson_sqrt_var(lam: float, n: int = 4000, rng: np.random.Generator | None = None) -> float:
    rng = rng or np.random.default_rng(0)
    if lam <= 0:
        return 0.0
    # Closed-form approx for lambda >= 1 is ~0.25; still sample lightly below.
    if lam >= 1.0:
        return 0.25
    samples = rng.poisson(lam, size=n)
    return float(np.var(np.sqrt(samples)))


def sigma_noise(
    X: np.ndarray,
    mask_a: np.ndarray,
    V: np.ndarray,
    s: np.ndarray,
    *,
    rng: np.random.Generator | None = None,
) -> float:
    """Analytic lower bound: sqrt(tr V.T diag(v_i/s_i^2) V)."""
    lam = X[mask_a].mean(axis=0)
    v = np.array([poisson_sqrt_var(float(l), rng=rng) for l in lam])
    ss = np.where(s < 1e-8, 1.0, s)
    D = v / (ss * ss)
    M = V.T @ (D[:, None] * V)
    return float(np.sqrt(max(0.0, np.trace(M))))


def sigma_split_units(
    X: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    *,
    d: int,
    drop_u: bool,
    seed: int,
) -> float:
    """Empirical upper bound via two unit halves on A, distance on B."""
    rng = np.random.default_rng(seed)
    n_u = X.shape[1]
    rates = X[mask_a].mean(axis=0)
    order = np.argsort(rates)
    # balanced by rate: alternate into halves
    h0 = order[0::2]
    h1 = order[1::2]
    if h0.size < d + 1 or h1.size < d + 1:
        return float("nan")
    fa = fit_pca_half(X[:, h0], mask_a, d=d, drop_u=drop_u)
    fb = fit_pca_half(X[:, h1], mask_a, d=d, drop_u=drop_u)
    # affine align fa<-fb on A, evaluate on B
    A = fb["C"][mask_a]
    B = fa["C"][mask_a]
    W, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
    Cb = fb["C"][mask_b] @ W
    Ca = fa["C"][mask_b]
    return float(np.median(np.linalg.norm(Ca - Cb, axis=1)))


def tile_level(
    C_a: np.ndarray,
    C_b_al: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    retained: np.ndarray,
    *,
    M: int,
    sigma_n: float,
    times_s: np.ndarray,
    dt_s: float,
    seed: int,
) -> dict[str, Any]:
    """k-means tiles on each half; match; ARI; frac shift < radius."""
    kept = np.asarray(retained, dtype=bool)
    Xa = C_a[mask_a]
    Xb = C_b_al[mask_b]
    if Xa.shape[0] < M or Xb.shape[0] < M:
        return {"error": "too_few_bins", "M": M}

    km_a = KMeans(n_clusters=M, n_init=20, random_state=seed, algorithm="lloyd")
    km_b = KMeans(n_clusters=M, n_init=20, random_state=seed + 1, algorithm="lloyd")
    lab_a_fit = km_a.fit_predict(Xa)
    lab_b_fit = km_b.fit_predict(Xb)
    mu_a = km_a.cluster_centers_
    mu_b = km_b.cluster_centers_

    # drop small tiles
    keep_a = np.array([(lab_a_fit == k).sum() >= 50 for k in range(M)])
    keep_b = np.array([(lab_b_fit == k).sum() >= 50 for k in range(M)])

    # radii
    r_a = np.zeros(M)
    for k in range(M):
        pts = Xa[lab_a_fit == k]
        if pts.shape[0] == 0:
            continue
        r_a[k] = float(np.sqrt(np.mean(np.sum((pts - mu_a[k]) ** 2, axis=1))))

    # Hungarian on centroid distances
    cost = np.linalg.norm(mu_a[:, None, :] - mu_b[None, :, :], axis=2)
    # large cost for dropped
    cost[~keep_a, :] = 1e6
    cost[:, ~keep_b] = 1e6
    ra, rb = linear_sum_assignment(cost)
    matched = [(int(i), int(j)) for i, j in zip(ra, rb) if keep_a[i] and keep_b[j] and cost[i, j] < 1e5]

    # membership labels on all kept bins
    # nearest A-tile by C_a, nearest B-tile by C_b_al
    def _nn(C, mus):
        d2 = ((C[:, None, :] - mus[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)

    lab_a_all = _nn(C_a, mu_a)
    lab_b_all = _nn(C_b_al, mu_b)
    # remap B labels to A via matching
    remap = {j: i for i, j in matched}
    lab_b_matched = np.array([remap.get(int(k), -1) for k in lab_b_all])

    both = kept & (lab_b_matched >= 0)
    if both.sum() < 10:
        ari_all = float("nan")
    else:
        ari_all = float(adjusted_rand_score(lab_a_all[both], lab_b_matched[both]))

    # frac tiles with centroid shift < own radius
    n_close = 0
    n_matched = 0
    for i, j in matched:
        shift = float(np.linalg.norm(mu_a[i] - mu_b[j]))
        n_matched += 1
        if shift < max(r_a[i], 1e-12):
            n_close += 1
    frac_close = float(n_close / n_matched) if n_matched else float("nan")

    med_r_over_noise = float(
        np.median(r_a[keep_a] / max(sigma_n, 1e-12)) if np.any(keep_a) else float("nan")
    )

    # qualifying targets on A half (occupancy/visits on mask_a timeline)
    # build label series on full time for visits: use lab_a_all but only count mask_a
    qual = []
    for k in range(M):
        if not keep_a[k]:
            continue
        in_k = (lab_a_all == k) & mask_a
        occ = float(in_k.sum() / max(1, mask_a.sum()))
        # visit count needs a time series; use binary on full length
        binary = (lab_a_all == k) & mask_a
        # sparse: count visits within mask_a runs
        vis = count_visits(binary, times_s=times_s, dt_s=dt_s, min_outside_s=1.0)
        n_vis = int(vis["n_visits"])
        if occ >= OCC_MIN and n_vis >= VISITS_MIN:
            qual.append({"label": k, "occupancy": occ, "n_visits": n_vis})

    return {
        "M": M,
        "n_matched": n_matched,
        "frac_shift_lt_radius": frac_close,
        "ari_all": ari_all,
        "median_r_over_sigma_noise": med_r_over_noise,
        "n_qualifying_A": len(qual),
        "n_dropped_A": int((~keep_a).sum()),
        "n_dropped_B": int((~keep_b).sum()),
    }


def evaluate_split(
    X: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    retained: np.ndarray,
    *,
    d: int,
    drop_u: bool,
    times_s: np.ndarray,
    dt_s: float,
    seed: int,
    M: int | None = None,
    light: bool = False,
) -> dict[str, Any]:
    """One split × one d × one u-mode: levels 1–3 + resolution.

    light=True: skip resolution bracket (for N1 floor percentiles only).
    """
    fa = fit_pca_half(X, mask_a, d=d, drop_u=drop_u)
    fb = fit_pca_half(X, mask_b, d=d, drop_u=drop_u)
    d_use = min(fa["d"], fb["d"])
    Va, Vb = fa["V"][:, :d_use], fb["V"][:, :d_use]
    Ca, Cb = fa["C"][:, :d_use], fb["C"][:, :d_use]
    angles = principal_angles_deg(Va, Vb)
    R = rigid_R(Va, Vb)
    Cb_al = Cb @ R
    errs = None if light else coord_errors(Ca, Cb_al, R, mask_a, mask_b, retained)
    if light:
        sig_n = 1.0
        sig_s = None
    else:
        sig_n = sigma_noise(X, mask_a, Va, fa["s"], rng=np.random.default_rng(seed))
        sig_s = sigma_split_units(
            X, mask_a, mask_b, d=d_use, drop_u=drop_u, seed=seed + 17,
        )

    if M is None:
        M, _meta = choose_k(Ca[mask_a], seed=seed)
    tiles = tile_level(
        Ca, Cb_al, mask_a, mask_b, retained,
        M=int(M), sigma_n=sig_n, times_s=times_s, dt_s=dt_s, seed=seed,
    )
    return {
        "d": d_use,
        "drop_u": drop_u,
        "angles_deg": angles.tolist(),
        "n_stable": n_stable(angles),
        "coord": errs,
        "sigma_noise": sig_n if not light else None,
        "sigma_split": sig_s,
        "tiles": tiles,
        "M": int(M),
    }


def n1_shift_X(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-unit circular shift with offset in [0.25, 0.75] of T."""
    T, n_u = X.shape
    lo, hi = int(0.25 * T), max(int(0.25 * T) + 1, int(0.75 * T))
    shifts = rng.integers(lo, hi, size=n_u)
    out = np.empty_like(X)
    for j, sh in enumerate(shifts):
        out[:, j] = np.roll(X[:, j], int(sh))
    return out
