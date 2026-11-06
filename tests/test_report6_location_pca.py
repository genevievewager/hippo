"""Amendment §12 synthetic fixture tests for PCA hard-split levels.

(i) two halves from one low-rank model: angles ~0, e ~0, ARI ~1
(ii) N1 on that fixture: n_stable 0 with u projected out
(iii) S0 on the fixture: e ~0
(iv) Hungarian matching recovers a known permutation of tiles
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from agents.report6.location.pca_core import (
    evaluate_split,
    n1_shift_X,
    rigid_R,
)


def _low_rank_counts(
    T: int = 4000,
    N: int = 40,
    d_true: int = 3,
    seed: int = 0,
    *,
    smooth: bool = True,
) -> np.ndarray:
    """High-SNR positive observations of a shared low-rank latent.

    Uses X = (offset + lat @ L.T)^2 so the pipeline's sqrt step recovers a
    nearly linear low-rank matrix (deterministic; no Poisson), which is what
    (i)/(iii) need to isolate alignment. smooth=False makes latents white in
    time (for the N1 floor fixture).
    """
    rng = np.random.default_rng(seed)
    L, _ = np.linalg.qr(rng.normal(size=(N, d_true)))
    L = L[:, :d_true]
    lat = rng.normal(size=(T, d_true))
    if smooth:
        for _ in range(40):
            lat[1:] = 0.99 * lat[:-1] + 0.01 * rng.normal(size=(T - 1, d_true))
    sig = 4.0 + lat @ L.T
    return np.clip(sig, 0.2, None) ** 2


def _halves_ab(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained = np.ones(T, dtype=bool)
    block = 40
    mask_a = np.zeros(T, dtype=bool)
    mask_b = np.zeros(T, dtype=bool)
    for b0 in range(0, T, block):
        b1 = min(T, b0 + block)
        if (b0 // block) % 2 == 0:
            mask_a[b0:b1] = True
        else:
            mask_b[b0:b1] = True
    return retained, mask_a & retained, mask_b & retained


def _s0_masks(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    retained = np.ones(T, dtype=bool)
    idx = np.arange(T)
    return retained, retained & (idx % 2 == 0), retained & (idx % 2 == 1)


def test_i_shared_model_angles_e_ari():
    """(i) Shared low-rank model → angles near 0, e near 0, ARI near 1."""
    X = _low_rank_counts(seed=1)
    T = X.shape[0]
    retained, ma, mb = _halves_ab(T)
    times = np.arange(T, dtype=float) * 0.05
    ev = evaluate_split(
        X, ma, mb, retained,
        d=3, drop_u=True, times_s=times, dt_s=0.05, seed=0, M=3,
    )
    angles = np.asarray(ev["angles_deg"])
    e_med = ev["coord"]["e_rigid_all"]["median"]
    ari = ev["tiles"]["ari_all"]
    assert np.all(angles < 5.0), angles
    assert e_med < 0.15, e_med
    assert ari > 0.5, ari


def test_ii_n1_n_stable_zero_u_out():
    """(ii) Per-unit shift destroys shared structure → n_stable 0 with u out."""
    X = _low_rank_counts(T=1600, seed=2, smooth=False)
    T = X.shape[0]
    retained, ma, mb = _halves_ab(T)
    times = np.arange(T, dtype=float) * 0.05
    ns = []
    for i in range(20):
        Xs = n1_shift_X(X, np.random.default_rng(1000 + i))
        ev = evaluate_split(
            Xs, ma, mb, retained,
            d=3, drop_u=True, times_s=times, dt_s=0.05, seed=i, M=4, light=True,
        )
        ns.append(ev["n_stable"])
    assert float(np.median(ns)) == 0.0, ns
    # Chance alignments of one axis are allowed; the floor is not "many" stables.
    assert float(np.percentile(ns, 95)) < 2.0, ns


def test_iii_s0_e_near_zero():
    """(iii) Interleaved S0 → rigid e near 0 (catches double-R / wrong apply)."""
    X = _low_rank_counts(T=1600, seed=3)
    T = X.shape[0]
    retained, ma, mb = _s0_masks(T)
    times = np.arange(T, dtype=float) * 0.05
    ev = evaluate_split(
        X, ma, mb, retained,
        d=3, drop_u=True, times_s=times, dt_s=0.05, seed=0, M=4,
    )
    e_med = ev["coord"]["e_rigid_all"]["median"]
    e_aff = ev["coord"]["e_affine_B_median"]
    angles = np.asarray(ev["angles_deg"])
    assert ev["n_stable"] == 3
    assert np.all(angles < 2.0), angles
    assert e_med < 0.05, e_med
    assert e_aff < 0.05, e_aff


def test_iv_hungarian_recovers_permutation():
    """(iv) Hungarian matching recovers a known tile permutation."""
    rng = np.random.default_rng(4)
    M = 5
    mu = rng.normal(size=(M, 2)) * 3.0
    perm = np.array([2, 0, 4, 1, 3])
    mu_b = mu[perm]
    cost = np.linalg.norm(mu[:, None, :] - mu_b[None, :, :], axis=2)
    ra, rb = linear_sum_assignment(cost)
    recovered = np.empty(M, dtype=int)
    for i, j in zip(ra, rb):
        recovered[j] = i
    assert np.array_equal(recovered, perm), (recovered, perm)


def test_alignment_R_definition_and_apply():
    """R = U W^T from SVD(V_b.T V_a); C_b @ R equals R.T @ c_B; reflections ok."""
    rng = np.random.default_rng(5)
    A, _ = np.linalg.qr(rng.normal(size=(20, 3)))
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(Q) > 0:
        Q[:, -1] *= -1
    assert np.linalg.det(Q) < 0
    B = A @ Q
    R = rigid_R(A, B)
    assert np.allclose(B @ R, A, atol=1e-8)
    assert abs(abs(np.linalg.det(R)) - 1.0) < 1e-8
    c_b = rng.normal(size=3)
    assert np.allclose(R.T @ c_b, c_b @ R)
