"""Temporal decoding controls on Isomap coordinates (static vs history)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from realtime.manifolds.isomap import IsomapManifoldEncoder
from realtime.temporal.flattened import (
    average_latent_history,
    flatten_latent_sequences,
    shuffle_sequence_order,
)
from realtime.temporal.sequences import build_causal_latent_sequences


def _ring_with_oscillating_velocity(n: int = 500, n_units: int = 24, seed: int = 0):
    """Ring manifold; signed speed oscillates for the whole session."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    omega = np.sin(0.05 * t)  # varies on train and test
    theta = np.cumsum(0.15 * omega)
    z = np.column_stack([np.cos(theta), np.sin(theta)])
    # Also expose finite differences in the latent mixing so history helps
    dz = np.gradient(z, axis=0)
    latents = np.column_stack([z, dz])
    mixing = rng.normal(size=(4, n_units))
    X = latents @ mixing + 0.01 * rng.normal(size=(n, n_units))
    X = np.maximum(X - X.min() + 0.01, 0.0)
    return X, omega.astype(float)


def test_isomap_history_beats_shuffled_for_velocity():
    X, y_vel = _ring_with_oscillating_velocity()
    split = int(0.7 * len(X))
    enc = IsomapManifoldEncoder(
        n_components=3,
        n_neighbors=20,
        transform="sqrt_counts",
        standardize=True,
        pre_pca_enabled=False,
        require_connected_graph=True,
    )
    enc.fit(X[:split])
    Z = enc.transform(X)

    seq5, idx5 = build_causal_latent_sequences(Z, 5)
    flat5 = flatten_latent_sequences(seq5)
    train5 = idx5 < split
    test5 = idx5 >= split
    assert train5.any() and test5.any()

    r2_hist = r2_score(
        y_vel[idx5[test5]],
        Ridge().fit(flat5[train5], y_vel[idx5[train5]]).predict(flat5[test5]),
    )

    rng = np.random.default_rng(1)
    flat_shuf = flatten_latent_sequences(shuffle_sequence_order(seq5, rng=rng))
    r2_shuf = r2_score(
        y_vel[idx5[test5]],
        Ridge().fit(flat_shuf[train5], y_vel[idx5[train5]]).predict(flat_shuf[test5]),
    )

    flat_avg = average_latent_history(seq5)
    r2_avg = r2_score(
        y_vel[idx5[test5]],
        Ridge().fit(flat_avg[train5], y_vel[idx5[train5]]).predict(flat_avg[test5]),
    )

    seq1, idx1 = build_causal_latent_sequences(Z, 1)
    flat1 = flatten_latent_sequences(seq1)
    train1 = idx1 < split
    test1 = idx1 >= split
    r2_static = r2_score(
        y_vel[idx1[test1]],
        Ridge().fit(flat1[train1], y_vel[idx1[train1]]).predict(flat1[test1]),
    )

    # Ordered history should beat order-destroying / order-agnostic controls
    assert r2_hist > r2_shuf
    assert r2_hist >= r2_avg - 1e-6
    assert r2_hist > r2_static
