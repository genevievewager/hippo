"""Synthetic dataset verifying distinct roles of W (integration) and L (history)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from realtime.manifolds.pca import PCAManifoldEncoder
from realtime.temporal.flattened import (
    flatten_latent_sequences,
    shuffle_sequence_order,
)
from realtime.temporal.sequences import build_causal_latent_sequences


def _make_synthetic(n_times: int = 400, n_units: int = 20, seed: int = 0):
    """
    Current latent state encodes position; temporal direction encodes velocity.

    Shuffling the latent sequence destroys velocity information.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_times)
    position = np.sin(0.05 * t)
    velocity = np.gradient(position)
    # Latent-like neural state: first dims carry position, deltas carry velocity
    z_true = np.zeros((n_times, 4))
    z_true[:, 0] = position
    z_true[:, 1] = position
    z_true[:, 2] = velocity
    z_true[:, 3] = velocity
    # Observations are noisy linear mixtures
    mixing = rng.normal(size=(4, n_units))
    X = z_true @ mixing + 0.05 * rng.normal(size=(n_times, n_units))
    y_pos = position
    y_vel = velocity
    return X, y_pos, y_vel


def test_current_state_decodes_position_history_decodes_velocity():
    X, y_pos, y_vel = _make_synthetic()
    n = len(X)
    split = int(0.7 * n)
    enc = PCAManifoldEncoder(n_components=4, transform="counts", standardize=True)
    enc.fit(X[:split])
    Z = enc.transform(X)

    # L=1: position OK, velocity weak
    seq1, idx1 = build_causal_latent_sequences(Z, 1)
    flat1 = flatten_latent_sequences(seq1)
    train = idx1 < split
    test = idx1 >= split
    model_pos = Ridge().fit(flat1[train], y_pos[idx1[train]])
    r2_pos_L1 = r2_score(y_pos[idx1[test]], model_pos.predict(flat1[test]))

    model_vel = Ridge().fit(flat1[train], y_vel[idx1[train]])
    r2_vel_L1 = r2_score(y_vel[idx1[test]], model_vel.predict(flat1[test]))

    # L=5 ordered: velocity should improve
    seq5, idx5 = build_causal_latent_sequences(Z, 5)
    flat5 = flatten_latent_sequences(seq5)
    train5 = idx5 < split
    test5 = idx5 >= split
    model_vel5 = Ridge().fit(flat5[train5], y_vel[idx5[train5]])
    r2_vel_L5 = r2_score(y_vel[idx5[test5]], model_vel5.predict(flat5[test5]))

    rng = np.random.default_rng(1)
    flat5_shuf = flatten_latent_sequences(shuffle_sequence_order(seq5, rng=rng))
    model_shuf = Ridge().fit(flat5_shuf[train5], y_vel[idx5[train5]])
    r2_vel_shuf = r2_score(y_vel[idx5[test5]], model_shuf.predict(flat5_shuf[test5]))

    assert r2_pos_L1 > 0.5
    assert r2_vel_L5 > r2_vel_L1
    assert r2_vel_L5 > r2_vel_shuf


def test_longer_integration_smoothes_noisy_counts():
    """Larger W (simulated by averaging neighboring frames) improves SNR then oversmooths."""
    rng = np.random.default_rng(2)
    t = np.arange(300)
    signal = np.sin(0.1 * t)
    noise = rng.normal(scale=1.0, size=t.shape)
    obs = signal + noise

    def window_r2(W: int) -> float:
        x = np.convolve(obs, np.ones(W) / W, mode="same").reshape(-1, 1)
        split = 200
        model = Ridge().fit(x[:split], signal[:split])
        return float(r2_score(signal[split:], model.predict(x[split:])))

    r2_w1 = window_r2(1)
    r2_w5 = window_r2(5)
    r2_w50 = window_r2(50)
    assert r2_w5 > r2_w1
    # Extreme smoothing eventually hurts high-frequency signal
    assert r2_w5 >= r2_w50 or r2_w5 > 0.3
