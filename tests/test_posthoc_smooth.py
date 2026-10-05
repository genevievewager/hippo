"""Unit tests for causal EMA post-hoc smoothing."""

from __future__ import annotations

import numpy as np

from analysis.real_quadrant.posthoc_smooth import causal_ema, select_ema_tau


def test_causal_ema_is_causal_and_constant_signal_fixed():
    t = np.arange(0, 5, 0.05)
    pred = np.column_stack([np.ones(len(t)), np.full(len(t), 2.0)])
    sm = causal_ema(pred, t, tau_s=0.5)
    np.testing.assert_allclose(sm, pred)
    # Changing a future point must not affect the past.
    pred2 = pred.copy()
    pred2[-1] = 100.0
    sm2 = causal_ema(pred2, t, tau_s=0.5)
    np.testing.assert_allclose(sm2[:-1], sm[:-1])


def test_select_ema_tau_returns_grid_value():
    rng = np.random.default_rng(0)
    n = 600
    t = np.arange(n) * 0.05
    y = np.cumsum(rng.normal(size=(n, 2)), axis=0)
    Z = y + rng.normal(scale=0.5, size=y.shape)
    train = np.zeros(n, dtype=bool)
    train[:480] = True
    tau, med = select_ema_tau(Z, y, t, train, ridge_alpha=1.0, n_blocks=4, gap_s=1.0)
    assert tau in (0.1, 0.25, 0.5, 1.0, 2.0)
    assert np.isfinite(med)
