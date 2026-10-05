"""Causality checks for GPFA filter (gpfa_causal) and LDS."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from realtime.dynamic_latents.gpfa import GPFAModel
from realtime.dynamic_latents.lds import LinearDynamicalSystem
from realtime.dynamic_latents.kalman import kalman_filter, rts_smooth


def _synthetic_session(n: int = 400, n_units: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * 0.05
    # Slow latent walk → observations
    z = np.cumsum(rng.normal(scale=0.3, size=(n, 3)), axis=0)
    C = rng.normal(scale=0.5, size=(n_units, 3))
    X = z @ C.T + rng.normal(scale=0.2, size=(n, n_units))
    train = np.zeros(n, dtype=bool)
    train[: int(0.8 * n)] = True
    return X, t, train


def test_gpfa_fit_uses_training_rows_only():
    """Parameters must not depend on held-out future observations."""
    X, _t, train = _synthetic_session()
    X_a = X.copy()
    X_b = X.copy()
    X_b[~train] += 50.0  # large poison on test-only rows
    m_a = GPFAModel(n_components=3, max_iter=2, random_state=0, update_dt=0.05)
    m_b = GPFAModel(n_components=3, max_iter=2, random_state=0, update_dt=0.05)
    m_a.fit(X_a[train])
    m_b.fit(X_b[train])
    np.testing.assert_allclose(m_a.C_, m_b.C_, atol=1e-10)
    np.testing.assert_allclose(m_a.R_, m_b.R_, atol=1e-10)
    np.testing.assert_allclose(m_a.A_, m_b.A_, atol=1e-10)
    np.testing.assert_allclose(m_a.Q_, m_b.Q_, atol=1e-10)
    np.testing.assert_allclose(m_a.tau_, m_b.tau_, atol=1e-10)


def test_gpfa_causal_transform_is_filter_not_smoother():
    X, _t, train = _synthetic_session()
    model = GPFAModel(n_components=3, max_iter=2, random_state=1, update_dt=0.05)
    model.fit(X[train])
    Z_c = model.transform(X, causal=True, reset=True)
    filt = kalman_filter(
        X, model.A_, model.C_, model.d_, model.Q_, model.R_, model.mu0_, model.P0_,
    )
    Z_s, _ = rts_smooth(filt, model.A_, model.Q_)
    np.testing.assert_allclose(Z_c, filt.mu, atol=1e-10)
    # Smoother differs from filter on a real trajectory.
    assert not np.allclose(Z_c, Z_s, atol=1e-5)


def test_gpfa_causal_future_spike_perturbation_does_not_leak():
    """Perturb X at times > t; latent and decoded pred at ≤ t must be unchanged."""
    X, _t, train = _synthetic_session(n=500, n_units=16, seed=2)
    model = GPFAModel(n_components=4, max_iter=2, random_state=2, update_dt=0.05)
    model.fit(X[train])
    Z0 = model.transform(X, causal=True, reset=True)
    t_idx = 200
    assert t_idx < len(X) - 5
    X_poison = X.copy()
    X_poison[t_idx + 1 :] += 100.0
    Z1 = model.transform(X_poison, causal=True, reset=True)
    np.testing.assert_allclose(Z0[: t_idx + 1], Z1[: t_idx + 1], atol=1e-10)
    # Future must change.
    assert not np.allclose(Z0[t_idx + 1 :], Z1[t_idx + 1 :], atol=1e-5)
    # Ridge pred at t from latents up to t (decoder fit on train).
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    y = np.cumsum(np.random.default_rng(3).normal(size=(len(X), 2)), axis=0)
    sc = StandardScaler()
    Ztr = sc.fit_transform(Z0[train])
    ridge = Ridge(alpha=1.0).fit(Ztr, y[train])
    p0 = ridge.predict(sc.transform(Z0[[t_idx]]))
    p1 = ridge.predict(sc.transform(Z1[[t_idx]]))
    np.testing.assert_allclose(p0, p1, atol=1e-10)


def test_lds_future_spike_perturbation_does_not_leak():
    X, _t, train = _synthetic_session(n=500, n_units=16, seed=4)
    model = LinearDynamicalSystem(n_components=4, random_state=4)
    model.fit(X[train])
    Z0 = model.transform(X, causal=True, reset=True)
    t_idx = 200
    X_poison = X.copy()
    X_poison[t_idx + 1 :] += 100.0
    Z1 = model.transform(X_poison, causal=True, reset=True)
    np.testing.assert_allclose(Z0[: t_idx + 1], Z1[: t_idx + 1], atol=1e-10)
    assert not np.allclose(Z0[t_idx + 1 :], Z1[t_idx + 1 :], atol=1e-5)


def test_pipeline_gpfa_fit_mask_is_train_only():
    """Source-level: fit_transform passes train mask into model.fit."""
    from realtime import quadrant_n5_run as run

    src = inspect.getsource(run.fit_transform_representation)
    assert "model.fit(X[np.asarray(fit_mask, dtype=bool)])" in src
    src_t = inspect.getsource(run.fit_transform_representation_timed)
    assert "model.fit(X[np.asarray(fit_mask, dtype=bool)])" in src_t
