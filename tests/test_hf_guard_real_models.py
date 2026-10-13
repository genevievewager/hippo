"""Fit real pipeline model classes; HF width guard must pass/raise correctly."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.neighbors import KNeighborsRegressor

from analysis.real_quadrant.hf_guard import (
    HFFeatureWidthError,
    assert_model_obs_eq_n_units,
    assert_n_features_eq_n_units,
    model_n_obs,
)
from analysis.real_quadrant.posthoc_smooth import causal_ema
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import _make_rep, fit_transform_representation


N_UNITS = 16
N_T = 120
D = 3


def _synth_X(n_t: int = N_T, n_u: int = N_UNITS, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_t, n_u)).astype(float)


def _fit_rep(name: str, X: np.ndarray):
    cfg = load_quadrant_n5_yaml()
    # Keep LDS/GPFA cheap on tiny data
    if name == "lds":
        cfg = dict(cfg)
        cfg["representations"] = dict(cfg["representations"])
        cfg["representations"]["lds"] = dict(cfg["representations"]["lds"], em_iters=2)
    if name == "gpfa":
        cfg = dict(cfg)
        cfg["representations"] = dict(cfg["representations"])
        cfg["representations"]["gpfa"] = dict(
            cfg["representations"]["gpfa"], max_iter=5, factor_analysis_max_iter=20,
        )
    fit_mask = np.zeros(len(X), dtype=bool)
    fit_mask[: max(40, len(X) // 2)] = True
    model = _make_rep(name, D, cfg, seed=0, n_fit=int(fit_mask.sum()))
    fit_transform_representation(name, model, X, fit_mask)
    return model


@pytest.mark.parametrize(
    "name",
    ["raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"],
)
def test_real_rep_models_obs_width(name: str):
    X = _synth_X()
    assert_n_features_eq_n_units(X, N_UNITS, context=f"{name}.X")
    model = _fit_rep(name, X)
    # Wrapper or inner model must expose obs dim == n_units
    assert model_n_obs(model) == N_UNITS
    assert_model_obs_eq_n_units(model, N_UNITS, context=name)
    with pytest.raises(HFFeatureWidthError, match="!="):
        assert_model_obs_eq_n_units(model, N_UNITS + 1, context=name)
    # Wrong feature matrix at entry
    with pytest.raises(HFFeatureWidthError, match="n_features"):
        assert_n_features_eq_n_units(
            _synth_X(n_u=N_UNITS + 1), N_UNITS, context=f"{name}.badX",
        )


def test_gpfa_causal_uses_same_gpfa_model():
    """gpfa_causal is GPFAModel.transform(causal=True); same obs dim as gpfa."""
    X = _synth_X()
    emb = _fit_rep("gpfa", X)
    inner = emb.model_
    assert model_n_obs(inner) == N_UNITS
    assert_model_obs_eq_n_units(inner, N_UNITS, context="gpfa_causal")
    with pytest.raises(HFFeatureWidthError):
        assert_model_obs_eq_n_units(inner, N_UNITS + 1, context="gpfa_causal")
    # causal transform still requires matching X width
    Z = inner.transform(X, causal=True, reset=True)
    assert Z.shape[0] == len(X)


def test_knn_readout_n_features_in():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(80, D))
    y = rng.normal(size=(80, 2))
    knn = KNeighborsRegressor(n_neighbors=5)
    knn.fit(Z, y)
    assert model_n_obs(knn) == D
    assert_model_obs_eq_n_units(knn, D, context="knn")
    with pytest.raises(HFFeatureWidthError):
        assert_model_obs_eq_n_units(knn, D + 1, context="knn")


def test_smooth_path_guards_feature_entry_not_ema():
    """EMA smoothers are not feature models; guard applies to X at entry."""
    X = _synth_X()
    assert_n_features_eq_n_units(X, N_UNITS, context="smooth.X")
    pred = np.cumsum(np.ones((len(X), 2)), axis=0)
    times = np.arange(len(X), dtype=float) * 0.05
    out = causal_ema(pred, times, tau_s=0.25)
    assert out.shape == pred.shape
    with pytest.raises(HFFeatureWidthError, match="n_features"):
        assert_n_features_eq_n_units(
            _synth_X(n_u=N_UNITS + 1), N_UNITS, context="smooth.badX",
        )
    # EMA has no obs dim → must raise, never return -1
    with pytest.raises(HFFeatureWidthError, match="cannot determine"):
        model_n_obs(object())


def test_unknown_attrs_raise_never_minus_one():
    class Empty:
        pass

    with pytest.raises(HFFeatureWidthError, match="cannot determine"):
        n = model_n_obs(Empty())
        assert n != -1  # pragma: no cover

    class NegSentinel:
        n_features_in_ = -1

    with pytest.raises(HFFeatureWidthError, match="not a valid"):
        model_n_obs(NegSentinel())
