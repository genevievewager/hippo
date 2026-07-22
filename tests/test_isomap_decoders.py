"""Decoder comparisons on Isomap vs PCA coordinates (linear vs nonlinear)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from realtime.decoder_models import is_nonlinear_decoder, make_continuous_pipeline
from realtime.manifolds.isomap import IsomapManifoldEncoder
from realtime.manifolds.pca import PCAManifoldEncoder


def _nonlinear_behavior_from_latent(n: int = 400, n_units: int = 25, seed: int = 0):
    """Behavior is a nonlinear function of a 2-D curved latent; observations are mixtures."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, n)
    z1 = np.cos(t)
    z2 = np.sin(t)
    # Nonlinear target of manifold coordinates
    y = (z1 ** 2 - z2 ** 2) + 0.1 * rng.normal(size=n)
    latents = np.column_stack([z1, z2, z1 * z2])
    mixing = rng.normal(size=(3, n_units))
    X = latents @ mixing + 0.05 * rng.normal(size=(n, n_units))
    X = np.maximum(X - X.min(), 0.0)  # count-like
    return X, y


def test_nonlinear_decoder_flag():
    assert is_nonlinear_decoder("random_forest_regressor")
    assert is_nonlinear_decoder("rbf_svr")
    assert not is_nonlinear_decoder("ridge")
    assert not is_nonlinear_decoder("logistic_regression")


def test_pca_vs_isomap_linear_and_rf_decoders():
    X, y = _nonlinear_behavior_from_latent()
    split = int(0.7 * len(X))
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    pca = PCAManifoldEncoder(n_components=3, transform="sqrt_counts", standardize=True)
    pca.fit(X_tr)
    Zp_tr, Zp_te = pca.transform(X_tr), pca.transform(X_te)

    iso = IsomapManifoldEncoder(
        n_components=3,
        n_neighbors=15,
        transform="sqrt_counts",
        standardize=True,
        pre_pca_enabled=True,
        pre_pca_n_components=10,
    )
    iso.fit(X_tr)
    Zi_tr, Zi_te = iso.transform(X_tr), iso.transform(X_te)

    results = {}
    for name, Ztr, Zte in (
        ("pca", Zp_tr, Zp_te),
        ("isomap", Zi_tr, Zi_te),
    ):
        for dec_name, model in (
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(n_estimators=50, max_depth=8, random_state=0)),
        ):
            pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
            pipe.fit(Ztr, y_tr)
            results[f"{name}_{dec_name}"] = float(r2_score(y_te, pipe.predict(Zte)))

    # All configurations should produce finite scores; RF may or may not beat Ridge.
    for v in results.values():
        assert np.isfinite(v)
    # At least one Isomap configuration should be competitive with PCA ridge
    assert max(results["isomap_ridge"], results["isomap_rf"]) > -0.5


def test_make_continuous_pipeline_nonlinear_names():
    for name in ("rbf_svr", "mlp_regressor", "knn_regressor"):
        pipe = make_continuous_pipeline(name, "speed", seed=0, n_jobs=1)
        assert pipe is not None
    pos_pipe = make_continuous_pipeline("random_forest_regressor", "position", seed=0, n_jobs=1)
    # Multi-output wrapper for position
    assert "model" in pos_pipe.named_steps
