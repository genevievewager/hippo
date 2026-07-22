"""Tests for realtime-eligible distilled Isomap feature mode."""

from __future__ import annotations

import numpy as np

from realtime.manifold_features import (
    is_realtime_compatible_feature_mode,
    load_feature_transformer,
    make_feature_transformer,
)
from realtime.manifolds.isomap_distilled_features import IsomapDistilledManifold


def _toy_counts(n=180, d=24, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, n)
    z = np.column_stack([np.cos(t), np.sin(t), 0.3 * t])
    mix = rng.normal(size=(3, d))
    X = np.maximum(z @ mix + 0.05 * rng.normal(size=(n, d)), 0.0)
    return X


def test_distilled_feature_mode_is_realtime_eligible_by_default():
    assert is_realtime_compatible_feature_mode("global_isomap") is False
    assert is_realtime_compatible_feature_mode("global_isomap_distilled") is True


def test_isomap_distilled_manifold_fit_transform_save_load(tmp_path):
    X = _toy_counts()
    tr = make_feature_transformer(
        "global_isomap_distilled",
        decode_window=0.25,
        n_components=3,
        n_neighbors=12,
        isomap_pre_pca_n_components=10,
        random_state=0,
    )
    assert isinstance(tr, IsomapDistilledManifold)
    tr.fit(X[:120])
    Z = tr.transform(X[120:130])
    assert Z.shape == (10, 3)
    assert "runtime_per_transform_ms" in tr.get_metadata().get("distillation_metrics", {})
    # Latency gate should pass for ridge on this tiny problem
    assert tr.realtime_compatible is True

    out = tmp_path / "distilled_feat"
    tr.save(out)
    loaded = load_feature_transformer(out)
    np.testing.assert_allclose(
        tr.transform(X[120:130]),
        loaded.transform(X[120:130]),
        rtol=1e-5,
        atol=1e-5,
    )
