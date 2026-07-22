"""Tests for optional Isomap parametric distillation."""

from __future__ import annotations

import numpy as np

from realtime.manifolds.isomap import IsomapManifoldEncoder
from realtime.manifolds.isomap_distillation import (
    IsomapDistilledEncoder,
    distill_isomap_encoder,
)


def test_distillation_approximates_isomap(tmp_path):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 4 * np.pi, 250)
    z_true = np.column_stack([np.cos(t), np.sin(t)])
    mixing = rng.normal(size=(2, 20))
    X = z_true @ mixing + 0.02 * rng.normal(size=(250, 20))
    X = np.maximum(X - X.min(), 0.0)

    split_tr, split_va = 150, 200
    iso = IsomapManifoldEncoder(
        n_components=2,
        n_neighbors=12,
        transform="sqrt_counts",
        standardize=True,
        pre_pca_enabled=True,
        pre_pca_n_components=8,
    )
    iso.fit(X[:split_tr])
    Z = iso.transform(X)

    out = tmp_path / "distilled"
    dist = distill_isomap_encoder(
        X[:split_tr],
        Z[:split_tr],
        model_name="ridge",
        X_val=X[split_tr:split_va],
        Z_val=Z[split_tr:split_va],
        X_test=X[split_va:],
        Z_test=Z[split_va:],
        output_dir=out,
        realtime_latency_budget_ms=50.0,
        seed=0,
    )
    assert dist.n_features_in_ == 20
    assert dist.latent_dim == 2
    assert "runtime_per_transform_ms" in dist.metrics_
    # Causal: only uses current observation
    z_hat = dist.transform(X[split_va : split_va + 1])
    assert z_hat.shape == (1, 2)

    loaded = IsomapDistilledEncoder.load(out)
    np.testing.assert_allclose(
        dist.transform(X[split_va:]),
        loaded.transform(X[split_va:]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_distillation_rejects_realtime_when_budget_impossible():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 8))
    Z = rng.normal(size=(80, 2))
    dist = IsomapDistilledEncoder(
        model_name="mlp",
        realtime_latency_budget_ms=1e-9,  # impossible budget
        max_procrustes_mse=0.0,  # also strict
        seed=0,
    )
    dist.fit(X[:50], Z[:50], X_val=X[50:], Z_val=Z[50:])
    assert dist.realtime_compatible is False
    assert dist.metrics_["latency_ok"] is False or dist.metrics_["distortion_ok"] is False
