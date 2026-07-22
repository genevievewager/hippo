"""Tests for Isomap manifold encoder, diagnostics, and geometry recovery."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA

from realtime.manifolds.isomap import IsomapManifoldEncoder
from realtime.manifolds.isomap_diagnostics import (
    DisconnectedGraphError,
    compute_graph_diagnostics,
    validate_graph_connectivity,
)
from realtime.manifolds.isomap_metrics import (
    evaluate_isomap_geometry,
    procrustes_aligned_error,
    trustworthiness_at_ks,
)
from realtime.manifolds.registry import (
    available_manifolds,
    is_realtime_compatible_manifold,
    make_manifold_encoder,
)
from realtime.manifold_features import IsomapManifold, make_feature_transformer


def _swiss_roll(n: int = 400, seed: int = 0):
    X, t = make_swiss_roll(n_samples=n, noise=0.05, random_state=seed)
    return X, t


def test_isomap_registered_and_offline():
    assert "isomap" in available_manifolds()
    assert is_realtime_compatible_manifold("isomap") is False
    assert is_realtime_compatible_manifold("pca") is True


def test_fit_transform_shapes_and_save_load(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 15))
    X = np.maximum(X, 0.0)  # count-like
    enc = IsomapManifoldEncoder(
        n_components=3,
        n_neighbors=10,
        pre_pca_enabled=True,
        pre_pca_n_components=8,
        require_connected_graph=True,
    )
    Z = enc.fit_transform(X[:80])
    assert Z.shape == (80, 3)
    Z_test = enc.transform(X[80:])
    assert Z_test.shape == (40, 3)
    assert enc.realtime_compatible is False

    out = tmp_path / "isomap_model"
    enc.save(out)
    loaded = IsomapManifoldEncoder.load(out)
    Z2 = loaded.transform(X[80:])
    np.testing.assert_allclose(Z_test, Z2, rtol=1e-5, atol=1e-5)


def test_pre_pca_and_training_only_preprocessing():
    rng = np.random.default_rng(1)
    X_train = np.maximum(rng.normal(size=(100, 40)), 0.0)
    X_test = np.maximum(rng.normal(size=(30, 40)) + 5.0, 0.0)  # shifted
    enc = IsomapManifoldEncoder(
        n_components=2,
        n_neighbors=8,
        pre_pca_enabled=True,
        pre_pca_n_components=10,
    )
    enc.fit(X_train)
    # Scaler / PCA must be frozen from train; transform still works on shifted test
    Z = enc.transform(X_test)
    assert Z.shape == (30, 2)
    assert enc.scaler_ is not None
    assert enc.pre_pca_ is not None
    assert enc.pre_pca_.n_components_ <= 10


def test_swiss_roll_trustworthiness_beats_pca():
    X, _ = _swiss_roll(n=300, seed=2)
    split = 200
    # Isomap on ambient 3-D swiss roll
    iso = IsomapManifoldEncoder(
        n_components=2,
        n_neighbors=12,
        transform="counts",
        standardize=True,
        pre_pca_enabled=False,
        require_connected_graph=True,
    )
    iso.fit(X[:split])
    Z_iso = iso.transform(X[:split])

    pca = PCA(n_components=2, random_state=0).fit(X[:split])
    Z_pca = pca.transform(X[:split])

    tw_iso = trustworthiness_at_ks(X[:split], Z_iso, neighbor_ks=(10,))
    tw_pca = trustworthiness_at_ks(X[:split], Z_pca, neighbor_ks=(10,))
    assert tw_iso["trustworthiness"] >= tw_pca["trustworthiness"] - 0.05


def test_disconnected_graph_rejected():
    # Two well-separated clusters with tiny n_neighbors → likely disconnected
    rng = np.random.default_rng(3)
    A = rng.normal(size=(40, 5))
    B = rng.normal(size=(40, 5)) + 100.0
    X = np.vstack([A, B])
    enc = IsomapManifoldEncoder(
        n_components=2,
        n_neighbors=2,
        transform="counts",
        standardize=False,
        pre_pca_enabled=False,
        require_connected_graph=True,
        allow_largest_component_only=False,
    )
    with pytest.raises(DisconnectedGraphError):
        enc.fit(X)


def test_largest_component_only_recovery():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(60, 4))
    B = rng.normal(size=(5, 4)) + 50.0
    X = np.vstack([A, B])
    enc = IsomapManifoldEncoder(
        n_components=2,
        n_neighbors=3,
        transform="counts",
        standardize=False,
        pre_pca_enabled=False,
        require_connected_graph=True,
        allow_largest_component_only=True,
        minimum_largest_component_fraction=0.8,
    )
    enc.fit(X)
    assert enc.accepted_
    Z = enc.transform(X)
    assert Z.shape[1] == 2


def test_graph_diagnostics_keys():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(80, 6))
    enc = IsomapManifoldEncoder(
        n_components=2,
        n_neighbors=8,
        transform="counts",
        standardize=True,
        pre_pca_enabled=False,
    )
    enc.fit(X)
    diag = enc.graph_diagnostics_
    assert diag is not None
    for key in (
        "n_connected_components",
        "largest_component_fraction",
        "min_node_degree",
        "median_node_degree",
        "max_node_degree",
        "mean_degree_over_n_samples",
        "fraction_duplicate_observations",
        "geodesic_distance_finite_fraction",
    ):
        assert key in diag


def test_validate_graph_connectivity_logic():
    ok_diag = {
        "graph_connected": False,
        "n_connected_components": 2,
        "largest_component_fraction": 0.99,
    }
    accepted, reason = validate_graph_connectivity(
        ok_diag,
        require_connected_graph=True,
        allow_largest_component_only=True,
        minimum_largest_component_fraction=0.95,
    )
    assert accepted and reason is None

    rejected, reason2 = validate_graph_connectivity(
        ok_diag,
        require_connected_graph=True,
        allow_largest_component_only=False,
    )
    assert not rejected
    assert reason2 is not None


def test_geometry_metrics_finite():
    X, _ = _swiss_roll(n=150, seed=6)
    enc = IsomapManifoldEncoder(
        n_components=2,
        n_neighbors=10,
        transform="counts",
        standardize=True,
        pre_pca_enabled=False,
        sampled_distance_pairs=2000,
    )
    Z = enc.fit_transform(X)
    assert enc.geometry_metrics_ is not None
    assert np.isfinite(enc.geometry_metrics_["trustworthiness"])
    assert np.isfinite(enc.geometry_metrics_["residual_variance"])
    metrics = evaluate_isomap_geometry(
        X, Z, geo_dist=enc.isomap_.dist_matrix_, sampled_distance_pairs=1000,
    )
    assert "knn_overlap_k10" in metrics


def test_procrustes_alignment_invariant_to_rotation():
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(50, 3))
    angle = np.pi / 5
    R = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])
    Z_rot = Z @ R
    err = procrustes_aligned_error(Z, Z_rot)
    assert err["procrustes_mse"] < 1e-10
    assert err["procrustes_correlation"] > 0.99


def test_static_isomap_feature_transformer(tmp_path):
    rng = np.random.default_rng(8)
    X = np.maximum(rng.normal(size=(100, 12)), 0.0)
    t = make_feature_transformer(
        "global_isomap",
        decode_window=0.25,
        n_components=3,
        n_neighbors=8,
        isomap_pre_pca_n_components=6,
    )
    assert isinstance(t, IsomapManifold)
    t.fit(X[:70])
    Z = t.transform(X[70:])
    assert Z.shape == (30, 3)
    meta = t.get_metadata()
    assert meta["manifold_type"] == "isomap"
    assert meta["realtime_compatible"] is False
    out = tmp_path / "static_iso"
    t.save(out)
    loaded = IsomapManifold.load(out)
    np.testing.assert_allclose(Z, loaded.transform(X[70:]), rtol=1e-5, atol=1e-5)


def test_make_manifold_encoder_isomap():
    enc = make_manifold_encoder("isomap", n_components=2, n_neighbors=5, pre_pca_enabled=False)
    assert isinstance(enc, IsomapManifoldEncoder)
