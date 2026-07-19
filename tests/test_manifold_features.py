"""Tests for manifold feature transformers and train-only fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realtime.manifold_features import (
    GlobalPCAManifold,
    GroupwisePCAManifold,
    IdentityFeatures,
    make_feature_transformer,
)


def test_identity_rates_divide_by_window():
    X = np.ones((5, 3))
    t = IdentityFeatures("rates", decode_window=0.5).fit(X)
    np.testing.assert_allclose(t.transform(X), 2.0)


def test_global_pca_fit_transform_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 20))
    pca = GlobalPCAManifold(n_components=5).fit(X[:70])
    Z = pca.transform(X[70:])
    assert Z.shape == (30, 5)
    assert pca.actual_n_components_ == 5


def test_groupwise_pca_concatenates_groups():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 6))
    labels = ["CA1", "CA1", "CA3", "CA3", "DG", "DG"]
    g = GroupwisePCAManifold(labels, n_components=2).fit(X[:50])
    Z = g.transform(X[50:])
    assert Z.shape[0] == 30
    assert Z.shape[1] == 2 + 2 + 2
    assert g.actual_n_features_ == 6


def test_groupwise_caps_components_by_group_size():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3))
    labels = ["A", "A", "B"]
    g = GroupwisePCAManifold(labels, n_components=10).fit(X)
    assert g.group_n_components_["A"] <= 2
    assert g.group_n_components_["B"] == 1


def test_make_feature_transformer_skips_missing_column():
    units = pd.DataFrame({"unit_id": [0, 1], "region": ["CA1", "CA3"]})
    # layer missing
    out = make_feature_transformer(
        "layer_pca", decode_window=0.25, n_components=2,
        units_df=units, unit_ids=[0, 1],
    )
    assert out is None


def test_pca_does_not_need_test_data_to_fit():
    rng = np.random.default_rng(3)
    X_train = rng.normal(size=(50, 10))
    X_test = rng.normal(size=(20, 10)) + 100  # shifted
    pca = GlobalPCAManifold(n_components=3).fit(X_train)
    # Transform still works; mean is train mean (causality of fit)
    Z = pca.transform(X_test)
    assert Z.shape == (20, 3)
    # Refitting on test would change components; frozen transform is stable
    Z2 = pca.transform(X_test)
    np.testing.assert_allclose(Z, Z2)
