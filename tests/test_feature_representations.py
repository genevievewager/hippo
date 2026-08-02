"""Tests for causal spike feature representations (F)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realtime.feature_representations import (
    SpikeFeatureTransformer,
    make_spike_feature_transformer,
    resolve_feature_types,
)


def test_quick_feature_types_default():
    assert resolve_feature_types(None, max_models="quick") == (
        "counts", "rates", "sqrt_counts",
    )


def test_zscore_fit_on_train_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(loc=5.0, scale=2.0, size=(80, 4))
    X_test = rng.normal(loc=50.0, scale=2.0, size=(20, 4))
    t = SpikeFeatureTransformer("zscore_counts").fit(X_train)
    Z = t.transform(X_test)
    # Train mean/std used: test mean should not be ~0
    assert abs(Z.mean()) > 1.0
    Z_train = t.transform(X_train)
    assert abs(Z_train.mean()) < 0.2


def test_rates_divide_by_window():
    X = np.ones((5, 3))
    t = SpikeFeatureTransformer("rates", decode_window=0.5).fit(X)
    np.testing.assert_allclose(t.transform(X), 2.0)


def test_region_normalized_requires_labels():
    X = np.ones((10, 3))
    with pytest.raises(ValueError):
        SpikeFeatureTransformer("region_normalized_counts").fit(X)


def test_region_normalized_grouped():
    X = np.array([
        [1.0, 10.0, 100.0],
        [3.0, 14.0, 120.0],
        [2.0, 12.0, 110.0],
    ])
    labels = ["CA1", "CA1", "CA3"]
    t = SpikeFeatureTransformer(
        "region_normalized_counts", group_labels=labels,
    ).fit(X)
    Z = t.transform(X)
    assert Z.shape == X.shape
    # Within CA1 group, mean over samples of z-scored features ~0
    assert abs(Z[:, :2].mean()) < 1e-6


def test_make_transformer_with_units():
    units = pd.DataFrame({
        "unit_id": [0, 1, 2],
        "region": ["CA1", "CA1", "DG"],
        "cell_type": ["pyr", "pyr", "gc"],
    })
    t = make_spike_feature_transformer(
        "cell_type_normalized_counts",
        decode_window=0.25,
        units_df=units,
        unit_ids=[0, 1, 2],
    )
    X = np.random.default_rng(1).normal(size=(20, 3))
    t.fit(X[:14])
    assert t.transform(X[14:]).shape == (6, 3)
