"""raw_lag history control: causal stack of current + previous 5 steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from realtime.manifold_features import LaggedRawFeatures, make_feature_transformer
from realtime.representation_registry import QUADRANT_UNKNOWN, get_spec


def test_raw_lag_registry_is_control_not_quadrant():
    spec = get_spec("raw_lag")
    assert spec.implemented is True
    assert spec.quadrant == QUADRANT_UNKNOWN
    assert spec.realtime_capable is True
    assert spec.offline_only is False


def test_raw_lag_stacks_current_and_five_previous():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    tf = LaggedRawFeatures(n_lags=5).fit(X)
    Z = tf.transform(X)
    assert Z.shape == (20, 4 * 6)
    np.testing.assert_allclose(Z[10, :4], X[10])
    np.testing.assert_allclose(Z[10, 4:8], X[9])
    np.testing.assert_allclose(Z[10, 20:24], X[5])
    assert tf.valid_mask(20).sum() == 15
    assert not tf.valid_mask(20)[:5].any()


def test_raw_lag_is_causal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 3))
    tf = LaggedRawFeatures(n_lags=5).fit(X)
    Z = tf.transform(X)
    X2 = X.copy()
    X2[12:] += 10.0
    Z2 = tf.transform(X2)
    np.testing.assert_allclose(Z[:12], Z2[:12])


def test_raw_lag_transform_one_matches_batch_after_warmup(tmp_path: Path):
    rng = np.random.default_rng(2)
    X = rng.normal(size=(12, 5))
    tf = make_feature_transformer("raw_lag", decode_window=0.250, n_lags=5)
    tf.fit(X)
    Z = tf.transform(X)
    tf.reset_state()
    rows = [tf.transform_one(X[i]) for i in range(len(X))]
    got = np.vstack(rows)
    np.testing.assert_allclose(got[5:], Z[5:])
    tf.save(tmp_path / "raw_lag")
    loaded = LaggedRawFeatures.load(tmp_path / "raw_lag")
    assert loaded.n_lags == 5
    np.testing.assert_allclose(loaded.transform(X)[5:], Z[5:])
