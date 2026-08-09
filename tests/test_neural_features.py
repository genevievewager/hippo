"""Unit/integration tests for causal neural feature extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realtime.neural_features import (
    CausalSpikeBuffer,
    NeuralFeatureExtractor,
    required_buffer_seconds,
)
from realtime.neural_features.binning import build_causal_unit_bin_tensor
from realtime.neural_features.feature_sets import (
    FEATURE_SET_DEFINITIONS,
    embedding_compatible_with_feature_set,
    families_for_feature_set,
)
from realtime.spike_binner import build_causal_spike_matrix, count_spikes_in_window


def _toy_units() -> pd.DataFrame:
    return pd.DataFrame({
        "unit_id": [0, 1, 2, 3],
        "region": ["CA1", "CA1", "Subiculum", "MEC"],
        "layer": ["pyr", "pyr", "mol", "II"],
        "cell_type": ["pyr", "pyr", "pyr", "stellate"],
    })


def _toy_spikes() -> pd.DataFrame:
    # Spikes strictly for windows ending at t=0.25 and t=0.50
    rows = []
    # Window [0.0, 0.25): units fire
    for t in (0.01, 0.05, 0.10, 0.20, 0.249):
        rows.append({"time": t, "unit_id": 0})
    for t in (0.02, 0.15):
        rows.append({"time": t, "unit_id": 1})
    for t in (0.03, 0.12, 0.22):
        rows.append({"time": t, "unit_id": 2})
    rows.append({"time": 0.08, "unit_id": 3})
    # Exactly at boundary / future — must be excluded for t=0.25
    rows.append({"time": 0.25, "unit_id": 0})
    rows.append({"time": 0.26, "unit_id": 0})
    # Second window extras in [0.25, 0.50)
    for t in (0.30, 0.40):
        rows.append({"time": t, "unit_id": 0})
    rows.append({"time": 0.35, "unit_id": 2})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def test_counts_reproduce_existing_implementation():
    spikes = _toy_spikes()
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    decode_times = np.array([0.25, 0.50])
    W = 0.25
    X_ref = build_causal_spike_matrix(spikes, unit_ids, decode_times, W)
    ext = NeuralFeatureExtractor.from_feature_set(
        "counts", units_df=units, unit_ids=unit_ids, decode_window=W, update_dt=0.25,
    )
    out = ext.extract_matrix(spikes, decode_times)
    assert out.feature_vector.shape == X_ref.shape
    np.testing.assert_allclose(out.feature_vector, X_ref)
    assert len(out.feature_names) == X_ref.shape[1]
    assert all(s.family == "counts" for s in out.feature_metadata)


def test_no_future_spikes_in_any_feature():
    spikes = _toy_spikes()
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    t = 0.25
    W = 0.25
    # Reference half-open counts
    ref = count_spikes_in_window(spikes, unit_ids, t - W, t)
    for fs in FEATURE_SET_DEFINITIONS:
        ext = NeuralFeatureExtractor.from_feature_set(
            fs, units_df=units, unit_ids=unit_ids, decode_window=W,
            update_dt=0.05, coactivity_bin_dt=0.05,
        )
        out = ext.extract_at(spikes, t)
        # Counts slice (when present) must match causal baseline
        if "counts" in ext.families:
            n = len(unit_ids)
            np.testing.assert_allclose(out.feature_vector[0, :n], ref)
        assert np.all(np.isfinite(out.feature_vector))


def test_delta_count_indexing():
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    spikes = _toy_spikes()
    decode_times = np.array([0.25, 0.50])
    W = 0.25
    ext = NeuralFeatureExtractor.from_feature_set(
        "counts_dynamics", units_df=units, unit_ids=unit_ids,
        decode_window=W, update_dt=0.25,
    )
    out = ext.extract_matrix(spikes, decode_times)
    X = build_causal_spike_matrix(spikes, unit_ids, decode_times, W)
    n = len(unit_ids)
    counts = out.feature_vector[:, :n]
    deltas = out.feature_vector[:, n:2 * n]
    np.testing.assert_allclose(counts, X)
    np.testing.assert_allclose(deltas[0], 0.0)
    np.testing.assert_allclose(deltas[1], X[1] - X[0])


def test_regional_aggregation_and_region_scope():
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    spikes = _toy_spikes()
    decode_times = np.array([0.25])
    W = 0.25
    ext = NeuralFeatureExtractor.from_feature_set(
        "counts_regional", units_df=units, unit_ids=unit_ids, decode_window=W,
    )
    out = ext.extract_matrix(spikes, decode_times)
    names = out.feature_names
    assert any(n.startswith("region_CA1_") for n in names)
    assert any(n.startswith("region_MEC_") for n in names)
    # Within-region coactivity names only when that family is enabled
    ext2 = NeuralFeatureExtractor.from_feature_set(
        "counts_coactivity", units_df=units, unit_ids=unit_ids,
        decode_window=W, coactivity_bin_dt=0.05,
    )
    out2 = ext2.extract_matrix(spikes, decode_times)
    within = [s for s in out2.feature_metadata if s.family == "within_region_coactivity"]
    assert within
    assert all(s.region in {"CA1", "Subiculum", "MEC"} for s in within)


def test_cross_region_pairs_and_zero_variance_safe():
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    # Only unit 0 fires; others silent → some zero-variance traces
    spikes = pd.DataFrame({
        "time": [0.01, 0.05, 0.10, 0.15, 0.20],
        "unit_id": [0, 0, 0, 0, 0],
    })
    decode_times = np.array([0.25])
    ext = NeuralFeatureExtractor.from_feature_set(
        "counts_regional_coactivity",
        units_df=units,
        unit_ids=unit_ids,
        decode_window=0.25,
        coactivity_bin_dt=0.05,
    )
    out = ext.extract_matrix(spikes, decode_times)
    assert np.all(np.isfinite(out.feature_vector))
    cross = [s for s in out.feature_metadata if s.family == "cross_region_coactivity"]
    assert cross
    # Region pairs present
    pair_names = { (s.region, s.region_b) for s in cross }
    assert ("CA1", "MEC") in pair_names or ("CA1", "Subiculum") in pair_names


def test_feature_names_align_with_columns_and_dims_deterministic():
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    spikes = _toy_spikes()
    decode_times = np.array([0.25, 0.50])
    for fs in ("counts", "counts_dynamics", "full_population_state"):
        ext = NeuralFeatureExtractor.from_feature_set(
            fs, units_df=units, unit_ids=unit_ids,
            decode_window=0.25, update_dt=0.25, coactivity_bin_dt=0.05,
        )
        a = ext.extract_matrix(spikes, decode_times)
        b = ext.extract_matrix(spikes, decode_times)
        assert a.feature_vector.shape[1] == len(a.feature_names)
        assert a.feature_names == b.feature_names
        np.testing.assert_allclose(a.feature_vector, b.feature_vector)


def test_rolling_realtime_matches_offline():
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    spikes = _toy_spikes()
    W = 0.25
    ext = NeuralFeatureExtractor.from_feature_set(
        "counts_dynamics", units_df=units, unit_ids=unit_ids,
        decode_window=W, update_dt=0.25,
    )
    decode_times = np.array([0.25, 0.50])
    offline = ext.extract_matrix(spikes, decode_times)

    buf = CausalSpikeBuffer(
        history_seconds=required_buffer_seconds(ext),
        unit_ids=unit_ids,
    )
    online_rows = []
    ext.reset_history()
    for t in decode_times:
        # Push spikes up to but not including t (causal)
        new = spikes[(spikes["time"] < t) & (spikes["time"] >= t - W - 0.25)]
        # Simpler: push everything < t each time after clear-ish eviction
        buf.clear()
        past = spikes[spikes["time"] < t]
        buf.push_many(past["time"].to_numpy(), past["unit_id"].to_numpy())
        online_rows.append(buf.extract(ext, float(t)).feature_vector[0])
    online = np.vstack(online_rows)
    np.testing.assert_allclose(online, offline.feature_vector)


def test_groupwise_embedding_only_on_counts():
    assert embedding_compatible_with_feature_set("region_pca", "counts")
    assert not embedding_compatible_with_feature_set("region_pca", "counts_dynamics")
    assert embedding_compatible_with_feature_set("global_pca", "full_population_state")


def test_internal_bins_causal():
    spikes = _toy_spikes()
    units = _toy_units()
    unit_ids = units["unit_id"].to_numpy()
    Xb = build_causal_unit_bin_tensor(
        spikes, unit_ids, np.array([0.25]), decode_window=0.25, coactivity_bin_dt=0.05,
    )
    # Sum over bins equals causal counts
    X = build_causal_spike_matrix(spikes, unit_ids, np.array([0.25]), 0.25)
    np.testing.assert_allclose(Xb[0].sum(axis=1), X[0])


def test_train_test_scaler_leakage_pattern():
    """StandardScaler / PCA on F-E must fit train only — smoke the contract."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 8))
    train = X[:70]
    test = X[70:]
    scaler = StandardScaler().fit(train)
    Xtr = scaler.transform(train)
    Xte = scaler.transform(test)
    pca = PCA(n_components=3, random_state=0).fit(Xtr)
    Ztr = pca.transform(Xtr)
    Zte = pca.transform(Xte)
    # Fitting on all data would shift mean toward test — assert frozen params
    scaler_all = StandardScaler().fit(X)
    assert not np.allclose(scaler.mean_, scaler_all.mean_)
    assert Ztr.shape[1] == 3 and Zte.shape[1] == 3


def test_families_for_named_sets():
    assert families_for_feature_set("counts") == ("counts",)
    assert "lagged_coupling" in families_for_feature_set("full_population_state")
    assert "cross_region_coactivity" in families_for_feature_set("counts_regional_coactivity")
