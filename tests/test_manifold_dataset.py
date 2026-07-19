"""Tests for shared ManifoldDataset loader (Phase 1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hippo.dataset import load_manifold_dataset
from realtime.spike_features import build_causal_spike_matrix
from realtime.timing import DEFAULT_BEHAVIOR_SAMPLING_RATE_HZ


SESSION = Path("outputs/ratinabox_003")


@pytest.mark.skipif(not SESSION.exists(), reason="requires outputs/ratinabox_003")
def test_load_manifold_dataset_shapes_and_timing():
    ds = load_manifold_dataset(
        SESSION,
        spike_source="sorted",
        integration_window_s=0.250,
        activity_representation="counts",
    )
    assert ds.activity.ndim == 2
    assert ds.activity.shape[0] == len(ds.timestamps_s)
    assert ds.activity.shape[1] == len(ds.unit_ids)
    assert len(ds.behavior) == ds.n_times
    assert abs(ds.update_interval_s - 1.0 / DEFAULT_BEHAVIOR_SAMPLING_RATE_HZ) < 0.01
    assert ds.train_mask is not None and ds.test_mask is not None
    assert ds.train_mask.sum() + ds.test_mask.sum() == ds.n_times
    assert set(ds.unit_metadata["unit_id"]) == set(ds.unit_ids)


@pytest.mark.skipif(not SESSION.exists(), reason="requires outputs/ratinabox_003")
def test_no_future_spikes_in_window():
    from realtime.data_loading import load_simulation_data
    from realtime.data_loading import make_decode_times
    from realtime.timing import extract_behavior_times, resolve_update_dt_s

    data = load_simulation_data(SESSION, "sorted")
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(data["summary"], behavior_times=behavior_times)
    W = 0.250
    decode_times = make_decode_times(
        data["session_duration"], W, update_dt, behavior_times=behavior_times,
    )
    X = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, W,
    )
    # Spot-check: for a mid-session time, no spike in the matrix window is >= t
    spikes = data["spikes_df"]
    for cand in ("time", "spike_time_s", "time_s"):
        if cand in spikes.columns:
            tcol = cand
            break
    else:
        raise AssertionError(f"No spike time column in {list(spikes.columns)}")
    t = float(decode_times[len(decode_times) // 2])
    window_spikes = spikes[(spikes[tcol] >= t - W) & (spikes[tcol] < t)]
    assert (window_spikes[tcol] < t).all()
    # Matrix row must be finite nonnegative counts
    row = X[len(decode_times) // 2]
    assert np.all(row >= 0)
    assert np.all(np.isfinite(row))


@pytest.mark.skipif(not SESSION.exists(), reason="requires outputs/ratinabox_003")
def test_subset_units_preserves_times():
    ds = load_manifold_dataset(SESSION, spike_source="sorted", integration_window_s=0.1)
    keep = ds.unit_ids[: max(5, len(ds.unit_ids) // 4)]
    sub = ds.subset_units(keep)
    assert sub.n_times == ds.n_times
    assert sub.n_units == len(keep)
    assert np.array_equal(sub.timestamps_s, ds.timestamps_s)
