"""Tests for causal latent sequences, lags, and controls."""

from __future__ import annotations

import numpy as np
import pytest

from realtime.temporal.flattened import (
    average_latent_history,
    flatten_latent_sequences,
    shuffle_sequence_order,
)
from realtime.temporal.sequences import (
    apply_prediction_lag,
    build_causal_latent_sequences,
    lag_seconds_to_frames,
)
from realtime.temporal.splits import (
    causal_train_val_test_split,
    mask_sequences_within_split,
    required_split_gap_s,
)


def test_sequence_length_matches_history_frames():
    Z = np.arange(20 * 3, dtype=float).reshape(20, 3)
    for L in (1, 2, 5, 10, 20):
        seq, idx = build_causal_latent_sequences(Z, L, pad_mode="drop")
        assert seq.shape == (20 - L + 1, L, 3)
        assert len(idx) == 20 - L + 1
        # Last frame of each sequence is the target time state
        for i, t in enumerate(idx):
            np.testing.assert_allclose(seq[i, -1], Z[t])
            np.testing.assert_allclose(seq[i, 0], Z[t - L + 1])


def test_sequences_never_use_future_latents():
    Z = np.arange(10, dtype=float).reshape(10, 1)
    seq, idx = build_causal_latent_sequences(Z, 3, pad_mode="drop")
    for i, t in enumerate(idx):
        assert seq[i].max() <= Z[t]


def test_prediction_lag_indexing():
    end = np.array([2, 3, 4, 5])
    end2, tgt = apply_prediction_lag(end, lag_frames=2, n_times=7)
    np.testing.assert_array_equal(tgt, end2 + 2)
    assert tgt.max() < 7


def test_negative_lag_rejected():
    with pytest.raises(ValueError):
        lag_seconds_to_frames(-0.05, 0.05)


def test_shuffled_control_preserves_states_changes_order():
    rng = np.random.default_rng(0)
    seq = np.arange(5 * 4 * 2, dtype=float).reshape(5, 4, 2)
    shuffled = shuffle_sequence_order(seq, rng=rng)
    for i in range(5):
        orig_sorted = np.sort(seq[i].reshape(-1))
        shuf_sorted = np.sort(shuffled[i].reshape(-1))
        np.testing.assert_allclose(orig_sorted, shuf_sorted)
    # With L>1, order should differ for at least one sequence almost surely
    assert not np.allclose(seq, shuffled)


def test_averaged_history_uses_only_sequence_states():
    seq = np.ones((3, 4, 2))
    seq[:, 0, :] = 0.0
    avg = average_latent_history(seq)
    np.testing.assert_allclose(avg, 0.75)


def test_split_gap_and_no_cross_boundary_sequences():
    # Long enough session so gap_s = W_max + L_max + tau does not empty val
    t = np.arange(0.0, 60.0, 0.05)
    gap = required_split_gap_s(1.0, 1.0, 0.25)
    train, val, test = causal_train_val_test_split(t, train_frac=0.6, val_frac=0.15, gap_s=gap)
    assert train.any() and val.any() and test.any()
    assert not np.any(train & val)
    assert not np.any(val & test)

    # Build dummy end indices in val region and ensure history stays in val
    end_idx = np.where(val)[0][10:20]
    keep = mask_sequences_within_split(end_idx, history_frames=5, split_mask=val)
    for t_idx, ok in zip(end_idx, keep):
        if ok:
            assert val[t_idx - 4 : t_idx + 1].all()


def test_flatten_shape():
    seq = np.zeros((6, 5, 3))
    flat = flatten_latent_sequences(seq)
    assert flat.shape == (6, 15)
