"""Causal spike windows must exclude spikes at/after the prediction time."""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime.spike_binner import build_causal_spike_matrix, count_spikes_in_window


def test_count_spikes_half_open_excludes_stop_time():
    spikes = pd.DataFrame({
        "time": [0.10, 0.149999, 0.15, 0.16],
        "unit_id": [0, 0, 0, 0],
    })
    counts = count_spikes_in_window(spikes, unit_ids=[0], t_start=0.10, t_end=0.15)
    # Includes 0.10 and 0.149999, excludes 0.15 and 0.16
    assert counts[0] == 2


def test_build_matrix_no_future_spikes():
    spikes = pd.DataFrame({
        "time": np.array([0.0, 0.049, 0.05, 0.051, 0.10]),
        "unit_id": np.zeros(5, dtype=int),
    })
    decode_times = np.array([0.05, 0.10])
    X = build_causal_spike_matrix(spikes, [0], decode_times, decode_window=0.05)
    # At t=0.05, window [0.0, 0.05) -> spikes 0.0 and 0.049 (0.05 excluded)
    assert X[0, 0] == 2
    # At t=0.10, window [0.05, 0.10) -> spikes 0.05 and 0.051 (0.10 excluded)
    assert X[1, 0] == 2
