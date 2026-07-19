"""Spike-time sorting for causal window counts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from realtime.data_loading import load_simulation_data
from realtime.spike_binner import count_spikes_in_window


def test_load_simulation_data_sorts_unsorted_ground_truth_spikes(tmp_path: Path):
    behavior = pd.DataFrame(
        {
            "time": [0.0, 0.05, 0.10],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "speed": [0.0, 1.0, 1.0],
            "head_direction": [0.0, 0.1, 0.2],
        }
    )
    units = pd.DataFrame({"unit_id": [1, 2]})
    # Deliberately unsorted spike times (as in spikes_ground_truth.csv exports).
    spikes = pd.DataFrame(
        {
            "unit_id": [1, 2, 1, 2],
            "spike_time_s": [0.08, 0.02, 0.03, 0.09],
        }
    )
    behavior.to_csv(tmp_path / "behavior.csv", index=False)
    units.to_csv(tmp_path / "units.csv", index=False)
    spikes.to_csv(tmp_path / "spikes_ground_truth.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps({"session_duration_s": 0.1}))

    data = load_simulation_data(tmp_path, spike_source="ground_truth")
    times = data["spikes_df"]["time"].to_numpy()
    assert np.all(times[:-1] <= times[1:])

    counts = count_spikes_in_window(data["spikes_df"], [1, 2], 0.0, 0.05)
    assert counts.tolist() == [1.0, 1.0]  # spikes at 0.02 and 0.03
