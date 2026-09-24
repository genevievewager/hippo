"""Phase 1 correctness-gate regressions.

Each test encodes a contract that used to fail silently. They must fail on
the pre-fix implementations and pass now.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtime.live.spike_buffer import CausalSpikeBuffer, resolve_spike_columns
from realtime.live.spike_stream import ReplaySpikeStream
from realtime.pipeline_invariants import PipelineInvariantError, assert_times_in_seconds
from realtime.spike_binner import count_spikes_in_window


def _brute_counts(times, units, unit_ids, t_start, t_end):
    out = np.zeros(len(unit_ids), dtype=float)
    idx = {int(u): i for i, u in enumerate(unit_ids)}
    for t, u in zip(times, units):
        if t_start <= float(t) < t_end:
            j = idx.get(int(u))
            if j is not None:
                out[j] += 1.0
    return out


def test_buffer_shuffled_arrival_matches_sorted():
    unit_ids = [1, 2]
    times = np.array([0.10, 0.25, 0.40, 0.55, 0.70, 0.85])
    units = np.array([1, 2, 1, 2, 1, 2])
    rng = np.random.default_rng(0)
    order = rng.permutation(len(times))

    sorted_buf = CausalSpikeBuffer(unit_ids, history_s=10.0)
    sorted_buf.extend(times, units)

    shuffled_buf = CausalSpikeBuffer(unit_ids, history_s=10.0)
    shuffled_buf.extend(times[order], units[order])

    for t in (0.30, 0.60, 0.90):
        np.testing.assert_array_equal(
            shuffled_buf.counts_at(t, 0.50),
            sorted_buf.counts_at(t, 0.50),
        )


def test_buffer_one_nan_matches_clean_stream():
    unit_ids = [1]
    clean_times = np.array([0.10, 0.20, 0.30, 0.40])
    dirty_times = np.array([0.10, np.nan, 0.20, 0.30, 0.40])
    clean = CausalSpikeBuffer(unit_ids, history_s=10.0)
    dirty = CausalSpikeBuffer(unit_ids, history_s=10.0)
    clean.extend(clean_times, np.ones(len(clean_times), dtype=int))
    dirty.extend(dirty_times, np.ones(len(dirty_times), dtype=int))

    assert dirty.n_rejected_nonfinite == 1
    assert dirty.n_spikes == clean.n_spikes
    np.testing.assert_array_equal(dirty.counts_at(0.45, 0.40), clean.counts_at(0.45, 0.40))


def test_count_spikes_in_window_unsorted_raises():
    spikes = pd.DataFrame({"time": [0.9, 0.1, 0.5], "unit_id": [1, 1, 1]})
    with pytest.raises(ValueError, match="sorted ascending"):
        count_spikes_in_window(spikes, [1], 0.0, 1.0)


def test_count_spikes_in_window_sorted_matches_brute_force():
    times = np.array([0.05, 0.15, 0.25, 0.35])
    units = np.array([1, 2, 1, 1])
    spikes = pd.DataFrame({"time": times, "unit_id": units})
    got = count_spikes_in_window(spikes, [1, 2], 0.10, 0.30)
    np.testing.assert_array_equal(got, _brute_counts(times, units, [1, 2], 0.10, 0.30))


@pytest.mark.parametrize(
    "time_col,unit_col",
    [
        ("times", "cluster_id"),
        ("times", "clusters"),
        ("spike_time_s", "cluster_id"),
        ("spike_time_s", "clusters"),
        ("time", "unit_id"),
    ],
)
def test_replay_stream_phy_column_variants_load(time_col, unit_col):
    df = pd.DataFrame({time_col: [0.1, 0.2, 0.3], unit_col: [7, 7, 8]})
    stream = ReplaySpikeStream(spikes_df=df)
    stream.connect()
    assert stream.connected
    assert sorted(stream.list_unit_ids()) == [7, 8]
    chunk = stream.get_new_spikes(up_to_time=0.25)
    assert list(chunk.columns) == ["time", "unit_id"]
    assert len(chunk) == 2


def test_shared_alias_table_used_by_binner_stream_and_buffer():
    from realtime.live import spike_stream as stream_mod
    from realtime.spike_binner import _resolve_spike_columns

    df = pd.DataFrame({"spike_time_s": [0.1], "clusters": [3]})
    assert _resolve_spike_columns(df) == resolve_spike_columns(df)
    mapping = stream_mod.ReplaySpikeStream._column_map(df)
    assert mapping["spike_time_s"] == "time"
    assert mapping["clusters"] == "unit_id"


def test_load_simulation_data_rejects_sample_index_spike_times(tmp_path: Path):
    from realtime.data_loading import load_simulation_data

    behavior = pd.DataFrame(
        {
            "time": [0.0, 0.05, 0.10],
            "x": [1.0, 2.0, 3.0],
            "y": [1.0, 2.0, 3.0],
            "speed": [0.0, 1.0, 1.0],
            "head_direction": [0.0, 0.1, 0.2],
        }
    )
    units = pd.DataFrame(
        {
            "unit_id": [1, 2],
            "region": ["CA1", "CA1"],
            "cell_type": ["CA1_pyr", "CA1_pyr"],
        }
    )
    # 30 kHz sample indices for a 1 s session, written as if they were seconds.
    spikes = pd.DataFrame(
        {"unit_id": [1, 2, 1], "spike_time_s": [0, 15000, 30000]}
    )
    behavior.to_csv(tmp_path / "behavior.csv", index=False)
    units.to_csv(tmp_path / "units.csv", index=False)
    spikes.to_csv(tmp_path / "spikes_ground_truth.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps({"session_duration_s": 1.0}))

    with pytest.raises(PipelineInvariantError, match="not in seconds"):
        load_simulation_data(tmp_path, spike_source="ground_truth")


def test_replay_stream_rejects_integer_sample_indices():
    times = np.arange(0, 30_000, 100, dtype=np.int64)
    df = pd.DataFrame({"time": times, "unit_id": np.ones(len(times), dtype=int)})
    stream = ReplaySpikeStream(spikes_df=df)
    with pytest.raises(PipelineInvariantError, match="integer dtype"):
        stream.connect()


def test_assert_times_in_seconds_accepts_short_session():
    assert_times_in_seconds(np.array([0.0, 0.05, 0.10]), session_length_s=0.1)


def test_latency_budget_cli_default_is_50ms():
    from agents.pressure_test.findings import RunContext
    from agents.pressure_test.runner import build_parser

    args = build_parser().parse_args([])
    assert args.latency_budget_ms == pytest.approx(50.0)
    assert RunContext(repo_root=".").latency_budget_ms == pytest.approx(50.0)
