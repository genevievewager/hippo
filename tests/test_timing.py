"""Tests for 20 Hz timing and behavior-aligned decoder updates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from realtime.data_loading import make_decode_times
from realtime.timing import (
    DEFAULT_UPDATE_DT_S,
    assert_alignment,
    extract_behavior_times,
    make_behavior_aligned_decode_times,
    resolve_update_dt_s,
    validate_behavior_timestamps,
)


def test_default_update_dt_is_50ms():
    assert DEFAULT_UPDATE_DT_S == pytest.approx(0.050)


def test_resolve_update_dt_from_summary_behavior_dt():
    dt = resolve_update_dt_s({"behavior_dt": 0.05}, derive_from_behavior=True)
    assert dt == pytest.approx(0.05)


def test_resolve_update_dt_from_behavior_rate():
    dt = resolve_update_dt_s(
        None, behavior_sampling_rate_hz=20.0, derive_from_behavior=True,
    )
    assert dt == pytest.approx(0.05)


def test_behavior_timestamps_validate_at_20hz():
    t = np.arange(0.0, 1.0, 0.05)
    result = validate_behavior_timestamps(t, expected_dt_s=0.05)
    assert result.ok
    assert result.median_dt_s == pytest.approx(0.05)


def test_make_behavior_aligned_decode_times_matches_frames():
    t = np.arange(0.0, 2.0, 0.05)
    decode, _ = make_behavior_aligned_decode_times(t, integration_window_s=0.25)
    assert np.all(decode >= 0.25 - 1e-12)
    assert_alignment(decode, t, alignment_tolerance_s=0.005)
    # One prediction per behavioral frame
    assert len(decode) == len(t[t >= 0.25 - 1e-12])


def test_make_decode_times_prefers_behavior_times():
    beh = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    times = make_decode_times(1.0, 0.20, 0.05, behavior_times=beh)
    np.testing.assert_allclose(times, [0.20, 0.25, 0.30])


def test_extract_behavior_times_time_s():
    df = pd.DataFrame({"time_s": [0.0, 0.05], "x_cm": [1.0, 2.0]})
    np.testing.assert_allclose(extract_behavior_times(df), [0.0, 0.05])
