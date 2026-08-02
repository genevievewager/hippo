"""Tests for public-workflow output contracts and sorted-only hardening."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtime.output_contracts import (
    OutputContractError,
    assert_decode_outputs,
    assert_simulation_outputs,
    validate_simulation_outputs,
)
from realtime.realtime_decoder import RealTimeDecoder
from realtime.train_decoder import TrainedDecoders


class _ConstModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        n = len(X)
        if isinstance(self.value, str):
            return np.array([self.value] * n, dtype=object)
        v = np.asarray(self.value)
        if v.ndim == 0:
            return np.full(n, float(v))
        return np.tile(v.reshape(1, -1), (n, 1))

    def predict_proba(self, X):
        return np.tile([[0.7, 0.3]], (len(X), 1))

    @property
    def classes_(self):
        return np.array(["center", "wall"])

    @property
    def named_steps(self):
        return {"model": self}


def test_validate_simulation_reports_missing(tmp_path: Path):
    missing = validate_simulation_outputs(tmp_path)
    assert "behavior.csv" in missing
    assert "spikes_sorted.csv" in missing


def test_assert_simulation_raises(tmp_path: Path):
    with pytest.raises(OutputContractError, match="Simulation outputs incomplete"):
        assert_simulation_outputs(tmp_path)


def test_assert_decode_raises_without_registry(tmp_path: Path):
    with pytest.raises(OutputContractError, match="Decode/deployment"):
        assert_decode_outputs(tmp_path)


def test_realtime_replay_includes_window_metadata():
    models = TrainedDecoders(
        position=_ConstModel(np.array([1.0, 2.0])),
        speed=_ConstModel(5.0),
        spatial_context=_ConstModel("center"),
        movement_state=_ConstModel("slow"),
        spatial_context_classes=["center", "wall"],
        movement_state_classes=["slow", "fast"],
    )
    models.movement_state.predict_proba = lambda X: np.tile([[0.6, 0.4]], (len(X), 1))

    dec = RealTimeDecoder(
        models=models,
        unit_ids=[0, 1],
        decode_window=0.25,
        update_dt=0.05,
        feature_type="counts",
    )
    spikes = pd.DataFrame({
        "time": [0.10, 0.20, 0.24, 0.30],
        "unit_id": [0, 1, 0, 1],
    })
    beh = pd.DataFrame({
        "time": [0.50],
        "x": [0.0],
        "y": [0.0],
        "spatial_context": ["center"],
        "movement_state": ["slow"],
        "speed": [1.0],
    })
    out = dec.replay(spikes, beh)
    for col in (
        "decode_time", "window_start", "window_end", "decode_window_s",
        "update_dt_s", "n_spikes_in_window", "n_active_units_in_window",
    ):
        assert col in out.columns
    assert out.loc[0, "window_start"] == pytest.approx(0.25)
    assert out.loc[0, "window_end"] == pytest.approx(0.50)
    assert out.loc[0, "update_dt_s"] == pytest.approx(0.05)
    # Spikes in [0.25, 0.50): times 0.30 only
    assert int(out.loc[0, "n_spikes_in_window"]) == 1
