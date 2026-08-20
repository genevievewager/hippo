"""Tests for global active dataset / spike-source session helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ui import state


class _FakeSession(dict):
    """Minimal stand-in for st.session_state in unit tests."""

    def get(self, key, default=None):
        return super().get(key, default)


@pytest.fixture
def session(monkeypatch):
    store = _FakeSession({state.KEY_SPIKE_SOURCE: "ground_truth"})
    monkeypatch.setattr(state.st, "session_state", store)
    return store


def test_resolve_spike_source_clamps_invalid_value(session, tmp_path: Path):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "spikes_sorted.csv").write_text("time,unit_id\n")
    resolved = state.resolve_spike_source(exp, sources=["sorted"])
    assert resolved == "sorted"
    assert session[state.KEY_SPIKE_SOURCE] == "sorted"


def test_resolve_spike_source_keeps_valid_value(session, tmp_path: Path):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "spikes_sorted.csv").write_text("time,unit_id\n")
    (exp / "spikes_ground_truth.csv").write_text("time,unit_id\n")
    session[state.KEY_SPIKE_SOURCE] = "ground_truth"
    resolved = state.resolve_spike_source(exp)
    assert resolved == "ground_truth"


def test_resolve_spike_source_defaults_when_no_files(session, tmp_path: Path):
    exp = tmp_path / "empty"
    exp.mkdir()
    session.pop(state.KEY_SPIKE_SOURCE, None)
    resolved = state.resolve_spike_source(exp)
    assert resolved == "sorted"


def test_get_spike_source_delegates_to_resolve(session, tmp_path: Path):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "spikes_sorted.csv").write_text("time,unit_id\n")
    session[state.KEY_SPIKE_SOURCE] = "sorted"
    assert state.get_spike_source(exp) == "sorted"
