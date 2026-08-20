"""Unit tests for non-increasing job remaining-time math."""

from __future__ import annotations

from ui.components.run_status import remaining_s


def test_remaining_never_negative():
    assert remaining_s(elapsed=999.0, step=0, total=10, estimate_s=10.0) == 0.0
    assert remaining_s(elapsed=50.0, step=10, total=10, estimate_s=None) == 0.0


def test_estimate_minus_elapsed_before_pace():
    assert remaining_s(elapsed=20.0, step=0, total=100, estimate_s=100.0) == 80.0
    assert remaining_s(elapsed=20.0, step=1, total=100, estimate_s=100.0) == 80.0


def test_pace_after_two_steps():
    # elapsed/step * remaining steps: (10/2) * 8 = 40
    assert remaining_s(elapsed=10.0, step=2, total=10, estimate_s=100.0) == 40.0


def test_remaining_never_increases():
    later = remaining_s(
        elapsed=100.0,
        step=3,
        total=10,
        estimate_s=50.0,
        prev_remaining=30.0,
    )
    assert later == 30.0
    ticked = remaining_s(
        elapsed=100.0,
        step=3,
        total=10,
        estimate_s=50.0,
        prev_remaining=30.0,
        dt=5.0,
    )
    assert ticked == 25.0


def test_none_without_estimate_or_steps():
    assert remaining_s(elapsed=5.0, step=0, total=10) is None
