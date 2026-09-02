"""Unit tests for work-based remaining-time (ETA) math."""

from __future__ import annotations

from ui.components.run_status import remaining_s


def test_remaining_none_before_two_units():
    assert remaining_s(
        completed=0, total=10, ema_unit_s=None, n_measured=0,
    ) is None
    assert remaining_s(
        completed=1, total=10, ema_unit_s=5.0, n_measured=1,
    ) is None


def test_pace_after_two_units():
    assert remaining_s(
        completed=2, total=10, ema_unit_s=5.0, n_measured=2,
    ) == 40.0


def test_remaining_zero_when_done():
    assert remaining_s(
        completed=10, total=10, ema_unit_s=5.0, n_measured=4,
    ) == 0.0


def test_prior_only_before_live_pace():
    assert remaining_s(
        completed=0, total=10, ema_unit_s=None, n_measured=0, prior_unit_s=2.0,
    ) == 20.0


def test_live_ema_overrides_prior():
    assert remaining_s(
        completed=2, total=10, ema_unit_s=5.0, n_measured=2, prior_unit_s=2.0,
    ) == 40.0
