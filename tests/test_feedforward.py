"""Tests for trisynaptic / entorhinal feedforward coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.feedforward import (
    FEEDFORWARD_PARAMS,
    apply_int_to_ca1_inhibition,
    apply_trisynaptic_feedforward,
)


@dataclass
class _G:
    name: str
    rates_hz: np.ndarray


def _const_group(name: str, n_cells: int, n_steps: int, value: float) -> _G:
    return _G(name=name, rates_hz=np.full((n_cells, n_steps), value, dtype=float))


def test_feedforward_raises_dg_when_mec_active():
    t = 20
    groups = [
        _const_group("MEC_grid", 5, t, 10.0),
        _const_group("DG_place", 4, t, 1.0),
        _const_group("CA3_place", 3, t, 1.0),
        _const_group("CA1_place_pp", 6, t, 1.0),
        _const_group("CA2_place", 2, t, 1.0),
    ]
    dg_before = groups[1].rates_hz.copy()
    ca3_before = groups[2].rates_hz.copy()
    ca1_before = groups[3].rates_hz.copy()

    meta = apply_trisynaptic_feedforward(groups, ratinabox_params={"apply_feedforward": True})
    assert meta["applied"] is True
    assert groups[1].rates_hz.mean() > dg_before.mean()
    assert groups[2].rates_hz.mean() > ca3_before.mean()
    assert groups[3].rates_hz.mean() > ca1_before.mean()


def test_feedforward_can_be_disabled():
    t = 10
    groups = [
        _const_group("MEC_grid", 2, t, 10.0),
        _const_group("DG_place", 2, t, 1.0),
    ]
    before = groups[1].rates_hz.copy()
    meta = apply_trisynaptic_feedforward(groups, ratinabox_params={"apply_feedforward": False})
    assert meta["applied"] is False
    assert np.allclose(groups[1].rates_hz, before)


def test_int_inhibits_ca1():
    t = 15
    groups = [
        _const_group("CA1_place_pp", 4, t, 5.0),
        _const_group("CA1_int", 3, t, 20.0),
    ]
    before = groups[0].rates_hz.copy()
    meta = apply_int_to_ca1_inhibition(groups)
    assert meta["applied"] is True
    assert groups[0].rates_hz.mean() < before.mean()
    assert groups[0].rates_hz.min() >= 0.0


def test_default_weights_match_documented_hypotheses():
    assert FEEDFORWARD_PARAMS["w_mec_to_dg"] == 0.20
    assert FEEDFORWARD_PARAMS["w_dg_to_ca3"] == 0.25
    assert FEEDFORWARD_PARAMS["w_ca3_to_ca1"] == 0.20
    assert FEEDFORWARD_PARAMS["w_mec_to_ca1"] == 0.15
    assert FEEDFORWARD_PARAMS["w_int_to_ca1"] == 0.30
