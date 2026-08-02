"""Tests for trisynaptic / entorhinal feedforward coupling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.feedforward import (
    FEEDFORWARD_PARAMS,
    apply_int_to_ca1_inhibition,
    apply_local_int_inhibition,
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
    assert meta["feedforward_profile"] == "trisynaptic_full"
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
        _const_group("INT_CA1", 3, t, 20.0),
    ]
    before = groups[0].rates_hz.copy()
    meta = apply_local_int_inhibition(groups)
    assert meta["applied"] is True
    assert groups[0].rates_hz.mean() < before.mean()
    assert groups[0].rates_hz.min() >= 0.0
    assert meta["edges"]["INT_CA1_to_CA1"]["applied"] is True


def test_legacy_interneuron_group_still_inhibits_ca1():
    t = 15
    groups = [
        _const_group("CA1_place_pp", 4, t, 5.0),
        _const_group("interneuron", 3, t, 20.0),
    ]
    before = groups[0].rates_hz.copy()
    meta = apply_int_to_ca1_inhibition(groups)
    assert meta["applied"] is True
    assert groups[0].rates_hz.mean() < before.mean()


def test_local_int_inhibits_each_home_region():
    t = 12
    groups = [
        _const_group("CA1_place_pp", 2, t, 5.0),
        _const_group("CA3_place", 2, t, 5.0),
        _const_group("DG_place", 2, t, 5.0),
        _const_group("INT_CA1", 2, t, 20.0),
        _const_group("INT_CA3", 2, t, 20.0),
        _const_group("INT_DG", 2, t, 20.0),
    ]
    before = {g.name: g.rates_hz.copy() for g in groups[:3]}
    meta = apply_local_int_inhibition(groups)
    assert meta["applied"] is True
    assert groups[0].rates_hz.mean() < before["CA1_place_pp"].mean()
    assert groups[1].rates_hz.mean() < before["CA3_place"].mean()
    assert groups[2].rates_hz.mean() < before["DG_place"].mean()
    # INT pools themselves are unchanged.
    assert np.allclose(groups[3].rates_hz, 20.0)


def test_default_weights_match_documented_hypotheses():
    assert FEEDFORWARD_PARAMS["w_mec_to_dg"] == 0.20
    assert FEEDFORWARD_PARAMS["w_dg_to_ca3"] == 0.25
    assert FEEDFORWARD_PARAMS["w_ca3_to_ca1"] == 0.20
    assert FEEDFORWARD_PARAMS["w_mec_to_ca1"] == 0.15
    assert FEEDFORWARD_PARAMS["w_int_to_ca1"] == 0.30
    assert FEEDFORWARD_PARAMS["w_int_to_ca3"] == 0.25
    assert FEEDFORWARD_PARAMS["w_int_to_dg"] == 0.25
    assert FEEDFORWARD_PARAMS["w_int_to_ca2"] == 0.20
    assert FEEDFORWARD_PARAMS["w_int_to_sub"] == 0.20


def test_sub_ent_profile_drives_sub_from_mec():
    t = 20
    groups = [
        _const_group("MEC_grid", 5, t, 10.0),
        _const_group("Sub_bvc", 6, t, 1.0),
    ]
    before = groups[1].rates_hz.copy()
    meta = apply_trisynaptic_feedforward(
        groups,
        ratinabox_params={"apply_feedforward": True, "feedforward": {"profile": "auto"}},
    )
    assert meta["feedforward_profile"] == "sub_ent_dominant"
    assert meta["feedforward_weights"]["w_mec_to_sub"] > 0
    assert groups[1].rates_hz.mean() > before.mean()


def test_missing_ca1_skips_ca_targets():
    t = 10
    groups = [
        _const_group("MEC_grid", 3, t, 8.0),
        _const_group("Sub_bvc", 4, t, 1.0),
    ]
    meta = apply_trisynaptic_feedforward(groups)
    assert "CA1" not in meta.get("present_nodes", [])
    assert "SUB" in meta["present_nodes"]
    assert meta["applied"] is True
