"""Tests for maximally hippocampal RiaB population table and backend."""

from __future__ import annotations

import numpy as np
import pytest

from hippo_sim.config import (
    CELL_TYPES,
    RATE_PARAMS,
    REGION_TO_CELL_TYPE,
    SimConfig,
)
from hippo_sim.hippocampal_populations import (
    HIPPOCAMPAL_RIA_B_POPULATIONS,
    population_count,
    population_table_summary,
)
from hippo.unit_metadata import normalize_unit_metadata
import pandas as pd


def test_population_table_has_expected_groups():
    names = [s["name"] for s in HIPPOCAMPAL_RIA_B_POPULATIONS]
    assert names == [
        "CA1_place_pp",
        "CA3_place",
        "DG_place",
        "CA2_place",
        "MEC_grid",
        "MEC_hd",
        "Sub_bvc",
        "MEC_speed",
        "INT_CA1",
        "INT_CA3",
        "INT_DG",
        "INT_CA2",
        "INT_SUB",
    ]


def test_population_table_total_default_count():
    total = sum(population_count(s, {}) for s in HIPPOCAMPAL_RIA_B_POPULATIONS)
    assert total == 305  # 290 principals + 15 local INT


def test_population_cell_types_have_rate_params():
    for spec in HIPPOCAMPAL_RIA_B_POPULATIONS:
        assert spec["cell_type"] in RATE_PARAMS
        assert spec["cell_type"] in CELL_TYPES


def test_region_mapping_is_anatomically_plausible():
    assert REGION_TO_CELL_TYPE[("CA1", "oriens")] == "INT_CA1"
    assert REGION_TO_CELL_TYPE[("CA1", "pyramidal")] == "CA1_pyr"
    assert REGION_TO_CELL_TYPE[("MEC", "layer2")] == "MEC_grid"
    assert REGION_TO_CELL_TYPE[("MEC", "layer3")] == "MEC_hd"
    assert REGION_TO_CELL_TYPE[("Subiculum", "pyramidal")] == "Sub_bvc"


def test_interneuron_is_inhibitory_in_metadata():
    df = normalize_unit_metadata(pd.DataFrame({
        "unit_id": [0, 1, 2],
        "cell_type": ["INT_CA1", "INT_DG", "CA1_pyr"],
        "region": ["CA1", "DG", "CA1"],
        "layer": ["oriens", "hilus", "pyramidal"],
    }))
    assert df.loc[0, "cell_class"] == "inhibitory"
    assert df.loc[1, "cell_class"] == "inhibitory"
    assert df.loc[2, "cell_class"] == "excitatory"


def test_local_int_specs_have_home_nodes():
    homes = {
        s["name"]: s["home_node"]
        for s in HIPPOCAMPAL_RIA_B_POPULATIONS
        if s["name"].startswith("INT_")
    }
    assert homes == {
        "INT_CA1": "CA1",
        "INT_CA3": "CA3",
        "INT_DG": "DG",
        "INT_CA2": "CA2",
        "INT_SUB": "SUB",
    }


def test_population_table_summary_rows():
    rows = population_table_summary(dict(SimConfig().ratinabox_params))
    assert len(rows) == len(HIPPOCAMPAL_RIA_B_POPULATIONS)
    assert rows[0]["ratinabox_class"] == "PhasePrecessingPlaceCells"
    assert "phase precession" in rows[0]["notes"].lower() or "Phase" in rows[0]["notes"]


def test_ratinabox_backend_smoke_short_session(tmp_path):
    pytest.importorskip("ratinabox")
    from hippo_sim.behavior import simulate_behavior
    from hippo_sim.ratinabox_neural_backend import simulate_ratinabox_neural_activity

    config = SimConfig(
        output_dir=tmp_path,
        seed=0,
        session_duration_s=2.0,
    )
    # Shrink populations for a fast smoke test.
    for key in list(config.ratinabox_params):
        if key.startswith("n_"):
            config.ratinabox_params[key] = min(int(config.ratinabox_params[key]), 4)

    behavior = simulate_behavior(config)
    units, rates, meta = simulate_ratinabox_neural_activity(
        config, behavior.trace, env=behavior.env, agent=behavior.agent,
    )
    assert len(units) == rates.shape[0]
    assert rates.shape[1] == config.n_behavior_steps
    assert rates.min() >= 0
    cell_types = {u.cell_type for u in units}
    assert "CA1_pyr" in cell_types
    assert "MEC_grid" in cell_types or "MEC_hd" in cell_types
    assert any(ct.startswith("INT_") for ct in cell_types)
    assert "population_table" in str(meta) or "ratinabox_population_table" in meta
    assert meta["ratinabox_cell_groups"]
    ff = meta.get("feedforward") or {}
    assert "local_int_inhibition" in ff or "int_to_ca1" in ff
