"""Tests for hippocampal-system allowlist (decode / manifold exclusion)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from hippo.anatomy.hippocampal_system import (
    ALLOWED_CELL_TYPES,
    annotate_units_for_analysis,
    canonicalize_region,
    filter_unit_ids_for_analysis,
    filter_units_for_analysis,
    geometry_summary,
    infer_circuit_profile,
    is_allowed_cell_type,
    is_hippocampal_region,
    recommended_partitions,
    unit_include_in_decoder,
)
from realtime.data_loading import load_simulation_data


def test_allowed_cell_types_match_ratinabox_vocabulary():
    assert "CA1_pyr" in ALLOWED_CELL_TYPES
    assert "Sub_bvc" in ALLOWED_CELL_TYPES
    assert "MEC_grid" in ALLOWED_CELL_TYPES
    assert "visual_units_optional" not in ALLOWED_CELL_TYPES
    assert not is_allowed_cell_type("visual_units_optional")
    assert not is_allowed_cell_type("HATA_optional")


def test_region_aliases_canonicalize_lab_names():
    assert canonicalize_region("subiculum") == "Subiculum"
    assert canonicalize_region("entorhinal_cortex") == "MEC"
    assert canonicalize_region("dentate_gyrus") == "DG"
    assert canonicalize_region("HPF_ProS_transition") == "Subiculum"
    assert canonicalize_region("deep_entorhinal_HATA") == "MEC"
    assert is_hippocampal_region("subiculum")
    assert not is_hippocampal_region("visual_cortex")


def test_visual_cortex_units_excluded_from_decoder():
    assert not unit_include_in_decoder(
        cell_type="visual_units_optional", region="visual_cortex",
    )
    # Even a RiaB type wrongly placed in VIS is excluded by region.
    assert not unit_include_in_decoder(
        cell_type="CA1_pyr", region="visual_cortex",
    )
    assert unit_include_in_decoder(
        cell_type="Sub_bvc", region="subiculum",
    )


def test_annotate_and_filter_units():
    df = pd.DataFrame([
        {"unit_id": 0, "cell_type": "Sub_bvc", "region": "subiculum"},
        {"unit_id": 1, "cell_type": "visual_units_optional", "region": "visual_cortex"},
        {"unit_id": 2, "cell_type": "MEC_grid", "region": "entorhinal_cortex"},
        {"unit_id": 3, "cell_type": "HATA_optional", "region": "deep_entorhinal_HATA"},
    ])
    annotated = annotate_units_for_analysis(df)
    assert annotated.loc[0, "include_in_decoder"]
    assert not annotated.loc[1, "include_in_decoder"]
    assert annotated.loc[2, "include_in_decoder"]
    assert not annotated.loc[3, "include_in_decoder"]
    assert annotated.loc[0, "region_canonical"] == "Subiculum"
    assert annotated.loc[2, "region_canonical"] == "MEC"

    filtered = filter_units_for_analysis(df)
    assert set(filtered["unit_id"]) == {0, 2}
    assert filter_unit_ids_for_analysis(df, [0, 1, 2, 3]) == [0, 2]


def test_sub_ent_circuit_profile_and_partitions():
    df = pd.DataFrame([
        {"unit_id": i, "cell_type": ct, "region": region}
        for i, (ct, region) in enumerate([
            ("Sub_bvc", "subiculum"),
            ("Sub_bvc", "subiculum"),
            ("MEC_grid", "entorhinal_cortex"),
            ("MEC_hd", "entorhinal_cortex"),
            ("visual_units_optional", "visual_cortex"),
        ])
    ])
    assert infer_circuit_profile(df) == "sub_ent_dominant"
    geom = geometry_summary(df)
    assert geom["sub_ent_dominant"] is True
    assert "Subiculum" in geom["present_regions"]
    assert "MEC" in geom["present_regions"]
    assert "CA1" not in geom["present_regions"]
    parts = recommended_partitions(units_df=df)
    assert "ca1_deep_superficial" not in parts
    assert "deep_superficial" in parts
    assert "subfield" in parts
    assert "cell_type" in parts


def test_trisynaptic_profile_when_ca_fields_present():
    df = pd.DataFrame([
        {"unit_id": 0, "cell_type": "CA1_pyr", "region": "CA1"},
        {"unit_id": 1, "cell_type": "CA3_pyr", "region": "CA3"},
        {"unit_id": 2, "cell_type": "DG_granule", "region": "DG"},
    ])
    assert infer_circuit_profile(df) == "trisynaptic_full"
    parts = recommended_partitions(units_df=df)
    assert "ca1_deep_superficial" in parts


def test_load_simulation_data_drops_non_hippocampal(tmp_path):
    exp = tmp_path / "run"
    exp.mkdir()
    units = pd.DataFrame([
        {"unit_id": 0, "cell_type": "Sub_bvc", "region": "subiculum", "layer": "SUB"},
        {"unit_id": 1, "cell_type": "visual_units_optional", "region": "visual_cortex", "layer": "VIS"},
        {"unit_id": 2, "cell_type": "MEC_hd", "region": "entorhinal_cortex", "layer": "ENT"},
    ])
    units.to_csv(exp / "units.csv", index=False)
    spikes = pd.DataFrame({
        "unit_id": [0, 0, 1, 1, 2],
        "spike_time_s": [0.1, 0.2, 0.15, 0.25, 0.3],
    })
    spikes.to_csv(exp / "spikes_sorted.csv", index=False)
    pd.DataFrame({"time_s": [0.0, 0.05, 0.1], "x_cm": [0, 1, 2], "y_cm": [0, 0, 0]}).to_csv(
        exp / "behavior.csv", index=False,
    )
    (exp / "summary.json").write_text('{"session_duration_s": 1.0}')

    data = load_simulation_data(exp, "sorted")
    assert data["unit_ids"] == [0, 2]
    assert data["n_units_excluded"] == 1
    assert set(data["spikes_df"]["unit_id"]) == {0, 2}
    assert 1 not in set(data["spikes_df"]["unit_id"])
