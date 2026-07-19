"""Tests for unit metadata normalization (Phase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hippo.unit_metadata import (
    REQUIRED_UNIT_COLUMNS,
    metadata_availability_table,
    normalize_unit_metadata,
)


def _minimal_units(n: int = 10) -> pd.DataFrame:
    regions = ["CA1", "CA1", "CA3", "DG", "CA2"] * 2
    layers = ["oriens", "pyramidal", "pyramidal", "granule", "pyramidal"] * 2
    return pd.DataFrame({
        "unit_id": np.arange(n),
        "cell_type": ["CA1_pyr", "CA1_pyr", "CA3_pyr", "DG_granule", "CA2_pyr"] * 2,
        "region": regions,
        "layer": layers,
        "channel": np.arange(n),
        "depth_um": np.linspace(0, 2000, n),
        "place_x_cm": np.zeros(n),
        "place_y_cm": np.zeros(n),
        "hd_pref_rad": np.zeros(n),
        "rate_model": ["custom"] * n,
        "ratinabox_class": ["PlaceCells"] * n,
    })


def test_required_columns_enforced():
    df = pd.DataFrame({"unit_id": [0], "region": ["CA1"]})
    with pytest.raises(ValueError, match="missing required"):
        normalize_unit_metadata(df)


def test_normalize_derives_cell_class_and_subfield():
    df = normalize_unit_metadata(_minimal_units())
    assert "cell_class" in df.columns
    assert "subfield" in df.columns
    assert set(df["subfield"]) == set(df["region"])
    assert (df["cell_class"] == "excitatory").all()


def test_deep_superficial_from_layer():
    df = normalize_unit_metadata(_minimal_units())
    ca1 = df[df["region"] == "CA1"]
    assert set(ca1["deep_superficial_group"]).issubset(
        {"deep", "superficial", "intermediate", "unknown"}
    )
    oriens = ca1[ca1["layer"] == "oriens"]
    assert (oriens["deep_superficial_group"] == "deep").all()
    assert (oriens["deep_superficial_coordinate"] == 0.0).all()


def test_anatomical_coordinates_in_unit_interval_when_finite():
    df = normalize_unit_metadata(_minimal_units())
    for col in ("deep_superficial_coordinate",):
        vals = df[col].dropna()
        assert ((vals >= 0) & (vals <= 1)).all()


def test_optional_columns_reserved():
    df = normalize_unit_metadata(_minimal_units())
    assert "proximodistal_coordinate" in df.columns
    assert "projection_target" in df.columns
    assert df["metadata_schema_version"].iloc[0] == "phase1"


def test_metadata_availability_table():
    raw = _minimal_units()
    table = metadata_availability_table(raw)
    assert "column" in table.columns
    assert REQUIRED_UNIT_COLUMNS[0] in set(table["column"])
    unit_id_row = table[table["column"] == "unit_id"].iloc[0]
    assert bool(unit_id_row["in_raw_units_csv"])
