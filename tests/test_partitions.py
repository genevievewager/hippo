"""Tests for population partition registry (Phase 1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hippo.partitions import (
    PLANNED_PARTITIONS,
    apply_partitions,
    available_partitions,
    make_partition,
)
from hippo.unit_metadata import normalize_unit_metadata


def _units(n_per_region: int = 8) -> pd.DataFrame:
    rows = []
    uid = 0
    for region, layer, ct in [
        ("CA1", "oriens", "CA1_pyr"),
        ("CA1", "pyramidal", "CA1_pyr"),
        ("CA1", "radiatum", "CA1_pyr"),
        ("CA3", "pyramidal", "CA3_pyr"),
        ("DG", "granule", "DG_granule"),
    ]:
        for _ in range(n_per_region):
            rows.append({
                "unit_id": uid,
                "cell_type": ct,
                "region": region,
                "layer": layer,
                "channel": uid,
                "depth_um": float(uid),
                "place_x_cm": 0.0,
                "place_y_cm": 0.0,
                "hd_pref_rad": 0.0,
                "rate_model": "custom",
                "ratinabox_class": "PlaceCells",
            })
            uid += 1
    return normalize_unit_metadata(pd.DataFrame(rows))


def test_available_partitions_includes_anatomical():
    avail = available_partitions()
    for name in (
        "all_units", "subfield", "layer", "ca1_deep_superficial",
        "deep_superficial", "cell_class", "cell_type",
    ):
        assert name in avail


def test_all_units_partition():
    units = _units()
    part = make_partition("all_units").apply(units, min_units_per_group=5)
    assert "all" in part.group_labels
    assert len(part.group_labels["all"]) == len(units)


def test_subfield_partition_groups():
    units = _units()
    part = make_partition("subfield").apply(units, min_units_per_group=5)
    assert "CA1" in part.group_labels
    assert "CA3" in part.group_labels
    assert "DG" in part.group_labels


def test_min_units_filtering():
    units = _units(n_per_region=2)  # CA3/DG have only 2 each
    part = make_partition("subfield").apply(units, min_units_per_group=5)
    # CA1 has 3 layers × 2 = 6 units → kept; CA3/DG excluded
    assert "CA1" in part.group_labels
    assert "CA3" in part.exclusion_reason or "CA3" not in part.group_labels


def test_ca1_deep_superficial():
    units = _units()
    part = make_partition("ca1_deep_superficial").apply(units, min_units_per_group=5)
    assert any("deep" in g for g in part.group_labels)
    assert any("superficial" in g for g in part.group_labels)
    for ids in part.group_labels.values():
        assert set(units.set_index("unit_id").loc[ids, "region"]) == {"CA1"}


def test_deep_superficial_general_and_sub_ent():
    rows = []
    uid = 0
    for region, layer, ct in [
        ("subiculum", "deep", "Sub_bvc"),
        ("subiculum", "superficial", "Sub_bvc"),
        ("entorhinal_cortex", "deep", "MEC_grid"),
        ("entorhinal_cortex", "superficial", "MEC_hd"),
    ]:
        for _ in range(6):
            rows.append({
                "unit_id": uid,
                "cell_type": ct,
                "region": region,
                "layer": layer,
                "channel": uid,
                "depth_um": float(uid),
                "place_x_cm": 0.0,
                "place_y_cm": 0.0,
                "hd_pref_rad": 0.0,
                "rate_model": "custom",
                "ratinabox_class": "PlaceCells",
            })
            uid += 1
    units = normalize_unit_metadata(pd.DataFrame(rows))
    # CA1-only partition should be empty
    ca1_part = make_partition("ca1_deep_superficial").apply(units, min_units_per_group=5)
    assert ca1_part.group_labels == {}
    part = make_partition("deep_superficial").apply(units, min_units_per_group=5)
    assert any(g.startswith("Subiculum_") for g in part.group_labels)
    assert any(g.startswith("MEC_") for g in part.group_labels)


def test_cell_type_partition_sub_ent():
    rows = []
    uid = 0
    for ct, region in [("Sub_bvc", "subiculum"), ("MEC_grid", "entorhinal_cortex")]:
        for _ in range(6):
            rows.append({
                "unit_id": uid,
                "cell_type": ct,
                "region": region,
                "layer": "SUB" if ct == "Sub_bvc" else "ENT",
                "channel": uid,
                "depth_um": float(uid),
                "place_x_cm": 0.0,
                "place_y_cm": 0.0,
                "hd_pref_rad": 0.0,
                "rate_model": f"ratinabox_{ct}",
                "ratinabox_class": ct,
            })
            uid += 1
    units = normalize_unit_metadata(pd.DataFrame(rows))
    part = make_partition("cell_type").apply(units, min_units_per_group=5)
    assert "Sub_bvc" in part.group_labels
    assert "MEC_grid" in part.group_labels


def test_planned_partition_raises():
    with pytest.raises(ValueError, match="Unknown partition"):
        make_partition("physiology_cluster")
    assert "physiology_cluster" in PLANNED_PARTITIONS


def test_apply_partitions_batch():
    units = _units()
    results = apply_partitions(units, ["all_units", "cell_class"], min_units_per_group=5)
    assert set(results) == {"all_units", "cell_class"}
    assert results["cell_class"].group_labels
