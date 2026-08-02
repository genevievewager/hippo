"""Tests for publication probe-trajectory figures."""

from __future__ import annotations

import pandas as pd

from visualization.constants import (
    ANALYSIS_INTERNEURON_CELL_CLASS,
    analysis_cell_class,
    cell_class_order_for_counts,
)


def test_analysis_cell_class_collapses_local_int_pools():
    assert analysis_cell_class("INT_CA1") == ANALYSIS_INTERNEURON_CELL_CLASS
    assert analysis_cell_class("INT_CA3") == ANALYSIS_INTERNEURON_CELL_CLASS
    assert analysis_cell_class("interneuron") == ANALYSIS_INTERNEURON_CELL_CLASS
    assert analysis_cell_class("CA1_pyr") == "CA1_pyr"


def test_panel_d_count_matrix_uses_one_interneuron_column():
    units = pd.DataFrame({
        "region": ["CA1", "CA1", "CA2", "CA3", "CA3", "DG", "Subiculum"],
        "cell_type": [
            "CA1_pyr", "INT_CA1", "CA2_pyr", "CA3_pyr", "INT_CA3",
            "INT_DG", "INT_SUB",
        ],
    })
    cell_classes = units["cell_type"].astype(str).map(analysis_cell_class)
    ct = pd.crosstab(units["region"], cell_classes)
    cols = cell_class_order_for_counts(ct.columns)

    assert ANALYSIS_INTERNEURON_CELL_CLASS in cols
    assert "INT_CA1" not in cols
    assert "INT_CA3" not in cols

    table = ct.reindex(columns=cols).fillna(0).astype(int)
    assert table.loc["CA1", ANALYSIS_INTERNEURON_CELL_CLASS] == 1
    assert table.loc["CA3", ANALYSIS_INTERNEURON_CELL_CLASS] == 1
    assert table.loc["DG", ANALYSIS_INTERNEURON_CELL_CLASS] == 1
    assert table.loc["Subiculum", ANALYSIS_INTERNEURON_CELL_CLASS] == 1
    assert table.loc["CA2", ANALYSIS_INTERNEURON_CELL_CLASS] == 0
