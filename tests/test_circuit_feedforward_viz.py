"""Circuit feedforward diagram aggregates local INT pools into one node."""

from pathlib import Path

import pandas as pd

from visualization.constants import CIRCUIT_NODE_COLORS, circuit_node_colors
from visualization.publication_circuit_plots import (
    _arc3_point,
    _diagram_node,
    _edge_label_xy,
    _int_homes_with_pools,
    _node_summary,
)
from visualization.load_outputs import SimulationOutputs


def _mock_outputs(units_df: pd.DataFrame) -> SimulationOutputs:
    return SimulationOutputs(
        input_dir=Path("/tmp"),
        units=units_df,
        spikes_gt=pd.DataFrame({"time": [], "unit_id": []}),
        spikes_sorted=pd.DataFrame({"time": [], "unit_id": []}),
        behavior=pd.DataFrame({"time_s": [0.0], "x": [0.0], "y": [0.0]}),
        summary={},
        unit_mean_rates_gt=pd.Series(
            {int(uid): 10.0 for uid in units_df["unit_id"]},
        ),
    )


def test_diagram_node_collapses_int_pools():
    assert _diagram_node("INT_CA1") == "INT"
    assert _diagram_node("INT_DG") == "INT"
    assert _diagram_node("CA1") == "CA1"


def test_node_summary_aggregates_int_pools():
    units = pd.DataFrame({
        "unit_id": [0, 1, 2, 3],
        "cell_type": ["CA1_pyr", "INT_CA1", "INT_CA3", "INT_DG"],
        "region": ["CA1", "CA1", "CA3", "DG"],
    })
    summary = _node_summary(_mock_outputs(units))
    nodes = set(summary["node"])
    assert "INT" in nodes
    assert "INT_CA1" not in nodes
    assert "INT_CA3" not in nodes
    int_row = summary.loc[summary["node"] == "INT"].iloc[0]
    assert int_row["n_units"] == 3
    assert int_row["cell_classes_short"] == "int"


def test_int_homes_with_pools():
    units = pd.DataFrame({
        "unit_id": [0, 1, 2],
        "cell_type": ["CA1_pyr", "INT_CA1", "INT_DG"],
        "region": ["CA1", "CA1", "DG"],
    })
    units["circuit_node"] = ["CA1", "INT_CA1", "INT_DG"]
    homes = _int_homes_with_pools(units)
    assert homes == {"CA1", "DG"}


def test_feedforward_diagram_uses_circuit_node_palette():
    colors = circuit_node_colors(["CA1", "INT"])
    assert colors["CA1"] == CIRCUIT_NODE_COLORS["CA1"]
    assert colors["INT"] == CIRCUIT_NODE_COLORS["INT"]


def test_edge_label_xy_follows_arc_midpoint():
    x0, y0, x1, y1 = 0.0, 0.0, 1.0, 0.0
    assert _arc3_point(x0, y0, x1, y1, 0.0, 0.5) == (0.5, 0.0)
    mx, my = _edge_label_xy("MEC", "DG", 0.28, 0.82, 0.86, 0.82)
    assert abs(mx - 0.57) < 0.02 and abs(my - 0.82) < 0.02
