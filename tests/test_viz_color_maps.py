"""Stable color maps for cell classes / circuit nodes / regions."""

from visualization.constants import (
    CELL_CLASS_COLORS,
    CELL_CLASS_ORDER,
    CIRCUIT_NODE_COLORS,
    CIRCUIT_NODE_ORDER,
    cell_class_colors,
    circuit_node_colors,
    region_colors,
)


def test_cell_class_colors_stable_across_subsets():
    full = cell_class_colors()
    subset = cell_class_colors(["MEC_grid", "INT_CA1", "DG_granule"])
    assert subset["INT_CA1"] == full["INT_CA1"] == CELL_CLASS_COLORS["INT_CA1"]
    assert subset["MEC_grid"] == full["MEC_grid"]
    # Subset order must not reassign colors.
    flipped = cell_class_colors(["DG_granule", "INT_CA1", "MEC_grid"])
    assert flipped["INT_CA1"] == subset["INT_CA1"]
    # Legacy alias shares INT_CA1 color.
    assert CELL_CLASS_COLORS["interneuron"] == CELL_CLASS_COLORS["INT_CA1"]


def test_circuit_node_colors_stable_across_subsets():
    full = circuit_node_colors()
    subset = circuit_node_colors(["INT_CA1", "MEC", "CA1"])
    assert subset["INT_CA1"] == full["INT_CA1"] == CIRCUIT_NODE_COLORS["INT_CA1"]
    assert subset["MEC"] == full["MEC"]
    flipped = circuit_node_colors(["CA1", "MEC", "INT_CA1"])
    assert flipped["INT_CA1"] == subset["INT_CA1"]
    assert CIRCUIT_NODE_COLORS["INT"] == CIRCUIT_NODE_COLORS["INT_CA1"]


def test_region_colors_align_to_circuit_nodes():
    colors = region_colors(["CA1", "Subiculum", "MEC"])
    assert colors["CA1"] == CIRCUIT_NODE_COLORS["CA1"]
    assert colors["Subiculum"] == CIRCUIT_NODE_COLORS["SUB"]
    assert colors["MEC"] == CIRCUIT_NODE_COLORS["MEC"]


def test_canonical_orders_cover_color_maps():
    assert set(CELL_CLASS_ORDER).issubset(set(CELL_CLASS_COLORS))
    assert set(CIRCUIT_NODE_ORDER).issubset(set(CIRCUIT_NODE_COLORS))
