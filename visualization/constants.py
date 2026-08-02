"""Shared constants for visualization."""

from __future__ import annotations

from typing import Iterable

CELL_CLASS_ORDER = [
    "CA1_pyr",
    "INT_CA1",  # 3×3 INT panel; pools all local INT_* (see neural_plots)
    "CA2_pyr",
    "CA3_pyr",
    "DG_granule",
    "Sub_bvc",
    "MEC_grid",
    "MEC_hd",
    "MEC_speed",
]

# Extra class labels that appear in population / color maps but not the 3×3 grid.
CELL_CLASS_ORDER_EXTRA = [
    "INT_CA2",
    "INT_CA3",
    "INT_DG",
    "INT_SUB",
    "interneuron",  # legacy alias
]

# Region-local INT pools share one rate model; collapse for count heatmaps.
LOCAL_INT_CELL_TYPES = frozenset({
    "INT_CA1", "INT_CA2", "INT_CA3", "INT_DG", "INT_SUB",
    "interneuron", "CA1_int",
})
ANALYSIS_INTERNEURON_CELL_CLASS = "interneuron"
# Canonical analysis regions first; lab trajectory aliases follow for figure order.
REGION_ORDER = [
    "CA1",
    "CA2",
    "CA3",
    "DG",
    "Subiculum",
    "MEC",
    "HPF_ProS_transition",
    "subiculum",
    "dentate_gyrus",
    "entorhinal_cortex",
    "deep_entorhinal_HATA",
    "visual_cortex",
]

# RatInABox population / rate_model display order (matches hippocampal_populations).
RATE_MODEL_ORDER = [
    "ratinabox_CA1_place_pp",
    "ratinabox_CA3_place",
    "ratinabox_DG_place",
    "ratinabox_CA2_place",
    "ratinabox_MEC_grid",
    "ratinabox_MEC_hd",
    "ratinabox_Sub_bvc",
    "ratinabox_MEC_speed",
    "ratinabox_INT_CA1",
    "ratinabox_INT_CA3",
    "ratinabox_INT_DG",
    "ratinabox_INT_CA2",
    "ratinabox_INT_SUB",
    "ratinabox_interneuron",  # legacy
]

# Circuit nodes for feedforward population activity panels.
# Default order follows trisynaptic cascade; local INT satellites follow home.
# SUB/ENT-dominant profiles reorder via ``circuit_node_order_for_units``.
CIRCUIT_NODE_ORDER = [
    "MEC", "DG", "INT_DG", "CA3", "INT_CA3", "CA2", "INT_CA2",
    "CA1", "INT_CA1", "SUB", "INT_SUB",
]
CIRCUIT_NODE_ORDER_SUB_ENT = [
    "MEC", "SUB", "INT_SUB", "DG", "INT_DG", "CA3", "INT_CA3",
    "CA2", "INT_CA2", "CA1", "INT_CA1",
]
REGION_TO_CIRCUIT_NODE = {
    "MEC": "MEC",
    "DG": "DG",
    "CA3": "CA3",
    "CA2": "CA2",
    "CA1": "CA1",
    "Subiculum": "SUB",
    "subiculum": "SUB",
    "HPF_ProS_transition": "SUB",
    "dentate_gyrus": "DG",
    "entorhinal_cortex": "MEC",
    "deep_entorhinal_HATA": "MEC",
}
CELL_TYPE_TO_CIRCUIT_NODE = {
    "MEC_grid": "MEC",
    "MEC_hd": "MEC",
    "MEC_speed": "MEC",
    "DG_granule": "DG",
    "CA3_pyr": "CA3",
    "CA2_pyr": "CA2",
    "CA1_pyr": "CA1",
    "INT_CA1": "INT_CA1",
    "INT_CA2": "INT_CA2",
    "INT_CA3": "INT_CA3",
    "INT_DG": "INT_DG",
    "INT_SUB": "INT_SUB",
    "interneuron": "INT_CA1",  # legacy
    "CA1_int": "INT_CA1",  # legacy
    "Sub_bvc": "SUB",
}


def circuit_node_order_for_present(present_nodes: set[str] | list[str]) -> list[str]:
    """Prefer SUB-first ordering when CA fields are absent."""
    present = set(present_nodes)
    ca = present & {"CA1", "CA2", "CA3", "INT_CA1", "INT_CA2", "INT_CA3", "INT"}
    if (present & {"SUB", "MEC", "INT_SUB"}) and not ca:
        base = CIRCUIT_NODE_ORDER_SUB_ENT
    else:
        base = CIRCUIT_NODE_ORDER
    return [n for n in base if n in present] + sorted(present - set(base))


# Collapsed nodes in ``fig_circuit_feedforward`` (one INT, principals only).
FEEDFORWARD_DIAGRAM_NODE_ORDER = ["MEC", "DG", "CA3", "CA2", "CA1", "SUB", "INT"]


def _named_palette(names: list[str], cmap_name: str) -> dict[str, tuple]:
    """Stable name→RGBA map: color index is fixed by ``names`` order, not presence."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    try:
        cmap = plt.get_cmap(cmap_name)
        n_cmap = getattr(cmap, "N", 10) or 10
        return {name: cmap(i % n_cmap) for i, name in enumerate(names)}
    except ValueError:
        # Seaborn palette names (e.g. "deep") are not matplotlib colormaps.
        colors = sns.color_palette(cmap_name, n_colors=max(len(names), 1))
        return {name: colors[i % len(colors)] for i, name in enumerate(names)}


# Fixed across all figures: lookup by name, never by subset position.
_COLOR_CLASS_KEYS = CELL_CLASS_ORDER + [
    c for c in CELL_CLASS_ORDER_EXTRA if c not in CELL_CLASS_ORDER
]
CELL_CLASS_COLORS: dict[str, tuple] = _named_palette(_COLOR_CLASS_KEYS, "tab10")
# Legacy / shared INT colors.
CELL_CLASS_COLORS["interneuron"] = CELL_CLASS_COLORS["INT_CA1"]
CELL_CLASS_COLORS["CA1_int"] = CELL_CLASS_COLORS["INT_CA1"]
for _int_ct in ("INT_CA2", "INT_CA3", "INT_DG", "INT_SUB"):
    # Keep distinct hues from the extended palette; already assigned above.
    pass
CIRCUIT_NODE_COLORS: dict[str, tuple] = _named_palette(CIRCUIT_NODE_ORDER, "deep")
# Legacy single-INT node (older runs) shares INT_CA1 color.
CIRCUIT_NODE_COLORS["INT"] = CIRCUIT_NODE_COLORS["INT_CA1"]
_FALLBACK_COLOR = (0.55, 0.55, 0.55, 1.0)


def analysis_cell_class(cell_type: str) -> str:
    """Map simulation INT_* labels onto one analysis interneuron class."""
    key = str(cell_type)
    if key in LOCAL_INT_CELL_TYPES:
        return ANALYSIS_INTERNEURON_CELL_CLASS
    return key


def cell_class_order_for_counts(present: Iterable[str] | None = None) -> list[str]:
    """Column order for region × cell-class count heatmaps (one interneuron class)."""
    order = [
        ANALYSIS_INTERNEURON_CELL_CLASS if c == "INT_CA1" else c
        for c in CELL_CLASS_ORDER
    ]
    if present is None:
        return order
    present_set = {str(c) for c in present}
    return [c for c in order if c in present_set]


def cell_class_colors(
    present: Iterable[str] | None = None,
) -> dict[str, tuple]:
    """Return cell-class → color for the given labels (full map if ``present`` is None)."""
    if present is None:
        return dict(CELL_CLASS_COLORS)
    out: dict[str, tuple] = {}
    extras = [str(c) for c in present if str(c) not in CELL_CLASS_COLORS]
    extra_palette = (
        _named_palette([f"_extra_{i}" for i in range(len(extras))], "Pastel1")
        if extras else {}
    )
    for raw in present:
        key = str(raw)
        if key in CELL_CLASS_COLORS:
            out[key] = CELL_CLASS_COLORS[key]
        else:
            out[key] = extra_palette.get(f"_extra_{extras.index(key)}", _FALLBACK_COLOR)
    return out


def circuit_node_colors(
    present: Iterable[str] | None = None,
) -> dict[str, tuple]:
    """Return circuit-node → color (fixed by ``CIRCUIT_NODE_ORDER``)."""
    if present is None:
        return dict(CIRCUIT_NODE_COLORS)
    out: dict[str, tuple] = {}
    for raw in present:
        key = str(raw)
        out[key] = CIRCUIT_NODE_COLORS.get(key, _FALLBACK_COLOR)
    return out


def region_colors(
    present: Iterable[str] | None = None,
) -> dict[str, tuple]:
    """Region colors aligned to circuit nodes via ``REGION_TO_CIRCUIT_NODE``."""
    keys = list(present) if present is not None else list(REGION_ORDER)
    out: dict[str, tuple] = {}
    for raw in keys:
        key = str(raw)
        node = REGION_TO_CIRCUIT_NODE.get(key, key)
        out[key] = CIRCUIT_NODE_COLORS.get(node, _FALLBACK_COLOR)
    return out


FIGURE_DPI = 300
MAX_LINE_POINTS = 5000

# Subfolders under experiment figures/. Only these dirs (plus output.pdf)
# should appear at the figures root — never loose PNGs/CSVs.
FIGURE_SUBDIR_TRAJECTORY = "trajectory"
FIGURE_SUBDIR_BEHAVIOR = "behavior"
FIGURE_SUBDIR_FEATURES = "features"
FIGURE_SUBDIR_NEURAL = "neural"
FIGURE_SUBDIR_SORTING = "sorting"
FIGURE_SUBDIR_REPORT = "report"
FIGURE_SUBDIR_DECODER = "decoder_comparison"
FIGURE_SUBDIR_REALTIME = "realtime_decoding"
FIGURE_SUBDIR_TEMPORAL = "temporal_decoding"
FIGURE_SUBDIR_LATENCY = "latency"
FIGURE_SUBDIR_DEPLOYMENT = "deployment_decoder_selection"
