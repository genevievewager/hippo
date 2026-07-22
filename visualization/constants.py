"""Shared constants for visualization."""

CELL_CLASS_ORDER = [
    "CA1_pyr",
    "CA1_int",
    "CA2_pyr",
    "CA3_pyr",
    "DG_granule",
    "Sub_bvc",
    "MEC_grid",
    "MEC_hd",
    "MEC_speed",
]
REGION_ORDER = ["CA1", "CA2", "CA3", "DG", "Subiculum", "MEC"]

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
    "ratinabox_CA1_int",
]

# Circuit nodes for feedforward population activity panels.
CIRCUIT_NODE_ORDER = ["MEC", "DG", "CA3", "CA2", "CA1", "INT", "SUB"]
REGION_TO_CIRCUIT_NODE = {
    "MEC": "MEC",
    "DG": "DG",
    "CA3": "CA3",
    "CA2": "CA2",
    "CA1": "CA1",
    "Subiculum": "SUB",
}
CELL_TYPE_TO_CIRCUIT_NODE = {
    "MEC_grid": "MEC",
    "MEC_hd": "MEC",
    "MEC_speed": "MEC",
    "DG_granule": "DG",
    "CA3_pyr": "CA3",
    "CA2_pyr": "CA2",
    "CA1_pyr": "CA1",
    "CA1_int": "INT",
    "Sub_bvc": "SUB",
}

FIGURE_DPI = 300
MAX_LINE_POINTS = 5000

# Subfolders under experiment figures/. Only these dirs (plus output.pdf)
# should appear at the figures root — never loose PNGs/CSVs.
FIGURE_SUBDIR_BEHAVIOR = "behavior"
FIGURE_SUBDIR_FEATURES = "features"
FIGURE_SUBDIR_NEURAL = "neural"
FIGURE_SUBDIR_SORTING = "sorting"
FIGURE_SUBDIR_REPORT = "report"
FIGURE_SUBDIR_DECODER = "decoder_comparison"
FIGURE_SUBDIR_REALTIME = "realtime_decoding"
FIGURE_SUBDIR_TEMPORAL = "temporal_decoding"
