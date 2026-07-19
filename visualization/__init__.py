"""Visualization suite for hippocampal Neuropixels simulation outputs.

Public entry point: ``run_visualizations.py`` (calls
``visualization.experiment_viz.generate_experiment_figures``).
"""

from visualization.constants import (
    CELL_CLASS_ORDER,
    FIGURE_DPI,
    MAX_LINE_POINTS,
    REGION_ORDER,
)
from visualization.experiment_viz import generate_experiment_figures
from visualization.load_outputs import SimulationOutputs, load_simulation_outputs
from visualization.pdf import compile_figures_pdf

__all__ = [
    "CELL_CLASS_ORDER",
    "REGION_ORDER",
    "FIGURE_DPI",
    "MAX_LINE_POINTS",
    "SimulationOutputs",
    "load_simulation_outputs",
    "generate_experiment_figures",
    "compile_figures_pdf",
]
