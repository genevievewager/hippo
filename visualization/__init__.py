"""Visualization suite for hippocampal Neuropixels simulation outputs.

Public entry point: ``run_visualizations.py`` (calls
``visualization.experiment_viz.generate_experiment_figures``).

Keep this package ``__init__`` lightweight so UI / tooling imports of
submodules (e.g. ``artifact_manifest``) do not pull in the full plotting stack.
"""

from visualization.constants import (
    CELL_CLASS_ORDER,
    FIGURE_DPI,
    MAX_LINE_POINTS,
    REGION_ORDER,
    cell_class_colors,
    circuit_node_colors,
)

__all__ = [
    "CELL_CLASS_ORDER",
    "REGION_ORDER",
    "FIGURE_DPI",
    "MAX_LINE_POINTS",
    "cell_class_colors",
    "circuit_node_colors",
    "SimulationOutputs",
    "load_simulation_outputs",
    "generate_experiment_figures",
    "compile_figures_pdf",
]


def __getattr__(name: str):
    """Lazy exports for heavy plotting entry points."""
    if name in {"SimulationOutputs", "load_simulation_outputs"}:
        from visualization.load_outputs import SimulationOutputs, load_simulation_outputs

        return {
            "SimulationOutputs": SimulationOutputs,
            "load_simulation_outputs": load_simulation_outputs,
        }[name]
    if name == "generate_experiment_figures":
        from visualization.experiment_viz import generate_experiment_figures

        return generate_experiment_figures
    if name == "compile_figures_pdf":
        from visualization.pdf import compile_figures_pdf

        return compile_figures_pdf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
