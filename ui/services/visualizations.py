"""Generate experiment figures from the UI (wraps visualization.experiment_viz)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from visualization.experiment_viz import (
    VizResult,
    generate_experiment_figures,
    has_decoder_comparison,
    has_realtime_decoding,
    has_simulation_outputs,
)

ProgressCallback = Callable[[str, int, int], None]


@dataclass
class VisualizationRequest:
    """UI request mirroring ``run_visualizations.py`` flags."""

    experiment_dir: Path
    include_simulation: bool = True
    include_comparison: bool = True
    include_realtime: bool = True
    compile_pdf: bool = False
    rate_bin_size: float = 0.250


def detect_viz_inputs(experiment_dir: Path) -> dict[str, bool]:
    """What on-disk outputs can drive figure generation."""
    exp = Path(experiment_dir)
    return {
        "simulation": has_simulation_outputs(exp),
        "comparison": has_decoder_comparison(exp),
        "realtime": has_realtime_decoding(exp),
        "figures_dir_exists": (exp / "figures").is_dir(),
    }


def default_viz_request(experiment_dir: Path) -> VisualizationRequest:
    """Smart defaults: generate every figure type whose inputs exist."""
    avail = detect_viz_inputs(experiment_dir)
    return VisualizationRequest(
        experiment_dir=Path(experiment_dir),
        include_simulation=avail["simulation"],
        include_comparison=avail["comparison"],
        include_realtime=avail["realtime"],
        compile_pdf=False,
    )


def generate_visualizations(
    req: VisualizationRequest,
    *,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Populate ``figures/`` for an experiment without retraining decoders."""
    exp = Path(req.experiment_dir)
    if not exp.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp}")

    if progress_callback:
        progress_callback("Generating experiment figures…", 1, 2)

    result: VizResult = generate_experiment_figures(
        experiment_dir=exp,
        figures_dir=exp / "figures",
        include_simulation=req.include_simulation,
        include_comparison=req.include_comparison,
        include_realtime=req.include_realtime,
        compile_pdf=req.compile_pdf,
        rate_bin_size=req.rate_bin_size,
    )

    if progress_callback:
        progress_callback("Figures ready", 2, 2)

    generated: list[str] = []
    if result.simulation:
        generated.append("simulation")
    if result.comparison:
        generated.append("decoder_comparison")
    if result.realtime:
        generated.append("realtime")

    return {
        "experiment_dir": str(exp),
        "figures_dir": str(result.figures_dir),
        "generated": generated,
        "pdf_path": str(result.pdf_path) if result.pdf_path else None,
        "simulation": result.simulation,
        "comparison": result.comparison,
        "realtime": result.realtime,
    }
