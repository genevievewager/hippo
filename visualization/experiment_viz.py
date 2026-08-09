"""Unified experiment visualization: simulation + decoder figures + optional PDF.

Plot-only: reads saved CSV/JSON outputs and never retrains decoders.
Public entry point: ``run_visualizations.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from visualization.behavior_plots import generate_behavior_plots
from visualization.cell_class_plots import generate_cell_class_plots
from visualization.constants import (
    FIGURE_SUBDIR_BEHAVIOR,
    FIGURE_SUBDIR_FEATURES,
    FIGURE_SUBDIR_NEURAL,
    FIGURE_SUBDIR_REPORT,
    FIGURE_SUBDIR_SORTING,
    FIGURE_SUBDIR_TRAJECTORY,
)
from visualization.feature_plots import generate_feature_plots
from visualization.load_outputs import load_simulation_outputs
from visualization.neural_plots import generate_neural_plots
from visualization.pdf import compile_figures_pdf
from visualization.report_figures import generate_report_figures


@dataclass
class VizResult:
    """Summary of what was generated for an experiment."""

    figures_dir: Path
    simulation: bool = False
    comparison: bool = False
    realtime: bool = False
    pdf_path: Path | None = None


def has_simulation_outputs(experiment_dir: Path) -> bool:
    """Return True if core simulation files are present."""
    experiment_dir = Path(experiment_dir)
    required = ("behavior.csv", "units.csv", "spikes_ground_truth.csv", "summary.json")
    return all((experiment_dir / name).exists() for name in required)


def has_decoder_comparison(experiment_dir: Path) -> bool:
    """Return True if decoder comparison outputs exist."""
    root = Path(experiment_dir) / "decoder_comparison"
    if (root / "decoder_comparison_metrics.csv").exists():
        return True
    if (root / "sorted" / "decoder_comparison_metrics.csv").exists():
        return True
    if (root / "ground_truth" / "decoder_comparison_metrics.csv").exists():
        return True
    return False


def has_realtime_decoding(experiment_dir: Path) -> bool:
    """Return True if realtime decoding outputs exist."""
    root = Path(experiment_dir) / "realtime_decoding"
    if not root.exists():
        return False
    for _path in root.rglob("decoded_realtime.csv"):
        return True
    return False


def _regenerate_probe_trajectory(experiment_dir: Path, figures_dir: Path, data) -> None:
    """Rewrite ``figures/trajectory/fig_probe_trajectory.png`` from saved anatomy."""
    import json

    import pandas as pd

    from visualization.publication_trajectory_plots import (
        generate_publication_trajectory_figures,
    )

    anatomy_path = Path(experiment_dir) / "anatomy_regions.csv"
    if not anatomy_path.exists():
        anatomy_path = Path(experiment_dir) / "trajectory" / "anatomy_regions_used.csv"
    if not anatomy_path.exists():
        return
    anatomy = pd.read_csv(anatomy_path)
    n_channels = 384
    meta: dict = {}
    summary_path = Path(experiment_dir) / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            n_channels = int(summary.get("n_channels", n_channels))
            meta = summary.get("trajectory_meta") or {}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    generate_publication_trajectory_figures(
        anatomy,
        data.units,
        figures_dir,
        n_channels=n_channels,
        meta=meta if isinstance(meta, dict) else {},
    )


def generate_simulation_figures(
    experiment_dir: Path,
    figures_dir: Path,
    rate_bin_size: float = 0.250,
) -> None:
    """Generate simulation figures into categorized subfolders under figures_dir.

    Layout (figures root contains only subfolders + output.pdf)::

        figures/
          trajectory/   behavior/   features/   neural/   sorting/   report/
          decoder_comparison/   realtime_decoding/   temporal_decoding/
          output.pdf
    """
    data = load_simulation_outputs(experiment_dir)
    figures_dir = Path(figures_dir)
    generate_behavior_plots(data, figures_dir / FIGURE_SUBDIR_BEHAVIOR)
    generate_feature_plots(data, figures_dir / FIGURE_SUBDIR_FEATURES)
    generate_neural_plots(
        data, figures_dir / FIGURE_SUBDIR_NEURAL, rate_bin_size=rate_bin_size,
    )
    generate_cell_class_plots(
        data, figures_dir / FIGURE_SUBDIR_SORTING, rate_bin_size=rate_bin_size,
    )
    generate_report_figures(
        data, figures_dir / FIGURE_SUBDIR_REPORT, rate_bin_size=rate_bin_size,
    )
    _regenerate_probe_trajectory(experiment_dir, figures_dir, data)


def generate_experiment_figures(
    experiment_dir: Path,
    figures_dir: Path | None = None,
    *,
    include_simulation: bool = True,
    include_comparison: bool = True,
    include_realtime: bool = True,
    compile_pdf: bool = False,
    rate_bin_size: float = 0.250,
) -> VizResult:
    """
    Detect available experiment outputs and generate figures.

    Never retrains decoders or recomputes comparison metrics. Only reads
    saved simulation / decoder outputs and writes under ``figures/``.
    """
    from visualization.publication_style import enable_open_axes

    enable_open_axes()

    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    result = VizResult(figures_dir=figures_dir)

    if include_simulation and has_simulation_outputs(experiment_dir):
        print(f"Generating simulation figures from {experiment_dir}...")
        generate_simulation_figures(experiment_dir, figures_dir, rate_bin_size=rate_bin_size)
        result.simulation = True

    # Publication multi-panel suite: decoding, manifolds/Isomap, closed-loop,
    # deployment, latency, optional temporal W×L (replaces legacy single-panel sprawl).
    needs_publication = False
    if include_comparison and has_decoder_comparison(experiment_dir):
        needs_publication = True
        result.comparison = True
    deploy_dir = experiment_dir / "deployment_decoder_selection"
    scores_sorted = (
        experiment_dir / "decoder_comparison" / "sorted" / "all_window_scores_sorted.csv"
    )
    if include_comparison and (deploy_dir.exists() or scores_sorted.exists()):
        needs_publication = True
    if include_realtime and has_realtime_decoding(experiment_dir):
        needs_publication = True
        result.realtime = True
    temporal_dir = experiment_dir / "decoding"
    if include_comparison and (temporal_dir / "comparison").exists():
        needs_publication = True
    lat_dir = experiment_dir / "latency_profiling"
    rt_has_latency = False
    if (experiment_dir / "realtime_decoding").exists():
        rt_has_latency = any(
            (experiment_dir / "realtime_decoding").rglob("latency/latency_by_stage.csv")
        )
    if lat_dir.exists() or rt_has_latency:
        needs_publication = True
        try:
            from visualization.latency_plots import plot_latency_outputs

            print(f"Generating latency / pipeline-timing figures from {experiment_dir}...")
            plot_latency_outputs(experiment_dir, figures_dir)
        except Exception as exc:
            print(f"  warning: latency figures skipped ({exc})")

    if needs_publication:
        try:
            from visualization.publication_decoding_plots import (
                generate_publication_decoding_figures,
            )

            print(f"Generating publication decoding/manifold figures from {experiment_dir}...")
            generate_publication_decoding_figures(experiment_dir, figures_dir)
        except Exception as exc:
            print(f"  warning: publication decoding figures skipped ({exc})")

    if compile_pdf:
        print(f"Compiling PDF under {figures_dir}...")
        result.pdf_path = compile_figures_pdf(
            figures_dir=figures_dir,
            experiment_dir=experiment_dir,
        )

    return result
