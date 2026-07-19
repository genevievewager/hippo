"""Full decoder workflow orchestration (comparison → best closed-loop replay)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realtime.decoder_comparison import (
    DEFAULT_DECODE_WINDOWS,
    DEFAULT_MANIFOLD_N_COMPONENTS,
    ComparisonRunConfig,
    run_compare_sources,
    run_decoder_comparison,
)
from realtime.evaluate_realtime import run_realtime_with_best_decoder
from realtime.manifold_features import QUICK_FEATURE_MODES
from realtime.timing import DEFAULT_UPDATE_DT_S


@dataclass
class WorkflowResult:
    """Paths and summaries produced by the full decoder workflow."""

    comparison_dir: Path
    realtime_dir: Path
    figures_dir: Path | None = None
    pdf_path: Path | None = None
    comparison_summary: Any = None
    temporal_dir: Path | None = None
    temporal_summary: Any = None


def run_full_decoder_workflow(
    input_dir: Path,
    output_dir: Path,
    *,
    compare_sources: bool = False,
    spike_source: str = "sorted",
    decode_windows: tuple[float, ...] = DEFAULT_DECODE_WINDOWS,
    max_models: str = "quick",
    closed_loop_target: str = "spatial_context",
    selection_policy: str = "shortest_near_optimal",
    update_dt: float = DEFAULT_UPDATE_DT_S,
    train_frac: float = 0.70,
    n_jobs: int = -1,
    seed: int = 42,
    feature_modes: tuple[str, ...] = QUICK_FEATURE_MODES,
    manifold_n_components: tuple[int, ...] = DEFAULT_MANIFOLD_N_COMPONENTS,
    region_ablation: bool = False,
    layer_ablation: bool = False,
    skip_visualization: bool = False,
    compile_pdf: bool = False,
    include_simulation_figures: bool = False,
    enable_temporal_manifold: bool = False,
    representations: tuple[str, ...] = ("raw", "pca"),
    latent_history_frames: tuple[int, ...] = (1, 2, 5, 10, 20),
    prediction_lags: tuple[float, ...] = (0.0,),
    temporal_models: tuple[str, ...] = (
        "raw_static",
        "static_latent",
        "flattened_history",
        "shuffled_sequence",
        "averaged_history",
    ),
) -> WorkflowResult:
    """
    Run decoder comparison, best-decoder closed-loop replay, and optional figures.

    Causal features always use spikes from ``[t - decode_window, t)`` only.
    Default ``update_dt`` matches the 20 Hz behavioral frame rate (0.050 s).

    When ``enable_temporal_manifold`` is True, also run Phase-1 joint
    integration-window × latent-history comparison under ``decoding/``.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    comparison_dir = output_dir / "decoder_comparison"
    realtime_dir = output_dir / "realtime_decoding"
    figures_dir = output_dir / "figures"
    temporal_dir = output_dir / "decoding"
    temporal_summary = None

    print("Step 1/4: decoder comparison...")
    if compare_sources:
        comparison_summary = run_compare_sources(
            input_dir=input_dir,
            output_dir=comparison_dir,
            decode_windows=tuple(decode_windows),
            update_dt=update_dt,
            train_frac=train_frac,
            feature_modes=tuple(feature_modes),
            manifold_n_components=tuple(manifold_n_components),
            max_models=max_models,
            n_jobs=n_jobs,
            seed=seed,
            region_ablation=region_ablation,
            layer_ablation=layer_ablation,
        )
        sources = ("ground_truth", "sorted")
    else:
        comparison_summary = run_decoder_comparison(ComparisonRunConfig(
            input_dir=input_dir,
            output_dir=comparison_dir,
            spike_source=spike_source,
            decode_windows=tuple(decode_windows),
            update_dt=update_dt,
            train_frac=train_frac,
            feature_modes=tuple(feature_modes),
            manifold_n_components=tuple(manifold_n_components),
            max_models=max_models,
            n_jobs=n_jobs,
            seed=seed,
            region_ablation=region_ablation,
            layer_ablation=layer_ablation,
        ))
        sources = (spike_source,)

    if enable_temporal_manifold:
        print("Step 2/4: temporal manifold comparison (W × L)...")
        from realtime.temporal.comparison import (
            TemporalComparisonConfig,
            run_temporal_manifold_comparison,
        )

        for source in sources:
            temporal_summary = run_temporal_manifold_comparison(
                TemporalComparisonConfig(
                    input_dir=input_dir,
                    output_dir=temporal_dir / "comparison" / source,
                    spike_source=source,
                    integration_windows_s=tuple(decode_windows),
                    latent_history_frames=tuple(latent_history_frames),
                    prediction_lags_s=tuple(prediction_lags),
                    representations=tuple(representations),
                    temporal_models=tuple(temporal_models),
                    max_models=max_models,
                    n_jobs=n_jobs,
                    seed=seed,
                )
            )
    else:
        print("Step 2/4: temporal manifold comparison skipped "
              "(pass --enable-temporal-manifold)")

    print("Step 3/4: best-decoder closed-loop replay...")
    for source in sources:
        run_realtime_with_best_decoder(
            input_dir=input_dir,
            output_dir=realtime_dir,
            comparison_dir=comparison_dir,
            closed_loop_target=closed_loop_target,
            spike_source=source,
            selection_policy=selection_policy,
            update_dt=update_dt,
            train_frac=train_frac,
        )

    pdf_path = None
    if not skip_visualization:
        print("Step 4/4: visualizations...")
        from visualization.experiment_viz import generate_experiment_figures

        viz = generate_experiment_figures(
            experiment_dir=output_dir,
            figures_dir=figures_dir,
            include_simulation=include_simulation_figures,
            include_comparison=True,
            include_realtime=True,
            compile_pdf=compile_pdf,
        )
        if enable_temporal_manifold and (temporal_dir / "comparison").exists():
            from visualization.temporal_plots import plot_temporal_comparison_outputs
            plot_temporal_comparison_outputs(temporal_dir, figures_dir)
        pdf_path = viz.pdf_path
    elif compile_pdf:
        from visualization.pdf import compile_figures_pdf
        figures_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = compile_figures_pdf(figures_dir=figures_dir)

    return WorkflowResult(
        comparison_dir=comparison_dir,
        realtime_dir=realtime_dir,
        figures_dir=None if skip_visualization else figures_dir,
        pdf_path=pdf_path,
        comparison_summary=comparison_summary,
        temporal_dir=temporal_dir if enable_temporal_manifold else None,
        temporal_summary=temporal_summary,
    )
