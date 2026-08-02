"""Full decoder workflow orchestration (comparison → best closed-loop replay)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from realtime.adaptive_windows import COARSE_DECODE_WINDOWS, windows_from_comparison_dir
from realtime.decoder_comparison import (
    ComparisonRunConfig,
    run_compare_sources,
    run_decoder_comparison,
)
from realtime.deployment_selection import (
    DEPLOYMENT_SPIKE_SOURCE,
    tag_ground_truth_outputs_as_oracle,
    write_deployment_selection_artifacts,
)
from realtime.evaluate_realtime import run_realtime_with_best_decoder
from realtime.timing import DEFAULT_UPDATE_DT_S
from realtime.workflow_profiles import (
    WorkflowProfile,
    get_profile,
)


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
    profile: str | None = None
    deployment_dir: Path | None = None
    best_realtime_json: Path | None = None


def _print_manifold_usefulness(comparison_dir: Path, sources: tuple[str, ...]) -> None:
    """Print Step-1 manifold vs counts interpretations so users see if manifolds help."""
    comparison_dir = Path(comparison_dir)
    printed_any = False
    for source in sources:
        path = comparison_dir / source / "manifold_vs_counts_summary.csv"
        if not path.exists():
            path = comparison_dir / "manifold_vs_counts_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "spike_source" in df.columns:
            df = df[df["spike_source"].astype(str) == source]
        if df.empty or "interpretation" not in df.columns:
            continue
        label = source
        if source == "ground_truth":
            label = f"{source} [oracle / non-deployable]"
        print(f"  Manifold vs counts ({label}):")
        for _, row in df.iterrows():
            target = row.get("target_name", "?")
            interp = row.get("interpretation", "")
            print(f"    {target}: {interp}")
        printed_any = True
    if not printed_any:
        print(
            "  Manifold vs counts summary not available "
            "(need both counts and manifold feature modes)."
        )


def run_full_decoder_workflow(
    input_dir: Path,
    output_dir: Path,
    *,
    profile: str = "manifolds",
    compare_sources: bool | None = None,
    spike_source: str = "sorted",
    deployment_only: bool = True,
    include_ground_truth_diagnostics: bool = False,
    decode_windows: tuple[float, ...] | None = None,
    adaptive_windows: bool | None = None,
    max_models: str | None = None,
    closed_loop_target: str = "position",
    selection_policy: str = "shortest_near_optimal",
    update_dt: float = DEFAULT_UPDATE_DT_S,
    train_frac: float = 0.70,
    n_jobs: int = -1,
    seed: int = 42,
    feature_modes: tuple[str, ...] | None = None,
    manifold_n_components: tuple[int, ...] | None = None,
    isomap_n_neighbors: tuple[int, ...] | None = None,
    isomap_latent_dim: int | None = None,
    enable_isomap_distillation: bool | None = None,
    region_ablation: bool = False,
    layer_ablation: bool = False,
    skip_visualization: bool = False,
    skip_comparison: bool = False,
    compile_pdf: bool = False,
    include_simulation_figures: bool = False,
    enable_temporal_manifold: bool | None = None,
    representations: tuple[str, ...] | None = None,
    latent_history_frames: tuple[int, ...] | None = None,
    prediction_lags: tuple[float, ...] | None = None,
    temporal_models: tuple[str, ...] | None = None,
) -> WorkflowResult:
    """
    Run decoder comparison, best-decoder closed-loop replay, and optional figures.

    Causal features always use spikes from ``[t - decode_window, t)`` only.
    Default ``update_dt`` matches the 20 Hz behavioral frame rate (0.050 s).

    **Deployment rule:** by default (``deployment_only=True``) only sorted spikes
    are used for model selection and realtime replay. Ground-truth spikes are
    optional oracle diagnostics (``include_ground_truth_diagnostics=True``) and
    never written into deployable best-model registries.

    Profiles (``manifolds`` / ``standard`` / ``quick`` / ``full``) set lean
    defaults. Explicit keyword arguments override the profile. Default
    ``manifolds`` searches all realtime-relevant embeddings (PCA family +
    classic / distilled Isomap) on a bounded W / k / nn grid. ``standard``
    evaluates counts+PCA on the full causal-window grid
    ``[0.050, 0.100, 0.250, 0.500, 1.000]`` independently per target —
    0.250 s is never hard-coded as the winner.
    """
    prof: WorkflowProfile = get_profile(profile)

    # Deployment-only is the default: sorted spikes select deployable models.
    if deployment_only and not include_ground_truth_diagnostics:
        compare_sources = False
        spike_source = DEPLOYMENT_SPIKE_SOURCE
    elif include_ground_truth_diagnostics:
        compare_sources = True
        spike_source = DEPLOYMENT_SPIKE_SOURCE
    elif compare_sources is None:
        compare_sources = prof.compare_sources

    if decode_windows is None:
        decode_windows = prof.decode_windows
    if adaptive_windows is None:
        adaptive_windows = prof.adaptive_windows
    if max_models is None:
        max_models = prof.max_models
    if feature_modes is None:
        feature_modes = prof.feature_modes
    if manifold_n_components is None:
        manifold_n_components = prof.manifold_n_components
    if isomap_n_neighbors is None:
        isomap_n_neighbors = prof.isomap_n_neighbors
    if isomap_latent_dim is None:
        isomap_latent_dim = 8
    if enable_isomap_distillation is None:
        enable_isomap_distillation = prof.enable_isomap_distillation
    if enable_isomap_distillation:
        modes = list(feature_modes)
        for mode in ("global_isomap", "global_isomap_distilled"):
            if mode not in modes:
                modes.append(mode)
        feature_modes = tuple(modes)
    if enable_temporal_manifold is None:
        enable_temporal_manifold = prof.enable_temporal_manifold
    if representations is None:
        representations = prof.representations
    if latent_history_frames is None:
        latent_history_frames = prof.latent_history_frames
    if prediction_lags is None:
        prediction_lags = prof.prediction_lags
    if temporal_models is None:
        temporal_models = prof.temporal_models

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    comparison_dir = output_dir / "decoder_comparison"
    realtime_dir = output_dir / "realtime_decoding"
    figures_dir = output_dir / "figures"
    temporal_dir = output_dir / "decoding"
    temporal_summary = None
    comparison_summary = None
    deployment_paths: dict[str, Path] = {}

    print(
        f"Step 1/4: decoder comparison (profile={prof.name}, "
        f"deployment_only={deployment_only and not include_ground_truth_diagnostics}, "
        f"windows={list(decode_windows)}, adaptive_windows={adaptive_windows}, "
        f"feature_modes={list(feature_modes)}, "
        f"manifold_k={list(manifold_n_components)}, "
        f"isomap_nn={list(isomap_n_neighbors)})..."
    )
    print(
        "  Deployment selection uses sorted spikes only. "
        "Ground-truth is oracle / non-deployable"
        + (" (enabled this run)." if include_ground_truth_diagnostics else " (skipped).")
    )

    if skip_comparison and (
        (comparison_dir / "sorted" / "best_decoder_by_target.json").exists()
        or (comparison_dir / "best_decoder_by_target.json").exists()
    ):
        print("  skip_comparison: reusing existing decoder_comparison/ outputs")
        sources = (DEPLOYMENT_SPIKE_SOURCE,)
        if include_ground_truth_diagnostics and (comparison_dir / "ground_truth").exists():
            sources = ("ground_truth", DEPLOYMENT_SPIKE_SOURCE)
    elif include_ground_truth_diagnostics or compare_sources:
        comparison_summary = run_compare_sources(
            input_dir=input_dir,
            output_dir=comparison_dir,
            decode_windows=tuple(decode_windows),
            update_dt=update_dt,
            train_frac=train_frac,
            feature_modes=tuple(feature_modes),
            manifold_n_components=tuple(manifold_n_components),
            isomap_n_neighbors=tuple(isomap_n_neighbors),
            max_models=max_models,
            n_jobs=n_jobs,
            seed=seed,
            region_ablation=region_ablation,
            layer_ablation=layer_ablation,
            adaptive_windows=adaptive_windows,
        )
        tag_ground_truth_outputs_as_oracle(comparison_dir)
        sources = ("ground_truth", DEPLOYMENT_SPIKE_SOURCE)
    else:
        comparison_summary = run_decoder_comparison(ComparisonRunConfig(
            input_dir=input_dir,
            output_dir=comparison_dir / DEPLOYMENT_SPIKE_SOURCE,
            spike_source=DEPLOYMENT_SPIKE_SOURCE,
            decode_windows=tuple(decode_windows),
            update_dt=update_dt,
            train_frac=train_frac,
            feature_modes=tuple(feature_modes),
            manifold_n_components=tuple(manifold_n_components),
            isomap_n_neighbors=tuple(isomap_n_neighbors),
            max_models=max_models,
            n_jobs=n_jobs,
            seed=seed,
            region_ablation=region_ablation,
            layer_ablation=layer_ablation,
            adaptive_windows=adaptive_windows,
        ))
        src_best = comparison_dir / DEPLOYMENT_SPIKE_SOURCE / "best_decoder_by_target.csv"
        if src_best.exists():
            import shutil
            shutil.copy2(src_best, comparison_dir / "best_decoder_by_target.csv")
            src_json = (
                comparison_dir / DEPLOYMENT_SPIKE_SOURCE / "best_decoder_by_target.json"
            )
            if src_json.exists():
                shutil.copy2(src_json, comparison_dir / "best_decoder_by_target.json")
        sources = (DEPLOYMENT_SPIKE_SOURCE,)

    _print_manifold_usefulness(comparison_dir, sources)

    try:
        deployment_paths = write_deployment_selection_artifacts(
            experiment_dir=output_dir,
            comparison_dir=comparison_dir,
            input_dir=input_dir,
            update_dt_s=update_dt,
            seed=seed,
            selection_policy=selection_policy,
        )
    except Exception as exc:
        print(f"  WARNING: could not write deployment selection artifacts: {exc}")

    if enable_temporal_manifold:
        temporal_sources = (
            sources if include_ground_truth_diagnostics else (DEPLOYMENT_SPIKE_SOURCE,)
        )
        if prof.temporal_inherit_windows:
            temporal_windows = windows_from_comparison_dir(
                comparison_dir,
                (DEPLOYMENT_SPIKE_SOURCE,),
                fallback=tuple(decode_windows) or COARSE_DECODE_WINDOWS,
            )
            print(
                "Step 2/4: temporal manifold comparison (W × L) "
                f"inheriting W={list(temporal_windows)} from sorted Step 1..."
            )
        else:
            temporal_windows = tuple(decode_windows)
            print(
                "Step 2/4: temporal manifold comparison (W × L) "
                f"using profile windows W={list(temporal_windows)}..."
            )
        from realtime.temporal.comparison import (
            TemporalComparisonConfig,
            run_temporal_manifold_comparison,
        )

        for source in temporal_sources:
            print(
                f"  temporal spike_source={source}"
                + (" [oracle]" if source == "ground_truth" else "")
                + "..."
            )
            temporal_summary = run_temporal_manifold_comparison(
                TemporalComparisonConfig(
                    input_dir=input_dir,
                    output_dir=temporal_dir / "comparison" / source,
                    spike_source=source,
                    integration_windows_s=tuple(temporal_windows),
                    latent_history_frames=tuple(latent_history_frames),
                    prediction_lags_s=tuple(prediction_lags),
                    representations=tuple(representations),
                    isomap_latent_dim=int(isomap_latent_dim),
                    isomap_n_neighbors=int(isomap_n_neighbors[0]),
                    temporal_models=tuple(temporal_models),
                    max_models=max_models if max_models == "full" else "quick",
                    n_jobs=n_jobs,
                    seed=seed,
                )
            )
    else:
        print(
            "Step 2/4: temporal manifold comparison skipped "
            "(pass --enable-temporal-manifold or --profile full with the flag)"
        )

    print("Step 3/4: best-decoder closed-loop replay (sorted / deployable only)...")
    run_realtime_with_best_decoder(
        input_dir=input_dir,
        output_dir=realtime_dir,
        comparison_dir=comparison_dir,
        closed_loop_target=closed_loop_target,
        spike_source=DEPLOYMENT_SPIKE_SOURCE,
        selection_policy=selection_policy,
        update_dt=update_dt,
        train_frac=train_frac,
        experiment_dir=output_dir,
    )

    print("  latency profiling (feature transforms + realtime stages)...")
    try:
        from realtime.latency_benchmark import run_latency_benchmark

        run_latency_benchmark(
            output_dir,
            spike_source=DEPLOYMENT_SPIKE_SOURCE,
            decode_window_s=0.250,
            update_dt_s=update_dt,
            train_frac=train_frac,
            isomap_n_components=int(isomap_latent_dim),
            isomap_n_neighbors=int(isomap_n_neighbors[0]),
            seed=seed,
        )
    except Exception as exc:
        print(f"  warning: latency benchmark skipped ({exc})")

    pdf_path = None
    if not skip_visualization:
        print("Step 4/4: visualizations...")
        from visualization.deployment_plots import plot_deployment_selection_outputs
        from visualization.experiment_viz import generate_experiment_figures
        from visualization.latency_plots import plot_latency_outputs

        viz = generate_experiment_figures(
            experiment_dir=output_dir,
            figures_dir=figures_dir,
            include_simulation=include_simulation_figures,
            include_comparison=True,
            include_realtime=True,
            compile_pdf=False,
        )
        plot_deployment_selection_outputs(output_dir, figures_dir)
        plot_latency_outputs(output_dir, figures_dir)
        if enable_temporal_manifold and (temporal_dir / "comparison").exists():
            from visualization.temporal_plots import plot_temporal_comparison_outputs
            plot_temporal_comparison_outputs(temporal_dir, figures_dir)
        if compile_pdf:
            from visualization.pdf import compile_figures_pdf

            pdf_path = compile_figures_pdf(
                figures_dir=figures_dir,
                experiment_dir=output_dir,
            )
            viz.pdf_path = pdf_path
        else:
            pdf_path = viz.pdf_path
    elif compile_pdf:
        from visualization.pdf import compile_figures_pdf
        figures_dir.mkdir(parents=True, exist_ok=True)
        from visualization.latency_plots import plot_latency_outputs

        plot_latency_outputs(output_dir, figures_dir)
        pdf_path = compile_figures_pdf(figures_dir=figures_dir)

    return WorkflowResult(
        comparison_dir=comparison_dir,
        realtime_dir=realtime_dir,
        figures_dir=None if skip_visualization else figures_dir,
        pdf_path=pdf_path,
        comparison_summary=comparison_summary,
        temporal_dir=temporal_dir if enable_temporal_manifold else None,
        temporal_summary=temporal_summary,
        profile=prof.name,
        deployment_dir=deployment_paths.get("deployment_dir"),
        best_realtime_json=deployment_paths.get("models_best_realtime_json"),
    )
