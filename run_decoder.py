#!/usr/bin/env python3
"""Public decoder entry point (full workflow).

Runs comparison → best-window selection → causal closed-loop replay →
optional figures/PDF. For comparison-only research grids over
``F × E × D × W × C``, use ``run_decoder_comparison.py``.

Default ``--profile manifolds`` searches realtime-relevant embeddings
(counts + global/region/layer PCA + classic/distilled Isomap) on a lean
W / k / nn grid with the quick decoder zoo. Use ``--profile standard`` for a
faster counts+PCA smoke test, or ``--profile full`` for dense research grids.
Prefer ``run_visualizations.py`` for figures (pass ``--skip-visualization``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realtime.manifold_features import ALL_FEATURE_MODES
from realtime.timing import DEFAULT_UPDATE_DT_S
from realtime.workflow import run_full_decoder_workflow
from realtime.workflow_profiles import PROFILES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full decoder workflow: compare models/windows, select the best "
            "setup, run causal closed-loop replay, optionally visualize. "
            "Default profile=manifolds (RT-relevant embeddings + Isomap "
            "teacher/student). Use --profile standard for a faster counts+PCA "
            "smoke test, or --profile full for research grids. Prefer "
            "run_visualizations.py for figures."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Simulation output directory")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Experiment directory (writes decoder_comparison/, realtime_decoding/, figures/)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES.keys()),
        default="manifolds",
        help=(
            "Workflow profile: manifolds (default; all RT embeddings + Isomap "
            "teacher/student, lean grid), standard (counts+PCA, full W), "
            "quick (coarse W smoke), full (dense research grid + full model zoo). "
            "Explicit flags below override the profile."
        ),
    )
    parser.add_argument("--compare-sources", action="store_true", default=None,
                        help="Deprecated alias: prefer --include-ground-truth-diagnostics")
    parser.add_argument("--no-compare-sources", action="store_true",
                        help="Force sorted-only even if a flag would compare sources")
    parser.add_argument(
        "--deployment-only",
        action="store_true",
        default=True,
        help="Select deployable models from sorted spikes only (default)",
    )
    parser.add_argument(
        "--no-deployment-only",
        action="store_true",
        help="Allow non-default spike_source behavior (advanced)",
    )
    parser.add_argument(
        "--include-ground-truth-diagnostics",
        action="store_true",
        help=(
            "Also run ground-truth oracle comparisons (non-deployable). "
            "Deployable models are still selected from sorted spikes only."
        ),
    )
    parser.add_argument("--spike-source", choices=["sorted", "ground_truth"], default="sorted",
                        help="Ignored for deployment selection (always sorted)")
    parser.add_argument(
        "--decode-windows", type=float, nargs="+", default=None,
        help=(
            "Neural integration windows W in seconds "
            "(default from profile; manifolds: 0.050 0.250 0.500 1.000)"
        ),
    )
    parser.add_argument(
        "--adaptive-windows", action="store_true", default=None,
        help="After coarse W pass, densify near per-target optima",
    )
    parser.add_argument(
        "--no-adaptive-windows", action="store_true",
        help="Disable adaptive W refine",
    )
    parser.add_argument("--max-models", choices=["quick", "full"], default=None)
    parser.add_argument(
        "--feature-modes", nargs="+", default=None, choices=list(ALL_FEATURE_MODES),
        help="Feature modes (default from profile; manifolds includes PCA family + Isomap)",
    )
    parser.add_argument("--manifold-n-components", type=int, default=None,
                        help="Single latent dim for manifold modes (overrides profile)")
    parser.add_argument(
        "--manifold-components-list", type=int, nargs="+", default=None,
        help="Latent-dim grid for manifold modes (overrides profile)",
    )
    parser.add_argument(
        "--isomap-neighbors", type=int, nargs="+", default=None,
        help="Isomap n_neighbors grid for global_isomap (default: 10)",
    )
    parser.add_argument(
        "--isomap-latent-dim", type=int, default=None,
        help="Latent dim for Isomap / distilled Isomap (default: 8)",
    )
    parser.add_argument(
        "--enable-isomap-distillation",
        action="store_true",
        help=(
            "Include offline global_isomap and realtime-eligible "
            "global_isomap_distilled in the feature-mode comparison"
        ),
    )
    parser.add_argument("--region-ablation", action="store_true")
    parser.add_argument("--layer-ablation", action="store_true")
    parser.add_argument(
        "--closed-loop-target",
        default="position",
        help="Primary closed-loop / realtime target (default: continuous position)",
    )
    parser.add_argument(
        "--selection-policy",
        choices=["best_accuracy", "shortest_near_optimal"],
        default="shortest_near_optimal",
    )
    parser.add_argument(
        "--update-dt", type=float, default=DEFAULT_UPDATE_DT_S,
        help="Decoder update interval (default 0.050 s = 20 Hz behavior rate)",
    )
    parser.add_argument(
        "--behavior-rate", type=float, default=20.0,
        help="Behavioral / video sampling rate in Hz (used when deriving update_dt)",
    )
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Reuse existing decoder_comparison/ outputs (skip Step 1)",
    )
    parser.add_argument(
        "--include-simulation-figures",
        action="store_true",
        help="Also regenerate simulation figures during the workflow viz step",
    )
    parser.add_argument("--compile-pdf", action="store_true")
    parser.add_argument(
        "--enable-temporal-manifold",
        action="store_true",
        default=None,
        help="Run Phase-1 W×L temporal comparison (lean grid unless --profile full)",
    )
    parser.add_argument(
        "--representations", nargs="+", default=None,
        help="Temporal manifold representations (default: pca for quick/standard)",
    )
    parser.add_argument(
        "--latent-history-frames", type=int, nargs="+", default=None,
        help="Latent history lengths L in video frames (default from profile)",
    )
    parser.add_argument(
        "--temporal-models", nargs="+", default=None,
        help="Temporal model classes (default: core 3; full adds shuffle/average controls)",
    )
    parser.add_argument(
        "--prediction-lags", type=float, nargs="+", default=None,
        help="Neural-to-behavior prediction lags tau in seconds (tau >= 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_dt = args.update_dt
    if args.behavior_rate and abs(args.update_dt - DEFAULT_UPDATE_DT_S) < 1e-12:
        update_dt = 1.0 / float(args.behavior_rate)

    if args.manifold_components_list:
        n_comps: tuple[int, ...] | None = tuple(args.manifold_components_list)
    elif args.manifold_n_components is not None:
        n_comps = (int(args.manifold_n_components),)
    else:
        n_comps = None  # use profile defaults

    compare_sources: bool | None
    if args.no_compare_sources:
        compare_sources = False
    elif args.compare_sources or args.include_ground_truth_diagnostics:
        compare_sources = True
    else:
        compare_sources = None

    adaptive_windows: bool | None
    if args.no_adaptive_windows:
        adaptive_windows = False
    elif args.adaptive_windows:
        adaptive_windows = True
    else:
        adaptive_windows = None

    deployment_only = not args.no_deployment_only
    include_gt = bool(args.include_ground_truth_diagnostics or args.compare_sources)

    feature_modes = tuple(args.feature_modes) if args.feature_modes else None
    # CLI flag forces distillation on; profile=manifolds also enables it in workflow.
    enable_isomap_distillation: bool | None = True if args.enable_isomap_distillation else None
    if args.enable_isomap_distillation:
        base = list(feature_modes) if feature_modes else ["counts", "global_pca", "region_pca"]
        for mode in ("global_isomap", "global_isomap_distilled"):
            if mode not in base:
                base.append(mode)
        feature_modes = tuple(base)

    result = run_full_decoder_workflow(
        input_dir=args.input,
        output_dir=args.output,
        profile=args.profile,
        compare_sources=compare_sources,
        spike_source=args.spike_source,
        deployment_only=deployment_only,
        include_ground_truth_diagnostics=include_gt,
        decode_windows=tuple(args.decode_windows) if args.decode_windows else None,
        adaptive_windows=adaptive_windows,
        max_models=args.max_models,
        closed_loop_target=args.closed_loop_target,
        selection_policy=args.selection_policy,
        update_dt=update_dt,
        train_frac=args.train_frac,
        n_jobs=args.n_jobs,
        seed=args.seed,
        feature_modes=feature_modes,
        manifold_n_components=n_comps,
        isomap_n_neighbors=(
            tuple(args.isomap_neighbors) if args.isomap_neighbors else None
        ),
        isomap_latent_dim=args.isomap_latent_dim,
        enable_isomap_distillation=enable_isomap_distillation,
        region_ablation=args.region_ablation,
        layer_ablation=args.layer_ablation,
        skip_visualization=args.skip_visualization,
        skip_comparison=args.skip_comparison,
        compile_pdf=args.compile_pdf,
        include_simulation_figures=args.include_simulation_figures,
        enable_temporal_manifold=True if args.enable_temporal_manifold else None,
        representations=tuple(args.representations) if args.representations else None,
        latent_history_frames=(
            tuple(args.latent_history_frames) if args.latent_history_frames else None
        ),
        prediction_lags=tuple(args.prediction_lags) if args.prediction_lags else None,
        temporal_models=tuple(args.temporal_models) if args.temporal_models else None,
    )
    print("Full decoder workflow complete.")
    if result.profile:
        print(f"  profile:    {result.profile}")
    print(f"  comparison: {result.comparison_dir}")
    if result.deployment_dir is not None:
        print(f"  deployment: {result.deployment_dir}")
    if result.best_realtime_json is not None:
        print(f"  realtime models: {result.best_realtime_json}")
    print(f"  realtime:   {result.realtime_dir}")
    if result.temporal_dir is not None:
        print(f"  temporal:   {result.temporal_dir}")
    if result.figures_dir is not None:
        print(f"  figures:    {result.figures_dir}")
    if result.pdf_path is not None:
        print(f"  pdf:        {result.pdf_path}")


if __name__ == "__main__":
    main()
