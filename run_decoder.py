#!/usr/bin/env python3
"""Public decoder entry point (only recommended decoding script).

Runs comparison → best-window selection → causal closed-loop replay →
optional figures/PDF. Prefer this over ``run_decoder_comparison.py`` or
``run_realtime_decoding.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realtime.decoder_comparison import (
    DEFAULT_DECODE_WINDOWS,
    DEFAULT_MANIFOLD_N_COMPONENTS,
)
from realtime.manifold_features import ALL_FEATURE_MODES, QUICK_FEATURE_MODES
from realtime.timing import (
    DEFAULT_LATENT_HISTORY_FRAMES,
    DEFAULT_UPDATE_DT_S,
)
from realtime.workflow import run_full_decoder_workflow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Full decoder workflow: compare models/windows, select the best "
            "setup, run causal closed-loop replay, optionally visualize. "
            "Default update interval matches 20 Hz behavior (0.050 s)."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Simulation output directory")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Experiment directory (writes decoder_comparison/, realtime_decoding/, figures/)",
    )
    parser.add_argument("--compare-sources", action="store_true")
    parser.add_argument("--spike-source", choices=["sorted", "ground_truth"], default="sorted")
    parser.add_argument(
        "--decode-windows", type=float, nargs="+",
        default=list(DEFAULT_DECODE_WINDOWS),
        help="Neural integration windows W in seconds",
    )
    parser.add_argument("--max-models", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--feature-modes", nargs="+", default=None, choices=list(ALL_FEATURE_MODES),
        help="Feature modes (default quick: counts global_pca region_pca)",
    )
    parser.add_argument("--manifold-n-components", type=int, default=3)
    parser.add_argument(
        "--manifold-components-list", type=int, nargs="+", default=None,
    )
    parser.add_argument("--region-ablation", action="store_true")
    parser.add_argument("--layer-ablation", action="store_true")
    parser.add_argument("--closed-loop-target", default="spatial_context")
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
        "--include-simulation-figures",
        action="store_true",
        help="Also regenerate simulation figures during the workflow viz step",
    )
    parser.add_argument("--compile-pdf", action="store_true")
    parser.add_argument(
        "--enable-temporal-manifold",
        action="store_true",
        help="Run Phase-1 joint W×L temporal manifold comparison under decoding/",
    )
    parser.add_argument(
        "--representations", nargs="+", default=["raw", "pca"],
        help="Manifold representations for temporal comparison",
    )
    parser.add_argument(
        "--latent-history-frames", type=int, nargs="+",
        default=list(DEFAULT_LATENT_HISTORY_FRAMES),
        help="Latent history lengths L in video frames",
    )
    parser.add_argument(
        "--temporal-models", nargs="+",
        default=[
            "raw_static", "static_latent", "flattened_history",
            "shuffled_sequence", "averaged_history",
        ],
        help="Temporal model classes (Phase 1; GRU/TCN in later phases)",
    )
    parser.add_argument(
        "--prediction-lags", type=float, nargs="+",
        default=[0.0],
        help="Neural-to-behavior prediction lags tau in seconds (tau >= 0)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_dt = args.update_dt
    if args.behavior_rate and abs(args.update_dt - DEFAULT_UPDATE_DT_S) < 1e-12:
        update_dt = 1.0 / float(args.behavior_rate)

    if args.feature_modes is None:
        feature_modes = QUICK_FEATURE_MODES
    else:
        feature_modes = tuple(args.feature_modes)
    if args.manifold_components_list:
        n_comps = tuple(args.manifold_components_list)
    else:
        n_comps = (int(args.manifold_n_components),)

    result = run_full_decoder_workflow(
        input_dir=args.input,
        output_dir=args.output,
        compare_sources=args.compare_sources,
        spike_source=args.spike_source,
        decode_windows=tuple(args.decode_windows),
        max_models=args.max_models,
        closed_loop_target=args.closed_loop_target,
        selection_policy=args.selection_policy,
        update_dt=update_dt,
        train_frac=args.train_frac,
        n_jobs=args.n_jobs,
        seed=args.seed,
        feature_modes=feature_modes,
        manifold_n_components=n_comps,
        region_ablation=args.region_ablation,
        layer_ablation=args.layer_ablation,
        skip_visualization=args.skip_visualization,
        compile_pdf=args.compile_pdf,
        include_simulation_figures=args.include_simulation_figures,
        enable_temporal_manifold=args.enable_temporal_manifold,
        representations=tuple(args.representations),
        latent_history_frames=tuple(args.latent_history_frames),
        prediction_lags=tuple(args.prediction_lags),
        temporal_models=tuple(args.temporal_models),
    )
    print("Full decoder workflow complete.")
    print(f"  comparison: {result.comparison_dir}")
    print(f"  realtime:   {result.realtime_dir}")
    if result.temporal_dir is not None:
        print(f"  temporal:   {result.temporal_dir}")
    if result.figures_dir is not None:
        print(f"  figures:    {result.figures_dir}")
    if result.pdf_path is not None:
        print(f"  pdf:        {result.pdf_path}")


if __name__ == "__main__":
    main()
