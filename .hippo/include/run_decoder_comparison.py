#!/usr/bin/env python3
"""Developer utility: decoder comparison only (static + manifold feature modes).

Most users should run the public end-to-end workflow instead::

    python run_full_decoder_workflow.py --input ... --output ...

This script wraps ``realtime.decoder_comparison`` for debugging a single
comparison step without closed-loop replay.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realtime.decoder_comparison import (
    DEFAULT_DECODE_WINDOWS,
    DEFAULT_MANIFOLD_N_COMPONENTS,
    ComparisonRunConfig,
    run_compare_sources,
    run_decoder_comparison,
)
from realtime.manifold_features import ALL_FEATURE_MODES, QUICK_FEATURE_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "[Developer] Compare decoder models and causal windows. "
            "Prefer run_full_decoder_workflow.py for end-to-end runs."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spike-source", choices=["sorted", "ground_truth"], default="sorted")
    parser.add_argument("--compare-sources", action="store_true")
    parser.add_argument(
        "--decode-windows", type=float, nargs="+", default=list(DEFAULT_DECODE_WINDOWS),
    )
    parser.add_argument("--update-dt", type=float, default=0.050)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument(
        "--feature-modes", nargs="+", default=None, choices=list(ALL_FEATURE_MODES),
    )
    parser.add_argument("--manifold-n-components", type=int, default=3)
    parser.add_argument("--manifold-components-list", type=int, nargs="+", default=None)
    parser.add_argument("--max-models", choices=["quick", "full"], default="quick")
    parser.add_argument("--region-ablation", action="store_true")
    parser.add_argument("--layer-ablation", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_modes = tuple(args.feature_modes) if args.feature_modes else QUICK_FEATURE_MODES
    n_comps = (
        tuple(args.manifold_components_list)
        if args.manifold_components_list
        else (int(args.manifold_n_components),)
    )
    if args.compare_sources:
        run_compare_sources(
            input_dir=args.input,
            output_dir=args.output,
            decode_windows=tuple(args.decode_windows),
            update_dt=args.update_dt,
            train_frac=args.train_frac,
            feature_modes=feature_modes,
            manifold_n_components=n_comps,
            max_models=args.max_models,
            n_jobs=args.n_jobs,
            seed=args.seed,
            region_ablation=args.region_ablation,
            layer_ablation=args.layer_ablation,
        )
    else:
        run_decoder_comparison(ComparisonRunConfig(
            input_dir=args.input,
            output_dir=args.output,
            spike_source=args.spike_source,
            decode_windows=tuple(args.decode_windows),
            update_dt=args.update_dt,
            train_frac=args.train_frac,
            feature_modes=feature_modes,
            manifold_n_components=n_comps,
            max_models=args.max_models,
            n_jobs=args.n_jobs,
            seed=args.seed,
            region_ablation=args.region_ablation,
            layer_ablation=args.layer_ablation,
        ))
    print(f"[developer] Decoder comparison written to {args.output}")


if __name__ == "__main__":
    main()
