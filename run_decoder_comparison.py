#!/usr/bin/env python3
"""Model and causal window optimization across many decoder configurations.

Use before closed-loop replay to choose the best decoder model and decode_window.
For single-setup replay with closed-loop triggers, use run_realtime_decoding.py.
For figures, use run_decoder_visualization.py after computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realtime.decoder_comparison import (
    DEFAULT_DECODE_WINDOWS,
    ComparisonRunConfig,
    run_compare_sources,
    run_decoder_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare decoder models and causal spike-history windows for "
            "real-time hippocampal decoding"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Simulation output directory (e.g. outputs/run_001)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for decoder comparison outputs",
    )
    parser.add_argument(
        "--spike-source",
        choices=["sorted", "ground_truth"],
        default="sorted",
        help="Spike table to decode from (default: sorted)",
    )
    parser.add_argument(
        "--compare-sources",
        action="store_true",
        help="Run comparison separately for ground_truth and sorted spikes",
    )
    parser.add_argument(
        "--decode-windows",
        type=float,
        nargs="+",
        default=list(DEFAULT_DECODE_WINDOWS),
        help="Causal spike-history windows in seconds",
    )
    parser.add_argument(
        "--update-dt",
        type=float,
        default=0.025,
        help="Decoder update interval in seconds (default: 0.025)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of session used for training (default: 0.70)",
    )
    parser.add_argument(
        "--feature-modes",
        choices=["counts", "rates"],
        nargs="+",
        default=["counts"],
        help="Spike feature modes: raw counts or rate-normalized counts",
    )
    parser.add_argument(
        "--max-models",
        choices=["quick", "full"],
        default="quick",
        help="Model set to evaluate (default: quick)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for applicable sklearn models (default: -1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stochastic models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decode_windows = tuple(sorted(set(args.decode_windows)))
    feature_modes = tuple(dict.fromkeys(args.feature_modes))

    if args.compare_sources:
        comparison = run_compare_sources(
            input_dir=args.input,
            output_dir=args.output,
            decode_windows=decode_windows,
            update_dt=args.update_dt,
            train_frac=args.train_frac,
            feature_modes=feature_modes,
            max_models=args.max_models,
            n_jobs=args.n_jobs,
            seed=args.seed,
        )
        print("Source comparison complete.")
        if not comparison.empty:
            print(comparison.to_string(index=False))
        return

    metrics_df = run_decoder_comparison(ComparisonRunConfig(
        input_dir=args.input,
        output_dir=args.output,
        spike_source=args.spike_source,
        decode_windows=decode_windows,
        update_dt=args.update_dt,
        train_frac=args.train_frac,
        feature_modes=feature_modes,
        max_models=args.max_models,
        n_jobs=args.n_jobs,
        seed=args.seed,
    ))
    print("Decoder comparison complete.")
    print(f"  Evaluations: {len(metrics_df)}")
    print(f"  Outputs: {args.output}")


if __name__ == "__main__":
    main()
