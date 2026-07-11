#!/usr/bin/env python3
"""Single closed-loop replay: one decoder setup, causal session replay, trigger evaluation.

Use when decoder settings (spike_source, decode_window, update_dt) are already chosen.
For model/window optimization, use run_decoder_comparison.py instead.
For figures, use run_decoder_visualization.py after computation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realtime.evaluate_realtime import run_compare_sources, run_realtime_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal real-time decoding from hippocampal Neuropixels spikes",
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
        help="Directory for real-time decoding outputs",
    )
    parser.add_argument(
        "--spike-source",
        choices=["sorted", "ground_truth"],
        default="sorted",
        help="Spike table to decode from (default: sorted)",
    )
    parser.add_argument(
        "--update-dt",
        type=float,
        default=0.025,
        help="Decoder update interval in seconds (default: 0.025 = 40 Hz)",
    )
    parser.add_argument(
        "--decode-window",
        type=float,
        default=0.250,
        help="Causal spike-history window in seconds (default: 0.250)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Fraction of session used for training (default: 0.70)",
    )
    parser.add_argument(
        "--trigger-context",
        default="wall",
        help="Spatial context that triggers closed-loop events (default: wall)",
    )
    parser.add_argument(
        "--trigger-confidence",
        type=float,
        default=0.80,
        help="Minimum decoder confidence for context trigger (default: 0.80)",
    )
    parser.add_argument(
        "--trigger-movement",
        default="none",
        help="Movement state trigger (still/slow/fast) or 'none' (default: none)",
    )
    parser.add_argument(
        "--compare-sources",
        action="store_true",
        help="Run pipeline for both ground_truth and sorted spikes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trigger_context = None if args.trigger_context.lower() == "none" else args.trigger_context
    trigger_movement = None if args.trigger_movement.lower() == "none" else args.trigger_movement

    if args.compare_sources:
        comparison = run_compare_sources(
            input_dir=args.input,
            output_dir=args.output,
            update_dt=args.update_dt,
            decode_window=args.decode_window,
            train_frac=args.train_frac,
            trigger_context=trigger_context,
            trigger_confidence=args.trigger_confidence,
            trigger_movement=trigger_movement,
        )
        print("Comparison complete.")
        print(comparison.to_string(index=False))
        return

    metrics = run_realtime_pipeline(
        input_dir=args.input,
        output_dir=args.output / args.spike_source,
        spike_source=args.spike_source,
        update_dt=args.update_dt,
        decode_window=args.decode_window,
        train_frac=args.train_frac,
        trigger_context=trigger_context,
        trigger_confidence=args.trigger_confidence,
        trigger_movement=trigger_movement,
    )

    print("Real-time decoding complete.")
    for key, value in metrics.metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
