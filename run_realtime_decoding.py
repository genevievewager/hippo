#!/usr/bin/env python3
"""Single closed-loop replay: one decoder setup, causal session replay, trigger evaluation.

Manual mode: specify --decode-window / --decoder-name yourself.
Automatic mode: --use-best-decoder reads best_decoder_by_target.csv from Step 2A.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from realtime.evaluate_realtime import (
    run_compare_sources,
    run_realtime_pipeline,
    run_realtime_with_best_decoder,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal real-time decoding from hippocampal Neuropixels spikes",
    )
    parser.add_argument("--input", type=Path, required=True, help="Simulation output directory")
    parser.add_argument("--output", type=Path, required=True, help="Realtime decoding output directory")
    parser.add_argument(
        "--spike-source", choices=["sorted", "ground_truth"], default="sorted",
    )
    parser.add_argument("--update-dt", type=float, default=0.025)
    parser.add_argument("--decode-window", type=float, default=0.250)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument(
        "--decoder-name", default=None,
        help="Optional primary decoder for closed-loop target (manual mode)",
    )
    parser.add_argument(
        "--feature-type", choices=["counts", "rates"], default="counts",
    )
    parser.add_argument(
        "--closed-loop-target",
        default="spatial_context",
        choices=[
            "position", "spatial_context", "distance_to_wall", "wall_distance_bin",
            "speed", "movement_state", "head_direction", "acceleration",
        ],
        help="Latent variable used for closed-loop triggering",
    )
    parser.add_argument("--trigger-context", default="wall")
    parser.add_argument("--trigger-confidence", type=float, default=0.80)
    parser.add_argument("--trigger-movement", default="none")
    parser.add_argument("--trigger-wall-bin", default="near_wall")
    parser.add_argument("--trigger-distance-lt-cm", type=float, default=10.0)
    parser.add_argument("--trigger-speed-gt-cm-s", type=float, default=10.0)
    parser.add_argument("--trigger-zone", default="wall")
    parser.add_argument("--trigger-hd-center-deg", type=float, default=90.0)
    parser.add_argument("--trigger-hd-width-deg", type=float, default=30.0)
    parser.add_argument(
        "--compare-sources", action="store_true",
        help="Run pipeline for both ground_truth and sorted spikes (manual mode)",
    )
    parser.add_argument(
        "--comparison-dir", type=Path, default=None,
        help="Decoder comparison output directory (required with --use-best-decoder)",
    )
    parser.add_argument(
        "--use-best-decoder", action="store_true",
        help="Automatically select decoder/window from comparison results",
    )
    parser.add_argument(
        "--selection-policy",
        choices=["best_accuracy", "shortest_near_optimal"],
        default="shortest_near_optimal",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trigger_context = None if args.trigger_context.lower() == "none" else args.trigger_context
    trigger_movement = None if args.trigger_movement.lower() == "none" else args.trigger_movement

    if args.use_best_decoder:
        if args.comparison_dir is None:
            raise SystemExit("--use-best-decoder requires --comparison-dir")
        if not args.closed_loop_target:
            raise SystemExit("--use-best-decoder requires --closed-loop-target")

        result = run_realtime_with_best_decoder(
            input_dir=args.input,
            output_dir=args.output,
            comparison_dir=args.comparison_dir,
            closed_loop_target=args.closed_loop_target,
            spike_source=args.spike_source,
            selection_policy=args.selection_policy,
            update_dt=args.update_dt,
            train_frac=args.train_frac,
            trigger_context=trigger_context,
            trigger_confidence=args.trigger_confidence,
            trigger_movement=trigger_movement,
            trigger_wall_bin=args.trigger_wall_bin,
            trigger_distance_lt_cm=args.trigger_distance_lt_cm,
            trigger_speed_gt_cm_s=args.trigger_speed_gt_cm_s,
            trigger_zone=args.trigger_zone,
            trigger_hd_center_deg=args.trigger_hd_center_deg,
            trigger_hd_width_deg=args.trigger_hd_width_deg,
        )
        print("Best-decoder realtime replay complete.")
        if result.selected_config:
            cfg = result.selected_config
            print(f"  selected_decoder_name: {cfg['selected_decoder_name']}")
            print(f"  selected_decode_window_s: {cfg['selected_decode_window_s']}")
            print(f"  selection_policy: {cfg['selection_policy']}")
            print(f"  from_file: {cfg['from_file']}")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
        return

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
            closed_loop_target=args.closed_loop_target,
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
        closed_loop_target=args.closed_loop_target,
        trigger_context=trigger_context,
        trigger_confidence=args.trigger_confidence,
        trigger_movement=trigger_movement,
        trigger_wall_bin=args.trigger_wall_bin,
        trigger_distance_lt_cm=args.trigger_distance_lt_cm,
        trigger_speed_gt_cm_s=args.trigger_speed_gt_cm_s,
        trigger_zone=args.trigger_zone,
        trigger_hd_center_deg=args.trigger_hd_center_deg,
        trigger_hd_width_deg=args.trigger_hd_width_deg,
        decoder_name=args.decoder_name,
        feature_type=args.feature_type,
    )

    print("Real-time decoding complete.")
    for key, value in metrics.metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
