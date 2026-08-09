#!/usr/bin/env python3
"""Causal realtime closed-loop replay CLI.

Public end-to-end workflows usually go through ``run_decoder.py``. This script
exposes replay alone so the Streamlit UI and ad-hoc debugging share the same
backend as::

    from realtime.evaluate_realtime import (
        run_realtime_pipeline,
        run_realtime_with_best_decoder,
    )
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realtime.evaluate_realtime import (
    PipelineResult,
    run_compare_sources,
    run_realtime_pipeline,
    run_realtime_with_best_decoder,
)


@dataclass
class RealtimeReplayConfig:
    """Configuration for a single closed-loop replay run."""

    input_dir: Path
    output_dir: Path
    spike_source: str = "sorted"
    update_dt: float = 0.050
    decode_window: float = 0.250
    train_frac: float = 0.70
    decoder_name: str | None = None
    feature_type: str = "counts"
    manifold_n_components: int = 3
    closed_loop_target: str = "spatial_context"
    trigger_context: str | None = "wall"
    trigger_confidence: float = 0.80
    trigger_movement: str | None = None
    trigger_wall_bin: str = "near_wall"
    trigger_distance_lt_cm: float = 10.0
    trigger_speed_gt_cm_s: float = 10.0
    trigger_zone: str = "wall"
    trigger_hd_center_deg: float = 90.0
    trigger_hd_width_deg: float = 30.0
    compare_sources: bool = False
    comparison_dir: Path | None = None
    use_best_decoder: bool = False
    selection_policy: str = "shortest_near_optimal"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for realtime replay."""
    parser = argparse.ArgumentParser(
        description=(
            "Causal realtime replay on recorded / simulated spikes. "
            "Prefer run_decoder.py for end-to-end search + deployment."
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spike-source", choices=["sorted", "ground_truth"], default="sorted")
    parser.add_argument("--update-dt", type=float, default=0.050)
    parser.add_argument("--decode-window", type=float, default=0.250)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--decoder-name", default=None)
    parser.add_argument(
        "--feature-type",
        default="counts",
        help="counts | rates | global_pca | region_pca | layer_pca | global_lds | ...",
    )
    parser.add_argument(
        "--representation",
        default=None,
        help=(
            "Alias for --feature-type. Use e.g. global_lds for dynamic latent "
            "realtime replay. Offline-only methods (gpfa, global_isomap) are rejected."
        ),
    )
    parser.add_argument("--manifold-n-components", type=int, default=3)
    parser.add_argument(
        "--dynamic-latent-dims",
        type=int,
        default=None,
        help="Alias for --manifold-n-components when using dynamic latents.",
    )
    parser.add_argument("--closed-loop-target", default="spatial_context")
    parser.add_argument("--trigger-context", default="wall")
    parser.add_argument("--trigger-confidence", type=float, default=0.80)
    parser.add_argument("--trigger-movement", default="none")
    parser.add_argument("--trigger-wall-bin", default="near_wall")
    parser.add_argument("--trigger-distance-lt-cm", type=float, default=10.0)
    parser.add_argument("--trigger-speed-gt-cm-s", type=float, default=10.0)
    parser.add_argument("--trigger-zone", default="wall")
    parser.add_argument("--trigger-hd-center-deg", type=float, default=90.0)
    parser.add_argument("--trigger-hd-width-deg", type=float, default=30.0)
    parser.add_argument("--compare-sources", action="store_true")
    parser.add_argument("--comparison-dir", type=Path, default=None)
    parser.add_argument("--use-best-decoder", action="store_true")
    parser.add_argument(
        "--selection-policy",
        choices=["best_accuracy", "shortest_near_optimal"],
        default="shortest_near_optimal",
    )
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RealtimeReplayConfig:
    """Build a :class:`RealtimeReplayConfig` from parsed CLI arguments."""
    trigger_context = None if str(args.trigger_context).lower() == "none" else args.trigger_context
    trigger_movement = None if str(args.trigger_movement).lower() == "none" else args.trigger_movement
    feature_type = args.representation if args.representation else args.feature_type
    n_comp = (
        int(args.dynamic_latent_dims)
        if getattr(args, "dynamic_latent_dims", None) is not None
        else int(args.manifold_n_components)
    )
    # Organize dynamic vs static realtime outputs when using nested convention.
    output_dir = Path(args.output)
    return RealtimeReplayConfig(
        input_dir=Path(args.input),
        output_dir=output_dir,
        spike_source=args.spike_source,
        update_dt=float(args.update_dt),
        decode_window=float(args.decode_window),
        train_frac=float(args.train_frac),
        decoder_name=args.decoder_name,
        feature_type=feature_type,
        manifold_n_components=n_comp,
        closed_loop_target=args.closed_loop_target,
        trigger_context=trigger_context,
        trigger_confidence=float(args.trigger_confidence),
        trigger_movement=trigger_movement,
        trigger_wall_bin=args.trigger_wall_bin,
        trigger_distance_lt_cm=float(args.trigger_distance_lt_cm),
        trigger_speed_gt_cm_s=float(args.trigger_speed_gt_cm_s),
        trigger_zone=args.trigger_zone,
        trigger_hd_center_deg=float(args.trigger_hd_center_deg),
        trigger_hd_width_deg=float(args.trigger_hd_width_deg),
        compare_sources=bool(args.compare_sources),
        comparison_dir=Path(args.comparison_dir) if args.comparison_dir else None,
        use_best_decoder=bool(args.use_best_decoder),
        selection_policy=args.selection_policy,
    )


def run_realtime_decoding(config: RealtimeReplayConfig) -> Any:
    """Execute realtime replay from a structured config.

    Returns a :class:`PipelineResult`, a metrics dict, or a comparison DataFrame
    depending on the selected mode.
    """
    if config.use_best_decoder:
        if config.comparison_dir is None:
            raise ValueError("use_best_decoder requires comparison_dir")
        return run_realtime_with_best_decoder(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            comparison_dir=config.comparison_dir,
            closed_loop_target=config.closed_loop_target,
            spike_source=config.spike_source,
            selection_policy=config.selection_policy,
            update_dt=config.update_dt,
            train_frac=config.train_frac,
            trigger_context=config.trigger_context,
            trigger_confidence=config.trigger_confidence,
            trigger_movement=config.trigger_movement,
            trigger_wall_bin=config.trigger_wall_bin,
            trigger_distance_lt_cm=config.trigger_distance_lt_cm,
            trigger_speed_gt_cm_s=config.trigger_speed_gt_cm_s,
            trigger_zone=config.trigger_zone,
            trigger_hd_center_deg=config.trigger_hd_center_deg,
            trigger_hd_width_deg=config.trigger_hd_width_deg,
        )

    if config.compare_sources:
        return run_compare_sources(
            input_dir=config.input_dir,
            output_dir=config.output_dir,
            update_dt=config.update_dt,
            decode_window=config.decode_window,
            train_frac=config.train_frac,
            trigger_context=config.trigger_context,
            trigger_confidence=config.trigger_confidence,
            trigger_movement=config.trigger_movement,
            closed_loop_target=config.closed_loop_target,
        )

    return run_realtime_pipeline(
        input_dir=config.input_dir,
        output_dir=config.output_dir / config.spike_source,
        spike_source=config.spike_source,
        update_dt=config.update_dt,
        decode_window=config.decode_window,
        train_frac=config.train_frac,
        closed_loop_target=config.closed_loop_target,
        trigger_context=config.trigger_context,
        trigger_confidence=config.trigger_confidence,
        trigger_movement=config.trigger_movement,
        trigger_wall_bin=config.trigger_wall_bin,
        trigger_distance_lt_cm=config.trigger_distance_lt_cm,
        trigger_speed_gt_cm_s=config.trigger_speed_gt_cm_s,
        trigger_zone=config.trigger_zone,
        trigger_hd_center_deg=config.trigger_hd_center_deg,
        trigger_hd_width_deg=config.trigger_hd_width_deg,
        decoder_name=config.decoder_name,
        feature_type=config.feature_type,
        manifold_n_components=config.manifold_n_components,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = config_from_args(args)
    result = run_realtime_decoding(config)

    if config.use_best_decoder:
        assert isinstance(result, PipelineResult)
        print("Best-decoder realtime replay complete.")
        if result.selected_config:
            for key in (
                "selected_decoder_name", "selected_decode_window_s",
                "feature_type", "manifold_type", "manifold_n_components",
                "manifold_transform_path",
            ):
                if key in result.selected_config:
                    print(f"  {key}: {result.selected_config[key]}")
        for key, value in result.metrics.items():
            print(f"  {key}: {value}")
        return

    if config.compare_sources:
        print("Comparison complete.")
        print(result.to_string(index=False))
        return

    assert isinstance(result, PipelineResult)
    print("Real-time decoding complete.")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
