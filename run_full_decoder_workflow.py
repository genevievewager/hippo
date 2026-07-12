#!/usr/bin/env python3
"""Run decoder comparison → best-decoder realtime replay → visualization."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full decoder workflow: compare → best realtime replay → visualize",
    )
    parser.add_argument("--input", type=Path, required=True, help="Simulation output directory")
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Experiment directory (writes decoder_comparison/ and realtime_decoding/)",
    )
    parser.add_argument("--compare-sources", action="store_true")
    parser.add_argument("--spike-source", choices=["sorted", "ground_truth"], default="sorted")
    parser.add_argument(
        "--decode-windows", type=float, nargs="+",
        default=[0.025, 0.050, 0.100, 0.250, 0.500, 1.000],
    )
    parser.add_argument("--max-models", choices=["quick", "full"], default="quick")
    parser.add_argument("--closed-loop-target", default="spatial_context")
    parser.add_argument(
        "--selection-policy",
        choices=["best_accuracy", "shortest_near_optimal"],
        default="shortest_near_optimal",
    )
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--compile-pdf", action="store_true")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.output)
    comparison_dir = root / "decoder_comparison"
    realtime_dir = root / "realtime_decoding"

    compare_cmd = [
        sys.executable, "run_decoder_comparison.py",
        "--input", str(args.input),
        "--output", str(comparison_dir),
        "--decode-windows", *[str(w) for w in args.decode_windows],
        "--max-models", args.max_models,
    ]
    if args.compare_sources:
        compare_cmd.append("--compare-sources")
    else:
        compare_cmd.extend(["--spike-source", args.spike_source])
    _run(compare_cmd)

    sources = ("ground_truth", "sorted") if args.compare_sources else (args.spike_source,)
    for source in sources:
        _run([
            sys.executable, "run_realtime_decoding.py",
            "--input", str(args.input),
            "--output", str(realtime_dir),
            "--comparison-dir", str(comparison_dir),
            "--use-best-decoder",
            "--closed-loop-target", args.closed_loop_target,
            "--selection-policy", args.selection_policy,
            "--spike-source", source,
        ])

    if not args.skip_visualization:
        viz_cmd = [
            sys.executable, "run_decoder_visualization.py",
            "--experiment", str(root),
        ]
        if args.compile_pdf:
            viz_cmd.append("--compile-pdf")
        _run(viz_cmd)

    print("Full decoder workflow complete.")


if __name__ == "__main__":
    main()
