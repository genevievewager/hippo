#!/usr/bin/env python3
"""Generate visualization figures from hippocampal simulation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from visualization.behavior_plots import generate_behavior_plots
from visualization.cell_class_plots import generate_cell_class_plots
from visualization.feature_plots import generate_feature_plots
from visualization.load_outputs import load_simulation_outputs
from visualization.neural_plots import generate_neural_plots
from visualization.report_figures import generate_report_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate visualization figures from simulation outputs",
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Simulation output directory (e.g. outputs/run_001)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Directory for saved figures",
    )
    parser.add_argument(
        "--rate-bin-size", type=float, default=0.250,
        help="Bin size in seconds for rate/population plots (default: 0.250)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading simulation outputs from {args.input}...")
    data = load_simulation_outputs(args.input)

    print("Generating behavioral figures...")
    generate_behavior_plots(data, output_dir)

    print("Generating feature figures...")
    generate_feature_plots(data, output_dir)

    print("Generating neural / raster figures...")
    generate_neural_plots(data, output_dir, rate_bin_size=args.rate_bin_size)

    print("Generating cell-class and anatomy figures...")
    generate_cell_class_plots(data, output_dir, rate_bin_size=args.rate_bin_size)

    print("Generating report summary...")
    generate_report_figures(data, output_dir, rate_bin_size=args.rate_bin_size)

    n_figs = len(list(output_dir.glob("*.png")))
    n_csvs = len(list(output_dir.glob("*.csv")))
    print(f"Done. Saved {n_figs} figures and {n_csvs} CSV summaries to {output_dir}")


if __name__ == "__main__":
    main()
