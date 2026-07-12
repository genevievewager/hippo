#!/usr/bin/env python3
"""Compile all PNGs under an experiment figures/ folder into one sectioned PDF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from visualization.compile_figures_pdf import compile_figures_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile figures/**/*.png into a single PDF with section title pages "
            "for simulation, realtime decoding, and decoder comparison results"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Experiment directory (uses <experiment>/figures)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Figures directory (default: <experiment>/figures)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <figures-dir>/output.pdf)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.figures_dir is None and args.experiment is None:
        raise SystemExit("Provide --experiment or --figures-dir")

    figures_dir = args.figures_dir or (Path(args.experiment) / "figures")
    output_pdf = compile_figures_pdf(figures_dir, output_pdf=args.output)
    print(f"Wrote PDF: {output_pdf}")


if __name__ == "__main__":
    main()
