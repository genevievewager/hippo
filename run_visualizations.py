#!/usr/bin/env python3
"""Public visualization entry point (only recommended plotting script).

Generates all available figures (simulation, decoder comparison, realtime)
under ``<experiment>/figures/``. Reads saved outputs only — never retrains
decoders or recomputes comparison metrics.

Example::

    python run_visualizations.py --experiment outputs/ratinabox_002 --all --compile-pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from visualization.experiment_viz import generate_experiment_figures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all available experiment visualizations under figures/. "
            "Reads saved outputs only; does not retrain decoders."
        ),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Experiment directory (e.g. outputs/ratinabox_002). Default figures: <experiment>/figures",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Alias for --experiment (backward compatible)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Figures directory (default: <experiment>/figures)",
    )
    parser.add_argument(
        "--include-simulation",
        action="store_true",
        help="Generate simulation behavior/neural/probe figures",
    )
    parser.add_argument(
        "--include-decoder",
        action="store_true",
        help="Generate decoder comparison and realtime figures when available",
    )
    parser.add_argument(
        "--include-comparison",
        action="store_true",
        help="Generate decoder comparison figures when available",
    )
    parser.add_argument(
        "--include-realtime",
        action="store_true",
        help="Generate realtime / closed-loop figures when available",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate every available figure type (default when only --experiment is set)",
    )
    parser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="Compile figures/**/*.png into figures/output.pdf",
    )
    parser.add_argument(
        "--rate-bin-size",
        type=float,
        default=0.250,
        help="Bin size in seconds for rate/population plots (default: 0.250)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment or args.input
    if experiment_dir is None:
        raise SystemExit("Provide --experiment or --input")

    experiment_dir = Path(experiment_dir)
    figures_dir = Path(args.output) if args.output is not None else experiment_dir / "figures"

    # Default: if no include flags, behave like --all (user-friendly)
    any_flag = (
        args.all
        or args.include_simulation
        or args.include_decoder
        or args.include_comparison
        or args.include_realtime
    )
    if not any_flag or args.all:
        include_simulation = True
        include_comparison = True
        include_realtime = True
    else:
        include_simulation = args.include_simulation
        include_comparison = args.include_comparison or args.include_decoder
        include_realtime = args.include_realtime or args.include_decoder

    result = generate_experiment_figures(
        experiment_dir=experiment_dir,
        figures_dir=figures_dir,
        include_simulation=include_simulation,
        include_comparison=include_comparison,
        include_realtime=include_realtime,
        compile_pdf=args.compile_pdf,
        rate_bin_size=args.rate_bin_size,
    )

    parts = []
    if result.simulation:
        parts.append("simulation")
    if result.comparison:
        parts.append("decoder_comparison")
    if result.realtime:
        parts.append("realtime_decoding")
    if not parts:
        print(f"No matching outputs found under {experiment_dir}")
    else:
        print(f"Wrote figures ({', '.join(parts)}) to {result.figures_dir}")
    if result.pdf_path is not None:
        print(f"Compiled PDF: {result.pdf_path}")


if __name__ == "__main__":
    main()
