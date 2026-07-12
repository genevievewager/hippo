#!/usr/bin/env python3
"""Plot-only: generate decoder figures from saved computation outputs.

Reads CSV/JSON/prediction tables. Does not retrain models or recompute comparisons.
All figures are written under the experiment's shared figures/ folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from visualization.decoder_plots import (
    plot_decoder_comparison_outputs,
    plot_realtime_outputs,
    resolve_experiment_dir,
)
from visualization.compile_figures_pdf import compile_figures_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate decoder visualization figures under "
            "<experiment>/figures/{realtime_decoding,decoder_comparison}/"
        ),
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        default=None,
        help="Experiment directory (e.g. outputs/ratinabox_002). Figures go to <experiment>/figures/",
    )
    parser.add_argument(
        "--realtime-dir",
        type=Path,
        default=None,
        help="Realtime decoding output directory (default: <experiment>/realtime_decoding)",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Decoder comparison output directory (default: <experiment>/decoder_comparison)",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Override figures directory (default: <experiment>/figures)",
    )
    parser.add_argument(
        "--compile-pdf",
        action="store_true",
        help="Also compile all figures/**/*.png into figures/output.pdf with section titles",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    experiment_dir = resolve_experiment_dir(
        experiment_dir=args.experiment,
        realtime_dir=args.realtime_dir,
        comparison_dir=args.comparison_dir,
    )

    realtime_dir = args.realtime_dir
    comparison_dir = args.comparison_dir
    if args.experiment is not None:
        if realtime_dir is None and (experiment_dir / "realtime_decoding").exists():
            realtime_dir = experiment_dir / "realtime_decoding"
        if comparison_dir is None and (experiment_dir / "decoder_comparison").exists():
            comparison_dir = experiment_dir / "decoder_comparison"

    if realtime_dir is None and comparison_dir is None and not args.compile_pdf:
        raise SystemExit(
            "Provide --experiment with decoder outputs, or at least one of "
            "--realtime-dir / --comparison-dir (or use --compile-pdf alone)"
        )

    figures_dir = args.figures_dir or (experiment_dir / "figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    if realtime_dir is not None:
        out = plot_realtime_outputs(realtime_dir, figures_dir)
        print(f"Realtime figures written to: {out}")

    if comparison_dir is not None:
        out = plot_decoder_comparison_outputs(comparison_dir, figures_dir)
        print(f"Decoder comparison figures written to: {out}")

    if args.compile_pdf:
        pdf_path = compile_figures_pdf(figures_dir)
        print(f"Compiled PDF written to: {pdf_path}")


if __name__ == "__main__":
    main()
