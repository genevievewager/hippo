#!/usr/bin/env python3
"""Plot-only: generate decoder figures from saved computation outputs.

Reads CSV/JSON/prediction tables. Does not retrain models or recompute comparisons.
Run after run_realtime_decoding.py and/or run_decoder_comparison.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from visualization.decoder_plots import (
    plot_decoder_comparison_outputs,
    plot_realtime_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate decoder visualization figures from saved outputs",
    )
    parser.add_argument(
        "--realtime-dir",
        type=Path,
        default=None,
        help="Realtime decoding output directory (e.g. outputs/run_001/realtime_decoding)",
    )
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=None,
        help="Decoder comparison output directory (e.g. outputs/run_001/decoder_comparison)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.realtime_dir is None and args.comparison_dir is None:
        raise SystemExit("Provide at least one of --realtime-dir or --comparison-dir")

    if args.realtime_dir is not None:
        plot_realtime_outputs(args.realtime_dir)
        print(f"Realtime figures written under: {args.realtime_dir}")

    if args.comparison_dir is not None:
        plot_decoder_comparison_outputs(args.comparison_dir)
        print(f"Decoder comparison figures written under: {args.comparison_dir}")


if __name__ == "__main__":
    main()
