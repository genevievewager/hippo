#!/usr/bin/env python3
"""Run the hippocampal Neuropixels simulation."""

from __future__ import annotations

import argparse
from pathlib import Path

from hippo_sim.config import NEURAL_BACKENDS, SimConfig
from hippo_sim.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Hippocampal Neuropixels simulation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/run_001"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Session duration in seconds (default 600 = 10 min)",
    )
    parser.add_argument(
        "--neural-backend",
        choices=list(NEURAL_BACKENDS),
        default="custom_rate_equations",
        help="Neural activity backend (default: custom_rate_equations)",
    )
    args = parser.parse_args()

    config = SimConfig(
        output_dir=args.output,
        seed=args.seed,
        session_duration_s=args.duration,
        neural_backend=args.neural_backend,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
