#!/usr/bin/env python3
"""Run the hippocampal Neuropixels simulation (RatInABox neural + behavior)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hippo.anatomy.trajectory_config import (
    DEFAULT_TRAJECTORY_CONFIG,
    list_trajectory_configs,
    resolve_trajectory_config,
)
from hippo_sim.config import SimConfig
from hippo_sim.pipeline import apply_trajectory_to_config, run_pipeline


def main() -> None:
    available = ", ".join(r["name"] for r in list_trajectory_configs()) or "(none)"
    parser = argparse.ArgumentParser(
        description="Hippocampal Neuropixels simulation (RatInABox rates + behavior)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/run_001"),
        help="Trial output directory (trajectory coords are snapshotted under <output>/trajectory/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--duration",
        type=float,
        default=600.0,
        help="Session duration in seconds (default 600 = 10 min)",
    )
    parser.add_argument(
        "--trajectory",
        "--trajectory-config",
        "--trajectory-name",
        dest="trajectory_config",
        type=str,
        default=None,
        help=(
            "Active lab insertion: config name under configs/trajectories/ "
            f"(e.g. lab_npx2_default) or a YAML path. Available: {available}. "
            f"Default: {DEFAULT_TRAJECTORY_CONFIG}."
        ),
    )
    parser.add_argument(
        "--list-trajectories",
        action="store_true",
        help="List selectable trajectory configs and exit",
    )
    parser.add_argument(
        "--trajectory-export",
        type=Path,
        default=None,
        help="Neuropixels Trajectory Explorer export (overrides approximate region table)",
    )
    parser.add_argument(
        "--anatomy-regions-file",
        type=Path,
        default=None,
        help="Override anatomy region-depth CSV from the trajectory config",
    )
    parser.add_argument(
        "--cell-capture-config",
        type=Path,
        default=None,
        help="Override cell-capture YAML from the trajectory config",
    )
    parser.add_argument(
        "--fallback-schematic-anatomy",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Fall back to schematic CA1–MEC geometry if trajectory anatomy is missing",
    )
    parser.add_argument(
        "--include-non-hippocampal-regions",
        action="store_true",
        help="Include visual cortex (etc.) units; default excludes them",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="Ignore trajectory configs and use schematic hippocampal geometry",
    )
    args = parser.parse_args()

    if args.list_trajectories:
        rows = list_trajectory_configs(include_templates=True)
        if not rows:
            print("No trajectory configs found under configs/trajectories/")
            return
        for row in rows:
            mark = " (default)" if row["is_default"] else ""
            tmpl = " [template]" if row["is_template"] else ""
            print(f"{row['name']}{mark}{tmpl}\t{row['path']}")
        return

    config = SimConfig(
        output_dir=args.output,
        seed=args.seed,
        session_duration_s=args.duration,
    )

    if not args.no_trajectory:
        traj_cfg = args.trajectory_config
        if traj_cfg is None and DEFAULT_TRAJECTORY_CONFIG.exists():
            traj_cfg = str(DEFAULT_TRAJECTORY_CONFIG)
        if traj_cfg is not None or args.trajectory_export is not None:
            resolved = (
                resolve_trajectory_config(traj_cfg) if traj_cfg is not None else None
            )
            apply_trajectory_to_config(
                config,
                trajectory_config=resolved,
                trajectory_export=args.trajectory_export,
                anatomy_regions_file=args.anatomy_regions_file,
                cell_capture_config=args.cell_capture_config,
                fallback_schematic=args.fallback_schematic_anatomy,
                include_non_hippocampal_regions=args.include_non_hippocampal_regions,
            )

    run_pipeline(config)


if __name__ == "__main__":
    main()
