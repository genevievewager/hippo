#!/usr/bin/env python3
"""Advanced staged workflow (simulate, manifolds, partitions, decode).

Most users should use the three public scripts instead::

    python run_simulation.py ...
    python run_full_decoder_workflow.py ...
    python run_visualizations.py --experiment ... --all --compile-pdf

This entry point is for multi-stage manifold / partition experiments.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hippo.dataset import load_manifold_dataset
from hippo.partitions import apply_partitions, available_partitions
from hippo.unit_metadata import metadata_availability_table, normalize_unit_metadata
from realtime.manifold_features import QUICK_FEATURE_MODES
from realtime.timing import DEFAULT_UPDATE_DT_S
from realtime.workflow import run_full_decoder_workflow


STAGES = (
    "simulate",
    "decode-static",
    "fit-manifolds",
    "compare-partitions",
    "decode-temporal",
    "analyze-communication",  # Phase 5 — stub
    "replay-realtime",
    "visualize",
    "report",
    "all",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Hippocampal simulation + manifold decoding workflow (staged)",
    )
    p.add_argument(
        "--stage",
        default="decode-static",
        choices=list(STAGES),
        help="Workflow stage to run",
    )
    p.add_argument("--input", type=Path, default=None, help="Existing simulation directory")
    p.add_argument("--output", type=Path, required=True, help="Experiment output directory")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--duration", type=float, default=600.0)
    p.add_argument(
        "--neural-backend",
        choices=["custom_rate_equations", "ratinabox_neurons"],
        default="ratinabox_neurons",
    )
    p.add_argument("--spike-source", choices=["sorted", "ground_truth"], default="sorted")
    p.add_argument("--compare-sources", action="store_true")
    p.add_argument("--behavior-rate", type=float, default=20.0)
    p.add_argument("--decode-windows", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    p.add_argument("--feature-modes", nargs="+", default=list(QUICK_FEATURE_MODES))
    p.add_argument("--manifold-components-list", type=int, nargs="+", default=[3])
    p.add_argument("--max-models", choices=["quick", "full"], default="quick")
    p.add_argument("--closed-loop-target", default="spatial_context")
    p.add_argument(
        "--selection-policy",
        choices=["best_accuracy", "shortest_near_optimal"],
        default="shortest_near_optimal",
    )
    p.add_argument(
        "--partitions",
        nargs="+",
        default=["all_units", "subfield", "layer", "ca1_deep_superficial", "cell_class"],
        help=f"Partition names. Available: {available_partitions()}",
    )
    p.add_argument("--integration-window", type=float, default=0.250)
    p.add_argument("--enable-temporal-manifold", action="store_true")
    p.add_argument(
        "--representations",
        nargs="+",
        default=["raw", "pca"],
        help="Temporal manifold representations (decode-temporal)",
    )
    p.add_argument(
        "--latent-history-frames",
        type=int,
        nargs="+",
        default=[1, 2, 5, 10, 20],
    )
    p.add_argument(
        "--prediction-lags",
        type=float,
        nargs="+",
        default=[0.0],
    )
    p.add_argument("--compile-pdf", action="store_true")
    p.add_argument("--skip-visualization", action="store_true")
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def _require_input(args: argparse.Namespace, stage: str) -> Path:
    input_dir = Path(args.input) if args.input is not None else Path(args.output)
    if not input_dir.exists():
        raise SystemExit(
            f"--input/--output directory does not exist for stage {stage!r}: {input_dir}"
        )
    return input_dir


def stage_simulate(args: argparse.Namespace) -> Path:
    from hippo_sim.config import SimConfig
    from hippo_sim.pipeline import run_pipeline

    out = Path(args.output)
    config = SimConfig(
        output_dir=out,
        seed=args.seed,
        session_duration_s=args.duration,
        neural_backend=args.neural_backend,
    )
    run_pipeline(config)
    print(f"Simulation written to {out}")
    return out


def stage_fit_manifolds(args: argparse.Namespace) -> None:
    """Build shared ManifoldDataset caches and unit metadata tables."""
    input_dir = _require_input(args, "fit-manifolds")
    out = Path(args.output) / "manifolds" / "datasets"
    out.mkdir(parents=True, exist_ok=True)
    sources = ("sorted", "ground_truth") if args.compare_sources else (args.spike_source,)
    for source in sources:
        ds = load_manifold_dataset(
            input_dir,
            spike_source=source,
            integration_window_s=args.integration_window,
            activity_representation="counts",
        )
        dest = out / f"{source}_w{int(round(args.integration_window * 1000)):04d}ms"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            dest / "activity.npz",
            activity=ds.activity,
            timestamps_s=ds.timestamps_s,
            unit_ids=ds.unit_ids,
            train_mask=ds.train_mask,
            test_mask=ds.test_mask,
        )
        ds.behavior.to_csv(dest / "behavior_aligned.csv", index=False)
        ds.unit_metadata.to_csv(dest / "unit_metadata.csv", index=False)
        metadata_availability_table(ds.unit_metadata).to_csv(
            dest / "metadata_availability.csv",
            index=False,
        )
        with open(dest / "dataset_meta.json", "w") as f:
            json.dump(
                {
                    "spike_source": ds.spike_source,
                    "integration_window_s": ds.integration_window_s,
                    "update_interval_s": ds.update_interval_s,
                    "activity_representation": ds.activity_representation,
                    "n_times": ds.n_times,
                    "n_units": ds.n_units,
                    "timing_validation": {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in ds.timing_validation.items()
                    },
                },
                f,
                indent=2,
                default=str,
            )
        print(f"  ManifoldDataset cached: {dest} ({ds.n_times}×{ds.n_units})")


def stage_compare_partitions(args: argparse.Namespace) -> None:
    input_dir = _require_input(args, "compare-partitions")
    units_path = input_dir / "units.csv"
    if not units_path.exists():
        raise FileNotFoundError(f"Missing {units_path}")
    units = normalize_unit_metadata(pd.read_csv(units_path))
    results = apply_partitions(units, args.partitions, min_units_per_group=5)
    out = Path(args.output) / "manifolds" / "partitions"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, part in results.items():
        part.group_metadata.to_csv(out / f"{name}_groups.csv", index=False)
        with open(out / f"{name}_exclusion.json", "w") as f:
            json.dump(part.exclusion_reason, f, indent=2)
        for g, ids in part.group_labels.items():
            summary.append({"partition": name, "group": g, "n_units": int(len(ids))})
            pd.Series(ids, name="unit_id").to_csv(
                out / f"{name}__{g}__unit_ids.csv",
                index=False,
            )
    pd.DataFrame(summary).to_csv(out / "partition_summary.csv", index=False)
    print(f"  Partition summary written to {out}")


def stage_decode(
    args: argparse.Namespace,
    *,
    enable_temporal: bool,
    skip_viz: bool,
) -> None:
    input_dir = _require_input(args, "decode")
    update_dt = 1.0 / float(args.behavior_rate) if args.behavior_rate else DEFAULT_UPDATE_DT_S
    run_full_decoder_workflow(
        input_dir=input_dir,
        output_dir=Path(args.output),
        compare_sources=args.compare_sources,
        spike_source=args.spike_source,
        decode_windows=tuple(args.decode_windows),
        max_models=args.max_models,
        closed_loop_target=args.closed_loop_target,
        selection_policy=args.selection_policy,
        update_dt=update_dt,
        feature_modes=tuple(args.feature_modes),
        manifold_n_components=tuple(args.manifold_components_list),
        skip_visualization=skip_viz,
        compile_pdf=args.compile_pdf and not skip_viz,
        enable_temporal_manifold=enable_temporal,
        representations=tuple(args.representations),
        latent_history_frames=tuple(args.latent_history_frames),
        prediction_lags=tuple(args.prediction_lags),
        n_jobs=args.n_jobs,
    )


def stage_visualize(args: argparse.Namespace) -> None:
    from visualization.experiment_viz import generate_experiment_figures

    generate_experiment_figures(
        experiment_dir=Path(args.output),
        include_simulation=True,
        include_comparison=True,
        include_realtime=True,
        compile_pdf=args.compile_pdf,
    )


def stage_report(args: argparse.Namespace) -> None:
    report_dir = Path(args.output) / "manifolds" / "summary"
    report_dir.mkdir(parents=True, exist_ok=True)
    parts = Path(args.output) / "manifolds" / "partitions" / "partition_summary.csv"
    lines = [
        "# Experiment report (Phase 1)",
        "",
        f"Output directory: `{args.output}`",
        "",
        "## Implemented artifacts",
        "",
        "- Simulation: `behavior.csv`, `units.csv`, spikes, `summary.json`",
        "- Static decoding: `decoder_comparison/`",
        "- Realtime replay: `realtime_decoding/`",
        "- Manifold datasets: `manifolds/datasets/`",
        "- Partitions: `manifolds/partitions/`",
        "",
        "## Planned (not yet aggregated here)",
        "",
        "- `best_by_partition.json`, communication summaries, topology (Phases 3–5)",
        "",
    ]
    if parts.exists():
        lines.append("## Partition summary")
        lines.append("")
        lines.append("```")
        lines.append(parts.read_text().strip())
        lines.append("```")
        lines.append("")
    (report_dir / "report.md").write_text("\n".join(lines))
    print(f"  Wrote {report_dir / 'report.md'}")


def main() -> None:
    args = parse_args()
    stage = args.stage
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if stage in ("simulate", "all"):
        stage_simulate(args)
        args.input = args.output

    if stage in ("fit-manifolds", "all"):
        stage_fit_manifolds(args)

    if stage in ("compare-partitions", "all"):
        stage_compare_partitions(args)

    if stage == "decode-static":
        stage_decode(args, enable_temporal=False, skip_viz=args.skip_visualization)

    if stage == "decode-temporal":
        stage_decode(args, enable_temporal=True, skip_viz=args.skip_visualization)

    if stage == "replay-realtime":
        # Replay uses comparison artifacts; run decoder workflow without re-viz by default.
        stage_decode(args, enable_temporal=False, skip_viz=True)

    if stage == "all":
        stage_decode(
            args,
            enable_temporal=bool(args.enable_temporal_manifold),
            skip_viz=args.skip_visualization,
        )

    if stage == "analyze-communication":
        print(
            "analyze-communication is planned for Phase 5 "
            "(lagged reduced-rank regression). No analysis run."
        )

    if stage == "visualize":
        stage_visualize(args)

    if stage in ("report", "all"):
        stage_report(args)

    print(f"Stage {stage!r} complete → {out}")


if __name__ == "__main__":
    main()
