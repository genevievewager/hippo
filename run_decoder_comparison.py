#!/usr/bin/env python3
"""Decoder comparison CLI: search F × E × D × W × C for lab-deployable decoders.

Figures are a separate inspection step (``run_visualizations.py``) and never retrain.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from realtime.cross_run_summary import build_cross_run_decoder_summary
from realtime.decoder_comparison import (
    DEFAULT_DECODE_WINDOWS,
    ComparisonRunConfig,
    run_compare_sources,
    run_decoder_comparison,
)
from realtime.feature_representations import ALL_FEATURE_TYPES, QUICK_FEATURE_TYPES
from realtime.lab_deployable import select_best_lab_deployable_decoders
from realtime.manifold_features import ALL_FEATURE_MODES, QUICK_FEATURE_MODES
from realtime.search_space import (
    ALL_EMBEDDING_TYPES,
    DEFAULT_MANIFOLD_N_COMPONENTS,
    QUICK_EMBEDDING_TYPES,
)
from realtime.sorting_robustness import build_sorted_information_loss_summary
from realtime.timing import DEFAULT_UPDATE_DT_S


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Search spike features F × embeddings E × decoders D × windows W × "
            "trigger rules C. Gate by accuracy, calibration, sorting robustness, "
            "cross-run generalization, and realtime latency. Export lab-deployable "
            "decoder profiles. Figures are never required to retrain."
        ),
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path, help="Single simulation output directory")
    src.add_argument(
        "--inputs", type=Path, nargs="+",
        help="Multiple simulation dirs for cross-run generalization",
    )
    p.add_argument("--output", type=Path, required=True, help="Comparison output directory")
    p.add_argument(
        "--spike-source", choices=["sorted", "ground_truth"], default="sorted",
    )
    p.add_argument(
        "--compare-sources", action="store_true",
        help="Also run ground-truth oracle comparison (single-input mode)",
    )
    p.add_argument(
        "--decode-windows", type=float, nargs="+",
        default=list(DEFAULT_DECODE_WINDOWS),
    )
    p.add_argument("--update-dt", type=float, default=DEFAULT_UPDATE_DT_S)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--max-models", choices=["quick", "full"], default="quick")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)

    # F
    p.add_argument(
        "--feature-types", nargs="+", default=None, choices=list(ALL_FEATURE_TYPES),
        help=f"Spike feature representations F (quick default: {QUICK_FEATURE_TYPES})",
    )
    # E
    p.add_argument(
        "--embedding-types", nargs="+", default=None, choices=list(ALL_EMBEDDING_TYPES),
        help=f"Embeddings E (quick default: {QUICK_EMBEDDING_TYPES})",
    )
    # Legacy combined modes
    p.add_argument(
        "--feature-modes", nargs="+", default=None, choices=list(ALL_FEATURE_MODES),
        help="Legacy combined modes (used when --feature-types/--embedding-types omitted)",
    )
    p.add_argument(
        "--manifold-components-list", type=int, nargs="+",
        default=None,
        help="Latent-dim grid for PCA/PLS embeddings",
    )
    p.add_argument("--manifold-n-components", type=int, default=None)

    # Realtime gates
    p.add_argument("--max-compute-ms", type=float, default=25.0)
    p.add_argument("--max-effective-history-s", type=float, default=0.500)

    # Controls / ablations
    p.add_argument("--include-controls", action="store_true")
    p.add_argument("--population-ablation", action="store_true")
    p.add_argument("--region-ablation", action="store_true")
    p.add_argument("--layer-ablation", action="store_true")
    p.add_argument("--adaptive-windows", action="store_true")

    # Trigger rules C
    p.add_argument("--trigger-context", default="wall")
    p.add_argument("--trigger-confidence", type=float, default=0.80)
    p.add_argument("--trigger-wall-bin", default="near_wall")
    p.add_argument("--trigger-distance-lt-cm", type=float, default=10.0)
    p.add_argument("--trigger-speed-gt-cm-s", type=float, default=10.0)
    p.add_argument("--trigger-movement", default="fast")
    p.add_argument("--trigger-hd-center-deg", type=float, default=90.0)
    p.add_argument("--trigger-hd-width-deg", type=float, default=30.0)
    p.add_argument("--no-trigger-search", action="store_true")
    return p.parse_args()


def _manifold_components(args: argparse.Namespace) -> tuple[int, ...]:
    if args.manifold_components_list:
        return tuple(int(x) for x in args.manifold_components_list)
    if args.manifold_n_components is not None:
        return (int(args.manifold_n_components),)
    if args.feature_types is not None or args.embedding_types is not None:
        return DEFAULT_MANIFOLD_N_COMPONENTS
    return (3,)


def _config_for(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
    *,
    spike_source: str,
    run_id: str | None = None,
) -> ComparisonRunConfig:
    use_fe = args.feature_types is not None or args.embedding_types is not None
    feature_modes = tuple(args.feature_modes) if args.feature_modes else QUICK_FEATURE_MODES
    return ComparisonRunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        spike_source=spike_source,
        decode_windows=tuple(float(w) for w in args.decode_windows),
        update_dt=args.update_dt,
        train_frac=args.train_frac,
        feature_modes=feature_modes,
        feature_types=tuple(args.feature_types) if args.feature_types else None,
        embedding_types=tuple(args.embedding_types) if args.embedding_types else None,
        use_fe_grid=use_fe,
        manifold_n_components=_manifold_components(args),
        max_models=args.max_models,
        n_jobs=args.n_jobs,
        seed=args.seed,
        region_ablation=args.region_ablation,
        layer_ablation=args.layer_ablation,
        population_ablation=args.population_ablation,
        adaptive_windows=args.adaptive_windows,
        include_controls=args.include_controls,
        max_compute_ms=args.max_compute_ms,
        max_effective_history_s=args.max_effective_history_s,
        enable_trigger_search=not args.no_trigger_search,
        trigger_context=args.trigger_context,
        trigger_confidence=args.trigger_confidence,
        trigger_wall_bin=args.trigger_wall_bin,
        trigger_distance_lt_cm=args.trigger_distance_lt_cm,
        trigger_speed_gt_cm_s=args.trigger_speed_gt_cm_s,
        trigger_movement=args.trigger_movement,
        trigger_hd_center_deg=args.trigger_hd_center_deg,
        trigger_hd_width_deg=args.trigger_hd_width_deg,
        run_id=run_id,
    )


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    if args.inputs:
        metrics_by_run: dict[str, pd.DataFrame] = {}
        for inp in args.inputs:
            run_id = Path(inp).name
            run_out = output / run_id
            print(f"=== Cross-run comparison: {run_id} ===")
            cfg = _config_for(
                Path(inp), run_out, args,
                spike_source=args.spike_source, run_id=run_id,
            )
            metrics_by_run[run_id] = run_decoder_comparison(cfg)

        cross = build_cross_run_decoder_summary(metrics_by_run)
        if not cross.empty:
            cross.to_csv(output / "cross_run_decoder_summary.csv", index=False)

        # Combined metrics + robustness (if GT also present in any run)
        combined = pd.concat(
            [df.assign(run_id=rid) for rid, df in metrics_by_run.items()],
            ignore_index=True,
        )
        combined.to_csv(output / "decoder_comparison_metrics.csv", index=False)
        loss = build_sorted_information_loss_summary(combined)
        if not loss.empty:
            loss.to_csv(output / "sorted_information_loss_summary.csv", index=False)

        control_path = None
        for rid in metrics_by_run:
            p = output / rid / "decoder_control_summary.csv"
            if p.exists():
                control_path = p
                break
        control_df = pd.read_csv(control_path) if control_path else None
        select_best_lab_deployable_decoders(
            combined[combined["spike_source"].astype(str) == "sorted"]
            if "spike_source" in combined.columns else combined,
            cross_run_df=cross,
            control_df=control_df,
            output_dir=output,
        )
        print(f"Wrote cross-run comparison → {output}")
        return

    # Single-input mode
    input_dir = Path(args.input)
    if args.compare_sources:
        print("=== Comparing ground_truth vs sorted ===")
        run_compare_sources(
            input_dir=input_dir,
            output_dir=output,
            decode_windows=tuple(float(w) for w in args.decode_windows),
            update_dt=args.update_dt,
            train_frac=args.train_frac,
            feature_modes=(
                tuple(args.feature_modes) if args.feature_modes else QUICK_FEATURE_MODES
            ),
            feature_types=tuple(args.feature_types) if args.feature_types else None,
            embedding_types=tuple(args.embedding_types) if args.embedding_types else None,
            manifold_n_components=_manifold_components(args),
            max_models=args.max_models,
            n_jobs=args.n_jobs,
            seed=args.seed,
            region_ablation=args.region_ablation,
            layer_ablation=args.layer_ablation,
            population_ablation=args.population_ablation,
            adaptive_windows=args.adaptive_windows,
            include_controls=args.include_controls,
            max_compute_ms=args.max_compute_ms,
            max_effective_history_s=args.max_effective_history_s,
            enable_trigger_search=not args.no_trigger_search,
            trigger_context=args.trigger_context,
            trigger_confidence=args.trigger_confidence,
            trigger_wall_bin=args.trigger_wall_bin,
            trigger_distance_lt_cm=args.trigger_distance_lt_cm,
            trigger_speed_gt_cm_s=args.trigger_speed_gt_cm_s,
            trigger_movement=args.trigger_movement,
            trigger_hd_center_deg=args.trigger_hd_center_deg,
            trigger_hd_width_deg=args.trigger_hd_width_deg,
        )
        sorted_metrics_path = output / "sorted" / "decoder_comparison_metrics.csv"
        if sorted_metrics_path.exists():
            sorted_metrics = pd.read_csv(sorted_metrics_path)
            select_best_lab_deployable_decoders(
                sorted_metrics, output_dir=output / "sorted",
            )
            for name in (
                "best_lab_deployable_decoders.csv",
                "lab_deployable_decoder_profile.json",
                "closed_loop_trigger_comparison.csv",
                "sorted_information_loss_summary.csv",
            ):
                src = output / "sorted" / name
                root = output / name
                if src.exists() and not root.exists():
                    root.write_bytes(src.read_bytes())
                # Prefer root-level sorted_information_loss from run_compare_sources
                alt = output / name
                if name == "sorted_information_loss_summary.csv" and alt.exists():
                    pass
    else:
        cfg = _config_for(input_dir, output, args, spike_source=args.spike_source)
        run_decoder_comparison(cfg)

    print(f"Wrote decoder comparison → {output}")


if __name__ == "__main__":
    main()
