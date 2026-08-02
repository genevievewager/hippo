"""Benchmark causal-update latency across feature modes and decoder stages.

Fits common feature transformers (counts, PCA, classic Isomap, distilled Isomap)
on the training split and measures per-update transform cost, then aggregates
closed-loop stage latency from any existing realtime run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.decoding_targets import align_extended_behavior_to_decoder_times
from realtime.latency_profiler import (
    DEFAULT_UPDATE_BUDGET_MS,
    benchmark_feature_transforms,
    save_latency_artifacts,
)
from realtime.manifold_features import (
    DEFAULT_ISOMAP_N_NEIGHBORS,
    make_feature_transformer,
)
from realtime.spike_features import build_causal_spike_matrix
from realtime.timing import extract_behavior_times, resolve_update_dt_s
from realtime.train_decoder import causal_train_test_split, infer_arena_bounds


def run_latency_benchmark(
    experiment_dir: Path,
    *,
    spike_source: str = "sorted",
    decode_window_s: float = 0.250,
    update_dt_s: float | None = None,
    train_frac: float = 0.70,
    isomap_n_components: int = 8,
    isomap_n_neighbors: int = DEFAULT_ISOMAP_N_NEIGHBORS,
    n_probe: int = 80,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Fit feature front-ends and write latency comparison artifacts."""
    experiment_dir = Path(experiment_dir)
    output_dir = Path(output_dir) if output_dir else experiment_dir / "latency_profiling"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_simulation_data(experiment_dir, spike_source)
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"],
        derive_from_behavior=True,
        update_dt_s=update_dt_s,
        behavior_times=behavior_times,
    )
    decode_times = make_decode_times(
        data["session_duration"],
        decode_window_s,
        update_dt,
        behavior_times=behavior_times,
    )
    X_counts = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, decode_window_s
    )
    train_mask, test_mask = causal_train_test_split(decode_times, train_frac)
    X_train = X_counts[train_mask]
    X_test = X_counts[test_mask]
    # Cap Isomap fit size for latency benchmarking (geometry approx. is enough).
    max_isomap_fit = 2500
    if len(X_train) > max_isomap_fit:
        idx = np.linspace(0, len(X_train) - 1, max_isomap_fit).astype(int)
        X_train_iso = X_train[idx]
    else:
        X_train_iso = X_train
    probe = X_test[: min(n_probe, len(X_test))]
    if len(probe) == 0:
        probe = X_train[: min(n_probe, len(X_train))]

    modes = {
        "counts": dict(n_components=3, n_neighbors=None, fit_X="full"),
        "global_pca": dict(n_components=3, n_neighbors=None, fit_X="full"),
        "region_pca": dict(n_components=3, n_neighbors=None, fit_X="full"),
        "global_isomap": dict(
            n_components=isomap_n_components,
            n_neighbors=isomap_n_neighbors,
            fit_X="iso",
        ),
        "global_isomap_distilled": dict(
            n_components=isomap_n_components,
            n_neighbors=isomap_n_neighbors,
            fit_X="iso",
        ),
    }

    transformers: dict[str, Any] = {}
    fit_meta: dict[str, Any] = {}
    for mode, cfg in modes.items():
        try:
            tr = make_feature_transformer(
                mode,
                decode_window=decode_window_s,
                n_components=int(cfg["n_components"]),
                units_df=data["units_df"],
                unit_ids=data["unit_ids"],
                random_state=seed,
                n_neighbors=int(cfg["n_neighbors"] or DEFAULT_ISOMAP_N_NEIGHBORS),
            )
            if tr is None:
                fit_meta[mode] = {"status": "skipped", "reason": "missing metadata"}
                continue
            fit_X = X_train_iso if cfg.get("fit_X") == "iso" else X_train
            tr.fit(fit_X)
            transformers[mode] = tr
            meta = tr.get_metadata() if hasattr(tr, "get_metadata") else {}
            fit_meta[mode] = {
                "status": "ok",
                "realtime_compatible": bool(meta.get("realtime_compatible", True)),
                "actual_n_features": meta.get("actual_n_features"),
                "runtime_per_transform_ms": meta.get("runtime_per_transform_ms"),
                "distillation_metrics": meta.get("distillation_metrics"),
            }
            # Persist distilled transform for optional realtime reuse
            if mode == "global_isomap_distilled":
                save_path = output_dir / "global_isomap_distilled_transform"
                tr.save(save_path)
                fit_meta[mode]["transform_path"] = str(save_path)
        except Exception as exc:
            fit_meta[mode] = {"status": "failed", "reason": str(exc)}
            print(f"  latency benchmark: skip {mode}: {exc}")

    feature_latency = benchmark_feature_transforms(
        probe, transformers, n_warmup=5, n_repeats=min(60, len(probe)),
    )
    feature_latency.to_csv(output_dir / "feature_transform_latency.csv", index=False)

    # Classic Isomap teacher vs distilled (same probe) if both available
    teacher_vs_student = []
    if "global_isomap" in transformers and "global_isomap_distilled" in transformers:
        iso = transformers["global_isomap"]
        dist = transformers["global_isomap_distilled"]
        # Already measured mean_ms in feature_latency; also measure teacher OOS cost.
        from realtime.latency_profiler import time_callable_ms

        _, teacher_ms = time_callable_ms(
            lambda: iso.transform(probe[:1]), repeats=40
        )
        _, student_ms = time_callable_ms(
            lambda: dist.transform(probe[:1]), repeats=40
        )
        teacher_vs_student.append({
            "method": "global_isomap_teacher",
            "mean_ms": teacher_ms,
            "realtime_compatible": False,
        })
        teacher_vs_student.append({
            "method": "global_isomap_distilled",
            "mean_ms": student_ms,
            "realtime_compatible": bool(
                getattr(dist, "realtime_compatible", False)
            ),
        })
        pd.DataFrame(teacher_vs_student).to_csv(
            output_dir / "isomap_teacher_vs_distilled_latency.csv", index=False
        )

    # Merge any existing realtime stage latency
    stage_frames = []
    rt_root = experiment_dir / "realtime_decoding"
    if rt_root.exists():
        for lat_csv in rt_root.rglob("latency/latency_by_stage.csv"):
            df = pd.read_csv(lat_csv)
            df["source"] = str(lat_csv.relative_to(rt_root).parent.parent)
            stage_frames.append(df)
    if stage_frames:
        stages = pd.concat(stage_frames, ignore_index=True)
        stages.to_csv(output_dir / "realtime_stage_latency_combined.csv", index=False)
    else:
        stages = pd.DataFrame()

    # Unified long-form table for plotting "everything"
    plot_rows = []
    for _, row in feature_latency.iterrows():
        plot_rows.append({
            "category": "feature_transform",
            "name": row["feature_mode"],
            "mean_ms": float(row["mean_ms"]),
            "median_ms": float(row["median_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "realtime_compatible": bool(row.get("realtime_compatible", True)),
        })
    for item in teacher_vs_student:
        plot_rows.append({
            "category": "isomap_compare",
            "name": item["method"],
            "mean_ms": float(item["mean_ms"]),
            "median_ms": float(item["mean_ms"]),
            "p95_ms": float(item["mean_ms"]),
            "realtime_compatible": bool(item["realtime_compatible"]),
        })
    if not stages.empty:
        # Prefer the sorted deployable run if present
        prefer = stages
        if "source" in stages.columns:
            sorted_rows = stages[stages["source"].astype(str).str.contains("sorted")]
            if not sorted_rows.empty:
                prefer = sorted_rows
        # Take first source group's stages
        if "source" in prefer.columns:
            first_src = prefer["source"].iloc[0]
            prefer = prefer[prefer["source"] == first_src]
        for _, row in prefer.iterrows():
            plot_rows.append({
                "category": "realtime_stage",
                "name": row["stage"],
                "mean_ms": float(row["mean_ms"]),
                "median_ms": float(row.get("median_ms", row["mean_ms"])),
                "p95_ms": float(row.get("p95_ms", row["mean_ms"])),
                "realtime_compatible": True,
            })

    plot_df = pd.DataFrame(plot_rows)
    plot_df.to_csv(output_dir / "latency_everything.csv", index=False)

    summary = {
        "spike_source": spike_source,
        "decode_window_s": float(decode_window_s),
        "update_dt_s": float(update_dt),
        "update_budget_ms": float(update_dt) * 1000.0,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_probe": int(len(probe)),
        "isomap_n_components": int(isomap_n_components),
        "isomap_n_neighbors": int(isomap_n_neighbors),
        "feature_fit": fit_meta,
        "feature_transform_latency": feature_latency.to_dict(orient="records"),
        "isomap_teacher_vs_distilled": teacher_vs_student,
        "arena_bounds": list(infer_arena_bounds(data["behavior_df"], data["summary"])),
    }
    with open(output_dir / "latency_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Canonical summary artifacts expected by README / viz.
    summary_rows: list[dict[str, Any]] = []
    for _, row in feature_latency.iterrows():
        summary_rows.append({
            "category": "feature_transform",
            "name": row["feature_mode"],
            "mean_ms": float(row["mean_ms"]),
            "median_ms": float(row["median_ms"]),
            "p95_ms": float(row["p95_ms"]),
            "update_budget_ms": float(summary["update_budget_ms"]),
            "within_budget": bool(float(row["mean_ms"]) <= float(summary["update_budget_ms"])),
            "realtime_compatible": bool(row.get("realtime_compatible", True)),
        })
    if not stages.empty:
        prefer = stages
        if "source" in stages.columns:
            sorted_rows = stages[stages["source"].astype(str).str.contains("sorted")]
            if not sorted_rows.empty:
                prefer = sorted_rows
        if "source" in prefer.columns:
            prefer = prefer[prefer["source"] == prefer["source"].iloc[0]]
        for _, row in prefer.iterrows():
            mean_ms = float(row["mean_ms"])
            summary_rows.append({
                "category": "realtime_stage",
                "name": row["stage"],
                "mean_ms": mean_ms,
                "median_ms": float(row.get("median_ms", mean_ms)),
                "p95_ms": float(row.get("p95_ms", mean_ms)),
                "update_budget_ms": float(summary["update_budget_ms"]),
                "within_budget": bool(mean_ms <= float(summary["update_budget_ms"])),
                "realtime_compatible": True,
            })
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df.to_csv(output_dir / "latency_summary.csv", index=False)
    # Also mirror under the canonical JSON name.
    total_row = None
    if not summary_df.empty:
        tot = summary_df[summary_df["name"] == "total_update"]
        if not tot.empty:
            total_row = tot.iloc[0].to_dict()
    latency_summary = {
        **summary,
        "total_update": total_row,
        "rows": summary_rows,
    }
    with open(output_dir / "latency_summary.json", "w") as f:
        json.dump(latency_summary, f, indent=2)

    # Annotate deployable registry with budget compatibility when present.
    _annotate_registry_latency(experiment_dir, latency_summary)

    print(
        f"  latency benchmark → {output_dir} "
        f"(budget={summary['update_budget_ms']:.1f} ms)"
    )
    if not feature_latency.empty:
        for _, row in feature_latency.iterrows():
            flag = "RT" if row.get("realtime_compatible", True) else "offline"
            print(
                f"    {row['feature_mode']:28s}  "
                f"mean={row['mean_ms']:.3f} ms  [{flag}]"
            )
    return summary


def _annotate_registry_latency(experiment_dir: Path, latency_summary: dict[str, Any]) -> None:
    """Record whether selected targets clear the update budget in the registry."""
    from realtime.deployment_selection import load_best_realtime_decoders

    experiment_dir = Path(experiment_dir)
    try:
        payload = load_best_realtime_decoders(experiment_dir)
    except FileNotFoundError:
        return

    budget = float(latency_summary.get("update_budget_ms", DEFAULT_UPDATE_BUDGET_MS))
    total = latency_summary.get("total_update") or {}
    total_ms = total.get("mean_ms")
    within = None if total_ms is None else bool(float(total_ms) <= budget)

    # Feature-mode level compatibility from feature_transform rows.
    feat_ok: dict[str, bool] = {}
    for row in latency_summary.get("rows") or []:
        if row.get("category") == "feature_transform":
            feat_ok[str(row["name"])] = bool(row.get("realtime_compatible", True)) and bool(
                row.get("within_budget", True)
            )

    for target, tgt in (payload.get("targets") or {}).items():
        feature = str(tgt.get("selected_feature_mode", "counts"))
        decoder = str(tgt.get("selected_decoder", ""))
        feature_ok = feat_ok.get(feature, True)
        # Random-forest classifier heads are often over budget at 20 Hz; surface clearly.
        decoder_warn = None
        if "random_forest_classifier" in decoder and within is False:
            decoder_warn = (
                f"{decoder} selected for {target} but measured total_update "
                f"mean={total_ms:.2f} ms exceeds {budget:.1f} ms budget"
            )
            print(f"  WARNING: {decoder_warn}")
        tgt["update_budget_ms"] = budget
        tgt["measured_total_update_mean_ms"] = total_ms
        tgt["within_update_budget"] = within
        tgt["feature_realtime_compatible"] = feature_ok
        tgt["realtime_compatible"] = bool(
            tgt.get("realtime_compatible", True) and feature_ok and (within is not False)
        )
        if decoder_warn:
            tgt["realtime_budget_warning"] = decoder_warn
            tgt["realtime_compatible"] = False

    payload["update_budget_ms"] = budget
    payload["measured_total_update_mean_ms"] = total_ms
    def _safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_safe(v) for v in obj]
        if isinstance(obj, Path):
            return str(obj)
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                pass
        try:
            if pd.isna(obj):
                return None
        except Exception:
            pass
        return obj

    for path in (
        experiment_dir / "models" / "best_realtime_decoders.json",
        experiment_dir / "deployment_decoder_selection" / "best_realtime_decoders.json",
    ):
        if path.exists():
            with open(path, "w") as f:
                json.dump(_safe(payload), f, indent=2)
