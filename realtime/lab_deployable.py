"""Lab-deployable decoder profile export and final selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from realtime.realtime_gate import apply_realtime_gate


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
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


def build_lab_deployable_profile(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    r = dict(row) if not isinstance(row, dict) else row
    return {
        "target_name": r.get("target_name"),
        "target_family": r.get("target_family"),
        "feature_type": r.get("feature_type"),
        "embedding_type": r.get("embedding_type", "identity"),
        "decoder_name": r.get("decoder_name"),
        "decode_window_s": r.get("decode_window_s"),
        "update_dt_s": r.get("update_dt_s"),
        "trigger_rule": r.get("trigger_rule"),
        "spike_source": r.get("spike_source", "sorted"),
        "expected_sorted_performance": r.get("metric_value", r.get("primary_metric_value")),
        "latency_profile": {
            "feature_compute_ms": r.get("feature_compute_ms"),
            "embedding_transform_ms": r.get("embedding_transform_ms"),
            "decoder_predict_ms": r.get("decoder_predict_ms"),
            "total_compute_ms": r.get("total_compute_ms"),
        },
        "calibration_profile": {
            "confidence_metric_name": r.get("confidence_metric_name"),
            "confidence_metric_value": r.get("confidence_metric_value"),
            "is_well_calibrated": bool(r.get("is_well_calibrated", False)),
        },
        "cross_run_profile": {
            "mean_metric": r.get("cross_run_mean_metric", r.get("mean_metric")),
            "std_metric": r.get("cross_run_std_metric", r.get("std_metric")),
            "worst_run_metric": r.get("cross_run_worst_metric", r.get("worst_run_metric")),
            "n_runs": r.get("n_runs", 1),
        },
        "required_inputs": ["spike_time", "unit_id"],
        "unit_metadata_required": ["region", "cell_type"],
        "model_path": r.get("model_path"),
        "embedding_transform_path": r.get(
            "embedding_transform_path", r.get("manifold_transform_path")
        ),
    }


def _candidate_passes_selection(row: pd.Series) -> bool:
    if str(row.get("spike_source", "sorted")) != "sorted":
        return False
    if not bool(row.get("passes_realtime_gate", False)):
        return False
    # Calibration required for categorical; continuous may omit.
    family = str(row.get("target_family", ""))
    if family == "categorical" and not bool(row.get("is_well_calibrated", False)):
        return False
    label = str(row.get("sorting_robustness_label", "minimal_loss"))
    if label == "large_loss":
        return False
    if "beats_controls" in row.index and not bool(row.get("beats_controls", True)):
        return False
    return True


def select_best_lab_deployable_decoders(
    metrics_df: pd.DataFrame,
    *,
    cross_run_df: pd.DataFrame | None = None,
    control_df: pd.DataFrame | None = None,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Select one best lab-deployable decoder per target and write profiles.

    Criteria:
      1. Sorted spikes
      2. Realtime latency gate
      3. Calibration (categorical)
      4. Cross-run generalization when available
      5. Beats negative controls when available
      6. Acceptable sorting robustness
      7. Clear deployment profile
    """
    from realtime.decoder_comparison import PRIMARY_METRIC

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    profiles_dir = output_dir / "lab_deployable_profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    df = metrics_df.copy()
    if "target_name" in df.columns:
        df = df[df["target_name"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    # Merge cross-run stats when present.
    if cross_run_df is not None and not cross_run_df.empty:
        merge_cols = [
            c for c in (
                "target_name", "feature_type", "embedding_type",
                "decoder_name", "decode_window_s", "spike_source",
            )
            if c in df.columns and c in cross_run_df.columns
        ]
        if merge_cols:
            cr = cross_run_df.rename(columns={
                "mean_metric": "cross_run_mean_metric",
                "std_metric": "cross_run_std_metric",
                "worst_run_metric": "cross_run_worst_metric",
            })
            keep = merge_cols + [
                c for c in (
                    "cross_run_mean_metric", "cross_run_std_metric",
                    "cross_run_worst_metric", "n_runs", "fraction_runs_passing_gate",
                )
                if c in cr.columns
            ]
            df = df.merge(cr[keep], on=merge_cols, how="left")

    # Mark beats_controls using control summary when available.
    if control_df is not None and not control_df.empty and "balanced_accuracy" in control_df.columns:
        ctrl_best = (
            control_df.groupby("target_name")["balanced_accuracy"].max().to_dict()
            if "target_name" in control_df.columns
            else {}
        )
        beats = []
        for _, row in df.iterrows():
            target = str(row.get("target_name"))
            if target not in PRIMARY_METRIC:
                beats.append(True)
                continue
            metric_name, direction = PRIMARY_METRIC[target]
            if metric_name not in row or target not in ctrl_best:
                beats.append(True)
                continue
            val = float(row[metric_name])
            baseline = float(ctrl_best[target])
            if direction == "higher":
                beats.append(val > baseline)
            else:
                beats.append(True)  # continuous controls handled loosely
        df["beats_controls"] = beats
    else:
        df["beats_controls"] = True

    # Ensure gate columns exist (estimate if missing).
    if "passes_realtime_gate" not in df.columns:
        gate_rows = []
        for _, row in df.iterrows():
            gate = apply_realtime_gate(
                feature_compute_ms=float(row.get("feature_compute_ms", 1.0) or 1.0),
                embedding_transform_ms=float(row.get("embedding_transform_ms", 1.0) or 1.0),
                decoder_predict_ms=float(row.get("decoder_predict_ms", 1.0) or 1.0),
                decode_window_s=float(row.get("decode_window_s", 0.25) or 0.25),
                update_dt_s=float(row.get("update_dt_s", 0.05) or 0.05),
                max_compute_ms=float(row.get("max_compute_ms", 25.0) or 25.0),
                max_effective_history_s=float(row.get("max_effective_history_s", 0.5) or 0.5),
            )
            gate_rows.append(gate)
        gate_df = pd.DataFrame(gate_rows)
        for col in gate_df.columns:
            df[col] = gate_df[col].to_numpy()

    if "is_well_calibrated" not in df.columns:
        df["is_well_calibrated"] = df.get("target_family", pd.Series(["continuous"] * len(df))) != "categorical"
        # Continuous targets: treat as calibrated for selection purposes.
        df.loc[df["target_family"] == "continuous", "is_well_calibrated"] = True

    if "sorting_robustness_label" not in df.columns:
        df["sorting_robustness_label"] = "minimal_loss"

    selected_rows: list[dict[str, Any]] = []
    for target, group in df.groupby("target_name"):
        target = str(target)
        if target not in PRIMARY_METRIC:
            continue
        metric_name, direction = PRIMARY_METRIC[target]
        cand = group[group.apply(_candidate_passes_selection, axis=1)].copy()
        if cand.empty:
            continue
        # Prefer cross-run mean when available.
        score_col = "cross_run_mean_metric" if "cross_run_mean_metric" in cand.columns else metric_name
        if score_col not in cand.columns:
            score_col = metric_name
        scored = cand.dropna(subset=[score_col])
        if scored.empty:
            continue
        if direction == "lower":
            best = scored.loc[scored[score_col].astype(float).idxmin()]
        else:
            best = scored.loc[scored[score_col].astype(float).idxmax()]

        profile = build_lab_deployable_profile(best)
        profile_path = profiles_dir / f"{target}_lab_deployable_decoder_profile.json"
        with open(profile_path, "w") as f:
            json.dump(_json_safe(profile), f, indent=2)

        # Also write a convenience copy at output root for the overall winner set.
        selected_rows.append({
            "target_name": target,
            "target_family": best.get("target_family"),
            "feature_type": best.get("feature_type"),
            "embedding_type": best.get("embedding_type", "identity"),
            "decoder_name": best.get("decoder_name"),
            "decode_window_s": best.get("decode_window_s"),
            "update_dt_s": best.get("update_dt_s"),
            "trigger_rule": best.get("trigger_rule"),
            "spike_source": best.get("spike_source", "sorted"),
            "primary_metric": metric_name,
            "metric_value": float(best.get(metric_name, best.get(score_col))),
            "passes_realtime_gate": bool(best.get("passes_realtime_gate", False)),
            "is_well_calibrated": bool(best.get("is_well_calibrated", False)),
            "sorting_robustness_label": best.get("sorting_robustness_label"),
            "cross_run_mean_metric": best.get("cross_run_mean_metric"),
            "cross_run_std_metric": best.get("cross_run_std_metric"),
            "cross_run_worst_metric": best.get("cross_run_worst_metric"),
            "deployment_profile_path": str(profile_path),
        })

    out = pd.DataFrame(selected_rows)
    if not out.empty:
        out.to_csv(output_dir / "best_lab_deployable_decoders.csv", index=False)
        # Combined profiles JSON (array) for convenience.
        combined = []
        for p in out["deployment_profile_path"]:
            with open(p) as f:
                combined.append(json.load(f))
        with open(output_dir / "lab_deployable_decoder_profile.json", "w") as f:
            json.dump(_json_safe(combined), f, indent=2)
    return out
