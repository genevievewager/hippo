"""Sorted-vs-ground-truth information-loss summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd


def sorting_robustness_label(relative_drop: float) -> str:
    if relative_drop < 0.10:
        return "minimal_loss"
    if relative_drop <= 0.30:
        return "moderate_loss"
    return "large_loss"


def relative_performance_drop(
    gt_value: float,
    sorted_value: float,
    *,
    direction: str,
) -> tuple[float, float]:
    """Return (absolute_drop, relative_drop) where drop > 0 means sorted is worse."""
    if direction == "lower":
        absolute = float(sorted_value - gt_value)
        denom = abs(gt_value) if abs(gt_value) > 1e-12 else 1.0
        relative = absolute / denom
    else:
        absolute = float(gt_value - sorted_value)
        denom = abs(gt_value) if abs(gt_value) > 1e-12 else 1.0
        relative = absolute / denom
    return absolute, relative


def build_sorted_information_loss_summary(
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare ground-truth vs sorted performance for matching candidate families.

    Matching key: target × feature_type × embedding_type × decoder × window.
    """
    from realtime.decoder_comparison import PRIMARY_METRIC

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()
    df = metrics_df.copy()
    if "target_name" in df.columns:
        df = df[df["target_name"].notna()].copy()
    if "spike_source" not in df.columns or df.empty:
        return pd.DataFrame()

    src = df["spike_source"].astype(str)
    gt = df[src == "ground_truth"]
    sorted_df = df[src == "sorted"]
    if gt.empty or sorted_df.empty:
        return pd.DataFrame()

    key_cols = [
        c for c in (
            "target_name",
            "feature_set",
            "feature_type",
            "embedding_type",
            "feature_mode",
            "decoder_name",
            "decode_window_s",
            "manifold_n_components",
        )
        if c in df.columns
    ]

    rows: list[dict[str, Any]] = []
    # Prefer embedding-aware keys when present.
    merge_cols = [c for c in key_cols if c != "feature_mode" or "embedding_type" not in key_cols]
    if "embedding_type" not in merge_cols and "feature_mode" in key_cols:
        merge_cols = [c for c in key_cols if c != "embedding_type"]

    gt_idx = gt.set_index(merge_cols, drop=False)
    for _, srow in sorted_df.iterrows():
        key = tuple(srow[c] for c in merge_cols)
        if key not in gt_idx.index:
            continue
        grow = gt_idx.loc[key]
        if isinstance(grow, pd.DataFrame):
            grow = grow.iloc[0]
        target = str(srow["target_name"])
        if target not in PRIMARY_METRIC:
            continue
        metric_name, direction = PRIMARY_METRIC[target]
        if metric_name not in srow or metric_name not in grow:
            continue
        if pd.isna(srow[metric_name]) or pd.isna(grow[metric_name]):
            continue
        gt_val = float(grow[metric_name])
        sorted_val = float(srow[metric_name])
        abs_drop, rel_drop = relative_performance_drop(
            gt_val, sorted_val, direction=direction,
        )
        rows.append({
            "target_name": target,
            "target_family": srow.get("target_family"),
            "feature_set": srow.get("feature_set", "counts"),
            "feature_type": srow.get("feature_type"),
            "embedding_type": srow.get("embedding_type", srow.get("feature_type")),
            "decoder_name": srow.get("decoder_name"),
            "decode_window_s": float(srow.get("decode_window_s")),
            "primary_metric": metric_name,
            "ground_truth_performance": gt_val,
            "sorted_performance": sorted_val,
            "absolute_drop": abs_drop,
            "relative_drop": rel_drop,
            "sorting_robustness_label": sorting_robustness_label(rel_drop),
        })
    return pd.DataFrame(rows)
