"""Helpers for feature-set × manifold comparison."""

from __future__ import annotations

from typing import Any

import pandas as pd

from realtime.neural_features.feature_sets import embedding_compatible_with_feature_set
from realtime.search_space import resolve_manifold_alias

# Mirrored from decoder_comparison.PRIMARY_METRIC to avoid circular imports.
_PRIMARY_METRIC = {
    "position": ("mean_position_error_cm", "lower"),
    "speed": ("r2", "higher"),
    "acceleration": ("r2", "higher"),
    "head_direction": ("mean_circular_error_deg", "lower"),
    "distance_to_wall": ("r2", "higher"),
    "spatial_context": ("balanced_accuracy", "higher"),
    "movement_state": ("balanced_accuracy", "higher"),
    "wall_distance_bin": ("balanced_accuracy", "higher"),
}


def resolve_manifolds_arg(
    manifolds: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    """Map CLI ``--manifolds`` aliases (including ``counts`` → ``identity``)."""
    if not manifolds:
        return None
    return tuple(resolve_manifold_alias(m) for m in manifolds)


def filter_fe_jobs_for_feature_set(
    fe_jobs: list[tuple[str, str, int | None, int | None]],
    feature_set: str,
) -> list[tuple[str, str, int | None, int | None]]:
    """Drop groupwise PCA jobs that require unit-aligned count features."""
    out = []
    for job in fe_jobs:
        _f, emb, _k, _nn = job
        if embedding_compatible_with_feature_set(emb, feature_set):
            out.append(job)
    return out


def effective_spike_feature_type(feature_set: str, feature_type: str) -> str:
    """Richer feature sets skip count-specific F transforms (rates/sqrt/…)."""
    if feature_set == "counts":
        return feature_type
    return "counts"


def summarize_feature_set_performance(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Compact table: best primary metric per feature_set × target."""
    if metrics_df is None or metrics_df.empty or "feature_set" not in metrics_df.columns:
        return pd.DataFrame()
    df = metrics_df.copy()
    if "target_name" in df.columns:
        df = df[df["target_name"].notna()]
    rows = []
    for (fs, target), g in df.groupby(["feature_set", "target_name"], dropna=False):
        if target not in _PRIMARY_METRIC:
            continue
        metric, direction = _PRIMARY_METRIC[str(target)]
        if metric not in g.columns:
            continue
        sub = g.dropna(subset=[metric])
        if sub.empty:
            continue
        if direction == "lower":
            best = sub.loc[sub[metric].idxmin()]
        else:
            best = sub.loc[sub[metric].idxmax()]
        rows.append({
            "feature_set": fs,
            "target_name": target,
            "embedding_type": best.get("embedding_type"),
            "decoder_name": best.get("decoder_name"),
            "decode_window_s": best.get("decode_window_s"),
            "feature_dimension": best.get("feature_dimension"),
            "latent_dimension": best.get(
                "actual_n_features", best.get("latent_dimension")
            ),
            "primary_metric": metric,
            "performance_value": float(best[metric]),
            "feature_extract_ms": best.get("feature_extract_ms"),
            "spike_source": best.get("spike_source"),
        })
    return pd.DataFrame(rows)
