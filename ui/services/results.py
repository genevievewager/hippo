"""Load decoder-comparison results from disk (no recomputation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realtime.decoder_comparison import ALL_TARGETS, PRIMARY_METRIC


@dataclass
class ComparisonArtifacts:
    """Resolved paths + loaded frames for one comparison output tree."""

    root: Path
    metrics: pd.DataFrame
    best_by_target: pd.DataFrame | None = None
    manifold_explained_variance: pd.DataFrame | None = None
    feature_set_summary: pd.DataFrame | None = None
    loss_summary: pd.DataFrame | None = None


def find_metrics_csv(experiment_or_comparison: Path) -> Path | None:
    """Locate ``decoder_comparison_metrics.csv`` under common layouts."""
    root = Path(experiment_or_comparison)
    candidates = [
        root / "decoder_comparison_metrics.csv",
        root / "sorted" / "decoder_comparison_metrics.csv",
        root / "decoder_comparison" / "sorted" / "decoder_comparison_metrics.csv",
        root / "decoder_comparison" / "decoder_comparison_metrics.csv",
        root / "decoder_comparison" / "ground_truth" / "decoder_comparison_metrics.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Prefer sorted if multiple nested copies exist.
    matches = sorted(root.rglob("decoder_comparison_metrics.csv"))
    for path in matches:
        if "sorted" in path.parts:
            return path
    return matches[0] if matches else None


def list_comparison_roots(experiment_dir: Path) -> list[Path]:
    """Return directories that contain comparison metrics for an experiment."""
    exp = Path(experiment_dir)
    found: list[Path] = []
    for metrics in exp.rglob("decoder_comparison_metrics.csv"):
        found.append(metrics.parent)
    # Deduplicate while preserving order
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def load_comparison_artifacts(root: Path) -> ComparisonArtifacts:
    """Load comparison tables without triggering scientific recomputation."""
    root = Path(root)
    metrics_path = find_metrics_csv(root)
    if metrics_path is None:
        raise FileNotFoundError(f"No decoder_comparison_metrics.csv under {root}")
    metrics = pd.read_csv(metrics_path)
    base = metrics_path.parent

    def _optional(name: str) -> pd.DataFrame | None:
        p = base / name
        if p.exists():
            return pd.read_csv(p)
        return None

    return ComparisonArtifacts(
        root=base,
        metrics=metrics,
        best_by_target=_optional("best_decoder_by_target.csv"),
        manifold_explained_variance=_optional("manifold_explained_variance.csv"),
        feature_set_summary=_optional("feature_set_performance_summary.csv"),
        loss_summary=_optional("sorted_information_loss_summary.csv"),
    )


def primary_score_column(target: str) -> str | None:
    if target not in PRIMARY_METRIC:
        return None
    return PRIMARY_METRIC[target][0]


def primary_direction(target: str) -> str | None:
    if target not in PRIMARY_METRIC:
        return None
    return PRIMARY_METRIC[target][1]


def add_score_column(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach a unified ``score`` column from each row's primary metric."""
    df = metrics.copy()
    scores: list[float] = []
    for _, row in df.iterrows():
        target = row.get("target_name")
        metric = PRIMARY_METRIC.get(str(target), (None, None))[0]
        if metric and metric in df.columns and pd.notna(row.get(metric)):
            scores.append(float(row[metric]))
        else:
            scores.append(float("nan"))
    df["score"] = scores
    return df


def leaderboard(
    metrics: pd.DataFrame,
    *,
    target: str | None = None,
) -> pd.DataFrame:
    """Compact sortable leaderboard with common identity columns."""
    df = add_score_column(metrics)
    if target is not None and "target_name" in df.columns:
        df = df[df["target_name"] == target].copy()
    cols = [
        c for c in (
            "target_name",
            "feature_set",
            "embedding_type",
            "manifold",
            "feature_mode",
            "decoder_name",
            "decode_window_s",
            "spike_source",
            "primary_metric",
            "score",
            "mean_position_error_cm",
            "r2",
            "balanced_accuracy",
            "total_compute_ms",
            "passes_realtime_gate",
        )
        if c in df.columns
    ]
    out = df[cols].copy() if cols else df
    if "score" in out.columns:
        # Sort within each target using primary-metric direction when possible.
        pieces = []
        group_col = "target_name" if "target_name" in out.columns else None
        groups = out.groupby(group_col, dropna=False) if group_col else [(None, out)]
        for tgt, g in groups:
            direction = primary_direction(str(tgt)) if tgt is not None else "higher"
            ascending = direction == "lower"
            pieces.append(g.sort_values("score", ascending=ascending, na_position="last"))
        out = pd.concat(pieces, ignore_index=True) if pieces else out
    return out


def highlight_best_mask(metrics: pd.DataFrame) -> pd.Series:
    """Boolean mask: best row per target by PRIMARY_METRIC."""
    df = add_score_column(metrics)
    mask = pd.Series(False, index=df.index)
    if "target_name" not in df.columns:
        return mask
    for target, g in df.groupby("target_name"):
        metric, direction = PRIMARY_METRIC.get(str(target), (None, None))
        if metric is None or metric not in g.columns:
            continue
        sub = g.dropna(subset=[metric])
        if sub.empty:
            continue
        idx = sub[metric].idxmin() if direction == "lower" else sub[metric].idxmax()
        mask.loc[idx] = True
    return mask


def filter_metrics(
    metrics: pd.DataFrame,
    *,
    targets: list[str] | None = None,
    feature_sets: list[str] | None = None,
    manifolds: list[str] | None = None,
    decoders: list[str] | None = None,
    decode_windows: list[float] | None = None,
    spike_sources: list[str] | None = None,
) -> pd.DataFrame:
    df = metrics.copy()
    if targets and "target_name" in df.columns:
        df = df[df["target_name"].isin(targets)]
    if feature_sets and "feature_set" in df.columns:
        df = df[df["feature_set"].isin(feature_sets)]
    manifold_col = "embedding_type" if "embedding_type" in df.columns else (
        "manifold" if "manifold" in df.columns else None
    )
    if manifolds and manifold_col:
        df = df[df[manifold_col].isin(manifolds)]
    if decoders and "decoder_name" in df.columns:
        df = df[df["decoder_name"].isin(decoders)]
    if decode_windows is not None and "decode_window_s" in df.columns:
        wins = {float(w) for w in decode_windows}
        df = df[df["decode_window_s"].astype(float).isin(wins)]
    if spike_sources and "spike_source" in df.columns:
        df = df[df["spike_source"].isin(spike_sources)]
    return df.reset_index(drop=True)


def load_decoded_example(comparison_root: Path, target: str = "position") -> dict[str, Any] | None:
    """Load a saved decoded-example NPZ/CSV if present."""
    examples = Path(comparison_root) / "decoded_examples"
    if not examples.exists():
        return None
    # Prefer npz with target in name
    for path in sorted(examples.glob(f"*{target}*")):
        if path.suffix == ".npz":
            data = np.load(path, allow_pickle=True)
            return {k: data[k] for k in data.files}
        if path.suffix == ".csv":
            return {"csv": pd.read_csv(path), "path": path}
    csvs = list(examples.glob("*.csv"))
    if csvs:
        return {"csv": pd.read_csv(csvs[0]), "path": csvs[0]}
    return None


def known_targets() -> tuple[str, ...]:
    return ALL_TARGETS
