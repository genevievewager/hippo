"""Lazy loaders for held-out decoder-comparison prediction traces.

The UI should look up traces by ``config_id`` only. Filename conventions live
in ``realtime.prediction_artifacts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from realtime.comparison_metrics_union import ensure_config_ids, make_config_id
from realtime.decoder_comparison import ALL_TARGETS, PRIMARY_METRIC
from realtime.decoding_diagnostics_prep import (
    PAIR_MISMATCH_MESSAGE,
    PairAlignment,
    align_config_pair,
    header_metrics_from_row,
    target_diagnostic_family,
)
from realtime.prediction_artifacts import (
    prediction_parquet_path,
    read_prediction_trace,
)
from ui.services.results import find_metrics_csv

LEGACY_TRACE_MESSAGE = (
    "Prediction-level diagnostics are unavailable for this legacy run. "
    "Re-run Decoder Benchmark to generate diagnostic traces."
)

TARGET_DISPLAY_NAMES: dict[str, str] = {
    "position": "Position",
    "speed": "Speed",
    "acceleration": "Acceleration",
    "head_direction": "Head Direction",
    "distance_to_wall": "Distance to Wall",
    "spatial_context": "Spatial Context",
    "movement_state": "Movement State",
    "wall_distance_bin": "Wall Distance Bin",
}


def _maybe_cache(fn):
    try:
        import streamlit as st

        return st.cache_data(show_spinner=False)(fn)
    except Exception:
        return fn


def comparison_source_root(experiment_dir: Path, spike_source: str = "sorted") -> Path:
    exp = Path(experiment_dir)
    preferred = exp / "decoder_comparison" / str(spike_source)
    if preferred.exists():
        return preferred
    metrics = find_metrics_csv(exp)
    if metrics is not None:
        return metrics.parent
    return preferred


def load_comparison_metrics(
    experiment_dir: Path,
    *,
    spike_source: str | None = None,
) -> pd.DataFrame:
    """Load the comparison table (no prediction files)."""
    path = find_metrics_csv(Path(experiment_dir))
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    df = ensure_config_ids(df)
    if spike_source and "spike_source" in df.columns:
        df = df[df["spike_source"].astype(str) == str(spike_source)]
    return df.reset_index(drop=True)


def available_targets(metrics: pd.DataFrame) -> list[str]:
    if metrics.empty or "target_name" not in metrics.columns:
        return list(ALL_TARGETS)
    present = {str(t) for t in metrics["target_name"].dropna().unique()}
    ordered = [t for t in ALL_TARGETS if t in present]
    extra = sorted(present - set(ordered))
    return ordered + extra


def target_configs(metrics: pd.DataFrame, target: str) -> pd.DataFrame:
    if metrics.empty or "target_name" not in metrics.columns:
        return metrics
    out = metrics[metrics["target_name"].astype(str) == str(target)].copy()
    if "exclusion_reason" in out.columns:
        out = out[out["exclusion_reason"].isna() | (out["exclusion_reason"].astype(str) == "")]
    return out.reset_index(drop=True)


def best_config_row(metrics: pd.DataFrame, target: str) -> pd.Series | None:
    sub = target_configs(metrics, target)
    if sub.empty:
        return None
    metric_key, direction = PRIMARY_METRIC.get(target, (None, "higher"))
    if metric_key is None or metric_key not in sub.columns:
        return sub.iloc[0]
    scored = pd.to_numeric(sub[metric_key], errors="coerce")
    if scored.notna().any():
        idx = scored.idxmin() if direction == "lower" else scored.idxmax()
        return sub.loc[idx]
    return sub.iloc[0]


@_maybe_cache
def _read_parquet_cached(path_str: str, mtime_ns: int) -> tuple[pd.DataFrame, dict[str, str]]:
    df, meta = read_prediction_trace(Path(path_str))
    return df, meta


@dataclass
class PredictionBundle:
    config_id: str
    target: str
    family: str
    metrics_row: pd.Series | None
    frame: pd.DataFrame
    meta: dict[str, str]
    path: Path


def prediction_path_for(
    experiment_dir: Path,
    config_id: str,
    *,
    spike_source: str = "sorted",
) -> Path:
    root = comparison_source_root(experiment_dir, spike_source)
    return prediction_parquet_path(root, config_id)


def has_prediction_trace(
    experiment_dir: Path,
    config_id: str,
    *,
    spike_source: str = "sorted",
) -> bool:
    return prediction_path_for(
        experiment_dir, config_id, spike_source=spike_source,
    ).is_file()


def any_prediction_traces(
    experiment_dir: Path,
    *,
    spike_source: str = "sorted",
) -> bool:
    root = comparison_source_root(experiment_dir, spike_source)
    pred_dir = root / "predictions"
    if not pred_dir.is_dir():
        return False
    return any(pred_dir.glob("*.parquet"))


def load_prediction_trace(
    experiment_dir: Path,
    config_id: str,
    *,
    metrics_row: pd.Series | None = None,
    spike_source: str = "sorted",
) -> PredictionBundle | None:
    path = prediction_path_for(experiment_dir, config_id, spike_source=spike_source)
    if not path.is_file():
        return None
    frame, meta = _read_parquet_cached(str(path), path.stat().st_mtime_ns)
    target = str(
        (metrics_row.get("target_name") if metrics_row is not None else None)
        or meta.get("target_name")
        or ""
    )
    return PredictionBundle(
        config_id=str(config_id),
        target=target,
        family=target_diagnostic_family(target),
        metrics_row=metrics_row,
        frame=frame,
        meta=meta,
        path=path,
    )


def load_config_pair(
    experiment_dir: Path,
    config_id_a: str,
    config_id_b: str,
    *,
    row_a: pd.Series | None = None,
    row_b: pd.Series | None = None,
    spike_source: str = "sorted",
) -> tuple[PredictionBundle | None, PredictionBundle | None, PairAlignment | None]:
    a = load_prediction_trace(
        experiment_dir, config_id_a, metrics_row=row_a, spike_source=spike_source,
    )
    b = load_prediction_trace(
        experiment_dir, config_id_b, metrics_row=row_b, spike_source=spike_source,
    )
    if a is None or b is None:
        return a, b, None
    alignment = align_config_pair(a.frame, b.frame)
    return a, b, alignment


def config_label(row: pd.Series | dict[str, Any]) -> str:
    getter = row.get if hasattr(row, "get") else lambda k, d=None: row[k] if k in row else d
    emb = getter("embedding_type") or getter("feature_mode") or "?"
    dec = getter("decoder_name") or "?"
    fs = getter("feature_set") or "?"
    w = getter("decode_window_s")
    try:
        wtxt = f"{float(w):.3f}s" if w is not None and not pd.isna(w) else "?"
    except (TypeError, ValueError):
        wtxt = str(w)
    k = getter("manifold_n_components")
    ktxt = ""
    try:
        if k is not None and not pd.isna(k):
            ktxt = f" · k={int(k)}"
    except (TypeError, ValueError):
        pass
    return f"{emb} · {dec} · {fs} · W={wtxt}{ktxt}"


def row_by_config_id(metrics: pd.DataFrame, config_id: str) -> pd.Series | None:
    if metrics.empty or "config_id" not in metrics.columns:
        return None
    hit = metrics[metrics["config_id"].astype(str) == str(config_id)]
    if hit.empty:
        return None
    return hit.iloc[0]


def counts_baseline_row(
    metrics: pd.DataFrame,
    *,
    target: str,
    decoder_name: str | None = None,
    decode_window_s: float | None = None,
) -> pd.Series | None:
    """Same target (and optionally decoder/W) with a counts / identity embedding."""
    sub = target_configs(metrics, target)
    if sub.empty:
        return None
    emb = sub["embedding_type"].astype(str) if "embedding_type" in sub.columns else None
    mode = sub["feature_mode"].astype(str) if "feature_mode" in sub.columns else None
    is_counts = pd.Series(False, index=sub.index)
    if emb is not None:
        is_counts = is_counts | emb.isin(("counts", "identity"))
    if mode is not None:
        is_counts = is_counts | mode.isin(("counts", "identity"))
    pool = sub[is_counts]
    if pool.empty:
        return None
    if decoder_name is not None and "decoder_name" in pool.columns:
        hit = pool[pool["decoder_name"].astype(str) == str(decoder_name)]
        if not hit.empty:
            pool = hit
    if decode_window_s is not None and "decode_window_s" in pool.columns:
        hit = pool[np_isclose_window(pool["decode_window_s"], decode_window_s)]
        if not hit.empty:
            pool = hit
    return best_config_row(pool, target) if "target_name" in pool.columns else pool.iloc[0]


def np_isclose_window(series: pd.Series, window: float) -> pd.Series:
    nums = pd.to_numeric(series, errors="coerce")
    return (nums - float(window)).abs() < 1e-9


def same_decoder_other_embedding(
    metrics: pd.DataFrame,
    row: pd.Series,
) -> pd.DataFrame:
    sub = target_configs(metrics, str(row.get("target_name")))
    if sub.empty:
        return sub
    mask = sub["decoder_name"].astype(str) == str(row.get("decoder_name"))
    if "decode_window_s" in sub.columns:
        mask = mask & np_isclose_window(sub["decode_window_s"], float(row["decode_window_s"]))
    if "feature_set" in sub.columns:
        mask = mask & (sub["feature_set"].astype(str) == str(row.get("feature_set")))
    mask = mask & (sub["config_id"].astype(str) != str(row.get("config_id")))
    return sub[mask].reset_index(drop=True)


def same_embedding_other_decoder(
    metrics: pd.DataFrame,
    row: pd.Series,
) -> pd.DataFrame:
    sub = target_configs(metrics, str(row.get("target_name")))
    if sub.empty:
        return sub
    mask = sub["embedding_type"].astype(str) == str(row.get("embedding_type"))
    if "decode_window_s" in sub.columns:
        mask = mask & np_isclose_window(sub["decode_window_s"], float(row["decode_window_s"]))
    if "feature_set" in sub.columns:
        mask = mask & (sub["feature_set"].astype(str) == str(row.get("feature_set")))
    mask = mask & (sub["config_id"].astype(str) != str(row.get("config_id")))
    return sub[mask].reset_index(drop=True)


__all__ = [
    "LEGACY_TRACE_MESSAGE",
    "PAIR_MISMATCH_MESSAGE",
    "TARGET_DISPLAY_NAMES",
    "PredictionBundle",
    "any_prediction_traces",
    "available_targets",
    "best_config_row",
    "comparison_source_root",
    "config_label",
    "counts_baseline_row",
    "has_prediction_trace",
    "header_metrics_from_row",
    "load_comparison_metrics",
    "load_config_pair",
    "load_prediction_trace",
    "make_config_id",
    "row_by_config_id",
    "same_decoder_other_embedding",
    "same_embedding_other_decoder",
    "target_configs",
]
