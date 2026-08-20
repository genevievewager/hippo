"""Shared diagnostic calculations for Plotly and publication matplotlib.

Metrics and stored parquet columns are always computed on the **full** held-out
set. Functions marked ``for_display`` may subsample for plotting only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from realtime.prediction_artifacts import (
    CATEGORICAL_TARGETS,
    HEAD_DIRECTION_TARGET,
    POSITION_TARGET,
    SCALAR_TARGETS,
    class_labels_from_meta,
    proba_column_name,
    target_diagnostic_family,
)

# Visualization-only cap. Metrics always use the full held-out set.
MAX_DISPLAY_POINTS = 5000

# Occupancy mask: bins with fewer than this many samples are treated as missing.
DEFAULT_MIN_OCCUPANCY = 5
DEFAULT_SPATIAL_BINS = 12
DEFAULT_MAGNITUDE_BINS = 8
DEFAULT_ANGLE_BINS = 12
DEFAULT_SPARSE_LINKS = 250

PAIR_MISMATCH_MESSAGE = (
    "Held-out timestamps do not match (often different decode windows). "
    "Per-sample error difference is unavailable; both traces are shown separately."
)


@dataclass(frozen=True)
class PairAlignment:
    aligned: bool
    frame_a: pd.DataFrame
    frame_b: pd.DataFrame
    message: str | None = None


def downsample_for_display(
    df: pd.DataFrame,
    *,
    max_points: int = MAX_DISPLAY_POINTS,
) -> pd.DataFrame:
    """Evenly subsample rows for plotting. Do not use for metrics."""
    n = len(df)
    if n <= max_points:
        return df
    idx = np.linspace(0, n - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def sparse_link_indices(n: int, *, max_links: int = DEFAULT_SPARSE_LINKS) -> np.ndarray:
    """Indices for sparse true→pred displacement segments."""
    if n <= 0:
        return np.array([], dtype=int)
    if n <= max_links:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_links).astype(int)


def traces_time_aligned(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> bool:
    if "time" not in frame_a.columns or "time" not in frame_b.columns:
        return False
    ta = np.asarray(frame_a["time"], dtype=float)
    tb = np.asarray(frame_b["time"], dtype=float)
    if ta.shape != tb.shape:
        return False
    return bool(np.allclose(ta, tb, rtol=0.0, atol=1e-9, equal_nan=True))


def align_config_pair(frame_a: pd.DataFrame, frame_b: pd.DataFrame) -> PairAlignment:
    a = frame_a.reset_index(drop=True)
    b = frame_b.reset_index(drop=True)
    if traces_time_aligned(a, b):
        return PairAlignment(aligned=True, frame_a=a, frame_b=b, message=None)
    return PairAlignment(
        aligned=False, frame_a=a, frame_b=b, message=PAIR_MISMATCH_MESSAGE,
    )


def error_column(family: str) -> str:
    if family == "position":
        return "error_cm"
    if family == "head_direction":
        return "circular_error_deg"
    if family == "scalar":
        return "residual"
    return ""


def absolute_error_series(df: pd.DataFrame, family: str) -> np.ndarray | None:
    if family == "position" and "error_cm" in df.columns:
        return np.asarray(df["error_cm"], dtype=float)
    if family == "head_direction" and "circular_error_deg" in df.columns:
        return np.asarray(df["circular_error_deg"], dtype=float)
    if family == "scalar" and "residual" in df.columns:
        return np.abs(np.asarray(df["residual"], dtype=float))
    if family == "categorical" and "true" in df.columns and "pred" in df.columns:
        return (df["true"].astype(str) != df["pred"].astype(str)).to_numpy(dtype=float)
    return None


def arena_limits_from_trace(
    df: pd.DataFrame,
    *,
    pad_frac: float = 0.02,
) -> tuple[float, float, float, float]:
    xs = np.concatenate([
        np.asarray(df["true_x"], dtype=float),
        np.asarray(df["pred_x"], dtype=float),
    ])
    ys = np.concatenate([
        np.asarray(df["true_y"], dtype=float),
        np.asarray(df["pred_y"], dtype=float),
    ])
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    span = max(x_max - x_min, y_max - y_min, 1.0)
    pad = pad_frac * span
    # Shared square limits so Panel A/B cannot autoscaledifferently.
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half = 0.5 * span + pad
    return cx - half, cx + half, cy - half, cy + half


def spatial_error_map(
    df: pd.DataFrame,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    n_bins: int = DEFAULT_SPATIAL_BINS,
    min_occupancy: int = DEFAULT_MIN_OCCUPANCY,
) -> dict[str, np.ndarray]:
    """Mean error on a true-position grid; low-occupancy bins are NaN."""
    x = np.asarray(df["true_x"], dtype=float)
    y = np.asarray(df["true_y"], dtype=float)
    err = np.asarray(df["error_cm"], dtype=float)
    x_edges = np.linspace(x_min, x_max, n_bins + 1)
    y_edges = np.linspace(y_min, y_max, n_bins + 1)
    ix = np.clip(np.digitize(x, x_edges) - 1, 0, n_bins - 1)
    iy = np.clip(np.digitize(y, y_edges) - 1, 0, n_bins - 1)
    sums = np.zeros((n_bins, n_bins), dtype=float)
    counts = np.zeros((n_bins, n_bins), dtype=float)
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err)
    for i, j, e, ok in zip(ix, iy, err, valid):
        if not ok:
            continue
        sums[j, i] += e
        counts[j, i] += 1.0
    mean = np.full_like(sums, np.nan)
    mask = counts >= min_occupancy
    mean[mask] = sums[mask] / counts[mask]
    return {
        "mean_error": mean,
        "counts": counts,
        "x_edges": x_edges,
        "y_edges": y_edges,
        "mask": mask,
    }


def radial_shrinkage(
    df: pd.DataFrame,
    *,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> dict[str, Any]:
    """Compare distance-to-center of true vs predicted positions.

    Descriptive only: a slope < 1 is consistent with central regression of
    predictions, not proof of a causal mechanism.
    """
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    r_true = np.hypot(
        np.asarray(df["true_x"], dtype=float) - cx,
        np.asarray(df["true_y"], dtype=float) - cy,
    )
    r_pred = np.hypot(
        np.asarray(df["pred_x"], dtype=float) - cx,
        np.asarray(df["pred_y"], dtype=float) - cy,
    )
    ok = np.isfinite(r_true) & np.isfinite(r_pred)
    rt, rp = r_true[ok], r_pred[ok]
    slope, intercept = _linreg_slope_intercept(rt, rp)
    eps = 1e-6
    ratio = np.mean(rp[rt > eps] / rt[rt > eps]) if np.any(rt > eps) else float("nan")
    return {
        "center_xy": (cx, cy),
        "r_true": r_true,
        "r_pred": r_pred,
        "slope": slope,
        "intercept": intercept,
        "mean_radius_ratio": float(ratio),
    }


def _linreg_slope_intercept(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if x.size < 2:
        return float("nan"), float("nan")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    var = float(np.sum((x - xm) ** 2))
    if var <= 0:
        return float("nan"), float(ym)
    slope = float(np.sum((x - xm) * (y - ym)) / var)
    intercept = ym - slope * xm
    return slope, intercept


def compression_slope(true_vals: np.ndarray, pred_vals: np.ndarray) -> dict[str, float]:
    """Ordinary least-squares slope of pred ~ true (identity would be 1)."""
    t = np.asarray(true_vals, dtype=float).ravel()
    p = np.asarray(pred_vals, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    slope, intercept = _linreg_slope_intercept(t[ok], p[ok])
    return {"slope": slope, "intercept": intercept}


def magnitude_bins(
    true_vals: np.ndarray,
    pred_vals: np.ndarray,
    *,
    n_bins: int = DEFAULT_MAGNITUDE_BINS,
) -> pd.DataFrame:
    t = np.asarray(true_vals, dtype=float).ravel()
    p = np.asarray(pred_vals, dtype=float).ravel()
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]
    if t.size == 0:
        return pd.DataFrame(columns=["bin_center", "mean_true", "mean_pred", "mean_residual", "n"])
    edges = np.quantile(t, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0] -= 1e-12
    edges[-1] += 1e-12
    # Unique edges if true values are nearly constant.
    edges = np.unique(edges)
    if edges.size < 3:
        return pd.DataFrame([{
            "bin_center": float(np.mean(t)),
            "mean_true": float(np.mean(t)),
            "mean_pred": float(np.mean(p)),
            "mean_residual": float(np.mean(p - t)),
            "n": int(t.size),
        }])
    idx = np.clip(np.digitize(t, edges) - 1, 0, edges.size - 2)
    rows = []
    for b in range(edges.size - 1):
        mask = idx == b
        if not np.any(mask):
            continue
        tb, pb = t[mask], p[mask]
        rows.append({
            "bin_center": float(0.5 * (edges[b] + edges[b + 1])),
            "mean_true": float(np.mean(tb)),
            "mean_pred": float(np.mean(pb)),
            "mean_residual": float(np.mean(pb - tb)),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def circular_error_by_angle(
    true_deg: np.ndarray,
    circ_err: np.ndarray,
    *,
    n_bins: int = DEFAULT_ANGLE_BINS,
) -> pd.DataFrame:
    t = np.asarray(true_deg, dtype=float) % 360.0
    e = np.asarray(circ_err, dtype=float)
    edges = np.linspace(0.0, 360.0, n_bins + 1)
    idx = np.clip(np.digitize(t, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        rows.append({
            "bin_center_deg": float(0.5 * (edges[b] + edges[b + 1])),
            "mean_circular_error_deg": float(np.nanmean(e[mask])),
            "n": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def wrap_head_direction_series(true_deg: np.ndarray, pred_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift predicted series by ±360° per sample so overlays do not jump 359→1."""
    t = np.asarray(true_deg, dtype=float)
    p = np.asarray(pred_deg, dtype=float)
    # Bring pred into (true - 180, true + 180].
    delta = (p - t + 180.0) % 360.0 - 180.0
    return t, t + delta


def confusion_from_trace(
    df: pd.DataFrame,
    *,
    class_labels: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    true = df["true"].astype(str).to_numpy()
    pred = df["pred"].astype(str).to_numpy()
    labels = class_labels or sorted(set(true) | set(pred))
    mat = confusion_matrix(true, pred, labels=labels)
    return mat, list(labels)


def recall_by_true_class(
    df: pd.DataFrame,
    *,
    class_labels: list[str] | None = None,
) -> pd.DataFrame:
    mat, labels = confusion_from_trace(df, class_labels=class_labels)
    rows = []
    for i, lab in enumerate(labels):
        denom = float(mat[i].sum())
        rec = float(mat[i, i] / denom) if denom else float("nan")
        rows.append({"class": lab, "recall": rec, "n_true": int(denom)})
    return pd.DataFrame(rows)


def predicted_class_probability(df: pd.DataFrame, class_labels: list[str]) -> np.ndarray | None:
    """Probability assigned to the predicted label, or None if no proba columns."""
    used: set[str] = set()
    cols = []
    for lab in class_labels:
        col = proba_column_name(lab, used=used)
        cols.append(col)
        if col not in df.columns:
            return None
    pred = df["pred"].astype(str).to_numpy()
    out = np.full(len(df), np.nan, dtype=float)
    label_to_col = {str(lab): cols[i] for i, lab in enumerate(class_labels)}
    for i, lab in enumerate(pred):
        col = label_to_col.get(str(lab))
        if col is None:
            continue
        out[i] = float(df.iloc[i][col])
    return out


def top_class_entropy(df: pd.DataFrame, class_labels: list[str]) -> np.ndarray | None:
    used: set[str] = set()
    cols = [proba_column_name(lab, used=used) for lab in class_labels]
    if any(c not in df.columns for c in cols):
        return None
    P = df[cols].to_numpy(dtype=float)
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return -np.sum(P * np.log(P), axis=1)


def header_metrics_from_row(row: Mapping[str, Any], target: str) -> dict[str, float | None]:
    """Compact metric header from a comparison metrics row (full held-out set)."""
    family = target_diagnostic_family(str(target))

    def _f(name: str) -> float | None:
        val = row.get(name)
        try:
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return None
            return float(val)
        except (TypeError, ValueError):
            return None

    if family == "position":
        return {
            "mean_error_cm": _f("mean_position_error_cm"),
            "median_error_cm": _f("median_position_error_cm"),
            "p90_error_cm": _f("p90_position_error_cm"),
        }
    if family == "head_direction":
        return {
            "mean_circular_error_deg": _f("mean_circular_error_deg"),
            "median_circular_error_deg": _f("median_circular_error_deg"),
            "p90_circular_error_deg": _f("p90_circular_error_deg"),
        }
    if family == "categorical":
        return {
            "balanced_accuracy": _f("balanced_accuracy"),
            "macro_f1": _f("macro_f1"),
            "accuracy": _f("accuracy"),
        }
    out = {
        "r2": _f("r2"),
        "mae": _f("mae") if _f("mae") is not None else _f("mae_cm"),
        "rmse": _f("rmse") if _f("rmse") is not None else _f("rmse_cm"),
    }
    return out


def metric_delta(
    a: dict[str, float | None],
    b: dict[str, float | None],
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in a:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            out[key] = None
        else:
            out[key] = va - vb
    return out


__all__ = [
    "CATEGORICAL_TARGETS",
    "HEAD_DIRECTION_TARGET",
    "PAIR_MISMATCH_MESSAGE",
    "POSITION_TARGET",
    "SCALAR_TARGETS",
    "PairAlignment",
    "absolute_error_series",
    "align_config_pair",
    "arena_limits_from_trace",
    "class_labels_from_meta",
    "circular_error_by_angle",
    "compression_slope",
    "confusion_from_trace",
    "downsample_for_display",
    "header_metrics_from_row",
    "magnitude_bins",
    "metric_delta",
    "predicted_class_probability",
    "radial_shrinkage",
    "recall_by_true_class",
    "sparse_link_indices",
    "spatial_error_map",
    "target_diagnostic_family",
    "top_class_entropy",
    "traces_time_aligned",
    "wrap_head_direction_series",
]
