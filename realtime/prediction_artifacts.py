"""Held-out prediction traces for decoder-comparison diagnostics.

Each successful F×E×D×W evaluation writes a snappy Parquet file keyed by
``config_id``. Scoring is unchanged; this only persists ``y_true`` / ``y_pred``
(and class probabilities when available) for the causal held-out test split.

Residual convention (scalars): ``residual = pred - true``.
Circular error: shortest arc in degrees (see ``circular_error_from_degrees``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from realtime.comparison_metrics_union import make_config_id
from realtime.decoder_models import TARGET_FAMILY
from realtime.decoding_targets import angles_from_sin_cos, circular_error_deg

SPLIT_HELD_OUT_TEST = "held_out_test"

POSITION_TARGET = "position"
HEAD_DIRECTION_TARGET = "head_direction"
SCALAR_TARGETS = ("speed", "acceleration", "distance_to_wall")
CATEGORICAL_TARGETS = tuple(
    name for name, family in TARGET_FAMILY.items() if family == "categorical"
)

META_CONFIG_ID = b"config_id"
META_TARGET = b"target_name"
META_SPLIT = b"split"
META_CLASS_LABELS = b"class_labels"


def prediction_dir(comparison_root: Path) -> Path:
    """``decoder_comparison/<spike_source>/predictions``."""
    return Path(comparison_root) / "predictions"


def prediction_parquet_path(comparison_root: Path, config_id: str) -> Path:
    return prediction_dir(comparison_root) / f"{config_id}.parquet"


def sanitize_class_token(label: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")
    return text or "class"


def proba_column_name(label: Any, *, used: set[str] | None = None) -> str:
    base = f"proba_{sanitize_class_token(label)}"
    if used is None:
        return base
    name = base
    n = 2
    while name in used:
        name = f"{base}_{n}"
        n += 1
    used.add(name)
    return name


def target_diagnostic_family(target: str) -> str:
    if target == POSITION_TARGET:
        return "position"
    if target == HEAD_DIRECTION_TARGET:
        return "head_direction"
    if target in SCALAR_TARGETS:
        return "scalar"
    if target in CATEGORICAL_TARGETS:
        return "categorical"
    family = TARGET_FAMILY.get(target)
    if family == "categorical":
        return "categorical"
    return "scalar"


def _window_frame(time: np.ndarray, decode_window_s: float, update_dt_s: float) -> dict[str, np.ndarray]:
    t = np.asarray(time, dtype=float)
    return {
        "time": t,
        "decode_time": t,
        "window_start": t - float(decode_window_s),
        "window_end": t,
        "decode_window_s": np.full(t.shape, float(decode_window_s)),
        "update_dt_s": np.full(t.shape, float(update_dt_s)),
    }


def build_prediction_frame(
    *,
    config_id: str,
    target: str,
    behavior_test: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    decode_window_s: float,
    update_dt_s: float,
    class_labels: list[Any] | None = None,
    proba: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Build the held-out trace DataFrame and parquet metadata (str values)."""
    time = behavior_test["time"].to_numpy(dtype=float)
    n = len(time)
    base = _window_frame(time, decode_window_s, update_dt_s)
    ids = np.full(n, str(config_id), dtype=object)
    split = np.full(n, SPLIT_HELD_OUT_TEST, dtype=object)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    family = target_diagnostic_family(target)
    meta = {
        "config_id": str(config_id),
        "target_name": str(target),
        "split": SPLIT_HELD_OUT_TEST,
        "target_family": family,
    }

    if family == "position":
        if y_pred_arr.ndim == 1:
            raise ValueError("Position predictions must be 2D")
        true_x = behavior_test["x"].to_numpy(dtype=float) if "x" in behavior_test.columns else y_true_arr[:, 0]
        true_y = behavior_test["y"].to_numpy(dtype=float) if "y" in behavior_test.columns else y_true_arr[:, 1]
        pred_x = y_pred_arr[:, 0].astype(float)
        pred_y = y_pred_arr[:, 1].astype(float)
        error_cm = np.hypot(pred_x - true_x, pred_y - true_y)
        df = pd.DataFrame({
            "config_id": ids,
            "split": split,
            **base,
            "true_x": true_x,
            "true_y": true_y,
            "pred_x": pred_x,
            "pred_y": pred_y,
            "error_cm": error_cm,
        })
        return df, meta

    if family == "head_direction":
        if y_true_arr.ndim == 1:
            raise ValueError("Head-direction targets must be sin/cos columns")
        true_sin = y_true_arr[:, 0].astype(float)
        true_cos = y_true_arr[:, 1].astype(float)
        pred_sin = y_pred_arr[:, 0].astype(float)
        pred_cos = y_pred_arr[:, 1].astype(float)
        true_rad = angles_from_sin_cos(true_sin, true_cos)
        pred_rad = angles_from_sin_cos(pred_sin, pred_cos)
        true_deg = np.degrees(true_rad)
        pred_deg = np.degrees(pred_rad)
        circ = circular_error_deg(true_rad, pred_rad)
        df = pd.DataFrame({
            "config_id": ids,
            "split": split,
            **base,
            "true_deg": true_deg,
            "pred_deg": pred_deg,
            "circular_error_deg": circ,
            "true_sin": true_sin,
            "true_cos": true_cos,
            "pred_sin": pred_sin,
            "pred_cos": pred_cos,
        })
        return df, meta

    if family == "categorical":
        labels = [str(x) for x in (class_labels or [])]
        if not labels:
            labels = sorted({str(x) for x in np.ravel(y_true_arr)} | {str(x) for x in np.ravel(y_pred_arr)})
        meta["class_labels"] = json.dumps(labels)
        data: dict[str, Any] = {
            "config_id": ids,
            "split": split,
            **base,
            "true": np.asarray(y_true_arr).astype(str).ravel(),
            "pred": np.asarray(y_pred_arr).astype(str).ravel(),
        }
        if proba is not None:
            proba_arr = np.asarray(proba, dtype=float)
            if proba_arr.ndim != 2 or proba_arr.shape[0] != n:
                raise ValueError(
                    f"proba shape {proba_arr.shape} incompatible with n={n}"
                )
            used: set[str] = set()
            n_cols = min(proba_arr.shape[1], len(labels)) if labels else proba_arr.shape[1]
            if not labels:
                labels = [str(i) for i in range(proba_arr.shape[1])]
                meta["class_labels"] = json.dumps(labels)
                n_cols = proba_arr.shape[1]
            col_names = []
            for i in range(n_cols):
                col = proba_column_name(labels[i], used=used)
                col_names.append(col)
                data[col] = proba_arr[:, i]
            meta["proba_columns"] = json.dumps(col_names)
        df = pd.DataFrame(data)
        return df, meta

    true_1d = np.asarray(y_true_arr, dtype=float).ravel()
    pred_1d = np.asarray(y_pred_arr, dtype=float).ravel()
    residual = pred_1d - true_1d
    df = pd.DataFrame({
        "config_id": ids,
        "split": split,
        **base,
        "true": true_1d,
        "pred": pred_1d,
        "residual": residual,
    })
    return df, meta


def write_prediction_trace(
    comparison_root: Path,
    *,
    config_id: str,
    target: str,
    behavior_test: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    decode_window_s: float,
    update_dt_s: float,
    class_labels: list[Any] | None = None,
    proba: np.ndarray | None = None,
) -> Path:
    """Write held-out predictions as snappy Parquet. Returns the file path."""
    df, meta = build_prediction_frame(
        config_id=config_id,
        target=target,
        behavior_test=behavior_test,
        y_true=y_true,
        y_pred=y_pred,
        decode_window_s=decode_window_s,
        update_dt_s=update_dt_s,
        class_labels=class_labels,
        proba=proba,
    )
    path = prediction_parquet_path(comparison_root, config_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    encoded = {key.encode("utf-8"): str(val).encode("utf-8") for key, val in meta.items()}
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        **encoded,
    })
    pq.write_table(table, path, compression="snappy")
    return path


def read_prediction_metadata(path: Path) -> dict[str, str]:
    pf = pq.ParquetFile(path)
    raw = pf.schema_arrow.metadata or {}
    out: dict[str, str] = {}
    for key, val in raw.items():
        k = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        v = val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
        out[k] = v
    return out


def read_prediction_trace(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    table = pq.read_table(path)
    meta_raw = table.schema.metadata or {}
    meta: dict[str, str] = {}
    for key, val in meta_raw.items():
        k = key.decode("utf-8") if isinstance(key, (bytes, bytearray)) else str(key)
        if k.startswith("pandas") or k.startswith("ARROW:"):
            continue
        v = val.decode("utf-8") if isinstance(val, (bytes, bytearray)) else str(val)
        meta[k] = v
    df = table.to_pandas()
    return df, meta


def class_labels_from_meta(meta: Mapping[str, str]) -> list[str]:
    raw = meta.get("class_labels")
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, list):
        return [str(x) for x in parsed]
    return []


def stamp_config_id(row: dict[str, Any]) -> str:
    """Set ``row['config_id']`` from identity fields; return the id."""
    cid = make_config_id(row)
    row["config_id"] = cid
    return cid
