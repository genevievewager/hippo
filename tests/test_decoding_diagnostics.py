"""Held-out prediction artifacts, config_id, and diagnostic prep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import confusion_matrix

from realtime.comparison_metrics_union import ensure_config_ids, make_config_id
from realtime.decoding_diagnostics_prep import (
    PAIR_MISMATCH_MESSAGE,
    align_config_pair,
    confusion_from_trace,
    traces_time_aligned,
)
from realtime.decoding_targets import circular_error_from_degrees
from realtime.prediction_artifacts import (
    build_prediction_frame,
    read_prediction_trace,
    write_prediction_trace,
)


def _identity(**overrides) -> dict:
    row = {
        "spike_source": "sorted",
        "target_name": "speed",
        "decoder_name": "ridge",
        "feature_set": "counts",
        "feature_mode": "global_pca",
        "embedding_type": "global_pca",
        "decode_window_s": 0.250,
        "manifold_n_components": 3,
        "n_neighbors": None,
        "n_landmarks": None,
    }
    row.update(overrides)
    return row


def test_config_id_is_deterministic_and_unique():
    a = make_config_id(_identity())
    b = make_config_id(_identity())
    c = make_config_id(_identity(decoder_name="random_forest"))
    assert a == b
    assert a != c
    assert "/" not in a and " " not in a


def test_ensure_config_ids_unique_per_metrics_row():
    rows = [
        _identity(decoder_name="ridge"),
        _identity(decoder_name="elasticnet"),
        _identity(target_name="position", decoder_name="ridge"),
    ]
    df = ensure_config_ids(pd.DataFrame(rows))
    assert df["config_id"].nunique() == 3
    assert df["config_id"].is_unique


def test_position_error_cm_matches_euclidean(tmp_path: Path):
    n = 8
    beh = pd.DataFrame({
        "time": np.linspace(0, 1, n),
        "x": np.arange(n, dtype=float),
        "y": np.arange(n, dtype=float) * 0.5,
    })
    y_true = beh[["x", "y"]].to_numpy()
    y_pred = y_true + np.array([3.0, 4.0])
    cid = make_config_id(_identity(target_name="position"))
    path = write_prediction_trace(
        tmp_path,
        config_id=cid,
        target="position",
        behavior_test=beh,
        y_true=y_true,
        y_pred=y_pred,
        decode_window_s=0.25,
        update_dt_s=0.05,
    )
    df, meta = read_prediction_trace(path)
    expected = np.hypot(y_pred[:, 0] - y_true[:, 0], y_pred[:, 1] - y_true[:, 1])
    np.testing.assert_allclose(df["error_cm"].to_numpy(), expected)
    assert np.allclose(expected, 5.0)
    assert meta["config_id"] == cid
    assert meta["target_name"] == "position"
    assert (df["config_id"] == cid).all()


def test_scalar_residual_is_pred_minus_true(tmp_path: Path):
    n = 5
    true = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    pred = np.array([1.5, 1.0, 3.0, 6.0, 8.0])
    beh = pd.DataFrame({"time": np.arange(n, dtype=float)})
    cid = make_config_id(_identity())
    path = write_prediction_trace(
        tmp_path,
        config_id=cid,
        target="speed",
        behavior_test=beh,
        y_true=true,
        y_pred=pred,
        decode_window_s=0.25,
        update_dt_s=0.05,
    )
    df, _ = read_prediction_trace(path)
    np.testing.assert_allclose(df["residual"].to_numpy(), pred - true)


def test_head_direction_wraparound_359_vs_1():
    err = circular_error_from_degrees(np.array([359.0]), np.array([1.0]))
    np.testing.assert_allclose(err, [2.0], atol=1e-6)
    err2 = circular_error_from_degrees(np.array([1.0]), np.array([359.0]))
    np.testing.assert_allclose(err2, [2.0], atol=1e-6)


def test_head_direction_parquet_circular_error(tmp_path: Path):
    true_deg = np.array([359.0, 90.0])
    pred_deg = np.array([1.0, 90.0])
    true_rad = np.deg2rad(true_deg)
    pred_rad = np.deg2rad(pred_deg)
    y_true = np.column_stack([np.sin(true_rad), np.cos(true_rad)])
    y_pred = np.column_stack([np.sin(pred_rad), np.cos(pred_rad)])
    beh = pd.DataFrame({"time": np.array([0.0, 1.0])})
    cid = make_config_id(_identity(target_name="head_direction"))
    path = write_prediction_trace(
        tmp_path,
        config_id=cid,
        target="head_direction",
        behavior_test=beh,
        y_true=y_true,
        y_pred=y_pred,
        decode_window_s=0.25,
        update_dt_s=0.05,
    )
    df, _ = read_prediction_trace(path)
    np.testing.assert_allclose(df["circular_error_deg"].to_numpy()[0], 2.0, atol=1e-5)


def test_categorical_proba_order_and_confusion(tmp_path: Path):
    labels = ["center", "corner", "wall"]
    y_true = np.array(["center", "corner", "wall", "center"])
    y_pred = np.array(["center", "wall", "wall", "corner"])
    proba = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.2, 0.7],
        [0.05, 0.1, 0.85],
        [0.2, 0.6, 0.2],
    ])
    beh = pd.DataFrame({"time": np.arange(4, dtype=float)})
    cid = make_config_id(_identity(target_name="spatial_context", decoder_name="logreg"))
    df, meta = build_prediction_frame(
        config_id=cid,
        target="spatial_context",
        behavior_test=beh,
        y_true=y_true,
        y_pred=y_pred,
        decode_window_s=0.25,
        update_dt_s=0.05,
        class_labels=labels,
        proba=proba,
    )
    path = write_prediction_trace(
        tmp_path,
        config_id=cid,
        target="spatial_context",
        behavior_test=beh,
        y_true=y_true,
        y_pred=y_pred,
        decode_window_s=0.25,
        update_dt_s=0.05,
        class_labels=labels,
        proba=proba,
    )
    loaded, loaded_meta = read_prediction_trace(path)
    assert loaded_meta["class_labels"] == meta["class_labels"]
    np.testing.assert_allclose(loaded["proba_center"] + loaded["proba_corner"] + loaded["proba_wall"], 1.0, atol=1e-6)
    mat, labs = confusion_from_trace(loaded, class_labels=labels)
    expected = confusion_matrix(y_true, y_pred, labels=labels)
    np.testing.assert_array_equal(mat, expected)
    assert labs == labels


def test_pair_alignment_refuses_mismatched_times():
    a = pd.DataFrame({"time": [0.0, 0.05, 0.10], "error_cm": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"time": [0.0, 0.05, 0.20], "error_cm": [1.0, 2.0, 3.0]})
    assert not traces_time_aligned(a, b)
    pair = align_config_pair(a, b)
    assert pair.aligned is False
    assert PAIR_MISMATCH_MESSAGE in (pair.message or "")


def test_pair_alignment_accepts_matching_times():
    a = pd.DataFrame({"time": [0.0, 0.05, 0.10], "error_cm": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"time": [0.0, 0.05, 0.10], "error_cm": [0.5, 2.5, 1.0]})
    pair = align_config_pair(a, b)
    assert pair.aligned is True
    np.testing.assert_allclose(
        pair.frame_a["error_cm"] - pair.frame_b["error_cm"],
        [0.5, -0.5, 2.0],
    )


def test_export_position_diagnostics_png(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from visualization.publication_decoding_diagnostics import export_diagnostics_figure

    n = 40
    rng = np.random.default_rng(0)
    true_xy = rng.normal(size=(n, 2))
    pred_xy = true_xy * 0.6
    beh = pd.DataFrame({"time": np.linspace(0, 10, n), "x": true_xy[:, 0], "y": true_xy[:, 1]})
    cid = make_config_id(_identity(target_name="position"))
    df, meta = build_prediction_frame(
        config_id=cid,
        target="position",
        behavior_test=beh,
        y_true=true_xy,
        y_pred=pred_xy,
        decode_window_s=0.25,
        update_dt_s=0.05,
    )
    path = export_diagnostics_figure(
        experiment_dir=tmp_path,
        target="position",
        frame_a=df,
        meta_a=meta,
        figures_dir=tmp_path / "figures",
    )
    assert path.is_file()
    assert path.stat().st_size > 0

