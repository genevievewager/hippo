"""Tests for sorted-only deployment decoder selection."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from realtime.deployment_selection import (
    DEPLOYMENT_SPIKE_SOURCE,
    build_all_window_scores_table,
    build_best_realtime_decoders_payload,
    load_best_realtime_decoders,
    warn_if_uniform_window,
    write_deployment_selection_artifacts,
)


def _fake_metrics() -> pd.DataFrame:
    rows = []
    for target, metric, higher in (
        ("position", "mean_position_error_cm", False),
        ("speed", "r2", True),
        ("spatial_context", "balanced_accuracy", True),
    ):
        for w in (0.05, 0.10, 0.25, 0.50, 1.0):
            for dec in ("ridge", "random_forest_regressor") if target != "spatial_context" else (
                "logistic_regression", "random_forest_classifier",
            ):
                # Make different windows win for different targets
                if target == "position":
                    val = abs(w - 0.5) + (0.1 if "ridge" in dec else 0.0)
                elif target == "speed":
                    val = 1.0 - abs(w - 0.1) - (0.05 if "ridge" in dec else 0.0)
                else:
                    val = 0.9 - abs(w - 0.25) * 0.2
                rows.append({
                    "spike_source": "sorted",
                    "target_name": target,
                    "decoder_name": dec,
                    "feature_type": "counts",
                    "decode_window_s": w,
                    "primary_metric": metric,
                    metric: val,
                    "n_train_samples": 100,
                    "n_test_samples": 40,
                    "realtime_compatible": True,
                })
    return pd.DataFrame(rows)


def _fake_best() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "target_name": "position",
            "target_family": "continuous",
            "primary_metric": "mean_position_error_cm",
            "best_decoder_name": "random_forest_regressor",
            "best_decode_window_s": 0.5,
            "recommended_realtime_window_s": 0.5,
            "recommended_realtime_decoder_name": "random_forest_regressor",
            "best_feature_type": "counts",
            "best_manifold_type": "none",
            "best_manifold_grouping": None,
            "best_manifold_n_components": None,
            "best_metric_value": 0.0,
            "spike_source": "sorted",
            "model_path": "models/best_position_decoder.joblib",
            "best_window_model_path": "models/by_window/counts/kna/w0.500s/position.joblib",
            "realtime_model_path": "models/by_window/counts/kna/w0.500s/position.joblib",
            "manifold_transform_path": None,
            "decoder_config_json": json.dumps({
                "decoder_name": "random_forest_regressor",
                "decode_window_s": 0.5,
                "feature_type": "counts",
                "target_name": "position",
            }),
            "realtime_decoder_config_json": json.dumps({
                "decoder_name": "random_forest_regressor",
                "decode_window_s": 0.5,
                "feature_type": "counts",
                "target_name": "position",
            }),
        },
        {
            "target_name": "speed",
            "target_family": "continuous",
            "primary_metric": "r2",
            "best_decoder_name": "ridge",
            "best_decode_window_s": 0.1,
            "recommended_realtime_window_s": 0.1,
            "recommended_realtime_decoder_name": "ridge",
            "best_feature_type": "counts",
            "best_manifold_type": "none",
            "best_manifold_grouping": None,
            "best_manifold_n_components": None,
            "best_metric_value": 0.95,
            "spike_source": "sorted",
            "model_path": "models/best_speed_decoder.joblib",
            "best_window_model_path": "models/by_window/counts/kna/w0.100s/speed.joblib",
            "realtime_model_path": "models/by_window/counts/kna/w0.100s/speed.joblib",
            "manifold_transform_path": None,
            "decoder_config_json": json.dumps({
                "decoder_name": "ridge",
                "decode_window_s": 0.1,
                "feature_type": "counts",
                "target_name": "speed",
            }),
            "realtime_decoder_config_json": json.dumps({
                "decoder_name": "ridge",
                "decode_window_s": 0.1,
                "feature_type": "counts",
                "target_name": "speed",
            }),
        },
    ])


def test_scores_table_has_all_windows():
    scores = build_all_window_scores_table(_fake_metrics(), best_df=_fake_best())
    assert not scores.empty
    assert set(scores["causal_window_s"]) == {0.05, 0.10, 0.25, 0.50, 1.0}
    assert scores["spike_source"].eq(DEPLOYMENT_SPIKE_SOURCE).all()
    assert {"target", "decoder", "feature_mode", "metric_value", "selected_best"} <= set(scores.columns)


def test_scores_table_preserves_composite_feature_modes():
    """F×E schema: feature_type is base spikes; feature_mode is the composite key."""
    from realtime.deployment_selection import _composite_feature_mode

    rows = []
    for mode, emb in (
        ("counts", "identity"),
        ("global_pca", "global_pca"),
        ("layer_pca", "layer_pca"),
        ("global_isomap", "global_isomap"),
    ):
        rows.append({
            "spike_source": "sorted",
            "target_name": "speed",
            "decoder_name": "ridge",
            "feature_type": "counts",
            "feature_mode": mode,
            "embedding_type": emb,
            "decode_window_s": 0.25,
            "primary_metric": "r2",
            "r2": 0.5,
            "n_train_samples": 100,
            "n_test_samples": 40,
            "realtime_compatible": mode != "global_isomap",
            "manifold_n_components": None if mode == "counts" else 8,
        })
    metrics = pd.DataFrame(rows)
    scores = build_all_window_scores_table(metrics)
    assert set(scores["feature_mode"]) == {
        "counts", "global_pca", "layer_pca", "global_isomap",
    }
    # Helper must not collapse composite modes back to base feature_type.
    assert _composite_feature_mode(metrics.iloc[2]) == "layer_pca"


def test_uniform_window_warning_when_collapsed():
    best = _fake_best()
    best["recommended_realtime_window_s"] = 0.25
    best["best_decode_window_s"] = 0.25
    # Only one tested window → strong warning
    scores = build_all_window_scores_table(_fake_metrics())
    scores = scores[scores["causal_window_s"] == 0.25]
    with pytest.warns(UserWarning, match="same causal window"):
        msg = warn_if_uniform_window(best, scores)
    assert msg is not None
    assert "0.250" in msg or "0.25" in msg


def test_write_and_load_deployable_registry(tmp_path: Path):
    exp = tmp_path / "run001"
    comp = exp / "decoder_comparison" / "sorted"
    comp.mkdir(parents=True)
    metrics = _fake_metrics()
    best = _fake_best()
    metrics.to_csv(comp / "decoder_comparison_metrics.csv", index=False)
    best.to_csv(comp / "best_decoder_by_target.csv", index=False)
    (tmp_path / "sorting").mkdir()
    (tmp_path / "sorting" / "sorted_spikes.csv").write_text("time,unit_id\n")

    paths = write_deployment_selection_artifacts(
        experiment_dir=exp,
        comparison_dir=exp / "decoder_comparison",
        input_dir=tmp_path,
        metrics_df=metrics,
        best_df=best,
        seed=7,
    )
    assert paths["models_best_realtime_json"].exists()
    assert paths["scores_csv"].exists()

    payload = load_best_realtime_decoders(exp)
    assert payload["deployable"] is True
    assert payload["spike_source"] == "sorted"
    assert payload["update_interval_s"] == 0.05
    assert payload["update_rate_hz"] == 20.0
    assert "position" in payload["targets"]
    assert payload["targets"]["position"]["selected_causal_window_s"] == 0.5
    assert payload["targets"]["speed"]["selected_causal_window_s"] == 0.1
    # No ground-truth targets
    assert all(t["spike_source"] == "sorted" for t in payload["targets"].values())


def test_payload_rejects_empty_sorted():
    gt_only = pd.DataFrame([{
        "target_name": "position",
        "best_decoder_name": "ridge",
        "best_decode_window_s": 0.25,
        "recommended_realtime_window_s": 0.25,
        "recommended_realtime_decoder_name": "ridge",
        "best_feature_type": "counts",
        "best_metric_value": 1.0,
        "primary_metric": "mean_position_error_cm",
        "spike_source": "ground_truth",
        "decoder_config_json": "{}",
        "realtime_decoder_config_json": "{}",
    }])
    with pytest.raises(ValueError, match="sorted"):
        build_best_realtime_decoders_payload(
            gt_only,
            comparison_dir=Path("/tmp"),
            input_dir=Path("/tmp"),
        )


def test_write_artifacts_rejects_ground_truth_only(tmp_path: Path):
    exp = tmp_path / "run_gt"
    comp = exp / "decoder_comparison"
    comp.mkdir(parents=True)
    gt_only = pd.DataFrame([{
        "target_name": "position",
        "best_decoder_name": "ridge",
        "best_decode_window_s": 0.25,
        "recommended_realtime_window_s": 0.25,
        "recommended_realtime_decoder_name": "ridge",
        "best_feature_type": "counts",
        "best_metric_value": 1.0,
        "primary_metric": "mean_position_error_cm",
        "spike_source": "ground_truth",
        "decoder_config_json": "{}",
        "realtime_decoder_config_json": "{}",
    }])
    with pytest.raises(ValueError, match="sorted"):
        write_deployment_selection_artifacts(
            experiment_dir=exp,
            comparison_dir=comp,
            input_dir=tmp_path,
            metrics_df=pd.DataFrame(),
            best_df=gt_only,
        )


def test_isomap_best_accuracy_remaps_model_path():
    row = {
        "target_name": "position",
        "best_decoder_name": "ridge",
        "best_decode_window_s": 0.5,
        "recommended_realtime_window_s": 0.25,
        "recommended_realtime_decoder_name": "ridge",
        "best_feature_type": "global_isomap",
        "best_metric_value": 1.0,
        "primary_metric": "mean_position_error_cm",
        "spike_source": "sorted",
        "best_window_model_path": "models/by_window/global_isomap/k8/w0.500s/position.joblib",
        "realtime_model_path": "models/by_window/counts/kna/w0.250s/position.joblib",
        "manifold_transform_path": "models/manifold_transforms/global_isomap_k8_w0500ms",
        "decoder_config_json": json.dumps({
            "decoder_name": "ridge",
            "feature_type": "global_isomap",
            "manifold_type": "isomap",
            "manifold_transform_path": "models/manifold_transforms/global_isomap_k8_w0500ms",
        }),
        "realtime_decoder_config_json": json.dumps({
            "decoder_name": "ridge",
            "feature_type": "counts",
            "manifold_type": "none",
        }),
    }
    payload = build_best_realtime_decoders_payload(
        pd.DataFrame([row]),
        comparison_dir=Path("/tmp/comp"),
        input_dir=Path("/tmp"),
        selection_policy="best_accuracy",
    )
    tgt = payload["targets"]["position"]
    assert tgt["selected_feature_mode"] == "counts"
    assert tgt["model_artifact_path"] == "models/by_window/counts/kna/w0.250s/position.joblib"
    assert tgt["manifold_transform_path"] is None
    assert tgt["selected_causal_window_s"] == 0.25
    assert tgt.get("remapped_from_offline_isomap") is True
