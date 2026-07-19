"""Tests for closed-loop decoder / window selection from comparison tables."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from realtime.best_decoder_selection import select_best_decoder_row


def _write_comparison_table(comparison_dir: Path, spike_source: str = "sorted") -> Path:
    src_dir = comparison_dir / spike_source
    models_dir = src_dir / "models"
    models_dir.mkdir(parents=True)

    best_transform = models_dir / "manifold_transforms" / "region_pca_k3_w1000ms"
    rt_transform = models_dir / "manifold_transforms" / "region_pca_k3_w0500ms"
    best_transform.mkdir(parents=True)
    rt_transform.mkdir(parents=True)

    best_model = models_dir / "by_window" / "region_pca" / "k3" / "w1.000s" / "spatial_context.joblib"
    rt_model = models_dir / "by_window" / "region_pca" / "k3" / "w0.500s" / "spatial_context.joblib"
    best_model.parent.mkdir(parents=True)
    rt_model.parent.mkdir(parents=True)
    best_model.write_bytes(b"")
    rt_model.write_bytes(b"")

    best_cfg = {
        "decoder_name": "random_forest_classifier",
        "decode_window_s": 1.0,
        "feature_type": "region_pca",
        "manifold_n_components": 3,
        "manifold_transform_path": str(best_transform),
    }
    rt_cfg = {
        "decoder_name": "random_forest_classifier",
        "decode_window_s": 0.5,
        "feature_type": "region_pca",
        "manifold_n_components": 3,
        "manifold_transform_path": str(rt_transform),
    }
    df = pd.DataFrame(
        [
            {
                "target_name": "spatial_context",
                "primary_metric": "balanced_accuracy",
                "best_decoder_name": "random_forest_classifier",
                "best_decode_window_s": 1.0,
                "recommended_realtime_window_s": 0.5,
                "recommended_realtime_decoder_name": "random_forest_classifier",
                "best_feature_type": "region_pca",
                "best_manifold_n_components": 3.0,
                "best_metric_value": 0.87,
                "spike_source": spike_source,
                "best_window_model_path": str(best_model),
                "realtime_model_path": str(rt_model),
                # Top-level column intentionally stores the *best-W* transform.
                "manifold_transform_path": str(best_transform),
                "decoder_config_json": json.dumps(best_cfg),
                "realtime_decoder_config_json": json.dumps(rt_cfg),
            }
        ]
    )
    table_path = src_dir / "best_decoder_by_target.csv"
    df.to_csv(table_path, index=False)
    return table_path


def test_shortest_near_optimal_uses_window_matched_transform(tmp_path: Path):
    comparison_dir = tmp_path / "decoder_comparison"
    _write_comparison_table(comparison_dir)

    selected, _ = select_best_decoder_row(
        comparison_dir=comparison_dir,
        spike_source="sorted",
        closed_loop_target="spatial_context",
        selection_policy="shortest_near_optimal",
    )

    assert selected["selected_decode_window_s"] == 0.5
    assert "w0.500s" in selected["selected_model_path"]
    assert selected["selected_manifold_transform_path"].endswith("region_pca_k3_w0500ms")
    assert "w1000ms" not in selected["selected_manifold_transform_path"]


def test_best_accuracy_keeps_best_window_transform(tmp_path: Path):
    comparison_dir = tmp_path / "decoder_comparison"
    _write_comparison_table(comparison_dir)

    selected, _ = select_best_decoder_row(
        comparison_dir=comparison_dir,
        spike_source="sorted",
        closed_loop_target="spatial_context",
        selection_policy="best_accuracy",
    )

    assert selected["selected_decode_window_s"] == 1.0
    assert selected["selected_manifold_transform_path"].endswith("region_pca_k3_w1000ms")


def test_best_row_at_window_ignores_nan_n_components():
    """counts features store NaN n_components; lookup must not filter them out."""
    import numpy as np
    from realtime.decoder_comparison import _best_row_at_window

    df = pd.DataFrame(
        [
            {
                "decode_window_s": 0.25,
                "feature_type": "counts",
                "manifold_n_components": np.nan,
                "manifold_transform_path": "counts_w0250ms",
                "decoder_name": "random_forest_classifier",
                "balanced_accuracy": 0.91,
            },
            {
                "decode_window_s": 0.5,
                "feature_type": "counts",
                "manifold_n_components": np.nan,
                "manifold_transform_path": "counts_w0500ms",
                "decoder_name": "random_forest_classifier",
                "balanced_accuracy": 0.93,
            },
        ]
    )
    row = _best_row_at_window(
        df, 0.25, "counts", "balanced_accuracy", "higher", n_components=np.nan,
    )
    assert row is not None
    assert row["manifold_transform_path"] == "counts_w0250ms"
