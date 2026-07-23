"""Tests for manifold_vs_spikes_onepager figure generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from visualization.manifold_plots import (
    _metrics_to_score_frame,
    _plot_manifold_vs_spikes_onepager,
)


def _tiny_metrics() -> pd.DataFrame:
    rows = []
    for feat, dec, w, err in [
        ("counts", "ridge", 0.25, 12.0),
        ("counts", "random_forest_regressor", 0.5, 10.0),
        ("region_pca", "random_forest_regressor", 0.5, 8.0),
        ("global_pca", "ridge", 0.25, 15.0),
    ]:
        rows.append({
            "spike_source": "sorted",
            "target_name": "position",
            "target_family": "continuous",
            "decoder_name": dec,
            "feature_type": feat,
            "decode_window_s": w,
            "primary_metric": "mean_position_error_cm",
            "mean_position_error_cm": err,
            "realtime_compatible": True,
            "manifold_n_components": 3.0 if "pca" in feat else None,
        })
    rows.append({
        "spike_source": "sorted",
        "target_name": "speed",
        "target_family": "continuous",
        "decoder_name": "pca_ridge",
        "feature_type": "counts",
        "decode_window_s": 0.25,
        "primary_metric": "r2",
        "r2": 0.4,
        "realtime_compatible": True,
        "manifold_n_components": None,
    })
    rows.append({
        "spike_source": "sorted",
        "target_name": "speed",
        "target_family": "continuous",
        "decoder_name": "ridge",
        "feature_type": "global_pca",
        "decode_window_s": 0.05,
        "primary_metric": "r2",
        "r2": 0.0,
        "realtime_compatible": True,
        "manifold_n_components": 3.0,
    })
    return pd.DataFrame(rows)


def test_metrics_to_score_frame():
    scores = _metrics_to_score_frame(_tiny_metrics())
    assert not scores.empty
    assert set(scores["target"]) >= {"position", "speed"}
    assert "feature_mode" in scores.columns


def test_manifold_vs_spikes_onepager_writes_png(tmp_path: Path):
    metrics = _tiny_metrics()
    comparison_dir = tmp_path / "decoder_comparison" / "sorted"
    comparison_dir.mkdir(parents=True)
    vs = pd.DataFrame(
        [
            {
                "spike_source": "sorted",
                "target_name": "position",
                "primary_metric": "mean_position_error_cm",
                "best_counts_feature_type": "counts",
                "best_counts_decoder": "random_forest_regressor",
                "best_counts_window_s": 0.5,
                "best_counts_metric_value": 10.0,
                "best_manifold_feature_type": "region_pca",
                "best_manifold_decoder": "random_forest_regressor",
                "best_manifold_window_s": 0.5,
                "best_manifold_metric_value": 8.0,
                "performance_difference": 2.0,
                "interpretation": "manifold improves decoding",
            },
            {
                "spike_source": "sorted",
                "target_name": "speed",
                "primary_metric": "r2",
                "best_counts_feature_type": "counts",
                "best_counts_decoder": "pca_ridge",
                "best_counts_window_s": 0.25,
                "best_counts_metric_value": 0.4,
                "best_manifold_feature_type": "global_pca",
                "best_manifold_decoder": "ridge",
                "best_manifold_window_s": 0.05,
                "best_manifold_metric_value": 0.0,
                "performance_difference": -0.4,
                "interpretation": "manifold reduces decoding",
            },
        ]
    )
    vs.to_csv(comparison_dir / "manifold_vs_counts_summary.csv", index=False)
    metrics.to_csv(comparison_dir / "decoder_comparison_metrics.csv", index=False)

    out = tmp_path / "figures" / "decoder_comparison"
    out.mkdir(parents=True)
    path = _plot_manifold_vs_spikes_onepager(
        metrics=metrics,
        comparison_dir=tmp_path / "decoder_comparison",
        out_dir=out,
        experiment_dir=tmp_path,
    )
    assert path is not None
    assert path.exists()
    assert path.name == "fig_manifold_vs_spikes_onepager.png"
