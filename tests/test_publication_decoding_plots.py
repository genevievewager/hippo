"""Smoke and caption tests for publication decoder / Isomap figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from visualization.figure_captions import CAPTIONS, caption_for
from visualization.publication_decoding_plots import (
    plot_fig_closed_loop,
    plot_fig_decoding_performance,
    plot_fig_deployment,
    plot_fig_latency,
    plot_fig_manifold_decoding,
    plot_fig_temporal_wl,
)
from visualization.publication_isomap_plots import (
    plot_fig_isomap_diagnostics,
    plot_fig_isomap_story,
    plot_fig_latent_geometry,
)

PUB_STEMS = (
    "fig_decoding_performance",
    "fig_manifold_decoding",
    "fig_latent_geometry",
    "fig_isomap_diagnostics",
    "fig_isomap_story",
    "fig_closed_loop",
    "fig_deployment",
    "fig_latency",
    "fig_temporal_wl",
)


@pytest.fixture()
def mini_experiment(tmp_path: Path) -> Path:
    """Synthetic experiment tree with enough CSVs for all plot_fig_* writers."""
    exp = tmp_path / "exp"
    dec = exp / "decoder_comparison" / "sorted"
    dec.mkdir(parents=True)
    deploy = exp / "deployment_decoder_selection"
    deploy.mkdir(parents=True)
    lat = exp / "latency_profiling"
    lat.mkdir(parents=True)
    rt = exp / "realtime_decoding" / "sorted" / "spatial_context_shortest_near_optimal"
    rt.mkdir(parents=True)
    temporal = exp / "decoding" / "comparison" / "sorted"
    temporal.mkdir(parents=True)

    rows = []
    for w in (0.05, 0.10, 0.25):
        for feat in ("counts", "global_pca", "region_pca"):
            for target, family, metric, val_base in (
                ("position", "continuous", "mean_position_error_cm", 20.0),
                ("speed", "continuous", "r2", 0.5),
                ("spatial_context", "categorical", "balanced_accuracy", 0.7),
                ("movement_state", "categorical", "balanced_accuracy", 0.6),
                ("head_direction", "continuous", "mean_circular_error_deg", 40.0),
            ):
                for decoder in ("ridge", "random_forest_regressor"):
                    v = val_base + (0.05 if feat == "global_pca" else 0.0)
                    if "error" in metric:
                        v = val_base - (1.0 if feat == "global_pca" else 0.0)
                    row = {
                        "spike_source": "sorted",
                        "source": "sorted",
                        "feature_type": feat,
                        "decode_window_s": w,
                        "update_dt_s": 0.05,
                        "n_units": 50,
                        "manifold_type": "pca" if "pca" in feat else "none",
                        "manifold_grouping": "global" if feat == "global_pca" else (
                            "region" if feat == "region_pca" else ""
                        ),
                        "manifold_n_components": 3 if "pca" in feat else np.nan,
                        "n_neighbors": np.nan,
                        "target_name": target,
                        "target_family": family,
                        "decoder_name": decoder,
                        "primary_metric": metric,
                        "mean_position_error_cm": v if metric == "mean_position_error_cm" else np.nan,
                        "r2": v if metric == "r2" else np.nan,
                        "balanced_accuracy": v if metric == "balanced_accuracy" else np.nan,
                        "mean_circular_error_deg": v if metric == "mean_circular_error_deg" else np.nan,
                        "trustworthiness": np.nan,
                        "residual_variance": np.nan,
                        "geodesic_distance_correlation": np.nan,
                        "largest_component_fraction": np.nan,
                        "graph_connected": np.nan,
                    }
                    rows.append(row)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(dec / "decoder_comparison_metrics.csv", index=False)

    best = pd.DataFrame([
        {
            "target_name": "position",
            "target_family": "continuous",
            "primary_metric": "mean_position_error_cm",
            "best_decoder_name": "ridge",
            "best_decode_window_s": 0.10,
            "recommended_realtime_window_s": 0.10,
            "recommended_realtime_decoder_name": "ridge",
            "best_feature_type": "global_pca",
            "best_metric_value": 18.0,
        },
        {
            "target_name": "spatial_context",
            "target_family": "categorical",
            "primary_metric": "balanced_accuracy",
            "best_decoder_name": "random_forest_classifier",
            "best_decode_window_s": 0.25,
            "recommended_realtime_window_s": 0.25,
            "recommended_realtime_decoder_name": "random_forest_classifier",
            "best_feature_type": "counts",
            "best_metric_value": 0.75,
        },
    ])
    best.to_csv(dec / "best_decoder_by_target.csv", index=False)
    best.to_csv(deploy / "best_decoder_by_target_sorted.csv", index=False)

    mvc = pd.DataFrame([
        {
            "spike_source": "sorted",
            "target_name": "position",
            "primary_metric": "mean_position_error_cm",
            "best_counts_metric_value": 20.0,
            "best_manifold_metric_value": 18.0,
            "performance_difference": 2.0,
            "interpretation": "manifold improves decoding",
        },
        {
            "spike_source": "sorted",
            "target_name": "spatial_context",
            "primary_metric": "balanced_accuracy",
            "best_counts_metric_value": 0.70,
            "best_manifold_metric_value": 0.72,
            "performance_difference": 0.02,
            "interpretation": "manifold improves decoding",
        },
    ])
    mvc.to_csv(dec / "manifold_vs_counts_summary.csv", index=False)

    ev = pd.DataFrame([
        {
            "spike_source": "sorted",
            "decode_window_s": 0.1,
            "feature_type": "region_pca",
            "grouping_name": "region",
            "group_name": "CA1",
            "n_units": 20,
            "n_components": 3,
            "explained_variance_sum": 0.4,
        },
        {
            "spike_source": "sorted",
            "decode_window_s": 0.1,
            "feature_type": "region_pca",
            "grouping_name": "region",
            "group_name": "CA3",
            "n_units": 15,
            "n_components": 3,
            "explained_variance_sum": 0.35,
        },
    ])
    ev.to_csv(dec / "manifold_explained_variance.csv", index=False)

    scores = []
    for w in (0.05, 0.10, 0.25):
        for dec_name in ("ridge", "random_forest_regressor"):
            for target in ("position", "spatial_context"):
                scores.append({
                    "spike_source": "sorted",
                    "target": target,
                    "decoder": dec_name,
                    "feature_mode": "counts",
                    "causal_window_s": w,
                    "metric_name": "mean_position_error_cm" if target == "position" else "balanced_accuracy",
                    "metric_value": 20.0 - 10 * w if target == "position" else 0.6 + w,
                    "higher_is_better": target != "position",
                    "realtime_compatible": True,
                })
    pd.DataFrame(scores).to_csv(deploy / "all_sorted_window_scores.csv", index=False)

    n = 80
    rng = np.random.default_rng(0)
    decoded = pd.DataFrame({
        "time": np.linspace(0, 40, n),
        "true_x": rng.uniform(0, 100, n),
        "true_y": rng.uniform(0, 100, n),
        "decoded_x": rng.uniform(0, 100, n),
        "decoded_y": rng.uniform(0, 100, n),
        "position_error_cm": rng.uniform(5, 40, n),
        "true_spatial_context": rng.choice(["open", "near_wall"], n),
        "decoded_spatial_context": rng.choice(["open", "near_wall"], n),
        "true_movement_state": rng.choice(["moving", "still"], n),
        "decoded_movement_state": rng.choice(["moving", "still"], n),
    })
    decoded.to_csv(rt / "decoded_realtime.csv", index=False)
    events = pd.DataFrame({
        "time": rng.uniform(0, 40, 20),
        "event_type": ["trigger"] * 20,
        "closed_loop_target": ["spatial_context"] * 20,
        "correct_trigger": rng.choice([True, False], 20),
        "true_spatial_context": rng.choice(["open", "near_wall"], 20),
        "decoded_spatial_context": rng.choice(["open", "near_wall"], 20),
    })
    events.to_csv(rt / "closed_loop_events.csv", index=False)

    pd.DataFrame([
        {"category": "feature_transform", "name": "counts", "mean_ms": 0.01, "p95_ms": 0.02, "realtime_compatible": True},
        {"category": "feature_transform", "name": "global_pca", "mean_ms": 0.1, "p95_ms": 0.2, "realtime_compatible": True},
        {"category": "realtime_stage", "name": "spike_binning", "mean_ms": 1.0, "p95_ms": 1.5, "realtime_compatible": True},
        {"category": "isomap_compare", "name": "global_isomap_teacher", "mean_ms": 8.0, "p95_ms": 10.0, "realtime_compatible": False},
    ]).to_csv(lat / "latency_everything.csv", index=False)
    pd.DataFrame([
        {"name": "counts", "mean_ms": 0.01, "realtime_compatible": True},
        {"name": "global_pca", "mean_ms": 0.1, "realtime_compatible": True},
    ]).to_csv(lat / "feature_transform_latency.csv", index=False)
    pd.DataFrame([
        {"method": "global_isomap_teacher", "mean_ms": 8.0, "realtime_compatible": False},
        {"method": "global_isomap_distilled", "mean_ms": 0.1, "realtime_compatible": True},
    ]).to_csv(lat / "isomap_teacher_vs_distilled_latency.csv", index=False)
    pd.DataFrame([
        {"stage": "spike_binning", "mean_ms": 1.0, "source": "sorted/x"},
        {"stage": "feature_transform", "mean_ms": 0.2, "source": "sorted/x"},
        {"stage": "decode_position", "mean_ms": 2.0, "source": "sorted/x"},
    ]).to_csv(lat / "realtime_stage_latency_combined.csv", index=False)
    (lat / "latency_benchmark_summary.json").write_text('{"update_budget_ms": 50.0}\n')

    # Temporal W×L
    twl = []
    for w in (0.05, 0.1, 0.25):
        for L in (1, 2, 4):
            twl.append({
                "target": "position",
                "representation": "pca",
                "temporal_model": "flattened_history",
                "integration_window_s": w,
                "latent_history_frames": L,
                "validation_metric": "mean_position_error_cm",
                "validation_metric_value": 25.0 - 5 * w - L,
            })
            twl.append({
                "target": "speed",
                "representation": "raw",
                "temporal_model": "static_latent",
                "integration_window_s": w,
                "latent_history_frames": L,
                "validation_metric": "r2",
                "validation_metric_value": 0.4 + 0.1 * w + 0.01 * L,
            })
    pd.DataFrame(twl).to_csv(temporal / "all_configurations.csv", index=False)

    return exp


def test_publication_caption_stems_registered():
    for stem in PUB_STEMS:
        assert stem in CAPTIONS
        text = caption_for(Path(f"figures/decoder_comparison/{stem}.png"), figure_number=4)
        assert text.startswith("Figure 4. ")
        assert len(CAPTIONS[stem]) > 40



def test_plot_figs_write_pngs(mini_experiment: Path, tmp_path: Path):
    figures = tmp_path / "figures"
    figures.mkdir()

    assert plot_fig_decoding_performance(mini_experiment, figures) is not None
    assert plot_fig_manifold_decoding(mini_experiment, figures) is not None
    assert plot_fig_latent_geometry(mini_experiment, figures) is not None
    assert plot_fig_isomap_diagnostics(mini_experiment, figures) is not None
    assert plot_fig_isomap_story(mini_experiment, figures) is not None
    assert plot_fig_closed_loop(mini_experiment, figures) is not None
    assert plot_fig_deployment(mini_experiment, figures) is not None
    assert plot_fig_latency(mini_experiment, figures) is not None
    assert plot_fig_temporal_wl(mini_experiment, figures) is not None

    expected = {
        "decoder_comparison/fig_decoding_performance.png",
        "decoder_comparison/fig_manifold_decoding.png",
        "decoder_comparison/fig_latent_geometry.png",
        "decoder_comparison/fig_isomap_diagnostics.png",
        "decoder_comparison/fig_isomap_story.png",
        "realtime_decoding/fig_closed_loop.png",
        "deployment_decoder_selection/fig_deployment.png",
        "latency/fig_latency.png",
        "temporal_decoding/fig_temporal_wl.png",
    }
    for rel in expected:
        path = figures / rel
        assert path.exists(), f"missing {rel}"
        assert path.stat().st_size > 1000


def test_isomap_empty_state_does_not_crash(mini_experiment: Path, tmp_path: Path):
    """PCA-only metrics still produce Isomap figs with empty-state annotations."""
    figures = tmp_path / "figures"
    path = plot_fig_isomap_diagnostics(mini_experiment, figures)
    assert path is not None and path.exists()
    path2 = plot_fig_latent_geometry(mini_experiment, figures)
    assert path2 is not None and path2.exists()


@pytest.mark.skipif(
    not Path("outputs/ratinabox_005/decoder_comparison/sorted/decoder_comparison_metrics.csv").exists(),
    reason="ratinabox_005 comparison outputs not present",
)
def test_optional_ratinabox_005_smoke():
    from visualization.publication_decoding_plots import generate_publication_decoding_figures

    exp = Path("outputs/ratinabox_005")
    figures = exp / "figures"
    written = generate_publication_decoding_figures(exp, figures, cleanup_legacy=True)
    assert written
    assert (figures / "decoder_comparison" / "fig_decoding_performance.png").exists()
