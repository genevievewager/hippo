"""Tests for quadrant comparison figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualization.publication_quadrant_plots import (
    STABILITY_METRICS,
    make_quadrant_behavior_figure,
    make_quadrant_stability_figure,
    plot_quadrant_behavior,
    plot_quadrant_stability,
)


def _stability_df() -> pd.DataFrame:
    rows = []
    for emb in ("global_pca", "diffusion_nystrom", "global_lds"):
        rows.append({
            "embedding_type": emb,
            "decode_window_s": 0.25,
            "pairwise_distance_preservation": 0.4,
            "neighborhood_trustworthiness": 0.7,
            "latent_velocity_mean": 0.1,
            "latent_smoothness": 0.8,
            "latent_dimensionality_proxy": 2.5,
            "procrustes_alignment_error": np.nan,
        })
    return pd.DataFrame(rows)


def test_stability_figure_has_six_axes():
    fig = make_quadrant_stability_figure(_stability_df(), decode_window_s=0.25)
    assert len(STABILITY_METRICS) == 6
    assert len(fig.axes) == 6
    plt.close(fig)


def test_plot_quadrant_stability_writes(tmp_path: Path):
    out = plot_quadrant_stability(_stability_df(), tmp_path / "fig_quadrant_stability.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_behavior_figure_has_six_axes():
    t = np.linspace(0, 1, 20)
    decoded = pd.DataFrame({
        "true_x": np.cos(t),
        "true_y": np.sin(t),
        "decoded_x": np.cos(t) + 0.05,
        "decoded_y": np.sin(t) + 0.05,
    })
    by_q = {
        "static_linear": decoded,
        "static_nonlinear": decoded,
        "dynamic_linear": decoded,
        "dynamic_nonlinear": None,
    }
    metrics = {
        "static_linear": {"mean_predict_ms": 2.0, "mean_position_error_cm": 12.0},
        "static_nonlinear": {"mean_predict_ms": 3.0, "mean_position_error_cm": 10.0},
        "dynamic_linear": {"mean_predict_ms": 4.0, "mean_position_error_cm": 11.0},
    }
    fig = make_quadrant_behavior_figure(by_q, metrics, target="position")
    assert len(fig.axes) >= 6
    plt.close(fig)


def test_behavior_figure_time_colors_true_and_decoded():
    t = np.linspace(0, 10, 20)
    decoded = pd.DataFrame({
        "time": t,
        "true_x": np.cos(t),
        "true_y": np.sin(t),
        "decoded_x": np.cos(t) + 0.05,
        "decoded_y": np.sin(t) + 0.05,
    })
    fig = make_quadrant_behavior_figure(
        {"static_linear": decoded, "static_nonlinear": decoded, "dynamic_linear": decoded},
        {
            "static_linear": {"mean_predict_ms": 2.0, "mean_position_error_cm": 12.0},
            "static_nonlinear": {"mean_predict_ms": 3.0, "mean_position_error_cm": 10.0},
            "dynamic_linear": {"mean_predict_ms": 4.0, "mean_position_error_cm": 11.0},
        },
        target="position",
    )
    from matplotlib.collections import PathCollection
    n_scatter = sum(
        isinstance(artist, PathCollection)
        for ax in fig.axes
        for artist in ax.collections
    )
    assert n_scatter >= 6  # true + decoded on three occupied panels
    plt.close(fig)


def test_inspect_csv_time_colored_plotly_has_two_traces():
    from ui.views.realtime_replay import _true_vs_decoded_time_colored

    t = np.linspace(0, 5, 8)
    fig = _true_vs_decoded_time_colored(pd.DataFrame({
        "time": t,
        "true_x": np.cos(t),
        "true_y": np.sin(t),
        "decoded_x": np.cos(t) + 0.1,
        "decoded_y": np.sin(t) + 0.1,
    }))
    assert len(fig.data) == 2
    assert fig.data[0].name == "True"
    assert fig.data[1].name == "Decoded"


def test_plot_quadrant_behavior_writes(tmp_path: Path):
    decoded = pd.DataFrame({
        "time": [0.0, 1.0],
        "true_x": [0.0, 1.0],
        "true_y": [0.0, 1.0],
        "decoded_x": [0.1, 1.1],
        "decoded_y": [0.1, 0.9],
    })
    out = plot_quadrant_behavior(
        {"static_linear": decoded},
        {"static_linear": {"mean_predict_ms": 1.5, "mean_position_error_cm": 8.0}},
        tmp_path / "fig_quadrant_behavior.png",
        target="position",
    )
    assert out.exists()
    assert out.stat().st_size > 0
