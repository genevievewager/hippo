"""Tests for ridge quadrant center-bias diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from realtime.quadrant_ridge_diagnostics import (
    CalibrationMetrics,
    MethodDiagnostics,
    compute_calibration_metrics,
    compute_occupancy_weights,
    generate_interpretation_summary,
)


def test_calibration_metrics_shrinkage_toward_center():
    center = (50.0, 50.0)
    true = np.array([[20.0, 50.0], [80.0, 50.0], [50.0, 20.0], [50.0, 80.0]])
    pred = 0.5 * true + np.array([25.0, 25.0])  # shrink toward center
    m = compute_calibration_metrics(true, pred, center)
    assert m.slope_x < 1.0
    assert m.slope_y < 1.0
    assert m.slope_r < 1.0
    assert m.mean_radial_bias < 0.0
    assert m.contraction_ratio < 1.0


def test_occupancy_weights_inverse_frequency():
    xy = np.vstack([
        np.tile([0.0, 0.0], (10, 1)),
        np.tile([10.0, 10.0], (2, 1)),
    ])
    w = compute_occupancy_weights(xy, (0.0, 20.0, 0.0, 20.0), n_bins=5)
    assert len(w) == 12
    assert w.sum() == pytest.approx(12.0, rel=1e-6)
    assert w[10] > w[0]


def test_interpretation_summary_mentions_best_method():
    def mk(err, bias, sr):
        return CalibrationMetrics(
            slope_x=0.8, intercept_x=0.0, r2_x=0.5,
            slope_y=0.8, intercept_y=0.0, r2_y=0.5,
            slope_r=sr, intercept_r=0.0, r2_r=0.5,
            mean_radial_bias=bias,
            contraction_ratio=0.7,
            mean_position_error_cm=err,
        )

    results = {
        "counts": MethodDiagnostics(
            "counts", "Counts", "baseline", "counts", True,
            metrics=mk(10.0, -1.0, 0.9),
        ),
        "global_pca": MethodDiagnostics(
            "global_pca", "Global PCA", "static_linear", "global_pca", True,
            metrics=mk(15.0, -5.0, 0.4),
        ),
    }
    text = generate_interpretation_summary(results)
    assert "Counts" in text
    assert "Global PCA" in text
