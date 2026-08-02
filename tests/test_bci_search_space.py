"""Tests for F×E×D×W×C helpers and realtime gating."""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime.calibration_metrics import (
    brier_score,
    expected_calibration_error,
    is_well_calibrated,
)
from realtime.realtime_gate import apply_realtime_gate
from realtime.search_space import compose_feature_mode, expand_fe_jobs, legacy_mode_to_fe
from realtime.sorting_robustness import sorting_robustness_label
from realtime.trigger_comparison import default_trigger_rules


def test_compose_feature_mode_legacy_compatible():
    assert compose_feature_mode("counts", "identity") == "counts"
    assert compose_feature_mode("rates", "identity") == "rates"
    assert compose_feature_mode("counts", "global_pca") == "global_pca"
    assert compose_feature_mode("sqrt_counts", "global_pca") == "sqrt_counts__global_pca"


def test_legacy_mode_to_fe():
    assert legacy_mode_to_fe("counts") == ("counts", "identity")
    assert legacy_mode_to_fe("global_pca") == ("counts", "global_pca")


def test_expand_legacy_feature_modes():
    jobs = expand_fe_jobs(
        feature_modes=("counts", "global_pca"),
        manifold_n_components=(3,),
        use_fe_grid=False,
    )
    pairs = {(f, e) for f, e, _, _ in jobs}
    assert ("counts", "identity") in pairs
    assert ("counts", "global_pca") in pairs


def test_expand_fe_grid():
    jobs = expand_fe_jobs(
        feature_types=("counts", "rates"),
        embedding_types=("identity", "global_pca"),
        manifold_n_components=(2,),
        use_fe_grid=True,
    )
    pairs = {(f, e) for f, e, _, _ in jobs}
    assert ("counts", "identity") in pairs
    assert ("rates", "global_pca") in pairs


def test_realtime_gate_reasons():
    ok = apply_realtime_gate(
        feature_compute_ms=1, embedding_transform_ms=1, decoder_predict_ms=1,
        decode_window_s=0.25, update_dt_s=0.05,
        max_compute_ms=25, max_effective_history_s=0.5,
    )
    assert ok["passes_realtime_gate"]
    assert ok["realtime_gate_reason"] == "passes"

    slow = apply_realtime_gate(
        feature_compute_ms=20, embedding_transform_ms=20, decoder_predict_ms=20,
        decode_window_s=0.25, update_dt_s=0.05,
        max_compute_ms=25, max_effective_history_s=0.5,
    )
    assert slow["realtime_gate_reason"] == "compute_too_slow"

    long = apply_realtime_gate(
        feature_compute_ms=1, embedding_transform_ms=1, decoder_predict_ms=1,
        decode_window_s=1.0, update_dt_s=0.05,
        max_compute_ms=25, max_effective_history_s=0.5,
    )
    assert long["realtime_gate_reason"] == "history_window_too_long"


def test_calibration_helpers():
    y = np.array(["a", "b", "a", "b"])
    proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.8, 0.2],
        [0.3, 0.7],
    ])
    brier = brier_score(y, proba, ["a", "b"])
    ece = expected_calibration_error(y, proba, ["a", "b"])
    assert 0 <= brier <= 1
    assert 0 <= ece <= 1
    assert is_well_calibrated(brier=0.1, ece=0.05)


def test_sorting_robustness_labels():
    assert sorting_robustness_label(0.05) == "minimal_loss"
    assert sorting_robustness_label(0.2) == "moderate_loss"
    assert sorting_robustness_label(0.5) == "large_loss"


def test_default_trigger_rules_nonempty():
    rules = default_trigger_rules()
    assert any(r.closed_loop_target == "spatial_context" for r in rules)
    assert any(r.closed_loop_target == "speed" for r in rules)
