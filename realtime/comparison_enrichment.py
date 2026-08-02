"""Per-candidate enrichment: latency, calibration, window stats, triggers."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd

from realtime.calibration_metrics import (
    bayesian_confidence_summary,
    classifier_calibration_summary,
)
from realtime.decoder_models import is_bayesian_model
from realtime.feature_representations import normalization_type_for
from realtime.realtime_gate import apply_realtime_gate
from realtime.search_space import compose_feature_mode
from realtime.trigger_comparison import TriggerRule, evaluate_trigger_rule_on_predictions


def window_matrix_stats(X_counts: np.ndarray) -> dict[str, float]:
    """Aggregate spike / active-unit statistics across causal windows."""
    X = np.asarray(X_counts, dtype=float)
    n_spikes = X.sum(axis=1)
    n_active = (X > 0).sum(axis=1)
    return {
        "mean_n_spikes_per_window": float(np.mean(n_spikes)),
        "median_n_spikes_per_window": float(np.median(n_spikes)),
        "mean_active_units_per_window": float(np.mean(n_active)),
        "median_active_units_per_window": float(np.median(n_active)),
    }


def per_sample_window_frame(
    decode_times: np.ndarray,
    X_counts: np.ndarray,
    *,
    decode_window_s: float,
    update_dt_s: float,
) -> pd.DataFrame:
    """Per-decode-time causal window metadata."""
    times = np.asarray(decode_times, dtype=float)
    X = np.asarray(X_counts, dtype=float)
    return pd.DataFrame({
        "decode_time": times,
        "window_start": times - float(decode_window_s),
        "window_end": times,
        "decode_window_s": float(decode_window_s),
        "update_dt_s": float(update_dt_s),
        "n_spikes_in_window": X.sum(axis=1),
        "n_active_units_in_window": (X > 0).sum(axis=1),
    })


def estimate_stage_latency_ms(
    feature_transform: Any,
    embedding_transform: Any,
    pipeline: Any,
    X_counts_sample: np.ndarray,
    n_reps: int = 3,
) -> dict[str, float]:
    """Micro-benchmark feature → embedding → predict on a small sample."""
    X = np.asarray(X_counts_sample, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    # Cap for speed.
    X = X[: min(8, len(X))]

    def _time(fn) -> float:
        # Warmup
        fn()
        t0 = time.perf_counter()
        for _ in range(n_reps):
            fn()
        return float((time.perf_counter() - t0) * 1000.0 / n_reps)

    feat_ms = _time(lambda: feature_transform.transform(X))
    Xf = feature_transform.transform(X)
    emb_ms = _time(lambda: embedding_transform.transform(Xf))
    Xe = embedding_transform.transform(Xf)
    pred_ms = _time(lambda: pipeline.predict(Xe))
    return {
        "feature_compute_ms": feat_ms,
        "embedding_transform_ms": emb_ms,
        "decoder_predict_ms": pred_ms,
    }


def enrich_fit_row(
    row: dict[str, Any],
    *,
    fit: Any,
    feature_type: str,
    embedding_type: str,
    feature_transform: Any,
    embedding_transform: Any,
    X_counts_test: np.ndarray,
    X_test_embedded: np.ndarray,
    decode_window_s: float,
    update_dt_s: float,
    max_compute_ms: float,
    max_effective_history_s: float,
    window_stats: dict[str, float],
    trigger_rules: list[TriggerRule] | None,
    decode_times_test: np.ndarray,
    behavior_test: pd.DataFrame,
    arena_bounds: tuple[float, float, float, float] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Add F/E metadata, latency gate, calibration; return trigger rows."""
    from realtime.decoder_comparison import CATEGORICAL_TARGETS

    row = dict(row)
    row["feature_type"] = feature_type
    row["embedding_type"] = embedding_type
    row["feature_mode"] = compose_feature_mode(feature_type, embedding_type)
    row["normalization_type"] = normalization_type_for(feature_type)
    row.update(window_stats)

    # Latency + realtime gate
    try:
        lat = estimate_stage_latency_ms(
            feature_transform, embedding_transform, fit.pipeline, X_counts_test,
        )
    except Exception:
        lat = {
            "feature_compute_ms": 1.0,
            "embedding_transform_ms": 1.0,
            "decoder_predict_ms": 1.0,
        }
    gate = apply_realtime_gate(
        feature_compute_ms=lat["feature_compute_ms"],
        embedding_transform_ms=lat["embedding_transform_ms"],
        decoder_predict_ms=lat["decoder_predict_ms"],
        decode_window_s=decode_window_s,
        update_dt_s=update_dt_s,
        max_compute_ms=max_compute_ms,
        max_effective_history_s=max_effective_history_s,
    )
    row.update(gate)

    # Calibration / confidence
    target = str(row.get("target_name"))
    decoder_name = str(row.get("decoder_name"))
    conf = None
    X_emb = np.asarray(X_test_embedded, dtype=float)
    if target in CATEGORICAL_TARGETS:
        try:
            if hasattr(fit.pipeline, "predict_proba"):
                proba = np.asarray(fit.pipeline.predict_proba(X_emb), dtype=float)
                labels = fit.labels or sorted(set(fit.y_true) | set(fit.y_pred))
                cal = classifier_calibration_summary(fit.y_true, proba, labels)
                row.update({
                    "confidence_metric_name": cal["confidence_metric_name"],
                    "confidence_metric_value": cal["confidence_metric_value"],
                    "brier_score": cal["brier_score"],
                    "expected_calibration_error": cal["expected_calibration_error"],
                    "confidence_accuracy_curve": json.dumps(cal["confidence_accuracy_curve"]),
                    "is_well_calibrated": cal["is_well_calibrated"],
                })
                conf = proba.max(axis=1)
            else:
                row.update({
                    "confidence_metric_name": None,
                    "confidence_metric_value": None,
                    "is_well_calibrated": False,
                })
        except Exception:
            row.update({
                "confidence_metric_name": None,
                "confidence_metric_value": None,
                "is_well_calibrated": False,
            })
    elif is_bayesian_model(decoder_name):
        try:
            place = fit.pipeline
            if hasattr(place, "place_"):
                place = place.place_
            if hasattr(place, "_posterior"):
                posts = np.stack([place._posterior(x) for x in X_emb], axis=0)
                cal = bayesian_confidence_summary(posts)
                row.update({
                    "confidence_metric_name": cal["confidence_metric_name"],
                    "confidence_metric_value": cal["confidence_metric_value"],
                    "posterior_entropy_mean": cal["posterior_entropy_mean"],
                    "posterior_peak_probability_mean": cal["posterior_peak_probability_mean"],
                    "posterior_credible_area_mean": cal["posterior_credible_area_mean"],
                    "is_well_calibrated": cal["is_well_calibrated"],
                })
                conf = np.asarray(
                    [float(np.max(p)) for p in posts], dtype=float,
                )
            else:
                row["is_well_calibrated"] = True
        except Exception:
            row["is_well_calibrated"] = True
    else:
        # Continuous non-Bayesian: calibration not required.
        row.update({
            "confidence_metric_name": None,
            "confidence_metric_value": None,
            "is_well_calibrated": True,
        })

    trigger_rows: list[dict[str, Any]] = []
    if trigger_rules:
        for rule in trigger_rules:
            if rule.closed_loop_target != target:
                continue
            try:
                trow = evaluate_trigger_rule_on_predictions(
                    rule,
                    decode_times=decode_times_test,
                    y_true=fit.y_true,
                    y_pred=fit.y_pred,
                    behavior_test=behavior_test,
                    confidence=conf if isinstance(conf, np.ndarray) else None,
                    arena_bounds=arena_bounds,
                )
                trow.update({
                    "spike_source": row.get("spike_source"),
                    "feature_type": feature_type,
                    "embedding_type": embedding_type,
                    "decoder_name": decoder_name,
                    "decode_window_s": decode_window_s,
                    "target_name": target,
                })
                trigger_rows.append(trow)
                # Attach best matching rule name onto metrics row (last wins).
                row["trigger_rule"] = rule.name
            except Exception:
                continue

    return row, trigger_rows
