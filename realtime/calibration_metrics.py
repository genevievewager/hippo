"""Confidence calibration metrics for closed-loop decoder gating."""

from __future__ import annotations

from typing import Any

import numpy as np


def brier_score(y_true: np.ndarray, proba: np.ndarray, classes: list[Any] | np.ndarray) -> float:
    """Multiclass Brier score."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    one_hot = np.zeros_like(proba)
    for i, lab in enumerate(y_true):
        if lab in class_to_idx:
            one_hot[i, class_to_idx[lab]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[Any] | np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE using max-class confidence bins."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    correct = np.array(
        [class_to_idx.get(lab, -1) == pred_idx[i] for i, lab in enumerate(y_true)],
        dtype=float,
    )
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(conf)
    if n == 0:
        return float("nan")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if not np.any(mask):
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def confidence_accuracy_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[Any] | np.ndarray,
    thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
) -> list[dict[str, float]]:
    """Accuracy among predictions with confidence >= threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    classes = list(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    conf = proba.max(axis=1)
    pred_idx = proba.argmax(axis=1)
    correct = np.array(
        [class_to_idx.get(lab, -1) == pred_idx[i] for i, lab in enumerate(y_true)],
        dtype=float,
    )
    rows: list[dict[str, float]] = []
    for thr in thresholds:
        mask = conf >= thr
        if not np.any(mask):
            rows.append({
                "confidence_threshold": float(thr),
                "coverage": 0.0,
                "accuracy": float("nan"),
            })
            continue
        rows.append({
            "confidence_threshold": float(thr),
            "coverage": float(mask.mean()),
            "accuracy": float(correct[mask].mean()),
        })
    return rows


def posterior_entropy(posteriors: np.ndarray) -> np.ndarray:
    """Shannon entropy (nats) for each flattened posterior."""
    P = np.asarray(posteriors, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    return -np.sum(P * np.log(P), axis=1)


def posterior_peak_probability(posteriors: np.ndarray) -> np.ndarray:
    P = np.asarray(posteriors, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    return P.reshape(P.shape[0], -1).max(axis=1)


def posterior_credible_area(
    posteriors: np.ndarray,
    mass: float = 0.95,
) -> np.ndarray:
    """Fraction of bins needed to accumulate ``mass`` posterior probability."""
    P = np.asarray(posteriors, dtype=float)
    if P.ndim == 1:
        P = P.reshape(1, -1)
    flat = P.reshape(P.shape[0], -1)
    flat = flat / np.maximum(flat.sum(axis=1, keepdims=True), 1e-12)
    areas = np.zeros(flat.shape[0], dtype=float)
    n = flat.shape[1]
    for i in range(flat.shape[0]):
        order = np.sort(flat[i])[::-1]
        csum = np.cumsum(order)
        k = int(np.searchsorted(csum, mass) + 1)
        areas[i] = k / max(n, 1)
    return areas


def is_well_calibrated(
    *,
    brier: float | None = None,
    ece: float | None = None,
    ece_threshold: float = 0.15,
    brier_threshold: float = 0.35,
) -> bool:
    """Heuristic gate for whether confidence thresholds are usable."""
    ok = True
    if ece is not None and not np.isnan(ece):
        ok = ok and ece <= ece_threshold
    if brier is not None and not np.isnan(brier):
        ok = ok and brier <= brier_threshold
    return bool(ok)


def classifier_calibration_summary(
    y_true: np.ndarray,
    proba: np.ndarray,
    classes: list[Any] | np.ndarray,
) -> dict[str, Any]:
    brier = brier_score(y_true, proba, classes)
    ece = expected_calibration_error(y_true, proba, classes)
    curve = confidence_accuracy_curve(y_true, proba, classes)
    return {
        "confidence_metric_name": "expected_calibration_error",
        "confidence_metric_value": ece,
        "brier_score": brier,
        "expected_calibration_error": ece,
        "confidence_accuracy_curve": curve,
        "is_well_calibrated": is_well_calibrated(brier=brier, ece=ece),
    }


def bayesian_confidence_summary(posteriors: np.ndarray) -> dict[str, Any]:
    ent = posterior_entropy(posteriors)
    peak = posterior_peak_probability(posteriors)
    area = posterior_credible_area(posteriors)
    # Use mean peak probability as primary confidence metric.
    mean_peak = float(np.mean(peak))
    return {
        "confidence_metric_name": "posterior_peak_probability",
        "confidence_metric_value": mean_peak,
        "posterior_entropy_mean": float(np.mean(ent)),
        "posterior_peak_probability_mean": mean_peak,
        "posterior_credible_area_mean": float(np.mean(area)),
        # Bayesian peak confidence is "usable" if typically decisive.
        "is_well_calibrated": bool(mean_peak >= 0.2),
    }
