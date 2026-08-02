"""Baseline and negative-control comparisons for decoder information tests."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score


CONTROL_NAMES = (
    "shuffle_spike_times",
    "shuffle_unit_ids",
    "circular_shift_behavior",
    "occupancy_prior_only",
    "previous_state_baseline",
    "random_classifier",
)


def shuffle_spike_times(spikes_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute spike times across events (destroys temporal structure, keeps rates)."""
    out = spikes_df.copy()
    time_col = "time" if "time" in out.columns else "spike_time"
    times = out[time_col].to_numpy().copy()
    rng.shuffle(times)
    out[time_col] = times
    return out.sort_values(time_col).reset_index(drop=True)


def shuffle_unit_ids(spikes_df: pd.DataFrame, unit_ids: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    out = spikes_df.copy()
    unit_col = "unit_id" if "unit_id" in out.columns else "unit"
    mapping = {int(u): int(v) for u, v in zip(unit_ids, rng.permutation(unit_ids))}
    out[unit_col] = out[unit_col].map(lambda u: mapping.get(int(u), int(u)))
    return out


def circular_shift_behavior(behavior: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = behavior.copy()
    n = len(out)
    if n < 2:
        return out
    shift = int(rng.integers(1, n))
    for col in out.columns:
        if col in {"time", "timestamp"}:
            continue
        out[col] = np.roll(out[col].to_numpy(), shift)
    return out


class OccupancyPriorClassifier(BaseEstimator, ClassifierMixin):
    """Predict the most frequent training class (occupancy / label prior)."""

    def fit(self, X, y):
        y = np.asarray(y)
        vals, counts = np.unique(y, return_counts=True)
        self.classes_ = vals
        self.prior_ = counts / counts.sum()
        self.majority_ = vals[int(np.argmax(counts))]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_, dtype=object)

    def predict_proba(self, X):
        return np.tile(self.prior_, (len(X), 1))


class PreviousStateBaseline(BaseEstimator, ClassifierMixin):
    """Predict previous label (temporal autocorrelation baseline)."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._y_train_ = y
        return self

    def predict(self, X):
        # At test time without streaming state, use shifted train mode as proxy:
        # predict the training majority (conservative). Callers should prefer
        # ``predict_from_labels`` for true previous-state evaluation.
        vals, counts = np.unique(self._y_train_, return_counts=True)
        return np.full(len(X), vals[int(np.argmax(counts))], dtype=object)


def previous_state_predictions(y_true: np.ndarray) -> np.ndarray:
    y = np.asarray(y_true)
    out = np.empty_like(y)
    out[0] = y[0]
    out[1:] = y[:-1]
    return out


class RandomClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.rng_ = np.random.default_rng(self.random_state)
        return self

    def predict(self, X):
        return self.rng_.choice(self.classes_, size=len(X))


def evaluate_categorical_controls(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate label-only negative controls on a categorical target."""
    rows: list[dict[str, Any]] = []
    controls: dict[str, Any] = {
        "occupancy_prior_only": OccupancyPriorClassifier(),
        "random_classifier": RandomClassifier(random_state=seed),
    }
    for name, clf in controls.items():
        model = clone(clf)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "control_name": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        })

    prev = previous_state_predictions(y_test)
    rows.append({
        "control_name": "previous_state_baseline",
        "accuracy": float(accuracy_score(y_test, prev)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prev)),
    })
    return rows
