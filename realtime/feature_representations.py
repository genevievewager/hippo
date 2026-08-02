"""Causal spike feature representations (search dimension F).

All normalizations are fit on the training portion only and then frozen for
held-out / realtime application. Never fit on the full session before splitting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ALL_FEATURE_TYPES = (
    "counts",
    "rates",
    "zscore_counts",
    "sqrt_counts",
    "log1p_counts",
    "region_normalized_counts",
    "cell_type_normalized_counts",
)

QUICK_FEATURE_TYPES = ("counts", "rates", "sqrt_counts")
FULL_FEATURE_TYPES = ALL_FEATURE_TYPES

# Feature types that need unit metadata grouping.
GROUPED_FEATURE_TYPES = {
    "region_normalized_counts": "region",
    "cell_type_normalized_counts": "cell_type",
}

# Feature types that store training statistics for z-score / group norm.
NORMALIZATION_TYPES = {
    "counts": "none",
    "rates": "rate",
    "zscore_counts": "zscore",
    "sqrt_counts": "sqrt",
    "log1p_counts": "log1p",
    "region_normalized_counts": "region_zscore",
    "cell_type_normalized_counts": "cell_type_zscore",
}


def resolve_feature_types(
    feature_types: tuple[str, ...] | list[str] | None,
    *,
    max_models: str = "quick",
) -> tuple[str, ...]:
    if feature_types:
        unknown = [f for f in feature_types if f not in ALL_FEATURE_TYPES]
        if unknown:
            raise ValueError(f"Unknown feature type(s): {unknown}")
        return tuple(dict.fromkeys(feature_types))
    if max_models == "full":
        return FULL_FEATURE_TYPES
    return QUICK_FEATURE_TYPES


def normalization_type_for(feature_type: str) -> str:
    if feature_type not in NORMALIZATION_TYPES:
        raise ValueError(f"Unknown feature type: {feature_type}")
    return NORMALIZATION_TYPES[feature_type]


class SpikeFeatureTransformer(BaseEstimator, TransformerMixin):
    """Train-only spike feature representation F."""

    def __init__(
        self,
        feature_type: str = "counts",
        decode_window: float = 0.250,
        group_labels: list[str] | None = None,
        eps: float = 1e-6,
    ):
        if feature_type not in ALL_FEATURE_TYPES:
            raise ValueError(
                f"Unknown feature_type {feature_type!r}; expected one of {ALL_FEATURE_TYPES}"
            )
        self.feature_type = feature_type
        self.decode_window = float(decode_window)
        self.group_labels = group_labels
        self.eps = float(eps)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.group_mean_: dict[str, np.ndarray] | None = None
        self.group_std_: dict[str, np.ndarray] | None = None
        self.n_features_in_: int | None = None

    @property
    def normalization_type(self) -> str:
        return normalization_type_for(self.feature_type)

    def fit(self, X: np.ndarray, y: Any = None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = int(X.shape[1])
        ft = self.feature_type

        if ft == "zscore_counts":
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0)
            self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)
        elif ft in GROUPED_FEATURE_TYPES:
            if self.group_labels is None or len(self.group_labels) != X.shape[1]:
                raise ValueError(
                    f"{ft} requires group_labels with length n_units={X.shape[1]}"
                )
            labels = np.asarray(self.group_labels, dtype=object)
            self.group_mean_ = {}
            self.group_std_ = {}
            for g in sorted(set(labels.tolist())):
                idx = np.where(labels == g)[0]
                if idx.size == 0:
                    continue
                mu = np.mean(X[:, idx], axis=0)
                sd = np.std(X[:, idx], axis=0)
                sd = np.where(sd < self.eps, 1.0, sd)
                self.group_mean_[str(g)] = mu
                self.group_std_[str(g)] = sd
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        ft = self.feature_type
        if ft == "counts":
            return X
        if ft == "rates":
            return X / self.decode_window
        if ft == "sqrt_counts":
            return np.sqrt(np.maximum(X, 0.0))
        if ft == "log1p_counts":
            return np.log1p(np.maximum(X, 0.0))
        if ft == "zscore_counts":
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("SpikeFeatureTransformer must be fit before transform")
            return (X - self.mean_) / self.std_
        if ft in GROUPED_FEATURE_TYPES:
            if self.group_mean_ is None or self.group_std_ is None or self.group_labels is None:
                raise RuntimeError("SpikeFeatureTransformer must be fit before transform")
            out = np.zeros_like(X, dtype=float)
            labels = np.asarray(self.group_labels, dtype=object)
            for g, mu in self.group_mean_.items():
                idx = np.where(labels == g)[0]
                if idx.size == 0:
                    continue
                out[:, idx] = (X[:, idx] - mu) / self.group_std_[g]
            return out
        raise ValueError(f"Unknown feature_type: {ft}")

    def get_metadata(self) -> dict[str, Any]:
        return {
            "feature_type": self.feature_type,
            "normalization_type": self.normalization_type,
            "decode_window_s": self.decode_window,
            "n_features_in": self.n_features_in_,
        }

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_dir / "feature_transform.joblib")
        with open(output_dir / "meta.json", "w") as f:
            json.dump(self.get_metadata(), f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> "SpikeFeatureTransformer":
        return joblib.load(Path(input_dir) / "feature_transform.joblib")


def group_labels_for_feature_type(
    feature_type: str,
    units_df: pd.DataFrame | None,
    unit_ids: list[int] | np.ndarray | None,
) -> list[str] | None:
    """Return per-unit group labels for grouped feature types, else None."""
    col = GROUPED_FEATURE_TYPES.get(feature_type)
    if col is None:
        return None
    if units_df is None or unit_ids is None:
        raise ValueError(f"{feature_type} requires units_df and unit_ids")
    if col not in units_df.columns:
        raise ValueError(f"{feature_type} requires units.csv column {col!r}")
    indexed = units_df.set_index("unit_id")
    labels: list[str] = []
    for uid in np.asarray(unit_ids, dtype=int):
        if uid not in indexed.index:
            labels.append("unknown")
        else:
            labels.append(str(indexed.loc[uid, col]))
    return labels


def make_spike_feature_transformer(
    feature_type: str,
    *,
    decode_window: float,
    units_df: pd.DataFrame | None = None,
    unit_ids: list[int] | np.ndarray | None = None,
) -> SpikeFeatureTransformer:
    labels = None
    if feature_type in GROUPED_FEATURE_TYPES:
        labels = group_labels_for_feature_type(feature_type, units_df, unit_ids)
    return SpikeFeatureTransformer(
        feature_type=feature_type,
        decode_window=decode_window,
        group_labels=labels,
    )
