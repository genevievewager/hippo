"""Behavioral association of latent dimensions (not latent biological labels)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


CONTINUOUS_BEHAVIOR_VARS = (
    "x",
    "y",
    "speed",
    "acceleration",
    "distance_to_wall",
)
CIRCULAR_BEHAVIOR_VARS = ("head_direction",)
CATEGORICAL_BEHAVIOR_VARS = (
    "spatial_context",
    "movement_state",
    "reward_state",
    "in_reward_zone",
)


def _safe_series(behavior: pd.DataFrame, name: str) -> np.ndarray | None:
    if name not in behavior.columns:
        # Common aliases
        aliases = {
            "reward_state": ["reward", "reward_zone", "in_reward_zone"],
            "in_reward_zone": ["reward_zone", "reward_state"],
        }
        for alt in aliases.get(name, []):
            if alt in behavior.columns:
                return behavior[alt].to_numpy()
        return None
    return behavior[name].to_numpy()


def _circular_association(z: np.ndarray, angle: np.ndarray) -> float:
    """Association via decoding sin/cos of angle from a single latent dim."""
    y = np.column_stack([np.sin(angle), np.cos(angle)])
    model = Ridge(alpha=1.0)
    try:
        model.fit(z.reshape(-1, 1), y)
        pred = model.predict(z.reshape(-1, 1))
        return float(r2_score(y, pred))
    except Exception:
        return float("nan")


def associate_latent_with_behavior(
    Z: np.ndarray,
    behavior: pd.DataFrame,
    *,
    representation: str,
    causal_status: str = "causal_filtered",
    max_cv_splits: int = 3,
) -> pd.DataFrame:
    """Build a table of latent-dimension × behavior association scores.

    Scores are labeled as **behavioral association**, not latent meaning.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 1:
        assert Z.ndim == 2
    else:
        Z = Z.reshape(-1, 1)
    T, k = Z.shape
    if len(behavior) != T:
        n = min(T, len(behavior))
        Z = Z[:n]
        behavior = behavior.iloc[:n].reset_index(drop=True)
        T, k = Z.shape

    rows: list[dict[str, Any]] = []

    for dim in range(k):
        z = Z[:, dim]
        z2 = z.reshape(-1, 1)

        for var in CONTINUOUS_BEHAVIOR_VARS:
            y = _safe_series(behavior, var)
            if y is None:
                continue
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(y) & np.isfinite(z)
            if mask.sum() < 10:
                continue
            corr = float(np.corrcoef(z[mask], y[mask])[0, 1])
            model = Ridge(alpha=1.0)
            model.fit(z2[mask], y[mask])
            pred = model.predict(z2[mask])
            r2 = float(r2_score(y[mask], pred))
            rows.append({
                "latent_representation": representation,
                "latent_dimension": dim + 1,
                "behavioral_variable": var,
                "variable_type": "continuous",
                "association_score": r2,
                "correlation": corr,
                "metric": "R2",
                "causal_status": causal_status,
            })

        for var in CIRCULAR_BEHAVIOR_VARS:
            y = _safe_series(behavior, var)
            if y is None:
                continue
            y = np.asarray(y, dtype=float)
            mask = np.isfinite(y) & np.isfinite(z)
            if mask.sum() < 10:
                continue
            score = _circular_association(z[mask], y[mask])
            rows.append({
                "latent_representation": representation,
                "latent_dimension": dim + 1,
                "behavioral_variable": var,
                "variable_type": "circular",
                "association_score": score,
                "correlation": float("nan"),
                "metric": "circular_R2",
                "causal_status": causal_status,
            })

        for var in CATEGORICAL_BEHAVIOR_VARS:
            y = _safe_series(behavior, var)
            if y is None:
                continue
            y = np.asarray(y)
            mask = np.isfinite(z) & pd.notna(y)
            if mask.sum() < 10:
                continue
            le = LabelEncoder()
            try:
                y_enc = le.fit_transform(y[mask].astype(str))
            except Exception:
                continue
            if len(np.unique(y_enc)) < 2:
                continue
            clf = LogisticRegression(max_iter=500, random_state=0)
            try:
                n_splits = min(max_cv_splits, len(np.unique(y_enc)), int(mask.sum()) // 5)
                if n_splits >= 2:
                    scores = cross_val_score(clf, z2[mask], y_enc, cv=n_splits, scoring="accuracy")
                    acc = float(np.mean(scores))
                else:
                    clf.fit(z2[mask], y_enc)
                    acc = float(accuracy_score(y_enc, clf.predict(z2[mask])))
            except Exception:
                continue
            rows.append({
                "latent_representation": representation,
                "latent_dimension": dim + 1,
                "behavioral_variable": var,
                "variable_type": "categorical",
                "association_score": acc,
                "correlation": float("nan"),
                "metric": "classification_accuracy",
                "causal_status": causal_status,
            })

        # Position as 2D target from single dim (also report combined xy if both exist).
        if "x" in behavior.columns and "y" in behavior.columns:
            xy = behavior[["x", "y"]].to_numpy(dtype=float)
            mask = np.isfinite(xy).all(axis=1) & np.isfinite(z)
            if mask.sum() >= 10:
                model = Ridge(alpha=1.0)
                model.fit(z2[mask], xy[mask])
                pred = model.predict(z2[mask])
                r2 = float(r2_score(xy[mask], pred, multioutput="uniform_average"))
                rows.append({
                    "latent_representation": representation,
                    "latent_dimension": dim + 1,
                    "behavioral_variable": "position",
                    "variable_type": "continuous",
                    "association_score": r2,
                    "correlation": float("nan"),
                    "metric": "R2",
                    "causal_status": causal_status,
                })

    return pd.DataFrame(rows)
