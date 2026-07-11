"""Decoder model zoo: scikit-learn model factories for comparison experiments."""

from __future__ import annotations

from sklearn.base import clone
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

QUICK_CONTINUOUS = ("ridge", "random_forest_regressor")
FULL_CONTINUOUS = (
    "ridge",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "knn_regressor",
)

QUICK_CATEGORICAL = ("logistic_regression", "random_forest_classifier")
FULL_CATEGORICAL = (
    "logistic_regression",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "knn_classifier",
)

MULTI_OUTPUT_CONTINUOUS = frozenset({"position", "head_direction"})


def continuous_model_names(mode: str) -> tuple[str, ...]:
    if mode == "full":
        return FULL_CONTINUOUS
    return QUICK_CONTINUOUS


def categorical_model_names(mode: str) -> tuple[str, ...]:
    if mode == "full":
        return FULL_CATEGORICAL
    return QUICK_CATEGORICAL


def _base_continuous_estimator(name: str, seed: int, n_jobs: int):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "gradient_boosting_regressor":
        return GradientBoostingRegressor(random_state=seed)
    if name == "knn_regressor":
        return KNeighborsRegressor(n_neighbors=15, weights="distance")
    raise ValueError(f"Unknown continuous model: {name}")


def _base_categorical_estimator(name: str, seed: int, n_jobs: int):
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    if name == "random_forest_classifier":
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "gradient_boosting_classifier":
        return GradientBoostingClassifier(random_state=seed)
    if name == "knn_classifier":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    raise ValueError(f"Unknown categorical model: {name}")


def make_continuous_pipeline(
    name: str,
    target_name: str,
    seed: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build a scaled continuous decoder pipeline."""
    estimator = _base_continuous_estimator(name, seed, n_jobs)
    if target_name in MULTI_OUTPUT_CONTINUOUS:
        estimator = MultiOutputRegressor(estimator)
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def make_categorical_pipeline(
    name: str,
    seed: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    """Build a scaled categorical decoder pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", _base_categorical_estimator(name, seed, n_jobs)),
    ])


def clone_pipeline(pipeline: Pipeline) -> Pipeline:
    return clone(pipeline)
