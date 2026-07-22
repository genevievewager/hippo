"""Decoder model zoo: scikit-learn and Bayesian factories for comparison experiments."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR

from realtime.bayesian_decoder import (
    BayesianDistanceToWallDecoder,
    BayesianPlaceDecoder,
    BayesianPlaceDerivedDecoder,
)

# Continuous model families
QUICK_CONTINUOUS = ("ridge", "pca_ridge", "random_forest_regressor")
FULL_CONTINUOUS = (
    "ridge",
    "elastic_net",
    "pca_ridge",
    "pls_regression",
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
    "knn_regressor",
    "rbf_svr",
    "mlp_regressor",
    "bayesian_place_decoder",
    "bayesian_place_decoder_smoothed",
)

QUICK_CATEGORICAL = ("logistic_regression", "random_forest_classifier")
FULL_CATEGORICAL = (
    "logistic_regression",
    "linear_svm_classifier",
    "random_forest_classifier",
    "hist_gradient_boosting_classifier",
    "knn_classifier",
    "rbf_svc",
    "mlp_classifier",
    "bayesian_place_decoder_derived_context",
)

# Nonlinear output mappers (still compared against linear baselines on same features)
NONLINEAR_CONTINUOUS = frozenset({
    "random_forest_regressor",
    "hist_gradient_boosting_regressor",
    "knn_regressor",
    "rbf_svr",
    "mlp_regressor",
})
NONLINEAR_CATEGORICAL = frozenset({
    "random_forest_classifier",
    "hist_gradient_boosting_classifier",
    "knn_classifier",
    "rbf_svc",
    "mlp_classifier",
})

MULTI_OUTPUT_CONTINUOUS = frozenset({"position", "head_direction"})

# Targets for which Bayesian place / derived models apply
BAYESIAN_POSITION_TARGETS = frozenset({"position"})
BAYESIAN_DISTANCE_TARGETS = frozenset({"distance_to_wall"})
BAYESIAN_DERIVED_TARGETS = frozenset({"spatial_context", "wall_distance_bin"})

DEFAULT_BAYESIAN_PARAMS = {
    "n_bins": 20,
    "smooth_alpha": 0.7,
}

TARGET_FAMILY = {
    "position": "continuous",
    "speed": "continuous",
    "acceleration": "continuous",
    "head_direction": "continuous",
    "distance_to_wall": "continuous",
    "spatial_context": "categorical",
    "movement_state": "categorical",
    "wall_distance_bin": "categorical",
}


def continuous_model_names(mode: str, target: str | None = None) -> tuple[str, ...]:
    if mode == "full":
        names = list(FULL_CONTINUOUS)
    else:
        names = list(QUICK_CONTINUOUS)
        if target in BAYESIAN_POSITION_TARGETS or target in BAYESIAN_DISTANCE_TARGETS:
            names.append("bayesian_place_decoder")
    if target is not None:
        names = [n for n in names if _continuous_model_allowed(n, target)]
    return tuple(names)


def categorical_model_names(mode: str, target: str | None = None) -> tuple[str, ...]:
    if mode == "full":
        names = list(FULL_CATEGORICAL)
    else:
        names = list(QUICK_CATEGORICAL)
        if target in BAYESIAN_DERIVED_TARGETS:
            names.append("bayesian_place_decoder_derived_context")
    if target is not None:
        names = [n for n in names if _categorical_model_allowed(n, target)]
    return tuple(names)


def _continuous_model_allowed(name: str, target: str) -> bool:
    if name in ("bayesian_place_decoder", "bayesian_place_decoder_smoothed"):
        return target in BAYESIAN_POSITION_TARGETS or target in BAYESIAN_DISTANCE_TARGETS
    return True


def _categorical_model_allowed(name: str, target: str) -> bool:
    if name == "bayesian_place_decoder_derived_context":
        return target in BAYESIAN_DERIVED_TARGETS
    return True


def is_nonlinear_decoder(name: str) -> bool:
    return name in NONLINEAR_CONTINUOUS or name in NONLINEAR_CATEGORICAL


def default_model_params(name: str, seed: int = 42, n_jobs: int = -1) -> dict[str, Any]:
    """Serializable default hyperparameters for decoder_config_json."""
    if name == "ridge":
        return {"alpha": 1.0}
    if name == "elastic_net":
        return {"alpha": 0.1, "l1_ratio": 0.5, "max_iter": 5000, "random_state": seed}
    if name == "pca_ridge":
        return {"pca_n_components": 0.95, "ridge_alpha": 1.0}
    if name == "pls_regression":
        return {"n_components": 10}
    if name == "random_forest_regressor":
        return {"n_estimators": 100, "max_depth": 12, "random_state": seed, "n_jobs": n_jobs}
    if name == "hist_gradient_boosting_regressor":
        return {"max_depth": 8, "random_state": seed}
    if name == "knn_regressor":
        return {"n_neighbors": 15, "weights": "distance"}
    if name == "rbf_svr":
        return {"C": 1.0, "gamma": "scale", "kernel": "rbf"}
    if name == "mlp_regressor":
        return {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 400,
            "random_state": seed,
        }
    if name in ("bayesian_place_decoder", "bayesian_place_decoder_smoothed"):
        return {
            **DEFAULT_BAYESIAN_PARAMS,
            "smooth": name.endswith("smoothed"),
        }
    if name == "logistic_regression":
        return {"max_iter": 1000, "class_weight": "balanced", "random_state": seed}
    if name == "linear_svm_classifier":
        return {"max_iter": 5000, "class_weight": "balanced", "random_state": seed, "dual": False}
    if name == "random_forest_classifier":
        return {
            "n_estimators": 100,
            "max_depth": 12,
            "class_weight": "balanced",
            "random_state": seed,
            "n_jobs": n_jobs,
        }
    if name == "hist_gradient_boosting_classifier":
        return {"max_depth": 8, "random_state": seed}
    if name == "knn_classifier":
        return {"n_neighbors": 15, "weights": "distance"}
    if name == "rbf_svc":
        return {"C": 1.0, "gamma": "scale", "kernel": "rbf", "class_weight": "balanced"}
    if name == "mlp_classifier":
        return {
            "hidden_layer_sizes": (64, 32),
            "max_iter": 400,
            "random_state": seed,
        }
    if name == "bayesian_place_decoder_derived_context":
        return {**DEFAULT_BAYESIAN_PARAMS, "smooth": False}
    return {}


def _base_continuous_estimator(name: str, seed: int, n_jobs: int):
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "elastic_net":
        return ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000, random_state=seed)
    if name == "pca_ridge":
        return Pipeline([
            ("pca", PCA(n_components=0.95, random_state=seed)),
            ("ridge", Ridge(alpha=1.0)),
        ])
    if name == "pls_regression":
        return PLSRegression(n_components=10)
    if name == "random_forest_regressor":
        return RandomForestRegressor(
            n_estimators=100, max_depth=12, random_state=seed, n_jobs=n_jobs,
        )
    if name == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(max_depth=8, random_state=seed)
    if name == "knn_regressor":
        return KNeighborsRegressor(n_neighbors=15, weights="distance")
    if name == "rbf_svr":
        return SVR(kernel="rbf", C=1.0, gamma="scale")
    if name == "mlp_regressor":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed,
        )
    raise ValueError(f"Unknown continuous model: {name}")


def _base_categorical_estimator(name: str, seed: int, n_jobs: int):
    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed,
        )
    if name == "linear_svm_classifier":
        return LinearSVC(
            max_iter=5000, class_weight="balanced", random_state=seed, dual=False,
        )
    if name == "random_forest_classifier":
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, class_weight="balanced",
            random_state=seed, n_jobs=n_jobs,
        )
    if name == "hist_gradient_boosting_classifier":
        return HistGradientBoostingClassifier(max_depth=8, random_state=seed)
    if name == "knn_classifier":
        return KNeighborsClassifier(n_neighbors=15, weights="distance")
    if name == "rbf_svc":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    if name == "mlp_classifier":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=400, random_state=seed,
        )
    raise ValueError(f"Unknown categorical model: {name}")


def make_continuous_pipeline(
    name: str,
    target_name: str,
    seed: int = 42,
    n_jobs: int = -1,
    arena_bounds: tuple[float, float, float, float] | None = None,
) -> Any:
    """Build a continuous decoder (Pipeline or Bayesian estimator)."""
    if name in ("bayesian_place_decoder", "bayesian_place_decoder_smoothed"):
        smooth = name.endswith("smoothed")
        if target_name == "distance_to_wall":
            return BayesianDistanceToWallDecoder(
                n_bins=DEFAULT_BAYESIAN_PARAMS["n_bins"],
                smooth=smooth,
                smooth_alpha=DEFAULT_BAYESIAN_PARAMS["smooth_alpha"],
                arena_bounds=arena_bounds,
            )
        return BayesianPlaceDecoder(
            n_bins=DEFAULT_BAYESIAN_PARAMS["n_bins"],
            smooth=smooth,
            smooth_alpha=DEFAULT_BAYESIAN_PARAMS["smooth_alpha"],
            arena_bounds=arena_bounds,
        )

    estimator = _base_continuous_estimator(name, seed, n_jobs)
    needs_multi = target_name in MULTI_OUTPUT_CONTINUOUS and name != "pls_regression"
    if needs_multi:
        estimator = MultiOutputRegressor(estimator)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def make_categorical_pipeline(
    name: str,
    seed: int = 42,
    n_jobs: int = -1,
    target_name: str = "spatial_context",
    arena_bounds: tuple[float, float, float, float] | None = None,
) -> Any:
    """Build a categorical decoder (Pipeline or Bayesian derived estimator)."""
    if name == "bayesian_place_decoder_derived_context":
        return BayesianPlaceDerivedDecoder(
            derived_target=target_name,
            n_bins=DEFAULT_BAYESIAN_PARAMS["n_bins"],
            smooth=False,
            smooth_alpha=DEFAULT_BAYESIAN_PARAMS["smooth_alpha"],
            arena_bounds=arena_bounds,
        )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", _base_categorical_estimator(name, seed, n_jobs)),
    ])


def clone_pipeline(pipeline: Any) -> Any:
    return clone(pipeline)


def is_bayesian_model(name: str) -> bool:
    return name.startswith("bayesian_place_decoder")
