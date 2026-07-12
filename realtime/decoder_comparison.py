"""Decoder comparison and causal window-size optimization experiments.

Spike counts are causal population-vector features: at time t, each unit's
count in [t - window, t) forms the feature vector. Different windows trade
latency against spike-count reliability — sparse hippocampal firing may need
longer windows for position/spatial context, while speed/movement may decode
from shorter windows when many units are speed-modulated.

Head direction uses sine/cosine targets because angles are circular.
Ground-truth spikes estimate best-case neural information; sorted spikes
estimate what remains after Neuropixels degradation and spike sorting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from realtime.bayesian_decoder import (
    BayesianDistanceToWallDecoder,
    BayesianPlaceDecoder,
    BayesianPlaceDerivedDecoder,
)
from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.decoder_models import (
    TARGET_FAMILY,
    categorical_model_names,
    continuous_model_names,
    default_model_params,
    is_bayesian_model,
    make_categorical_pipeline,
    make_continuous_pipeline,
)
from realtime.decoding_targets import (
    align_extended_behavior_to_decoder_times,
    angles_from_sin_cos,
    circular_error_deg,
)
from realtime.spike_features import apply_feature_mode, build_causal_spike_matrix
from realtime.train_decoder import causal_train_test_split, infer_arena_bounds

DEFAULT_DECODE_WINDOWS = (0.025, 0.050, 0.100, 0.250, 0.500, 1.000)

CONTINUOUS_TARGETS = (
    "position",
    "speed",
    "acceleration",
    "head_direction",
    "distance_to_wall",
)
CATEGORICAL_TARGETS = (
    "spatial_context",
    "movement_state",
    "wall_distance_bin",
)
ALL_TARGETS = CONTINUOUS_TARGETS + CATEGORICAL_TARGETS

PRIMARY_METRIC = {
    "position": ("mean_position_error_cm", "lower"),
    "speed": ("r2", "higher"),
    "acceleration": ("r2", "higher"),
    "head_direction": ("mean_circular_error_deg", "lower"),
    "distance_to_wall": ("r2", "higher"),
    "spatial_context": ("balanced_accuracy", "higher"),
    "movement_state": ("balanced_accuracy", "higher"),
    "wall_distance_bin": ("balanced_accuracy", "higher"),
}


@dataclass
class FitResult:
    metrics: dict[str, Any]
    y_true: np.ndarray | pd.Series
    y_pred: np.ndarray
    pipeline: Any
    labels: list[str] | None = None


@dataclass
class ComparisonRunConfig:
    input_dir: Path
    output_dir: Path
    spike_source: str = "sorted"
    decode_windows: tuple[float, ...] = DEFAULT_DECODE_WINDOWS
    update_dt: float = 0.025
    train_frac: float = 0.70
    feature_modes: tuple[str, ...] = ("counts",)
    max_models: str = "quick"
    n_jobs: int = -1
    seed: int = 42


def _get_y(behavior: pd.DataFrame, target: str) -> np.ndarray:
    if target == "position":
        return behavior[["x", "y"]].to_numpy()
    if target == "head_direction":
        return behavior[["head_direction_sin", "head_direction_cos"]].to_numpy()
    if target in behavior.columns:
        return behavior[target].to_numpy()
    raise KeyError(f"Unknown target: {target}")


def _fit_estimator(
    estimator: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    behavior_train: pd.DataFrame,
    decoder_name: str,
) -> Any:
    if is_bayesian_model(decoder_name):
        position_xy = behavior_train[["x", "y"]].to_numpy()
        if isinstance(estimator, (BayesianPlaceDerivedDecoder, BayesianDistanceToWallDecoder)):
            estimator.fit(X_train, position_xy=position_xy)
        else:
            estimator.fit(X_train, position_xy)
        return estimator
    estimator.fit(X_train, y_train)
    return estimator


def _predict_labels(estimator: Any) -> list[str] | None:
    if hasattr(estimator, "classes_"):
        return list(estimator.classes_)
    if hasattr(estimator, "named_steps") and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
        if hasattr(model, "classes_"):
            return list(model.classes_)
    return None


def _fit_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    behavior_train: pd.DataFrame,
    behavior_test: pd.DataFrame,
    target: str,
    decoder_name: str,
    seed: int,
    n_jobs: int,
    arena_bounds: tuple[float, float, float, float] | None,
) -> FitResult:
    y_train = _get_y(behavior_train, target)
    y_test = _get_y(behavior_test, target)

    if target in CATEGORICAL_TARGETS:
        pipeline = make_categorical_pipeline(
            decoder_name, seed=seed, n_jobs=n_jobs,
            target_name=target, arena_bounds=arena_bounds,
        )
        pipeline = _fit_estimator(pipeline, X_train, y_train, behavior_train, decoder_name)
        y_pred = pipeline.predict(X_test)
        labels = _predict_labels(pipeline) or sorted(set(y_test) | set(y_pred))
        metrics = _classification_metrics(y_test, y_pred, labels)
        return FitResult(
            metrics=metrics, y_true=y_test, y_pred=y_pred, pipeline=pipeline, labels=labels,
        )

    pipeline = make_continuous_pipeline(
        decoder_name, target, seed=seed, n_jobs=n_jobs, arena_bounds=arena_bounds,
    )
    pipeline = _fit_estimator(pipeline, X_train, y_train, behavior_train, decoder_name)
    y_pred = np.asarray(pipeline.predict(X_test))
    if target == "distance_to_wall" and y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    metrics = _continuous_metrics(target, y_test, y_pred, behavior_test)
    return FitResult(metrics=metrics, y_true=y_test, y_pred=y_pred, pipeline=pipeline)


def _classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_labels": labels,
    }


def _continuous_metrics(
    target: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    behavior_test: pd.DataFrame,
) -> dict[str, Any]:
    if target == "position":
        y_pred = np.asarray(y_pred)
        if y_pred.ndim == 1:
            raise ValueError("Position predictions must be 2D")
        pos_err = np.linalg.norm(y_pred - y_true, axis=1)
        return {
            "mean_position_error_cm": float(np.mean(pos_err)),
            "median_position_error_cm": float(np.median(pos_err)),
            "p90_position_error_cm": float(np.percentile(pos_err, 90)),
            "r2_x": float(r2_score(y_true[:, 0], y_pred[:, 0])),
            "r2_y": float(r2_score(y_true[:, 1], y_pred[:, 1])),
        }

    if target == "head_direction":
        true_angles = angles_from_sin_cos(y_true[:, 0], y_true[:, 1])
        pred_angles = angles_from_sin_cos(y_pred[:, 0], y_pred[:, 1])
        circ_err = circular_error_deg(true_angles, pred_angles)
        return {
            "mean_circular_error_deg": float(np.mean(circ_err)),
            "median_circular_error_deg": float(np.median(circ_err)),
            "p90_circular_error_deg": float(np.percentile(circ_err, 90)),
        }

    y_true_1d = np.asarray(y_true).ravel()
    y_pred_1d = np.asarray(y_pred).ravel()
    corr = float(np.corrcoef(y_true_1d, y_pred_1d)[0, 1]) if len(y_true_1d) > 1 else float("nan")
    metrics: dict[str, Any] = {
        "r2": float(r2_score(y_true_1d, y_pred_1d)),
        "correlation": corr,
        "mae": float(mean_absolute_error(y_true_1d, y_pred_1d)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_1d, y_pred_1d))),
    }
    if target == "distance_to_wall":
        metrics["mae_cm"] = metrics.pop("mae")
        metrics["rmse_cm"] = metrics.pop("rmse")
    return metrics


def _is_better(value: float, direction: str, other: float) -> bool:
    if direction == "lower":
        return value < other
    return value > other


def _near_optimal(value: float, best: float, direction: str) -> bool:
    if direction == "lower":
        return value <= 1.05 * best
    return value >= 0.95 * best


def run_decoder_comparison(config: ComparisonRunConfig) -> pd.DataFrame:
    """Run full decoder/window comparison for one spike source."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    examples_dir = output_dir / "decoded_examples"
    models_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    data = load_simulation_data(config.input_dir, config.spike_source)
    arena_bounds = infer_arena_bounds(data["behavior_df"], data["summary"])

    rows: list[dict[str, Any]] = []
    best_fits: dict[str, FitResult] = {}
    best_meta: dict[str, dict[str, Any]] = {}

    for feature_mode in config.feature_modes:
        for decode_window in config.decode_windows:
            decode_times = make_decode_times(
                data["session_duration"], decode_window, config.update_dt
            )
            aligned = align_extended_behavior_to_decoder_times(
                data["behavior_df"], decode_times, data["summary"]
            )
            X_all = build_causal_spike_matrix(
                data["spikes_df"],
                data["unit_ids"],
                decode_times,
                decode_window,
            )
            X_all = apply_feature_mode(X_all, feature_mode, decode_window)

            train_mask, test_mask = causal_train_test_split(decode_times, config.train_frac)
            X_train = X_all[train_mask]
            X_test = X_all[test_mask]
            beh_train = aligned.loc[train_mask].reset_index(drop=True)
            beh_test = aligned.loc[test_mask].reset_index(drop=True)

            for target in CONTINUOUS_TARGETS:
                for decoder_name in continuous_model_names(config.max_models, target):
                    fit = _fit_and_evaluate(
                        X_train, X_test, beh_train, beh_test,
                        target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                    )
                    row = _base_row(config, data, feature_mode, decode_window)
                    row.update({
                        "target_name": target,
                        "target_family": TARGET_FAMILY[target],
                        "decoder_name": decoder_name,
                        "n_train_samples": int(train_mask.sum()),
                        "n_test_samples": int(test_mask.sum()),
                    })
                    row.update(fit.metrics)
                    rows.append(row)
                    _maybe_update_best(best_fits, best_meta, target, row, fit, beh_test)

            for target in CATEGORICAL_TARGETS:
                for decoder_name in categorical_model_names(config.max_models, target):
                    fit = _fit_and_evaluate(
                        X_train, X_test, beh_train, beh_test,
                        target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                    )
                    row = _base_row(config, data, feature_mode, decode_window)
                    row.update({
                        "target_name": target,
                        "target_family": TARGET_FAMILY[target],
                        "decoder_name": decoder_name,
                        "n_train_samples": int(train_mask.sum()),
                        "n_test_samples": int(test_mask.sum()),
                    })
                    row.update(fit.metrics)
                    rows.append(row)
                    _maybe_update_best(best_fits, best_meta, target, row, fit, beh_test)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "decoder_comparison_metrics.csv", index=False)
    with open(output_dir / "decoder_comparison_metrics.json", "w") as f:
        json.dump(_json_safe(rows), f, indent=2)

    best_df = _build_best_decoder_table(metrics_df, best_fits, models_dir, config)
    best_df.to_csv(output_dir / "best_decoder_by_target.csv", index=False)
    with open(output_dir / "best_decoder_by_target.json", "w") as f:
        json.dump(_json_safe(best_df.to_dict(orient="records")), f, indent=2)

    _save_best_models(best_fits, best_meta, models_dir)
    _save_best_predictions(best_fits, best_meta, examples_dir)
    return metrics_df


def _base_row(
    config: ComparisonRunConfig,
    data: dict,
    feature_mode: str,
    decode_window: float,
) -> dict[str, Any]:
    return {
        "spike_source": config.spike_source,
        "source": config.spike_source,
        "feature_type": feature_mode,
        "decode_window_s": decode_window,
        "update_dt_s": config.update_dt,
        "n_units": len(data["unit_ids"]),
    }


def _maybe_update_best(
    best_fits: dict[str, FitResult],
    best_meta: dict[str, dict[str, Any]],
    target: str,
    row: dict[str, Any],
    fit: FitResult,
    beh_test: pd.DataFrame,
) -> None:
    metric_key, direction = PRIMARY_METRIC[target]
    value = float(row[metric_key])
    if target not in best_fits or _is_better(
        value, direction, best_meta[target]["primary_metric_value"]
    ):
        best_fits[target] = fit
        best_meta[target] = {
            **row,
            "primary_metric_value": value,
            "behavior_test": beh_test,
        }


def _build_decoder_config(
    decoder_name: str,
    decode_window_s: float,
    feature_type: str,
    target_name: str,
    seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    params = default_model_params(decoder_name, seed=seed, n_jobs=n_jobs)
    config: dict[str, Any] = {
        "decoder_name": decoder_name,
        "decode_window_s": decode_window_s,
        "feature_type": feature_type,
        "target_name": target_name,
        "target_family": TARGET_FAMILY[target_name],
        "model_params": params,
    }
    if is_bayesian_model(decoder_name):
        config["bayesian_params"] = {
            k: params[k] for k in ("n_bins", "smooth", "smooth_alpha") if k in params
        }
    else:
        config["bayesian_params"] = {}
    return config


def _build_best_decoder_table(
    metrics_df: pd.DataFrame,
    best_fits: dict[str, FitResult],
    models_dir: Path,
    config: ComparisonRunConfig,
) -> pd.DataFrame:
    summary_rows = []
    for target in ALL_TARGETS:
        target_df = metrics_df[metrics_df["target_name"] == target].copy()
        if target_df.empty:
            continue
        metric_key, direction = PRIMARY_METRIC[target]
        if direction == "lower":
            best_idx = target_df[metric_key].idxmin()
        else:
            best_idx = target_df[metric_key].idxmax()
        best_row = target_df.loc[best_idx]
        best_value = float(best_row[metric_key])
        recommended_window = _recommended_window(target_df, metric_key, direction, best_value)

        model_path = models_dir / f"best_{target}_decoder.joblib"
        decoder_config = _build_decoder_config(
            str(best_row["decoder_name"]),
            float(best_row["decode_window_s"]),
            str(best_row["feature_type"]),
            target,
            config.seed,
            config.n_jobs,
        )

        summary_rows.append({
            "target_name": target,
            "target_family": TARGET_FAMILY[target],
            "primary_metric": metric_key,
            "best_decoder_name": best_row["decoder_name"],
            "best_decode_window_s": float(best_row["decode_window_s"]),
            "recommended_realtime_window_s": recommended_window,
            "best_feature_type": best_row["feature_type"],
            "best_metric_value": best_value,
            "spike_source": best_row["spike_source"],
            "source": best_row.get("source", best_row["spike_source"]),
            "model_path": str(model_path) if target in best_fits else "",
            "decoder_config_json": json.dumps(decoder_config),
        })
    return pd.DataFrame(summary_rows)


def _recommended_window(
    target_df: pd.DataFrame,
    metric_key: str,
    direction: str,
    best_value: float,
) -> float:
    windows = sorted(target_df["decode_window_s"].unique())
    for window in windows:
        window_df = target_df[target_df["decode_window_s"] == window]
        if window_df.empty:
            continue
        window_best = (
            window_df[metric_key].min() if direction == "lower"
            else window_df[metric_key].max()
        )
        if _near_optimal(float(window_best), best_value, direction):
            return float(window)
    return float(windows[0])


def _save_best_models(
    best_fits: dict[str, FitResult],
    best_meta: dict[str, dict[str, Any]],
    models_dir: Path,
) -> None:
    for target, fit in best_fits.items():
        meta = {k: v for k, v in best_meta[target].items() if k != "behavior_test"}
        joblib.dump(fit.pipeline, models_dir / f"best_{target}_decoder.joblib")
        with open(models_dir / f"best_{target}_meta.json", "w") as f:
            json.dump(_json_safe(meta), f, indent=2)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _save_best_predictions(
    best_fits: dict[str, FitResult],
    best_meta: dict[str, dict[str, Any]],
    examples_dir: Path,
) -> None:
    for target, fit in best_fits.items():
        beh = best_meta[target]["behavior_test"]
        time = beh["time"].to_numpy()

        if target == "position":
            pred = np.asarray(fit.y_pred)
            pd.DataFrame({
                "time": time,
                "true_x": beh["x"].to_numpy(),
                "true_y": beh["y"].to_numpy(),
                "pred_x": pred[:, 0],
                "pred_y": pred[:, 1],
            }).to_csv(examples_dir / "best_position_predictions.csv", index=False)
            continue

        if target == "head_direction":
            true_angles = np.degrees(angles_from_sin_cos(
                fit.y_true[:, 0], fit.y_true[:, 1],
            ))
            pred_angles = np.degrees(angles_from_sin_cos(
                fit.y_pred[:, 0], fit.y_pred[:, 1],
            ))
            pd.DataFrame({
                "time": time,
                "true": true_angles,
                "pred": pred_angles,
            }).to_csv(examples_dir / "best_head_direction_predictions.csv", index=False)
            continue

        if target in CATEGORICAL_TARGETS:
            pd.DataFrame({
                "time": time,
                "true": fit.y_true,
                "pred": fit.y_pred,
            }).to_csv(examples_dir / f"best_{target}_predictions.csv", index=False)
            continue

        pd.DataFrame({
            "time": time,
            "true": np.asarray(fit.y_true).ravel(),
            "pred": np.asarray(fit.y_pred).ravel(),
        }).to_csv(examples_dir / f"best_{target}_predictions.csv", index=False)


def run_compare_sources(
    input_dir: Path,
    output_dir: Path,
    decode_windows: tuple[float, ...] = DEFAULT_DECODE_WINDOWS,
    update_dt: float = 0.025,
    train_frac: float = 0.70,
    feature_modes: tuple[str, ...] = ("counts",),
    max_models: str = "quick",
    n_jobs: int = -1,
    seed: int = 42,
) -> pd.DataFrame:
    """Run decoder comparison for ground_truth and sorted spikes."""
    output_dir = Path(output_dir)
    best_tables = []

    for source in ("ground_truth", "sorted"):
        run_decoder_comparison(ComparisonRunConfig(
            input_dir=input_dir,
            output_dir=output_dir / source,
            spike_source=source,
            decode_windows=decode_windows,
            update_dt=update_dt,
            train_frac=train_frac,
            feature_modes=feature_modes,
            max_models=max_models,
            n_jobs=n_jobs,
            seed=seed,
        ))
        best_path = output_dir / source / "best_decoder_by_target.csv"
        if best_path.exists():
            bt = pd.read_csv(best_path)
            bt["source"] = source
            best_tables.append(bt)

    if not best_tables:
        return pd.DataFrame()

    comparison_df = pd.concat(best_tables, ignore_index=True)
    comparison_df.to_csv(output_dir / "source_comparison_metrics.csv", index=False)
    # Also write combined best table at root for convenience
    comparison_df.to_csv(output_dir / "best_decoder_by_target.csv", index=False)
    with open(output_dir / "best_decoder_by_target.json", "w") as f:
        json.dump(_json_safe(comparison_df.to_dict(orient="records")), f, indent=2)
    return comparison_df
