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
    is_nonlinear_decoder,
    make_categorical_pipeline,
    make_continuous_pipeline,
)
from realtime.decoding_targets import (
    align_extended_behavior_to_decoder_times,
    angles_from_sin_cos,
    circular_error_deg,
)
from realtime.comparison_enrichment import (
    enrich_fit_row,
    per_sample_window_frame,
    window_matrix_stats,
)
from realtime.feature_representations import make_spike_feature_transformer
from realtime.manifold_features import (
    DEFAULT_ISOMAP_N_NEIGHBORS,
    OFFLINE_ONLY_FEATURE_MODES,
    QUICK_FEATURE_MODES,
    is_realtime_compatible_feature_mode,
    make_feature_transformer,
    manifold_transform_dirname,
    manifold_type_for_feature_mode,
)
from realtime.search_space import (
    compose_feature_mode,
    embedding_needs_targets,
    expand_fe_jobs,
)
from realtime.spike_features import build_causal_spike_matrix
from realtime.train_decoder import causal_train_test_split, infer_arena_bounds
from realtime.trigger_comparison import default_trigger_rules

DEFAULT_DECODE_WINDOWS = (0.025, 0.050, 0.100, 0.250, 0.500, 1.000)
DEFAULT_MANIFOLD_N_COMPONENTS = (3,)

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
    update_dt: float = 0.050
    train_frac: float = 0.70
    # Legacy combined modes (backward compatible with --feature-modes).
    feature_modes: tuple[str, ...] = QUICK_FEATURE_MODES
    # New F × E search dimensions (when set, expand Cartesian product).
    feature_types: tuple[str, ...] | None = None
    embedding_types: tuple[str, ...] | None = None
    use_fe_grid: bool = False
    manifold_n_components: tuple[int, ...] = DEFAULT_MANIFOLD_N_COMPONENTS
    isomap_n_neighbors: tuple[int, ...] = (DEFAULT_ISOMAP_N_NEIGHBORS,)
    isomap_pre_pca_enabled: bool = True
    isomap_pre_pca_n_components: int = 50
    isomap_require_connected_graph: bool = True
    max_models: str = "quick"
    n_jobs: int = -1
    seed: int = 42
    align_to_behavior: bool = True
    region_ablation: bool = False
    layer_ablation: bool = False
    population_ablation: bool = False
    adaptive_windows: bool = False
    include_controls: bool = False
    max_compute_ms: float = 25.0
    max_effective_history_s: float = 0.500
    # Closed-loop trigger search (C)
    enable_trigger_search: bool = True
    trigger_context: str | None = "wall"
    trigger_confidence: float = 0.80
    trigger_wall_bin: str | None = "near_wall"
    trigger_distance_lt_cm: float | None = 10.0
    trigger_speed_gt_cm_s: float | None = 10.0
    trigger_movement: str | None = "fast"
    trigger_hd_center_deg: float | None = 90.0
    trigger_hd_width_deg: float | None = 30.0
    # Multi-run id (optional annotation)
    run_id: str | None = None


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


def _expand_feature_jobs(
    feature_modes: tuple[str, ...],
    manifold_n_components: tuple[int, ...],
    isomap_n_neighbors: tuple[int, ...] = (DEFAULT_ISOMAP_N_NEIGHBORS,),
) -> list[tuple[str, int | None, int | None]]:
    """Legacy helper: return (feature_mode, n_components, n_neighbors) jobs."""
    jobs = expand_fe_jobs(
        feature_modes=feature_modes,
        manifold_n_components=manifold_n_components,
        isomap_n_neighbors=isomap_n_neighbors,
        use_fe_grid=False,
    )
    # Map (F, E, k, nn) → legacy (feature_mode, k, nn)
    return [
        (compose_feature_mode(f, e), k, nn) for f, e, k, nn in jobs
    ]


def _get_y_for_embedding(behavior: pd.DataFrame, target: str) -> np.ndarray:
    """Supervised embedding target; use position for categorical / place-like."""
    if target in CATEGORICAL_TARGETS or target in ("position", "distance_to_wall"):
        return behavior[["x", "y"]].to_numpy()
    if target == "head_direction":
        return behavior[["head_direction_sin", "head_direction_cos"]].to_numpy()
    return _get_y(behavior, target)


def run_decoder_comparison(config: ComparisonRunConfig) -> pd.DataFrame:
    """Run full F×E×D×W(+C) comparison for one spike source."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    examples_dir = output_dir / "decoded_examples"
    manifold_dir = models_dir / "manifold_transforms"
    feature_dir = models_dir / "feature_transforms"
    models_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)
    manifold_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)

    data = load_simulation_data(config.input_dir, config.spike_source)
    arena_bounds = infer_arena_bounds(data["behavior_df"], data["summary"])
    from realtime.timing import extract_behavior_times, resolve_update_dt_s

    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"],
        derive_from_behavior=config.align_to_behavior,
        update_dt_s=config.update_dt,
        behavior_times=behavior_times,
    )

    fe_jobs = expand_fe_jobs(
        feature_types=config.feature_types,
        embedding_types=config.embedding_types,
        feature_modes=config.feature_modes,
        manifold_n_components=tuple(config.manifold_n_components),
        isomap_n_neighbors=tuple(config.isomap_n_neighbors),
        max_models=config.max_models,
        use_fe_grid=bool(
            config.use_fe_grid
            or config.feature_types is not None
            or config.embedding_types is not None
        ),
    )

    trigger_rules = (
        default_trigger_rules(
            trigger_context=config.trigger_context,
            trigger_confidence=config.trigger_confidence,
            trigger_wall_bin=config.trigger_wall_bin,
            trigger_distance_lt_cm=config.trigger_distance_lt_cm,
            trigger_speed_gt_cm_s=config.trigger_speed_gt_cm_s,
            trigger_movement=config.trigger_movement,
            trigger_hd_center_deg=config.trigger_hd_center_deg,
            trigger_hd_width_deg=config.trigger_hd_width_deg,
        )
        if config.enable_trigger_search
        else []
    )

    rows: list[dict[str, Any]] = []
    explained_rows: list[dict[str, Any]] = []
    trigger_rows_all: list[dict[str, Any]] = []
    control_rows_all: list[dict[str, Any]] = []
    best_fits: dict[str, FitResult] = {}
    best_meta: dict[str, dict[str, Any]] = {}
    best_by_window: dict[tuple[str, float, str, str], FitResult] = {}
    best_by_window_meta: dict[tuple[str, float, str, str], dict[str, Any]] = {}

    window_queue = list(dict.fromkeys(float(w) for w in config.decode_windows))
    tested_windows: list[float] = []
    refined_once = False
    wi = 0
    while wi < len(window_queue):
        decode_window = float(window_queue[wi])
        wi += 1
        print(f"  window W={decode_window:.3f}s "
              f"({len(tested_windows) + 1}/{len(window_queue)} queued)...")
        decode_times = make_decode_times(
            data["session_duration"],
            decode_window,
            update_dt,
            behavior_times=behavior_times if config.align_to_behavior else None,
        )
        aligned = align_extended_behavior_to_decoder_times(
            data["behavior_df"], decode_times, data["summary"]
        )
        X_counts = build_causal_spike_matrix(
            data["spikes_df"],
            data["unit_ids"],
            decode_times,
            decode_window,
        )
        w_stats = window_matrix_stats(X_counts)
        train_mask, test_mask = causal_train_test_split(decode_times, config.train_frac)
        beh_train = aligned.loc[train_mask].reset_index(drop=True)
        beh_test = aligned.loc[test_mask].reset_index(drop=True)
        decode_times_test = decode_times[test_mask]
        X_counts_test = X_counts[test_mask]

        # Persist per-sample window metadata for this W (overwrite each window).
        per_sample_window_frame(
            decode_times, X_counts,
            decode_window_s=decode_window, update_dt_s=update_dt,
        ).to_csv(output_dir / f"window_samples_w{decode_window:.3f}s.csv", index=False)

        for feature_type, embedding_type, n_components, n_neighbors in fe_jobs:
            feature_mode = compose_feature_mode(feature_type, embedding_type)
            try:
                f_transform = make_spike_feature_transformer(
                    feature_type,
                    decode_window=decode_window,
                    units_df=data["units_df"],
                    unit_ids=data["unit_ids"],
                )
            except Exception as exc:
                print(f"  skip F={feature_type}: {exc}")
                continue

            X_train_raw = X_counts[train_mask]
            X_test_raw = X_counts[test_mask]
            f_transform.fit(X_train_raw)
            X_train_f = f_transform.transform(X_train_raw)
            X_test_f = f_transform.transform(X_test_raw)

            f_name = f"{feature_type}_w{int(round(decode_window * 1000)):04d}ms"
            f_path = feature_dir / f_name
            f_transform.save(f_path)

            # Supervised embeddings are fit inside the target loop.
            if embedding_needs_targets(embedding_type):
                unsupervised_embed = None
            else:
                unsupervised_embed = make_feature_transformer(
                    embedding_type if embedding_type != "identity" else "identity",
                    decode_window=decode_window,
                    n_components=n_components or 3,
                    units_df=data["units_df"],
                    unit_ids=data["unit_ids"],
                    random_state=config.seed,
                    n_neighbors=n_neighbors or DEFAULT_ISOMAP_N_NEIGHBORS,
                    isomap_pre_pca_enabled=config.isomap_pre_pca_enabled,
                    isomap_pre_pca_n_components=config.isomap_pre_pca_n_components,
                    isomap_require_connected_graph=config.isomap_require_connected_graph,
                    n_jobs=config.n_jobs,
                )
                if unsupervised_embed is None:
                    print(f"  skip E={embedding_type} (missing unit metadata)")
                    continue
                try:
                    unsupervised_embed.fit(X_train_f)
                except Exception as exc:
                    from realtime.manifolds.isomap_diagnostics import DisconnectedGraphError

                    if isinstance(exc, DisconnectedGraphError) or "disconnected" in str(exc).lower():
                        print(
                            f"  exclude {feature_mode} k={n_components} "
                            f"nn={n_neighbors}: {exc}"
                        )
                        rows.append({
                            **_base_row(
                                config, data, feature_type, decode_window, update_dt,
                                embedding_type=embedding_type,
                            ),
                            "manifold_type": manifold_type_for_feature_mode(embedding_type),
                            "manifold_n_components": n_components,
                            "n_neighbors": n_neighbors,
                            "exclusion_reason": str(exc),
                            "graph_connected": False,
                            "realtime_compatible": False,
                            "target_name": None,
                            "n_train_samples": int(train_mask.sum()),
                            "n_test_samples": int(test_mask.sum()),
                        })
                        continue
                    raise

                X_train_shared = unsupervised_embed.transform(X_train_f)
                X_test_shared = unsupervised_embed.transform(X_test_f)
                meta = unsupervised_embed.get_metadata()
                transform_name = manifold_transform_dirname(
                    feature_mode, decode_window, n_components, n_neighbors,
                )
                transform_path = manifold_dir / transform_name
                unsupervised_embed.save(transform_path)
                for group in meta.get("groups", []) or []:
                    explained_rows.append({
                        "spike_source": config.spike_source,
                        "decode_window_s": float(decode_window),
                        "feature_type": feature_type,
                        "embedding_type": embedding_type,
                        "group_name": group.get("group_name"),
                        "n_units": group.get("n_units"),
                        "n_components": group.get("n_components"),
                        "explained_variance_sum": group.get("explained_variance_sum"),
                        "explained_variance_by_component": json.dumps(
                            group.get("explained_variance_by_component")
                        ),
                    })
                geo = meta.get("geometry_metrics") or {}
                diag = meta.get("graph_diagnostics") or {}
                shared_extras = {
                    "manifold_type": meta.get("manifold_type")
                    or manifold_type_for_feature_mode(embedding_type),
                    "manifold_grouping": meta.get("manifold_grouping"),
                    "manifold_n_components": n_components,
                    "n_neighbors": n_neighbors if n_neighbors is not None else meta.get("n_neighbors"),
                    "pre_pca_enabled": meta.get("pre_pca_enabled"),
                    "pre_pca_dim": meta.get("pre_pca_dim"),
                    "actual_n_features": meta.get("actual_n_features"),
                    "explained_variance_ratio": (
                        json.dumps(meta.get("explained_variance_ratio"))
                        if meta.get("explained_variance_ratio") is not None else None
                    ),
                    "manifold_transform_path": str(transform_path),
                    "embedding_transform_path": str(transform_path),
                    "realtime_compatible": bool(
                        meta.get(
                            "realtime_compatible",
                            is_realtime_compatible_feature_mode(embedding_type)
                            if embedding_type != "identity"
                            else True,
                        )
                    ),
                    "graph_connected": diag.get("graph_connected"),
                    "largest_component_fraction": diag.get("largest_component_fraction"),
                    "trustworthiness": geo.get("trustworthiness"),
                    "residual_variance": geo.get("residual_variance"),
                    "geodesic_distance_correlation": geo.get("geodesic_distance_correlation"),
                }
            # Evaluate all targets/decoders for this (F, E, W)
            target_list = list(CONTINUOUS_TARGETS) + list(CATEGORICAL_TARGETS)
            for target in target_list:
                if embedding_needs_targets(embedding_type):
                    emb = make_feature_transformer(
                        embedding_type,
                        decode_window=decode_window,
                        n_components=n_components or 3,
                        units_df=data["units_df"],
                        unit_ids=data["unit_ids"],
                        random_state=config.seed,
                    )
                    if emb is None:
                        continue
                    y_emb = _get_y_for_embedding(beh_train, target)
                    try:
                        emb.fit(X_train_f, y_emb)
                    except Exception as exc:
                        print(f"  skip supervised E={embedding_type} target={target}: {exc}")
                        continue
                    X_train = emb.transform(X_train_f)
                    X_test = emb.transform(X_test_f)
                    meta = emb.get_metadata()
                    transform_name = manifold_transform_dirname(
                        f"{feature_mode}_{target}", decode_window, n_components, n_neighbors,
                    )
                    transform_path = manifold_dir / transform_name
                    emb.save(transform_path)
                    feature_row_extras = {
                        "manifold_type": meta.get("manifold_type")
                        or manifold_type_for_feature_mode(embedding_type),
                        "manifold_grouping": meta.get("manifold_grouping"),
                        "manifold_n_components": n_components,
                        "n_neighbors": n_neighbors,
                        "actual_n_features": meta.get("actual_n_features"),
                        "manifold_transform_path": str(transform_path),
                        "embedding_transform_path": str(transform_path),
                        "realtime_compatible": True,
                    }
                    e_transform = emb
                else:
                    X_train, X_test = X_train_shared, X_test_shared
                    feature_row_extras = shared_extras
                    e_transform = unsupervised_embed

                # Restrict Bayesian-place-tuning embedding to Bayesian / place-like targets.
                if embedding_type == "bayesian_place_tuning" and target not in (
                    "position", "distance_to_wall", "spatial_context", "wall_distance_bin",
                ):
                    continue

                model_names = (
                    continuous_model_names(config.max_models, target)
                    if target in CONTINUOUS_TARGETS
                    else categorical_model_names(config.max_models, target)
                )
                for decoder_name in model_names:
                    if embedding_type == "bayesian_place_tuning" and not is_bayesian_model(decoder_name):
                        continue
                    fit = _fit_and_evaluate(
                        X_train, X_test, beh_train, beh_test,
                        target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                    )
                    row = _base_row(
                        config, data, feature_type, decode_window, update_dt,
                        embedding_type=embedding_type,
                    )
                    row.update(feature_row_extras)
                    row.update({
                        "target_name": target,
                        "target_family": TARGET_FAMILY[target],
                        "decoder_name": decoder_name,
                        "decoder_nonlinear": is_nonlinear_decoder(decoder_name),
                        "n_train_samples": int(train_mask.sum()),
                        "n_test_samples": int(test_mask.sum()),
                        "primary_metric": PRIMARY_METRIC[target][0],
                        "run_id": config.run_id,
                    })
                    row.update(fit.metrics)
                    row, trows = enrich_fit_row(
                        row,
                        fit=fit,
                        feature_type=feature_type,
                        embedding_type=embedding_type,
                        feature_transform=f_transform,
                        embedding_transform=e_transform,
                        X_counts_test=X_counts_test,
                        X_test_embedded=X_test,
                        decode_window_s=decode_window,
                        update_dt_s=update_dt,
                        max_compute_ms=config.max_compute_ms,
                        max_effective_history_s=config.max_effective_history_s,
                        window_stats=w_stats,
                        trigger_rules=trigger_rules,
                        decode_times_test=decode_times_test,
                        behavior_test=beh_test,
                        arena_bounds=arena_bounds,
                    )
                    rows.append(row)
                    trigger_rows_all.extend(trows)
                    _maybe_update_best(best_fits, best_meta, target, row, fit, beh_test)
                    _maybe_update_best_by_window(
                        best_by_window, best_by_window_meta, target, row, fit,
                    )

        if config.include_controls and wi == 1:
            from realtime.negative_controls import evaluate_categorical_controls

            # Controls use raw train-fit counts features (identity embedding).
            ctrl_f = make_spike_feature_transformer(
                "counts", decode_window=decode_window,
                units_df=data["units_df"], unit_ids=data["unit_ids"],
            )
            ctrl_f.fit(X_counts[train_mask])
            Xtr = ctrl_f.transform(X_counts[train_mask])
            Xte = ctrl_f.transform(X_counts[test_mask])
            for target in CATEGORICAL_TARGETS:
                y_tr = _get_y(beh_train, target)
                y_te = _get_y(beh_test, target)
                for crow in evaluate_categorical_controls(
                    Xtr, Xte, y_tr, y_te, seed=config.seed,
                ):
                    control_rows_all.append({
                        **crow,
                        "spike_source": config.spike_source,
                        "target_name": target,
                        "feature_type": "counts",
                        "embedding_type": "identity",
                        "decode_window_s": decode_window,
                        "control_family": "negative_control",
                    })

        tested_windows.append(decode_window)
        if (
            config.adaptive_windows
            and not refined_once
            and wi >= len(window_queue)
        ):
            from realtime.adaptive_windows import propose_refined_windows

            best_ws = [
                float(meta["decode_window_s"])
                for meta in best_meta.values()
                if "decode_window_s" in meta
            ]
            extras = propose_refined_windows(tested_windows, best_ws)
            if extras:
                print(
                    "  adaptive refine: adding windows "
                    + ", ".join(f"{w:.3f}s" for w in extras)
                )
                window_queue.extend(float(w) for w in extras)
            refined_once = True

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "decoder_comparison_metrics.csv", index=False)
    with open(output_dir / "decoder_comparison_metrics.json", "w") as f:
        json.dump(_json_safe(rows), f, indent=2)

    explained_df = pd.DataFrame(explained_rows)
    if not explained_df.empty:
        explained_df.to_csv(output_dir / "manifold_explained_variance.csv", index=False)

    if trigger_rows_all:
        pd.DataFrame(trigger_rows_all).to_csv(
            output_dir / "closed_loop_trigger_comparison.csv", index=False,
        )

    if control_rows_all:
        pd.DataFrame(control_rows_all).to_csv(
            output_dir / "decoder_control_summary.csv", index=False,
        )

    scored_df = metrics_df
    if "target_name" in metrics_df.columns:
        scored_df = metrics_df[metrics_df["target_name"].notna()].copy()

    _save_windowed_models(best_by_window, best_by_window_meta, models_dir)
    best_df = _build_best_decoder_table(scored_df, best_fits, models_dir, config)
    from realtime.manifold_summaries import (
        build_best_manifold_decoder_table,
        build_manifold_vs_counts_summary,
        enrich_best_decoder_table,
        write_manifold_decoder_report,
    )
    best_df = enrich_best_decoder_table(best_df, scored_df)
    best_df.to_csv(output_dir / "best_decoder_by_target.csv", index=False)
    with open(output_dir / "best_decoder_by_target.json", "w") as f:
        json.dump(_json_safe(best_df.to_dict(orient="records")), f, indent=2)

    best_manifold = build_best_manifold_decoder_table(scored_df)
    if not best_manifold.empty:
        best_manifold.to_csv(output_dir / "best_manifold_decoder_by_target.csv", index=False)
    vs_counts = build_manifold_vs_counts_summary(scored_df)
    if not vs_counts.empty:
        vs_counts.to_csv(output_dir / "manifold_vs_counts_summary.csv", index=False)
    write_manifold_decoder_report(
        output_dir, scored_df, vs_counts, best_manifold, explained_df,
        units_df=data.get("units_df"),
    )

    if config.region_ablation:
        _run_unit_ablation(
            config, data, behavior_times, update_dt, arena_bounds,
            group_col="region", output_csv=output_dir / "region_ablation_metrics.csv",
        )
    if config.layer_ablation:
        _run_unit_ablation(
            config, data, behavior_times, update_dt, arena_bounds,
            group_col="layer", output_csv=output_dir / "layer_ablation_metrics.csv",
        )
    if config.population_ablation:
        _run_population_ablation(
            config, data, behavior_times, update_dt, arena_bounds,
            output_csv=output_dir / "population_ablation_summary.csv",
        )

    _save_best_models(best_fits, best_meta, models_dir)
    _save_best_predictions(best_fits, best_meta, examples_dir)

    # Lab-deployable selection for this single-run comparison.
    from realtime.lab_deployable import select_best_lab_deployable_decoders

    control_df = pd.DataFrame(control_rows_all) if control_rows_all else None
    select_best_lab_deployable_decoders(
        scored_df, control_df=control_df, output_dir=output_dir,
    )
    return metrics_df


def _base_row(
    config: ComparisonRunConfig,
    data: dict,
    feature_type: str,
    decode_window: float,
    update_dt: float | None = None,
    *,
    embedding_type: str = "identity",
) -> dict[str, Any]:
    return {
        "spike_source": config.spike_source,
        "source": config.spike_source,
        "feature_type": feature_type,
        "embedding_type": embedding_type,
        "feature_mode": compose_feature_mode(feature_type, embedding_type),
        "decode_window_s": decode_window,
        "update_dt_s": float(update_dt if update_dt is not None else config.update_dt),
        "n_units": len(data["unit_ids"]),
        "max_compute_ms": float(config.max_compute_ms),
        "max_effective_history_s": float(config.max_effective_history_s),
    }


def _run_unit_ablation(
    config: ComparisonRunConfig,
    data: dict,
    behavior_times: np.ndarray,
    update_dt: float,
    arena_bounds: tuple[float, float, float, float],
    *,
    group_col: str,
    output_csv: Path,
) -> None:
    """Decode using only units from each region/layer (+ all units control)."""
    units_df = data["units_df"]
    if group_col not in units_df.columns:
        print(f"  skip {group_col} ablation: column missing from units.csv")
        return

    unit_ids = np.asarray(data["unit_ids"], dtype=int)
    groups = ["__all__"] + sorted(units_df[group_col].astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []
    # Lightweight ablation: one window, counts features, quick models, key targets
    decode_window = float(sorted(config.decode_windows)[len(config.decode_windows) // 2])
    decode_times = make_decode_times(
        data["session_duration"], decode_window, update_dt,
        behavior_times=behavior_times if config.align_to_behavior else None,
    )
    aligned = align_extended_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"]
    )
    train_mask, test_mask = causal_train_test_split(decode_times, config.train_frac)
    beh_train = aligned.loc[train_mask].reset_index(drop=True)
    beh_test = aligned.loc[test_mask].reset_index(drop=True)

    for group in groups:
        if group == "__all__":
            keep_ids = unit_ids
            label = "all"
        else:
            keep_ids = units_df.loc[
                units_df[group_col].astype(str) == group, "unit_id"
            ].to_numpy(dtype=int)
            keep_ids = np.array([u for u in unit_ids if u in set(keep_ids)], dtype=int)
            label = group
        if len(keep_ids) == 0:
            continue
        X = build_causal_spike_matrix(
            data["spikes_df"], keep_ids, decode_times, decode_window,
        )
        X_train, X_test = X[train_mask], X[test_mask]
        for target in ("position", "spatial_context", "speed"):
            names = (
                continuous_model_names("quick", target)
                if target in CONTINUOUS_TARGETS
                else categorical_model_names("quick", target)
            )
            for decoder_name in names:
                fit = _fit_and_evaluate(
                    X_train, X_test, beh_train, beh_test,
                    target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                )
                metric_key = PRIMARY_METRIC[target][0]
                rows.append({
                    "spike_source": config.spike_source,
                    "ablation_group_column": group_col,
                    "ablation_group": label,
                    "n_units": int(len(keep_ids)),
                    "decode_window_s": decode_window,
                    "feature_type": "counts",
                    "target_name": target,
                    "decoder_name": decoder_name,
                    "primary_metric": metric_key,
                    "primary_metric_value": float(fit.metrics[metric_key]),
                    **{k: v for k, v in fit.metrics.items() if k != "confusion_matrix"},
                })
    if rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"  wrote {output_csv.name} ({len(rows)} rows)")


def _run_population_ablation(
    config: ComparisonRunConfig,
    data: dict,
    behavior_times: np.ndarray,
    update_dt: float,
    arena_bounds: tuple[float, float, float, float],
    *,
    output_csv: Path,
) -> None:
    """Region / layer / cell-type population ablations for key targets."""
    from realtime.population_ablation import expand_population_jobs, unit_ids_for_population

    units_df = data["units_df"]
    unit_ids = np.asarray(data["unit_ids"], dtype=int)
    jobs = expand_population_jobs(units_df)
    decode_window = float(sorted(config.decode_windows)[len(config.decode_windows) // 2])
    decode_times = make_decode_times(
        data["session_duration"], decode_window, update_dt,
        behavior_times=behavior_times if config.align_to_behavior else None,
    )
    aligned = align_extended_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"]
    )
    train_mask, test_mask = causal_train_test_split(decode_times, config.train_frac)
    beh_train = aligned.loc[train_mask].reset_index(drop=True)
    beh_test = aligned.loc[test_mask].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for job in jobs:
        keep_ids = unit_ids_for_population(
            units_df, unit_ids, job["population_mode"],
            layer=job.get("layer"), cell_type=job.get("cell_type"),
        )
        if len(keep_ids) == 0:
            continue
        X = build_causal_spike_matrix(
            data["spikes_df"], keep_ids, decode_times, decode_window,
        )
        X_train, X_test = X[train_mask], X[test_mask]
        for target in ("position", "spatial_context", "speed"):
            names = (
                continuous_model_names("quick", target)
                if target in CONTINUOUS_TARGETS
                else categorical_model_names("quick", target)
            )
            for decoder_name in names:
                fit = _fit_and_evaluate(
                    X_train, X_test, beh_train, beh_test,
                    target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                )
                metric_key = PRIMARY_METRIC[target][0]
                rows.append({
                    "spike_source": config.spike_source,
                    "population_mode": job["population_mode"],
                    "population_label": job["population_label"],
                    "n_units": int(len(keep_ids)),
                    "decode_window_s": decode_window,
                    "feature_type": "counts",
                    "embedding_type": "identity",
                    "target_name": target,
                    "decoder_name": decoder_name,
                    "primary_metric": metric_key,
                    "primary_metric_value": float(fit.metrics[metric_key]),
                    **{k: v for k, v in fit.metrics.items() if k != "confusion_matrix"},
                })
    if rows:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"  wrote {output_csv.name} ({len(rows)} rows)")


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


def _n_comp_key(n_components: Any) -> str:
    if n_components is None or (isinstance(n_components, float) and np.isnan(n_components)):
        return "na"
    return str(int(n_components))


def _window_model_key(
    target: str,
    decode_window: float,
    feature_type: str,
    n_components: Any = None,
) -> tuple[str, float, str, str]:
    return (target, float(decode_window), str(feature_type), _n_comp_key(n_components))


def windowed_model_path(
    models_dir: Path,
    target: str,
    decode_window: float,
    feature_type: str = "counts",
    n_components: Any = None,
) -> Path:
    """Path for the best model at a specific causal window / feature mode."""
    return (
        Path(models_dir)
        / "by_window"
        / str(feature_type)
        / f"k{_n_comp_key(n_components)}"
        / f"w{float(decode_window):.3f}s"
        / f"{target}.joblib"
    )


def _maybe_update_best_by_window(
    best_by_window: dict[tuple[str, float, str, str], FitResult],
    best_by_window_meta: dict[tuple[str, float, str, str], dict[str, Any]],
    target: str,
    row: dict[str, Any],
    fit: FitResult,
) -> None:
    metric_key, direction = PRIMARY_METRIC[target]
    value = float(row[metric_key])
    # Prefer composite feature_mode for path keys (backward compatible with E modes).
    mode_key = str(row.get("feature_mode") or row.get("feature_type"))
    key = _window_model_key(
        target,
        float(row["decode_window_s"]),
        mode_key,
        row.get("manifold_n_components"),
    )
    if key not in best_by_window or _is_better(
        value, direction, best_by_window_meta[key]["primary_metric_value"]
    ):
        best_by_window[key] = fit
        best_by_window_meta[key] = {
            **{k: v for k, v in row.items()},
            "primary_metric_value": value,
        }


def _save_windowed_models(
    best_by_window: dict[tuple[str, float, str, str], FitResult],
    best_by_window_meta: dict[tuple[str, float, str, str], dict[str, Any]],
    models_dir: Path,
) -> None:
    for key, fit in best_by_window.items():
        target, decode_window, feature_type, n_comp = key
        path = windowed_model_path(
            models_dir, target, decode_window, feature_type,
            None if n_comp == "na" else int(n_comp),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit.pipeline, path)
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(_json_safe(best_by_window_meta[key]), f, indent=2)


def _build_decoder_config(
    decoder_name: str,
    decode_window_s: float,
    feature_type: str,
    target_name: str,
    seed: int,
    n_jobs: int,
    *,
    manifold_n_components: Any = None,
    manifold_type: Any = None,
    manifold_grouping: Any = None,
    manifold_transform_path: Any = None,
) -> dict[str, Any]:
    params = default_model_params(decoder_name, seed=seed, n_jobs=n_jobs)
    config: dict[str, Any] = {
        "decoder_name": decoder_name,
        "decode_window_s": decode_window_s,
        "feature_type": feature_type,
        "target_name": target_name,
        "target_family": TARGET_FAMILY[target_name],
        "manifold_type": manifold_type,
        "manifold_grouping": manifold_grouping,
        "manifold_n_components": (
            None if manifold_n_components is None or (
                isinstance(manifold_n_components, float) and np.isnan(manifold_n_components)
            ) else int(manifold_n_components)
        ),
        "manifold_transform_path": manifold_transform_path,
        "model_params": params,
    }
    if is_bayesian_model(decoder_name):
        config["bayesian_params"] = {
            k: params[k] for k in ("n_bins", "smooth", "smooth_alpha") if k in params
        }
    else:
        config["bayesian_params"] = {}
    return config


def _best_row_at_window(
    target_df: pd.DataFrame,
    decode_window: float,
    feature_type: str,
    metric_key: str,
    direction: str,
    n_components: Any = None,
) -> pd.Series | None:
    mode_col = "feature_mode" if "feature_mode" in target_df.columns else "feature_type"
    mask = (
        np.isclose(target_df["decode_window_s"].astype(float), float(decode_window))
        & (target_df[mode_col].astype(str) == str(feature_type))
    )
    # counts / non-manifold rows store NaN for n_components; treat as "no filter".
    if (
        "manifold_n_components" in target_df.columns
        and n_components is not None
        and not pd.isna(n_components)
    ):
        mask = mask & (
            target_df["manifold_n_components"].fillna(-1).astype(float) == float(n_components)
        )
    window_df = target_df[mask]
    if window_df.empty:
        return None
    if direction == "lower":
        return window_df.loc[window_df[metric_key].idxmin()]
    return window_df.loc[window_df[metric_key].idxmax()]


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
        feature_type = str(best_row.get("feature_mode") or best_row["feature_type"])
        n_comp = best_row.get("manifold_n_components")

        # Prefer realtime-compatible feature modes for closed-loop recommendations.
        # Standard Isomap is offline_analysis_only.
        rt_pool = target_df
        if "realtime_compatible" in target_df.columns:
            rt_ok = target_df["realtime_compatible"].fillna(True).astype(bool)
            if rt_ok.any():
                rt_pool = target_df[rt_ok]
        emb_or_mode = str(best_row.get("embedding_type") or feature_type)
        if emb_or_mode in OFFLINE_ONLY_FEATURE_MODES and not rt_pool.empty:
            # Best offline result may be Isomap; recommend a realtime-capable alternative.
            if direction == "lower":
                rt_best_idx = rt_pool[metric_key].idxmin()
            else:
                rt_best_idx = rt_pool[metric_key].idxmax()
            rt_best_row = rt_pool.loc[rt_best_idx]
            rt_best_value = float(rt_best_row[metric_key])
            recommended_window = _recommended_window(
                rt_pool, metric_key, direction, rt_best_value,
            )
            rt_feature = str(rt_best_row.get("feature_mode") or rt_best_row["feature_type"])
            rt_n_comp = rt_best_row.get("manifold_n_components")
        else:
            recommended_window = _recommended_window(
                rt_pool if not rt_pool.empty else target_df,
                metric_key, direction, best_value,
            )
            rt_feature = feature_type
            rt_n_comp = n_comp

        model_path = models_dir / f"best_{target}_decoder.joblib"
        realtime_path = windowed_model_path(
            models_dir, target, recommended_window, rt_feature, rt_n_comp,
        )
        best_window_path = windowed_model_path(
            models_dir, target, float(best_row["decode_window_s"]), feature_type, n_comp,
        )

        rt_row = _best_row_at_window(
            rt_pool if not rt_pool.empty else target_df,
            recommended_window, rt_feature, metric_key, direction, rt_n_comp,
        )
        realtime_decoder_name = (
            str(rt_row["decoder_name"]) if rt_row is not None
            else str(best_row["decoder_name"])
        )
        if rt_row is not None:
            rt_n_comp = rt_row.get("manifold_n_components")
            rt_feature = str(rt_row.get("feature_mode") or rt_row["feature_type"])

        decoder_config = _build_decoder_config(
            str(best_row["decoder_name"]),
            float(best_row["decode_window_s"]),
            feature_type,
            target,
            config.seed,
            config.n_jobs,
            manifold_n_components=n_comp,
            manifold_type=best_row.get("manifold_type"),
            manifold_grouping=best_row.get("manifold_grouping"),
            manifold_transform_path=best_row.get("manifold_transform_path"),
        )
        realtime_decoder_config = _build_decoder_config(
            realtime_decoder_name,
            float(recommended_window),
            rt_feature,
            target,
            config.seed,
            config.n_jobs,
            manifold_n_components=rt_n_comp,
            manifold_type=(
                rt_row.get("manifold_type") if rt_row is not None
                else best_row.get("manifold_type")
            ),
            manifold_grouping=(
                rt_row.get("manifold_grouping") if rt_row is not None
                else best_row.get("manifold_grouping")
            ),
            manifold_transform_path=(
                rt_row.get("manifold_transform_path") if rt_row is not None
                else best_row.get("manifold_transform_path")
            ),
        )

        summary_rows.append({
            "target_name": target,
            "target_family": TARGET_FAMILY[target],
            "primary_metric": metric_key,
            "best_decoder_name": best_row["decoder_name"],
            "best_decode_window_s": float(best_row["decode_window_s"]),
            "recommended_realtime_window_s": recommended_window,
            "recommended_realtime_decoder_name": realtime_decoder_name,
            "best_feature_type": feature_type,
            "best_manifold_type": best_row.get("manifold_type"),
            "best_manifold_grouping": best_row.get("manifold_grouping"),
            "best_manifold_n_components": n_comp,
            "best_metric_value": best_value,
            "spike_source": best_row["spike_source"],
            "source": best_row.get("source", best_row["spike_source"]),
            "model_path": str(model_path) if target in best_fits else "",
            "best_window_model_path": str(best_window_path),
            "realtime_model_path": str(realtime_path),
            "manifold_transform_path": best_row.get("manifold_transform_path"),
            "decoder_config_json": json.dumps(decoder_config),
            "realtime_decoder_config_json": json.dumps(realtime_decoder_config),
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


def _prediction_window_columns(meta: dict[str, Any], time: np.ndarray) -> dict[str, Any]:
    """Causal window metadata shared by decoded-example tables."""
    decode_window = float(meta.get("decode_window_s", 0.25))
    update_dt = float(meta.get("update_dt_s", 0.05))
    t = np.asarray(time, dtype=float)
    return {
        "decode_time": t,
        "window_start": t - decode_window,
        "window_end": t,
        "decode_window_s": decode_window,
        "update_dt_s": update_dt,
        "n_spikes_in_window": meta.get("n_spikes_in_window"),
        "n_active_units_in_window": meta.get("n_active_units_in_window"),
    }


def _save_best_predictions(
    best_fits: dict[str, FitResult],
    best_meta: dict[str, dict[str, Any]],
    examples_dir: Path,
) -> None:
    for target, fit in best_fits.items():
        beh = best_meta[target]["behavior_test"]
        time = beh["time"].to_numpy()
        window_cols = _prediction_window_columns(best_meta[target], time)

        if target == "position":
            pred = np.asarray(fit.y_pred)
            pd.DataFrame({
                "time": time,
                **window_cols,
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
                **window_cols,
                "true": true_angles,
                "pred": pred_angles,
            }).to_csv(examples_dir / "best_head_direction_predictions.csv", index=False)
            continue

        if target in CATEGORICAL_TARGETS:
            pd.DataFrame({
                "time": time,
                **window_cols,
                "true": fit.y_true,
                "pred": fit.y_pred,
            }).to_csv(examples_dir / f"best_{target}_predictions.csv", index=False)
            continue

        pd.DataFrame({
            "time": time,
            **window_cols,
            "true": np.asarray(fit.y_true).ravel(),
            "pred": np.asarray(fit.y_pred).ravel(),
        }).to_csv(examples_dir / f"best_{target}_predictions.csv", index=False)


def run_compare_sources(
    input_dir: Path,
    output_dir: Path,
    decode_windows: tuple[float, ...] = DEFAULT_DECODE_WINDOWS,
    update_dt: float = 0.050,
    train_frac: float = 0.70,
    feature_modes: tuple[str, ...] = QUICK_FEATURE_MODES,
    feature_types: tuple[str, ...] | None = None,
    embedding_types: tuple[str, ...] | None = None,
    manifold_n_components: tuple[int, ...] = DEFAULT_MANIFOLD_N_COMPONENTS,
    isomap_n_neighbors: tuple[int, ...] = (DEFAULT_ISOMAP_N_NEIGHBORS,),
    max_models: str = "quick",
    n_jobs: int = -1,
    seed: int = 42,
    region_ablation: bool = False,
    layer_ablation: bool = False,
    population_ablation: bool = False,
    adaptive_windows: bool = False,
    include_controls: bool = False,
    max_compute_ms: float = 25.0,
    max_effective_history_s: float = 0.500,
    **config_extras: Any,
) -> pd.DataFrame:
    """Run decoder comparison for ground_truth and sorted spikes."""
    output_dir = Path(output_dir)
    best_tables = []
    metric_frames: list[pd.DataFrame] = []

    for source in ("ground_truth", "sorted"):
        run_decoder_comparison(ComparisonRunConfig(
            input_dir=input_dir,
            output_dir=output_dir / source,
            spike_source=source,
            decode_windows=decode_windows,
            update_dt=update_dt,
            train_frac=train_frac,
            feature_modes=feature_modes,
            feature_types=feature_types,
            embedding_types=embedding_types,
            use_fe_grid=bool(feature_types is not None or embedding_types is not None),
            manifold_n_components=manifold_n_components,
            isomap_n_neighbors=isomap_n_neighbors,
            max_models=max_models,
            n_jobs=n_jobs,
            seed=seed,
            region_ablation=region_ablation,
            layer_ablation=layer_ablation,
            population_ablation=population_ablation,
            adaptive_windows=adaptive_windows,
            include_controls=include_controls,
            max_compute_ms=max_compute_ms,
            max_effective_history_s=max_effective_history_s,
            **config_extras,
        ))
        best_path = output_dir / source / "best_decoder_by_target.csv"
        if best_path.exists():
            bt = pd.read_csv(best_path)
            bt["source"] = source
            best_tables.append(bt)
        mpath = output_dir / source / "decoder_comparison_metrics.csv"
        if mpath.exists():
            metric_frames.append(pd.read_csv(mpath))

    if metric_frames:
        combined_metrics = pd.concat(metric_frames, ignore_index=True)
        combined_metrics.to_csv(output_dir / "decoder_comparison_metrics.csv", index=False)
        from realtime.sorting_robustness import build_sorted_information_loss_summary

        loss = build_sorted_information_loss_summary(combined_metrics)
        if not loss.empty:
            loss.to_csv(output_dir / "sorted_information_loss_summary.csv", index=False)

    if not best_tables:
        return pd.DataFrame()

    comparison_df = pd.concat(best_tables, ignore_index=True)
    comparison_df.to_csv(output_dir / "source_comparison_metrics.csv", index=False)
    # Also write combined best table at root for convenience
    comparison_df.to_csv(output_dir / "best_decoder_by_target.csv", index=False)
    with open(output_dir / "best_decoder_by_target.json", "w") as f:
        json.dump(_json_safe(comparison_df.to_dict(orient="records")), f, indent=2)
    return comparison_df
