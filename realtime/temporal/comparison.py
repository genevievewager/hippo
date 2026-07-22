"""Phase-1 temporal manifold decoding: joint W × L comparison with controls.

Architecture:
    x_t^(W) -> z_t -> [z_{t-L+1}, ..., z_t] -> y_hat_t

Model classes
-------------
A. raw_static           : f(x_t^W)
B. static_latent        : D(z_t)  (L=1)
C. long_aggregate       : f(x_t^{W_long}) with L=1 (no latent history)
D. flattened_history    : D(flatten([z_{t-L+1},...,z_t]))
F. shuffled_sequence    : same states, permuted order
G. averaged_history     : D(mean_k z_{t-k})
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from realtime.data_loading import load_simulation_data
from realtime.decoder_comparison import (
    ALL_TARGETS,
    CATEGORICAL_TARGETS,
    PRIMARY_METRIC,
    _classification_metrics,
    _continuous_metrics,
    _fit_estimator,
    _get_y,
    _predict_labels,
)
from realtime.decoder_models import (
    is_nonlinear_decoder,
    make_categorical_pipeline,
    make_continuous_pipeline,
)
from realtime.decoding_targets import align_extended_behavior_to_decoder_times
from realtime.manifolds import is_realtime_compatible_manifold, make_manifold_encoder
from realtime.manifolds.isomap_diagnostics import DisconnectedGraphError
from realtime.spike_features import apply_feature_mode, build_causal_spike_matrix
from realtime.temporal.flattened import (
    average_latent_history,
    flatten_latent_sequences,
    shuffle_sequence_order,
)
from realtime.temporal.sequences import (
    apply_prediction_lag,
    build_causal_latent_sequences,
    lag_seconds_to_frames,
)
from realtime.temporal.splits import (
    causal_train_val_test_split,
    mask_sequences_within_split,
    required_split_gap_s,
)
from realtime.timing import (
    DEFAULT_INTEGRATION_WINDOWS_S,
    DEFAULT_PREDICTION_LAGS_S,
    assert_alignment,
    extract_behavior_times,
    make_behavior_aligned_decode_times,
    resolve_update_dt_s,
)
from realtime.train_decoder import infer_arena_bounds

# Lean Phase-1 default; --profile full still passes the dense L grid.
DEFAULT_LATENT_HISTORY_FRAMES = (1, 5, 20)

DEFAULT_DECODERS_CONTINUOUS = ("ridge", "random_forest_regressor")
DEFAULT_DECODERS_CATEGORICAL = ("logistic_regression", "random_forest_classifier")
REALTIME_BUDGET_S = 0.050


@dataclass
class TemporalComparisonConfig:
    input_dir: Path
    output_dir: Path
    spike_source: str = "sorted"
    integration_windows_s: tuple[float, ...] = DEFAULT_INTEGRATION_WINDOWS_S
    latent_history_frames: tuple[int, ...] = DEFAULT_LATENT_HISTORY_FRAMES
    prediction_lags_s: tuple[float, ...] = (0.0,)
    representations: tuple[str, ...] = ("pca",)
    pca_latent_dim: int = 16
    isomap_latent_dim: int = 8
    isomap_n_neighbors: int = 10
    isomap_pre_pca_enabled: bool = True
    isomap_pre_pca_n_components: int = 50
    temporal_models: tuple[str, ...] = (
        "raw_static",
        "static_latent",
        "flattened_history",
    )
    include_long_aggregate_control: bool = True
    targets: tuple[str, ...] = ALL_TARGETS
    train_frac: float = 0.60
    val_frac: float = 0.15
    seed: int = 42
    n_jobs: int = -1
    max_models: str = "quick"
    behavior_sampling_rate_hz: float = 20.0
    alignment_tolerance_s: float = 0.005
    feature_type: str = "counts"


def _decoder_names_for_target(target: str, max_models: str) -> tuple[str, ...]:
    if target in CATEGORICAL_TARGETS:
        return DEFAULT_DECODERS_CATEGORICAL if max_models == "quick" else DEFAULT_DECODERS_CATEGORICAL
    return DEFAULT_DECODERS_CONTINUOUS


def _primary_value(metrics: dict[str, Any], target: str) -> float:
    key, _ = PRIMARY_METRIC[target]
    return float(metrics[key])


def _fit_predict_eval(
    features_train: np.ndarray,
    features_eval: np.ndarray,
    y_train: np.ndarray,
    y_eval: np.ndarray,
    beh_train: pd.DataFrame,
    beh_eval: pd.DataFrame,
    target: str,
    decoder_name: str,
    seed: int,
    n_jobs: int,
    arena_bounds: tuple[float, float, float, float],
) -> tuple[dict[str, Any], float, Any]:
    t0 = time.perf_counter()
    if target in CATEGORICAL_TARGETS:
        pipeline = make_categorical_pipeline(
            decoder_name, seed=seed, n_jobs=n_jobs,
            target_name=target, arena_bounds=arena_bounds,
        )
    else:
        pipeline = make_continuous_pipeline(
            decoder_name, target, seed=seed, n_jobs=n_jobs, arena_bounds=arena_bounds,
        )
    pipeline = _fit_estimator(pipeline, features_train, y_train, beh_train, decoder_name)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = np.asarray(pipeline.predict(features_eval))
    infer_s = time.perf_counter() - t1
    per_update = infer_s / max(len(features_eval), 1)

    if target in CATEGORICAL_TARGETS:
        labels = _predict_labels(pipeline) or sorted(set(map(str, y_eval)) | set(map(str, y_pred)))
        metrics = _classification_metrics(y_eval, y_pred, labels)
    else:
        if target == "distance_to_wall" and y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        metrics = _continuous_metrics(target, y_eval, y_pred, beh_eval)
    metrics["mean_inference_latency_s"] = float(per_update)
    metrics["fit_time_s"] = float(fit_s)
    metrics["realtime_budget_ok"] = bool(per_update <= REALTIME_BUDGET_S)
    return metrics, per_update, pipeline


def run_temporal_manifold_comparison(config: TemporalComparisonConfig) -> pd.DataFrame:
    """Run Phase-1 joint integration-window / latent-history comparison."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    manifolds_dir = output_dir / "manifolds"
    features_dir = output_dir / "features"
    for d in (models_dir, manifolds_dir, features_dir):
        d.mkdir(parents=True, exist_ok=True)

    data = load_simulation_data(config.input_dir, config.spike_source)
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"],
        behavior_sampling_rate_hz=config.behavior_sampling_rate_hz,
        derive_from_behavior=True,
        behavior_times=behavior_times,
    )
    arena_bounds = infer_arena_bounds(data["behavior_df"], data["summary"])

    max_W = max(config.integration_windows_s)
    max_L_s = max(config.latent_history_frames) * update_dt
    max_tau = max(config.prediction_lags_s) if config.prediction_lags_s else 0.0
    gap_s = required_split_gap_s(max_W, max_L_s, max_tau)

    # Align once at the largest W so all windows share a common timestamp grid
    # subset; per-window we further require t >= W.
    decode_times_full, timing_validation = make_behavior_aligned_decode_times(
        behavior_times,
        integration_window_s=max_W,
        summary=data["summary"],
        expected_dt_s=update_dt,
        alignment_tolerance_s=config.alignment_tolerance_s,
        strict=False,
    )
    max_align_err = assert_alignment(
        decode_times_full, behavior_times,
        alignment_tolerance_s=config.alignment_tolerance_s,
    )
    with open(output_dir / "timing_metadata.json", "w") as f:
        json.dump({
            "update_dt_s": update_dt,
            "behavior_sampling_rate_hz": 1.0 / update_dt,
            "max_alignment_error_s": max_align_err,
            "split_gap_s": gap_s,
            "timing_validation": timing_validation.__dict__,
            "integration_windows_s": list(config.integration_windows_s),
            "latent_history_frames": list(config.latent_history_frames),
            "prediction_lags_s": list(config.prediction_lags_s),
            "representations": list(config.representations),
        }, f, indent=2)

    aligned = align_extended_behavior_to_decoder_times(
        data["behavior_df"], decode_times_full, data["summary"]
    )
    train_mask, val_mask, test_mask = causal_train_val_test_split(
        decode_times_full,
        train_frac=config.train_frac,
        val_frac=config.val_frac,
        gap_s=gap_s,
    )

    # Cache spike-count matrices by W (same decode_times_full; early times with
    # t < W are zero-padded in the matrix but masked out per window).
    X_by_window: dict[float, np.ndarray] = {}
    for W in config.integration_windows_s:
        X = build_causal_spike_matrix(
            data["spikes_df"], data["unit_ids"], decode_times_full, W,
        )
        X = apply_feature_mode(X, config.feature_type, W)
        X_by_window[float(W)] = X
        np.savez_compressed(
            features_dir / f"window_{int(round(W * 1000)):04d}ms.npz",
            X=X, decode_times=decode_times_full,
        )

    rows: list[dict[str, Any]] = []
    best_by_target: dict[str, dict[str, Any]] = {}
    rng = np.random.default_rng(config.seed)

    # Fit manifolds on train split for each (W, representation)
    encoders: dict[tuple[float, str], Any] = {}
    Z_by_key: dict[tuple[float, str], np.ndarray] = {}
    for W, X in X_by_window.items():
        # Valid times for this W: t >= W (all decode_times_full already >= max_W,
        # so if W <= max_W all are valid).
        for rep in config.representations:
            kwargs: dict[str, Any] = {}
            if rep == "pca":
                kwargs["n_components"] = config.pca_latent_dim
            elif rep == "isomap":
                kwargs.update({
                    "n_components": config.isomap_latent_dim,
                    "n_neighbors": config.isomap_n_neighbors,
                    "pre_pca_enabled": config.isomap_pre_pca_enabled,
                    "pre_pca_n_components": config.isomap_pre_pca_n_components,
                    "n_jobs": config.n_jobs,
                    "random_state": config.seed,
                })
            enc = make_manifold_encoder(rep, **kwargs)
            try:
                enc.fit(X[train_mask])
            except DisconnectedGraphError as exc:
                print(f"  exclude isomap W={W:.3f}s: {exc}")
                continue
            Z = enc.transform(X)
            encoders[(W, rep)] = enc
            Z_by_key[(W, rep)] = Z
            enc_dir = manifolds_dir / rep / f"window_{int(round(W * 1000)):04d}ms"
            enc.save(enc_dir)

    long_W = max(config.integration_windows_s)

    for target in config.targets:
        if target not in ALL_TARGETS:
            continue
        metric_key, direction = PRIMARY_METRIC[target]
        decoders = _decoder_names_for_target(target, config.max_models)

        for W in config.integration_windows_s:
            X = X_by_window[float(W)]
            for rep in config.representations:
                if (float(W), rep) not in Z_by_key:
                    continue
                Z = Z_by_key[(float(W), rep)]
                enc = encoders.get((float(W), rep))
                isomap_meta: dict[str, Any] = {}
                if rep == "isomap" and enc is not None and hasattr(enc, "get_metadata"):
                    isomap_meta = enc.get_metadata()
                geo = isomap_meta.get("geometry_metrics") or {}
                diag = isomap_meta.get("graph_diagnostics") or {}
                for L in config.latent_history_frames:
                    for tau in config.prediction_lags_s:
                        lag_frames = lag_seconds_to_frames(tau, update_dt)
                        sequences, end_idx = build_causal_latent_sequences(Z, L, pad_mode="drop")
                        if len(end_idx) == 0:
                            continue
                        end_idx, target_idx = apply_prediction_lag(
                            end_idx, lag_frames, len(decode_times_full),
                        )
                        if len(end_idx) == 0:
                            continue
                        sequences = np.stack(
                            [Z[int(t) - L + 1 : int(t) + 1] for t in end_idx],
                            axis=0,
                        )

                        train_keep = mask_sequences_within_split(end_idx, L, train_mask)
                        val_keep = mask_sequences_within_split(end_idx, L, val_mask)
                        test_keep = mask_sequences_within_split(end_idx, L, test_mask)
                        if not train_keep.any() or not val_keep.any():
                            continue

                        y_all = _get_y(aligned, target)
                        beh_all = aligned

                        model_specs: list[tuple[str, np.ndarray]] = []
                        # B / D family from latent sequences
                        if "static_latent" in config.temporal_models and L == 1:
                            model_specs.append(("static_latent", flatten_latent_sequences(sequences)))
                        if "flattened_history" in config.temporal_models and L >= 1:
                            model_specs.append(
                                ("flattened_history", flatten_latent_sequences(sequences))
                            )
                        if "shuffled_sequence" in config.temporal_models and L > 1:
                            shuffled = shuffle_sequence_order(sequences, rng=rng)
                            model_specs.append(
                                ("shuffled_sequence", flatten_latent_sequences(shuffled))
                            )
                        if "averaged_history" in config.temporal_models and L >= 1:
                            model_specs.append(
                                ("averaged_history", average_latent_history(sequences))
                            )

                        for temporal_model, feats in model_specs:
                            # Avoid duplicate static_latent vs flattened when L==1
                            if temporal_model == "flattened_history" and L == 1:
                                if "static_latent" in config.temporal_models:
                                    continue
                            for decoder_name in decoders:
                                print(
                                    f"    {target} W={W:.3f}s L={L} "
                                    f"rep={rep} model={temporal_model} "
                                    f"decoder={decoder_name}"
                                )
                                metrics_val, lat_val, pipeline = _fit_predict_eval(
                                    feats[train_keep],
                                    feats[val_keep],
                                    y_all[target_idx[train_keep]],
                                    y_all[target_idx[val_keep]],
                                    beh_all.iloc[target_idx[train_keep]].reset_index(drop=True),
                                    beh_all.iloc[target_idx[val_keep]].reset_index(drop=True),
                                    target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                                )
                                metrics_test = {}
                                if test_keep.any():
                                    y_pred_test = np.asarray(pipeline.predict(feats[test_keep]))
                                    if target in CATEGORICAL_TARGETS:
                                        labels = _predict_labels(pipeline) or sorted(
                                            set(map(str, y_all[target_idx[test_keep]]))
                                        )
                                        metrics_test = _classification_metrics(
                                            y_all[target_idx[test_keep]], y_pred_test, labels,
                                        )
                                    else:
                                        if target == "distance_to_wall" and y_pred_test.ndim > 1:
                                            y_pred_test = y_pred_test.ravel()
                                        metrics_test = _continuous_metrics(
                                            target,
                                            y_all[target_idx[test_keep]],
                                            y_pred_test,
                                            beh_all.iloc[target_idx[test_keep]].reset_index(drop=True),
                                        )

                                row = {
                                    "target": target,
                                    "representation": rep,
                                    "integration_window_s": float(W),
                                    "latent_history_frames": int(L),
                                    "latent_history_s": float(L) * update_dt,
                                    "prediction_lag_s": float(tau),
                                    "update_dt_s": update_dt,
                                    "temporal_model": temporal_model,
                                    "decoder": decoder_name,
                                    "decoder_nonlinear": is_nonlinear_decoder(decoder_name),
                                    "spike_source": config.spike_source,
                                    "model_class": temporal_model,
                                    "n_train": int(train_keep.sum()),
                                    "n_val": int(val_keep.sum()),
                                    "n_test": int(test_keep.sum()),
                                    "validation_metric": metric_key,
                                    "validation_metric_value": _primary_value(metrics_val, target),
                                    "validation_realtime_budget_ok": metrics_val.get(
                                        "realtime_budget_ok", True
                                    ),
                                    "mean_inference_latency_s": metrics_val.get(
                                        "mean_inference_latency_s", float("nan")
                                    ),
                                    "realtime_compatible": is_realtime_compatible_manifold(rep),
                                    "n_neighbors": (
                                        config.isomap_n_neighbors if rep == "isomap" else None
                                    ),
                                    "latent_dim": (
                                        config.isomap_latent_dim if rep == "isomap"
                                        else (config.pca_latent_dim if rep == "pca" else None)
                                    ),
                                    "graph_connected": diag.get("graph_connected"),
                                    "largest_component_fraction": diag.get(
                                        "largest_component_fraction"
                                    ),
                                    "trustworthiness": geo.get("trustworthiness"),
                                    "residual_variance": geo.get("residual_variance"),
                                }
                                for k, v in metrics_val.items():
                                    if k == "confusion_matrix":
                                        continue
                                    row[f"val_{k}"] = v
                                for k, v in metrics_test.items():
                                    if k == "confusion_matrix":
                                        continue
                                    row[f"test_{k}"] = v
                                rows.append(row)

                                cur_best = best_by_target.get(target)
                                value = row["validation_metric_value"]
                                better = (
                                    cur_best is None
                                    or (direction == "lower" and value < cur_best["validation_metric_value"])
                                    or (direction == "higher" and value > cur_best["validation_metric_value"])
                                )
                                # Prefer realtime-capable when tied-ish; still select on val only
                                if better:
                                    best_by_target[target] = {
                                        **{k: row[k] for k in (
                                            "target", "representation", "integration_window_s",
                                            "latent_history_frames", "latent_history_s",
                                            "prediction_lag_s", "update_dt_s", "temporal_model",
                                            "decoder", "validation_metric", "validation_metric_value",
                                            "mean_inference_latency_s",
                                        )},
                                        "model_path": str(
                                            models_dir / f"best_{target}_{temporal_model}.joblib"
                                        ),
                                    }
                                    joblib.dump(pipeline, models_dir / f"best_{target}_{temporal_model}.joblib")

            # A. raw static baseline at this W (no manifold / L=1)
            if "raw_static" in config.temporal_models:
                valid = decode_times_full >= (W - 1e-12)
                for decoder_name in decoders:
                    metrics_val, _, pipeline = _fit_predict_eval(
                        X[train_mask & valid],
                        X[val_mask & valid],
                        _get_y(aligned.loc[train_mask & valid].reset_index(drop=True), target),
                        _get_y(aligned.loc[val_mask & valid].reset_index(drop=True), target),
                        aligned.loc[train_mask & valid].reset_index(drop=True),
                        aligned.loc[val_mask & valid].reset_index(drop=True),
                        target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                    )
                    row = {
                        "target": target,
                        "representation": "none",
                        "integration_window_s": float(W),
                        "latent_history_frames": 1,
                        "latent_history_s": update_dt,
                        "prediction_lag_s": 0.0,
                        "update_dt_s": update_dt,
                        "temporal_model": "raw_static",
                        "decoder": decoder_name,
                        "spike_source": config.spike_source,
                        "model_class": "raw_static",
                        "n_train": int((train_mask & valid).sum()),
                        "n_val": int((val_mask & valid).sum()),
                        "n_test": int((test_mask & valid).sum()),
                        "validation_metric": metric_key,
                        "validation_metric_value": _primary_value(metrics_val, target),
                        "validation_realtime_budget_ok": metrics_val.get("realtime_budget_ok", True),
                        "mean_inference_latency_s": metrics_val.get("mean_inference_latency_s", float("nan")),
                    }
                    for k, v in metrics_val.items():
                        if k != "confusion_matrix":
                            row[f"val_{k}"] = v
                    rows.append(row)
                    value = row["validation_metric_value"]
                    cur_best = best_by_target.get(target)
                    better = (
                        cur_best is None
                        or (direction == "lower" and value < cur_best["validation_metric_value"])
                        or (direction == "higher" and value > cur_best["validation_metric_value"])
                    )
                    if better:
                        best_by_target[target] = {
                            **{k: row[k] for k in (
                                "target", "representation", "integration_window_s",
                                "latent_history_frames", "latent_history_s",
                                "prediction_lag_s", "update_dt_s", "temporal_model",
                                "decoder", "validation_metric", "validation_metric_value",
                                "mean_inference_latency_s",
                            )},
                            "model_path": str(models_dir / f"best_{target}_raw_static.joblib"),
                        }
                        joblib.dump(pipeline, models_dir / f"best_{target}_raw_static.joblib")

        # C. long aggregate window control (largest W, L=1, raw static)
        if config.include_long_aggregate_control and "raw_static" in config.temporal_models:
            X_long = X_by_window[float(long_W)]
            for decoder_name in decoders:
                metrics_val, _, pipeline = _fit_predict_eval(
                    X_long[train_mask],
                    X_long[val_mask],
                    _get_y(aligned.loc[train_mask].reset_index(drop=True), target),
                    _get_y(aligned.loc[val_mask].reset_index(drop=True), target),
                    aligned.loc[train_mask].reset_index(drop=True),
                    aligned.loc[val_mask].reset_index(drop=True),
                    target, decoder_name, config.seed, config.n_jobs, arena_bounds,
                )
                row = {
                    "target": target,
                    "representation": "none",
                    "integration_window_s": float(long_W),
                    "latent_history_frames": 1,
                    "latent_history_s": update_dt,
                    "prediction_lag_s": 0.0,
                    "update_dt_s": update_dt,
                    "temporal_model": "long_aggregate",
                    "decoder": decoder_name,
                    "spike_source": config.spike_source,
                    "model_class": "long_aggregate",
                    "validation_metric": metric_key,
                    "validation_metric_value": _primary_value(metrics_val, target),
                    "mean_inference_latency_s": metrics_val.get("mean_inference_latency_s", float("nan")),
                }
                for k, v in metrics_val.items():
                    if k != "confusion_matrix":
                        row[f"val_{k}"] = v
                rows.append(row)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(output_dir / "all_configurations.csv", index=False)
    with open(output_dir / "best_by_target.json", "w") as f:
        json.dump(best_by_target, f, indent=2)
    latency_df = metrics_df[[
        c for c in (
            "target", "temporal_model", "representation", "integration_window_s",
            "latent_history_frames", "decoder", "mean_inference_latency_s",
            "validation_realtime_budget_ok",
        ) if c in metrics_df.columns
    ]].copy()
    latency_df.to_csv(output_dir / "latency_results.csv", index=False)
    return metrics_df
