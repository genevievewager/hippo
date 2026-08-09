"""Closed-loop replay: train, causally decode, and evaluate one decoder setup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from realtime.best_decoder_selection import (
    copy_loaded_models_to_output,
    load_pretrained_suite,
    load_windowed_model,
    select_best_decoder_row,
)
from realtime.closed_loop_controller import evaluate_closed_loop
from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.decoder_models import (
    is_bayesian_model,
    make_categorical_pipeline,
    make_continuous_pipeline,
)
from realtime.decoding_targets import align_extended_behavior_to_decoder_times
from realtime.realtime_decoder import RealTimeDecoder
from realtime.spike_features import build_causal_spike_matrix
from realtime.train_decoder import (
    TrainedDecoders,
    align_behavior_to_decoder_times,
    causal_train_test_split,
    evaluate_offline_training,
    infer_arena_bounds,
    train_decoders,
)


@dataclass
class PipelineResult:
    metrics: dict
    decoded_df: pd.DataFrame
    closed_loop_df: pd.DataFrame
    offline_metrics: dict
    spike_source: str
    selected_config: dict | None = None


def _train_primary_model(
    decoder_name: str,
    target: str,
    X_train: np.ndarray,
    beh_train: pd.DataFrame,
    arena_bounds: tuple[float, float, float, float],
    seed: int = 42,
    n_jobs: int = -1,
) -> Any:
    from realtime.bayesian_decoder import (
        BayesianDistanceToWallDecoder,
        BayesianPlaceDerivedDecoder,
    )

    if target in ("spatial_context", "movement_state", "wall_distance_bin"):
        model = make_categorical_pipeline(
            decoder_name, seed=seed, n_jobs=n_jobs,
            target_name=target, arena_bounds=arena_bounds,
        )
    else:
        model = make_continuous_pipeline(
            decoder_name, target, seed=seed, n_jobs=n_jobs, arena_bounds=arena_bounds,
        )

    if is_bayesian_model(decoder_name):
        position_xy = beh_train[["x", "y"]].to_numpy()
        if isinstance(model, (BayesianPlaceDerivedDecoder, BayesianDistanceToWallDecoder)):
            model.fit(X_train, position_xy=position_xy)
        else:
            model.fit(X_train, position_xy)
    else:
        if target == "position":
            y = beh_train[["x", "y"]].to_numpy()
        elif target == "head_direction":
            y = beh_train[["head_direction_sin", "head_direction_cos"]].to_numpy()
        else:
            y = beh_train[target].to_numpy()
        model.fit(X_train, y)
    return model


def run_realtime_pipeline(
    input_dir: Path,
    output_dir: Path,
    spike_source: str = "sorted",
    update_dt: float = 0.050,
    decode_window: float = 0.250,
    train_frac: float = 0.70,
    closed_loop_target: str = "position",
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_movement: str | None = None,
    trigger_wall_bin: str | None = "near_wall",
    trigger_distance_lt_cm: float | None = 10.0,
    trigger_speed_gt_cm_s: float | None = 10.0,
    trigger_zone: str | None = "wall",
    trigger_hd_center_deg: float | None = 90.0,
    trigger_hd_width_deg: float | None = 30.0,
    decoder_name: str | None = None,
    feature_type: str = "counts",
    selected_config: dict | None = None,
    pretrained_decoders: TrainedDecoders | None = None,
    primary_model: Any | None = None,
    align_to_behavior: bool = True,
    feature_transformer: Any | None = None,
    manifold_n_components: int | None = None,
) -> PipelineResult:
    """Replay causally on the test period, evaluate, and save outputs.

    When ``pretrained_decoders`` (and optional ``primary_model``) are provided,
    models are reused as-is and not retrained. Manifold transforms must be fit
    only on training data (or loaded from a transform fit on train).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data = load_simulation_data(input_dir, spike_source)
    arena_bounds = infer_arena_bounds(data["behavior_df"], data["summary"])
    from realtime.timing import extract_behavior_times, resolve_update_dt_s
    from realtime.manifold_features import make_feature_transformer

    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = resolve_update_dt_s(
        data["summary"],
        derive_from_behavior=align_to_behavior,
        update_dt_s=update_dt,
        behavior_times=behavior_times,
    )
    decode_times = make_decode_times(
        data["session_duration"],
        decode_window,
        update_dt,
        behavior_times=behavior_times if align_to_behavior else None,
    )

    # Extended labels for richer closed-loop targets; fall back for basic columns.
    aligned_ext = align_extended_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"]
    )
    aligned_all = align_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"]
    )
    for col in ("acceleration", "distance_to_wall", "wall_distance_bin",
                "head_direction", "head_direction_sin", "head_direction_cos"):
        if col in aligned_ext.columns and col not in aligned_all.columns:
            aligned_all[col] = aligned_ext[col]
    if "wall_distance_bin" in aligned_ext.columns:
        aligned_all["true_wall_distance_bin"] = aligned_ext["wall_distance_bin"]

    X_counts = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, decode_window
    )

    train_mask, test_mask = causal_train_test_split(decode_times, train_frac)
    beh_train = aligned_all.loc[train_mask].reset_index(drop=True)
    beh_test = aligned_all.loc[test_mask].reset_index(drop=True)

    if feature_transformer is None:
        from realtime.manifold_features import (
            OFFLINE_ONLY_FEATURE_MODES,
            is_realtime_compatible_feature_mode,
        )
        from realtime.search_space import resolve_manifold_alias

        feature_type_resolved = resolve_manifold_alias(feature_type)
        if feature_type_resolved in OFFLINE_ONLY_FEATURE_MODES or not is_realtime_compatible_feature_mode(
            feature_type_resolved
        ):
            raise ValueError(
                f"Representation {feature_type!r} is offline-only / acausal and "
                "cannot be launched for realtime replay. Use global_lds, "
                "global_pca, counts, or another realtime-compatible method."
            )
        feature_transformer = make_feature_transformer(
            feature_type_resolved,
            decode_window=decode_window,
            n_components=manifold_n_components or 3,
            units_df=data["units_df"],
            unit_ids=data["unit_ids"],
            update_dt=update_dt,
            spike_source=spike_source,
        )
        if feature_transformer is None:
            raise ValueError(f"Could not build feature transformer for {feature_type}")
        feature_transformer.fit(X_counts[train_mask])
        feature_transformer.save(models_dir / "feature_transformer")

    from realtime.search_space import is_dynamic_embedding
    from realtime.search_space import resolve_manifold_alias as _resolve_alias

    emb_name = _resolve_alias(feature_type)
    if is_dynamic_embedding(emb_name) and hasattr(feature_transformer, "transform"):
        X_train = feature_transformer.transform(X_counts[train_mask], causal=True, reset=True)
        X_test = feature_transformer.transform(X_counts[test_mask], causal=True, reset=False)
    else:
        X_train = feature_transformer.transform(X_counts[train_mask])
        X_test = feature_transformer.transform(X_counts[test_mask])

    if pretrained_decoders is not None:
        decoders = pretrained_decoders
        print("  reusing pretrained comparison models (no retrain)")
    else:
        decoders = train_decoders(X_train, beh_train, models_dir=models_dir)

    offline_metrics = evaluate_offline_training(decoders, X_test, beh_test)

    if primary_model is None and decoder_name is not None and closed_loop_target is not None:
        primary_model = _train_primary_model(
            decoder_name, closed_loop_target, X_train, beh_train, arena_bounds,
        )
        joblib.dump(primary_model, models_dir / f"primary_{closed_loop_target}_decoder.joblib")
    elif primary_model is not None:
        joblib.dump(primary_model, models_dir / f"primary_{closed_loop_target}_decoder.joblib")

    rt_decoder = RealTimeDecoder(
        models=decoders,
        unit_ids=data["unit_ids"],
        decode_window=decode_window,
        update_dt=update_dt,
        feature_type=feature_type,
        primary_model=primary_model,
        primary_target=closed_loop_target if primary_model is not None else None,
        feature_transformer=feature_transformer,
    )
    replay_out = rt_decoder.replay(
        data["spikes_df"], beh_test, profile_latency=True,
    )
    decoded_df, latency_samples = replay_out

    import time as _time

    from realtime.latency_profiler import save_latency_artifacts

    # Time closed-loop policy separately (vectorized over decoded frames).
    t_cl0 = _time.perf_counter()
    closed_loop_df = evaluate_closed_loop(
        decoded_df,
        closed_loop_target=closed_loop_target,
        trigger_context=trigger_context,
        trigger_confidence=trigger_confidence,
        trigger_movement=trigger_movement,
        trigger_wall_bin=trigger_wall_bin,
        trigger_distance_lt_cm=trigger_distance_lt_cm,
        trigger_speed_gt_cm_s=trigger_speed_gt_cm_s,
        trigger_zone=trigger_zone,
        trigger_hd_center_deg=trigger_hd_center_deg,
        trigger_hd_width_deg=trigger_hd_width_deg,
        arena_bounds=arena_bounds,
    )
    closed_loop_total_ms = (_time.perf_counter() - t_cl0) * 1000.0
    per_frame_cl_ms = closed_loop_total_ms / max(len(decoded_df), 1)
    for sample in latency_samples:
        sample.stages_ms["closed_loop_policy"] = per_frame_cl_ms
        # Recompute total including closed-loop share
        sample.stages_ms["total_update"] = (
            sample.stages_ms.get("total_update", 0.0) + per_frame_cl_ms
        )

    latency_summary = save_latency_artifacts(
        latency_samples,
        output_dir / "latency",
        update_budget_ms=float(update_dt) * 1000.0,
        extra_meta={
            "spike_source": spike_source,
            "feature_type": feature_type,
            "decode_window_s": float(decode_window),
            "update_dt_s": float(update_dt),
            "closed_loop_target": closed_loop_target,
            "decoder_name": decoder_name,
            "closed_loop_total_ms": closed_loop_total_ms,
            "closed_loop_per_frame_ms": per_frame_cl_ms,
        },
    )

    metrics = compute_realtime_metrics(
        decoded_df=decoded_df,
        closed_loop_df=closed_loop_df,
        update_dt=update_dt,
        decode_window=decode_window,
        spike_source=spike_source,
        n_units=len(data["unit_ids"]),
        decoder_name=decoder_name,
        closed_loop_target=closed_loop_target,
        feature_type=feature_type,
    )
    metrics["models_reused_from_comparison"] = pretrained_decoders is not None
    metrics["latency"] = {
        "mean_total_ms": latency_summary.get("mean_total_ms"),
        "median_total_ms": latency_summary.get("median_total_ms"),
        "p95_total_ms": latency_summary.get("p95_total_ms"),
        "within_budget_frac": latency_summary.get("within_budget_frac"),
        "update_budget_ms": latency_summary.get("update_budget_ms"),
        "n_updates": latency_summary.get("n_updates"),
    }

    decoded_df.to_csv(output_dir / "decoded_realtime.csv", index=False)
    closed_loop_df.to_csv(output_dir / "closed_loop_events.csv", index=False)
    with open(output_dir / "realtime_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "offline_test_metrics.json", "w") as f:
        json.dump(offline_metrics, f, indent=2)

    if selected_config is not None:
        with open(output_dir / "selected_realtime_decoder_config.json", "w") as f:
            json.dump(selected_config, f, indent=2)

    return PipelineResult(
        metrics=metrics,
        decoded_df=decoded_df,
        closed_loop_df=closed_loop_df,
        offline_metrics=offline_metrics,
        spike_source=spike_source,
        selected_config=selected_config,
    )


def compute_realtime_metrics(
    decoded_df: pd.DataFrame,
    closed_loop_df: pd.DataFrame,
    update_dt: float,
    decode_window: float,
    spike_source: str,
    n_units: int,
    decoder_name: str | None = None,
    closed_loop_target: str | None = None,
    feature_type: str = "counts",
) -> dict:
    """Aggregate real-time decoding and closed-loop metrics."""
    pos_err = decoded_df["position_error_cm"].to_numpy()
    context_acc = float(accuracy_score(
        decoded_df["true_spatial_context"],
        decoded_df["decoded_spatial_context"],
    ))
    movement_acc = float(accuracy_score(
        decoded_df["true_movement_state"],
        decoded_df["decoded_movement_state"],
    ))

    y_speed = decoded_df["true_speed"].to_numpy()
    pred_speed = decoded_df["decoded_speed"].to_numpy()
    speed_r2 = float(r2_score(y_speed, pred_speed))
    speed_corr = float(np.corrcoef(y_speed, pred_speed)[0, 1])

    n_events = len(closed_loop_df)
    if n_events > 0:
        precision = float(closed_loop_df["correct_trigger"].mean())
        false_positive_rate = float(1.0 - precision)
    else:
        precision = float("nan")
        false_positive_rate = float("nan")

    return {
        "mean_position_error_cm": float(np.mean(pos_err)),
        "median_position_error_cm": float(np.median(pos_err)),
        "spatial_context_accuracy": context_acc,
        "movement_state_accuracy": movement_acc,
        "speed_r2": speed_r2,
        "speed_correlation": speed_corr,
        "n_decoder_updates": int(len(decoded_df)),
        "n_closed_loop_events": int(n_events),
        "closed_loop_precision": precision,
        "closed_loop_false_positive_rate": false_positive_rate,
        "update_dt": update_dt,
        "decode_window": decode_window,
        "spike_source": spike_source,
        "n_units": n_units,
        "decoder_name": decoder_name,
        "closed_loop_target": closed_loop_target,
        "feature_type": feature_type,
    }


def run_realtime_with_best_decoder(
    input_dir: Path,
    output_dir: Path,
    comparison_dir: Path,
    closed_loop_target: str,
    spike_source: str = "sorted",
    selection_policy: str = "shortest_near_optimal",
    update_dt: float = 0.050,
    train_frac: float = 0.70,
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_movement: str | None = None,
    trigger_wall_bin: str | None = "near_wall",
    trigger_distance_lt_cm: float | None = 10.0,
    trigger_speed_gt_cm_s: float | None = 10.0,
    trigger_zone: str | None = "wall",
    trigger_hd_center_deg: float | None = 90.0,
    trigger_hd_width_deg: float | None = 30.0,
    experiment_dir: Path | None = None,
) -> PipelineResult:
    """Select best decoder/window from comparison results and run closed-loop replay.

    Prefer ``models/best_realtime_decoders.json`` (sorted / deployable only) when
    present under ``experiment_dir``. Never loads ground-truth models for
    deployment. Reuses models already fit during comparison at the selected
    causal window (no retrain) when windowed artifacts are available.
    """
    if spike_source != "sorted":
        raise ValueError(
            f"Realtime deployment requires spike_source='sorted' "
            f"(got {spike_source!r}). Ground-truth models are oracle-only."
        )

    decode_window = None
    feature_type = None
    decoder_name = None
    n_comp = None
    transform_path = None
    model_path = None
    decoder_cfg: dict = {}
    from_file = None
    registry = None

    exp = Path(experiment_dir) if experiment_dir is not None else Path(comparison_dir).parent
    try:
        from realtime.deployment_selection import load_best_realtime_decoders
        registry = load_best_realtime_decoders(exp)
        tgt = registry["targets"].get(closed_loop_target)
        if tgt is None:
            raise KeyError(
                f"Target {closed_loop_target!r} missing from best_realtime_decoders.json"
            )
        decode_window = float(tgt["selected_causal_window_s"])
        feature_type = str(tgt["selected_feature_mode"])
        decoder_name = str(tgt["selected_decoder"])
        n_comp = tgt.get("manifold_n_components")
        transform_path = tgt.get("manifold_transform_path")
        model_path = tgt.get("model_artifact_path")
        decoder_cfg = tgt.get("decoder_config") or {}
        from_file = exp / "models" / "best_realtime_decoders.json"
        print(
            f"  loaded deployable registry: {closed_loop_target} → "
            f"{decoder_name} @ W={decode_window:.3f}s feature={feature_type}"
        )
    except FileNotFoundError:
        registry = None

    if registry is None:
        selected, from_file = select_best_decoder_row(
            comparison_dir=comparison_dir,
            spike_source=spike_source,
            closed_loop_target=closed_loop_target,
            selection_policy=selection_policy,
        )
        decode_window = float(selected["selected_decode_window_s"])
        feature_type = str(selected["selected_feature_type"])
        decoder_name = str(selected["selected_decoder_name"])
        models_dir = Path(selected["comparison_models_dir"])
        n_comp = selected.get("selected_manifold_n_components")
        transform_path = selected.get("selected_manifold_transform_path")
        model_path = selected.get("selected_model_path")
        decoder_cfg = selected.get("decoder_config") or {}
    else:
        models_dir = Path(comparison_dir) / spike_source / "models"
        if not models_dir.exists():
            models_dir = Path(comparison_dir) / "models"

    run_dir = (
        Path(output_dir)
        / spike_source
        / f"{closed_loop_target}_{selection_policy}"
    )

    if n_comp is not None and not (isinstance(n_comp, float) and np.isnan(n_comp)):
        n_comp = int(n_comp)
    else:
        n_comp = None

    selected_config = {
        "selection_mode": "use_best_decoder",
        "comparison_dir": str(comparison_dir),
        "closed_loop_target": closed_loop_target,
        "spike_source": spike_source,
        "selection_policy": selection_policy,
        "selected_decoder_name": decoder_name,
        "selected_decode_window_s": decode_window,
        "feature_type": feature_type,
        "selected_feature_type": feature_type,
        "manifold_type": decoder_cfg.get("manifold_type"),
        "manifold_grouping": decoder_cfg.get("manifold_grouping"),
        "manifold_n_components": n_comp,
        "manifold_transform_path": transform_path,
        "decoder_model_path": model_path,
        "selected_model_path": model_path,
        "from_file": str(from_file) if from_file else None,
        "decoder_config": decoder_cfg,
        "models_reused_from_comparison": False,
        "deployable": True,
        "update_dt_s": float(update_dt),
        "update_rate_hz": float(1.0 / update_dt),
    }

    from realtime.manifold_features import load_feature_transformer

    pretrained = None
    primary = None
    feature_transformer = None
    if transform_path and Path(transform_path).exists():
        feature_transformer = load_feature_transformer(Path(transform_path))
        print(f"  loaded manifold transform from {transform_path}")

    try:
        pretrained = load_pretrained_suite(
            models_dir, decode_window, feature_type, n_comp,
        )
        primary = load_windowed_model(
            models_dir, closed_loop_target, decode_window, feature_type, n_comp,
        )
        copy_loaded_models_to_output(
            models_dir,
            run_dir / "models",
            decode_window,
            feature_type,
            closed_loop_target,
            primary_model_path=Path(model_path) if model_path else None,
            n_components=n_comp,
            manifold_transform_path=(
                Path(transform_path) if transform_path else None
            ),
        )
        selected_config["models_reused_from_comparison"] = True
        print(
            f"  loaded comparison models @ window={decode_window}s "
            f"feature={feature_type} k={n_comp} ({spike_source})"
        )
    except FileNotFoundError as exc:
        print(
            f"  warning: comparison windowed models unavailable ({exc}); "
            "falling back to retrain"
        )

    return run_realtime_pipeline(
        input_dir=input_dir,
        output_dir=run_dir,
        spike_source=spike_source,
        update_dt=update_dt,
        decode_window=decode_window,
        train_frac=train_frac,
        closed_loop_target=closed_loop_target,
        trigger_context=trigger_context,
        trigger_confidence=trigger_confidence,
        trigger_movement=trigger_movement,
        trigger_wall_bin=trigger_wall_bin,
        trigger_distance_lt_cm=trigger_distance_lt_cm,
        trigger_speed_gt_cm_s=trigger_speed_gt_cm_s,
        trigger_zone=trigger_zone,
        trigger_hd_center_deg=trigger_hd_center_deg,
        trigger_hd_width_deg=trigger_hd_width_deg,
        decoder_name=str(decoder_name),
        feature_type=feature_type,
        selected_config=selected_config,
        pretrained_decoders=pretrained,
        primary_model=primary,
        feature_transformer=feature_transformer,
        manifold_n_components=n_comp,
    )


def run_compare_sources(
    input_dir: Path,
    output_dir: Path,
    update_dt: float = 0.050,
    decode_window: float = 0.250,
    train_frac: float = 0.70,
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_movement: str | None = None,
    closed_loop_target: str = "position",
) -> pd.DataFrame:
    """Run decoding for ground-truth and sorted spikes and save comparison metrics."""
    output_dir = Path(output_dir)
    results: dict[str, PipelineResult] = {}

    for source in ("ground_truth", "sorted"):
        source_out = output_dir / source
        results[source] = run_realtime_pipeline(
            input_dir=input_dir,
            output_dir=source_out,
            spike_source=source,
            update_dt=update_dt,
            decode_window=decode_window,
            train_frac=train_frac,
            closed_loop_target=closed_loop_target,
            trigger_context=trigger_context,
            trigger_confidence=trigger_confidence,
            trigger_movement=trigger_movement,
        )

    comparison_df = pd.DataFrame(
        [{**results[source].metrics, "source": source} for source in ("ground_truth", "sorted")]
    )
    comparison_df.to_csv(output_dir / "source_comparison_metrics.csv", index=False)
    return comparison_df
