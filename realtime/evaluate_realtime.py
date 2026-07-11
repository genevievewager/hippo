"""Closed-loop replay: train, causally decode, and evaluate one decoder setup."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score

from realtime.closed_loop_controller import evaluate_closed_loop
from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.realtime_decoder import RealTimeDecoder
from realtime.spike_binner import build_causal_spike_matrix
from realtime.train_decoder import (
    align_behavior_to_decoder_times,
    causal_train_test_split,
    evaluate_offline_training,
    train_decoders,
)


@dataclass
class PipelineResult:
    metrics: dict
    decoded_df: pd.DataFrame
    closed_loop_df: pd.DataFrame
    offline_metrics: dict
    spike_source: str


def run_realtime_pipeline(
    input_dir: Path,
    output_dir: Path,
    spike_source: str = "sorted",
    update_dt: float = 0.025,
    decode_window: float = 0.250,
    train_frac: float = 0.70,
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_movement: str | None = None,
) -> PipelineResult:
    """Train decoders, replay causally on test period, evaluate, and save outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"

    data = load_simulation_data(input_dir, spike_source)
    decode_times = make_decode_times(data["session_duration"], decode_window, update_dt)
    aligned_all = align_behavior_to_decoder_times(
        data["behavior_df"], decode_times, data["summary"]
    )

    X_all = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, decode_window
    )

    train_mask, test_mask = causal_train_test_split(decode_times, train_frac)
    X_train = X_all[train_mask]
    X_test = X_all[test_mask]
    beh_train = aligned_all.loc[train_mask].reset_index(drop=True)
    beh_test = aligned_all.loc[test_mask].reset_index(drop=True)

    decoders = train_decoders(X_train, beh_train, models_dir=models_dir)
    offline_metrics = evaluate_offline_training(decoders, X_test, beh_test)

    rt_decoder = RealTimeDecoder(
        models=decoders,
        unit_ids=data["unit_ids"],
        decode_window=decode_window,
        update_dt=update_dt,
    )
    decoded_df = rt_decoder.replay(data["spikes_df"], beh_test)

    closed_loop_df = evaluate_closed_loop(
        decoded_df,
        trigger_context=trigger_context,
        trigger_confidence=trigger_confidence,
        trigger_movement=trigger_movement,
    )

    metrics = compute_realtime_metrics(
        decoded_df=decoded_df,
        closed_loop_df=closed_loop_df,
        update_dt=update_dt,
        decode_window=decode_window,
        spike_source=spike_source,
        n_units=len(data["unit_ids"]),
    )

    decoded_df.to_csv(output_dir / "decoded_realtime.csv", index=False)
    closed_loop_df.to_csv(output_dir / "closed_loop_events.csv", index=False)
    with open(output_dir / "realtime_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(output_dir / "offline_test_metrics.json", "w") as f:
        json.dump(offline_metrics, f, indent=2)

    return PipelineResult(
        metrics=metrics,
        decoded_df=decoded_df,
        closed_loop_df=closed_loop_df,
        offline_metrics=offline_metrics,
        spike_source=spike_source,
    )


def compute_realtime_metrics(
    decoded_df: pd.DataFrame,
    closed_loop_df: pd.DataFrame,
    update_dt: float,
    decode_window: float,
    spike_source: str,
    n_units: int,
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
    }


def run_compare_sources(
    input_dir: Path,
    output_dir: Path,
    update_dt: float = 0.025,
    decode_window: float = 0.250,
    train_frac: float = 0.70,
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_movement: str | None = None,
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
            trigger_context=trigger_context,
            trigger_confidence=trigger_confidence,
            trigger_movement=trigger_movement,
        )

    comparison_df = pd.DataFrame(
        [{**results[source].metrics, "source": source} for source in ("ground_truth", "sorted")]
    )
    comparison_df.to_csv(output_dir / "source_comparison_metrics.csv", index=False)
    return comparison_df
