"""Load simulation outputs, evaluate real-time decoding, and generate plots.

Comparing spikes_ground_truth.csv to spikes_sorted.csv quantifies information
loss from Neuropixels recording degradation and spike sorting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

from realtime.closed_loop_controller import evaluate_closed_loop
from realtime.realtime_decoder import RealTimeDecoder
from realtime.spike_binner import build_causal_spike_matrix, _resolve_spike_columns
from realtime.train_decoder import (
    TrainedDecoders,
    align_behavior_to_decoder_times,
    causal_train_test_split,
    evaluate_offline_training,
    train_decoders,
)


def load_simulation_data(input_dir: Path, spike_source: str) -> dict:
    """Load behavior, units, spikes, and summary from a simulation output directory."""
    input_dir = Path(input_dir)
    required = ["behavior.csv", "units.csv", "summary.json"]
    for fname in required:
        if not (input_dir / fname).exists():
            raise FileNotFoundError(f"Required file not found: {input_dir / fname}")

    if spike_source == "sorted":
        spike_file = input_dir / "spikes_sorted.csv"
    elif spike_source == "ground_truth":
        spike_file = input_dir / "spikes_ground_truth.csv"
    else:
        raise ValueError(
            f"spike_source must be 'sorted' or 'ground_truth', got {spike_source!r}"
        )
    if not spike_file.exists():
        raise FileNotFoundError(f"Spike file not found: {spike_file}")

    behavior_df = pd.read_csv(input_dir / "behavior.csv")
    units_df = pd.read_csv(input_dir / "units.csv")
    spikes_df = pd.read_csv(spike_file)
    with open(input_dir / "summary.json") as f:
        summary = json.load(f)

    time_col, _ = _resolve_spike_columns(spikes_df)
    spikes_df = spikes_df.rename(columns={time_col: "time"})

    session_duration = summary.get("session_duration_s")
    if session_duration is None:
        session_duration = float(max(
            behavior_df.iloc[:, 0].max(),
            spikes_df["time"].max(),
        ))

    unit_ids = sorted(units_df["unit_id"].unique().tolist())

    return {
        "behavior_df": behavior_df,
        "units_df": units_df,
        "spikes_df": spikes_df,
        "summary": summary,
        "session_duration": float(session_duration),
        "unit_ids": unit_ids,
        "spike_source": spike_source,
    }


def make_decode_times(
    session_duration: float,
    decode_window: float,
    update_dt: float,
) -> np.ndarray:
    """Decoder times from decode_window to session end at update_dt steps."""
    if update_dt <= 0 or decode_window <= 0:
        raise ValueError("update_dt and decode_window must be positive")
    t_start = decode_window
    t_end = session_duration
    if t_start >= t_end:
        raise ValueError(
            f"decode_window ({decode_window}) must be less than session duration ({t_end})"
        )
    return np.arange(t_start, t_end + 1e-9, update_dt)


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
) -> dict:
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

    generate_plots(
        decoded_df=decoded_df,
        closed_loop_df=closed_loop_df,
        offline_metrics=offline_metrics,
        output_dir=output_dir,
    )

    return metrics


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


def generate_plots(
    decoded_df: pd.DataFrame,
    closed_loop_df: pd.DataFrame,
    offline_metrics: dict,
    output_dir: Path,
) -> None:
    """Save evaluation plots to output_dir."""
    output_dir = Path(output_dir)

    # Position error over time
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(decoded_df["time"], decoded_df["position_error_cm"], linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title("Real-time position decoding error")
    fig.tight_layout()
    fig.savefig(output_dir / "position_decoding_error_over_time.png", dpi=150)
    plt.close(fig)

    # True vs decoded position
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(decoded_df["true_x"], decoded_df["true_y"], s=4, alpha=0.3, label="True")
    ax.scatter(decoded_df["decoded_x"], decoded_df["decoded_y"], s=4, alpha=0.3, label="Decoded")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title("True vs decoded position")
    ax.legend(markerscale=3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "true_vs_decoded_position.png", dpi=150)
    plt.close(fig)

    # Spatial context confusion matrix
    _plot_confusion_matrix(
        decoded_df["true_spatial_context"],
        decoded_df["decoded_spatial_context"],
        offline_metrics.get("spatial_context_labels"),
        "Spatial context confusion matrix",
        output_dir / "spatial_context_confusion_matrix.png",
    )

    # Movement state confusion matrix
    _plot_confusion_matrix(
        decoded_df["true_movement_state"],
        decoded_df["decoded_movement_state"],
        offline_metrics.get("movement_state_labels"),
        "Movement state confusion matrix",
        output_dir / "movement_state_confusion_matrix.png",
    )

    # Closed-loop events over time
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(
        decoded_df["time"],
        decoded_df["position_error_cm"],
        s=2,
        alpha=0.2,
        color="gray",
        label="Position error",
    )
    if not closed_loop_df.empty:
        colors = closed_loop_df["correct_trigger"].map({True: "green", False: "red"})
        ax.scatter(
            closed_loop_df["time"],
            np.full(len(closed_loop_df), decoded_df["position_error_cm"].median()),
            c=colors,
            s=30,
            marker="|",
            label="Closed-loop trigger",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title("Closed-loop events over time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "closed_loop_events_over_time.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: list[str] | None,
    title: str,
    out_path: Path,
) -> None:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Decoded")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


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
    """Run decoding for ground-truth and sorted spikes and compare metrics."""
    output_dir = Path(output_dir)
    results = []

    for source in ("ground_truth", "sorted"):
        source_out = output_dir / source
        metrics = run_realtime_pipeline(
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
        metrics["source"] = source
        results.append(metrics)

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(output_dir / "source_comparison_metrics.csv", index=False)

    _plot_source_comparison(comparison_df, output_dir)
    return comparison_df


def _plot_source_comparison(comparison_df: pd.DataFrame, output_dir: Path) -> None:
    sources = comparison_df["source"].tolist()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sources, comparison_df["mean_position_error_cm"])
    ax.set_ylabel("Mean position error (cm)")
    ax.set_title("Ground truth vs sorted: position error")
    fig.tight_layout()
    fig.savefig(output_dir / "ground_truth_vs_sorted_position_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sources, comparison_df["spatial_context_accuracy"])
    ax.set_ylabel("Spatial context accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Ground truth vs sorted: context accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "ground_truth_vs_sorted_context_accuracy.png", dpi=150)
    plt.close(fig)
