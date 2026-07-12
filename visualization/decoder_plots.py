"""All decoder-related figures: realtime decoding and decoder comparison.

Figures are written under the experiment's shared figures/ folder so they
sit alongside simulation visualizations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from realtime.constants import SOURCE_LABELS

FIGURE_DPI = 150

ALL_TARGETS = (
    "position", "speed", "acceleration", "head_direction", "distance_to_wall",
    "spatial_context", "movement_state", "wall_distance_bin",
)
CATEGORICAL_TARGETS = ("spatial_context", "movement_state", "wall_distance_bin")

PLOT_METRIC = {
    "position": "mean_position_error_cm",
    "speed": "r2",
    "acceleration": "r2",
    "head_direction": "mean_circular_error_deg",
    "distance_to_wall": "r2",
    "spatial_context": "balanced_accuracy",
    "movement_state": "balanced_accuracy",
    "wall_distance_bin": "balanced_accuracy",
}

PLOT_FILENAME = {
    "position": "position_error_vs_window.png",
    "speed": "speed_r2_vs_window.png",
    "acceleration": "acceleration_r2_vs_window.png",
    "head_direction": "head_direction_error_vs_window.png",
    "distance_to_wall": "distance_to_wall_r2_vs_window.png",
    "spatial_context": "spatial_context_accuracy_vs_window.png",
    "movement_state": "movement_state_accuracy_vs_window.png",
    "wall_distance_bin": "wall_distance_bin_accuracy_vs_window.png",
}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV, returning an empty DataFrame if the file has no rows/columns."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def resolve_experiment_dir(
    experiment_dir: Path | None = None,
    realtime_dir: Path | None = None,
    comparison_dir: Path | None = None,
) -> Path:
    """Resolve the experiment root that owns the shared figures/ folder."""
    if experiment_dir is not None:
        return Path(experiment_dir)
    for candidate in (realtime_dir, comparison_dir):
        if candidate is not None:
            return Path(candidate).resolve().parent
    raise ValueError("Provide --experiment, --realtime-dir, or --comparison-dir")


def plot_realtime_outputs(
    realtime_dir: Path,
    figures_dir: Path,
) -> Path:
    """Generate realtime decoding figures under figures_dir/realtime_decoding/."""
    realtime_dir = Path(realtime_dir)
    out_root = Path(figures_dir) / "realtime_decoding"
    out_root.mkdir(parents=True, exist_ok=True)

    gt = realtime_dir / "ground_truth" / "decoded_realtime.csv"
    sorted_csv = realtime_dir / "sorted" / "decoded_realtime.csv"
    wrote_any = False

    if gt.exists() and sorted_csv.exists():
        _plot_realtime_compare_sources(realtime_dir, out_root / "comparison")
        wrote_any = True

    for source in ("sorted", "ground_truth"):
        source_dir = realtime_dir / source
        if (source_dir / "decoded_realtime.csv").exists():
            _plot_realtime_source(source_dir, out_root / source, source)
            wrote_any = True

    if not wrote_any:
        raise FileNotFoundError(
            f"No realtime decoding outputs found under {realtime_dir}"
        )
    return out_root


def plot_decoder_comparison_outputs(
    comparison_dir: Path,
    figures_dir: Path,
) -> Path:
    """Generate decoder comparison figures under figures_dir/decoder_comparison/."""
    comparison_dir = Path(comparison_dir)
    out_root = Path(figures_dir) / "decoder_comparison"
    out_root.mkdir(parents=True, exist_ok=True)

    gt_metrics = comparison_dir / "ground_truth" / "decoder_comparison_metrics.csv"
    sorted_metrics = comparison_dir / "sorted" / "decoder_comparison_metrics.csv"
    wrote_any = False

    if gt_metrics.exists() and sorted_metrics.exists():
        _plot_decoder_comparison_source_summary(comparison_dir, out_root)
        for source in ("ground_truth", "sorted"):
            _plot_decoder_comparison_single(
                comparison_dir / source,
                out_root / source,
            )
        wrote_any = True
    elif (comparison_dir / "decoder_comparison_metrics.csv").exists():
        _plot_decoder_comparison_single(comparison_dir, out_root)
        wrote_any = True

    if not wrote_any:
        raise FileNotFoundError(
            f"No decoder comparison outputs found under {comparison_dir}"
        )
    return out_root


def _plot_realtime_source(source_dir: Path, output_dir: Path, spike_source: str) -> None:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decoded_df = _read_csv(source_dir / "decoded_realtime.csv")
    if decoded_df.empty:
        raise FileNotFoundError(f"Missing or empty decoded_realtime.csv in {source_dir}")

    closed_loop_df = _read_csv(source_dir / "closed_loop_events.csv")
    offline_path = source_dir / "offline_test_metrics.json"
    offline_metrics = json.loads(offline_path.read_text()) if offline_path.exists() else {}

    source_label = SOURCE_LABELS.get(spike_source, spike_source)
    title_suffix = f" ({source_label})"

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(decoded_df["time"], decoded_df["position_error_cm"], linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title(f"Real-time position decoding error{title_suffix}")
    fig.tight_layout()
    fig.savefig(output_dir / "position_decoding_error_over_time.png", dpi=FIGURE_DPI)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    time = decoded_df["time"].to_numpy()
    true_scatter = ax.scatter(
        decoded_df["true_x"], decoded_df["true_y"], c=time, s=4, alpha=0.6,
        cmap="viridis", label="True",
    )
    ax.scatter(
        decoded_df["decoded_x"], decoded_df["decoded_y"], c=time, s=4, alpha=0.6,
        cmap="viridis", marker="x", label="Decoded",
    )
    fig.colorbar(true_scatter, ax=ax, label="Time (s)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_title(f"True vs decoded position colored by time{title_suffix}")
    ax.legend(markerscale=3)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(output_dir / "true_vs_decoded_position_time_colored.png", dpi=FIGURE_DPI)
    plt.close(fig)

    _save_confusion_matrix_plot(
        decoded_df["true_spatial_context"],
        decoded_df["decoded_spatial_context"],
        offline_metrics.get("spatial_context_labels"),
        f"Spatial context confusion matrix{title_suffix}",
        output_dir / "spatial_context_confusion_matrix.png",
    )
    _save_confusion_matrix_plot(
        decoded_df["true_movement_state"],
        decoded_df["decoded_movement_state"],
        offline_metrics.get("movement_state_labels"),
        f"Movement state confusion matrix{title_suffix}",
        output_dir / "movement_state_confusion_matrix.png",
    )

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.scatter(
        decoded_df["time"], decoded_df["position_error_cm"],
        s=2, alpha=0.2, color="gray", label="Position error",
    )
    if not closed_loop_df.empty and "correct_trigger" in closed_loop_df.columns:
        colors = closed_loop_df["correct_trigger"].map({True: "green", False: "red"})
        ax.scatter(
            closed_loop_df["time"],
            np.full(len(closed_loop_df), decoded_df["position_error_cm"].median()),
            c=colors, s=30, marker="|", label="Closed-loop trigger",
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (cm)")
    ax.set_title(f"Closed-loop events over time{title_suffix}")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_dir / "closed_loop_events_over_time.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_realtime_compare_sources(realtime_dir: Path, output_dir: Path) -> None:
    realtime_dir = Path(realtime_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for source in ("ground_truth", "sorted"):
        source_dir = realtime_dir / source
        results[source] = {
            "decoded_df": _read_csv(source_dir / "decoded_realtime.csv"),
            "closed_loop_df": _read_csv(source_dir / "closed_loop_events.csv"),
            "offline_metrics": json.loads(
                (source_dir / "offline_test_metrics.json").read_text()
            ),
            "metrics": json.loads((source_dir / "realtime_metrics.json").read_text()),
            "spike_source": source,
        }
        if results[source]["decoded_df"].empty:
            raise FileNotFoundError(f"Missing decoded_realtime.csv in {source_dir}")

    ground_truth = results["ground_truth"]
    sorted_result = results["sorted"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for ax, result in zip(axes, (ground_truth, sorted_result), strict=True):
        df = result["decoded_df"]
        ax.plot(df["time"], df["position_error_cm"], linewidth=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position error (cm)")
        ax.set_title(SOURCE_LABELS[result["spike_source"]])
    fig.suptitle("Real-time position decoding error", y=1.02)
    fig.tight_layout()
    fig.savefig(
        output_dir / "position_decoding_error_over_time.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    scatter = None
    for ax, result in zip(axes, (ground_truth, sorted_result), strict=True):
        decoded_df = result["decoded_df"]
        time = decoded_df["time"].to_numpy()
        scatter = ax.scatter(
            decoded_df["true_x"], decoded_df["true_y"], c=time, s=4, alpha=0.6,
            cmap="viridis", label="True",
        )
        ax.scatter(
            decoded_df["decoded_x"], decoded_df["decoded_y"], c=time, s=4, alpha=0.6,
            cmap="viridis", marker="x", label="Decoded",
        )
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(SOURCE_LABELS[result["spike_source"]])
        ax.legend(markerscale=3, loc="upper right")
        ax.set_aspect("equal", adjustable="box")
    if scatter is not None:
        cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
        cbar.set_label("Time (s)")
    fig.suptitle("True vs decoded position colored by time", y=1.02)
    fig.savefig(
        output_dir / "true_vs_decoded_position.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    for task, title, filename in (
        ("spatial_context", "Spatial context confusion matrix", "spatial_context_confusion_matrix.png"),
        ("movement_state", "Movement state confusion matrix", "movement_state_confusion_matrix.png"),
    ):
        _plot_realtime_side_by_side_confusion(
            ground_truth, sorted_result, task, title, output_dir / filename,
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5), sharey=True)
    for ax, result in zip(axes, (ground_truth, sorted_result), strict=True):
        decoded_df = result["decoded_df"]
        closed_loop_df = result["closed_loop_df"]
        ax.scatter(
            decoded_df["time"], decoded_df["position_error_cm"],
            s=2, alpha=0.2, color="gray", label="Position error",
        )
        if not closed_loop_df.empty and "correct_trigger" in closed_loop_df.columns:
            colors = closed_loop_df["correct_trigger"].map({True: "green", False: "red"})
            ax.scatter(
                closed_loop_df["time"],
                np.full(len(closed_loop_df), decoded_df["position_error_cm"].median()),
                c=colors, s=30, marker="|", label="Closed-loop trigger",
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position error (cm)")
        ax.set_title(SOURCE_LABELS[result["spike_source"]])
        ax.legend(loc="upper right")
    fig.suptitle("Closed-loop events over time", y=1.02)
    fig.tight_layout()
    fig.savefig(
        output_dir / "closed_loop_events_over_time.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(fig)

    _plot_realtime_metrics_summary(ground_truth, sorted_result, output_dir)


def _plot_realtime_side_by_side_confusion(
    ground_truth: dict,
    sorted_result: dict,
    task: str,
    title: str,
    out_path: Path,
) -> None:
    true_col = f"true_{task}"
    pred_col = f"decoded_{task}"
    label_key = f"{task}_labels"
    labels = ground_truth["offline_metrics"].get(label_key)
    if labels is None:
        labels = sorted(
            set(ground_truth["decoded_df"][true_col])
            | set(ground_truth["decoded_df"][pred_col])
            | set(sorted_result["decoded_df"][true_col])
            | set(sorted_result["decoded_df"][pred_col])
        )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    im = None
    for ax, result in zip(axes, (ground_truth, sorted_result), strict=True):
        cm = confusion_matrix(
            result["decoded_df"][true_col],
            result["decoded_df"][pred_col],
            labels=labels,
        )
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(SOURCE_LABELS[result["spike_source"]])
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Decoded")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
    fig.suptitle(title, y=1.02)
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def _plot_realtime_metrics_summary(
    ground_truth: dict,
    sorted_result: dict,
    output_dir: Path,
) -> None:
    metric_specs = [
        ("mean_position_error_cm", "Mean position error (cm)", None),
        ("median_position_error_cm", "Median position error (cm)", None),
        ("spatial_context_accuracy", "Spatial context accuracy", (0, 1)),
        ("movement_state_accuracy", "Movement state accuracy", (0, 1)),
        ("speed_r2", "Speed R²", None),
        ("speed_correlation", "Speed correlation", (-1, 1)),
    ]
    sources = ["ground_truth", "sorted"]
    labels = [SOURCE_LABELS[s] for s in sources]
    values_by_source = {
        s: results["metrics"]
        for s, results in (("ground_truth", ground_truth), ("sorted", sorted_result))
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (metric_key, ylabel, ylim) in zip(axes.ravel(), metric_specs, strict=True):
        values = [values_by_source[s][metric_key] for s in sources]
        ax.bar(labels, values, color=["#4C78A8", "#F58518"])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        if ylim is not None:
            ax.set_ylim(*ylim)
    fig.suptitle("Ground truth vs Neuropixels decoding performance", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "metrics_summary.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)

    for metric_key, filename, ylabel, ylim in (
        ("mean_position_error_cm", "mean_position_error.png", "Mean position error (cm)", None),
        ("spatial_context_accuracy", "spatial_context_accuracy.png", "Spatial context accuracy", (0, 1)),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        values = [values_by_source[s][metric_key] for s in sources]
        ax.bar(labels, values, color=["#4C78A8", "#F58518"])
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=FIGURE_DPI)
        plt.close(fig)


def _plot_decoder_comparison_single(comparison_dir: Path, figures_dir: Path) -> None:
    comparison_dir = Path(comparison_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = comparison_dir / "decoder_comparison_metrics.csv"
    best_path = comparison_dir / "best_decoder_by_target.csv"
    examples_dir = comparison_dir / "decoded_examples"
    if not metrics_path.exists() or not best_path.exists():
        return

    metrics_df = pd.read_csv(metrics_path)
    best_df = pd.read_csv(best_path)

    for target in ALL_TARGETS:
        _plot_metric_vs_window(
            metrics_df, target, figures_dir / PLOT_FILENAME[target],
        )

    _plot_best_decoder_summary(best_df, figures_dir / "best_decoder_by_target.png")
    _plot_recommended_windows(best_df, figures_dir / "recommended_window_by_target.png")

    for target in CATEGORICAL_TARGETS:
        pred_path = examples_dir / f"best_{target}_predictions.csv"
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path)
        labels = sorted(set(pred_df["true"]) | set(pred_df["pred"]))
        _save_confusion_matrix_plot(
            pred_df["true"],
            pred_df["pred"],
            labels,
            f"Best {target.replace('_', ' ')}",
            figures_dir / f"best_{target}_confusion_matrix.png",
        )

    _plot_comparison_prediction_examples(examples_dir, figures_dir)


def _plot_decoder_comparison_source_summary(comparison_dir: Path, figures_dir: Path) -> None:
    summary_path = comparison_dir / "source_comparison_metrics.csv"
    if not summary_path.exists():
        return
    comparison_df = pd.read_csv(summary_path)

    pos_df = comparison_df[comparison_df["target_name"] == "position"]
    if not pos_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [SOURCE_LABELS.get(s, s) for s in pos_df["source"]]
        ax.bar(labels, pos_df["best_metric_value"], color=["#4C78A8", "#F58518"])
        ax.set_ylabel("Mean position error (cm)")
        ax.set_title("Ground truth vs sorted: best position error")
        fig.tight_layout()
        fig.savefig(
            figures_dir / "ground_truth_vs_sorted_best_position_error.png",
            dpi=FIGURE_DPI,
        )
        plt.close(fig)

    ctx_df = comparison_df[comparison_df["target_name"] == "spatial_context"]
    if not ctx_df.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [SOURCE_LABELS.get(s, s) for s in ctx_df["source"]]
        ax.bar(labels, ctx_df["best_metric_value"], color=["#4C78A8", "#F58518"])
        ax.set_ylabel("Balanced accuracy")
        ax.set_ylim(0, 1)
        ax.set_title("Ground truth vs sorted: best spatial context accuracy")
        fig.tight_layout()
        fig.savefig(
            figures_dir / "ground_truth_vs_sorted_best_context_accuracy.png",
            dpi=FIGURE_DPI,
        )
        plt.close(fig)


def _plot_metric_vs_window(metrics_df: pd.DataFrame, target: str, out_path: Path) -> None:
    metric_key = PLOT_METRIC[target]
    target_df = metrics_df[metrics_df["target_name"] == target]
    if target_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for decoder_name in sorted(target_df["decoder_name"].unique()):
        sub = target_df[target_df["decoder_name"] == decoder_name]
        grouped = sub.groupby("decode_window_s")[metric_key].mean().reset_index()
        ax.plot(grouped["decode_window_s"], grouped[metric_key], marker="o", label=decoder_name)
    ax.set_xlabel("Decode window (s)")
    ax.set_ylabel(metric_key)
    ax.set_title(f"{target}: {metric_key} vs causal window")
    ax.legend(loc="best", fontsize=8)
    ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_best_decoder_summary(best_df: pd.DataFrame, out_path: Path) -> None:
    if best_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(best_df))
    ax.bar(x, best_df["best_metric_value"], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(best_df["target_name"], rotation=45, ha="right")
    ax.set_ylabel("Best primary metric value")
    ax.set_title("Best decoder performance by target")
    for i, row in enumerate(best_df.itertuples()):
        ax.text(
            i, row.best_metric_value,
            f"{row.best_decoder_name}\n{row.best_decode_window_s:.3f}s",
            ha="center", va="bottom", fontsize=7,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_recommended_windows(best_df: pd.DataFrame, out_path: Path) -> None:
    if best_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(best_df))
    width = 0.35
    ax.bar(
        x - width / 2, best_df["best_decode_window_s"], width,
        label="Best window", color="#4C78A8",
    )
    ax.bar(
        x + width / 2, best_df["recommended_realtime_window_s"], width,
        label="Recommended (95% optimal)", color="#F58518",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(best_df["target_name"], rotation=45, ha="right")
    ax.set_ylabel("Decode window (s)")
    ax.set_title("Best vs recommended realtime causal window")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _save_confusion_matrix_plot(
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
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_comparison_prediction_examples(examples_dir: Path, output_dir: Path) -> None:
    examples_dir = Path(examples_dir)
    output_dir = Path(output_dir)
    if not examples_dir.exists():
        return

    pos_path = examples_dir / "best_position_predictions.csv"
    if pos_path.exists():
        df = pd.read_csv(pos_path)
        time = df["time"].to_numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(
            df["true_x"], df["true_y"], c=time, s=4, alpha=0.6,
            cmap="viridis", label="True",
        )
        ax.scatter(
            df["pred_x"], df["pred_y"], c=time, s=4, alpha=0.6,
            cmap="viridis", marker="x", label="Predicted",
        )
        fig.colorbar(sc, ax=ax, label="Time (s)")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title("Best position: true vs predicted")
        ax.legend(markerscale=3)
        ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(output_dir / "best_position_true_vs_predicted.png", dpi=FIGURE_DPI)
        plt.close(fig)

    for target, filename, ylabel in (
        ("speed", "best_speed_true_vs_predicted_over_time.png", "Speed (cm/s)"),
        ("distance_to_wall", "best_distance_to_wall_true_vs_predicted_over_time.png", "Distance (cm)"),
        ("head_direction", "best_head_direction_true_vs_predicted_over_time.png", "Head direction (deg)"),
    ):
        path = examples_dir / f"best_{target}_predictions.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["time"], df["true"], label="True", linewidth=0.8)
        ax.plot(df["time"], df["pred"], label="Predicted", linewidth=0.8, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Best {target}: true vs predicted")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=FIGURE_DPI)
        plt.close(fig)
