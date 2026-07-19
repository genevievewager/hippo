"""Figures for Phase-1 temporal manifold decoding (W × L)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_temporal_comparison_outputs(temporal_dir: Path, figures_dir: Path) -> None:
    """Plot heatmaps and model comparisons from decoding/comparison outputs."""
    temporal_dir = Path(temporal_dir)
    figures_dir = Path(figures_dir) / "temporal_decoding"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for csv_path in sorted(temporal_dir.rglob("all_configurations.csv")):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue
        rel = csv_path.parent.relative_to(temporal_dir)
        out = figures_dir / rel.as_posix()
        out.mkdir(parents=True, exist_ok=True)
        _plot_wl_heatmaps(df, out)
        _plot_model_class_comparison(df, out)
        _plot_latency(df, out)
        _plot_timing_overview(csv_path.parent, out)


def _plot_wl_heatmaps(df: pd.DataFrame, out_dir: Path) -> None:
    latent = df[df["temporal_model"].isin(["flattened_history", "static_latent"])].copy()
    if latent.empty:
        return
    for (target, rep, model), sub in latent.groupby(
        ["target", "representation", "temporal_model"]
    ):
        pivot = sub.pivot_table(
            index="integration_window_s",
            columns="latent_history_frames",
            values="validation_metric_value",
            aggfunc="max" if sub["validation_metric"].iloc[0] != "mean_position_error_cm"
            and "error" not in str(sub["validation_metric"].iloc[0]).lower()
            else "min",
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(c) for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.3f}" for v in pivot.index])
        ax.set_xlabel("Latent history frames (L)")
        ax.set_ylabel("Integration window W (s)")
        ax.set_title(f"{target} | {rep} | {model}")
        fig.colorbar(im, ax=ax, label=sub["validation_metric"].iloc[0])
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{target}_{rep}_{model}.png", dpi=150)
        plt.close(fig)


def _plot_model_class_comparison(df: pd.DataFrame, out_dir: Path) -> None:
    for target, sub in df.groupby("target"):
        metric = sub["validation_metric"].iloc[0]
        direction = "lower" if "error" in metric or metric.endswith("_error_deg") else "higher"
        best_rows = []
        for model, g in sub.groupby("temporal_model"):
            if direction == "lower":
                best_rows.append(g.loc[g["validation_metric_value"].idxmin()])
            else:
                best_rows.append(g.loc[g["validation_metric_value"].idxmax()])
        best = pd.DataFrame(best_rows)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(best["temporal_model"], best["validation_metric_value"])
        ax.set_ylabel(metric)
        ax.set_title(f"Best validation score by model class: {target}")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(out_dir / f"model_comparison_{target}.png", dpi=150)
        plt.close(fig)


def _plot_latency(df: pd.DataFrame, out_dir: Path) -> None:
    if "mean_inference_latency_s" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for model, g in df.groupby("temporal_model"):
        vals = g["mean_inference_latency_s"].dropna().to_numpy() * 1000.0
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=20, alpha=0.4, label=model)
    ax.axvline(50.0, color="k", linestyle="--", label="50 ms budget")
    ax.set_xlabel("Mean inference latency (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Inference latency by temporal model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "latency_distribution.png", dpi=150)
    plt.close(fig)


def _plot_timing_overview(comparison_dir: Path, out_dir: Path) -> None:
    meta_path = Path(comparison_dir) / "timing_metadata.json"
    if not meta_path.exists():
        return
    import json
    with open(meta_path) as f:
        meta = json.load(f)
    dt = float(meta["update_dt_s"])
    windows = meta.get("integration_windows_s", [])
    histories = meta.get("latent_history_frames", [])
    lags = meta.get("prediction_lags_s", [0.0])

    fig, ax = plt.subplots(figsize=(9, 3))
    t0 = 1.0
    ax.axvline(t0, color="black", lw=2, label="behavior frame t")
    if windows:
        W = float(windows[len(windows) // 2])
        ax.plot([t0 - W, t0], [1, 1], color="C0", lw=6, solid_capstyle="butt", label=f"W={W}s spikes")
    if histories:
        L = int(histories[len(histories) // 2])
        ax.plot(
            [t0 - L * dt, t0], [2, 2], color="C1", lw=6, solid_capstyle="butt",
            label=f"L={L} frames ({L * dt:.2f}s) latent history",
        )
    tau = float(lags[0]) if lags else 0.0
    ax.plot([t0, t0 + tau], [3, 3], color="C2", lw=6, solid_capstyle="butt", label=f"tau={tau}s lag")
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["integration W", "latent history L", "prediction lag tau"])
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Timing roles (dt_update={dt:.3f}s)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "timing_overview.png", dpi=150)
    plt.close(fig)
