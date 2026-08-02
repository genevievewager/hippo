"""Latency figures for feature transforms and realtime decode stages."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from realtime.latency_profiler import STAGE_ORDER

FIGURE_DPI = 150
BUDGET_DEFAULT_MS = 50.0


def _read_update_budget_ms(experiment_dir: Path, latency_dir: Path | None = None) -> float:
    """Read update budget from per-run or experiment-level latency summaries."""
    if latency_dir is not None:
        summary = latency_dir / "latency_summary.json"
        if summary.exists():
            try:
                return float(json.loads(summary.read_text()).get("update_budget_ms", BUDGET_DEFAULT_MS))
            except Exception:
                pass
    bench = Path(experiment_dir) / "latency_profiling" / "latency_benchmark_summary.json"
    if bench.exists():
        try:
            return float(json.loads(bench.read_text()).get("update_budget_ms", BUDGET_DEFAULT_MS))
        except Exception:
            pass
    return BUDGET_DEFAULT_MS


def _load_latency_per_update(experiment_dir: Path) -> tuple[pd.DataFrame, float]:
    """Discover and load the richest per-update latency CSV for sorted realtime runs."""
    experiment_dir = Path(experiment_dir)
    rt_root = experiment_dir / "realtime_decoding"
    if not rt_root.exists():
        return pd.DataFrame(), BUDGET_DEFAULT_MS

    paths = sorted(rt_root.rglob("latency/latency_per_update.csv"))
    sorted_paths = [p for p in paths if "sorted" in str(p)]
    candidates = sorted_paths if sorted_paths else paths
    if not candidates:
        return pd.DataFrame(), BUDGET_DEFAULT_MS

    best_path = candidates[0]
    best_df = pd.DataFrame()
    for path in candidates:
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if len(df) > len(best_df):
            best_df = df
            best_path = path

    if best_df.empty:
        return pd.DataFrame(), _read_update_budget_ms(experiment_dir)

    budget_ms = _read_update_budget_ms(experiment_dir, best_path.parent)
    return best_df, budget_ms


def stage_ms_columns() -> list[tuple[str, str]]:
    """Return (stage_name, csv_column) pairs excluding total_update."""
    return [(s, f"{s}_ms") for s in STAGE_ORDER if s != "total_update"]


def plot_latency_outputs(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Write latency PNGs under figures/latency/ for PDF compilation.

    Default: publication multi-panel ``fig_latency`` and ``fig_latency_realtime``.
    Legacy single-panel helpers remain as private functions.
    """
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    from visualization.publication_decoding_plots import (
        plot_fig_latency,
        plot_fig_latency_realtime,
    )

    wrote = False
    if plot_fig_latency(experiment_dir, figures_dir) is not None:
        wrote = True
    if plot_fig_latency_realtime(experiment_dir, figures_dir) is not None:
        wrote = True
    return (figures_dir / "latency") if wrote else None


def _plot_everything(df: pd.DataFrame, out: Path, budget_ms: float) -> None:
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    categories = [
        ("feature_transform", "Feature transform\n(per causal update)"),
        ("realtime_stage", "Realtime decode stages"),
        ("isomap_compare", "Isomap teacher vs distilled"),
    ]
    for ax, (cat, title) in zip(axes, categories):
        sub = df[df["category"] == cat].copy()
        if sub.empty:
            ax.set_visible(False)
            continue
        sub = sub.sort_values("mean_ms", ascending=True)
        colors = [
            "#2ca02c" if bool(r.realtime_compatible) else "#d62728"
            for r in sub.itertuples()
        ]
        y = np.arange(len(sub))
        ax.barh(y, sub["mean_ms"], color=colors, edgecolor="k", linewidth=0.4)
        if "p95_ms" in sub.columns:
            ax.errorbar(
                sub["mean_ms"], y,
                xerr=[
                    np.zeros(len(sub)),
                    np.maximum(0.0, sub["p95_ms"] - sub["mean_ms"]),
                ],
                fmt="none", color="k", capsize=2, linewidth=0.8,
            )
        ax.axvline(budget_ms, color="0.3", linestyle="--", linewidth=1.0, label=f"{budget_ms:.0f} ms budget")
        ax.set_yticks(y)
        ax.set_yticklabels(sub["name"].tolist(), fontsize=8)
        ax.set_xlabel("Latency (ms)")
        ax.set_title(title, fontsize=10)
        ax.legend(loc="lower right", fontsize=7)
    fig.suptitle(
        "Causal update latency — all measured stages "
        "(green = realtime-compatible, red = offline-only)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out / "latency_everything.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_feature_transforms(df: pd.DataFrame, out: Path, budget_ms: float) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    df = df.sort_values("mean_ms")
    x = np.arange(len(df))
    colors = [
        "#2ca02c" if bool(r.realtime_compatible) else "#d62728"
        for r in df.itertuples()
    ]
    ax.bar(x, df["mean_ms"], color=colors, edgecolor="k", linewidth=0.5, label="mean")
    if "p95_ms" in df.columns:
        ax.plot(x, df["p95_ms"], "ko", markersize=4, label="p95")
    ax.axhline(budget_ms, color="0.25", linestyle="--", label=f"{budget_ms:.0f} ms budget")
    ax.set_xticks(x)
    ax.set_xticklabels(df["feature_mode"], rotation=20, ha="right")
    ax.set_ylabel("Per-update feature transform (ms)")
    ax.set_title("Feature-front-end latency (single causal count vector)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "feature_transform_latency.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_isomap_teacher_vs_distilled(
    df: pd.DataFrame, out: Path, budget_ms: float
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(df))
    colors = [
        "#2ca02c" if bool(r.realtime_compatible) else "#d62728"
        for r in df.itertuples()
    ]
    ax.bar(x, df["mean_ms"], color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(budget_ms, color="0.25", linestyle="--", label=f"{budget_ms:.0f} ms budget")
    ax.set_xticks(x)
    ax.set_xticklabels(df["method"], rotation=15, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Classic Isomap (offline) vs parametric distilled (realtime)")
    ax.legend(fontsize=8)
    for i, v in enumerate(df["mean_ms"]):
        ax.text(i, v, f" {v:.3f} ms", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "isomap_teacher_vs_distilled_latency.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_realtime_stages(df: pd.DataFrame, out: Path, budget_ms: float) -> None:
    if df.empty or "stage" not in df.columns:
        return
    order = [
        "spike_binning",
        "feature_transform",
        "decode_position",
        "decode_speed",
        "decode_spatial_context",
        "decode_movement_state",
        "decode_primary",
        "closed_loop_policy",
        "total_update",
    ]
    df = df.copy()
    df["_ord"] = df["stage"].map({s: i for i, s in enumerate(order)}).fillna(50)
    df = df.sort_values("_ord")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(df))
    ax.bar(x, df["mean_ms"], color="C0", edgecolor="k", linewidth=0.5, label="mean")
    if "p95_ms" in df.columns:
        ax.plot(x, df["p95_ms"], "ko", markersize=4, label="p95")
    ax.axhline(budget_ms, color="0.25", linestyle="--", label=f"{budget_ms:.0f} ms budget")
    ax.set_xticks(x)
    ax.set_xticklabels(df["stage"], rotation=25, ha="right")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Closed-loop realtime stage latency (sorted spikes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "realtime_stage_latency.png", dpi=FIGURE_DPI)
    plt.close(fig)
