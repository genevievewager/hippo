"""Quadrant (static/dynamic × linear/nonlinear) comparison figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualization.constants import FIGURE_DPI

QUADRANT_ORDER: tuple[str, ...] = (
    "static_linear",
    "static_nonlinear",
    "dynamic_linear",
    "dynamic_nonlinear",
)
QUADRANT_LABELS: dict[str, str] = {
    "static_linear": "Static linear",
    "static_nonlinear": "Static nonlinear",
    "dynamic_linear": "Dynamic linear",
    "dynamic_nonlinear": "Dynamic nonlinear",
}
QUADRANT_DEFAULTS: dict[str, str | None] = {
    "static_linear": "global_pca",
    "static_nonlinear": "diffusion_nystrom",
    "dynamic_linear": "global_lds",
    "dynamic_nonlinear": None,
}

STABILITY_METRICS: tuple[tuple[str, str], ...] = (
    ("pairwise_distance_preservation", "Pairwise distance preservation"),
    ("neighborhood_trustworthiness", "Neighborhood trustworthiness"),
    ("latent_velocity_mean", "Latent velocity (mean)"),
    ("latent_smoothness", "Latent smoothness"),
    ("latent_dimensionality_proxy", "Latent dimensionality proxy"),
    ("procrustes_alignment_error", "Procrustes alignment error"),
)

_PROC_EMPTY_NOTE = "no reference embedding"


def _quadrant_series(
    stability_df: pd.DataFrame | None,
    *,
    decode_window_s: float | None = None,
) -> dict[str, dict[str, float | None]]:
    """Mean metric per occupied quadrant embedding."""
    out: dict[str, dict[str, float | None]] = {}
    if stability_df is None or stability_df.empty:
        return out
    df = stability_df.copy()
    emb_col = "embedding_type" if "embedding_type" in df.columns else None
    if emb_col is None:
        return out
    if decode_window_s is not None and "decode_window_s" in df.columns:
        ms = int(round(float(decode_window_s) * 1000))
        df = df[np.round(df["decode_window_s"].astype(float) * 1000).astype(int) == ms]
    for qid in QUADRANT_ORDER:
        emb = QUADRANT_DEFAULTS.get(qid)
        if not emb:
            continue
        sub = df[df[emb_col].astype(str) == emb]
        row: dict[str, float | None] = {}
        for col, _title in STABILITY_METRICS:
            if col not in sub.columns or sub.empty:
                row[col] = None
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            row[col] = float(vals.mean()) if not vals.empty else None
        out[qid] = row
    return out


def make_quadrant_stability_figure(
    stability_df: pd.DataFrame | None,
    *,
    decode_window_s: float | None = None,
) -> plt.Figure:
    """2×3 figure, one panel per stability metric."""
    series = _quadrant_series(stability_df, decode_window_s=decode_window_s)
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.2))
    occupied = [q for q in QUADRANT_ORDER if QUADRANT_DEFAULTS.get(q)]
    labels = [QUADRANT_LABELS[q] for q in occupied]
    colors = ["#3d5a80", "#ee6c4d", "#98c1d9"]
    for ax, (col, title) in zip(axes.ravel(), STABILITY_METRICS):
        vals = []
        for q in occupied:
            v = (series.get(q) or {}).get(col)
            vals.append(np.nan if v is None else float(v))
        finite = [v for v in vals if np.isfinite(v)]
        if col == "procrustes_alignment_error" and not finite:
            ax.set_axis_off()
            ax.set_title(title)
            ax.text(
                0.5, 0.5, _PROC_EMPTY_NOTE,
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
            )
            continue
        ax.bar(range(len(labels)), vals, color=colors[: len(labels)])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(title)
        ax.set_ylabel("")
    fig.tight_layout()
    return fig


def plot_quadrant_stability(
    stability_df: pd.DataFrame | None,
    output_path: Path,
    *,
    decode_window_s: float | None = None,
) -> Path:
    fig = make_quadrant_stability_figure(stability_df, decode_window_s=decode_window_s)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _time_column(decoded: pd.DataFrame) -> str | None:
    for col in ("time", "decode_time"):
        if col in decoded.columns:
            return col
    return None


def _plot_true_vs_pred(ax, decoded: pd.DataFrame | None, title: str):
    """True vs decoded; position uses time-colored scatter (circles / crosses).

    Returns a ScalarMappable for a shared time colorbar, else None.
    """
    ax.set_title(title)
    if decoded is None or decoded.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no replay", ha="center", va="center", transform=ax.transAxes)
        return None
    if {"true_x", "true_y", "decoded_x", "decoded_y"}.issubset(decoded.columns):
        tcol = _time_column(decoded)
        time = None
        if tcol is not None:
            time = pd.to_numeric(decoded[tcol], errors="coerce").to_numpy()
            if not np.isfinite(time).any():
                time = None
        if time is not None:
            sc = ax.scatter(
                decoded["true_x"], decoded["true_y"],
                c=time, s=8, alpha=0.55, cmap="viridis", label="True",
            )
            ax.scatter(
                decoded["decoded_x"], decoded["decoded_y"],
                c=time, s=12, alpha=0.55, cmap="viridis", marker="x", label="Decoded",
            )
        else:
            sc = ax.scatter(
                decoded["true_x"], decoded["true_y"],
                c="#4C72B0", s=8, alpha=0.55, label="True",
            )
            ax.scatter(
                decoded["decoded_x"], decoded["decoded_y"],
                c="#DD8452", s=12, alpha=0.55, marker="x", label="Decoded",
            )
        x0 = float(np.nanmin([decoded["true_x"].min(), decoded["decoded_x"].min()]))
        x1 = float(np.nanmax([decoded["true_x"].max(), decoded["decoded_x"].max()]))
        y0 = float(np.nanmin([decoded["true_y"].min(), decoded["decoded_y"].min()]))
        y1 = float(np.nanmax([decoded["true_y"].max(), decoded["decoded_y"].max()]))
        lo = min(x0, y0)
        hi = max(x1, y1)
        pad = 0.02 * max(hi - lo, 1.0)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="best", markerscale=1.4)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        return sc if time is not None else None
    tcol = _time_column(decoded)
    ytrue = next((c for c in ("true_value", "y_true", "speed") if c in decoded.columns), None)
    ypred = next((c for c in ("decoded_value", "y_pred", "decoded_speed") if c in decoded.columns), None)
    if tcol and ytrue and ypred:
        ax.plot(decoded[tcol], decoded[ytrue], lw=0.9, color="0.25", label="true")
        ax.plot(decoded[tcol], decoded[ypred], lw=0.9, ls="--", color="#ee6c4d", label="decoded")
        ax.legend(fontsize=7, loc="best")
        return None
    ax.set_axis_off()
    ax.text(0.5, 0.5, "true/pred columns missing", ha="center", va="center", transform=ax.transAxes)
    return None


def make_quadrant_behavior_figure(
    decoded_by_quadrant: Mapping[str, pd.DataFrame | None],
    metrics_by_quadrant: Mapping[str, Mapping[str, Any]],
    *,
    target: str = "position",
) -> plt.Figure:
    """3×2: four true-vs-pred panels, latency bars, accuracy bars."""
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 11.0), layout="constrained")
    q_axes = {
        "static_linear": axes[0, 0],
        "static_nonlinear": axes[0, 1],
        "dynamic_linear": axes[1, 0],
        "dynamic_nonlinear": axes[1, 1],
    }
    time_mappable = None
    occupied_xy_axes = []
    for qid, ax in q_axes.items():
        label = QUADRANT_LABELS[qid]
        emb = QUADRANT_DEFAULTS.get(qid)
        if not emb:
            ax.set_axis_off()
            ax.set_title(f"{label} (not implemented)")
            ax.text(
                0.5, 0.5, "not implemented",
                ha="center", va="center", transform=ax.transAxes, fontsize=10,
            )
            continue
        sm = _plot_true_vs_pred(
            ax,
            decoded_by_quadrant.get(qid),
            f"{label} · {emb}",
        )
        occupied_xy_axes.append(ax)
        if sm is not None:
            time_mappable = sm
    if time_mappable is not None and occupied_xy_axes:
        cbar = fig.colorbar(
            time_mappable, ax=occupied_xy_axes, fraction=0.025, pad=0.02,
        )
        cbar.set_label("Time (s)", fontsize=8)
        cbar.ax.tick_params(labelsize=6)

    occupied = [q for q in QUADRANT_ORDER if QUADRANT_DEFAULTS.get(q)]
    labels = [QUADRANT_LABELS[q] for q in occupied]
    lat_ax, acc_ax = axes[2, 0], axes[2, 1]
    lat_vals = []
    acc_vals = []
    acc_name = "accuracy"
    for q in occupied:
        m = dict(metrics_by_quadrant.get(q) or {})
        lat = m.get("mean_predict_ms", m.get("p99_total_ms", m.get("mean_total_update_ms")))
        try:
            lat_vals.append(float(lat) if lat is not None else np.nan)
        except (TypeError, ValueError):
            lat_vals.append(np.nan)
        for key in (
            "mean_position_error_cm",
            "r2",
            "balanced_accuracy",
            "mean_circular_error_deg",
        ):
            if key in m and m[key] is not None:
                try:
                    acc_vals.append(float(m[key]))
                    acc_name = key
                    break
                except (TypeError, ValueError):
                    continue
        else:
            acc_vals.append(np.nan)

    lat_ax.bar(range(len(labels)), lat_vals, color="#3d5a80")
    lat_ax.set_xticks(range(len(labels)))
    lat_ax.set_xticklabels(labels, rotation=20, ha="right")
    lat_ax.set_title("Latency (ms)")
    lat_ax.set_ylabel("mean predict ms")

    acc_ax.bar(range(len(labels)), acc_vals, color="#ee6c4d")
    acc_ax.set_xticks(range(len(labels)))
    acc_ax.set_xticklabels(labels, rotation=20, ha="right")
    acc_ax.set_title(f"Accuracy · {target}")
    acc_ax.set_ylabel(acc_name)
    return fig


def plot_quadrant_behavior(
    decoded_by_quadrant: Mapping[str, pd.DataFrame | None],
    metrics_by_quadrant: Mapping[str, Mapping[str, Any]],
    output_path: Path,
    *,
    target: str = "position",
) -> Path:
    fig = make_quadrant_behavior_figure(
        decoded_by_quadrant, metrics_by_quadrant, target=target,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output_path
