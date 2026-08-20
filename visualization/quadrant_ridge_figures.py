"""Publication figures for ridge quadrant center-bias diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from realtime.quadrant_ridge_diagnostics import MethodDiagnostics
from visualization.constants import FIGURE_DPI
from visualization.publication_style import (
    apply_publication_theme,
    panel_label,
    save_pub_figure,
    style_figure_axes,
)

_METHOD_COLORS = {
    "counts": "#4C72B0",
    "global_pca": "#DD8452",
    "region_pca": "#55A868",
    "diffusion_nystrom": "#C44E52",
    "global_lds": "#8172B3",
}


def _color(method_id: str) -> str:
    return _METHOD_COLORS.get(method_id, "#333333")


def _time_scatter(ax, df: pd.DataFrame, *, title: str, show_decoded_only: bool = False):
    ax.set_title(title, fontsize=10)
    if df is None or df.empty:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
        return None
    t = pd.to_numeric(df["time"], errors="coerce").to_numpy()
    if not show_decoded_only:
        sc = ax.scatter(
            df["true_x"], df["true_y"], c=t, s=10, alpha=0.55,
            cmap="viridis", marker="o",
        )
    else:
        sc = ax.scatter(
            df["decoded_x"], df["decoded_y"], c=t, s=12, alpha=0.65,
            cmap="viridis", marker="x",
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    return sc


def _set_arena(ax, arena_bounds: tuple[float, float, float, float]) -> None:
    x0, x1, y0, y1 = arena_bounds
    pad = 0.02 * max(x1 - x0, y1 - y0, 1.0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)


def plot_fig_quadrant_ridge_trajectory(
    true_df: pd.DataFrame,
    results: Mapping[str, MethodDiagnostics],
    metrics_df: pd.DataFrame,
    *,
    center: tuple[float, float],
    arena_bounds: tuple[float, float, float, float],
    output_dir: Path,
) -> Path | None:
    apply_publication_theme()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    traj_methods = [
        ("true", "True trajectory", None),
        ("counts", "Counts + ridge", results.get("counts")),
        ("global_pca", "Global PCA + ridge", results.get("global_pca")),
        ("region_pca", "Region PCA + ridge", results.get("region_pca")),
        ("diffusion_nystrom", "Diffusion + Nyström + ridge", results.get("diffusion_nystrom")),
        ("global_lds", "LDS + ridge", results.get("global_lds")),
    ]

    fig = plt.figure(figsize=(14.0, 16.0))
    gs = GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.28)

    mappable = None
    panel_letters = list("ABCDEF")
    for i, (mid, title, diag) in enumerate(traj_methods[:6]):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        if mid == "true":
            mappable = _time_scatter(ax, true_df, title=title)
        elif diag and diag.available and diag.decoded is not None:
            mappable = _time_scatter(ax, diag.decoded, title=title, show_decoded_only=True)
        else:
            ax.set_title(title, fontsize=10)
            ax.set_axis_off()
            reason = diag.skip_reason if diag else "missing"
            ax.text(0.5, 0.5, reason or "unavailable", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        _set_arena(ax, arena_bounds)
        panel_label(ax, panel_letters[i])

    # G — radial scatter all methods
    ax_g = fig.add_subplot(gs[2, 0])
    cx, cy = center
    for mid, diag in results.items():
        if not diag.available or diag.decoded is None:
            continue
        df = diag.decoded
        r_true = np.hypot(df["true_x"] - cx, df["true_y"] - cy)
        r_pred = np.hypot(df["decoded_x"] - cx, df["decoded_y"] - cy)
        ax_g.scatter(r_true, r_pred, s=8, alpha=0.35, color=_color(mid), label=diag.label)
    lim = ax_g.get_xlim()
    if lim[1] > lim[0]:
        ax_g.plot(lim, lim, "k--", lw=0.8, alpha=0.6)
    ax_g.set_xlabel("True radius (cm)")
    ax_g.set_ylabel("Predicted radius (cm)")
    ax_g.set_title("Radial contraction")
    ax_g.legend(fontsize=7, frameon=False, loc="best")
    panel_label(ax_g, "G")

    # H — calibration slopes
    ax_h = fig.add_subplot(gs[2, 1])
    sub = metrics_df[metrics_df["available"] == True]  # noqa: E712
    if not sub.empty:
        x = np.arange(len(sub))
        w = 0.25
        ax_h.bar(x - w, sub["slope_x"], width=w, label="slope x", color="#4C72B0")
        ax_h.bar(x, sub["slope_y"], width=w, label="slope y", color="#DD8452")
        ax_h.bar(x + w, sub["slope_r"], width=w, label="slope r", color="#55A868")
        ax_h.axhline(1.0, color="0.3", ls="--", lw=0.8)
        ax_h.set_xticks(x)
        ax_h.set_xticklabels(sub["label"], rotation=25, ha="right", fontsize=8)
        ax_h.set_ylabel("Calibration slope")
        ax_h.legend(fontsize=7, frameon=False)
    panel_label(ax_h, "H")

    # I — mean error
    ax_i = fig.add_subplot(gs[2, 2])
    if not sub.empty:
        ax_i.bar(sub["label"], sub["mean_position_error_cm"], color=[_color(m) for m in sub["method_id"]])
        ax_i.set_ylabel("Mean error (cm)")
        ax_i.tick_params(axis="x", rotation=25, labelsize=8)
    panel_label(ax_i, "I")

    # J — mean radial bias
    ax_j = fig.add_subplot(gs[3, 0])
    if not sub.empty:
        ax_j.bar(sub["label"], sub["mean_radial_bias"], color=[_color(m) for m in sub["method_id"]])
        ax_j.axhline(0.0, color="0.3", ls="--", lw=0.8)
        ax_j.set_ylabel("Mean radial bias (cm)")
        ax_j.tick_params(axis="x", rotation=25, labelsize=8)
    panel_label(ax_j, "J")

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=fig.axes[:6], fraction=0.02, pad=0.02)
        cbar.set_label("Time (s)", fontsize=9)

    fig.suptitle(
        "Quadrant Comparison on Ridge: Trajectory Decoding and Center Bias "
        "Across Latent Representations",
        fontsize=13, y=0.995,
    )
    style_figure_axes(fig)
    png = save_pub_figure(fig, output_dir / "fig_quadrant_ridge_trajectory.png", dpi=FIGURE_DPI)
    pdf = output_dir / "fig_quadrant_ridge_trajectory.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def plot_fig_quadrant_ridge_shrinkage(
    results: Mapping[str, MethodDiagnostics],
    occupancy: dict,
    *,
    arena_bounds: tuple[float, float, float, float],
    output_dir: Path,
) -> Path | None:
    apply_publication_theme()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14.0, 16.0))
    gs = GridSpec(4, 3, figure=fig, hspace=0.42, wspace=0.32)

    # A — equation panel
    ax_a = fig.add_subplot(gs[0, :])
    ax_a.axis("off")
    ax_a.text(
        0.5, 0.72,
        r"$\min_W \|Y - XW\|_2^2 + \alpha \|W\|_2^2$",
        ha="center", va="center", fontsize=16, transform=ax_a.transAxes,
    )
    ax_a.text(
        0.5, 0.38,
        r"$\hat{Y} = XW + b$   ·   larger $\alpha$ shrinks $W$ and pulls predictions "
        "toward the training mean (center bias under occupancy imbalance).",
        ha="center", va="center", fontsize=11, transform=ax_a.transAxes,
    )
    panel_label(ax_a, "A", x=0.01, y=0.98, ha="left")

    def _alpha_panel(row, col, letter, colname, ylabel, title):
        ax = fig.add_subplot(gs[row, col])
        for mid, diag in results.items():
            if not diag.available or diag.alpha_sweep is None or diag.alpha_sweep.empty:
                continue
            df = diag.alpha_sweep.sort_values("alpha")
            ax.plot(df["alpha"], df[colname], marker="o", ms=3, lw=1.2, label=diag.label, color=_color(mid))
        ax.set_xscale("log")
        ax.set_xlabel("Ridge α")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        if col == 2:
            ax.legend(fontsize=7, frameon=False, loc="best")
        panel_label(ax, letter)

    _alpha_panel(1, 0, "B", "mean_position_error_cm", "Mean error (cm)", "Error vs α")
    _alpha_panel(1, 1, "C", "slope_r", "Radial slope", "Radial slope vs α")
    _alpha_panel(1, 2, "D", "mean_radial_bias", "Mean radial bias (cm)", "Radial bias vs α")

    # E/F — dimension sweeps
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    for mid, diag in results.items():
        if not diag.available or diag.dim_sweep is None or diag.dim_sweep.empty:
            continue
        df = diag.dim_sweep.sort_values("n_components")
        ax_e.plot(df["n_components"], df["mean_position_error_cm"], marker="o", label=diag.label, color=_color(mid))
        ax_f.plot(df["n_components"], df["slope_r"], marker="o", label=diag.label, color=_color(mid))
    ax_e.set_xlabel("Latent dimension k")
    ax_e.set_ylabel("Mean error (cm)")
    ax_e.set_title("Error vs k")
    ax_f.set_xlabel("Latent dimension k")
    ax_f.set_ylabel("Radial slope")
    ax_f.set_title("Radial slope vs k")
    ax_f.axhline(1.0, color="0.3", ls="--", lw=0.8)
    ax_e.legend(fontsize=7, frameon=False)
    panel_label(ax_e, "E")
    panel_label(ax_f, "F")

    # G — coef norm vs alpha (first available method overlay)
    ax_g = fig.add_subplot(gs[2, 2])
    for mid, diag in results.items():
        if not diag.available or diag.coef_norm_vs_alpha is None or diag.coef_norm_vs_alpha.empty:
            continue
        df = diag.coef_norm_vs_alpha.sort_values("alpha")
        ax_g.plot(df["alpha"], df["coef_l2_norm"], marker="o", ms=3, label=diag.label, color=_color(mid))
    ax_g.set_xscale("log")
    ax_g.set_xlabel("Ridge α")
    ax_g.set_ylabel(r"$\|W\|_2$")
    ax_g.set_title("Coefficient shrinkage")
    ax_g.legend(fontsize=7, frameon=False)
    panel_label(ax_g, "G")

    # H — occupancy map
    ax_h = fig.add_subplot(gs[3, 0])
    hist = np.asarray(occupancy.get("hist", []), dtype=float)
    x_edges = np.asarray(occupancy.get("x_edges", []), dtype=float)
    y_edges = np.asarray(occupancy.get("y_edges", []), dtype=float)
    if hist.size and x_edges.size and y_edges.size:
        im = ax_h.imshow(
            hist.T,
            origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="auto",
            cmap="magma",
        )
        fig.colorbar(im, ax=ax_h, fraction=0.046, pad=0.04, label="Visit count")
    ax_h.set_title("True occupancy")
    ax_h.set_xlabel("x (cm)")
    ax_h.set_ylabel("y (cm)")
    panel_label(ax_h, "H")

    # I — optional weighted vs unweighted
    ax_i = fig.add_subplot(gs[3, 1:])
    labels, unweighted, weighted = [], [], []
    for mid in ("counts", "global_pca"):
        diag = results.get(mid)
        if diag is None or not diag.available or diag.metrics is None:
            continue
        labels.append(diag.label)
        unweighted.append(diag.metrics.slope_r)
        weighted.append(
            diag.occupancy_weighted.slope_r
            if diag.occupancy_weighted is not None
            else np.nan
        )
    if labels:
        x = np.arange(len(labels))
        w = 0.35
        ax_i.bar(x - w / 2, unweighted, width=w, label="Unweighted ridge", color="#4C72B0")
        ax_i.bar(x + w / 2, weighted, width=w, label="Inverse-occupancy weighted", color="#C44E52")
        ax_i.axhline(1.0, color="0.3", ls="--", lw=0.8)
        ax_i.set_xticks(x)
        ax_i.set_xticklabels(labels)
        ax_i.set_ylabel("Radial slope")
        ax_i.set_title("Optional occupancy reweighting (diagnostic)")
        ax_i.legend(fontsize=8, frameon=False)
    else:
        ax_i.set_axis_off()
        ax_i.text(0.5, 0.5, "Weighted ridge not run", ha="center", va="center", transform=ax_i.transAxes)
    panel_label(ax_i, "I")

    fig.suptitle(
        "Quadrant Comparison on Ridge: Effect of Regularization and Latent Compression",
        fontsize=13, y=0.995,
    )
    style_figure_axes(fig)
    png = save_pub_figure(fig, output_dir / "fig_quadrant_ridge_shrinkage.png", dpi=FIGURE_DPI)
    pdf = output_dir / "fig_quadrant_ridge_shrinkage.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png
