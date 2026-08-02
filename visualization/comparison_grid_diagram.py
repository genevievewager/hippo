"""Schematic diagram of the decoder comparison grid search."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from visualization.constants import FIGURE_DPI, FIGURE_SUBDIR_DECODER
from visualization.publication_decoding_plots import load_comparison_metrics
from visualization.publication_style import panel_label, save_pub_figure


def _box(
    ax,
    xy: tuple[float, float],
    w: float,
    h: float,
    text: str,
    *,
    face: str = "#E3F2FD",
    edge: str = "#1565C0",
    fontsize: int = 9,
    lw: float = 1.4,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=lw, edgecolor=edge, facecolor=face,
        transform=ax.transAxes, clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2, xy[1] + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        transform=ax.transAxes, linespacing=1.15,
    )
    return patch


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "0.35") -> None:
    ax.add_patch(FancyArrowPatch(
        start, end,
        arrowstyle="-|>", mutation_scale=10, lw=1.2, color=color,
        transform=ax.transAxes, clip_on=False,
    ))


def _grid_stats(metrics) -> dict[str, str | int]:
    """Summarize an experiment metrics table for annotation."""
    if metrics is None or metrics.empty:
        return {}
    sub = metrics
    if "spike_source" in sub.columns:
        s = sub[sub["spike_source"].astype(str) == "sorted"]
        if not s.empty:
            sub = s
    sub = sub[sub["target_name"].notna()] if "target_name" in sub.columns else sub
    mode_col = "feature_mode" if "feature_mode" in sub.columns else "feature_type"
    return {
        "n_rows": len(sub),
        "n_w": sub["decode_window_s"].nunique() if "decode_window_s" in sub.columns else 0,
        "n_modes": sub[mode_col].nunique() if mode_col in sub.columns else 0,
        "n_decoders": sub["decoder_name"].nunique() if "decoder_name" in sub.columns else 0,
        "n_targets": sub["target_name"].nunique() if "target_name" in sub.columns else 0,
    }


def plot_fig_decoder_comparison_grid(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path:
    """Schematic: nested W × (F,E) × target × decoder search and selection."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = _grid_stats(load_comparison_metrics(experiment_dir))

    fig = plt.figure(figsize=(11.5, 8.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_x, left_w, left_gap = 0.02, 0.44, 0.045
    left_cx = left_x + left_w / 2

    # --- Left: nested loop schematic (top-down with fixed gaps) ---
    ax.text(
        left_x, 0.975, "Decoder comparison grid (one train/test split)",
        fontsize=11, fontweight="bold", va="top", ha="left",
    )

    y_top = 0.915
    left_boxes: list[tuple[float, str, str, str, float]] = [
        (0.058, "Profile bounds the search\n(Windows × features × decoders × targets)", "#FFF8E1", "#F9A825", 7.5),
        (0.068, "FOR each causal window W\n  build spike counts x_t(W) from [t−W, t)", "#E8F5E9", "#2E7D32", 7.5),
        (0.128, "FOR each (feature F, embedding E, k, nn)\n  fit encoder E on train → z_t (frozen at test)\n  FOR each target T · FOR each decoder D\n  fit D on train z → score test → 1 CSV row", "#E3F2FD", "#1565C0", 7),
        (0.088, "Key efficiency: E fit once per (W, F, E)\nDecoders D refit on same z_t (not full manifold refit)", "#F3E5F5", "#6A1B9A", 7),
        (0.118, "Per target T: best row by primary metric\nshortest_near_optimal: shortest W within 5% of best\n→ models/best_realtime_decoders.json", "#FBE9E7", "#BF360C", 7),
    ]
    prev_bottom: float | None = None
    for h, text, face, edge, fs in left_boxes:
        y = y_top - h
        _box(ax, (left_x, y), left_w, h, text, face=face, edge=edge, fontsize=fs)
        if prev_bottom is not None:
            _arrow(ax, (left_cx, prev_bottom), (left_cx, y + h))
        prev_bottom = y
        y_top = y - left_gap

    panel_label(ax, "A", x=left_x, y=0.99)

    # --- Right top: pipeline strip ---
    right_x, right_w = 0.58, 0.38
    ax.text(right_x, 0.975, "One configuration (one CSV row)", fontsize=10, fontweight="bold", ha="left")
    stages = [
        ("x_t(W)", "#C8E6C9"),
        ("E(·)", "#BBDEFB"),
        ("z_t", "#BBDEFB"),
        ("D(·)", "#FFE0B2"),
        ("ŷ_t", "#FFE0B2"),
        ("metric", "#F8BBD0"),
    ]
    x0, y0, w, h = right_x, 0.900, 0.046, 0.042
    gap = 0.026
    for i, (label, color) in enumerate(stages):
        xi = x0 + i * (w + gap)
        _box(ax, (xi, y0), w, h, label, face=color, edge="0.35", fontsize=7, lw=1.0)
        if i < len(stages) - 1:
            _arrow(ax, (xi + w, y0 + h / 2), (xi + w + gap * 0.55, y0 + h / 2))

    ax.text(
        right_x, 0.855,
        "Same W for encoder and decoder · train split fits E and D · test split scores only",
        fontsize=7.5, color="0.35", ha="left", va="top",
    )
    panel_label(ax, "B", x=right_x - 0.02, y=0.99)

    # --- Right middle: scale ---
    if stats:
        n_rows = stats["n_rows"]
        detail = (
            f"This experiment: {n_rows} scored configurations\n"
            f"  · {stats['n_w']} windows W\n"
            f"  · {stats['n_modes']} feature modes (F,E)\n"
            f"  · {stats['n_decoders']} decoder families D\n"
            f"  · {stats['n_targets']} targets T\n"
            f"≈ W × modes × (targets × decoders per target)\n"
            f"Manifold profile: ~15 embedding jobs × 4 W\n"
            f"≈ 60 encoder fits + ~{n_rows} decoder fits"
        )
    else:
        detail = (
            "Typical manifolds profile (bounded grid):\n"
            "  · 4 windows W\n"
            "  · 15 (F,E,k,nn) jobs → ~60 encoder fits\n"
            "  · 8 targets × 2–4 decoders each\n"
            "  · ~1,500 decoder fits total (one CSV row each)"
        )

    box_h = 0.28
    c_bottom = 0.42

    _box(ax, (right_x, c_bottom), right_w, box_h, detail, face="#FAFAFA", edge="0.45", fontsize=7.5)
    panel_label(ax, "C", x=right_x - 0.02, y=c_bottom + box_h + 0.02)

    out_path = out_dir / "fig_decoder_comparison_grid.png"
    save_pub_figure(fig, out_path, dpi=FIGURE_DPI, adjust=False)
    return out_path
