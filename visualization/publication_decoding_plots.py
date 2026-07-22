"""Publication multi-panel decoder / realtime / deployment / latency figures.

Paper figures 4, 5, 9, 10, 11 (+ optional Fig 12 temporal W×L).
Style matches ``population_activity_plots`` (seaborn ticks/paper, A–D panels).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

from visualization.constants import (
    FIGURE_DPI,
    FIGURE_SUBDIR_DECODER,
    FIGURE_SUBDIR_REALTIME,
    FIGURE_SUBDIR_TEMPORAL,
)
from visualization.publication_style import (
    expand_xlim_for_labels,
    legend_below,
    legend_outside,
    panel_label,
    save_pub_figure,
)

sns.set_theme(style="ticks", context="paper", font_scale=1.0)

CONTINUOUS_TARGETS = ("position", "speed", "head_direction", "distance_to_wall", "acceleration")
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

LOWER_IS_BETTER = {
    "mean_position_error_cm",
    "mean_circular_error_deg",
    "mae",
    "rmse",
}


def _read_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _empty_panel(ax, message: str) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=9, color="0.4", wrap=True)


def _is_lower_better(metric: str) -> bool:
    m = str(metric).lower()
    return m in LOWER_IS_BETTER or "error" in m


def _best_row(sub: pd.DataFrame, metric: str) -> pd.Series | None:
    if sub.empty or metric not in sub.columns:
        return None
    vals = pd.to_numeric(sub[metric], errors="coerce")
    if vals.notna().sum() == 0:
        return None
    idx = vals.idxmin() if _is_lower_better(metric) else vals.idxmax()
    return sub.loc[idx]


def load_comparison_metrics(experiment_dir: Path, prefer: str = "sorted") -> pd.DataFrame:
    """Load decoder comparison metrics, preferring sorted spike source."""
    root = Path(experiment_dir) / "decoder_comparison"
    candidates = [
        root / prefer / "decoder_comparison_metrics.csv",
        root / "decoder_comparison_metrics.csv",
        root / "ground_truth" / "decoder_comparison_metrics.csv",
        root / "sorted" / "decoder_comparison_metrics.csv",
    ]
    for path in candidates:
        df = _read_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def load_best_by_target(experiment_dir: Path) -> pd.DataFrame:
    root = Path(experiment_dir)
    for path in (
        root / "deployment_decoder_selection" / "best_decoder_by_target_sorted.csv",
        root / "decoder_comparison" / "sorted" / "best_decoder_by_target.csv",
        root / "decoder_comparison" / "best_decoder_by_target.csv",
    ):
        df = _read_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def load_window_scores(experiment_dir: Path) -> pd.DataFrame:
    root = Path(experiment_dir)
    for path in (
        root / "deployment_decoder_selection" / "all_sorted_window_scores.csv",
        root / "decoder_comparison" / "sorted" / "all_window_scores_sorted.csv",
    ):
        df = _read_csv(path)
        if not df.empty:
            return df
    return pd.DataFrame()


def _metric_for_target(df: pd.DataFrame, target: str) -> str:
    sub = df[df["target_name"] == target] if "target_name" in df.columns else df
    if not sub.empty and "primary_metric" in sub.columns:
        return str(sub["primary_metric"].iloc[0])
    return PLOT_METRIC.get(target, "r2")


def _best_per_window(sub: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Best decoder score per decode_window_s."""
    rows = []
    for w, g in sub.groupby("decode_window_s"):
        row = _best_row(g, metric)
        if row is not None:
            rows.append({"decode_window_s": float(w), "metric_value": float(row[metric]), "decoder_name": row.get("decoder_name", "")})
    return pd.DataFrame(rows)


def _cleanup_legacy_pngs(folder: Path, keep_prefixes: tuple[str, ...] = ("fig_",)) -> None:
    folder = Path(folder)
    if not folder.exists():
        return
    for png in folder.rglob("*.png"):
        if any(png.name.startswith(p) for p in keep_prefixes):
            continue
        png.unlink(missing_ok=True)
    # Remove empty nested directories left behind by legacy sprawl
    for sub in sorted(folder.rglob("*"), reverse=True):
        if sub.is_dir():
            try:
                next(sub.iterdir())
            except StopIteration:
                sub.rmdir()
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Fig 4 — Causal decoding performance
# ---------------------------------------------------------------------------

def plot_fig_decoding_performance(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 4: metric vs W, categorical vs W, best decoder, selected window."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir)
    best = load_best_by_target(experiment_dir)
    if metrics.empty and best.empty:
        return None

    # Prefer counts for window curves (fair decoder comparison across W)
    if not metrics.empty and "feature_type" in metrics.columns:
        counts = metrics[metrics["feature_type"] == "counts"].copy()
        if counts.empty:
            counts = metrics.copy()
    else:
        counts = metrics.copy()

    fig = plt.figure(figsize=(12.0, 7.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.55)

    # A — continuous targets vs window
    ax_a = fig.add_subplot(gs[0, 0])
    cont = [t for t in CONTINUOUS_TARGETS if not counts.empty and t in set(counts["target_name"])]
    if cont:
        for target in cont:
            sub = counts[counts["target_name"] == target]
            metric = _metric_for_target(sub, target)
            curve = _best_per_window(sub, metric)
            if curve.empty:
                continue
            # Short legend labels (metric in ylabel is shared)
            ax_a.plot(
                curve["decode_window_s"], curve["metric_value"],
                marker="o", ms=4, lw=1.3, label=target,
            )
        ax_a.set_xlabel("Causal window W (s)")
        ax_a.set_ylabel("Best primary metric")
        handles_a, labels_a = ax_a.get_legend_handles_labels()
        if ax_a.get_legend() is not None:
            ax_a.get_legend().remove()
        sns.despine(ax=ax_a)
    else:
        handles_a, labels_a = [], []
        _empty_panel(ax_a, "No continuous-target metrics")
    panel_label(ax_a, "A")

    # B — categorical vs window
    ax_b = fig.add_subplot(gs[0, 1])
    handles_b, labels_b = [], []
    cats = [t for t in CATEGORICAL_TARGETS if not counts.empty and t in set(counts["target_name"])]
    if cats:
        for target in cats:
            sub = counts[counts["target_name"] == target]
            metric = _metric_for_target(sub, target)
            curve = _best_per_window(sub, metric)
            if curve.empty:
                continue
            ax_b.plot(
                curve["decode_window_s"], curve["metric_value"],
                marker="o", ms=4, lw=1.3, label=target,
            )
        ax_b.set_xlabel("Causal window W (s)")
        ax_b.set_ylabel("Balanced accuracy")
        ax_b.set_ylim(0, 1.05)
        handles_b, labels_b = ax_b.get_legend_handles_labels()
        if ax_b.get_legend() is not None:
            ax_b.get_legend().remove()
        sns.despine(ax=ax_b)
    else:
        _empty_panel(ax_b, "No categorical-target metrics")
    panel_label(ax_b, "B")

    # C — best decoder by target
    ax_c = fig.add_subplot(gs[1, 0])
    if not best.empty:
        plot_df = best.copy()
        name_col = "best_decoder_name" if "best_decoder_name" in plot_df.columns else "decoder_name"
        val_col = "best_metric_value" if "best_metric_value" in plot_df.columns else None
        if val_col is None:
            _empty_panel(ax_c, "No best-decoder values")
        else:
            order = plot_df["target_name"].tolist()
            colors = sns.color_palette("deep", n_colors=len(order))
            y = np.arange(len(order))
            vals = plot_df[val_col].astype(float).to_numpy()
            ax_c.barh(y, vals, color=colors, edgecolor="k", lw=0.3)
            ax_c.set_yticks(y)
            ax_c.set_yticklabels(order, fontsize=8)
            for yi, (_, row) in enumerate(plot_df.iterrows()):
                v = float(row[val_col])
                name = str(row[name_col])
                short = name.replace("random_forest_", "rf_").replace("classifier", "clf").replace("regressor", "reg")
                ax_c.text(
                    v + 0.02 * max(float(np.max(vals)), 1.0), yi, short,
                    va="center", ha="left", fontsize=6, color="0.25",
                    clip_on=False,
                )
            ax_c.set_xlabel("Best metric value")
            expand_xlim_for_labels(ax_c, vals, pad_frac=0.45)
            sns.despine(ax=ax_c)
    else:
        _empty_panel(ax_c, "No best_decoder_by_target")
    panel_label(ax_c, "C")

    # D — recommended / selected window by target
    ax_d = fig.add_subplot(gs[1, 1])
    if not best.empty:
        w_col = (
            "recommended_realtime_window_s"
            if "recommended_realtime_window_s" in best.columns
            else "best_decode_window_s"
        )
        if w_col not in best.columns:
            _empty_panel(ax_d, "No selected windows")
        else:
            order = best["target_name"].tolist()
            windows = best[w_col].astype(float).to_numpy()
            y = np.arange(len(order))
            ax_d.barh(y, windows, color=sns.color_palette("muted", n_colors=len(order)), edgecolor="k", lw=0.3)
            ax_d.set_yticks(y)
            ax_d.set_yticklabels(order, fontsize=8)
            ax_d.set_xlabel("Selected window W (s)")
            sns.despine(ax=ax_d)
    else:
        _empty_panel(ax_d, "No selected windows")
    panel_label(ax_d, "D")

    # Combined figure legend under top row panels (never over data)
    all_h = list(handles_a) + list(handles_b)
    all_l = list(labels_a) + list(labels_b)
    if all_h:
        from visualization.publication_style import figure_legend_below
        figure_legend_below(fig, all_h, all_l, ncol=min(5, len(all_h)), fontsize=6, y=0.02)
    return save_pub_figure(
        fig, out_dir / "fig_decoding_performance.png", dpi=FIGURE_DPI,
        rect=(0.10, 0.12, 0.98, 0.92),
    )


# ---------------------------------------------------------------------------
# Fig 5 — Manifold vs spike counts
# ---------------------------------------------------------------------------

def plot_fig_manifold_decoding(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 5: feature-mode heatmap, counts vs manifold delta, region PCA, EV."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir)
    if metrics.empty or "feature_type" not in metrics.columns:
        return None

    mvc_path = (
        experiment_dir / "decoder_comparison" / "sorted" / "manifold_vs_counts_summary.csv"
    )
    mvc = _read_csv(mvc_path)
    if mvc.empty:
        mvc = _read_csv(experiment_dir / "decoder_comparison" / "manifold_vs_counts_summary.csv")

    ev = _read_csv(
        experiment_dir / "decoder_comparison" / "sorted" / "manifold_explained_variance.csv"
    )
    if ev.empty:
        ev = _read_csv(experiment_dir / "decoder_comparison" / "manifold_explained_variance.csv")

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # A — feature × target heatmap of best primary metric (normalized per target)
    ax_a = fig.add_subplot(gs[0, 0])
    targets = sorted(metrics["target_name"].unique())
    modes = [m for m in ("counts", "global_pca", "region_pca", "global_isomap", "global_isomap_distilled")
             if m in set(metrics["feature_type"])]
    if not modes:
        modes = sorted(metrics["feature_type"].dropna().unique())
    mat = np.full((len(targets), len(modes)), np.nan)
    for i, target in enumerate(targets):
        sub_t = metrics[metrics["target_name"] == target]
        metric = _metric_for_target(sub_t, target)
        for j, mode in enumerate(modes):
            sub = sub_t[sub_t["feature_type"] == mode]
            row = _best_row(sub, metric)
            if row is not None:
                mat[i, j] = float(row[metric])
        # Normalize per target for display (0=worst, 1=best)
        col = mat[i, :]
        if np.isfinite(col).sum() >= 2:
            if _is_lower_better(metric):
                lo, hi = np.nanmin(col), np.nanmax(col)
                mat[i, :] = (hi - col) / (hi - lo + 1e-12)
            else:
                lo, hi = np.nanmin(col), np.nanmax(col)
                mat[i, :] = (col - lo) / (hi - lo + 1e-12)
    if modes and targets:
        sns.heatmap(
            mat, ax=ax_a, annot=True, fmt=".2f",
            xticklabels=modes, yticklabels=targets,
            cmap="viridis", cbar_kws={"label": "Normalized best (1=best)"},
        )
        ax_a.set_xlabel("Feature mode")
        ax_a.tick_params(axis="x", rotation=25, labelsize=7)
        ax_a.tick_params(axis="y", labelsize=7)
    else:
        _empty_panel(ax_a, "No feature-mode metrics")
    panel_label(ax_a, "A")

    # B — counts vs best manifold delta
    ax_b = fig.add_subplot(gs[0, 1])
    if not mvc.empty and "performance_difference" in mvc.columns:
        order = mvc["target_name"].tolist()
        vals = mvc["performance_difference"].astype(float).to_numpy()
        colors = ["#2ca02c" if v > 0 else "#d62728" if v < 0 else "0.6" for v in vals]
        y = np.arange(len(order))
        ax_b.barh(y, vals, color=colors, edgecolor="k", lw=0.3)
        ax_b.axvline(0, color="k", lw=0.8)
        ax_b.set_yticks(y)
        ax_b.set_yticklabels(order, fontsize=8)
        ax_b.set_xlabel("Manifold − counts (signed)")
        sns.despine(ax=ax_b)
    else:
        _empty_panel(ax_b, "No manifold_vs_counts_summary")
    panel_label(ax_b, "B")

    # C — region PCA best by target
    ax_c = fig.add_subplot(gs[1, 0])
    region = metrics[metrics["feature_type"] == "region_pca"]
    if not region.empty:
        rows = []
        for target, g in region.groupby("target_name"):
            metric = _metric_for_target(g, target)
            row = _best_row(g, metric)
            if row is not None:
                rows.append({"target": target, "value": float(row[metric]), "metric": metric})
        rdf = pd.DataFrame(rows)
        if not rdf.empty:
            sns.barplot(
                data=rdf, y="target", x="value", ax=ax_c,
                color=sns.color_palette("deep")[2], orient="h",
            )
            ax_c.set_ylabel("")
            ax_c.set_xlabel("Best region-PCA metric")
            ax_c.tick_params(axis="y", labelsize=7)
            sns.despine(ax=ax_c)
        else:
            _empty_panel(ax_c, "No region-PCA rows")
    else:
        _empty_panel(ax_c, "region_pca not run")
    panel_label(ax_c, "C")

    # D — explained variance by group
    ax_d = fig.add_subplot(gs[1, 1])
    if not ev.empty and "explained_variance_sum" in ev.columns:
        # Prefer region_pca grouping if present
        sub = ev.copy()
        if "feature_type" in sub.columns and (sub["feature_type"] == "region_pca").any():
            sub = sub[sub["feature_type"] == "region_pca"]
        gname = "group_name" if "group_name" in sub.columns else None
        if gname:
            summary = sub.groupby(gname)["explained_variance_sum"].mean().sort_values(ascending=False)
            ax_d.bar(summary.index.astype(str), summary.values, color=sns.color_palette("deep")[0], edgecolor="k", lw=0.3)
            ax_d.set_ylabel("Mean explained variance")
            ax_d.tick_params(axis="x", rotation=30, labelsize=7)
            sns.despine(ax=ax_d)
        else:
            _empty_panel(ax_d, "No group_name in EV CSV")
    else:
        _empty_panel(ax_d, "No explained-variance CSV")
    panel_label(ax_d, "D")

    return save_pub_figure(
        fig, out_dir / "fig_manifold_decoding.png", dpi=FIGURE_DPI,
        rect=(0.10, 0.08, 0.98, 0.94),
    )


# ---------------------------------------------------------------------------
# Fig 9 — Closed-loop realtime
# ---------------------------------------------------------------------------

def _pick_realtime_run(realtime_dir: Path) -> Path | None:
    """Prefer a sorted run with events; else any decoded_realtime.csv."""
    realtime_dir = Path(realtime_dir)
    if not realtime_dir.exists():
        return None
    candidates: list[Path] = []
    for decoded in sorted(realtime_dir.rglob("decoded_realtime.csv")):
        run = decoded.parent
        if "sorted" not in str(run):
            continue
        candidates.append(run)
    if not candidates:
        for decoded in sorted(realtime_dir.rglob("decoded_realtime.csv")):
            candidates.append(decoded.parent)
    if not candidates:
        return None
    # Prefer run with closed-loop events
    with_events = [c for c in candidates if (c / "closed_loop_events.csv").exists()
                   and (c / "closed_loop_events.csv").stat().st_size > 10]
    pool = with_events or candidates
    # Prefer spatial_context in path
    for c in pool:
        if "spatial_context" in c.name:
            return c
    return pool[0]


def plot_fig_closed_loop(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 9: position/error, confusion, trigger reliability."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_REALTIME
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _pick_realtime_run(experiment_dir / "realtime_decoding")
    if run_dir is None:
        return None

    decoded = _read_csv(run_dir / "decoded_realtime.csv")
    events = _read_csv(run_dir / "closed_loop_events.csv")
    if decoded.empty:
        return None

    fig = plt.figure(figsize=(12.0, 7.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.50)
    handles_rt: list = []
    labels_rt: list = []

    ax_a = fig.add_subplot(gs[0, 0])
    if {"true_x", "true_y", "decoded_x", "decoded_y"}.issubset(decoded.columns):
        ax_a.scatter(decoded["true_x"], decoded["true_y"], s=4, c="#4C72B0", alpha=0.45, label="True")
        ax_a.scatter(decoded["decoded_x"], decoded["decoded_y"], s=4, c="#DD8452", alpha=0.45, marker="x", label="Decoded")
        ax_a.set_xlabel("x (cm)")
        ax_a.set_ylabel("y (cm)")
        ax_a.set_aspect("equal", adjustable="box")
        handles_rt, labels_rt = ax_a.get_legend_handles_labels()
        if ax_a.get_legend() is not None:
            ax_a.get_legend().remove()
        sns.despine(ax=ax_a)
    else:
        _empty_panel(ax_a, "No position columns")
    panel_label(ax_a, "A")

    # B — position error over time + triggers
    ax_b = fig.add_subplot(gs[0, 1])
    if "position_error_cm" in decoded.columns and "time" in decoded.columns:
        ax_b.plot(decoded["time"], decoded["position_error_cm"], lw=0.6, color="0.45", alpha=0.8)
        if not events.empty and "time" in events.columns:
            y_max = float(np.nanmax(decoded["position_error_cm"]))
            band = y_max + 0.12 * max(y_max, 1.0)
            if "correct_trigger" in events.columns:
                ok = events[events["correct_trigger"] == True]  # noqa: E712
                bad = events[events["correct_trigger"] == False]  # noqa: E712
                h1 = ax_b.scatter(ok["time"], np.full(len(ok), band), c="#2ca02c", s=18, marker="|", label="Correct trigger")
                h2 = ax_b.scatter(bad["time"], np.full(len(bad), band), c="#d62728", s=18, marker="|", label="False trigger")
                handles_rt = list(handles_rt) + [h1, h2]
                labels_rt = list(labels_rt) + ["Correct trigger", "False trigger"]
            else:
                h = ax_b.scatter(events["time"], np.full(len(events), band), c="C1", s=18, marker="|", label="Trigger")
                handles_rt = list(handles_rt) + [h]
                labels_rt = list(labels_rt) + ["Trigger"]
            ax_b.set_ylim(0, band * 1.2)
        ax_b.set_xlabel("Time (s)")
        ax_b.set_ylabel("Position error (cm)")
        sns.despine(ax=ax_b)
    else:
        _empty_panel(ax_b, "No position_error_cm")
    panel_label(ax_b, "B")

    # C — spatial context confusion
    ax_c = fig.add_subplot(gs[1, 0])
    if {"true_spatial_context", "decoded_spatial_context"}.issubset(decoded.columns):
        y_true = decoded["true_spatial_context"].astype(str)
        y_pred = decoded["decoded_spatial_context"].astype(str)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            cmn, ax=ax_c, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"label": "Row-norm"},
        )
        ax_c.set_xlabel("Decoded")
        ax_c.set_ylabel("True")
        ax_c.tick_params(labelsize=7)
    else:
        _empty_panel(ax_c, "No spatial_context columns")
    panel_label(ax_c, "C")

    # D — trigger precision / reliability summary
    ax_d = fig.add_subplot(gs[1, 1])
    if not events.empty and "correct_trigger" in events.columns:
        n = len(events)
        n_ok = int(events["correct_trigger"].astype(bool).sum())
        precision = n_ok / max(n, 1)
        # Approximate recall if true labels present: fraction of true target epochs hit
        bars = {"Triggers": n, "Correct": n_ok, "Precision×N": precision * n}
        # Show precision as text + bar of correct vs incorrect
        counts = [n_ok, n - n_ok]
        ax_d.bar(["Correct", "Incorrect"], counts, color=["#2ca02c", "#d62728"], edgecolor="k", lw=0.3)
        ax_d.set_ylabel(f"Trigger count  (precision={precision:.2f}, n={n})")
        sns.despine(ax=ax_d)
        _ = bars
    else:
        _empty_panel(ax_d, "No closed-loop events")
    panel_label(ax_d, "D")

    if handles_rt:
        from visualization.publication_style import figure_legend_below
        figure_legend_below(fig, handles_rt, labels_rt, ncol=min(4, len(handles_rt)), fontsize=7, y=0.02)
    return save_pub_figure(
        fig, out_dir / "fig_closed_loop.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.12, 0.98, 0.94),
    )


# ---------------------------------------------------------------------------
# Fig 10 — Deployment selection
# ---------------------------------------------------------------------------

def plot_fig_deployment(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 10: winner summary + window×decoder heatmaps."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / "deployment_decoder_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

    best = load_best_by_target(experiment_dir)
    scores = load_window_scores(experiment_dir)
    if best.empty and scores.empty:
        return None

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # A — winners table-like bars
    ax_a = fig.add_subplot(gs[0, :])
    if not best.empty:
        name_col = "best_decoder_name" if "best_decoder_name" in best.columns else "decoder_name"
        feat_col = "best_feature_type" if "best_feature_type" in best.columns else None
        w_col = "recommended_realtime_window_s" if "recommended_realtime_window_s" in best.columns else "best_decode_window_s"
        val_col = "best_metric_value" if "best_metric_value" in best.columns else None
        labels = []
        for _, row in best.iterrows():
            feat = f" / {row[feat_col]}" if feat_col else ""
            w = f" @ {float(row[w_col]):.2f}s" if w_col in best.columns else ""
            labels.append(f"{row['target_name']}: {row[name_col]}{feat}{w}")
        if val_col:
            y = np.arange(len(best))
            ax_a.barh(y, best[val_col].astype(float), color=sns.color_palette("deep", n_colors=len(best)), edgecolor="k", lw=0.3)
            ax_a.set_yticks(y)
            ax_a.set_yticklabels(labels, fontsize=7)
            ax_a.set_xlabel("Best metric value")
            sns.despine(ax=ax_a)
        else:
            _empty_panel(ax_a, "No metric values")
    else:
        _empty_panel(ax_a, "No deployment winners")
    panel_label(ax_a, "A")

    # B / C — heatmaps for two continuous targets if scores present
    target_col = "target" if not scores.empty and "target" in scores.columns else "target_name"
    dec_col = "decoder" if not scores.empty and "decoder" in scores.columns else "decoder_name"
    win_col = "causal_window_s" if not scores.empty and "causal_window_s" in scores.columns else "decode_window_s"
    met_col = "metric_value" if not scores.empty and "metric_value" in scores.columns else None

    heat_targets = []
    if not scores.empty and target_col in scores.columns:
        for t in ("position", "spatial_context", "speed", "movement_state"):
            if t in set(scores[target_col].astype(str)):
                heat_targets.append(t)
            if len(heat_targets) >= 2:
                break
    axes_bc = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, target, lab in zip(axes_bc, heat_targets + [None, None], ("B", "C")):
        panel_label(ax, lab)
        if target is None or met_col is None:
            _empty_panel(ax, "No window scores")
            continue
        sub = scores[scores[target_col].astype(str) == target].copy()
        # Prefer realtime-compatible / counts if available
        if "feature_mode" in sub.columns:
            if (sub["feature_mode"] == "counts").any():
                sub = sub[sub["feature_mode"] == "counts"]
        if "realtime_compatible" in sub.columns:
            rt = sub[sub["realtime_compatible"] == True]  # noqa: E712
            if not rt.empty:
                sub = rt
        if sub.empty:
            _empty_panel(ax, f"No scores for {target}")
            continue
        pivot = sub.pivot_table(index=dec_col, columns=win_col, values=met_col, aggfunc="mean")
        sns.heatmap(pivot, ax=ax, cmap="viridis", annot=False, cbar_kws={"label": met_col})
        ax.set_xlabel("W (s)")
        ax.set_ylabel("Decoder")
        ax.tick_params(labelsize=6)

    # D unused if only 2 panels in bottom — use leftover for feature-mode note if needed
    # (already used B and C)

    return save_pub_figure(
        fig, out_dir / "fig_deployment.png", dpi=FIGURE_DPI,
        rect=(0.10, 0.08, 0.98, 0.94),
    )


# ---------------------------------------------------------------------------
# Fig 11 — Latency budget
# ---------------------------------------------------------------------------

def plot_fig_latency(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 11: 2×2 latency budget panels."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)

    lat_dir = experiment_dir / "latency_profiling"
    everything = _read_csv(lat_dir / "latency_everything.csv")
    feature = _read_csv(lat_dir / "feature_transform_latency.csv")
    teacher = _read_csv(lat_dir / "isomap_teacher_vs_distilled_latency.csv")
    stage = _read_csv(lat_dir / "realtime_stage_latency_combined.csv")
    if stage.empty:
        stage_paths = sorted(
            (experiment_dir / "realtime_decoding").rglob("latency/latency_by_stage.csv")
        ) if (experiment_dir / "realtime_decoding").exists() else []
        for p in stage_paths:
            if "sorted" in str(p):
                stage = _read_csv(p)
                break
        if stage.empty and stage_paths:
            stage = _read_csv(stage_paths[0])

    budget_ms = 50.0
    summary_path = lat_dir / "latency_benchmark_summary.json"
    if summary_path.exists():
        try:
            budget_ms = float(json.loads(summary_path.read_text()).get("update_budget_ms", budget_ms))
        except Exception:
            pass

    if everything.empty and feature.empty and teacher.empty and stage.empty:
        return None

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    def _barh_latency(ax, df: pd.DataFrame, name_col: str, mean_col: str = "mean_ms") -> None:
        if df.empty or name_col not in df.columns or mean_col not in df.columns:
            _empty_panel(ax, "No data")
            return
        sub = df.sort_values(mean_col, ascending=True)
        y = np.arange(len(sub))
        colors = []
        for _, r in sub.iterrows():
            if "realtime_compatible" in sub.columns:
                colors.append("#2ca02c" if bool(r["realtime_compatible"]) else "#d62728")
            else:
                colors.append(sns.color_palette("deep")[0])
        ax.barh(y, sub[mean_col], color=colors, edgecolor="k", lw=0.3)
        ax.axvline(budget_ms, color="0.3", ls="--", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(sub[name_col].astype(str).tolist(), fontsize=7)
        ax.set_xlabel("Latency (ms)")
        sns.despine(ax=ax)

    ax_a = fig.add_subplot(gs[0, 0])
    feat_df = feature if not feature.empty else (
        everything[everything["category"] == "feature_transform"] if not everything.empty else pd.DataFrame()
    )
    _barh_latency(ax_a, feat_df, "name" if "name" in feat_df.columns else "method")
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    if not stage.empty:
        name_col = "stage" if "stage" in stage.columns else "name"
        # Aggregate mean across sources if needed
        if "source" in stage.columns and name_col in stage.columns:
            agg = stage.groupby(name_col, as_index=False)["mean_ms"].mean()
            if "realtime_compatible" in stage.columns:
                rt = stage.groupby(name_col)["realtime_compatible"].first().reset_index()
                agg = agg.merge(rt, on=name_col, how="left")
            _barh_latency(ax_b, agg, name_col)
        else:
            _barh_latency(ax_b, stage, name_col)
    else:
        _empty_panel(ax_b, "No realtime stage latency")
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    if not teacher.empty:
        name_col = "method" if "method" in teacher.columns else "name"
        _barh_latency(ax_c, teacher, name_col)
    else:
        iso = everything[everything["category"] == "isomap_compare"] if not everything.empty else pd.DataFrame()
        if not iso.empty:
            _barh_latency(ax_c, iso, "name")
        else:
            _empty_panel(ax_c, "Isomap teacher/distilled not profiled")
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    if not everything.empty:
        # Budget compliance summary
        sub = everything.copy()
        if "mean_ms" in sub.columns:
            sub = sub.sort_values("mean_ms", ascending=False).head(12)
            _barh_latency(ax_d, sub, "name" if "name" in sub.columns else "method")
        else:
            _empty_panel(ax_d, "No combined latency table")
    else:
        _empty_panel(ax_d, "No latency_everything.csv")
    panel_label(ax_d, "D")

    return save_pub_figure(
        fig, out_dir / "fig_latency.png", dpi=FIGURE_DPI,
        rect=(0.12, 0.08, 0.98, 0.94),
    )


# ---------------------------------------------------------------------------
# Fig 12 (suppl.) — Temporal W×L
# ---------------------------------------------------------------------------

def plot_fig_temporal_wl(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 12: compress temporal W×L heatmaps into one multi-panel figure."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    temporal_dir = experiment_dir / "decoding"
    csvs = list(temporal_dir.rglob("all_configurations.csv")) if temporal_dir.exists() else []
    if not csvs:
        return None

    df = pd.concat([_read_csv(p) for p in csvs], ignore_index=True)
    if df.empty:
        return None

    out_dir = figures_dir / FIGURE_SUBDIR_TEMPORAL
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick up to 4 target/model slices
    latent = df[df["temporal_model"].isin(["flattened_history", "static_latent"])].copy() if "temporal_model" in df.columns else df
    groups = list(latent.groupby(["target", "representation", "temporal_model"]))[:4]
    if not groups:
        return None

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)
    labels = "ABCD"
    for i, ((target, rep, model), sub) in enumerate(groups):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        metric = str(sub["validation_metric"].iloc[0]) if "validation_metric" in sub.columns else "metric"
        agg = "min" if "error" in metric.lower() else "max"
        pivot = sub.pivot_table(
            index="integration_window_s",
            columns="latent_history_frames",
            values="validation_metric_value",
            aggfunc=agg,
        )
        sns.heatmap(pivot, ax=ax, cmap="viridis", cbar_kws={"label": metric})
        ax.set_xlabel("L (frames)")
        ax.set_ylabel("W (s)")
        panel_label(ax, labels[i])

    for j in range(len(groups), 4):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        ax.axis("off")

    return save_pub_figure(
        fig, out_dir / "fig_temporal_wl.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.08, 0.98, 0.94),
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_publication_decoding_figures(
    experiment_dir: Path,
    figures_dir: Path | None = None,
    *,
    cleanup_legacy: bool = True,
) -> list[Path]:
    """Write publication decoder/realtime/deployment/latency figures."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for fn in (
        plot_fig_decoding_performance,
        plot_fig_manifold_decoding,
        plot_fig_closed_loop,
        plot_fig_deployment,
        plot_fig_latency,
        plot_fig_temporal_wl,
    ):
        try:
            path = fn(experiment_dir, figures_dir)
            if path is not None:
                written.append(path)
                print(f"  wrote {path.relative_to(figures_dir)}")
        except Exception as exc:
            print(f"  warning: {fn.__name__} skipped ({exc})")

    # Isomap publication suite
    try:
        from visualization.publication_isomap_plots import generate_publication_isomap_figures

        written.extend(
            generate_publication_isomap_figures(
                experiment_dir, figures_dir, cleanup_legacy=False,
            )
        )
    except Exception as exc:
        print(f"  warning: isomap publication figures skipped ({exc})")

    if cleanup_legacy:
        for sub in (
            FIGURE_SUBDIR_DECODER,
            FIGURE_SUBDIR_REALTIME,
            "deployment_decoder_selection",
            "latency",
            FIGURE_SUBDIR_TEMPORAL,
        ):
            _cleanup_legacy_pngs(figures_dir / sub)

    return written
