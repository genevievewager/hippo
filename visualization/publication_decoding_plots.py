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
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from sklearn.metrics import confusion_matrix

from visualization.constants import (
    FIGURE_DPI,
    FIGURE_SUBDIR_DECODER,
    FIGURE_SUBDIR_REALTIME,
    FIGURE_SUBDIR_TEMPORAL,
)
from realtime.decoder_comparison import PRIMARY_METRIC
from realtime.decoder_models import (
    categorical_model_names,
    continuous_model_names,
)
from visualization.publication_style import (
    apply_publication_theme,
    expand_xlim_for_labels,
    legend_below,
    legend_outside,
    panel_label,
    save_pub_figure,
)

apply_publication_theme()

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


def _scores_missing_manifold_modes(scores: pd.DataFrame, metrics: pd.DataFrame) -> bool:
    """True when window scores collapsed every mode to counts/rates but metrics have more."""
    if scores.empty or metrics.empty or "feature_mode" not in scores.columns:
        return False
    score_modes = {str(m) for m in scores["feature_mode"].dropna().unique()}
    if not score_modes <= {"counts", "rates"}:
        return False
    if "feature_mode" in metrics.columns:
        metric_modes = {str(m) for m in metrics["feature_mode"].dropna().unique()}
    elif "feature_type" in metrics.columns:
        metric_modes = {str(m) for m in metrics["feature_type"].dropna().unique()}
    else:
        return False
    return len(metric_modes - {"counts", "rates"}) > 0


def _load_window_scores_with_modes(experiment_dir: Path) -> pd.DataFrame:
    """Load window scores, repairing older CSVs that dropped composite feature modes."""
    scores = load_window_scores(experiment_dir)
    metrics = load_comparison_metrics(experiment_dir)
    if scores.empty or _scores_missing_manifold_modes(scores, metrics):
        rebuilt = _window_scores_from_metrics(metrics)
        if not rebuilt.empty:
            return rebuilt
    return scores


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
                    va="center", ha="left", fontsize=8, color="0.25",
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
        figure_legend_below(fig, all_h, all_l, ncol=min(5, len(all_h)), y=0.02)
    return save_pub_figure(
        fig, out_dir / "fig_decoding_performance.png", dpi=FIGURE_DPI,
        rect=(0.10, 0.16, 0.96, 0.92),
    )


# ---------------------------------------------------------------------------
# Fig 5 — Manifold vs spike counts
# ---------------------------------------------------------------------------

def _short_label(name: str, max_len: int = 14) -> str:
    s = str(name)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _load_registry_winners(experiment_dir: Path) -> dict[str, dict]:
    experiment_dir = Path(experiment_dir)
    for path in (
        experiment_dir / "models" / "best_realtime_decoders.json",
        experiment_dir / "deployment_decoder_selection" / "best_realtime_decoders.json",
    ):
        if not path.exists():
            continue
        try:
            with open(path) as f:
                registry = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        targets = registry.get("targets") if isinstance(registry, dict) else None
        if isinstance(targets, dict):
            return {str(t): dict(cfg) for t, cfg in targets.items()}
    return {}


_PREFERRED_TARGETS = (
    "position", "speed", "acceleration", "head_direction",
    "distance_to_wall", "spatial_context", "movement_state", "wall_distance_bin",
)
_PREFERRED_FEATURES = (
    "counts", "global_pca", "region_pca", "layer_pca",
    "global_isomap", "global_isomap_distilled",
)


def _ordered_targets(available: list[str] | set[str]) -> list[str]:
    avail = set(available)
    return [t for t in _PREFERRED_TARGETS if t in avail] + sorted(
        t for t in avail if t not in _PREFERRED_TARGETS
    )


def _ordered_features(available: list[str] | set[str]) -> list[str]:
    avail = set(available)
    return [f for f in _PREFERRED_FEATURES if f in avail] + sorted(
        f for f in avail if f not in _PREFERRED_FEATURES
    )


def _display_limits(display: np.ndarray) -> tuple[float, float]:
    finite = display[np.isfinite(display)]
    if not finite.size:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, 5))
    vmax = float(np.nanpercentile(finite, 95))
    if vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite) + 1e-9)
    return vmin, vmax


def _draw_feature_x_window_panel(
    ax,
    sub: pd.DataFrame,
    target: str,
    winners: dict[str, dict],
    *,
    cmap: str = "YlGn",
) -> None:
    """Feature × window heatmap; cell text = best decoder."""
    if sub.empty:
        ax.axis("off")
        return
    higher = bool(sub["higher_is_better"].iloc[0])
    metric = str(sub["metric_name"].iloc[0])
    features = _ordered_features(sub["feature_mode"].astype(str).unique().tolist())
    windows = sorted(float(w) for w in sub["causal_window_s"].unique())
    mat = np.full((len(features), len(windows)), np.nan)
    dec_ann = [["" for _ in windows] for _ in features]
    rt_mat = np.ones((len(features), len(windows)), dtype=bool)
    for (feat, w), g in sub.groupby(["feature_mode", "causal_window_s"], sort=False):
        fi = features.index(str(feat))
        wi = windows.index(float(w))
        idx = g["metric_value"].idxmax() if higher else g["metric_value"].idxmin()
        row = g.loc[idx]
        mat[fi, wi] = float(row["metric_value"])
        dec_ann[fi][wi] = _short_label(row["decoder"], 10)
        if "realtime_compatible" in row.index:
            rt_mat[fi, wi] = bool(row["realtime_compatible"])

    display = mat if higher else -mat
    vmin, vmax = _display_limits(display)
    ax.imshow(display, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for fi in range(len(features)):
        for wi in range(len(windows)):
            if not rt_mat[fi, wi] and np.isfinite(mat[fi, wi]):
                ax.add_patch(
                    plt.Rectangle(
                        (wi - 0.5, fi - 0.5), 1, 1,
                        fill=False, hatch="////", edgecolor="0.3", linewidth=0.4,
                    )
                )
            if np.isfinite(mat[fi, wi]):
                ax.text(
                    wi, fi,
                    f"{mat[fi, wi]:.3g}\n{dec_ann[fi][wi]}",
                    ha="center", va="center", fontsize=5.2,
                )
    if target in winners:
        wcfg = winners[target]
        wf = str(wcfg.get("selected_feature_mode", ""))
        ww = float(wcfg.get("selected_causal_window_s", float("nan")))
        if wf in features and ww in windows:
            ax.add_patch(
                plt.Rectangle(
                    (windows.index(ww) - 0.5, features.index(wf) - 0.5),
                    1, 1, fill=False, edgecolor="#d4a017", linewidth=2.2,
                )
            )
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([f"{w:.2g}" for w in windows], fontsize=7)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([_short_label(f, 14) for f in features], fontsize=7)
    ax.set_xlabel("W (s)", fontsize=8)
    ax.set_title(f"{target} ({metric})", fontsize=8)


def _draw_decoder_x_window_panel(
    ax,
    sub: pd.DataFrame,
    target: str,
    winners: dict[str, dict],
    *,
    cmap: str = "Blues",
) -> None:
    """Decoder × window heatmap; cell text = best realtime feature."""
    if sub.empty:
        ax.axis("off")
        return
    higher = bool(sub["higher_is_better"].iloc[0])
    metric = str(sub["metric_name"].iloc[0])
    decoders = sorted(sub["decoder"].astype(str).unique())
    windows = sorted(float(w) for w in sub["causal_window_s"].unique())
    mat = np.full((len(decoders), len(windows)), np.nan)
    feat_ann = [["" for _ in windows] for _ in decoders]
    for (dec, w), g in sub.groupby(["decoder", "causal_window_s"], sort=False):
        di = decoders.index(str(dec))
        wi = windows.index(float(w))
        idx = g["metric_value"].idxmax() if higher else g["metric_value"].idxmin()
        row = g.loc[idx]
        mat[di, wi] = float(row["metric_value"])
        feat_ann[di][wi] = _short_label(row["feature_mode"], 10)

    display = mat if higher else -mat
    vmin, vmax = _display_limits(display)
    ax.imshow(display, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for di in range(len(decoders)):
        for wi in range(len(windows)):
            if np.isfinite(mat[di, wi]):
                ax.text(
                    wi, di,
                    f"{mat[di, wi]:.3g}\n{feat_ann[di][wi]}",
                    ha="center", va="center", fontsize=5.2,
                )
    if target in winners:
        wcfg = winners[target]
        wd = str(wcfg.get("selected_decoder", ""))
        ww = float(wcfg.get("selected_causal_window_s", float("nan")))
        if wd in decoders and ww in windows:
            ax.add_patch(
                plt.Rectangle(
                    (windows.index(ww) - 0.5, decoders.index(wd) - 0.5),
                    1, 1, fill=False, edgecolor="#d4a017", linewidth=2.2,
                )
            )
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([f"{w:.2g}" for w in windows], fontsize=7)
    ax.set_yticks(np.arange(len(decoders)))
    ax.set_yticklabels([_short_label(d, 14) for d in decoders], fontsize=7)
    ax.set_xlabel("W (s)", fontsize=8)
    ax.set_title(f"{target} ({metric})", fontsize=8)


def _manifold_window_decoding_context(
    experiment_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict], list[str]] | None:
    """Load sorted window scores, realtime subset, registry winners, and targets."""
    scores = _load_window_scores_with_modes(experiment_dir)
    if scores.empty:
        return None

    if "spike_source" in scores.columns:
        sorted_scores = scores[scores["spike_source"].astype(str) == "sorted"]
        if not sorted_scores.empty:
            scores = sorted_scores

    winners = _load_registry_winners(experiment_dir)
    targets = _ordered_targets(
        list(winners.keys()) if winners else scores["target"].astype(str).unique().tolist()
    )
    targets = [t for t in targets if t in set(scores["target"].astype(str))]
    if not targets:
        return None

    rt_scores = scores
    if "realtime_compatible" in scores.columns:
        rt_only = scores[scores["realtime_compatible"] == True].copy()  # noqa: E712
        if not rt_only.empty:
            rt_scores = rt_only

    return scores, rt_scores, winners, targets


_WINDOW_GRID_ROWS = 4
_WINDOW_GRID_COLS = 2


def _plot_window_decoding_grid(
    *,
    targets: list[str],
    scores: pd.DataFrame,
    winners: dict[str, dict],
    draw_panel,
    title: str,
    subtitle: str,
    out_path: Path,
    cmap: str,
) -> Path:
    """One heatmap per target in a fixed 4×2 grid."""
    rows, cols = _WINDOW_GRID_ROWS, _WINDOW_GRID_COLS
    fig = plt.figure(figsize=(10, 2.35 * rows + 0.9))
    gs = GridSpec(rows, cols, figure=fig, hspace=0.55, wspace=0.35)

    for i, target in enumerate(targets):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        draw_panel(
            ax,
            scores[scores["target"].astype(str) == target],
            target,
            winners,
            cmap=cmap,
        )
    for j in range(len(targets), rows * cols):
        r, c = divmod(j, cols)
        fig.add_subplot(gs[r, c]).axis("off")

    fig.suptitle(title, fontsize=12, y=0.995)
    fig.text(0.5, 0.965, subtitle, ha="center", va="top", fontsize=10)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.04)
    fig.savefig(out_path, dpi=FIGURE_DPI)
    plt.close(fig)
    return out_path


def _cleanup_retired_window_decoding_pages(out_dir: Path) -> None:
    for stale in (
        out_dir / "fig_manifold_decoding.png",
        out_dir / "fig_manifold_decoder_window_threeway.png",
        out_dir / "fig_deployable_decoder_x_window_heatmaps.png",
    ):
        stale.unlink(missing_ok=True)


def plot_fig_feature_x_window(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 5a: per-target feature × causal-window heatmaps (4×2 grid).

    Each cell shows the metric and best decoder at that (feature, W); hatching
    marks offline-only features; gold outline marks the registry selection.
    """
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = _manifold_window_decoding_context(experiment_dir)
    if ctx is None:
        return None
    scores, _, winners, targets = ctx

    out_path = _plot_window_decoding_grid(
        targets=targets,
        scores=scores,
        winners=winners,
        draw_panel=_draw_feature_x_window_panel,
        title="Feature × window  (sorted spikes)",
        subtitle=(
            "Cell = best decoder @ that W; hatch = offline-only; gold = selected (feature, W)"
        ),
        out_path=out_dir / "fig_feature_x_window.png",
        cmap="YlGn",
    )
    _cleanup_retired_window_decoding_pages(out_dir)
    return out_path


def plot_fig_decoder_x_window(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 5b: per-target decoder × causal-window heatmaps (4×2 grid).

    Each cell shows the best realtime-compatible feature at that (decoder, W);
    gold outline marks the registry selection.
    """
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    ctx = _manifold_window_decoding_context(experiment_dir)
    if ctx is None:
        return None
    _, rt_scores, winners, targets = ctx

    out_path = _plot_window_decoding_grid(
        targets=targets,
        scores=rt_scores,
        winners=winners,
        draw_panel=_draw_decoder_x_window_panel,
        title="Decoder × window  (sorted spikes, realtime-compatible)",
        subtitle=(
            "Cell = best realtime feature @ that W; gold = selected (decoder, W)"
        ),
        out_path=out_dir / "fig_decoder_x_window.png",
        cmap="Blues",
    )
    _cleanup_retired_window_decoding_pages(out_dir)
    return out_path


def plot_fig_manifold_decoding(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Write feature×window and decoder×window as separate 4×2 figures."""
    plot_fig_decoder_x_window(experiment_dir, figures_dir)
    return plot_fig_feature_x_window(experiment_dir, figures_dir)


def _draw_fixed_decoder_feature_x_window_panel(
    ax,
    sub: pd.DataFrame,
    target: str,
    decoder: str,
    winners: dict[str, dict],
    *,
    cmap: str = "YlGn",
) -> None:
    """Feature × window heatmap for one fixed decoder (no decoder collapse)."""
    if sub.empty:
        ax.axis("off")
        return
    sub = sub[sub["decoder"].astype(str) == decoder]
    if sub.empty:
        ax.axis("off")
        ax.set_title(f"{target}\n(no data)", fontsize=8)
        return
    higher = bool(sub["higher_is_better"].iloc[0])
    metric = str(sub["metric_name"].iloc[0])
    features = _ordered_features(sub["feature_mode"].astype(str).unique().tolist())
    windows = sorted(float(w) for w in sub["causal_window_s"].unique())
    mat = np.full((len(features), len(windows)), np.nan)
    rt_mat = np.ones((len(features), len(windows)), dtype=bool)
    for (feat, w), g in sub.groupby(["feature_mode", "causal_window_s"], sort=False):
        fi = features.index(str(feat))
        wi = windows.index(float(w))
        row = g.iloc[0]
        if len(g) > 1:
            idx = g["metric_value"].idxmax() if higher else g["metric_value"].idxmin()
            row = g.loc[idx]
        mat[fi, wi] = float(row["metric_value"])
        if "realtime_compatible" in row.index:
            rt_mat[fi, wi] = bool(row["realtime_compatible"])

    display = mat if higher else -mat
    vmin, vmax = _display_limits(display)
    ax.imshow(display, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for fi in range(len(features)):
        for wi in range(len(windows)):
            if not rt_mat[fi, wi] and np.isfinite(mat[fi, wi]):
                ax.add_patch(
                    plt.Rectangle(
                        (wi - 0.5, fi - 0.5), 1, 1,
                        fill=False, hatch="////", edgecolor="0.3", linewidth=0.4,
                    )
                )
            if np.isfinite(mat[fi, wi]):
                ax.text(
                    wi, fi, f"{mat[fi, wi]:.3g}",
                    ha="center", va="center", fontsize=5.5,
                )
    if target in winners:
        wcfg = winners[target]
        wf = str(wcfg.get("selected_feature_mode", ""))
        ww = float(wcfg.get("selected_causal_window_s", float("nan")))
        wd = str(wcfg.get("selected_decoder", ""))
        if wd == decoder and wf in features and ww in windows:
            ax.add_patch(
                plt.Rectangle(
                    (windows.index(ww) - 0.5, features.index(wf) - 0.5),
                    1, 1, fill=False, edgecolor="#d4a017", linewidth=2.2,
                )
            )
    ax.set_xticks(np.arange(len(windows)))
    ax.set_xticklabels([f"{w:.2g}" for w in windows], fontsize=6)
    ax.set_yticks(np.arange(len(features)))
    ax.set_yticklabels([_short_label(f, 12) for f in features], fontsize=6)
    ax.set_xlabel("W (s)", fontsize=7)
    ax.set_title(f"{target} ({metric})", fontsize=8)


def _ordered_decoders_for_family(
    scores: pd.DataFrame,
    targets: tuple[str, ...],
    *,
    max_models: str = "quick",
) -> list[str]:
    """Decoders present in scores for a target family, in zoo order."""
    sub = scores[scores["target"].astype(str).isin(targets)]
    present = {str(d) for d in sub["decoder"].dropna().unique()}
    if not present:
        return []
    is_continuous = targets[0] in CONTINUOUS_TARGETS
    preferred = (
        continuous_model_names(max_models)
        if is_continuous
        else categorical_model_names(max_models)
    )
    ordered = [d for d in preferred if d in present]
    ordered += sorted(d for d in present if d not in ordered)
    return ordered


def _plot_decoder_family_feature_x_window(
    *,
    experiment_dir: Path,
    figures_dir: Path,
    family_label: str,
    targets: tuple[str, ...],
    out_name: str,
    cmap: str = "YlGn",
) -> Path | None:
    """One page: stacked decoder sections, each with feature×W panels per target."""
    ctx = _manifold_window_decoding_context(experiment_dir)
    if ctx is None:
        return None
    scores, _, winners, _all_targets = ctx
    family_targets = [t for t in _ordered_targets(targets) if t in set(scores["target"].astype(str))]
    if not family_targets:
        return None
    decoders = _ordered_decoders_for_family(scores, targets)
    if not decoders:
        return None

    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    n_targets = len(family_targets)
    target_cols = min(3, n_targets)
    target_rows = int(np.ceil(n_targets / target_cols))
    section_h = 0.35 + target_rows * 2.05
    fig_h = 0.7 + len(decoders) * section_h
    fig = plt.figure(figsize=(3.2 * target_cols + 0.8, fig_h))
    y_cursor = 0.98
    section_gap = 0.02

    for decoder in decoders:
        band_h = section_h / fig_h
        y_top = y_cursor
        y_bottom = y_cursor - band_h + section_gap
        gs = GridSpec(
            target_rows, target_cols, figure=fig,
            left=0.07, right=0.98,
            top=y_top - 0.04, bottom=y_bottom + 0.02,
            hspace=0.45, wspace=0.32,
        )
        fig.text(
            0.5, y_top - 0.01,
            _short_label(decoder, 40),
            ha="center", va="top", fontsize=11, fontweight="bold", color="#1a237e",
        )
        for i, target in enumerate(family_targets):
            r, c = divmod(i, target_cols)
            ax = fig.add_subplot(gs[r, c])
            _draw_fixed_decoder_feature_x_window_panel(
                ax,
                scores[scores["target"].astype(str) == target],
                target,
                decoder,
                winners,
                cmap=cmap,
            )
        for j in range(n_targets, target_rows * target_cols):
            r, c = divmod(j, target_cols)
            fig.add_subplot(gs[r, c]).axis("off")
        y_cursor = y_bottom - section_gap

    fig.suptitle(
        f"{family_label} decoders — feature × window  (sorted spikes)",
        fontsize=12, y=0.995,
    )
    fig.text(
        0.5, 0.01,
        "Each panel: feature × W at fixed decoder; hatch = offline-only; "
        "gold = deployable (feature, W) when this decoder is selected",
        ha="center", va="bottom", fontsize=9, color="0.35",
    )
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig_continuous_decoders_feature_x_window(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Continuous decoder suite: one section per decoder, panels = feature × W."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    return _plot_decoder_family_feature_x_window(
        experiment_dir=experiment_dir,
        figures_dir=figures_dir,
        family_label="Continuous",
        targets=CONTINUOUS_TARGETS,
        out_name="fig_continuous_decoders_feature_x_window.png",
        cmap="YlGn",
    )


def plot_fig_categorical_decoders_feature_x_window(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Categorical decoder suite: one section per decoder, panels = feature × W."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    return _plot_decoder_family_feature_x_window(
        experiment_dir=experiment_dir,
        figures_dir=figures_dir,
        family_label="Categorical",
        targets=CATEGORICAL_TARGETS,
        out_name="fig_categorical_decoders_feature_x_window.png",
        cmap="Blues",
    )


def _best_per_window_over_configs(
    metrics: pd.DataFrame,
    target: str,
    metric: str,
    *,
    feature_mode: str | None = None,
) -> pd.DataFrame:
    """Best held-out score at each W (optionally filtered to one feature mode)."""
    sub = metrics[metrics["target_name"].astype(str) == target].copy()
    if feature_mode is not None:
        mode_col = "feature_mode" if "feature_mode" in sub.columns else "feature_type"
        if mode_col in sub.columns:
            sub = sub[sub[mode_col].astype(str) == feature_mode]
    if sub.empty or metric not in sub.columns:
        return pd.DataFrame()
    rows = []
    for w, g in sub.groupby("decode_window_s"):
        row = _best_row(g, metric)
        if row is not None:
            rows.append({
                "decode_window_s": float(w),
                "metric_value": float(row[metric]),
                "decoder_name": str(row.get("decoder_name", "")),
                "feature_mode": str(
                    row.get("feature_mode") or row.get("feature_type", "counts")
                ),
            })
    return pd.DataFrame(rows).sort_values("decode_window_s")


def _metric_label(metric: str) -> str:
    labels = {
        "mean_position_error_cm": "Position error (cm, ↓)",
        "r2": "R² (↑)",
        "mean_circular_error_deg": "Circular error (deg, ↓)",
        "balanced_accuracy": "Balanced accuracy (↑)",
    }
    return labels.get(metric, metric)


def plot_fig_window_selection_story(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """How causal W is chosen, how manifolds enter, and how outputs are scored."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir)
    best = load_best_by_target(experiment_dir)
    winners = _load_registry_winners(experiment_dir)
    if metrics.empty and best.empty:
        return None
    if "spike_source" in metrics.columns:
        sorted_m = metrics[metrics["spike_source"].astype(str) == "sorted"]
        if not sorted_m.empty:
            metrics = sorted_m

    fig = plt.figure(figsize=(13.5, 9.2))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.9, 1.15, 1.15], hspace=0.42, wspace=0.34)

    # A — scoring reference
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.axis("off")
    table_rows = [["Target", "Primary metric", "Direction"]]
    for target in _ordered_targets(list(PRIMARY_METRIC.keys())):
        metric, direction = PRIMARY_METRIC[target]
        table_rows.append([
            target,
            metric,
            "lower" if direction == "lower" else "higher",
        ])
    tab = ax_a.table(
        cellText=table_rows[1:],
        colLabels=table_rows[0],
        loc="center",
        cellLoc="left",
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1.0, 1.25)
    ax_a.set_title(
        "Held-out scoring (test split only; no future spikes/behavior)",
        fontsize=10, pad=8,
    )
    panel_label(ax_a, "A", x=-0.08, y=1.05)

    # B — causal pipeline
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.axis("off")
    pipeline = (
        "Causal decode pipeline (each update at t)\n\n"
        "1. Spike counts  x_t(W)  in [t−W, t)\n"
        "2. Encoder E(·) fit on train → frozen at test\n"
        "     (counts / PCA / region PCA / … → z_t)\n"
        "3. Decoder D(·) fit on train latents → frozen\n"
        "4. Predict ŷ_t; score vs held-out behavior\n\n"
        "W is searched jointly with (feature, decoder).\n"
        "Selection policy (default): shortest_near_optimal\n"
        "  → shortest W within 5% of best metric\n"
        "     (≤1.05× best error, or ≥0.95× best score)"
    )
    ax_b.text(0.02, 0.98, pipeline, va="top", ha="left", fontsize=9, family="monospace")
    ax_b.set_title("Pipeline & window policy", fontsize=10, pad=8)
    panel_label(ax_b, "B", x=-0.08, y=1.05)

    # C — continuous: metric vs W with near-optimal band
    ax_c = fig.add_subplot(gs[1, 0])
    example_cont = "position"
    if example_cont in set(metrics.get("target_name", pd.Series(dtype=str)).astype(str)):
        metric = PRIMARY_METRIC[example_cont][0]
        direction = PRIMARY_METRIC[example_cont][1]
        colors = {"counts": "C0", "global_pca": "C1", "region_pca": "C2"}
        for feat, color in colors.items():
            curve = _best_per_window_over_configs(metrics, example_cont, metric, feature_mode=feat)
            if curve.empty:
                continue
            ax_c.plot(
                curve["decode_window_s"], curve["metric_value"],
                marker="o", ms=4, lw=1.4, color=color, label=feat,
            )
        overall = _best_per_window_over_configs(metrics, example_cont, metric)
        if not overall.empty:
            if direction == "lower":
                best_val = float(overall["metric_value"].min())
                ax_c.axhspan(best_val, 1.05 * best_val, color="#ffd54f", alpha=0.25, zorder=0)
            else:
                best_val = float(overall["metric_value"].max())
                ax_c.axhspan(0.95 * best_val, best_val, color="#ffd54f", alpha=0.25, zorder=0)
            best_w = float(overall.loc[overall["metric_value"].idxmin() if direction == "lower"
                                      else overall["metric_value"].idxmax(), "decode_window_s"])
            ax_c.axvline(best_w, color="#c62828", ls="--", lw=1.2, label=f"best W={best_w:.2g}s")
        if not best.empty:
            brow = best[best["target_name"] == example_cont]
            if not brow.empty:
                rec_w = float(brow.iloc[0].get("recommended_realtime_window_s", np.nan))
                if np.isfinite(rec_w):
                    ax_c.axvline(rec_w, color="#d4a017", ls="-", lw=2.0, label=f"selected W={rec_w:.2g}s")
        ax_c.set_xlabel("Causal window W (s)")
        ax_c.set_ylabel(_metric_label(metric))
        ax_c.set_title(f"Example: {example_cont} — score vs W by feature mode")
        ax_c.legend(fontsize=7, loc="best")
        sns.despine(ax=ax_c)
    else:
        _empty_panel(ax_c, "No continuous-target metrics")
    panel_label(ax_c, "C")

    # D — categorical example
    ax_d = fig.add_subplot(gs[1, 1])
    example_cat = "spatial_context"
    if example_cat in set(metrics.get("target_name", pd.Series(dtype=str)).astype(str)):
        metric = PRIMARY_METRIC[example_cat][0]
        direction = PRIMARY_METRIC[example_cat][1]
        for feat, color in (("counts", "C0"), ("global_pca", "C1"), ("region_pca", "C2")):
            curve = _best_per_window_over_configs(metrics, example_cat, metric, feature_mode=feat)
            if curve.empty:
                continue
            ax_d.plot(
                curve["decode_window_s"], curve["metric_value"],
                marker="o", ms=4, lw=1.4, color=color, label=feat,
            )
        overall = _best_per_window_over_configs(metrics, example_cat, metric)
        if not overall.empty:
            if direction == "lower":
                best_val = float(overall["metric_value"].min())
                ax_d.axhspan(best_val, 1.05 * best_val, color="#ffd54f", alpha=0.25, zorder=0)
            else:
                best_val = float(overall["metric_value"].max())
                ax_d.axhspan(0.95 * best_val, best_val, color="#ffd54f", alpha=0.25, zorder=0)
            best_w = float(overall.loc[overall["metric_value"].idxmax(), "decode_window_s"])
            ax_d.axvline(best_w, color="#c62828", ls="--", lw=1.2, label=f"best W={best_w:.2g}s")
        if not best.empty:
            brow = best[best["target_name"] == example_cat]
            if not brow.empty:
                rec_w = float(brow.iloc[0].get("recommended_realtime_window_s", np.nan))
                if np.isfinite(rec_w):
                    ax_d.axvline(rec_w, color="#d4a017", ls="-", lw=2.0, label=f"selected W={rec_w:.2g}s")
        ax_d.set_xlabel("Causal window W (s)")
        ax_d.set_ylabel(_metric_label(metric))
        ax_d.set_ylim(0, 1.05)
        ax_d.set_title(f"Example: {example_cat} — score vs W by feature mode")
        ax_d.legend(fontsize=7, loc="best")
        sns.despine(ax=ax_d)
    else:
        _empty_panel(ax_d, "No categorical-target metrics")
    panel_label(ax_d, "D")

    # E — all targets: best vs selected W
    ax_e = fig.add_subplot(gs[2, 0])
    if not best.empty:
        plot_df = best.copy()
        if "target_name" not in plot_df.columns:
            _empty_panel(ax_e, "No best_decoder_by_target")
        else:
            order = _ordered_targets(plot_df["target_name"].astype(str).tolist())
            plot_df = plot_df.set_index("target_name").loc[order].reset_index()
            y = np.arange(len(order))
            best_w = plot_df["best_decode_window_s"].astype(float).to_numpy()
            rec_w = plot_df.get(
                "recommended_realtime_window_s", plot_df["best_decode_window_s"]
            ).astype(float).to_numpy()
            ax_e.scatter(best_w, y, color="#c62828", s=36, zorder=3, label="Best-accuracy W")
            ax_e.scatter(rec_w, y, color="#d4a017", s=48, marker="|", linewidths=2.5, zorder=4,
                         label="Selected W (shortest near-optimal)")
            for yi, (bw, rw) in enumerate(zip(best_w, rec_w)):
                if abs(bw - rw) > 1e-6:
                    ax_e.plot([bw, rw], [yi, yi], color="0.6", lw=1.0, zorder=2)
            ax_e.set_yticks(y)
            ax_e.set_yticklabels(order, fontsize=8)
            ax_e.set_xlabel("Causal window W (s)")
            ax_e.set_title("Window choice per target")
            ax_e.legend(fontsize=7, loc="lower right")
            sns.despine(ax=ax_e)
    else:
        _empty_panel(ax_e, "No best_decoder_by_target")
    panel_label(ax_e, "E")

    # F — manifold role at selected W
    ax_f = fig.add_subplot(gs[2, 1])
    modes = ("counts", "global_pca", "region_pca")
    bar_rows: list[dict[str, float | str]] = []
    if not best.empty and not metrics.empty:
        mode_col = "feature_mode" if "feature_mode" in metrics.columns else "feature_type"
        for target in _ordered_targets(best["target_name"].astype(str).tolist()):
            brow = best[best["target_name"] == target]
            if brow.empty:
                continue
            rec_w = float(brow.iloc[0].get("recommended_realtime_window_s", np.nan))
            if not np.isfinite(rec_w):
                continue
            metric = PRIMARY_METRIC.get(target, (None, None))[0]
            if metric is None:
                continue
            sub = metrics[
                (metrics["target_name"].astype(str) == target)
                & (np.isclose(metrics["decode_window_s"].astype(float), rec_w))
            ]
            if sub.empty:
                continue
            row: dict[str, float | str] = {"target": target}
            for mode in modes:
                g = sub[sub[mode_col].astype(str) == mode] if mode_col in sub.columns else pd.DataFrame()
                best_row = _best_row(g, metric)
                row[mode] = float(best_row[metric]) if best_row is not None else np.nan
            bar_rows.append(row)

        if bar_rows:
            labels_f = [_short_label(str(r["target"]), 12) for r in bar_rows]
            x = np.arange(len(labels_f))
            w = 0.25
            for mi, mode in enumerate(modes):
                vals = [float(r[mode]) for r in bar_rows]
                ax_f.bar(x + (mi - 1) * w, vals, width=w, label=mode, color=f"C{mi}")
            ax_f.set_xticks(x)
            ax_f.set_xticklabels(labels_f, rotation=30, ha="right", fontsize=7)
            ax_f.set_ylabel("Best primary metric at selected W")
            ax_f.set_title("Manifold vs counts at deployable W")
            for xi, row in enumerate(bar_rows):
                target = str(row["target"])
                if target not in winners:
                    continue
                sel = str(winners[target].get("selected_feature_mode", ""))
                if sel in modes:
                    mi = modes.index(sel)
                    val = row.get(sel)
                    if val is not None and np.isfinite(float(val)):
                        ax_f.plot(xi + (mi - 1) * w, float(val), marker="*", ms=10, color="#d4a017", zorder=5)
            ax_f.legend(fontsize=7, ncol=3, loc="upper right")
            sns.despine(ax=ax_f)
        else:
            _empty_panel(ax_f, "No scores at selected W")
    else:
        _empty_panel(ax_f, "No manifold comparison data")
    panel_label(ax_f, "F")

    return save_pub_figure(
        fig, out_dir / "fig_window_selection_story.png", dpi=FIGURE_DPI,
        rect=(0.07, 0.04, 0.98, 0.96),
    )


def _window_scores_from_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a minimal window-scores frame from decoder_comparison_metrics."""
    required = {"target_name", "decode_window_s", "decoder_name"}
    if not required.issubset(metrics.columns):
        return pd.DataFrame()
    if "feature_mode" not in metrics.columns and "feature_type" not in metrics.columns:
        return pd.DataFrame()
    rows = []
    for _, row in metrics.iterrows():
        target = str(row["target_name"])
        metric = str(row["primary_metric"]) if "primary_metric" in row.index else _metric_for_target(metrics, target)
        if metric not in row.index or pd.isna(row[metric]):
            continue
        mode = row.get("feature_mode")
        if mode is None or (isinstance(mode, float) and pd.isna(mode)) or not str(mode).strip():
            mode = row.get("feature_type", "counts")
        mode_s = str(mode)
        rt = row.get("realtime_compatible")
        if rt is None or (isinstance(rt, float) and pd.isna(rt)):
            rt = "isomap" not in mode_s or "distilled" in mode_s
        rows.append({
            "spike_source": str(row.get("spike_source", row.get("source", "sorted"))),
            "target": target,
            "decoder": str(row["decoder_name"]),
            "feature_mode": mode_s,
            "causal_window_s": float(row["decode_window_s"]),
            "metric_name": metric,
            "metric_value": float(row[metric]),
            "higher_is_better": not _is_lower_better(metric),
            "realtime_compatible": bool(rt),
        })
    return pd.DataFrame(rows)


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
    # Prefer continuous-position primary runs (default closed-loop target).
    for c in pool:
        if c.name.startswith("position_"):
            return c
    for c in pool:
        if "position" in c.name:
            return c
    # Fall back to most recently modified run.
    return max(pool, key=lambda p: p.stat().st_mtime)


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

    # Physical layout in inches: four equal squares + cbar gutters so A is an
    # undistorted square maze and B–D share the same axes geometry.
    left_in, right_in = 0.85, 0.28
    bottom_in, top_in = 1.05, 0.48
    sq_in = 3.20
    cbar_in = 0.32
    gap_cbar = 0.10
    # Extra space so left colorbar labels do not collide with right y-labels.
    gap_x = 1.85
    gap_y = 1.25
    fig_w = left_in + sq_in + gap_cbar + cbar_in + gap_x + sq_in + gap_cbar + cbar_in + right_in
    fig_h = bottom_in + sq_in + gap_y + sq_in + top_in
    fig = plt.figure(figsize=(fig_w, fig_h))

    def _inch_box(x_in: float, y_in: float, w_in: float, h_in: float):
        return [x_in / fig_w, y_in / fig_h, w_in / fig_w, h_in / fig_h]

    y_top = bottom_in + sq_in + gap_y
    y_bot = bottom_in
    x_left = left_in
    x_right = left_in + sq_in + gap_cbar + cbar_in + gap_x

    ax_a = fig.add_axes(_inch_box(x_left, y_top, sq_in, sq_in))
    cax_a = fig.add_axes(_inch_box(x_left + sq_in + gap_cbar, y_top, cbar_in * 0.72, sq_in))
    ax_b = fig.add_axes(_inch_box(x_right, y_top, sq_in, sq_in))
    cax_b = fig.add_axes(_inch_box(x_right + sq_in + gap_cbar, y_top, cbar_in * 0.72, sq_in))
    ax_c = fig.add_axes(_inch_box(x_left, y_bot, sq_in, sq_in))
    cax_c = fig.add_axes(_inch_box(x_left + sq_in + gap_cbar, y_bot, cbar_in * 0.72, sq_in))
    ax_d = fig.add_axes(_inch_box(x_right, y_bot, sq_in, sq_in))
    cax_d = fig.add_axes(_inch_box(x_right + sq_in + gap_cbar, y_bot, cbar_in * 0.72, sq_in))
    cax_b.set_visible(False)
    cax_d.set_visible(False)

    handles_rt: list = []
    labels_rt: list = []

    if {"true_x", "true_y", "decoded_x", "decoded_y"}.issubset(decoded.columns):
        if "time" in decoded.columns:
            time = pd.to_numeric(decoded["time"], errors="coerce").to_numpy()
            sc = ax_a.scatter(
                decoded["true_x"], decoded["true_y"],
                c=time, s=4, alpha=0.55, cmap="viridis", label="True",
            )
            ax_a.scatter(
                decoded["decoded_x"], decoded["decoded_y"],
                c=time, s=4, alpha=0.55, cmap="viridis", marker="x", label="Decoded",
            )
            cbar = fig.colorbar(sc, cax=cax_a)
            cbar.set_label("Time (s)", fontsize=9)
            cbar.ax.tick_params(labelsize=6)
        else:
            cax_a.set_visible(False)
            ax_a.scatter(
                decoded["true_x"], decoded["true_y"],
                s=4, c="#4C72B0", alpha=0.45, label="True",
            )
            ax_a.scatter(
                decoded["decoded_x"], decoded["decoded_y"],
                s=4, c="#DD8452", alpha=0.45, marker="x", label="Decoded",
            )
        ax_a.set_xlabel("x (cm)")
        ax_a.set_ylabel("y (cm)")
        # Square arena: equal axis lengths on a square physical axes.
        x0 = float(np.nanmin([decoded["true_x"].min(), decoded["decoded_x"].min()]))
        x1 = float(np.nanmax([decoded["true_x"].max(), decoded["decoded_x"].max()]))
        y0 = float(np.nanmin([decoded["true_y"].min(), decoded["decoded_y"].min()]))
        y1 = float(np.nanmax([decoded["true_y"].max(), decoded["decoded_y"].max()]))
        lo = min(x0, y0)
        hi = max(x1, y1)
        pad = 0.02 * max(hi - lo, 1.0)
        ax_a.set_xlim(lo - pad, hi + pad)
        ax_a.set_ylim(lo - pad, hi + pad)
        ax_a.set_aspect("equal", adjustable="box")
        handles_rt = [
            plt.Line2D(
                [0], [0], marker="o", color="0.3", linestyle="None",
                markersize=5, label="True",
            ),
            plt.Line2D(
                [0], [0], marker="x", color="0.3", linestyle="None",
                markersize=5, label="Decoded",
            ),
        ]
        labels_rt = ["True", "Decoded"]
        if ax_a.get_legend() is not None:
            ax_a.get_legend().remove()
        sns.despine(ax=ax_a)
    else:
        cax_a.set_visible(False)
        _empty_panel(ax_a, "No position columns")
    panel_label(ax_a, "A")

    # B — position error over time + triggers
    if "position_error_cm" in decoded.columns and "time" in decoded.columns:
        ax_b.plot(decoded["time"], decoded["position_error_cm"], lw=0.6, color="0.45", alpha=0.8)
        if not events.empty and "time" in events.columns:
            y_max = float(np.nanmax(decoded["position_error_cm"]))
            band = y_max + 0.12 * max(y_max, 1.0)
            if "correct_trigger" in events.columns:
                ok = events[events["correct_trigger"] == True]  # noqa: E712
                bad = events[events["correct_trigger"] == False]  # noqa: E712
                h1 = ax_b.scatter(
                    ok["time"], np.full(len(ok), band),
                    c="#2ca02c", s=18, marker="|", label="Correct trigger",
                )
                h2 = ax_b.scatter(
                    bad["time"], np.full(len(bad), band),
                    c="#d62728", s=18, marker="|", label="False trigger",
                )
                handles_rt = list(handles_rt) + [h1, h2]
                labels_rt = list(labels_rt) + ["Correct trigger", "False trigger"]
            else:
                h = ax_b.scatter(
                    events["time"], np.full(len(events), band),
                    c="C1", s=18, marker="|", label="Trigger",
                )
                handles_rt = list(handles_rt) + [h]
                labels_rt = list(labels_rt) + ["Trigger"]
            ax_b.set_ylim(0, band * 1.2)
        ax_b.set_xlabel("Time (s)")
        ax_b.set_ylabel("Position error (cm)")
        sns.despine(ax=ax_b)
    else:
        _empty_panel(ax_b, "No position_error_cm")
    panel_label(ax_b, "B")

    # C — spatial context confusion (square cells on a square axes)
    if {"true_spatial_context", "decoded_spatial_context"}.issubset(decoded.columns):
        y_true = decoded["true_spatial_context"].astype(str)
        y_pred = decoded["decoded_spatial_context"].astype(str)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            cmn, ax=ax_c, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            cbar_ax=cax_c, cbar_kws={"label": "Row-norm"},
            square=True,
        )
        ax_c.set_xlabel("Decoded")
        ax_c.set_ylabel("True")
        ax_c.tick_params(labelsize=7)
        cax_c.tick_params(labelsize=6)
    else:
        cax_c.set_visible(False)
        _empty_panel(ax_c, "No spatial_context columns")
    panel_label(ax_c, "C")

    # D — trigger precision / reliability summary
    if not events.empty and "correct_trigger" in events.columns:
        n = len(events)
        n_ok = int(events["correct_trigger"].astype(bool).sum())
        precision = n_ok / max(n, 1)
        counts = [n_ok, n - n_ok]
        ax_d.bar(
            ["Correct", "Incorrect"], counts,
            color=["#2ca02c", "#d62728"], edgecolor="k", lw=0.3,
        )
        ax_d.set_ylabel("Trigger count")
        ax_d.set_title(f"precision={precision:.2f}, n={n}", fontsize=9, pad=4)
        sns.despine(ax=ax_d)
    else:
        _empty_panel(ax_d, "No closed-loop events")
    panel_label(ax_d, "D")

    if handles_rt:
        from visualization.publication_style import figure_legend_below
        figure_legend_below(fig, handles_rt, labels_rt, ncol=min(4, len(handles_rt)), y=0.02)
    return save_pub_figure(
        fig, out_dir / "fig_closed_loop.png", dpi=FIGURE_DPI,
        adjust=False,
    )


def _circular_error_deg(true_deg: np.ndarray, pred_deg: np.ndarray) -> np.ndarray:
    """Absolute circular error in degrees."""
    delta = np.deg2rad(pred_deg.astype(float) - true_deg.astype(float))
    return np.abs(np.rad2deg(np.arctan2(np.sin(delta), np.cos(delta))))


def plot_fig_closed_loop_suite(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Companion to fig_closed_loop: speed, movement state, and head direction."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_REALTIME
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _pick_realtime_run(experiment_dir / "realtime_decoding")
    if run_dir is None:
        return None

    decoded = _read_csv(run_dir / "decoded_realtime.csv")
    if decoded.empty or "time" not in decoded.columns:
        return None

    fig = plt.figure(figsize=(10.5, 7.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.34)
    run_label = run_dir.name.replace("_", " ")

    # A — speed
    ax_a = fig.add_subplot(gs[0, 0])
    if {"true_speed", "decoded_speed"}.issubset(decoded.columns):
        t = pd.to_numeric(decoded["time"], errors="coerce")
        ax_a.plot(t, decoded["true_speed"], lw=0.8, color="#4C72B0", alpha=0.85, label="True")
        ax_a.plot(t, decoded["decoded_speed"], lw=0.8, color="#DD8452", alpha=0.85, label="Decoded")
        if "speed_error" in decoded.columns:
            rmse = float(np.sqrt(np.nanmean(np.square(decoded["speed_error"]))))
            ax_a.set_title(f"Speed (RMSE={rmse:.2g} cm/s)", fontsize=9)
        else:
            ax_a.set_title("Speed", fontsize=9)
        ax_a.set_xlabel("Time (s)")
        ax_a.set_ylabel("Speed (cm/s)")
        ax_a.legend(fontsize=7, loc="best")
        sns.despine(ax=ax_a)
    else:
        _empty_panel(ax_a, "No speed columns")
    panel_label(ax_a, "A")

    # B — movement state confusion
    ax_b = fig.add_subplot(gs[0, 1])
    if {"true_movement_state", "decoded_movement_state"}.issubset(decoded.columns):
        y_true = decoded["true_movement_state"].astype(str)
        y_pred = decoded["decoded_movement_state"].astype(str)
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        acc = float((y_true.values == y_pred.values).mean())
        sns.heatmap(
            cmn, ax=ax_b, annot=True, fmt=".2f", cmap="Greens",
            xticklabels=labels, yticklabels=labels,
            cbar_kws={"label": "Row-norm"}, square=True,
        )
        ax_b.set_xlabel("Decoded")
        ax_b.set_ylabel("True")
        ax_b.set_title(f"Movement state (acc={acc:.2f})", fontsize=9)
        ax_b.tick_params(labelsize=7)
    else:
        _empty_panel(ax_b, "No movement_state columns")
    panel_label(ax_b, "B")

    # C — head direction (decoded when available)
    ax_c = fig.add_subplot(gs[1, 0])
    has_hd_dec = "decoded_head_direction_deg" in decoded.columns
    has_hd_true = "true_head_direction_deg" in decoded.columns
    if has_hd_dec and has_hd_true:
        t = pd.to_numeric(decoded["time"], errors="coerce")
        true_hd = pd.to_numeric(decoded["true_head_direction_deg"], errors="coerce")
        dec_hd = pd.to_numeric(decoded["decoded_head_direction_deg"], errors="coerce")
        ax_c.plot(t, true_hd, lw=0.8, color="#4C72B0", alpha=0.85, label="True")
        ax_c.plot(t, dec_hd, lw=0.8, color="#DD8452", alpha=0.85, label="Decoded")
        mean_err = float(np.nanmean(_circular_error_deg(true_hd.to_numpy(), dec_hd.to_numpy())))
        ax_c.set_title(f"Head direction (mean circ. err={mean_err:.1f}°)", fontsize=9)
        ax_c.set_xlabel("Time (s)")
        ax_c.set_ylabel("Heading (deg)")
        ax_c.legend(fontsize=7, loc="best")
        sns.despine(ax=ax_c)
    elif has_hd_true:
        t = pd.to_numeric(decoded["time"], errors="coerce")
        ax_c.plot(
            t, decoded["true_head_direction_deg"],
            lw=0.8, color="#4C72B0", alpha=0.85,
        )
        ax_c.set_xlabel("Time (s)")
        ax_c.set_ylabel("True heading (deg)")
        ax_c.set_title("Head direction (true only)", fontsize=9)
        ax_c.text(
            0.5, 0.08,
            "Decoded HD not in default suite;\nuse --closed-loop-target head_direction",
            transform=ax_c.transAxes, ha="center", va="bottom", fontsize=7, color="0.45",
        )
        sns.despine(ax=ax_c)
    else:
        _empty_panel(
            ax_c,
            "No head_direction columns\n(decode with --closed-loop-target head_direction)",
        )
    panel_label(ax_c, "C")

    # D — suite summary from realtime_metrics.json when present
    ax_d = fig.add_subplot(gs[1, 1])
    metrics_path = run_dir / "realtime_metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError):
            metrics = {}
        names: list[str] = []
        vals: list[float] = []
        mapping = (
            ("Position err (cm)", "mean_position_error_cm"),
            ("Context acc", "spatial_context_accuracy"),
            ("Movement acc", "movement_state_accuracy"),
            ("Speed R²", "speed_r2"),
        )
        for label, key in mapping:
            v = metrics.get(key)
            if v is not None and np.isfinite(float(v)):
                names.append(label)
                vals.append(float(v))
        if names:
            y = np.arange(len(names))
            ax_d.barh(y, vals, color="C0", edgecolor="k", linewidth=0.3)
            ax_d.set_yticks(y)
            ax_d.set_yticklabels(names, fontsize=8)
            ax_d.set_xlabel("Metric value")
            ax_d.set_title(f"Suite metrics ({run_label})", fontsize=9)
            expand_xlim_for_labels(ax_d, vals, pad_frac=0.15)
            sns.despine(ax=ax_d)
        else:
            _empty_panel(ax_d, "No suite metrics")
    else:
        _empty_panel(ax_d, "No realtime_metrics.json")
    panel_label(ax_d, "D")

    fig.suptitle(
        "Realtime suite targets (auxiliary heads alongside position)",
        fontsize=11, y=0.98,
    )
    return save_pub_figure(
        fig, out_dir / "fig_closed_loop_suite.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.06, 0.98, 0.93),
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
            ax_a.set_yticklabels(labels, fontsize=9)
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

    def _latency_label_col(df: pd.DataFrame, *preferred: str) -> str | None:
        for col in preferred:
            if col in df.columns:
                return col
        return None

    def _barh_latency(
        ax,
        df: pd.DataFrame,
        name_col: str | None,
        mean_col: str = "mean_ms",
        *,
        p95_col: str = "p95_ms",
    ) -> None:
        if df.empty or name_col is None or name_col not in df.columns or mean_col not in df.columns:
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
        if p95_col in sub.columns:
            p95_vals = pd.to_numeric(sub[p95_col], errors="coerce")
            mean_vals = pd.to_numeric(sub[mean_col], errors="coerce")
            ax.errorbar(
                mean_vals,
                y,
                xerr=[
                    np.zeros(len(sub)),
                    np.maximum(0.0, p95_vals - mean_vals),
                ],
                fmt="none",
                color="k",
                capsize=2,
                linewidth=0.8,
            )
        ax.axvline(budget_ms, color="0.3", ls="--", lw=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(sub[name_col].astype(str).tolist(), fontsize=9)
        ax.set_xlabel("Latency (ms)")
        sns.despine(ax=ax)

    ax_a = fig.add_subplot(gs[0, 0])
    feat_df = feature if not feature.empty else (
        everything[everything["category"] == "feature_transform"] if not everything.empty else pd.DataFrame()
    )
    # Benchmark CSV uses feature_mode; unified table uses name.
    _barh_latency(
        ax_a,
        feat_df,
        _latency_label_col(feat_df, "feature_mode", "name", "method"),
    )
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    if not stage.empty:
        name_col = "stage" if "stage" in stage.columns else "name"
        # Aggregate mean across sources if needed
        if "source" in stage.columns and name_col in stage.columns:
            agg_cols: dict[str, tuple[str, str]] = {"mean_ms": ("mean_ms", "mean")}
            if "p95_ms" in stage.columns:
                agg_cols["p95_ms"] = ("p95_ms", "mean")
            agg = stage.groupby(name_col, as_index=False).agg(**agg_cols)
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


def plot_fig_latency_realtime(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Per-update closed-loop latency: distribution, timeline, stage box plot, median stack."""
    from visualization.latency_plots import _load_latency_per_update, stage_ms_columns

    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / "latency"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, budget_ms = _load_latency_per_update(experiment_dir)
    if df.empty or "total_update_ms" not in df.columns:
        return None

    totals = pd.to_numeric(df["total_update_ms"], errors="coerce").dropna()
    if totals.empty:
        return None

    stage_pairs = [
        (stage, col) for stage, col in stage_ms_columns() if col in df.columns
    ]

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)
    palette = sns.color_palette("deep", max(len(stage_pairs), 1))

    # Panel A — total update distribution
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(
        totals,
        bins=min(30, max(10, len(totals) // 8)),
        density=True,
        alpha=0.55,
        color=palette[0],
        edgecolor="white",
        linewidth=0.4,
    )
    if len(totals) >= 2:
        sns.kdeplot(totals, ax=ax_a, color="0.15", lw=1.2)
    ax_a.axvline(budget_ms, color="0.3", ls="--", lw=1.0, label=f"{budget_ms:.0f} ms budget")
    mean_ms = float(totals.mean())
    p95_ms = float(np.percentile(totals, 95))
    within_frac = float((totals <= budget_ms).mean())
    ax_a.set_xlabel("Total update latency (ms)")
    ax_a.set_ylabel("Density")
    ax_a.text(
        0.98,
        0.97,
        f"mean {mean_ms:.1f} ms\np95 {p95_ms:.1f} ms\n{within_frac * 100:.0f}% within budget",
        transform=ax_a.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )
    ax_a.legend(loc="upper left", fontsize=7)
    sns.despine(ax=ax_a)
    panel_label(ax_a, "A")

    # Panel B — timeline
    ax_b = fig.add_subplot(gs[0, 1])
    plot_df = df.copy()
    plot_df["total_update_ms"] = pd.to_numeric(plot_df["total_update_ms"], errors="coerce")
    plot_df = plot_df.dropna(subset=["total_update_ms"])
    if "time_s" in plot_df.columns:
        plot_df["time_s"] = pd.to_numeric(plot_df["time_s"], errors="coerce")
        plot_df = plot_df.dropna(subset=["time_s"])
        ax_b.scatter(
            plot_df["time_s"],
            plot_df["total_update_ms"],
            s=12,
            alpha=0.55,
            color=palette[0],
            edgecolors="none",
        )
        ax_b.set_xlabel("Session time (s)")
    else:
        ax_b.scatter(
            np.arange(len(plot_df)),
            plot_df["total_update_ms"],
            s=12,
            alpha=0.55,
            color=palette[0],
            edgecolors="none",
        )
        ax_b.set_xlabel("Profiled update index")
    ax_b.axhline(budget_ms, color="0.3", ls="--", lw=1.0)
    ax_b.set_ylabel("Total update latency (ms)")
    sns.despine(ax=ax_b)
    panel_label(ax_b, "B")

    # Panel C — per-stage box plot
    ax_c = fig.add_subplot(gs[1, 0])
    if stage_pairs:
        box_data = [
            pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
            for _, col in stage_pairs
        ]
        labels = [s.replace("_", "\n") for s, _ in stage_pairs]
        bp = ax_c.boxplot(
            box_data,
            vert=True,
            patch_artist=True,
            showfliers=False,
        )
        for patch, color in zip(bp["boxes"], palette):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax_c.set_xticks(np.arange(1, len(labels) + 1))
        ax_c.set_xticklabels(labels, rotation=35, fontsize=7)
        ax_c.axhline(budget_ms, color="0.3", ls="--", lw=1.0)
        ax_c.set_ylabel("Latency (ms)")
    else:
        _empty_panel(ax_c, "No stage columns")
    sns.despine(ax=ax_c)
    panel_label(ax_c, "C")

    # Panel D — median stage contribution stack
    ax_d = fig.add_subplot(gs[1, 1])
    if stage_pairs:
        medians = [
            float(pd.to_numeric(df[col], errors="coerce").median())
            for _, col in stage_pairs
        ]
        left = 0.0
        for i, ((stage, _), med) in enumerate(zip(stage_pairs, medians)):
            ax_d.barh(
                0,
                med,
                left=left,
                height=0.45,
                color=palette[i % len(palette)],
                edgecolor="k",
                lw=0.3,
                label=stage.replace("_", " "),
            )
            left += med
        ax_d.axvline(budget_ms, color="0.3", ls="--", lw=1.0, label=f"{budget_ms:.0f} ms budget")
        ax_d.axvline(float(np.median(totals)), color="0.5", ls=":", lw=1.0, label="median total")
        ax_d.set_xlabel("Median latency (ms)")
        ax_d.set_yticks([])
        legend_below(ax_d, ncol=2, fontsize=7)
    else:
        _empty_panel(ax_d, "No stage columns")
    sns.despine(ax=ax_d, left=True)
    panel_label(ax_d, "D")

    return save_pub_figure(
        fig,
        out_dir / "fig_latency_realtime.png",
        dpi=FIGURE_DPI,
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

    from visualization.comparison_grid_diagram import plot_fig_decoder_comparison_grid

    written: list[Path] = []
    for fn in (
        plot_fig_decoding_performance,
        plot_fig_decoder_comparison_grid,
        plot_fig_window_selection_story,
        plot_fig_feature_x_window,
        plot_fig_decoder_x_window,
        plot_fig_continuous_decoders_feature_x_window,
        plot_fig_categorical_decoders_feature_x_window,
        plot_fig_closed_loop,
        plot_fig_closed_loop_suite,
        plot_fig_deployment,
        plot_fig_latency,
        plot_fig_latency_realtime,
        plot_fig_temporal_wl,
    ):
        try:
            path = fn(experiment_dir, figures_dir)
            if path is not None:
                written.append(path)
                print(f"  wrote {path.relative_to(figures_dir)}")
        except Exception as exc:
            print(f"  warning: {fn.__name__} skipped ({exc})")

    # Dense onepagers (ideal W/decoder + counts vs manifold) after manifold summary
    try:
        from visualization.deployment_plots import plot_deployable_selection_onepagers
        from visualization.manifold_plots import plot_fig_manifold_vs_spikes_onepager

        for path in plot_deployable_selection_onepagers(experiment_dir, figures_dir):
            written.append(path)
            print(f"  wrote {path.relative_to(figures_dir)}")
        path = plot_fig_manifold_vs_spikes_onepager(experiment_dir, figures_dir)
        if path is not None:
            written.append(path)
            print(f"  wrote {path.relative_to(figures_dir)}")
    except Exception as exc:
        print(f"  warning: deployable/manifold onepagers skipped ({exc})")

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
