"""Publication-style plots for neural feature-set comparisons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PRIMARY_METRIC = {
    "position": ("mean_position_error_cm", "lower"),
    "speed": ("r2", "higher"),
    "acceleration": ("r2", "higher"),
    "head_direction": ("mean_circular_error_deg", "lower"),
    "distance_to_wall": ("r2", "higher"),
    "spatial_context": ("balanced_accuracy", "higher"),
    "movement_state": ("balanced_accuracy", "higher"),
    "wall_distance_bin": ("balanced_accuracy", "higher"),
}


def _best_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    metric, direction = PRIMARY_METRIC[target]
    if metric not in df.columns:
        return pd.DataFrame()
    g = df[df["target_name"] == target].dropna(subset=[metric])
    if g.empty:
        return g
    keys = [c for c in ("feature_set", "embedding_type", "decode_window_s") if c in g.columns]
    rows = []
    for _, sub in g.groupby(keys, dropna=False):
        idx = sub[metric].idxmin() if direction == "lower" else sub[metric].idxmax()
        rows.append(sub.loc[idx])
    return pd.DataFrame(rows)


def plot_feature_set_performance(
    metrics_df: pd.DataFrame,
    output_path: Path,
    *,
    targets: tuple[str, ...] = ("position", "speed", "spatial_context"),
) -> Path | None:
    if metrics_df is None or metrics_df.empty or "feature_set" not in metrics_df.columns:
        return None
    df = metrics_df.copy()
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, target in zip(axes[0], targets, strict=False):
        if target not in PRIMARY_METRIC:
            ax.set_visible(False)
            continue
        metric, direction = PRIMARY_METRIC[target]
        best = _best_rows(df, target)
        if best.empty:
            ax.set_visible(False)
            continue
        order = best.groupby("feature_set")[metric].mean().sort_values(
            ascending=(direction == "lower"),
        )
        vals = [float(order.loc[fs]) for fs in order.index]
        ax.barh(list(order.index), vals, color="#3d5a80")
        ax.set_xlabel(metric)
        ax.set_title(target)
        ax.invert_yaxis()
    fig.suptitle("Feature set vs decoding performance (best decoder/window)")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_feature_set_manifold_heatmap(
    metrics_df: pd.DataFrame,
    output_path: Path,
    *,
    target: str = "position",
) -> Path | None:
    if metrics_df is None or metrics_df.empty:
        return None
    if "feature_set" not in metrics_df.columns or "embedding_type" not in metrics_df.columns:
        return None
    if target not in PRIMARY_METRIC:
        return None
    metric, direction = PRIMARY_METRIC[target]
    df = metrics_df.copy()
    sub = df[df["target_name"] == target].dropna(subset=[metric])
    if sub.empty:
        return None
    pivot = sub.pivot_table(
        index="feature_set",
        columns="embedding_type",
        values=metric,
        aggfunc="min" if direction == "lower" else "max",
    )
    fig, ax = plt.subplots(figsize=(1.4 * max(4, pivot.shape[1]), 1.0 * max(3, pivot.shape[0])))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis_r" if direction == "lower" else "viridis")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title(f"{target}: {metric}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_dim_cost_tradeoff(
    metrics_df: pd.DataFrame,
    output_path: Path,
    *,
    target: str = "position",
) -> Path | None:
    if metrics_df is None or metrics_df.empty or target not in PRIMARY_METRIC:
        return None
    metric, direction = PRIMARY_METRIC[target]
    df = metrics_df.copy()
    sub = df[df["target_name"] == target].dropna(subset=[metric])
    if sub.empty or "feature_dimension" not in sub.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for fs, g in sub.groupby("feature_set"):
        axes[0].scatter(g["feature_dimension"], g[metric], s=28, alpha=0.75, label=fs)
        if "feature_extract_ms" in g.columns:
            axes[1].scatter(g["feature_extract_ms"], g[metric], s=28, alpha=0.75, label=fs)
    axes[0].set_xlabel("feature dimension")
    axes[0].set_ylabel(metric)
    axes[0].set_title("Dimensionality vs performance")
    axes[1].set_xlabel("feature extract ms")
    axes[1].set_ylabel(metric)
    axes[1].set_title("Compute vs performance")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_latent_stability_summary(
    stability_df: pd.DataFrame,
    output_path: Path,
) -> Path | None:
    """Bar summary of Procrustes / trustworthiness by feature set (no severity axis)."""
    if stability_df is None or stability_df.empty:
        return None
    if "procrustes_alignment_error" not in stability_df.columns:
        return None
    if "feature_set" not in stability_df.columns:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    agg = (
        stability_df.groupby("feature_set", as_index=False)["procrustes_alignment_error"]
        .mean()
        .sort_values("procrustes_alignment_error")
    )
    ax.barh(list(agg["feature_set"]), list(agg["procrustes_alignment_error"]), color="#3d5a80")
    ax.set_xlabel("Procrustes alignment error (mean)")
    ax.set_title("Latent geometry stability by feature set")
    ax.invert_yaxis()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_feature_set_figures(
    comparison_dir: Path,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Load comparison CSVs and write the standard feature-set figure set."""
    comparison_dir = Path(comparison_dir)
    figures_dir = Path(figures_dir) if figures_dir else comparison_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = comparison_dir / "decoder_comparison_metrics.csv"
    if not metrics_path.exists():
        return []
    metrics = pd.read_csv(metrics_path)
    written: list[Path] = []
    for fn, path in (
        (plot_feature_set_performance, figures_dir / "fig_feature_set_performance.png"),
        (lambda m, p: plot_feature_set_manifold_heatmap(m, p, target="position"),
         figures_dir / "fig_feature_set_manifold_heatmap.png"),
        (lambda m, p: plot_dim_cost_tradeoff(m, p, target="position"),
         figures_dir / "fig_feature_dim_cost_tradeoff.png"),
    ):
        out = fn(metrics, path)
        if out is not None:
            written.append(out)

    stab_path = comparison_dir / "latent_stability_metrics.csv"
    if stab_path.exists():
        stab = pd.read_csv(stab_path)
        out = plot_latent_stability_summary(
            stab, figures_dir / "fig_latent_stability_summary.png",
        )
        if out is not None:
            written.append(out)
    return written
