"""Manifold decoding visualization figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 150


def plot_manifold_comparison_outputs(comparison_dir: Path, figures_dir: Path) -> None:
    """Generate manifold-related figures under figures_dir."""
    comparison_dir = Path(comparison_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = list(comparison_dir.rglob("decoder_comparison_metrics.csv"))
    if not metrics_paths and (comparison_dir / "decoder_comparison_metrics.csv").exists():
        metrics_paths = [comparison_dir / "decoder_comparison_metrics.csv"]

    frames = []
    for path in metrics_paths:
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if not df.empty:
            frames.append(df)
    if not frames:
        return
    metrics = pd.concat(frames, ignore_index=True)

    _plot_feature_performance_by_target(metrics, figures_dir)
    _plot_counts_vs_manifold_summary(comparison_dir, figures_dir)
    _plot_region_layer_by_target(metrics, figures_dir, "region_pca", "region_manifold_decoding_by_target.png")
    _plot_region_layer_by_target(metrics, figures_dir, "layer_pca", "layer_manifold_decoding_by_target.png")

    explained_paths = list(comparison_dir.rglob("manifold_explained_variance.csv"))
    for path in explained_paths:
        explained = pd.read_csv(path)
        if explained.empty:
            continue
        _plot_explained_variance(explained, figures_dir, "region")
        _plot_explained_variance(explained, figures_dir, "layer")

    _plot_latent_trajectories(comparison_dir, figures_dir)


def _plot_feature_performance_by_target(metrics: pd.DataFrame, out_dir: Path) -> None:
    if "feature_type" not in metrics.columns or "target_name" not in metrics.columns:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    targets = sorted(metrics["target_name"].unique())
    modes = sorted(metrics["feature_type"].unique())
    x = np.arange(len(targets))
    width = 0.8 / max(len(modes), 1)
    for i, mode in enumerate(modes):
        vals = []
        for target in targets:
            sub = metrics[(metrics["target_name"] == target) & (metrics["feature_type"] == mode)]
            if sub.empty or "primary_metric" not in sub.columns:
                vals.append(np.nan)
                continue
            metric = sub["primary_metric"].iloc[0]
            if metric not in sub.columns:
                vals.append(np.nan)
                continue
            # Use best value for this mode/target
            if "error" in metric or metric.endswith("_deg"):
                vals.append(float(sub[metric].min()))
            else:
                vals.append(float(sub[metric].max()))
        ax.bar(x + i * width, vals, width=width, label=mode)
    ax.set_xticks(x + width * (len(modes) - 1) / 2)
    ax.set_xticklabels(targets, rotation=30, ha="right")
    ax.set_ylabel("Best primary metric")
    ax.set_title("Manifold feature performance by target")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "manifold_feature_performance_by_target.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_counts_vs_manifold_summary(comparison_dir: Path, out_dir: Path) -> None:
    paths = list(Path(comparison_dir).rglob("manifold_vs_counts_summary.csv"))
    if not paths:
        return
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    labels = [f"{t}\n({s})" for t, s in zip(df["target_name"], df["spike_source"])]
    ax.bar(range(len(df)), df["performance_difference"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Manifold − counts (signed improvement)")
    ax.set_title("Counts vs manifold decoding summary")
    fig.tight_layout()
    fig.savefig(out_dir / "counts_vs_manifold_decoding_summary.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_region_layer_by_target(
    metrics: pd.DataFrame,
    out_dir: Path,
    feature_type: str,
    filename: str,
) -> None:
    sub = metrics[metrics["feature_type"] == feature_type]
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for target, g in sub.groupby("target_name"):
        metric = g["primary_metric"].iloc[0]
        if metric not in g.columns:
            continue
        best = g.groupby("decode_window_s")[metric].max() if "error" not in metric else g.groupby("decode_window_s")[metric].min()
        ax.plot(best.index, best.values, marker="o", label=target)
    ax.set_xlabel("Decode window (s)")
    ax.set_ylabel("Primary metric")
    ax.set_title(feature_type)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / filename, dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_explained_variance(explained: pd.DataFrame, out_dir: Path, grouping: str) -> None:
    sub = explained[explained["feature_type"] == f"{grouping}_pca"]
    if sub.empty:
        return
    summary = sub.groupby("group_name")["explained_variance_sum"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary.index.astype(str), summary.values)
    ax.set_ylabel("Mean explained variance (selected PCs)")
    ax.set_title(f"Manifold explained variance by {grouping}")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / f"manifold_explained_variance_by_{grouping}.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_latent_trajectories(comparison_dir: Path, out_dir: Path) -> None:
    """Plot PC1 vs PC2 colored by position using saved transforms if available."""
    try:
        from realtime.data_loading import load_simulation_data, make_decode_times
        from realtime.decoding_targets import align_extended_behavior_to_decoder_times
        from realtime.manifold_features import load_feature_transformer
        from realtime.spike_features import build_causal_spike_matrix
        from realtime.timing import extract_behavior_times
        from realtime.train_decoder import causal_train_test_split
    except Exception:
        return

    # Find a parent experiment dir with behavior.csv
    comparison_dir = Path(comparison_dir)
    candidates = [comparison_dir, comparison_dir.parent, comparison_dir.parent.parent]
    input_dir = None
    for c in candidates:
        if (c / "behavior.csv").exists():
            input_dir = c
            break
    if input_dir is None:
        return

    transform_dirs = list(comparison_dir.rglob("global_pca_k*_w*.ms")) + list(
        comparison_dir.rglob("region_pca_k*_w*.ms")
    )
    # directory names use ...ms without extra dots from glob - match manifold_transform dirs
    transform_dirs = [
        p for p in comparison_dir.rglob("meta.json")
        if p.parent.name.startswith(("global_pca_", "region_pca_"))
    ]
    if not transform_dirs:
        return

    data = load_simulation_data(input_dir, "sorted")
    behavior_times = extract_behavior_times(data["behavior_df"])

    for meta_path in transform_dirs[:6]:
        tdir = meta_path.parent
        try:
            transformer = load_feature_transformer(tdir)
        except Exception:
            continue
        # Infer window from dirname ..._w0250ms
        name = tdir.name
        try:
            w_ms = int(name.split("_w")[-1].replace("ms", ""))
            decode_window = w_ms / 1000.0
        except Exception:
            decode_window = 0.250
        decode_times = make_decode_times(
            data["session_duration"], decode_window, 0.05, behavior_times=behavior_times,
        )
        aligned = align_extended_behavior_to_decoder_times(
            data["behavior_df"], decode_times, data["summary"]
        )
        X = build_causal_spike_matrix(
            data["spikes_df"], data["unit_ids"], decode_times, decode_window,
        )
        train_mask, _ = causal_train_test_split(decode_times, 0.70)
        # Transform with already-fitted transform (do not refit)
        Z = transformer.transform(X)
        if Z.shape[1] < 2:
            continue
        fig, ax = plt.subplots(figsize=(5, 5))
        sc = ax.scatter(
            Z[:, 0], Z[:, 1], c=aligned["x"].to_numpy(), s=4, cmap="viridis", alpha=0.7,
        )
        fig.colorbar(sc, ax=ax, label="true x position (cm)")
        ax.set_xlabel("Latent PC1 (neural manifold)")
        ax.set_ylabel("Latent PC2 (neural manifold)")
        ax.set_title(f"{name}\nlatent neural coordinates (not physical position)")
        fig.tight_layout()
        safe = name.replace("/", "_")
        if "global_pca" in name:
            fig.savefig(out_dir / "global_pca_manifold_trajectory_position.png", dpi=FIGURE_DPI)
        elif "region_pca" in name:
            fig.savefig(out_dir / f"region_pca_manifold_trajectory_position.png", dpi=FIGURE_DPI)
        else:
            fig.savefig(out_dir / f"{safe}_trajectory_position.png", dpi=FIGURE_DPI)
        plt.close(fig)
