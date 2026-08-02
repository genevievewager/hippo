"""Manifold decoding visualization figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 150


def plot_manifold_comparison_outputs(comparison_dir: Path, figures_dir: Path) -> None:
    """Generate manifold-related figures under figures_dir.

    Default: publication multi-panel Isomap / manifold figures. Legacy
    single-panel helpers remain as private functions.
    """
    comparison_dir = Path(comparison_dir)
    figures_dir = Path(figures_dir)
    experiment_dir = _resolve_experiment_dir(comparison_dir) or comparison_dir.parent
    # figures_dir may already be figures/decoder_comparison
    root_figures = figures_dir if figures_dir.name != "decoder_comparison" else figures_dir.parent
    try:
        from visualization.publication_decoding_plots import plot_fig_manifold_decoding
        from visualization.publication_isomap_plots import generate_publication_isomap_figures

        plot_fig_manifold_decoding(experiment_dir, root_figures)
        generate_publication_isomap_figures(experiment_dir, root_figures, cleanup_legacy=False)
        plot_fig_manifold_vs_spikes_onepager(experiment_dir, root_figures)
    except Exception as exc:
        print(f"  warning: publication manifold/isomap figures skipped ({exc})")


def _resolve_experiment_dir(comparison_dir: Path) -> Path | None:
    """Walk up from decoder_comparison/ to the experiment root if present."""
    comparison_dir = Path(comparison_dir)
    for cand in (comparison_dir, comparison_dir.parent, comparison_dir.parent.parent):
        if (cand / "behavior.csv").exists() or (cand / "summary.json").exists():
            return cand
        if cand.name == "decoder_comparison" and cand.parent.exists():
            return cand.parent
    if comparison_dir.name == "decoder_comparison":
        return comparison_dir.parent
    return None


def _metrics_to_score_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    """Normalize decoder_comparison_metrics rows to the window-score schema."""
    from realtime.decoder_comparison import PRIMARY_METRIC

    rows = []
    for _, row in metrics.iterrows():
        target = str(row.get("target_name", ""))
        if not target:
            continue
        metric_name = str(row.get("primary_metric", ""))
        if not metric_name or metric_name not in row.index:
            if target in PRIMARY_METRIC:
                metric_name = PRIMARY_METRIC[target][0]
            else:
                continue
        if metric_name not in row.index or pd.isna(row[metric_name]):
            continue
        if target in PRIMARY_METRIC:
            higher = PRIMARY_METRIC[target][1] == "higher"
        else:
            higher = not (
                "error" in metric_name.lower()
                or metric_name.endswith("_deg")
                or metric_name in ("mae", "rmse", "mae_cm", "rmse_cm")
            )
        mode = row.get("feature_mode")
        if mode is None or (isinstance(mode, float) and pd.isna(mode)) or not str(mode).strip():
            mode = row.get("feature_type", "")
        rows.append({
            "spike_source": str(row.get("spike_source", "sorted")),
            "target": target,
            "decoder": str(row.get("decoder_name", "")),
            "feature_mode": str(mode),
            "causal_window_s": float(row.get("decode_window_s", float("nan"))),
            "metric_name": metric_name,
            "metric_value": float(row[metric_name]),
            "higher_is_better": higher,
            "realtime_compatible": bool(row.get("realtime_compatible", True)),
            "manifold_n_components": row.get("manifold_n_components"),
        })
    return pd.DataFrame(rows)


def _load_scores_prefer_deployment(
    experiment_dir: Path | None,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Prefer deployment window scores; fall back to metrics conversion."""
    rebuilt = _metrics_to_score_frame(metrics)
    if experiment_dir is not None:
        candidates = [
            Path(experiment_dir) / "deployment_decoder_selection" / "all_sorted_window_scores.csv",
            Path(experiment_dir) / "decoder_comparison" / "sorted" / "all_window_scores_sorted.csv",
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                if df.empty:
                    continue
                # Older exports wrote base feature_type (always counts) into
                # feature_mode; prefer metrics when that collapse is detected.
                score_modes = (
                    {str(m) for m in df["feature_mode"].dropna().unique()}
                    if "feature_mode" in df.columns
                    else set()
                )
                metric_modes = (
                    {str(m) for m in rebuilt["feature_mode"].dropna().unique()}
                    if not rebuilt.empty and "feature_mode" in rebuilt.columns
                    else set()
                )
                if score_modes <= {"counts", "rates"} and len(metric_modes - {"counts", "rates"}) > 0:
                    return rebuilt
                return df
    return rebuilt


def _load_manifold_vs_counts(comparison_dir: Path) -> pd.DataFrame:
    paths = list(Path(comparison_dir).rglob("manifold_vs_counts_summary.csv"))
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def _load_realtime_registry(experiment_dir: Path | None) -> dict | None:
    if experiment_dir is None:
        return None
    for path in (
        Path(experiment_dir) / "models" / "best_realtime_decoders.json",
        Path(experiment_dir) / "deployment_decoder_selection" / "best_realtime_decoders.json",
    ):
        if path.exists():
            import json
            with open(path) as f:
                return json.load(f)
    return None


def _short_label(name: str, max_len: int = 14) -> str:
    s = str(name)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _feature_column_order(features: list[str]) -> list[str]:
    preferred = [
        "counts", "rates",
        "global_pca", "region_pca", "layer_pca", "cell_type_pca", "rate_model_pca",
        "global_isomap_distilled", "global_isomap",
    ]
    return [f for f in preferred if f in features] + [
        f for f in sorted(features) if f not in preferred
    ]


def _best_over_windows_scores(sub: pd.DataFrame) -> pd.DataFrame:
    rows = []
    higher = bool(sub["higher_is_better"].iloc[0])
    for (dec, feat), g in sub.groupby(["decoder", "feature_mode"], sort=False):
        idx = g["metric_value"].idxmax() if higher else g["metric_value"].idxmin()
        row = g.loc[idx]
        rows.append({
            "decoder": str(dec),
            "feature_mode": str(feat),
            "causal_window_s": float(row["causal_window_s"]),
            "metric_value": float(row["metric_value"]),
            "realtime_compatible": bool(row.get("realtime_compatible", True)),
        })
    return pd.DataFrame(rows)


def _plot_manifold_vs_spikes_onepager(
    *,
    metrics: pd.DataFrame,
    comparison_dir: Path,
    out_dir: Path,
    experiment_dir: Path | None = None,
) -> Path | None:
    """One-pager: counts vs manifold features for sorted / deployable decoding.

    Top table: best counts vs best manifold vs deployable registry choice.
    Lower panels: decoder × feature heatmaps (best W per cell; gold = selected).
    """
    scores = _load_scores_prefer_deployment(experiment_dir, metrics)
    if scores.empty:
        return None
    if "spike_source" in scores.columns:
        scores = scores[scores["spike_source"].astype(str) == "sorted"].copy()
    if scores.empty:
        return None

    vs = _load_manifold_vs_counts(comparison_dir)
    if not vs.empty and "spike_source" in vs.columns:
        vs = vs[vs["spike_source"].astype(str) == "sorted"].copy()
    registry = _load_realtime_registry(experiment_dir)
    winners = {}
    if registry and "targets" in registry:
        winners = {str(t): dict(cfg) for t, cfg in registry["targets"].items()}

    preferred = [
        "position", "speed", "acceleration", "head_direction",
        "distance_to_wall", "spatial_context", "movement_state", "wall_distance_bin",
    ]
    targets = list(winners.keys()) if winners else sorted(scores["target"].unique())
    if not vs.empty:
        for t in vs["target_name"].astype(str).tolist():
            if t not in targets:
                targets.append(t)
    targets = [t for t in preferred if t in targets] + [
        t for t in targets if t not in preferred
    ]
    if not targets:
        return None

    n = len(targets)
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(16.5, 3.1 + 2.9 * rows_n))
    gs = fig.add_gridspec(
        rows_n + 1, cols,
        height_ratios=[1.35] + [1.0] * rows_n,
        hspace=0.55, wspace=0.35,
    )

    # ---- Top: counts vs manifold vs selected ----
    ax_tab = fig.add_subplot(gs[0, :])
    ax_tab.axis("off")
    table_rows = []
    vs_by_target = {}
    if not vs.empty:
        for _, row in vs.iterrows():
            vs_by_target[str(row["target_name"])] = row

    for t in targets:
        sub = scores[scores["target"] == t]
        metric = str(sub["metric_name"].iloc[0]) if not sub.empty else ""
        vrow = vs_by_target.get(t)
        if vrow is not None:
            counts_cell = (
                f"{_short_label(vrow['best_counts_decoder'], 16)} "
                f"W={float(vrow['best_counts_window_s']):.2f} "
                f"({float(vrow['best_counts_metric_value']):.3g})"
            )
            man_cell = (
                f"{_short_label(vrow['best_manifold_feature_type'], 12)}/"
                f"{_short_label(vrow['best_manifold_decoder'], 14)} "
                f"W={float(vrow['best_manifold_window_s']):.2f} "
                f"({float(vrow['best_manifold_metric_value']):.3g})"
            )
            verdict = str(vrow.get("interpretation", ""))
        else:
            counts_cell, man_cell, verdict = "—", "—", "—"

        if t in winners:
            cfg = winners[t]
            selected = (
                f"{_short_label(cfg.get('selected_feature_mode', ''), 12)} / "
                f"{_short_label(cfg.get('selected_decoder', ''), 14)} "
                f"W={float(cfg.get('selected_causal_window_s', float('nan'))):.2f}"
            )
        else:
            selected = "—"

        table_rows.append([
            t,
            _short_label(counts_cell, 42),
            _short_label(man_cell, 42),
            _short_label(verdict, 28),
            _short_label(selected, 36),
            metric,
        ])

    tbl = ax_tab.table(
        cellText=table_rows,
        colLabels=[
            "target",
            "best counts (spikes)",
            "best manifold",
            "verdict (±5%)",
            "deployable selected",
            "metric",
        ],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 1.4)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f4f6f")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f6f9")
        # Color verdict column
        if r > 0 and c == 3 and r - 1 < len(table_rows):
            text = str(table_rows[r - 1][3]).lower()
            if "improves" in text:
                cell.set_facecolor("#c7e9c0")
            elif "reduces" in text:
                cell.set_facecolor("#fcae91")
            elif "comparable" in text:
                cell.set_facecolor("#fff2a8")
    ax_tab.set_title(
        "Manifold vs spikes — sorted / Neuropixels  "
        "(heatmaps = best W per decoder×feature; gold box = deployable selection; "
        "hatch = offline-only)",
        fontsize=12,
        pad=8,
    )

    # ---- Per-target heatmaps ----
    for i, target in enumerate(targets):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r + 1, c])
        sub = scores[scores["target"] == target]
        if sub.empty:
            ax.axis("off")
            continue
        higher = bool(sub["higher_is_better"].iloc[0])
        metric = str(sub["metric_name"].iloc[0])
        collapsed = _best_over_windows_scores(sub)
        if collapsed.empty:
            ax.axis("off")
            continue
        decoders = sorted(collapsed["decoder"].unique())
        features = _feature_column_order(list(collapsed["feature_mode"].unique()))
        mat = np.full((len(decoders), len(features)), np.nan)
        win_mat = np.full((len(decoders), len(features)), np.nan)
        rt_mat = np.ones((len(decoders), len(features)), dtype=bool)
        for _, row in collapsed.iterrows():
            di = decoders.index(row["decoder"])
            fi = features.index(row["feature_mode"])
            mat[di, fi] = row["metric_value"]
            win_mat[di, fi] = row["causal_window_s"]
            rt_mat[di, fi] = bool(row["realtime_compatible"])

        display = mat.copy()
        if not higher:
            display = -display
        finite = display[np.isfinite(display)]
        if finite.size:
            vmin = float(np.nanpercentile(finite, 5))
            vmax = float(np.nanpercentile(finite, 95))
            if vmin == vmax:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite) + 1e-9)
        else:
            vmin, vmax = 0.0, 1.0
        ax.imshow(display, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)

        for di in range(len(decoders)):
            for fi in range(len(features)):
                if not rt_mat[di, fi] and np.isfinite(mat[di, fi]):
                    ax.add_patch(
                        plt.Rectangle(
                            (fi - 0.5, di - 0.5), 1, 1,
                            fill=False, hatch="////", edgecolor="0.3", linewidth=0.4,
                        )
                    )
                if np.isfinite(mat[di, fi]):
                    ax.text(
                        fi, di,
                        f"{mat[di, fi]:.3g}\nW={win_mat[di, fi]:.2f}",
                        ha="center", va="center", fontsize=5.5, color="black",
                    )

        if target in winners:
            w = winners[target]
            wd = str(w.get("selected_decoder", ""))
            wf = str(w.get("selected_feature_mode", ""))
            if wd in decoders and wf in features:
                ax.add_patch(
                    plt.Rectangle(
                        (features.index(wf) - 0.5, decoders.index(wd) - 0.5),
                        1, 1,
                        fill=False, edgecolor="#d4a017", linewidth=2.4,
                    )
                )

        # Vertical divider between spike features and manifold features
        n_spike = sum(1 for f in features if f in ("counts", "rates"))
        if 0 < n_spike < len(features):
            ax.axvline(n_spike - 0.5, color="0.35", lw=1.2, ls="--")

        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(decoders)))
        ax.set_xticklabels(
            [_short_label(f, 12) for f in features],
            rotation=35, ha="right", fontsize=6,
        )
        ax.set_yticklabels([_short_label(d, 16) for d in decoders], fontsize=6)
        ax.set_title(f"{target}\n({metric})", fontsize=8)

    for j in range(n, rows_n * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r + 1, c])
        ax.axis("off")

    fig.suptitle(
        "Manifold vs spikes one-pager (sorted / NPX) — choose feature mode per target",
        fontsize=14,
        y=0.995,
    )
    out_path = Path(out_dir) / "fig_manifold_vs_spikes_onepager.png"
    fig.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_fig_manifold_vs_spikes_onepager(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Write the counts-vs-manifold deployable onepager under decoder_comparison/."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else experiment_dir / "figures"
    out_dir = figures_dir / "decoder_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    from visualization.publication_decoding_plots import load_comparison_metrics

    comparison_dir = experiment_dir / "decoder_comparison"
    metrics = load_comparison_metrics(experiment_dir, prefer="sorted")
    if metrics.empty:
        # Fall back to empty frame; onepager can still use deployment scores.
        metrics = pd.DataFrame()
    return _plot_manifold_vs_spikes_onepager(
        metrics=metrics,
        comparison_dir=comparison_dir,
        out_dir=out_dir,
        experiment_dir=experiment_dir,
    )



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


def _plot_isomap_diagnostics(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Geometry / connectivity diagnostics for global_isomap comparison rows."""
    iso = metrics[metrics.get("feature_type", pd.Series(dtype=str)) == "global_isomap"].copy()
    if iso.empty and "manifold_type" in metrics.columns:
        iso = metrics[metrics["manifold_type"] == "isomap"].copy()
    if iso.empty:
        return

    sources = (
        sorted(iso["spike_source"].dropna().unique())
        if "spike_source" in iso.columns else [None]
    )
    for source in sources:
        sub = iso if source is None else iso[iso["spike_source"] == source]
        if sub.empty:
            continue
        suffix = f"_{source}" if source else ""

        if "n_neighbors" in sub.columns and "trustworthiness" in sub.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            g = sub.groupby("n_neighbors", as_index=False)["trustworthiness"].mean()
            ax.plot(g["n_neighbors"], g["trustworthiness"], marker="o")
            ax.set_xlabel("n_neighbors")
            ax.set_ylabel("trustworthiness")
            title = "Isomap trustworthiness vs n_neighbors"
            if source:
                title += f" ({source})"
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(
                out_dir / f"isomap_trustworthiness_vs_neighbors{suffix}.png",
                dpi=FIGURE_DPI,
            )
            plt.close(fig)

        if "n_neighbors" in sub.columns and "largest_component_fraction" in sub.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            g = sub.groupby("n_neighbors", as_index=False)["largest_component_fraction"].mean()
            ax.plot(g["n_neighbors"], g["largest_component_fraction"], marker="o", color="C1")
            ax.set_xlabel("n_neighbors")
            ax.set_ylabel("largest component fraction")
            ax.set_ylim(0, 1.05)
            title = "Isomap graph connectivity vs n_neighbors"
            if source:
                title += f" ({source})"
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(
                out_dir / f"isomap_connectivity_vs_neighbors{suffix}.png",
                dpi=FIGURE_DPI,
            )
            plt.close(fig)

        if "manifold_n_components" in sub.columns and "residual_variance" in sub.columns:
            fig, ax = plt.subplots(figsize=(6, 4))
            g = sub.groupby("manifold_n_components", as_index=False)["residual_variance"].mean()
            ax.plot(g["manifold_n_components"], g["residual_variance"], marker="o", color="C2")
            ax.set_xlabel("latent dim")
            ax.set_ylabel("residual variance")
            title = "Isomap residual variance vs latent dimension"
            if source:
                title += f" ({source})"
            ax.set_title(title)
            fig.tight_layout()
            fig.savefig(
                out_dir / f"isomap_residual_variance_vs_dim{suffix}.png",
                dpi=FIGURE_DPI,
            )
            plt.close(fig)


def _best_row_for_group(g: pd.DataFrame, metric: str) -> pd.Series | None:
    if metric not in g.columns or g.empty:
        return None
    lower = ("error" in metric) or metric.endswith("_deg")
    series = g[metric].dropna()
    if series.empty:
        return None
    idx = series.idxmin() if lower else series.idxmax()
    return g.loc[idx]


def _plot_isomap_decoder_comparisons(metrics: pd.DataFrame, out_dir: Path) -> None:
    """Isomap vs PCA vs counts figures, always stratified by spike_source.

    Mixing ground_truth and sorted rows before taking min/max was making raw
    counts look falsely dominant (GT counts beat sorted Isomap).
    """
    needed = {"feature_type", "target_name", "decoder_name", "primary_metric"}
    if not needed.issubset(metrics.columns):
        return
    keep_modes = ("counts", "global_pca", "global_isomap")
    sub = metrics[metrics["feature_type"].isin(keep_modes)].copy()
    if sub.empty:
        return

    sources = (
        sorted(sub["spike_source"].dropna().unique())
        if "spike_source" in sub.columns else [None]
    )
    for source in sources:
        src = sub if source is None else sub[sub["spike_source"] == source]
        if src.empty:
            continue
        # Prefer sources that actually include Isomap for these story plots.
        if "global_isomap" not in set(src["feature_type"]):
            continue
        suffix = f"_{source}" if source else ""
        _plot_position_feature_x_decoder(src, out_dir, suffix=suffix, source=source)
        _plot_best_representation_by_target(src, out_dir, suffix=suffix, source=source)
        _plot_position_best_bar(src, out_dir, suffix=suffix, source=source)


def _plot_position_feature_x_decoder(
    metrics: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str,
    source: str | None,
) -> None:
    pos = metrics[metrics["target_name"] == "position"]
    metric = "mean_position_error_cm"
    if pos.empty or metric not in pos.columns:
        return
    rows = []
    for (feat, dec), g in pos.groupby(["feature_type", "decoder_name"]):
        rows.append({
            "feature_type": feat,
            "decoder_name": dec,
            "error": float(g[metric].min()),
        })
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    # Stable order: counts, pca, isomap
    order = [f for f in ("counts", "global_pca", "global_isomap") if f in set(plot_df["feature_type"])]
    decoders = sorted(plot_df["decoder_name"].unique())
    x = np.arange(len(order))
    width = 0.8 / max(len(decoders), 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, dec in enumerate(decoders):
        vals = []
        for f in order:
            hit = plot_df[(plot_df["feature_type"] == f) & (plot_df["decoder_name"] == dec)]
            vals.append(float(hit["error"].min()) if not hit.empty else np.nan)
        ax.bar(x + i * width, vals, width=width, label=dec)

    # Mark overall best bar
    best_idx = plot_df["error"].idxmin()
    best = plot_df.loc[best_idx]
    ax.annotate(
        f"best: {best['feature_type']}\n{best['decoder_name']}\n{best['error']:.2f} cm",
        xy=(0.98, 0.98),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.6", alpha=0.9),
    )

    ax.set_xticks(x + width * (len(decoders) - 1) / 2)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel("mean position error (cm)  ↓ better")
    title = "Position decoding: counts / PCA / Isomap × decoder"
    if source:
        title += f"\n({source} spikes only — not mixed across sources)"
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    # Keep legacy filename for the primary/sorted-like story when possible
    legacy = out_dir / "isomap_vs_pca_decoder_position.png"
    path = out_dir / f"isomap_vs_pca_decoder_position{suffix}.png"
    fig.savefig(path, dpi=FIGURE_DPI)
    if source == "sorted" or (source is None and not legacy.exists()):
        fig.savefig(legacy, dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_position_best_bar(
    metrics: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str,
    source: str | None,
) -> None:
    """Single bar per representation: best decoder/window for position."""
    pos = metrics[metrics["target_name"] == "position"]
    metric = "mean_position_error_cm"
    if pos.empty or metric not in pos.columns:
        return
    rows = []
    for feat in ("counts", "global_pca", "global_isomap"):
        g = pos[pos["feature_type"] == feat]
        best = _best_row_for_group(g, metric)
        if best is None:
            continue
        rows.append({
            "feature_type": feat,
            "decoder_name": best["decoder_name"],
            "error": float(best[metric]),
            "k": best.get("manifold_n_components"),
            "nn": best.get("n_neighbors"),
        })
    if not rows:
        return
    plot_df = pd.DataFrame(rows)
    colors = {"counts": "C0", "global_pca": "C1", "global_isomap": "C2"}
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = np.arange(len(plot_df))
    bars = ax.bar(
        xs,
        plot_df["error"],
        color=[colors.get(f, "0.5") for f in plot_df["feature_type"]],
        edgecolor="k",
        linewidth=0.6,
    )
    winner = int(plot_df["error"].idxmin())
    bars[list(plot_df.index).index(winner)].set_linewidth(2.0)

    labels = []
    for _, r in plot_df.iterrows():
        extra = []
        if pd.notna(r.get("k")):
            extra.append(f"k={int(r['k'])}")
        if pd.notna(r.get("nn")):
            extra.append(f"nn={int(r['nn'])}")
        detail = f"{r['decoder_name']}"
        if extra:
            detail += "\n" + ", ".join(extra)
        labels.append(f"{r['feature_type']}\n{detail}")

    for i, (_, r) in enumerate(plot_df.iterrows()):
        ax.text(i, r["error"] + 0.15, f"{r['error']:.2f} cm", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("best mean position error (cm)  ↓ better")
    title = "Best position decoder per representation"
    if source:
        title += f" ({source})"
    ax.set_title(title)
    ax.set_ylim(0, max(plot_df["error"]) * 1.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"isomap_position_best_by_representation{suffix}.png", dpi=FIGURE_DPI)
    if source == "sorted":
        fig.savefig(out_dir / "isomap_position_best_by_representation.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_best_representation_by_target(
    metrics: pd.DataFrame,
    out_dir: Path,
    *,
    suffix: str,
    source: str | None,
) -> None:
    """Grouped bars: best score of counts / PCA / Isomap for each target."""
    keep = ("counts", "global_pca", "global_isomap")
    targets = [t for t in metrics["target_name"].dropna().unique() if pd.notna(t)]
    if not targets:
        return

    # Normalize each target to a "higher is better" score for display, plus
    # a companion raw-value annotation for the primary metric.
    records = []
    for target in sorted(targets):
        g = metrics[metrics["target_name"] == target]
        if g.empty:
            continue
        metric = str(g["primary_metric"].iloc[0])
        if metric not in g.columns:
            continue
        lower = ("error" in metric) or metric.endswith("_deg")
        for feat in keep:
            fg = g[g["feature_type"] == feat]
            best = _best_row_for_group(fg, metric)
            if best is None:
                continue
            raw = float(best[metric])
            records.append({
                "target": target,
                "feature_type": feat,
                "metric": metric,
                "raw": raw,
                "display": -raw if lower else raw,
                "decoder": best["decoder_name"],
                "lower_better": lower,
            })
    if not records:
        return
    plot_df = pd.DataFrame(records)
    targets_ord = sorted(plot_df["target"].unique())
    feats = [f for f in keep if f in set(plot_df["feature_type"])]
    x = np.arange(len(targets_ord))
    width = 0.8 / max(len(feats), 1)

    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, feat in enumerate(feats):
        vals = []
        for t in targets_ord:
            hit = plot_df[(plot_df["target"] == t) & (plot_df["feature_type"] == feat)]
            vals.append(float(hit["display"].iloc[0]) if not hit.empty else np.nan)
        ax.bar(x + i * width, vals, width=width, label=feat)

    # Star the winning feature per target
    for ti, t in enumerate(targets_ord):
        tg = plot_df[plot_df["target"] == t]
        if tg.empty:
            continue
        win = tg.loc[tg["display"].idxmax()]
        fi = feats.index(win["feature_type"]) if win["feature_type"] in feats else 0
        ax.plot(
            ti + fi * width + width / 2,
            win["display"],
            marker="*",
            color="k",
            markersize=10,
            linestyle="None",
        )

    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x + width * (len(feats) - 1) / 2)
    ax.set_xticklabels(targets_ord, rotation=25, ha="right")
    ax.set_ylabel("best score (errors negated so ↑ is better)")
    title = "Best counts / PCA / Isomap score by target"
    if source:
        title += f" ({source}; ★ = winning representation)"
    else:
        title += " (★ = winning representation)"
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"isomap_vs_pca_best_by_target{suffix}.png", dpi=FIGURE_DPI)
    if source == "sorted":
        fig.savefig(out_dir / "isomap_vs_pca_best_by_target.png", dpi=FIGURE_DPI)
    plt.close(fig)


# Backwards-compatible alias
def _plot_isomap_vs_pca_decoders(metrics: pd.DataFrame, out_dir: Path) -> None:
    _plot_isomap_decoder_comparisons(metrics, out_dir)
