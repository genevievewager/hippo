"""Publication multi-panel Isomap / latent-geometry figures (Figs 6–8).

Gracefully annotates empty-state panels when ``global_isomap`` was not run.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec

from visualization.constants import FIGURE_DPI, FIGURE_SUBDIR_DECODER
from visualization.publication_decoding_plots import (
    _best_row,
    _empty_panel,
    _is_lower_better,
    _metric_for_target,
    _read_csv,
    load_comparison_metrics,
)
from visualization.publication_style import legend_below, legend_outside, panel_label, save_pub_figure

sns.set_theme(style="ticks", context="paper", font_scale=1.0)

MAX_EMBED_POINTS = 2500


def _find_transform_dir(comparison_dir: Path, prefixes: tuple[str, ...]) -> Path | None:
    metas = [
        p for p in Path(comparison_dir).rglob("meta.json")
        if p.parent.name.startswith(prefixes)
    ]
    if not metas:
        return None
    # Prefer mid-window (~250 ms) if present
    for p in metas:
        if "_w0250ms" in p.parent.name or "_w0100ms" in p.parent.name:
            return p.parent
    return metas[0].parent


def _load_heldout_embedding(
    experiment_dir: Path,
    transform_dir: Path,
    *,
    color_keys: tuple[str, ...] = ("x", "speed", "spatial_context"),
) -> tuple[np.ndarray, pd.DataFrame, str] | None:
    """Transform held-out (or exploratory) spike windows with a saved encoder."""
    try:
        from realtime.data_loading import load_simulation_data, make_decode_times
        from realtime.decoding_targets import align_extended_behavior_to_decoder_times
        from realtime.manifold_features import load_feature_transformer
        from realtime.spike_features import build_causal_spike_matrix
        from realtime.timing import extract_behavior_times
        from realtime.train_decoder import causal_train_test_split
    except Exception:
        return None

    experiment_dir = Path(experiment_dir)
    if not (experiment_dir / "behavior.csv").exists():
        return None

    try:
        transformer = load_feature_transformer(transform_dir)
    except Exception:
        return None

    name = transform_dir.name
    try:
        w_ms = int(name.split("_w")[-1].replace("ms", ""))
        decode_window = w_ms / 1000.0
    except Exception:
        decode_window = 0.250

    try:
        data = load_simulation_data(experiment_dir, "sorted")
        behavior_times = extract_behavior_times(data["behavior_df"])
        decode_times = make_decode_times(
            data["session_duration"], decode_window, 0.05, behavior_times=behavior_times,
        )
        aligned = align_extended_behavior_to_decoder_times(
            data["behavior_df"], decode_times, data["summary"]
        )
        X = build_causal_spike_matrix(
            data["spikes_df"], data["unit_ids"], decode_times, decode_window,
        )
        train_mask, test_mask = causal_train_test_split(decode_times, 0.70)
        # Prefer held-out transform (train-fit / test-transform)
        if test_mask is not None and np.asarray(test_mask).any():
            mask = np.asarray(test_mask)
            tag = "held_out_test_transform"
        else:
            mask = np.ones(len(decode_times), dtype=bool)
            tag = "exploratory_full_session_embedding"
        Z = transformer.transform(X[mask])
        beh = aligned.iloc[np.where(mask)[0]].reset_index(drop=True)
        if Z.shape[0] > MAX_EMBED_POINTS:
            rng = np.random.default_rng(0)
            idx = rng.choice(Z.shape[0], size=MAX_EMBED_POINTS, replace=False)
            Z = Z[idx]
            beh = beh.iloc[idx].reset_index(drop=True)
        return Z, beh, tag
    except Exception:
        return None


def _scatter_latent(ax, Z: np.ndarray, color, *, cmap="viridis", label: str = "") -> None:
    is_series = isinstance(color, pd.Series)
    categorical = False
    if is_series:
        if color.dtype == object or str(color.dtype) == "category" or pd.api.types.is_string_dtype(color):
            categorical = True
        else:
            try:
                pd.to_numeric(color, errors="raise")
            except (TypeError, ValueError):
                categorical = True
    if categorical:
        cats = color.astype(str)
        uniq = sorted(cats.unique())
        palette = sns.color_palette("deep", n_colors=max(len(uniq), 1))
        lut = {u: palette[i] for i, u in enumerate(uniq)}
        c = [lut[u] for u in cats]
        ax.scatter(Z[:, 0], Z[:, 1], c=c, s=4, alpha=0.65)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=lut[u], markersize=6, label=u)
            for u in uniq[:8]
        ]
        legend_below(ax, handles=handles, labels=[h.get_label() for h in handles], ncol=min(3, len(handles)), fontsize=6)
    else:
        vals = np.asarray(color, dtype=float)
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=vals, s=4, cmap=cmap, alpha=0.65)
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    sns.despine(ax=ax)


# ---------------------------------------------------------------------------
# Fig 6 — Latent geometry
# ---------------------------------------------------------------------------

def plot_fig_latent_geometry(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 6: PCA + Isomap embeddings colored by behavior."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    comparison_dir = experiment_dir / "decoder_comparison"
    metrics = load_comparison_metrics(experiment_dir)
    has_isomap = (
        not metrics.empty
        and "feature_type" in metrics.columns
        and metrics["feature_type"].astype(str).str.contains("isomap").any()
    )

    pca_dir = _find_transform_dir(comparison_dir, ("global_pca_",))
    iso_dir = _find_transform_dir(comparison_dir, ("global_isomap_",)) if has_isomap else None

    pca_pack = _load_heldout_embedding(experiment_dir, pca_dir) if pca_dir else None
    iso_pack = _load_heldout_embedding(experiment_dir, iso_dir) if iso_dir else None

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.30)

    ax_a = fig.add_subplot(gs[0, 0])
    if pca_pack is not None:
        Z, beh, tag = pca_pack
        color = beh["x"] if "x" in beh.columns else np.arange(len(beh))
        _scatter_latent(ax_a, Z, color, label="position x (cm)")
    else:
        _empty_panel(ax_a, "PCA embedding unavailable")
    panel_label(ax_a, "A")

    ax_b = fig.add_subplot(gs[0, 1])
    if iso_pack is not None:
        Z, beh, tag = iso_pack
        color = beh["x"] if "x" in beh.columns else np.arange(len(beh))
        _scatter_latent(ax_b, Z, color, label="position x (cm)")
    else:
        msg = "Isomap not run\n(re-run with --feature-modes … global_isomap)"
        _empty_panel(ax_b, msg)
    panel_label(ax_b, "B")

    ax_c = fig.add_subplot(gs[1, 0])
    if iso_pack is not None:
        Z, beh, tag = iso_pack
        if "spatial_context" in beh.columns:
            _scatter_latent(ax_c, Z, beh["spatial_context"], label="spatial_context")
        elif "speed" in beh.columns:
            _scatter_latent(ax_c, Z, beh["speed"], cmap="magma", label="speed")
        else:
            _empty_panel(ax_c, "No context/speed columns")
    elif pca_pack is not None:
        Z, beh, tag = pca_pack
        if "spatial_context" in beh.columns:
            _scatter_latent(ax_c, Z, beh["spatial_context"], label="spatial_context")
        elif "speed" in beh.columns:
            _scatter_latent(ax_c, Z, beh["speed"], cmap="magma", label="speed")
        else:
            _empty_panel(ax_c, "Isomap not run")
    else:
        _empty_panel(ax_c, "Isomap not run")
    panel_label(ax_c, "C")

    ax_d = fig.add_subplot(gs[1, 1])
    if pca_pack is not None and iso_pack is not None:
        Zp, beh_p, _ = pca_pack
        Zi, beh_i, _ = iso_pack
        n = min(len(Zp), len(Zi), 800)
        ax_d.scatter(Zp[:n, 0], Zp[:n, 1], s=3, alpha=0.4, label="PCA", c="C0")
        ax_d.scatter(Zi[:n, 0], Zi[:n, 1], s=3, alpha=0.4, label="Isomap", c="C1")
        ax_d.set_xlabel("z₁")
        ax_d.set_ylabel("z₂")
        legend_below(ax_d, ncol=2, fontsize=6)
        sns.despine(ax=ax_d)
    elif pca_pack is not None:
        Z, beh, tag = pca_pack
        color = beh["speed"] if "speed" in beh.columns else (
            beh["x"] if "x" in beh.columns else np.arange(len(beh))
        )
        lab = "speed" if "speed" in beh.columns else "position x"
        _scatter_latent(ax_d, Z, color, cmap="magma", label=lab)
    else:
        _empty_panel(ax_d, "No embeddings available")
    panel_label(ax_d, "D")

    return save_pub_figure(
        fig, out_dir / "fig_latent_geometry.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.12, 0.98, 0.94),
    )


# ---------------------------------------------------------------------------
# Fig 7 — Isomap diagnostics
# ---------------------------------------------------------------------------

def plot_fig_isomap_diagnostics(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 7: trustworthiness, connectivity, residual variance, geodesic/knn."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir)
    iso = pd.DataFrame()
    if not metrics.empty:
        if "feature_type" in metrics.columns:
            iso = metrics[metrics["feature_type"].astype(str).str.contains("isomap", na=False)].copy()
        if iso.empty and "manifold_type" in metrics.columns:
            iso = metrics[metrics["manifold_type"] == "isomap"].copy()

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    def _diag_or_empty(ax, label: str, draw_fn) -> None:
        panel_label(ax, label)
        if iso.empty:
            _empty_panel(ax, "Isomap not run\n(no geometry diagnostics)")
            return
        draw_fn(ax)

    ax_a = fig.add_subplot(gs[0, 0])

    def draw_trust(ax):
        if "trustworthiness" not in iso.columns or "n_neighbors" not in iso.columns:
            _empty_panel(ax, "No trustworthiness column")
            return
        g = iso.groupby("n_neighbors")["trustworthiness"].mean()
        ax.plot(g.index, g.values, marker="o", lw=1.4)
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel("Trustworthiness")
        sns.despine(ax=ax)

    _diag_or_empty(ax_a, "A", draw_trust)

    ax_b = fig.add_subplot(gs[0, 1])

    def draw_conn(ax):
        ycol = "largest_component_fraction" if "largest_component_fraction" in iso.columns else None
        if ycol is None or "n_neighbors" not in iso.columns:
            _empty_panel(ax, "No connectivity columns")
            return
        g = iso.groupby("n_neighbors")[ycol].mean()
        ax.plot(g.index, g.values, marker="o", lw=1.4, color="C1")
        if "graph_connected" in iso.columns:
            conn = iso.groupby("n_neighbors")["graph_connected"].mean()
            ax.plot(conn.index, conn.values, marker="s", lw=1.0, color="C2", label="frac connected")
            legend_below(ax, ncol=1, fontsize=6)
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel("Largest component fraction")
        ax.set_ylim(0, 1.05)
        sns.despine(ax=ax)

    _diag_or_empty(ax_b, "B", draw_conn)

    ax_c = fig.add_subplot(gs[1, 0])

    def draw_resid(ax):
        if "residual_variance" not in iso.columns:
            _empty_panel(ax, "No residual_variance")
            return
        xcol = "manifold_n_components" if "manifold_n_components" in iso.columns else None
        if xcol is None:
            _empty_panel(ax, "No latent dim column")
            return
        g = iso.groupby(xcol)["residual_variance"].mean()
        ax.plot(g.index, g.values, marker="o", lw=1.4, color="C3")
        ax.set_xlabel("Latent dim")
        ax.set_ylabel("Residual variance")
        sns.despine(ax=ax)

    _diag_or_empty(ax_c, "C", draw_resid)

    ax_d = fig.add_subplot(gs[1, 1])

    def draw_geo(ax):
        plotted = False
        if "geodesic_distance_correlation" in iso.columns and "n_neighbors" in iso.columns:
            g = iso.groupby("n_neighbors")["geodesic_distance_correlation"].mean()
            if g.notna().any():
                ax.plot(g.index, g.values, marker="o", lw=1.4, label="geodesic corr")
                plotted = True
        knn_cols = [c for c in iso.columns if "knn_overlap" in c or c.startswith("continuity")]
        for c in knn_cols[:2]:
            if "n_neighbors" not in iso.columns:
                break
            g = iso.groupby("n_neighbors")[c].mean()
            if g.notna().any():
                ax.plot(g.index, g.values, marker="s", lw=1.2, label=c)
                plotted = True
        if not plotted:
            # Fall back: show trustworthiness vs residual as scatter summary
            if {"trustworthiness", "residual_variance"}.issubset(iso.columns):
                ax.scatter(iso["trustworthiness"], iso["residual_variance"], s=12, alpha=0.7)
                ax.set_xlabel("Trustworthiness")
                ax.set_ylabel("Residual variance")
                sns.despine(ax=ax)
            else:
                _empty_panel(ax, "No geodesic / knn_overlap columns")
            return
        ax.set_xlabel("n_neighbors")
        ax.set_ylabel("Preservation metric")
        legend_below(ax, ncol=2, fontsize=6)
        sns.despine(ax=ax)

    _diag_or_empty(ax_d, "D", draw_geo)

    return save_pub_figure(
        fig, out_dir / "fig_isomap_diagnostics.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.10, 0.98, 0.94),
    )


# ---------------------------------------------------------------------------
# Fig 8 — Isomap decoding story + distillation
# ---------------------------------------------------------------------------

def plot_fig_isomap_story(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Fig 8: counts vs PCA vs Isomap + distilled latency/accuracy."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir)
    teacher = _read_csv(
        experiment_dir / "latency_profiling" / "isomap_teacher_vs_distilled_latency.csv"
    )

    fig = plt.figure(figsize=(10.5, 7.2))
    gs = GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    modes = ("counts", "global_pca", "global_isomap", "global_isomap_distilled")
    has_iso = (
        not metrics.empty
        and "feature_type" in metrics.columns
        and any(m in set(metrics["feature_type"].astype(str)) for m in ("global_isomap", "global_isomap_distilled"))
    )

    # A — best score by representation × target (selected targets)
    ax_a = fig.add_subplot(gs[0, 0])
    if not metrics.empty and "feature_type" in metrics.columns:
        present = [m for m in modes if m in set(metrics["feature_type"])]
        targets = [t for t in ("position", "speed", "spatial_context", "head_direction")
                   if t in set(metrics["target_name"])]
        rows = []
        for target in targets:
            sub_t = metrics[metrics["target_name"] == target]
            metric = _metric_for_target(sub_t, target)
            for mode in present:
                sub = sub_t[sub_t["feature_type"] == mode]
                row = _best_row(sub, metric)
                if row is None:
                    continue
                val = float(row[metric])
                # Convert errors to a "higher is better" display via negation for mixed plot —
                # keep raw value and annotate metric in title.
                rows.append({"target": target, "feature": mode, "value": val, "metric": metric})
        rdf = pd.DataFrame(rows)
        if not rdf.empty:
            # Split continuous error targets vs accuracy for clarity: normalize per target
            for target in rdf["target"].unique():
                mask = rdf["target"] == target
                metric = rdf.loc[mask, "metric"].iloc[0]
                vals = rdf.loc[mask, "value"].to_numpy()
                if _is_lower_better(metric):
                    lo, hi = vals.min(), vals.max()
                    rdf.loc[mask, "norm"] = (hi - vals) / (hi - lo + 1e-12)
                else:
                    lo, hi = vals.min(), vals.max()
                    rdf.loc[mask, "norm"] = (vals - lo) / (hi - lo + 1e-12)
            sns.barplot(data=rdf, x="target", y="norm", hue="feature", ax=ax_a)
            ax_a.set_ylabel("Normalized best (1=best)")
            ax_a.set_xlabel("")
            legend_outside(ax_a, fontsize=6, ncol=1, bbox=(1.02, 1.0))
            ax_a.tick_params(axis="x", rotation=20, labelsize=7)
            ax_a.set_ylim(0, 1.15)
            sns.despine(ax=ax_a)
        else:
            _empty_panel(ax_a, "No representation scores")
    else:
        _empty_panel(ax_a, "No comparison metrics")
    # Empty-state note goes in the title, not over bars
    panel_label(ax_a, "A")

    # B — decoder family × representation for position (or first target)
    ax_b = fig.add_subplot(gs[0, 1])
    if not metrics.empty and "feature_type" in metrics.columns:
        target = "position" if "position" in set(metrics["target_name"]) else metrics["target_name"].iloc[0]
        sub = metrics[metrics["target_name"] == target]
        present = [m for m in modes if m in set(sub["feature_type"])]
        metric = _metric_for_target(sub, target)
        pivot = sub[sub["feature_type"].isin(present)].pivot_table(
            index="decoder_name", columns="feature_type", values=metric,
            aggfunc="min" if _is_lower_better(metric) else "max",
        )
        if not pivot.empty:
            # Reorder columns
            cols = [c for c in modes if c in pivot.columns]
            pivot = pivot[cols]
            sns.heatmap(pivot, ax=ax_b, annot=True, fmt=".2f", cmap="viridis",
                        cbar_kws={"label": metric})
            ax_b.tick_params(labelsize=6)
        else:
            _empty_panel(ax_b, "No decoder×feature matrix")
    else:
        _empty_panel(ax_b, "No metrics")
    panel_label(ax_b, "B")

    # C — teacher vs distilled latency
    ax_c = fig.add_subplot(gs[1, 0])
    if not teacher.empty:
        name_col = "method" if "method" in teacher.columns else "name"
        mean_col = "mean_ms" if "mean_ms" in teacher.columns else None
        if mean_col:
            y = np.arange(len(teacher))
            colors = [
                "#2ca02c" if bool(r.get("realtime_compatible", False)) else "#d62728"
                for _, r in teacher.iterrows()
            ]
            ax_c.barh(y, teacher[mean_col], color=colors, edgecolor="k", lw=0.3)
            ax_c.set_yticks(y)
            ax_c.set_yticklabels(teacher[name_col].astype(str).tolist(), fontsize=8)
            ax_c.axvline(50.0, color="0.3", ls="--", lw=1.0)
            ax_c.set_xlabel("Latency (ms)")
            sns.despine(ax=ax_c)
        else:
            _empty_panel(ax_c, "No latency means")
    else:
        _empty_panel(ax_c, "No teacher/distilled latency CSV")
    panel_label(ax_c, "C")

    # D — distilled vs classic accuracy if both feature modes present
    ax_d = fig.add_subplot(gs[1, 1])
    if has_iso and not metrics.empty:
        rows = []
        for mode in ("global_isomap", "global_isomap_distilled"):
            if mode not in set(metrics["feature_type"]):
                continue
            for target, g in metrics[metrics["feature_type"] == mode].groupby("target_name"):
                metric = _metric_for_target(g, target)
                row = _best_row(g, metric)
                if row is None:
                    continue
                rows.append({
                    "target": target,
                    "mode": "classic (offline)" if mode == "global_isomap" else "distilled",
                    "value": float(row[metric]),
                    "metric": metric,
                })
        rdf = pd.DataFrame(rows)
        if not rdf.empty and rdf["mode"].nunique() >= 2:
            # Normalize per target
            for target in rdf["target"].unique():
                mask = rdf["target"] == target
                metric = rdf.loc[mask, "metric"].iloc[0]
                vals = rdf.loc[mask, "value"].to_numpy()
                if _is_lower_better(metric):
                    lo, hi = vals.min(), vals.max()
                    rdf.loc[mask, "norm"] = (hi - vals) / (hi - lo + 1e-12) if hi > lo else 0.5
                else:
                    lo, hi = vals.min(), vals.max()
                    rdf.loc[mask, "norm"] = (vals - lo) / (hi - lo + 1e-12) if hi > lo else 0.5
            sns.barplot(data=rdf, x="target", y="norm", hue="mode", ax=ax_d)
            ax_d.set_ylabel("Normalized best")
            ax_d.tick_params(axis="x", rotation=25, labelsize=7)
            legend_below(ax_d, ncol=2, fontsize=6)
            sns.despine(ax=ax_d)
        elif not rdf.empty:
            sns.barplot(data=rdf, x="target", y="value", hue="mode", ax=ax_d)
            ax_d.tick_params(axis="x", rotation=25, labelsize=7)
            legend_below(ax_d, ncol=2, fontsize=6)
            sns.despine(ax=ax_d)
        else:
            _empty_panel(ax_d, "No Isomap accuracy rows")
    else:
        _empty_panel(ax_d, "Isomap not run\n(no teacher vs distilled accuracy)")
    panel_label(ax_d, "D")

    return save_pub_figure(
        fig, out_dir / "fig_isomap_story.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.12, 0.98, 0.94),
    )


def generate_publication_isomap_figures(
    experiment_dir: Path,
    figures_dir: Path | None = None,
    *,
    cleanup_legacy: bool = True,
) -> list[Path]:
    """Write Figs 6–8 under figures/decoder_comparison/."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    written: list[Path] = []
    for fn in (plot_fig_latent_geometry, plot_fig_isomap_diagnostics, plot_fig_isomap_story):
        try:
            path = fn(experiment_dir, figures_dir)
            if path is not None:
                written.append(path)
                print(f"  wrote {path.relative_to(figures_dir)}")
        except Exception as exc:
            print(f"  warning: {fn.__name__} skipped ({exc})")

    if cleanup_legacy:
        from visualization.publication_decoding_plots import _cleanup_legacy_pngs

        _cleanup_legacy_pngs(figures_dir / FIGURE_SUBDIR_DECODER)

    return written
