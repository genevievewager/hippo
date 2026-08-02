"""Deployment / sorted-spike decoder-selection figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGURE_DPI = 150


def plot_deployment_selection_outputs(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> None:
    """Write sorted-only deployment selection figures (publication + onepagers)."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else experiment_dir / "figures"
    from visualization.publication_decoding_plots import plot_fig_deployment

    plot_fig_deployment(experiment_dir, figures_dir)
    plot_deployable_selection_onepagers(experiment_dir, figures_dir)


def plot_deployable_selection_onepagers(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Clean retired deployable companion pages under decoder_comparison/.

    Decoder × window heatmaps live in ``fig_decoder_x_window``; feature ×
    window in ``fig_feature_x_window``. The old blue-table winner onepager is
    also retired (duplicated ``fig_manifold_vs_spikes_onepager``).
    """
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else experiment_dir / "figures"
    out = figures_dir / "decoder_comparison"
    out.mkdir(parents=True, exist_ok=True)
    for stale in (
        out / "fig_manifold_decoding.png",
        out / "fig_deployable_decoder_x_window_heatmaps.png",
        out / "fig_deployable_winner_onepager.png",
        out / "deployable_winner_onepager.png",
        out / "fig_manifold_decoder_window_threeway.png",
    ):
        stale.unlink(missing_ok=True)
    return []


def _plot_best_decoder_by_target(best: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.2))
    targets = best["target_name"].tolist()
    values = best["best_metric_value"].astype(float).tolist()
    decoders = best["best_decoder_name"].astype(str).tolist()
    windows = best.get(
        "recommended_realtime_window_s", best["best_decode_window_s"]
    ).astype(float).tolist()
    x = np.arange(len(targets))
    ax.bar(x, values, color="C0", edgecolor="k", linewidth=0.5)
    for i, (d, w) in enumerate(zip(decoders, windows)):
        ax.text(i, values[i], f"  {d}\n  W={w:.3f}s", fontsize=7, va="bottom", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=25, ha="right")
    ax.set_ylabel("best metric value")
    ax.set_title("Best decoder by target (sorted spikes / deployable)")
    fig.tight_layout()
    fig.savefig(out / "best_decoder_by_target_sorted.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_selected_window_by_target(
    best: pd.DataFrame,
    registry: dict | None,
    out: Path,
) -> None:
    """Make uniform-window collapse visually obvious."""
    fig, ax = plt.subplots(figsize=(9, 4))
    targets = []
    windows = []
    decoders = []
    if registry and "targets" in registry:
        for t, cfg in registry["targets"].items():
            targets.append(t)
            windows.append(float(cfg["selected_causal_window_s"]))
            decoders.append(str(cfg["selected_decoder"]))
    else:
        wcol = (
            "recommended_realtime_window_s"
            if "recommended_realtime_window_s" in best.columns
            else "best_decode_window_s"
        )
        dcol = (
            "recommended_realtime_decoder_name"
            if "recommended_realtime_decoder_name" in best.columns
            else "best_decoder_name"
        )
        for _, row in best.iterrows():
            targets.append(str(row["target_name"]))
            windows.append(float(row[wcol]))
            decoders.append(str(row[dcol]))

    unique_dec = sorted(set(decoders))
    colors = {d: f"C{i % 10}" for i, d in enumerate(unique_dec)}
    hatches = {d: ["", "//", "xx", "..", "\\\\"][i % 5] for i, d in enumerate(unique_dec)}
    x = np.arange(len(targets))
    for i, (w, d) in enumerate(zip(windows, decoders)):
        ax.bar(
            i, w, color=colors[d], hatch=hatches[d], edgecolor="k", linewidth=0.6, label=d,
        )
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")
    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=25, ha="right")
    ax.set_ylabel("selected causal window (s)")
    ax.set_title("Selected causal window by target (sorted / deployable)")
    uniq = sorted({round(w, 6) for w in windows})
    if len(uniq) == 1:
        ax.annotate(
            f"WARNING: all targets → {uniq[0]:.3f} s\n"
            "Inspect all_sorted_window_scores.csv",
            xy=(0.98, 0.95),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=8,
            color="darkred",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="darkred"),
        )
    fig.tight_layout()
    fig.savefig(out / "selected_window_by_target_sorted.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_metric_vs_window(scores: pd.DataFrame, out: Path) -> None:
    targets = sorted(scores["target"].unique())
    n = len(targets)
    cols = min(4, max(n, 1))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.6 * rows), squeeze=False)
    for ax, target in zip(axes.ravel(), targets):
        sub = scores[scores["target"] == target]
        metric = sub["metric_name"].iloc[0]
        for decoder, g in sub.groupby("decoder"):
            # Best feature per window
            piv = g.groupby("causal_window_s")["metric_value"]
            agg = piv.max() if bool(g["higher_is_better"].iloc[0]) else piv.min()
            ax.plot(agg.index, agg.values, marker="o", label=decoder, linewidth=1.2)
        ax.set_title(target, fontsize=9)
        ax.set_xlabel("W (s)", fontsize=8)
        ax.set_ylabel(metric, fontsize=7)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=5, loc="best")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Metric vs causal window (sorted spikes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "metric_vs_window_by_target_sorted.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _plot_deployment_model_summary(registry: dict, out: Path) -> None:
    targets = list(registry.get("targets", {}).keys())
    if not targets:
        return
    fig, ax = plt.subplots(figsize=(10, max(3, 0.45 * len(targets) + 1)))
    lines = []
    for t in targets:
        cfg = registry["targets"][t]
        lines.append(
            f"{t}: {cfg['selected_decoder']} | "
            f"W={cfg['selected_causal_window_s']:.3f}s | "
            f"feat={cfg['selected_feature_mode']}"
        )
    ax.axis("off")
    ax.set_title(
        f"Deployable realtime models (sorted only)\n"
        f"update={registry.get('update_rate_hz', 20)} Hz / "
        f"{registry.get('update_interval_s', 0.05)} s",
        fontsize=11,
    )
    ax.text(
        0.02, 0.95, "\n".join(lines),
        transform=ax.transAxes, va="top", ha="left",
        family="monospace", fontsize=9,
    )
    note = registry.get("deployment_rule", "")
    ax.text(
        0.02, 0.05, note,
        transform=ax.transAxes, va="bottom", ha="left",
        fontsize=8, style="italic", wrap=True,
    )
    fig.tight_layout()
    fig.savefig(out / "deployment_model_summary.png", dpi=FIGURE_DPI)
    plt.close(fig)


def _short(name: str, max_len: int = 14) -> str:
    s = str(name)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _registry_winners(registry: dict | None) -> dict[str, dict]:
    if not registry or "targets" not in registry:
        return {}
    return {str(t): dict(cfg) for t, cfg in registry["targets"].items()}


def _best_over_windows(sub: pd.DataFrame) -> pd.DataFrame:
    """Collapse causal windows: keep best metric per (decoder, feature_mode)."""
    rows = []
    higher = bool(sub["higher_is_better"].iloc[0])
    for (dec, feat), g in sub.groupby(["decoder", "feature_mode"], sort=False):
        if higher:
            idx = g["metric_value"].idxmax()
        else:
            idx = g["metric_value"].idxmin()
        row = g.loc[idx]
        rows.append({
            "decoder": str(dec),
            "feature_mode": str(feat),
            "causal_window_s": float(row["causal_window_s"]),
            "metric_value": float(row["metric_value"]),
            "realtime_compatible": bool(row.get("realtime_compatible", True)),
        })
    return pd.DataFrame(rows)


def _plot_deployable_winner_onepager(
    scores: pd.DataFrame,
    registry: dict | None,
    out: Path,
) -> None:
    """One-pager: deployable winners + decoder×feature heatmaps (sorted only).

    Each target panel collapses over windows by taking the best metric for that
    (decoder, feature) pair and annotates the winning ``W``. The registry
    selection is outlined. Offline-only features (e.g. classic Isomap) are
    shown but hatched.
    """
    scores = scores.copy()
    if "spike_source" in scores.columns:
        scores = scores[scores["spike_source"].astype(str) == "sorted"]
    if scores.empty:
        return

    winners = _registry_winners(registry)
    targets = list(winners.keys()) if winners else sorted(scores["target"].unique())
    # Prefer a stable scientific order when present
    preferred = [
        "position", "speed", "acceleration", "head_direction",
        "distance_to_wall", "spatial_context", "movement_state", "wall_distance_bin",
    ]
    targets = [t for t in preferred if t in targets] + [
        t for t in targets if t not in preferred
    ]

    n = len(targets)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(16.5, 2.8 + 2.9 * rows))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[1.15] + [1.0] * rows, hspace=0.55, wspace=0.35)

    # ---- Top: winner summary table ----
    ax_tab = fig.add_subplot(gs[0, :])
    ax_tab.axis("off")
    table_rows = []
    for t in targets:
        sub = scores[scores["target"] == t]
        metric = str(sub["metric_name"].iloc[0]) if not sub.empty else ""
        if t in winners:
            cfg = winners[t]
            table_rows.append([
                t,
                _short(cfg.get("selected_decoder", ""), 22),
                _short(cfg.get("selected_feature_mode", ""), 18),
                f"{float(cfg.get('selected_causal_window_s', float('nan'))):.3f}",
                f"{float(cfg.get('metric_value', float('nan'))):.3g}",
                metric,
            ])
        else:
            table_rows.append([t, "—", "—", "—", "—", metric])

    tbl = ax_tab.table(
        cellText=table_rows,
        colLabels=["target", "decoder", "feature", "W (s)", "metric", "metric name"],
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2f4f6f")
            cell.set_text_props(color="white", weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f3f6f9")
    ax_tab.set_title(
        "Deployable winners — sorted / Neuropixels only  "
        f"(policy={winners and next(iter(winners.values())).get('selection_policy', 'registry') or 'n/a'}; "
        "heatmaps = best W per decoder×feature; gold box = selected)",
        fontsize=12,
        pad=8,
    )

    # ---- Per-target heatmaps: decoder × feature ----
    for i, target in enumerate(targets):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r + 1, c])
        sub = scores[scores["target"] == target]
        if sub.empty:
            ax.axis("off")
            continue
        higher = bool(sub["higher_is_better"].iloc[0])
        metric = str(sub["metric_name"].iloc[0])
        collapsed = _best_over_windows(sub)
        decoders = sorted(collapsed["decoder"].unique())
        features = sorted(collapsed["feature_mode"].unique())
        # Matrix of metric values; NaN if missing
        mat = np.full((len(decoders), len(features)), np.nan)
        win_mat = np.full((len(decoders), len(features)), np.nan)
        rt_mat = np.ones((len(decoders), len(features)), dtype=bool)
        for _, row in collapsed.iterrows():
            di = decoders.index(row["decoder"])
            fi = features.index(row["feature_mode"])
            mat[di, fi] = row["metric_value"]
            win_mat[di, fi] = row["causal_window_s"]
            rt_mat[di, fi] = bool(row["realtime_compatible"])

        # Display so "better" is darker green regardless of metric direction
        display = mat.copy()
        if not higher:
            display = -display
        # Robust color scale over finite cells
        finite = display[np.isfinite(display)]
        if finite.size:
            vmin, vmax = float(np.nanpercentile(finite, 5)), float(np.nanpercentile(finite, 95))
            if vmin == vmax:
                vmin, vmax = float(np.nanmin(finite)), float(np.nanmax(finite) + 1e-9)
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(display, aspect="auto", cmap="YlGn", vmin=vmin, vmax=vmax)
        # Hatch offline-only cells
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
                    txt = f"{mat[di, fi]:.3g}\nW={win_mat[di, fi]:.2f}"
                    ax.text(fi, di, txt, ha="center", va="center", fontsize=5.5, color="black")

        # Outline registry winner
        if target in winners:
            w = winners[target]
            wd = str(w.get("selected_decoder", ""))
            wf = str(w.get("selected_feature_mode", ""))
            if wd in decoders and wf in features:
                di = decoders.index(wd)
                fi = features.index(wf)
                ax.add_patch(
                    plt.Rectangle(
                        (fi - 0.5, di - 0.5), 1, 1,
                        fill=False, edgecolor="#d4a017", linewidth=2.4,
                    )
                )

        ax.set_xticks(np.arange(len(features)))
        ax.set_yticks(np.arange(len(decoders)))
        ax.set_xticklabels([_short(f, 12) for f in features], rotation=35, ha="right", fontsize=6)
        ax.set_yticklabels([_short(d, 16) for d in decoders], fontsize=6)
        ax.set_title(f"{target}\n({metric})", fontsize=8)
        # colorbar per panel is crowded; skip — greener = better

    # Hide unused axes
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r + 1, c])
        ax.axis("off")

    fig.suptitle(
        "Deployable decoder selection one-pager (sorted / NPX spikes)",
        fontsize=14,
        y=0.995,
    )
    fig.savefig(out / "fig_deployable_winner_onepager.png", dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
