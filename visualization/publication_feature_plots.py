"""Panneled neural-feature overview figures (Feature Explorer).

Pages mirror Manifold Explorer's panneled layout:

- ``fig_feature_panel_variance.png``
- ``fig_feature_panel_traces.png``
- ``fig_feature_panel_correlation.png``

Each panel is one neural feature set. Window selection:

1. Before decoder comparison metrics exist → 250 ms for every feature set.
2. After decoding → each panel uses that feature set's own best decode
   window (ranked independently), even if another feature set won overall.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from realtime.decoder_comparison import PRIMARY_METRIC
from visualization.artifact_manifest import register_artifact
from visualization.constants import FIGURE_DPI, FIGURE_SUBDIR_FEATURES
from visualization.publication_style import apply_publication_theme, style_axes

DEFAULT_FEATURE_PANEL_WINDOW_S = 0.250
PANEL_KINDS = ("variance", "traces", "correlation")


def _figures_dir(experiment_dir: Path, figures_dir: Path | None = None) -> Path:
    root = Path(figures_dir) if figures_dir is not None else Path(experiment_dir) / "figures"
    out = root / FIGURE_SUBDIR_FEATURES
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_metrics(experiment_dir: Path) -> pd.DataFrame | None:
    from ui.services.results import find_metrics_csv

    path = find_metrics_csv(Path(experiment_dir))
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return df if df is not None and not df.empty else None


def _row_primary_score(row: pd.Series) -> float:
    target = str(row.get("target_name", ""))
    metric, direction = PRIMARY_METRIC.get(target, (None, None))
    if metric is None or metric not in row.index or pd.isna(row.get(metric)):
        return float("nan")
    value = float(row[metric])
    # Unified "higher is better" score for ranking within a feature set.
    return -value if direction == "lower" else value


def best_decode_window_by_feature_set(
    experiment_dir: Path,
    *,
    feature_sets: list[str] | tuple[str, ...] | None = None,
    fallback_s: float = DEFAULT_FEATURE_PANEL_WINDOW_S,
    spike_source: str = "sorted",
) -> dict[str, float]:
    """Return each feature set's best causal window from decoder metrics.

    Ranking is **per feature set**: for each set, take the decode window
    whose best (target×decoder) primary score averages highest. A feature
    set that did not win overall still gets its own best window.
    """
    df = _load_metrics(experiment_dir)
    if df is None or "feature_set" not in df.columns or "decode_window_s" not in df.columns:
        sets = list(feature_sets or [])
        return {fs: float(fallback_s) for fs in sets} if sets else {}

    work = df.copy()
    if spike_source and "spike_source" in work.columns:
        filtered = work[work["spike_source"].astype(str) == str(spike_source)]
        if not filtered.empty:
            work = filtered

    if feature_sets:
        wanted = set(feature_sets)
        work = work[work["feature_set"].astype(str).isin(wanted)]

    scores = work.apply(_row_primary_score, axis=1)
    work = work.assign(_score=scores)
    work = work[np.isfinite(work["_score"])]
    if work.empty:
        sets = list(feature_sets or sorted(df["feature_set"].astype(str).unique()))
        return {fs: float(fallback_s) for fs in sets}

    # Best score per (feature_set, window, target), then mean across targets.
    group_cols = ["feature_set", "decode_window_s"]
    if "target_name" in work.columns:
        best_per_target = (
            work.groupby(["feature_set", "decode_window_s", "target_name"], dropna=False)["_score"]
            .max()
            .reset_index()
        )
        summary = (
            best_per_target.groupby(group_cols, dropna=False)["_score"]
            .mean()
            .reset_index()
        )
    else:
        summary = (
            work.groupby(group_cols, dropna=False)["_score"]
            .max()
            .reset_index()
        )

    out: dict[str, float] = {}
    for fs, g in summary.groupby("feature_set", dropna=False):
        idx = g["_score"].idxmax()
        out[str(fs)] = float(g.loc[idx, "decode_window_s"])

    if feature_sets:
        for fs in feature_sets:
            out.setdefault(str(fs), float(fallback_s))
    return out


def resolve_feature_panel_windows(
    experiment_dir: Path,
    feature_sets: list[str] | tuple[str, ...],
    *,
    fallback_s: float = DEFAULT_FEATURE_PANEL_WINDOW_S,
    spike_source: str = "sorted",
) -> dict[str, dict[str, Any]]:
    """Map feature_set → {window_s, source} where source is metrics|default."""
    best = best_decode_window_by_feature_set(
        experiment_dir,
        feature_sets=feature_sets,
        fallback_s=fallback_s,
        spike_source=spike_source,
    )
    metrics = _load_metrics(experiment_dir)
    has_metrics = metrics is not None and "feature_set" in (metrics.columns if metrics is not None else [])
    resolved: dict[str, dict[str, Any]] = {}
    for fs in feature_sets:
        if has_metrics and fs in best and metrics is not None:
            present = set(metrics["feature_set"].astype(str).unique())
            if fs in present:
                resolved[fs] = {"window_s": float(best[fs]), "source": "decoder_metrics"}
                continue
        resolved[fs] = {"window_s": float(fallback_s), "source": "default_250ms"}
    return resolved


def _grid_shape(n: int) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    if n == 1:
        return 1, 1
    if n == 2:
        return 1, 2
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    cols = 3
    rows = int(np.ceil(n / cols))
    return rows, cols


def _draw_variance(ax, diag) -> None:
    if diag.variance is None or len(diag.variance) == 0:
        ax.text(0.5, 0.5, "no variance", ha="center", va="center", transform=ax.transAxes)
        return
    ax.plot(np.arange(len(diag.variance)), diag.variance, lw=1.2, color="steelblue")
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Variance")


def _draw_traces(ax, diag) -> None:
    traces = diag.example_traces
    value_cols = [c for c in traces.columns if c != "time_s"]
    if not value_cols or "time_s" not in traces.columns:
        ax.text(0.5, 0.5, "no traces", ha="center", va="center", transform=ax.transAxes)
        return
    t = traces["time_s"].to_numpy()
    for col in value_cols[:6]:
        ax.plot(t, traces[col].to_numpy(), lw=0.9, label=col)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    if len(value_cols) <= 6:
        ax.legend(fontsize=7, loc="upper right", frameon=False)


def _draw_correlation(ax, diag) -> None:
    corr = diag.correlation_subset
    if corr is None or corr.empty:
        ax.text(0.5, 0.5, "no correlation", ha="center", va="center", transform=ax.transAxes)
        return
    im = ax.imshow(corr.to_numpy(), aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def plot_feature_panel_pages(
    experiment_dir: Path,
    feature_sets: list[str] | tuple[str, ...],
    *,
    spike_source: str = "sorted",
    figures_dir: Path | None = None,
    fallback_window_s: float = DEFAULT_FEATURE_PANEL_WINDOW_S,
    run_id: str | None = None,
    progress_callback=None,
) -> dict[str, Any]:
    """Write the three panneled feature overview pages."""
    from ui.services.features import compute_feature_diagnostics

    apply_publication_theme()
    experiment_dir = Path(experiment_dir)
    out_dir = _figures_dir(experiment_dir, figures_dir)
    resolved = resolve_feature_panel_windows(
        experiment_dir,
        feature_sets,
        fallback_s=fallback_window_s,
        spike_source=spike_source,
    )

    diagnostics: dict[str, Any] = {}
    n = max(len(feature_sets), 1)
    for i, fs in enumerate(feature_sets, start=1):
        window = float(resolved[fs]["window_s"])
        if progress_callback:
            progress_callback(f"Feature panel `{fs}` @ {window:.3f}s", i, n + len(PANEL_KINDS))
        diagnostics[fs] = compute_feature_diagnostics(
            experiment_dir,
            fs,
            spike_source=spike_source,
            decode_window=window,
        )

    paths: dict[str, str] = {}
    drawers = {
        "variance": _draw_variance,
        "traces": _draw_traces,
        "correlation": _draw_correlation,
    }
    titles = {
        "variance": "Feature variance by feature set",
        "traces": "Top-variance feature traces by feature set",
        "correlation": "Feature correlation (subset) by feature set",
    }

    rows, cols = _grid_shape(len(feature_sets))
    for j, kind in enumerate(PANEL_KINDS, start=1):
        if progress_callback:
            progress_callback(f"Writing fig_feature_panel_{kind}", n + j, n + len(PANEL_KINDS))
        fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows), squeeze=False)
        last_im = None
        for idx, fs in enumerate(feature_sets):
            r, c = divmod(idx, cols)
            ax = axes[r][c]
            info = resolved[fs]
            w_ms = int(round(float(info["window_s"]) * 1000))
            src = "best W" if info["source"] == "decoder_metrics" else "250 ms"
            ax.set_title(f"{fs}\nW={w_ms} ms ({src})", fontsize=11)
            im = drawers[kind](ax, diagnostics[fs])
            if im is not None:
                last_im = im
            style_axes(ax)
        for idx in range(len(feature_sets), rows * cols):
            r, c = divmod(idx, cols)
            axes[r][c].axis("off")
        if kind == "correlation" and last_im is not None:
            fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        fig.suptitle(titles[kind], y=1.02, fontsize=14)
        fig.tight_layout()
        path = out_dir / f"fig_feature_panel_{kind}.png"
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        paths[kind] = str(path)
        register_artifact(
            path,
            category="Features",
            title=titles[kind],
            feature_set=",".join(feature_sets),
            run_id=run_id,
        )

    meta_path = out_dir / "feature_panel_windows.json"
    import json

    meta = {
        "feature_sets": list(feature_sets),
        "windows": resolved,
        "figures": paths,
        "spike_source": spike_source,
        "fallback_window_s": fallback_window_s,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def update_feature_panels_after_decoding(
    experiment_dir: Path,
    *,
    feature_sets: list[str] | tuple[str, ...] | None = None,
    spike_source: str = "sorted",
    progress_callback=None,
) -> dict[str, Any] | None:
    """Regenerate panneled feature pages using each set's best decode window."""
    experiment_dir = Path(experiment_dir)
    metrics = _load_metrics(experiment_dir)
    if metrics is None or "feature_set" not in metrics.columns:
        return None
    if feature_sets is None:
        feature_sets = tuple(sorted(metrics["feature_set"].astype(str).dropna().unique()))
    if not feature_sets:
        return None
    return plot_feature_panel_pages(
        experiment_dir,
        feature_sets,
        spike_source=spike_source,
        progress_callback=progress_callback,
    )
