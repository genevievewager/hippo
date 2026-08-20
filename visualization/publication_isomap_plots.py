"""Publication multi-panel Isomap / latent-geometry figures (Figs 6–8).

Default latent-geometry suite: one page per recovered latent variable, with
every embedding mode colored by that variable
(``fig_latent_geometry_<feature>.png``).

Pass ``compact=True`` to ``plot_fig_latent_geometry`` for a 2-page summary
(position page + other-features overview).

Gracefully annotates empty-state panels when a mode/transform is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
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
from visualization.publication_style import (
    apply_publication_theme,
    figure_legend_below,
    legend_below,
    legend_outside,
    panel_label,
    save_pub_figure,
)

apply_publication_theme()

MAX_EMBED_POINTS = 2500

# Stable panel order for embedding modes (skip missing).
EMBEDDING_MODE_ORDER = (
    "counts",
    "rates",
    "global_pca",
    "region_pca",
    "layer_pca",
    "cell_type_pca",
    "rate_model_pca",
    "global_isomap",
    "global_isomap_distilled",
    "diffusion_nystrom",
)

# One page per recovered / color feature (sorted-spike deployable view).
COLOR_FEATURES: tuple[tuple[str, str], ...] = (
    ("position", "Position"),
    ("speed", "Speed"),
    ("acceleration", "Acceleration"),
    ("head_direction", "Head direction"),
    ("distance_to_wall", "Distance to wall"),
    ("spatial_context", "Spatial context"),
    ("movement_state", "Movement state"),
    ("wall_distance_bin", "Wall-distance bin"),
)


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


def _mode_column(metrics: pd.DataFrame) -> str | None:
    """Column with embedding-mode labels.

    Newer decoder runs keep ``feature_type='counts'`` and put modes in
    ``feature_mode``; older runs store modes directly in ``feature_type``.
    """
    has_mode = "feature_mode" in metrics.columns
    has_type = "feature_type" in metrics.columns
    if has_mode and has_type:
        modes = {str(m) for m in metrics["feature_mode"].dropna().unique()}
        types = {str(m) for m in metrics["feature_type"].dropna().unique()}
        return "feature_mode" if len(modes) > len(types) else "feature_type"
    if has_mode:
        return "feature_mode"
    if has_type:
        return "feature_type"
    return None


def _row_embedding_mode(row: pd.Series) -> str:
    """Embedding mode label for one metrics row (schema-tolerant)."""
    if "feature_mode" in getattr(row, "index", []):
        fm = row.get("feature_mode")
        if pd.notna(fm):
            text = str(fm).strip()
            if text and text.lower() not in {"nan", "none"}:
                return text
    ft = row.get("feature_type")
    if pd.notna(ft):
        text = str(ft).strip()
        if text and text.lower() not in {"nan", "none"}:
            return text
    return ""


def _resolve_transform_dir(
    experiment_dir: Path,
    row: pd.Series,
) -> Path | None:
    """Resolve manifold transform directory from a metrics row."""
    experiment_dir = Path(experiment_dir)
    raw = row.get("manifold_transform_path")
    if raw is not None and not (isinstance(raw, float) and np.isnan(raw)):
        path = Path(str(raw))
        if not path.is_absolute():
            candidates = [
                path,
                Path.cwd() / path,
                experiment_dir / path.name,
                experiment_dir / "decoder_comparison" / "sorted" / "models" / "manifold_transforms" / path.name,
            ]
        else:
            candidates = [path]
        for c in candidates:
            if c.exists():
                return c

    feature = _row_embedding_mode(row) or str(row.get("feature_type", ""))
    w = float(row.get("decode_window_s", 0.25))
    w_ms = int(round(w * 1000))
    k = row.get("manifold_n_components")
    nn = row.get("n_neighbors")
    root = experiment_dir / "decoder_comparison" / "sorted" / "models" / "manifold_transforms"
    if not root.exists():
        root = experiment_dir / "decoder_comparison" / "models" / "manifold_transforms"
    if not root.exists():
        return None

    # Build candidate dirname patterns
    patterns = []
    if feature in ("counts", "rates"):
        patterns.append(f"{feature}_w{w_ms:04d}ms")
    else:
        k_part = ""
        if k is not None and not (isinstance(k, float) and np.isnan(k)):
            k_part = f"_k{int(k)}"
        nn_part = ""
        if nn is not None and not (isinstance(nn, float) and np.isnan(nn)):
            nn_part = f"_nn{int(nn)}"
        patterns.append(f"{feature}{k_part}{nn_part}_w{w_ms:04d}ms")
        patterns.append(f"{feature}{k_part}_w{w_ms:04d}ms")
        patterns.append(f"{feature}_w{w_ms:04d}ms")

    for name in patterns:
        cand = root / name
        if cand.exists():
            return cand
    # Fuzzy: any dir starting with feature and matching window
    for p in sorted(root.glob(f"{feature}*w{w_ms:04d}ms")):
        return p
    return None


def _best_row_for_mode_target(
    metrics: pd.DataFrame,
    *,
    feature_mode: str,
    target: str,
) -> pd.Series | None:
    if metrics.empty:
        return None
    df = metrics.copy()
    if "spike_source" in df.columns:
        df = df[df["spike_source"].astype(str) == "sorted"]
    mode_col = _mode_column(df)
    if mode_col is None:
        return None
    df = df[
        (df[mode_col].astype(str) == feature_mode)
        & (df["target_name"].astype(str) == target)
    ]
    if df.empty:
        return None
    metric = _metric_for_target(df, target)
    return _best_row(df, metric)


def _modes_present(metrics: pd.DataFrame) -> list[str]:
    mode_col = _mode_column(metrics)
    if metrics.empty or mode_col is None:
        return []
    df = metrics
    if "spike_source" in df.columns:
        df = df[df["spike_source"].astype(str) == "sorted"]
    present = sorted({str(m) for m in df[mode_col].dropna().unique()})
    ordered = [m for m in EMBEDDING_MODE_ORDER if m in present]
    ordered += [m for m in present if m not in ordered]
    return ordered


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
        Z = np.asarray(transformer.transform(X[mask]), dtype=float)
        # Display PCA when raw counts (or any) embedding is >2-D
        if Z.ndim == 1:
            Z = Z.reshape(-1, 1)
        if Z.shape[1] < 2:
            Z = np.column_stack([Z[:, 0], np.zeros(len(Z))])
        elif Z.shape[1] > 2:
            # Display-only 2D view. Never fit/persist a decoder or manifold model.
            # Ordered manifold embeddings: keep leading two axes. For raw counts /
            # rates, use an in-memory SVD projection (figures/ only; not saved under
            # decoder_comparison/ or models/).
            if name.startswith("counts") or name.startswith("rates"):
                Zc = Z - np.mean(Z, axis=0, keepdims=True)
                _, _, vt = np.linalg.svd(Zc, full_matrices=False)
                Z = Zc @ vt[:2].T
            else:
                Z = Z[:, :2]
        beh = aligned.iloc[np.where(mask)[0]].reset_index(drop=True)
        if Z.shape[0] > MAX_EMBED_POINTS:
            rng = np.random.default_rng(0)
            idx = rng.choice(Z.shape[0], size=MAX_EMBED_POINTS, replace=False)
            Z = Z[idx]
            beh = beh.iloc[idx].reset_index(drop=True)
        return Z, beh, tag
    except Exception:
        return None


def _rgb_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map position to RGB: x → hue (hsv), y → brightness."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    def _norm(a: np.ndarray) -> np.ndarray:
        lo, hi = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return np.zeros_like(a)
        return np.clip((a - lo) / (hi - lo), 0.0, 1.0)

    xn, yn = _norm(x), _norm(y)
    rgba = plt.cm.hsv(xn)
    brightness = 0.30 + 0.70 * yn
    out = rgba.copy()
    out[:, 0] *= brightness
    out[:, 1] *= brightness
    out[:, 2] *= brightness
    return out


def _scatter_latent(
    ax,
    Z: np.ndarray,
    color,
    *,
    cmap="viridis",
    label: str = "",
    show_colorbar: bool = True,
    show_legend: bool = True,
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict | None:
    """Scatter latent coords; optionally omit per-panel colorbar/legend.

    Returns a small dict describing the color mapping so callers can draw a
    single shared legend / colorbar for a multi-panel figure.
    """
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
        labels = [h.get_label() for h in handles]
        if show_legend:
            legend_below(ax, handles=handles, labels=labels, ncol=min(3, len(handles)))
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
        sns.despine(ax=ax)
        return {"kind": "categorical", "handles": handles, "labels": labels, "label": label}

    vals = np.asarray(color, dtype=float)
    sc = ax.scatter(
        Z[:, 0], Z[:, 1], c=vals, s=4, cmap=cmap, alpha=0.65,
        vmin=vmin, vmax=vmax,
    )
    if show_colorbar:
        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label=label)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    sns.despine(ax=ax)
    return {"kind": "continuous", "mappable": sc, "label": label}


def _behavior_column(beh: pd.DataFrame, *names: str) -> str | None:
    """First matching behavior column among canonical and unit-suffixed aliases."""
    for name in names:
        if name in beh.columns:
            return name
    return None


def _scatter_position_2d(ax, Z: np.ndarray, beh: pd.DataFrame, *, annotate: bool = True) -> dict | None:
    x_col = _behavior_column(beh, "x", "x_cm")
    y_col = _behavior_column(beh, "y", "y_cm")
    if x_col is None or y_col is None:
        color = beh[x_col] if x_col is not None else np.arange(len(beh))
        return _scatter_latent(ax, Z, color, label="position", show_colorbar=annotate)
    rgba = _rgb_from_xy(beh[x_col].to_numpy(), beh[y_col].to_numpy())
    ax.scatter(Z[:, 0], Z[:, 1], c=rgba, s=4, alpha=0.7)
    ax.set_xlabel("z₁")
    ax.set_ylabel("z₂")
    if annotate:
        ax.text(
            0.02, 0.98, "x→hue, y→brightness",
            transform=ax.transAxes, va="top", ha="left", fontsize=8, color="0.25",
        )
    sns.despine(ax=ax)
    return {"kind": "position_xy", "label": "x→hue, y→brightness"}


def _color_for_feature(beh: pd.DataFrame, feature: str):
    """Return color data for a recovered feature (Series or special marker)."""
    if feature == "position":
        return "position_xy"
    aliases = {
        "speed": ("speed", "speed_cm_s"),
        "acceleration": ("acceleration", "acceleration_cm_s2"),
        "head_direction": ("head_direction", "head_direction_rad"),
        "distance_to_wall": ("distance_to_wall", "distance_to_wall_cm"),
        "spatial_context": ("spatial_context",),
        "movement_state": ("movement_state",),
        "wall_distance_bin": ("wall_distance_bin",),
    }
    for name in aliases.get(feature, (feature,)):
        if name in beh.columns:
            return beh[name]
    # Fallbacks
    if feature == "head_direction" and "head_direction_sin" in beh.columns:
        return np.arctan2(beh["head_direction_sin"], beh["head_direction_cos"])
    return None


def _panel_title(mode: str, row: pd.Series | None) -> str:
    if row is None:
        return mode
    w = row.get("decode_window_s")
    k = row.get("manifold_n_components")
    nn = row.get("n_neighbors")
    parts = [mode]
    if w is not None and not (isinstance(w, float) and np.isnan(w)):
        parts.append(f"W={float(w):.2f}s")
    if k is not None and not (isinstance(k, float) and np.isnan(k)):
        parts.append(f"k={int(k)}")
    if nn is not None and not (isinstance(nn, float) and np.isnan(nn)):
        parts.append(f"nn={int(nn)}")
    return "\n".join([parts[0], " ".join(parts[1:])]) if len(parts) > 1 else parts[0]


def _robust_continuous_limits(values: np.ndarray) -> tuple[float, float]:
    """Percentile color limits so rare spikes do not collapse contrast."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0.0, 1.0
    vmin = float(np.nanpercentile(finite, 1))
    vmax = float(np.nanpercentile(finite, 99))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite) + 1e-9)
    return vmin, vmax


def _as_continuous_color_array(color) -> np.ndarray | None:
    """Float array for continuous coloring, or None if missing/categorical."""
    if color is None or (isinstance(color, str) and color == "position_xy"):
        return None
    if isinstance(color, pd.Series):
        if color.dtype == object or str(color.dtype) == "category" or pd.api.types.is_string_dtype(color):
            return None
        try:
            arr = pd.to_numeric(color, errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError):
            return None
    else:
        try:
            arr = np.asarray(color, dtype=float)
        except (TypeError, ValueError):
            return None
    return arr


def _global_color_limits(packs: list, feature: str) -> tuple[float | None, float | None]:
    """Shared vmin/vmax across panels so one colorbar matches all embeddings.

    Uses 1st–99th percentiles so rare spikes (e.g. acceleration outliers from
    forward differences) do not collapse contrast across the typical range.
    Values outside the limits still map to the colormap endpoints.
    """
    if feature == "position":
        return None, None
    vals: list[np.ndarray] = []
    for _mode, _row, pack in packs:
        if pack is None:
            continue
        Z, beh, _tag = pack
        del Z
        color = _color_for_feature(beh, feature)
        arr = _as_continuous_color_array(color)
        if color is not None and arr is None and not (isinstance(color, str) and color == "position_xy"):
            return None, None  # categorical
        if arr is None:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size:
            vals.append(finite)
    if not vals:
        return None, None
    return _robust_continuous_limits(np.concatenate(vals))


def _add_shared_color_guide(fig, axes: list, color_info: dict | None, *, feature_title: str) -> None:
    """Draw one shared colorbar / categorical legend / position note.

    Continuous colorbars use a dedicated figure axes so ``colorbar(ax=...)``
    does not steal vertical space and collapse the A–C / D–F gap.
    """
    if not color_info or not axes:
        return
    kind = color_info.get("kind")
    if kind == "continuous" and color_info.get("mappable") is not None:
        # Fixed right-hand strip; do not resize the panel GridSpec.
        cax = fig.add_axes([0.90, 0.18, 0.018, 0.62])
        cbar = fig.colorbar(color_info["mappable"], cax=cax)
        cbar.set_label(feature_title)
    elif kind == "categorical" and color_info.get("handles"):
        # Sit fully below bottom-row axis labels; enlarge for readability.
        # Negative figure y keeps panel GridSpec / rect dimensions unchanged.
        figure_legend_below(
            fig,
            color_info["handles"],
            color_info["labels"],
            ncol=min(4, max(len(color_info["handles"]), 1)),
            y=-0.06,
            title=feature_title,
            fontsize=14,
            title_fontsize=14,
            markerscale=2.2,
        )
    elif kind == "position_xy":
        fig.text(
            0.5, 0.01, "Color: x→hue, y→brightness",
            ha="center", va="bottom", fontsize=9, color="0.35",
        )


# ---------------------------------------------------------------------------
# Fig 6 — Latent geometry (one page per recovered feature)
# ---------------------------------------------------------------------------

@dataclass
class LatentGeometryPanel:
    """One embedding-mode panel for a latent-geometry page."""

    mode: str
    Z: np.ndarray
    behavior: pd.DataFrame
    decode_window_s: float | None = None
    n_components: int | None = None
    n_neighbors: int | None = None


def to_display_latent_2d(Z: np.ndarray, mode: str) -> np.ndarray:
    """Display-only 2-D view matching ``_load_heldout_embedding``.

    Ordered manifold embeddings keep leading two axes. Raw counts/rates use an
    in-memory SVD projection (never persisted as a model).
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape[0] == 0:
        return Z
    if Z.shape[1] < 2:
        return np.column_stack([Z[:, 0], np.zeros(len(Z))])
    if Z.shape[1] > 2:
        mode_l = str(mode).lower()
        if mode_l in {"counts", "rates", "identity"} or mode_l.startswith("counts"):
            Zc = Z - np.mean(Z, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(Zc, full_matrices=False)
            return Zc @ vt[:2].T
        return Z[:, :2]
    return Z


def _panel_title_from_meta(
    mode: str,
    *,
    decode_window_s: float | None = None,
    n_components: int | None = None,
    n_neighbors: int | None = None,
) -> str:
    parts = [mode]
    meta: list[str] = []
    if decode_window_s is not None and np.isfinite(decode_window_s):
        meta.append(f"W={float(decode_window_s):.2f}s")
    if n_components is not None and np.isfinite(n_components):
        meta.append(f"k={int(n_components)}")
    if n_neighbors is not None and np.isfinite(n_neighbors):
        meta.append(f"nn={int(n_neighbors)}")
    if meta:
        parts.append(" ".join(meta))
    return "\n".join(parts)


def plot_latent_geometry_page_from_embeddings(
    panels: list[LatentGeometryPanel],
    feature: str,
    feature_title: str,
    out_path: Path,
) -> Path | None:
    """Write one publication-format latent-geometry page from prepared embeddings.

    Matches ``fig_latent_geometry_<feature>.png`` layout used in the PDF suite:
    multi-panel grid, shared color guide, panel labels A…, position as x→hue /
    y→brightness.
    """
    if not panels:
        return None
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Match packing format expected by _global_color_limits
    plot_packs: list[tuple[str, None, tuple | None]] = []
    for panel in panels:
        Z = to_display_latent_2d(panel.Z, panel.mode)
        beh = panel.behavior
        if Z.ndim != 2 or Z.shape[1] < 2 or Z.shape[0] == 0:
            plot_packs.append((panel.mode, None, None))
            continue
        if Z.shape[0] > MAX_EMBED_POINTS:
            rng = np.random.default_rng(0)
            idx = rng.choice(Z.shape[0], size=MAX_EMBED_POINTS, replace=False)
            Z = Z[idx]
            beh = beh.iloc[idx].reset_index(drop=True)
        plot_packs.append((panel.mode, None, (Z, beh, "ui")))

    if not any(p[-1] is not None for p in plot_packs):
        return None

    n = len(plot_packs)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))
    height_ratios: list[float] = []
    for ri in range(rows_n):
        height_ratios.append(1.0)
        if ri < rows_n - 1:
            height_ratios.append(0.28)
    fig = plt.figure(figsize=(3.5 * cols + 1.4, 3.4 * rows_n + 0.9))
    gs = GridSpec(
        len(height_ratios), cols, figure=fig,
        height_ratios=height_ratios, hspace=0.05, wspace=0.28,
    )

    vmin, vmax = _global_color_limits(plot_packs, feature)
    axes: list = []
    color_info: dict | None = None
    cmap = "hsv" if feature == "head_direction" else (
        "magma" if feature in ("speed", "acceleration") else "viridis"
    )

    for i, (panel, (_mode, _row, pack)) in enumerate(zip(panels, plot_packs)):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r * 2, c])
        axes.append(ax)
        title = _panel_title_from_meta(
            panel.mode,
            decode_window_s=panel.decode_window_s,
            n_components=panel.n_components,
            n_neighbors=panel.n_neighbors,
        )
        if pack is None:
            _empty_panel(ax, f"{panel.mode}\nunavailable")
        else:
            Z, beh, _tag = pack
            color = _color_for_feature(beh, feature)
            if isinstance(color, str) and color == "position_xy":
                info = _scatter_position_2d(ax, Z, beh, annotate=False)
            elif color is None:
                _empty_panel(ax, f"no {feature}\nin behavior")
                info = None
            else:
                info = _scatter_latent(
                    ax, Z, color, cmap=cmap, label=feature,
                    show_colorbar=False, show_legend=False,
                    vmin=vmin, vmax=vmax,
                )
            if color_info is None and info is not None:
                color_info = info
            ax.set_title(title, fontsize=8, pad=6)
        panel_label(ax, chr(ord("A") + i))

    for j in range(n, rows_n * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r * 2, c])
        ax.axis("off")

    _add_shared_color_guide(fig, axes, color_info, feature_title=feature_title)
    fig.suptitle(
        f'Latent geometry colored by "{feature_title}"',
        fontsize=11,
        y=0.98,
    )
    bottom = 0.10 if (color_info or {}).get("kind") in {"categorical", "position_xy"} else 0.06
    right = 0.86 if (color_info or {}).get("kind") == "continuous" else 0.96
    return save_pub_figure(
        fig, out_path, dpi=FIGURE_DPI,
        rect=(0.08, bottom, right, 0.90),
        hspace=0.20,
        wspace=0.30,
        pad_inches=0.35,
    )


def plot_fig_latent_geometry_for_feature(
    experiment_dir: Path,
    feature: str,
    feature_title: str,
    figures_dir: Path | None = None,
) -> Path | None:
    """One page: all embedding modes colored by a single recovered feature."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir, prefer="sorted")
    if metrics.empty:
        return None
    if "spike_source" in metrics.columns:
        metrics = metrics[metrics["spike_source"].astype(str) == "sorted"].copy()

    modes = _modes_present(metrics)
    if not modes:
        return None

    # Decode target for choosing best W: position page uses target "position"
    target = "position" if feature == "position" else feature
    panels: list[LatentGeometryPanel] = []
    for mode in modes:
        row = _best_row_for_mode_target(metrics, feature_mode=mode, target=target)
        # If this mode never decoded this target, fall back to best W for position
        if row is None and target != "position":
            row = _best_row_for_mode_target(metrics, feature_mode=mode, target="position")
        if row is None:
            panels.append(LatentGeometryPanel(mode=mode, Z=np.zeros((0, 2)), behavior=pd.DataFrame()))
            continue
        tdir = _resolve_transform_dir(experiment_dir, row)
        pack = _load_heldout_embedding(experiment_dir, tdir) if tdir is not None else None
        if pack is None:
            panels.append(LatentGeometryPanel(mode=mode, Z=np.zeros((0, 2)), behavior=pd.DataFrame()))
            continue
        Z, beh, _tag = pack
        k = row.get("manifold_n_components")
        nn = row.get("n_neighbors")
        w = row.get("decode_window_s")
        panels.append(
            LatentGeometryPanel(
                mode=mode,
                Z=Z,
                behavior=beh,
                decode_window_s=float(w) if w is not None and pd.notna(w) else None,
                n_components=int(k) if k is not None and pd.notna(k) else None,
                n_neighbors=int(nn) if nn is not None and pd.notna(nn) else None,
            ),
        )

    stem = f"fig_latent_geometry_{feature}"
    return plot_latent_geometry_page_from_embeddings(
        panels, feature, feature_title, out_dir / f"{stem}.png",
    )


def plot_fig_latent_geometry(
    experiment_dir: Path,
    figures_dir: Path | None = None,
    *,
    compact: bool = False,
) -> Path | None:
    """Latent-geometry suite for the PDF.

    Default ``compact=False`` writes one page per recovered feature
    (``fig_latent_geometry_<feature>.png``): every embedding mode colored by
    that variable at its best causal window.

    Set ``compact=True`` for a 2-page summary:
      - ``fig_latent_geometry_position.png`` — all embeddings colored by position
      - ``fig_latent_geometry_features.png`` — remaining features in a compact grid
        (best embedding mode per feature)
    """
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    if not compact:
        written: list[Path] = []
        for feature, title in COLOR_FEATURES:
            try:
                path = plot_fig_latent_geometry_for_feature(
                    experiment_dir, feature, title, figures_dir=figures_dir,
                )
                if path is not None:
                    written.append(path)
                    print(f"  wrote {path.relative_to(figures_dir)}")
            except Exception as exc:
                print(f"  warning: latent geometry [{feature}] skipped ({exc})")
        (out_dir / "fig_latent_geometry.png").unlink(missing_ok=True)
        (out_dir / "fig_latent_geometry_features.png").unlink(missing_ok=True)
        if not written:
            return None
        return next(
            (p for p in written if p.stem == "fig_latent_geometry_position"),
            written[0],
        )

    # Compact suite ---------------------------------------------------------
    try:
        pos = plot_fig_latent_geometry_for_feature(
            experiment_dir, "position", "Position",
            figures_dir=figures_dir,
        )
        if pos is not None:
            print(f"  wrote {pos.relative_to(figures_dir)}")
    except Exception as exc:
        print(f"  warning: latent geometry [position] skipped ({exc})")
        pos = None

    other = [(f, t) for f, t in COLOR_FEATURES if f != "position"]
    overview = _plot_fig_latent_geometry_features_overview(
        experiment_dir, other, figures_dir=figures_dir,
    )
    if overview is not None:
        print(f"  wrote {overview.relative_to(figures_dir)}")

    # Remove per-feature pages from prior full suites.
    for feature, _ in COLOR_FEATURES:
        if feature == "position":
            continue
        (out_dir / f"fig_latent_geometry_{feature}.png").unlink(missing_ok=True)
    (out_dir / "fig_latent_geometry.png").unlink(missing_ok=True)

    return pos or overview


def _plot_fig_latent_geometry_features_overview(
    experiment_dir: Path,
    features: list[tuple[str, str]],
    *,
    figures_dir: Path,
) -> Path | None:
    """One page: each non-position feature colored on its best embedding."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir)
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir, prefer="sorted")
    if metrics.empty:
        return None
    if "spike_source" in metrics.columns:
        metrics = metrics[metrics["spike_source"].astype(str) == "sorted"].copy()

    modes = _modes_present(metrics)
    if not modes:
        return None
    # Prefer a global linear mode when available for a consistent overview grid.
    prefer = [m for m in ("global_pca", "region_pca", "layer_pca") if m in modes]
    default_mode = prefer[0] if prefer else modes[0]

    n = len(features)
    if n == 0:
        return None
    cols = 4
    rows_n = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.3 * cols + 0.8, 3.3 * rows_n + 0.9))
    gs = GridSpec(rows_n, cols, figure=fig, hspace=0.75, wspace=0.32)

    for i, (feature, title) in enumerate(features):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])
        target = feature
        row = _best_row_for_mode_target(metrics, feature_mode=default_mode, target=target)
        if row is None:
            row = _best_row_for_mode_target(metrics, feature_mode=default_mode, target="position")
        if row is None:
            # Fall back across modes.
            for mode in modes:
                row = _best_row_for_mode_target(metrics, feature_mode=mode, target=target)
                if row is not None:
                    break
        if row is None:
            _empty_panel(ax, f"{feature}\nunavailable")
            panel_label(ax, chr(ord("A") + i))
            continue
        tdir = _resolve_transform_dir(experiment_dir, row)
        pack = _load_heldout_embedding(experiment_dir, tdir) if tdir is not None else None
        if pack is None:
            _empty_panel(ax, f"{feature}\nunavailable")
        else:
            Z, beh, _tag = pack
            color = _color_for_feature(beh, feature)
            mode = _row_embedding_mode(row) or default_mode
            # Overview page: each panel is a different feature, so keep per-panel guides.
            if isinstance(color, str) and color == "position_xy":
                _scatter_position_2d(ax, Z, beh, annotate=True)
            elif color is None:
                _empty_panel(ax, f"no {feature}")
            else:
                cmap = "hsv" if feature == "head_direction" else (
                    "magma" if feature in ("speed", "acceleration") else "viridis"
                )
                arr = _as_continuous_color_array(color)
                vmin = vmax = None
                if arr is not None:
                    vmin, vmax = _robust_continuous_limits(arr)
                _scatter_latent(
                    ax, Z, color, cmap=cmap, label=feature, vmin=vmin, vmax=vmax,
                )
            ax.set_title(f"{title}\n({mode})", fontsize=9)
        panel_label(ax, chr(ord("A") + i))

    for j in range(n, rows_n * cols):
        r, c = divmod(j, cols)
        ax = fig.add_subplot(gs[r, c])
        ax.axis("off")

    fig.suptitle(
        'Latent geometry colored by other behavioral features',
        fontsize=11,
        y=0.98,
    )
    return save_pub_figure(
        fig, out_dir / "fig_latent_geometry_features.png", dpi=FIGURE_DPI,
        rect=(0.08, 0.08, 0.96, 0.90),
        hspace=0.78,
        wspace=0.35,
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
        mode_col = _mode_column(metrics)
        if mode_col is not None:
            iso = metrics[metrics[mode_col].astype(str).str.contains("isomap", na=False)].copy()
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
            legend_below(ax, ncol=1)
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
        legend_below(ax, ncol=2)
        sns.despine(ax=ax)

    _diag_or_empty(ax_d, "D", draw_geo)

    return save_pub_figure(
        fig, out_dir / "fig_isomap_diagnostics.png", dpi=FIGURE_DPI,
        rect=(0.10, 0.16, 0.96, 0.94),
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

    fig = plt.figure(figsize=(11.0, 7.8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.45)

    modes = ("counts", "global_pca", "global_isomap", "global_isomap_distilled")
    mode_col = _mode_column(metrics)
    mode_values = set(metrics[mode_col].astype(str)) if mode_col else set()
    has_iso = (
        not metrics.empty
        and mode_col is not None
        and any(m in mode_values for m in ("global_isomap", "global_isomap_distilled"))
    )

    # A — best score by representation × target (selected targets)
    ax_a = fig.add_subplot(gs[0, 0])
    if not metrics.empty and mode_col is not None:
        present = [m for m in modes if m in mode_values]
        targets = [t for t in ("position", "speed", "spatial_context", "head_direction")
                   if t in set(metrics["target_name"])]
        rows = []
        for target in targets:
            sub_t = metrics[metrics["target_name"] == target]
            metric = _metric_for_target(sub_t, target)
            for mode in present:
                sub = sub_t[sub_t[mode_col].astype(str) == mode]
                row = _best_row(sub, metric)
                if row is None:
                    continue
                val = float(row[metric])
                rows.append({"target": target, "feature": mode, "value": val, "metric": metric})
        rdf = pd.DataFrame(rows)
        if not rdf.empty:
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
            legend_outside(ax_a, ncol=1, bbox=(1.05, 1.0))
            ax_a.tick_params(axis="x", rotation=20, labelsize=8)
            ax_a.set_ylim(0, 1.15)
            sns.despine(ax=ax_a)
        else:
            _empty_panel(ax_a, "No representation scores")
    else:
        _empty_panel(ax_a, "No comparison metrics")
    panel_label(ax_a, "A")

    # B — decoder family × representation for position (or first target)
    ax_b = fig.add_subplot(gs[0, 1])
    if not metrics.empty and mode_col is not None:
        target = "position" if "position" in set(metrics["target_name"]) else metrics["target_name"].iloc[0]
        sub = metrics[metrics["target_name"] == target]
        present = [m for m in modes if m in set(sub[mode_col].astype(str))]
        metric = _metric_for_target(sub, target)
        pivot = sub[sub[mode_col].astype(str).isin(present)].pivot_table(
            index="decoder_name", columns=mode_col, values=metric,
            aggfunc="min" if _is_lower_better(metric) else "max",
        )
        if not pivot.empty:
            cols = [c for c in modes if c in pivot.columns]
            pivot = pivot[cols]
            sns.heatmap(pivot, ax=ax_b, annot=True, fmt=".2f", cmap="viridis",
                        cbar_kws={"label": metric})
            ax_b.tick_params(labelsize=8)
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
    if has_iso and not metrics.empty and mode_col is not None:
        rows = []
        for mode in ("global_isomap", "global_isomap_distilled"):
            if mode not in mode_values:
                continue
            for target, g in metrics[metrics[mode_col].astype(str) == mode].groupby("target_name"):
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
            legend_below(ax_d, ncol=2)
            sns.despine(ax=ax_d)
        elif not rdf.empty:
            sns.barplot(data=rdf, x="target", y="value", hue="mode", ax=ax_d)
            ax_d.tick_params(axis="x", rotation=25, labelsize=7)
            legend_below(ax_d, ncol=2)
            sns.despine(ax=ax_d)
        else:
            _empty_panel(ax_d, "No Isomap accuracy rows")
    else:
        _empty_panel(ax_d, "Isomap not run\n(no teacher vs distilled accuracy)")
    panel_label(ax_d, "D")

    return save_pub_figure(
        fig, out_dir / "fig_isomap_story.png", dpi=FIGURE_DPI,
        rect=(0.10, 0.14, 0.82, 0.94),
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

    try:
        from visualization.publication_decoder_geometry_plots import (
            generate_publication_decoder_geometry_figures,
        )

        for path in generate_publication_decoder_geometry_figures(experiment_dir, figures_dir):
            if path not in written:
                written.append(path)
    except Exception as exc:
        print(f"  warning: decoder geometry skipped ({exc})")

    try:
        from visualization.publication_winner_plots import generate_publication_winner_figures

        for path in generate_publication_winner_figures(experiment_dir, figures_dir):
            if path not in written:
                written.append(path)
    except Exception as exc:
        print(f"  warning: winner embedding figures skipped ({exc})")

    if cleanup_legacy:
        from visualization.publication_decoding_plots import _cleanup_legacy_pngs

        _cleanup_legacy_pngs(figures_dir / FIGURE_SUBDIR_DECODER)

    return written
