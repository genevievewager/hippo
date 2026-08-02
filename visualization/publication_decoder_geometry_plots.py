"""Publication decoder geometry: shared latent manifold + per-decoder prediction overlays."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from realtime.decoder_comparison import CATEGORICAL_TARGETS
from realtime.decoder_models import (
    TARGET_FAMILY,
    categorical_model_names,
    continuous_model_names,
    is_bayesian_model,
    make_categorical_pipeline,
    make_continuous_pipeline,
)
from visualization.constants import FIGURE_DPI, FIGURE_SUBDIR_DECODER
from visualization.publication_decoding_plots import (
    _best_row,
    _empty_panel,
    _metric_for_target,
    load_comparison_metrics,
)
from visualization.publication_isomap_plots import (
    COLOR_FEATURES,
    MAX_EMBED_POINTS,
    _add_shared_color_guide,
    _color_for_feature,
    _global_color_limits,
    _load_heldout_embedding,
    _resolve_transform_dir,
    _scatter_latent,
    _scatter_position_2d,
)
from visualization.publication_style import save_pub_figure

DEFAULT_SEED = 42
DEFAULT_N_JOBS = -1


def _row_mode(row: pd.Series) -> str:
    col = "feature_mode" if "feature_mode" in row.index else "feature_type"
    if col in row.index and pd.notna(row.get(col)):
        return str(row[col])
    return "counts"


def _registry_path(experiment_dir: Path) -> Path | None:
    for path in (
        experiment_dir / "models" / "best_realtime_decoders.json",
        experiment_dir / "deployment_decoder_selection" / "best_realtime_decoders.json",
    ):
        if path.exists():
            return path
    return None


def _shared_encoder_row(experiment_dir: Path, target: str, metrics: pd.DataFrame) -> pd.Series | None:
    """Deployable winner encoder config, else best metrics row for target."""
    reg_path = _registry_path(experiment_dir)
    if reg_path is not None:
        try:
            data = json.loads(reg_path.read_text())
            tgt = (data.get("targets") or {}).get(target)
            if tgt:
                row = {
                    "target_name": target,
                    "decoder_name": tgt.get("selected_decoder"),
                    "decode_window_s": tgt.get("selected_causal_window_s"),
                    "manifold_n_components": tgt.get("manifold_n_components"),
                    "manifold_transform_path": tgt.get("manifold_transform_path"),
                }
                mode_col = "selected_feature_mode"
                if mode_col in tgt:
                    row["feature_mode"] = tgt[mode_col]
                    row["feature_type"] = tgt[mode_col]
                return pd.Series(row)
        except (json.JSONDecodeError, OSError, TypeError):
            pass

    if metrics.empty:
        return None
    sub = metrics.copy()
    if "spike_source" in sub.columns:
        sub = sub[sub["spike_source"].astype(str) == "sorted"]
    if "target_name" in sub.columns:
        sub = sub[sub["target_name"].astype(str) == target]
    if sub.empty:
        return None
    metric = _metric_for_target(sub, target)
    return _best_row(sub, metric)


def _best_row_for_decoder_target(
    metrics: pd.DataFrame,
    decoder_name: str,
    target: str,
) -> pd.Series | None:
    if metrics.empty:
        return None
    sub = metrics.copy()
    if "spike_source" in sub.columns:
        sub = sub[sub["spike_source"].astype(str) == "sorted"]
    sub = sub[
        (sub["decoder_name"].astype(str) == decoder_name)
        & (sub["target_name"].astype(str) == target)
    ]
    if sub.empty:
        return None
    metric = _metric_for_target(sub, target)
    return _best_row(sub, metric)


def _decoders_for_target(target: str, *, max_models: str = "quick") -> tuple[str, ...]:
    if TARGET_FAMILY.get(target) == "categorical":
        return categorical_model_names(max_models, target)
    return continuous_model_names(max_models, target)


def _project_z_display(Z: np.ndarray, transform_dir_name: str) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.shape[1] < 2:
        return np.column_stack([Z[:, 0], np.zeros(len(Z))])
    if Z.shape[1] > 2:
        if transform_dir_name.startswith("counts") or transform_dir_name.startswith("rates"):
            Zc = Z - np.mean(Z, axis=0, keepdims=True)
            _, _, vt = np.linalg.svd(Zc, full_matrices=False)
            return Zc @ vt[:2].T
        return Z[:, :2]
    return Z


def _subsample_pack(
    Z: np.ndarray,
    beh: pd.DataFrame,
    y_pred: Any,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, pd.DataFrame, Any]:
    n = len(beh)
    if n <= MAX_EMBED_POINTS:
        return Z, beh, y_pred
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=MAX_EMBED_POINTS, replace=False)
    idx = np.sort(idx)
    beh_sub = beh.iloc[idx].reset_index(drop=True)
    if isinstance(y_pred, np.ndarray):
        if y_pred.ndim == 1:
            yp = y_pred[idx]
        else:
            yp = y_pred[idx]
    else:
        yp = np.asarray(y_pred)[idx]
    return Z[idx], beh_sub, yp


def _get_y(behavior: pd.DataFrame, target: str) -> np.ndarray:
    if target == "position":
        return behavior[["x", "y"]].to_numpy()
    if target == "head_direction":
        return behavior[["head_direction_sin", "head_direction_cos"]].to_numpy()
    if target in behavior.columns:
        return behavior[target].to_numpy()
    raise KeyError(f"Unknown target: {target}")


def _fit_estimator(estimator, X_train, y_train, behavior_train, decoder_name: str):
    if is_bayesian_model(decoder_name):
        position_xy = behavior_train[["x", "y"]].to_numpy()
        from realtime.bayesian_decoder import BayesianDistanceToWallDecoder, BayesianPlaceDerivedDecoder

        if isinstance(estimator, (BayesianPlaceDerivedDecoder, BayesianDistanceToWallDecoder)):
            estimator.fit(X_train, position_xy=position_xy)
        else:
            estimator.fit(X_train, position_xy)
        return estimator
    estimator.fit(X_train, y_train)
    return estimator


def _heldout_decoder_predictions(
    experiment_dir: Path,
    encoder_row: pd.Series,
    decoder_name: str,
    target: str,
    *,
    seed: int = DEFAULT_SEED,
    n_jobs: int = DEFAULT_N_JOBS,
) -> tuple[np.ndarray, pd.DataFrame, Any, Any] | None:
    """Refit decoder on train Z; return test Z (2D), behavior, y_true, y_pred."""
    try:
        from realtime.data_loading import load_simulation_data, make_decode_times
        from realtime.decoding_targets import align_extended_behavior_to_decoder_times
        from realtime.manifold_features import load_feature_transformer
        from realtime.spike_features import build_causal_spike_matrix
        from realtime.timing import extract_behavior_times
        from realtime.train_decoder import causal_train_test_split, infer_arena_bounds
    except Exception:
        return None

    experiment_dir = Path(experiment_dir)
    if not (experiment_dir / "behavior.csv").exists():
        return None

    tdir = _resolve_transform_dir(experiment_dir, encoder_row)
    if tdir is None:
        return None
    try:
        transformer = load_feature_transformer(tdir)
    except Exception:
        return None

    decode_window = float(encoder_row.get("decode_window_s", 0.25))
    try:
        data = load_simulation_data(experiment_dir, "sorted")
        arena_bounds = infer_arena_bounds(data["behavior_df"], data["summary"])
        behavior_times = extract_behavior_times(data["behavior_df"])
        decode_times = make_decode_times(
            data["session_duration"], decode_window, 0.05, behavior_times=behavior_times,
        )
        aligned = align_extended_behavior_to_decoder_times(
            data["behavior_df"], decode_times, data["summary"],
        )
        X = build_causal_spike_matrix(
            data["spikes_df"], data["unit_ids"], decode_times, decode_window,
        )
        train_mask, test_mask = causal_train_test_split(decode_times, 0.70)
        if test_mask is None or not np.asarray(test_mask).any():
            return None
        train_mask = np.asarray(train_mask)
        test_mask = np.asarray(test_mask)

        Z_train = np.asarray(transformer.transform(X[train_mask]), dtype=float)
        Z_test = np.asarray(transformer.transform(X[test_mask]), dtype=float)
        beh_train = aligned.iloc[np.where(train_mask)[0]].reset_index(drop=True)
        beh_test = aligned.iloc[np.where(test_mask)[0]].reset_index(drop=True)

        y_train = _get_y(beh_train, target)
        y_test = _get_y(beh_test, target)

        if target in CATEGORICAL_TARGETS:
            pipeline = make_categorical_pipeline(
                decoder_name, seed=seed, n_jobs=n_jobs,
                target_name=target, arena_bounds=arena_bounds,
            )
        else:
            pipeline = make_continuous_pipeline(
                decoder_name, target, seed=seed, n_jobs=n_jobs, arena_bounds=arena_bounds,
            )
        pipeline = _fit_estimator(pipeline, Z_train, y_train, beh_train, decoder_name)
        y_pred = pipeline.predict(Z_test)
        if target == "distance_to_wall" and isinstance(y_pred, np.ndarray) and y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        Z2 = _project_z_display(Z_test, tdir.name)
        Z2, beh_test, y_pred = _subsample_pack(Z2, beh_test, y_pred)
        y_test = _get_y(beh_test, target)
        return Z2, beh_test, y_test, y_pred
    except Exception:
        return None


def _metric_value_label(row: pd.Series | None, target: str) -> str:
    if row is None:
        return ""
    metric = _metric_for_target(pd.DataFrame([row]), target)
    if metric not in row.index or pd.isna(row[metric]):
        return ""
    val = float(row[metric])
    if "error" in metric or metric.endswith("_cm"):
        return f"{metric}={val:.2f}"
    return f"{metric}={val:.3f}"


def _draw_shared_latent_base(
    ax,
    Z: np.ndarray,
    beh: pd.DataFrame,
    feature: str,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> dict | None:
    color = _color_for_feature(beh, feature)
    if isinstance(color, str) and color == "position_xy":
        return _scatter_position_2d(ax, Z, beh, annotate=False)
    if color is None:
        _empty_panel(ax, f"no {feature}\nin behavior")
        return None
    return _scatter_latent(
        ax, Z, color, cmap=cmap, label=feature,
        show_colorbar=False, show_legend=False, vmin=vmin, vmax=vmax,
    )


def _draw_pred_panel(
    ax,
    feature: str,
    beh: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    *,
    is_categorical: bool = False,
) -> None:
    """Draw true vs predicted panel below the manifold scatter."""
    n = len(beh)
    ax.set_title("True vs predicted", fontsize=6, pad=10, y=1.02)
    if n == 0:
        _empty_panel(ax, "no predictions")
        return

    if is_categorical:
        yt = np.asarray(y_true).astype(str)
        yp = np.asarray(y_pred).astype(str)
        acc = float(np.mean(yt == yp))
        step = max(1, n // 80)
        sl = slice(None, None, step)
        t = np.arange(len(yt[sl]))
        uniq = sorted(set(yt) | set(yp))
        lut = {u: i for i, u in enumerate(uniq)}
        yt_i = np.array([lut[u] for u in yt[sl]], dtype=float)
        yp_i = np.array([lut[u] for u in yp[sl]], dtype=float)
        ax.scatter(t, yt_i + 0.12, s=6, c="0.45", alpha=0.8, label="true", linewidths=0)
        ax.scatter(t, yp_i - 0.12, s=6, c="#1565C0", alpha=0.85, label="pred", linewidths=0)
        ax.set_yticks(range(len(uniq)))
        ax.set_yticklabels(uniq, fontsize=5)
        ax.set_xlabel("held-out sample", fontsize=6)
        ax.set_ylabel("class", fontsize=6)
        ax.legend(fontsize=5, loc="upper right", frameon=False, markerscale=1.2)
        ax.text(
            0.02, 0.96, f"acc={acc:.2f}",
            transform=ax.transAxes, fontsize=7, va="top", ha="left",
        )
        return

    if feature == "position" and "x" in beh.columns and "y" in beh.columns:
        tx, ty = beh["x"].to_numpy(), beh["y"].to_numpy()
        pred = np.asarray(y_pred)
        if pred.ndim == 1 or pred.shape[1] < 2:
            _empty_panel(ax, "position pred\nunavailable")
            return
        ax.plot(tx, ty, color="0.55", lw=0.7, alpha=0.85, label="true")
        ax.plot(pred[:, 0], pred[:, 1], color="#1565C0", lw=0.9, alpha=0.9, label="pred")
        ax.set_xlabel("x (cm)", fontsize=6)
        ax.set_ylabel("y (cm)", fontsize=6)
        ax.legend(fontsize=5, loc="best", frameon=False)
        ax.set_aspect("equal", adjustable="box")
        return

    t = np.arange(n)
    step = max(1, n // 150)
    sl = slice(None, None, step)
    yt_arr = np.asarray(y_true, dtype=float)
    yp_arr = np.asarray(y_pred, dtype=float)
    if feature == "head_direction" and yt_arr.ndim == 2 and yt_arr.shape[1] >= 2:
        from realtime.decoding_targets import angles_from_sin_cos

        yt = np.degrees(angles_from_sin_cos(yt_arr[:, 0], yt_arr[:, 1]))
        yp = np.degrees(angles_from_sin_cos(yp_arr[:, 0], yp_arr[:, 1]))
        ylab = "angle (deg)"
    else:
        yt = yt_arr.ravel()
        yp = yp_arr.ravel()
        ylab = feature.replace("_", " ")
    ax.plot(t[sl], yt[sl], color="0.55", lw=0.8, alpha=0.9, label="true")
    ax.plot(t[sl], yp[sl], color="#1565C0", lw=0.9, alpha=0.9, label="pred")
    ax.set_xlabel("held-out sample", fontsize=6)
    ax.set_ylabel(ylab, fontsize=6)
    ax.legend(fontsize=5, loc="best", frameon=False)


def _short_decoder_title(decoder: str, metric_lbl: str) -> str:
    if not metric_lbl:
        return decoder
    return f"{decoder} ({metric_lbl})"


def _fill_decoder_header_ax(ax, label: str, title: str) -> None:
    """Render panel letter + decoder title in a dedicated header axes."""
    from textwrap import fill

    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    wrapped = fill(title, width=36)
    ax.text(
        0.0, 0.92, label,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9, fontweight="bold", clip_on=False,
    )
    ax.text(
        0.07, 0.92, wrapped,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=6.5, color="0.15", clip_on=False, linespacing=1.08,
    )


def plot_fig_decoder_geometry_for_feature(
    experiment_dir: Path,
    feature: str,
    feature_title: str,
    figures_dir: Path | None = None,
) -> Path | None:
    """One page: shared encoder; manifold above true/pred for each decoder."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    out_dir = figures_dir / FIGURE_SUBDIR_DECODER
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_comparison_metrics(experiment_dir, prefer="sorted")
    if "spike_source" in metrics.columns and not metrics.empty:
        metrics = metrics[metrics["spike_source"].astype(str) == "sorted"].copy()

    target = feature
    encoder_row = _shared_encoder_row(experiment_dir, target, metrics)
    if encoder_row is None and target != "position":
        encoder_row = _shared_encoder_row(experiment_dir, "position", metrics)

    decoders = _decoders_for_target(target)
    if not decoders:
        return None

    tdir = _resolve_transform_dir(experiment_dir, encoder_row) if encoder_row is not None else None
    shared_pack = _load_heldout_embedding(experiment_dir, tdir) if tdir is not None else None
    if shared_pack is None:
        Z_shared, beh_shared = None, None
    else:
        Z_shared, beh_shared, _tag = shared_pack

    n = len(decoders)
    cols = min(3, n)
    rows_n = int(np.ceil(n / cols))

    fig = plt.figure(figsize=(4.0 * cols + 0.8, 5.6 * rows_n + 1.6))
    outer = GridSpec(
        rows_n, cols, figure=fig,
        hspace=0.58, wspace=0.38,
        top=0.90, bottom=0.08, left=0.07, right=0.96,
    )

    mode = _row_mode(encoder_row) if encoder_row is not None else "encoder n/a"
    w = encoder_row.get("decode_window_s") if encoder_row is not None else None
    w_txt = f"{float(w):.2g}s" if w is not None and not pd.isna(w) else "?"
    fig.suptitle(
        f'Decoders on shared {mode} manifold (W={w_txt}) — colored by true {feature_title}',
        fontsize=10, y=0.975,
    )

    packs_for_limits = (
        [(mode, encoder_row, shared_pack)] if shared_pack is not None else []
    )
    vmin, vmax = _global_color_limits(packs_for_limits, feature)
    cmap = "hsv" if feature == "head_direction" else (
        "magma" if feature in ("speed", "acceleration") else "viridis"
    )

    latent_axes: list = []
    color_info: dict | None = None
    is_categorical = feature in CATEGORICAL_TARGETS or feature in (
        "spatial_context", "movement_state", "wall_distance_bin",
    )

    for i, decoder in enumerate(decoders):
        r, c = divmod(i, cols)
        inner = outer[r, c].subgridspec(
            3, 1, height_ratios=[0.28, 1.35, 1.0], hspace=0.38,
        )
        ax_header = fig.add_subplot(inner[0])
        ax_latent = fig.add_subplot(inner[1])
        ax_pred = fig.add_subplot(inner[2])
        latent_axes.append(ax_latent)

        dec_row = _best_row_for_decoder_target(metrics, decoder, target)
        metric_lbl = _metric_value_label(dec_row, target)
        _fill_decoder_header_ax(
            ax_header, chr(ord("A") + i), _short_decoder_title(decoder, metric_lbl),
        )

        if Z_shared is None or beh_shared is None:
            _empty_panel(ax_latent, "shared encoder\nunavailable")
            _empty_panel(ax_pred, "predictions\nunavailable")
            continue

        pred_pack = None
        if encoder_row is not None:
            pred_pack = _heldout_decoder_predictions(
                experiment_dir, encoder_row, decoder, target,
            )
        if pred_pack is None:
            _empty_panel(ax_latent, "decoder fit\nunavailable")
            _empty_panel(ax_pred, "predictions\nunavailable")
            continue

        _z_dec, beh_dec, y_true, y_pred = pred_pack
        info = _draw_shared_latent_base(
            ax_latent, Z_shared, beh_shared, feature, cmap=cmap, vmin=vmin, vmax=vmax,
        )
        if color_info is None and info is not None:
            color_info = info

        _draw_pred_panel(
            ax_pred, feature, beh_dec, y_true, y_pred, is_categorical=is_categorical,
        )

    for j in range(n, rows_n * cols):
        r, c = divmod(j, cols)
        inner = outer[r, c].subgridspec(3, 1, height_ratios=[0.28, 1.35, 1.0])
        fig.add_subplot(inner[0]).axis("off")
        fig.add_subplot(inner[1]).axis("off")
        fig.add_subplot(inner[2]).axis("off")

    _add_shared_color_guide(fig, latent_axes, color_info, feature_title=feature_title)

    stem = f"fig_decoder_geometry_{feature}"
    bottom = 0.12 if (color_info or {}).get("kind") in {"categorical", "position_xy"} else 0.08
    right = 0.86 if (color_info or {}).get("kind") == "continuous" else 0.96
    return save_pub_figure(
        fig, out_dir / f"{stem}.png", dpi=FIGURE_DPI,
        rect=(0.07, bottom, right, 0.93),
        pad_inches=0.45,
    )


def plot_fig_decoder_geometry(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> Path | None:
    """Write fig_decoder_geometry_<feature>.png for each behavioral variable."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"

    written: list[Path] = []
    for feature, title in COLOR_FEATURES:
        try:
            path = plot_fig_decoder_geometry_for_feature(
                experiment_dir, feature, title, figures_dir=figures_dir,
            )
            if path is not None:
                written.append(path)
                print(f"  wrote {path.relative_to(figures_dir)}")
        except Exception as exc:
            print(f"  warning: decoder geometry [{feature}] skipped ({exc})")

    if not written:
        return None
    return next(
        (p for p in written if p.stem == "fig_decoder_geometry_position"),
        written[0],
    )


def generate_publication_decoder_geometry_figures(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Orchestrator wrapper returning all written decoder-geometry paths."""
    experiment_dir = Path(experiment_dir)
    figures_dir = Path(figures_dir) if figures_dir else experiment_dir / "figures"
    written: list[Path] = []
    path = plot_fig_decoder_geometry(experiment_dir, figures_dir)
    if path is not None:
        out_dir = figures_dir / FIGURE_SUBDIR_DECODER
        written.extend(sorted(out_dir.glob("fig_decoder_geometry_*.png")))
    return written
