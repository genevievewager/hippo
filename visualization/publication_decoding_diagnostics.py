"""Publication-level held-out decoding diagnostic figures.

Uses ``publication_style`` constrained layout (no ``tight_layout``). Scientific
quantities come from ``realtime.decoding_diagnostics_prep`` so Streamlit Plotly
and these PNGs cannot disagree on residuals, circular error, or binning.

Display traces may be downsampled; stored parquet / header metrics are full test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec

from realtime.decoding_diagnostics_prep import (
    absolute_error_series,
    align_config_pair,
    arena_limits_from_trace,
    circular_error_by_angle,
    class_labels_from_meta,
    compression_slope,
    confusion_from_trace,
    downsample_for_display,
    magnitude_bins,
    predicted_class_probability,
    radial_shrinkage,
    recall_by_true_class,
    sparse_link_indices,
    spatial_error_map,
    target_diagnostic_family,
    wrap_head_direction_series,
)
from visualization.constants import FIGURE_DPI, FIGURE_SUBDIR_DECODER
from visualization.publication_style import (
    DIFF_CMAP,
    ERROR_CMAP,
    LINEWIDTH,
    PRED_B_COLOR,
    PRED_COLOR,
    PRED_MARKER,
    TIME_CMAP,
    TRUE_COLOR,
    TRUE_MARKER,
    new_decoding_figure,
    panel_label,
    save_decoding_figure,
    style_axes,
)

SPLIT_CAPTION = "Offline held-out test"


def diagnostics_figure_dir(experiment_dir: Path, figures_dir: Path | None = None) -> Path:
    root = Path(figures_dir) if figures_dir else Path(experiment_dir) / "figures"
    return root / FIGURE_SUBDIR_DECODER / "diagnostics"


def _raster_scatter(ax, x, y, *, c=None, cmap=None, marker="o", s=8, alpha=0.7, **kwargs):
    return ax.scatter(
        x, y, c=c, cmap=cmap, marker=marker, s=s, alpha=alpha,
        rasterized=True, linewidths=0.0 if marker != "x" else 0.6, **kwargs,
    )


def _add_time_colorbar(fig, mappable, *, label: str = "Time (s)"):
    cb = fig.colorbar(mappable, ax=fig.axes[:2] if len(fig.axes) >= 2 else fig.axes, shrink=0.65, pad=0.02)
    cb.set_label(label)
    return cb


def plot_position_diagnostics(
    frame: pd.DataFrame,
    *,
    title: str,
    path: Path | None = None,
    frame_b: pd.DataFrame | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> Path | Any:
    disp = downsample_for_display(frame)
    lims = arena_limits_from_trace(frame)
    x0, x1, y0, y1 = lims
    t = np.asarray(disp["time"], dtype=float)
    fig = plt.figure(figsize=(13.5, 10.5), layout="constrained")
    gs = GridSpec(3, 3, figure=fig, hspace=0.12, wspace=0.12)
    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[0, 1])
    ax_link = fig.add_subplot(gs[0, 2])
    ax_err = fig.add_subplot(gs[1, :])
    ax_map = fig.add_subplot(gs[2, 0])
    ax_rad = fig.add_subplot(gs[2, 1])
    ax_extra = fig.add_subplot(gs[2, 2])

    norm = Normalize(vmin=float(np.nanmin(t)), vmax=float(np.nanmax(t)))
    sc = _raster_scatter(
        ax_true, disp["true_x"], disp["true_y"],
        c=t, cmap=TIME_CMAP, marker=TRUE_MARKER, s=14, alpha=0.75,
    )
    ax_true.set_title("True trajectory")
    ax_true.set_xlabel("x (cm)")
    ax_true.set_ylabel("y (cm)")
    panel_label(ax_true, "A")

    _raster_scatter(
        ax_pred, disp["pred_x"], disp["pred_y"],
        c=t, cmap=TIME_CMAP, marker=PRED_MARKER, s=16, alpha=0.75,
    )
    ax_pred.set_title("Predicted trajectory")
    ax_pred.set_xlabel("x (cm)")
    ax_pred.set_ylabel("y (cm)")
    panel_label(ax_pred, "B")

    idx = sparse_link_indices(len(disp))
    segs = np.stack(
        [
            np.column_stack([disp["true_x"].iloc[idx], disp["true_y"].iloc[idx]]),
            np.column_stack([disp["pred_x"].iloc[idx], disp["pred_y"].iloc[idx]]),
        ],
        axis=1,
    )
    ax_link.add_collection(LineCollection(segs, colors="0.55", linewidths=0.4, alpha=0.5, rasterized=True))
    _raster_scatter(
        ax_link, disp["true_x"].iloc[idx], disp["true_y"].iloc[idx],
        c=TRUE_COLOR, marker=TRUE_MARKER, s=10, alpha=0.8,
    )
    _raster_scatter(
        ax_link, disp["pred_x"].iloc[idx], disp["pred_y"].iloc[idx],
        c=PRED_COLOR, marker=PRED_MARKER, s=12, alpha=0.8,
    )
    ax_link.set_title("True → predicted (sparse)")
    ax_link.set_xlabel("x (cm)")
    ax_link.set_ylabel("y (cm)")
    panel_label(ax_link, "C")

    for ax in (ax_true, ax_pred, ax_link):
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal", adjustable="box")
        style_axes(ax)

    fig.colorbar(sc, ax=[ax_true, ax_pred], location="right", shrink=0.8, pad=0.02, label="Time (s)")

    et = np.asarray(frame["time"], dtype=float)
    ee = np.asarray(frame["error_cm"], dtype=float)
    ax_err.plot(et, ee, color=PRED_COLOR, lw=0.8, rasterized=True)
    med = float(np.nanmedian(ee))
    p90 = float(np.nanpercentile(ee, 90))
    ax_err.axhline(med, color="0.25", ls="--", lw=1.0, label=f"median {med:.1f} cm")
    ax_err.axhline(p90, color="0.45", ls=":", lw=1.0, label=f"P90 {p90:.1f} cm")
    if frame_b is not None and "error_cm" in frame_b.columns:
        align = align_config_pair(frame, frame_b)
        if align.aligned:
            diff = (
                np.asarray(align.frame_a["error_cm"], dtype=float)
                - np.asarray(align.frame_b["error_cm"], dtype=float)
            )
            ax_err.plot(
                np.asarray(align.frame_a["time"], dtype=float),
                diff, color=PRED_B_COLOR, lw=0.7, alpha=0.85, label=f"{label_a}−{label_b}",
                rasterized=True,
            )
    ax_err.set_xlabel("Time (s)")
    ax_err.set_ylabel("Position error (cm)")
    ax_err.set_title("Error through time")
    ax_err.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, borderaxespad=0.0)
    panel_label(ax_err, "D")
    style_axes(ax_err)

    smap = spatial_error_map(frame, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
    mesh = ax_map.pcolormesh(
        smap["x_edges"], smap["y_edges"], smap["mean_error"],
        cmap=ERROR_CMAP, shading="auto",
    )
    ax_map.set_xlim(x0, x1)
    ax_map.set_ylim(y0, y1)
    ax_map.set_aspect("equal", adjustable="box")
    ax_map.set_title("Spatial error (true position)")
    ax_map.set_xlabel("x (cm)")
    ax_map.set_ylabel("y (cm)")
    fig.colorbar(mesh, ax=ax_map, label="Position error (cm)")
    panel_label(ax_map, "E")
    style_axes(ax_map)

    rad = radial_shrinkage(frame, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
    rt, rp = rad["r_true"], rad["r_pred"]
    _raster_scatter(ax_rad, rt, rp, c=TRUE_COLOR, s=8, alpha=0.35)
    hi = float(np.nanmax([np.nanmax(rt), np.nanmax(rp), 1.0]))
    ax_rad.plot([0, hi], [0, hi], color="0.4", ls="--", lw=1.0)
    if np.isfinite(rad["slope"]):
        xs = np.linspace(0, hi, 50)
        ax_rad.plot(xs, rad["slope"] * xs + rad["intercept"], color=PRED_COLOR, lw=1.4)
        ax_rad.set_xlabel("True radius from center (cm)")
        ax_rad.set_ylabel("Predicted radius (cm)")
        ax_rad.set_title(f"Radial shrinkage (slope={rad['slope']:.2f})")
    else:
        ax_rad.set_title("Radial shrinkage")
    ax_rad.set_aspect("equal", adjustable="box")
    panel_label(ax_rad, "F")
    style_axes(ax_rad)

    if frame_b is not None:
        align = align_config_pair(frame, frame_b)
        if align.aligned:
            tmp = frame.copy()
            tmp["error_cm"] = (
                np.asarray(align.frame_a["error_cm"], dtype=float)
                - np.asarray(align.frame_b["error_cm"], dtype=float)
            )
            dmap = spatial_error_map(tmp, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
            vmax = np.nanmax(np.abs(dmap["mean_error"]))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            im = ax_extra.pcolormesh(
                dmap["x_edges"], dmap["y_edges"], dmap["mean_error"],
                cmap=DIFF_CMAP, shading="auto", vmin=-vmax, vmax=vmax,
            )
            ax_extra.set_title(f"Local error {label_a} − {label_b}")
            fig.colorbar(im, ax=ax_extra, label="Δ error (cm)")
        else:
            ax_extra.text(0.5, 0.5, align.message or "not aligned", ha="center", va="center", wrap=True)
            ax_extra.set_axis_off()
    else:
        ax_extra.set_axis_off()
        ax_extra.text(
            0.5, 0.5,
            "Select a second configuration\nto map local error difference.",
            ha="center", va="center", transform=ax_extra.transAxes,
        )
    ax_extra.set_xlim(x0, x1)
    ax_extra.set_ylim(y0, y1)
    ax_extra.set_aspect("equal", adjustable="box")
    style_axes(ax_extra)

    fig.suptitle(f"{title}\n{SPLIT_CAPTION}", fontsize=13)
    if path is None:
        return fig
    return save_decoding_figure(fig, path, dpi=FIGURE_DPI)


def plot_scalar_diagnostics(
    frame: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    path: Path | None = None,
    frame_b: pd.DataFrame | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> Path | Any:
    disp = downsample_for_display(frame)
    fig, axes = new_decoding_figure(3, 2, figsize=(12.5, 10.0))
    ax_ts, ax_res = axes[0, 0], axes[0, 1]
    ax_sc, ax_rv = axes[1, 0], axes[1, 1]
    ax_bin, ax_cmp = axes[2, 0], axes[2, 1]

    ax_ts.plot(disp["time"], disp["true"], color=TRUE_COLOR, lw=LINEWIDTH, label="True")
    ax_ts.plot(disp["time"], disp["pred"], color=PRED_COLOR, lw=LINEWIDTH, ls="--", label="Predicted")
    ax_ts.set_title("True and predicted through time")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel(ylabel)
    ax_ts.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    panel_label(ax_ts, "A")

    ax_res.plot(disp["time"], disp["residual"], color=PRED_COLOR, lw=0.9, rasterized=True)
    ax_res.axhline(0.0, color="0.3", lw=1.0)
    ax_res.set_title("Residual (pred − true)")
    ax_res.set_xlabel("Time (s)")
    ax_res.set_ylabel("Residual")
    panel_label(ax_res, "B")

    t_all = np.asarray(frame["true"], dtype=float)
    p_all = np.asarray(frame["pred"], dtype=float)
    _raster_scatter(ax_sc, t_all, p_all, c=PRED_COLOR, s=8, alpha=0.35)
    lo = float(np.nanmin([np.nanmin(t_all), np.nanmin(p_all)]))
    hi = float(np.nanmax([np.nanmax(t_all), np.nanmax(p_all)]))
    ax_sc.plot([lo, hi], [lo, hi], color="0.35", ls="--", lw=1.0)
    slope = compression_slope(t_all, p_all)
    ax_sc.set_title(f"Predicted vs actual (slope={slope['slope']:.2f})")
    ax_sc.set_xlabel("True")
    ax_sc.set_ylabel("Predicted")
    ax_sc.set_aspect("equal", adjustable="box")
    panel_label(ax_sc, "C")

    ax_rv.axhline(0.0, color="0.3", lw=1.0)
    _raster_scatter(ax_rv, t_all, np.asarray(frame["residual"], dtype=float), c=PRED_COLOR, s=8, alpha=0.35)
    ax_rv.set_title("Residual vs actual")
    ax_rv.set_xlabel("True")
    ax_rv.set_ylabel("Residual")
    panel_label(ax_rv, "D")

    bins = magnitude_bins(t_all, p_all)
    if not bins.empty:
        ax_bin.plot(bins["mean_true"], bins["mean_pred"], color=PRED_COLOR, marker="o", lw=1.3)
        ax_bin.plot(bins["mean_true"], bins["mean_true"], color=TRUE_COLOR, ls="--", lw=1.0)
        ax_bin.set_title("Bias by true magnitude")
        ax_bin.set_xlabel("True (bin mean)")
        ax_bin.set_ylabel("Predicted (bin mean)")
    panel_label(ax_bin, "E")

    if frame_b is not None:
        align = align_config_pair(frame, frame_b)
        if align.aligned:
            ea = np.abs(np.asarray(align.frame_a["residual"], dtype=float))
            eb = np.abs(np.asarray(align.frame_b["residual"], dtype=float))
            ax_cmp.plot(align.frame_a["true"], ea - eb, color=PRED_B_COLOR, lw=0.0, marker="o", ms=2, alpha=0.5)
            ax_cmp.axhline(0.0, color="0.3", lw=1.0)
            ax_cmp.set_title(f"|err| {label_a} − {label_b} vs true")
            ax_cmp.set_xlabel("True")
            ax_cmp.set_ylabel("Δ |residual|")
        else:
            ax_cmp.text(0.5, 0.5, align.message or "", ha="center", va="center", transform=ax_cmp.transAxes)
            ax_cmp.set_axis_off()
    else:
        ax_cmp.set_axis_off()
    panel_label(ax_cmp, "F")

    fig.suptitle(f"{title}\n{SPLIT_CAPTION}", fontsize=13)
    if path is None:
        return fig
    return save_decoding_figure(fig, path, dpi=FIGURE_DPI)


def plot_head_direction_diagnostics(
    frame: pd.DataFrame,
    *,
    title: str,
    path: Path | None = None,
    frame_b: pd.DataFrame | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> Path | Any:
    disp = downsample_for_display(frame)
    tw, pw = wrap_head_direction_series(disp["true_deg"], disp["pred_deg"])
    fig, axes = new_decoding_figure(2, 2, figsize=(12.0, 8.8))
    ax_ts, ax_ce = axes[0, 0], axes[0, 1]
    ax_sc, ax_bin = axes[1, 0], axes[1, 1]

    ax_ts.plot(disp["time"], tw, color=TRUE_COLOR, lw=LINEWIDTH, label="True")
    ax_ts.plot(disp["time"], pw, color=PRED_COLOR, lw=LINEWIDTH, ls="--", label="Predicted (unwrapped to true)")
    ax_ts.set_title("Head direction through time")
    ax_ts.set_xlabel("Time (s)")
    ax_ts.set_ylabel("Degrees")
    ax_ts.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    panel_label(ax_ts, "A")

    ax_ce.plot(disp["time"], disp["circular_error_deg"], color=PRED_COLOR, lw=0.9, rasterized=True)
    ax_ce.set_title("Circular error (shortest arc)")
    ax_ce.set_xlabel("Time (s)")
    ax_ce.set_ylabel("Error (deg)")
    ax_ce.set_ylim(0, 180)
    panel_label(ax_ce, "B")

    _raster_scatter(ax_sc, frame["true_deg"] % 360.0, frame["pred_deg"] % 360.0, c=PRED_COLOR, s=8, alpha=0.35)
    ax_sc.plot([0, 360], [0, 360], color="0.35", ls="--", lw=1.0)
    ax_sc.set_xlim(0, 360)
    ax_sc.set_ylim(0, 360)
    ax_sc.set_aspect("equal", adjustable="box")
    ax_sc.set_title("Predicted vs true (mod 360°)")
    ax_sc.set_xlabel("True (deg)")
    ax_sc.set_ylabel("Predicted (deg)")
    panel_label(ax_sc, "C")

    by_ang = circular_error_by_angle(frame["true_deg"], frame["circular_error_deg"])
    if not by_ang.empty:
        ax_bin.bar(by_ang["bin_center_deg"], by_ang["mean_circular_error_deg"], width=20, color=PRED_COLOR, alpha=0.85)
        ax_bin.set_xlim(0, 360)
        ax_bin.set_xlabel("True head direction (deg)")
        ax_bin.set_ylabel("Mean circular error (deg)")
        ax_bin.set_title("Error by true angle")
    if frame_b is not None:
        align = align_config_pair(frame, frame_b)
        if align.aligned:
            diff = (
                np.asarray(align.frame_a["circular_error_deg"], dtype=float)
                - np.asarray(align.frame_b["circular_error_deg"], dtype=float)
            )
            ax_ce.plot(align.frame_a["time"], diff, color=PRED_B_COLOR, lw=0.7, alpha=0.8, label=f"{label_a}−{label_b}")
            ax_ce.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    panel_label(ax_bin, "D")

    fig.suptitle(f"{title}\n{SPLIT_CAPTION}", fontsize=13)
    if path is None:
        return fig
    return save_decoding_figure(fig, path, dpi=FIGURE_DPI)


def plot_categorical_diagnostics(
    frame: pd.DataFrame,
    *,
    title: str,
    class_labels: list[str] | None = None,
    path: Path | None = None,
    frame_b: pd.DataFrame | None = None,
    label_a: str = "A",
    label_b: str = "B",
) -> Path | Any:
    labels = class_labels or sorted(set(frame["true"].astype(str)) | set(frame["pred"].astype(str)))
    cmap = plt.get_cmap("tab10")
    color_of = {lab: cmap(i % 10) for i, lab in enumerate(labels)}
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    t_idx = np.array([lab_to_i.get(str(v), -1) for v in frame["true"]])
    p_idx = np.array([lab_to_i.get(str(v), -1) for v in frame["pred"]])
    correct = frame["true"].astype(str).to_numpy() == frame["pred"].astype(str).to_numpy()
    times = np.asarray(frame["time"], dtype=float)

    fig, axes = new_decoding_figure(3, 2, figsize=(13.0, 10.2))
    ax_true, ax_pred = axes[0, 0], axes[0, 1]
    ax_ok, ax_cm = axes[1, 0], axes[1, 1]
    ax_pr, ax_rec = axes[2, 0], axes[2, 1]

    def _strip(ax, idx, name, letter):
        ax.imshow(
            idx.reshape(1, -1), aspect="auto", interpolation="nearest",
            cmap=cmap, vmin=0, vmax=max(len(labels) - 1, 1),
            extent=[times.min(), times.max(), 0, 1],
        )
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")
        ax.set_title(name)
        panel_label(ax, letter)
        style_axes(ax)

    _strip(ax_true, t_idx, "True labels", "A")
    _strip(ax_pred, p_idx, "Predicted labels", "B")
    ax_ok.imshow(
        (~correct).reshape(1, -1).astype(float), aspect="auto", interpolation="nearest",
        cmap="Greys", vmin=0, vmax=1,
        extent=[times.min(), times.max(), 0, 1],
    )
    ax_ok.set_yticks([])
    ax_ok.set_xlabel("Time (s)")
    ax_ok.set_title("Incorrect (black)")
    panel_label(ax_ok, "C")

    mat, labs = confusion_from_trace(frame, class_labels=labels)
    im = ax_cm.imshow(mat, cmap="Blues")
    ax_cm.set_xticks(range(len(labs)))
    ax_cm.set_yticks(range(len(labs)))
    ax_cm.set_xticklabels(labs, rotation=30, ha="right")
    ax_cm.set_yticklabels(labs)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")
    ax_cm.set_title("Confusion (this configuration)")
    fig.colorbar(im, ax=ax_cm, shrink=0.8)
    panel_label(ax_cm, "D")

    proba_pred = predicted_class_probability(frame, labels)
    if proba_pred is not None:
        ax_pr.plot(times, proba_pred, color=PRED_COLOR, lw=0.9, rasterized=True)
        ax_pr.set_ylim(0, 1.05)
        ax_pr.set_title("P(predicted class)")
        ax_pr.set_xlabel("Time (s)")
        ax_pr.set_ylabel("Probability")
    else:
        ax_pr.text(0.5, 0.5, "Class probabilities not stored", ha="center", va="center", transform=ax_pr.transAxes)
        ax_pr.set_axis_off()
    panel_label(ax_pr, "E")

    rec = recall_by_true_class(frame, class_labels=labels)
    colors = [color_of[c] for c in rec["class"]]
    ax_rec.bar(rec["class"], rec["recall"], color=colors)
    ax_rec.set_ylim(0, 1.05)
    ax_rec.set_ylabel("Recall")
    ax_rec.set_title("Recall by true class")
    ax_rec.tick_params(axis="x", rotation=30)
    panel_label(ax_rec, "F")

    if frame_b is not None:
        align = align_config_pair(frame, frame_b)
        if align.aligned:
            ok_a = align.frame_a["true"].astype(str) == align.frame_a["pred"].astype(str)
            ok_b = align.frame_b["true"].astype(str) == align.frame_b["pred"].astype(str)
            only_a = ok_a & ~ok_b
            only_b = ok_b & ~ok_a
            ax_ok.plot(
                times[only_a.to_numpy()], np.full(int(only_a.sum()), 0.75),
                "|", color=PRED_COLOR, ms=4, label=f"{label_a} only correct",
            )
            ax_ok.plot(
                times[only_b.to_numpy()], np.full(int(only_b.sum()), 0.25),
                "|", color=PRED_B_COLOR, ms=4, label=f"{label_b} only correct",
            )
            ax_ok.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    fig.suptitle(f"{title}\n{SPLIT_CAPTION}", fontsize=13)
    if path is None:
        return fig
    return save_decoding_figure(fig, path, dpi=FIGURE_DPI)


def plot_where_decoding_succeeds_fails(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    target: str,
    title: str,
    label_a: str,
    label_b: str,
    class_labels: list[str] | None = None,
    path: Path,
) -> Path:
    family = target_diagnostic_family(target)
    align = align_config_pair(frame_a, frame_b)
    fig = plt.figure(figsize=(14.0, 9.2), layout="constrained")
    fig.suptitle(f"Where decoding succeeds and fails\n{title}\n{SPLIT_CAPTION}", fontsize=13)

    if family == "position":
        gs = GridSpec(2, 3, figure=fig)
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
        lims = arena_limits_from_trace(pd.concat([frame_a, frame_b], ignore_index=True))
        x0, x1, y0, y1 = lims
        disp_a = downsample_for_display(frame_a)
        disp_b = downsample_for_display(frame_b)
        panels = [
            (axes[0], disp_a["true_x"], disp_a["true_y"], disp_a["time"], TRUE_MARKER, "Ground truth"),
            (axes[1], disp_a["pred_x"], disp_a["pred_y"], disp_a["time"], PRED_MARKER, f"Predicted · {label_a}"),
            (axes[2], disp_b["pred_x"], disp_b["pred_y"], disp_b["time"], PRED_MARKER, f"Predicted · {label_b}"),
        ]
        letters = "ABC"
        sc0 = None
        for ax, x, y, t, mk, ttl in panels:
            sc0 = _raster_scatter(ax, x, y, c=t, cmap=TIME_CMAP, marker=mk, s=12, alpha=0.7)
            ax.set_title(ttl)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            style_axes(ax)
        for ax, lab in zip(axes[:3], letters):
            panel_label(ax, lab)
        if sc0 is not None:
            fig.colorbar(sc0, ax=axes[:3], shrink=0.7, label="Time (s)")

        sm_a = spatial_error_map(frame_a, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
        sm_b = spatial_error_map(frame_b, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
        vmax = np.nanmax([np.nanmax(sm_a["mean_error"]), np.nanmax(sm_b["mean_error"])])
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        for ax, sm, ttl, lab in (
            (axes[3], sm_a, f"Error · {label_a}", "D"),
            (axes[4], sm_b, f"Error · {label_b}", "E"),
        ):
            mesh = ax.pcolormesh(
                sm["x_edges"], sm["y_edges"], sm["mean_error"],
                cmap=ERROR_CMAP, vmin=0, vmax=vmax, shading="auto",
            )
            ax.set_title(ttl)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("x (cm)")
            ax.set_ylabel("y (cm)")
            fig.colorbar(mesh, ax=ax, label="Position error (cm)")
            panel_label(ax, lab)
            style_axes(ax)
        if align.aligned:
            tmp = frame_a.copy()
            tmp["error_cm"] = (
                np.asarray(align.frame_a["error_cm"]) - np.asarray(align.frame_b["error_cm"])
            )
            dm = spatial_error_map(tmp, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
            dv = np.nanmax(np.abs(dm["mean_error"]))
            if not np.isfinite(dv) or dv == 0:
                dv = 1.0
            im = axes[5].pcolormesh(
                dm["x_edges"], dm["y_edges"], dm["mean_error"],
                cmap=DIFF_CMAP, vmin=-dv, vmax=dv, shading="auto",
            )
            axes[5].set_title(f"Error {label_a} − {label_b}")
            fig.colorbar(im, ax=axes[5], label="Δ error (cm)")
        else:
            axes[5].text(0.5, 0.5, align.message or "", ha="center", va="center", transform=axes[5].transAxes)
            axes[5].set_axis_off()
        axes[5].set_xlim(x0, x1)
        axes[5].set_ylim(y0, y1)
        axes[5].set_aspect("equal", adjustable="box")
        panel_label(axes[5], "F")
        style_axes(axes[5])
        return save_decoding_figure(fig, path, dpi=FIGURE_DPI)

    if family == "categorical":
        labels = class_labels or sorted(
            set(frame_a["true"].astype(str)) | set(frame_a["pred"].astype(str))
            | set(frame_b["true"].astype(str)) | set(frame_b["pred"].astype(str))
        )
        gs = GridSpec(3, 2, figure=fig)
        ax = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
        cmap = plt.get_cmap("tab10")
        lab_to_i = {lab: i for i, lab in enumerate(labels)}
        times = np.asarray(frame_a["time"], dtype=float)
        for axi, series, ttl, letter in (
            (ax[0], frame_a["true"], "Ground truth", "A"),
            (ax[1], frame_a["pred"], f"Predicted · {label_a}", "B"),
            (ax[2], frame_b["pred"], f"Predicted · {label_b}", "C"),
        ):
            idx = np.array([lab_to_i.get(str(v), -1) for v in series])
            axi.imshow(
                idx.reshape(1, -1), aspect="auto", interpolation="nearest",
                cmap=cmap, vmin=0, vmax=max(len(labels) - 1, 1),
                extent=[times.min(), times.max(), 0, 1],
            )
            axi.set_yticks([])
            axi.set_title(ttl)
            axi.set_xlabel("Time (s)")
            panel_label(axi, letter)
        ok_a = frame_a["true"].astype(str) != frame_a["pred"].astype(str)
        ok_b = frame_b["true"].astype(str) != frame_b["pred"].astype(str)
        ax[3].imshow(ok_a.to_numpy().reshape(1, -1).astype(float), aspect="auto", cmap="Greys",
                     extent=[times.min(), times.max(), 0, 1], interpolation="nearest")
        ax[4].imshow(ok_b.to_numpy().reshape(1, -1).astype(float), aspect="auto", cmap="Greys",
                     extent=[times.min(), times.max(), 0, 1], interpolation="nearest")
        ax[3].set_title(f"Errors · {label_a}")
        ax[4].set_title(f"Errors · {label_b}")
        for axi, letter in ((ax[3], "D"), (ax[4], "E")):
            axi.set_yticks([])
            axi.set_xlabel("Time (s)")
            panel_label(axi, letter)
        if align.aligned:
            only_a = (~ok_a) & ok_b
            only_b = (~ok_b) & ok_a
            ax[5].plot(times, only_a.astype(float) - only_b.astype(float), color=PRED_COLOR, lw=0.8)
            ax[5].set_title(f"{label_a} unique correct minus {label_b}")
        else:
            ax[5].text(0.5, 0.5, align.message or "", ha="center", va="center", transform=ax[5].transAxes)
        panel_label(ax[5], "F")
        return save_decoding_figure(fig, path, dpi=FIGURE_DPI)

    # scalar / head_direction
    true_col = "true_deg" if family == "head_direction" else "true"
    pred_col = "pred_deg" if family == "head_direction" else "pred"
    gs = GridSpec(2, 3, figure=fig)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    da = downsample_for_display(frame_a)
    db = downsample_for_display(frame_b)
    axes[0].plot(da["time"], da[true_col], color=TRUE_COLOR, lw=LINEWIDTH)
    axes[0].set_title("Ground truth")
    panel_label(axes[0], "A")
    axes[1].plot(da["time"], da[true_col], color=TRUE_COLOR, lw=1.0, alpha=0.45)
    axes[1].plot(da["time"], da[pred_col], color=PRED_COLOR, lw=LINEWIDTH, ls="--")
    axes[1].set_title(f"Predicted · {label_a}")
    panel_label(axes[1], "B")
    axes[2].plot(db["time"], db[true_col] if true_col in db.columns else da[true_col], color=TRUE_COLOR, lw=1.0, alpha=0.45)
    axes[2].plot(db["time"], db[pred_col], color=PRED_B_COLOR, lw=LINEWIDTH, ls="--")
    axes[2].set_title(f"Predicted · {label_b}")
    panel_label(axes[2], "C")
    ea = absolute_error_series(frame_a, family)
    eb = absolute_error_series(frame_b, family)
    if ea is not None:
        axes[3].plot(frame_a["time"], ea, color=PRED_COLOR, lw=0.8)
    axes[3].set_title(f"Error · {label_a}")
    panel_label(axes[3], "D")
    if eb is not None:
        axes[4].plot(frame_b["time"], eb, color=PRED_B_COLOR, lw=0.8)
    axes[4].set_title(f"Error · {label_b}")
    panel_label(axes[4], "E")
    if align.aligned and ea is not None and eb is not None:
        axes[5].plot(align.frame_a["time"], ea - eb, color="0.2", lw=0.8)
        axes[5].axhline(0.0, color="0.5", lw=1.0)
        axes[5].set_title(f"Error {label_a} − {label_b}")
    else:
        axes[5].text(0.5, 0.5, (align.message if not align.aligned else ""), ha="center", va="center", transform=axes[5].transAxes)
    panel_label(axes[5], "F")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        style_axes(ax)
    return save_decoding_figure(fig, path, dpi=FIGURE_DPI)


def export_diagnostics_figure(
    *,
    experiment_dir: Path,
    target: str,
    frame_a: pd.DataFrame,
    meta_a: dict[str, str] | None = None,
    frame_b: pd.DataFrame | None = None,
    meta_b: dict[str, str] | None = None,
    label_a: str = "Config A",
    label_b: str = "Config B",
    figures_dir: Path | None = None,
    where_succeeds_fails: bool = False,
) -> Path:
    """Write PNG under ``figures/decoder_comparison/diagnostics/``."""
    out_dir = diagnostics_figure_dir(experiment_dir, figures_dir)
    family = target_diagnostic_family(target)
    labels = class_labels_from_meta(meta_a or {}) if meta_a else None
    if where_succeeds_fails and frame_b is not None:
        path = out_dir / f"fig_where_decoding_succeeds_fails_{target}.png"
        return plot_where_decoding_succeeds_fails(
            frame_a, frame_b, target=target, title=target,
            label_a=label_a, label_b=label_b, class_labels=labels, path=path,
        )
    path = out_dir / f"fig_decoding_diagnostics_{target}.png"
    title = f"{target} · {label_a}" + (f" vs {label_b}" if frame_b is not None else "")
    if family == "position":
        return plot_position_diagnostics(frame_a, title=title, path=path, frame_b=frame_b, label_a=label_a, label_b=label_b)
    if family == "head_direction":
        return plot_head_direction_diagnostics(frame_a, title=title, path=path, frame_b=frame_b, label_a=label_a, label_b=label_b)
    if family == "categorical":
        return plot_categorical_diagnostics(
            frame_a, title=title, class_labels=labels, path=path, frame_b=frame_b,
            label_a=label_a, label_b=label_b,
        )
    ylabel = target.replace("_", " ")
    return plot_scalar_diagnostics(
        frame_a, title=title, ylabel=ylabel, path=path, frame_b=frame_b,
        label_a=label_a, label_b=label_b,
    )


def generate_default_success_failure_figures(
    experiment_dir: Path,
    figures_dir: Path | None = None,
) -> list[Path]:
    """Best vs counts baseline for targets that have parquet traces; skip if missing."""
    from realtime.comparison_metrics_union import ensure_config_ids
    from realtime.decoder_comparison import PRIMARY_METRIC
    from realtime.prediction_artifacts import prediction_parquet_path, read_prediction_trace

    experiment_dir = Path(experiment_dir)
    metrics_path = (
        experiment_dir / "decoder_comparison" / "sorted" / "decoder_comparison_metrics.csv"
    )
    if not metrics_path.is_file():
        return []
    metrics = ensure_config_ids(pd.read_csv(metrics_path))
    if metrics.empty:
        return []
    pred_root = metrics_path.parent
    written: list[Path] = []

    def _best_row(df: pd.DataFrame, target: str) -> pd.Series | None:
        sub = df[df["target_name"].astype(str) == target]
        if "exclusion_reason" in sub.columns:
            sub = sub[sub["exclusion_reason"].isna() | (sub["exclusion_reason"].astype(str) == "")]
        if sub.empty:
            return None
        key, direction = PRIMARY_METRIC[target]
        if key not in sub.columns:
            return sub.iloc[0]
        scored = pd.to_numeric(sub[key], errors="coerce")
        idx = scored.idxmin() if direction == "lower" else scored.idxmax()
        return sub.loc[idx]

    for target in ("position", "speed", "movement_state"):
        best = _best_row(metrics, target)
        if best is None:
            continue
        pool = metrics[metrics["target_name"].astype(str) == target]
        is_counts = pd.Series(False, index=pool.index)
        if "embedding_type" in pool.columns:
            is_counts = is_counts | pool["embedding_type"].astype(str).isin(("counts", "identity"))
        if "feature_mode" in pool.columns:
            is_counts = is_counts | pool["feature_mode"].astype(str).isin(("counts", "identity"))
        counts_pool = pool[is_counts]
        if counts_pool.empty:
            continue
        counts = _best_row(counts_pool, target)
        if counts is None:
            continue
        if str(best["config_id"]) == str(counts["config_id"]):
            continue
        path_a = prediction_parquet_path(pred_root, str(best["config_id"]))
        path_b = prediction_parquet_path(pred_root, str(counts["config_id"]))
        if not path_a.is_file() or not path_b.is_file():
            continue
        frame_a, meta_a = read_prediction_trace(path_a)
        frame_b, meta_b = read_prediction_trace(path_b)
        path = export_diagnostics_figure(
            experiment_dir=experiment_dir,
            target=target,
            frame_a=frame_a,
            meta_a=meta_a,
            frame_b=frame_b,
            meta_b=meta_b,
            label_a="best",
            label_b="counts",
            figures_dir=figures_dir,
            where_succeeds_fails=True,
        )
        written.append(path)
    return written
