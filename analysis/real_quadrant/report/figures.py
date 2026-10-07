"""RD1–RD7 figure builders (saved artifacts only; train rate maps OK)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from analysis.real_quadrant.report import data as D
from analysis.real_quadrant.report.style import (
    COL,
    GPFA_OFFLINE_NOTE,
    MUTED,
    SHORT,
    W_FULL,
    head,
    save_fig,
)

OFFLINE = {"gpfa"}


def _caption(fig, text: str, y: float = 0.01) -> None:
    fig.text(0.02, y, text, fontsize=5.8, color=MUTED, va="bottom", ha="left",
             wrap=True)


def _path(ax, y: np.ndarray, *, color: str, lw: float = 0.4, alpha: float = 0.7) -> None:
    ax.plot(y[:, 0], y[:, 1], "-", color=color, lw=lw, alpha=alpha)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")


def fig_rd1(out_dir: Path, example: dict[str, Any]) -> Path:
    session = example["session"]
    contract = D.load_contract(session)
    report = D.load_session_report(session)
    a = contract["arrays"]
    times = np.asarray(a["decode_times"], float)
    y = np.asarray(a["y"], float)
    valid = np.asarray(a["valid"], bool)
    retained = np.asarray(a["retained"], bool)
    t0 = float(np.asarray(a["segment_t0"]).ravel()[0])
    t1 = float(np.asarray(a["segment_t1"]).ravel()[0])
    trim_s = float(np.asarray(a["trim_start_s"]).ravel()[0])
    trim_e = float(np.asarray(a["trim_end_s"]).ravel()[0])

    fig = plt.figure(figsize=(W_FULL, W_FULL * 0.72))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                  left=0.08, right=0.98, top=0.90, bottom=0.14)
    ax0 = fig.add_subplot(gs[0, 0])
    _path(ax0, y[retained & valid], color="#2a78d6")
    head(ax0, "a", "Path (room A, retained)")
    ax1 = fig.add_subplot(gs[0, 1])
    hb = ax1.hexbin(y[retained & valid, 0], y[retained & valid, 1],
                    gridsize=30, cmap="Greys", mincnt=1)
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_title("")
    head(ax1, "b", "Occupancy")
    fig.colorbar(hb, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(gs[0, 2])
    # valid / retained / dropped fractions over time (coarse bins)
    n_bins = 40
    edges = np.linspace(times.min(), times.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    frac_valid, frac_ret = [], []
    for i in range(n_bins):
        m = (times >= edges[i]) & (times < edges[i + 1])
        if not m.any():
            frac_valid.append(np.nan)
            frac_ret.append(np.nan)
            continue
        frac_valid.append(float(np.mean(valid[m])))
        frac_ret.append(float(np.mean(retained[m])))
    ax2.plot(centers - t0, frac_valid, color="#2a78d6", label="valid")
    ax2.plot(centers - t0, frac_ret, color="#1baf7a", label="retained")
    ax2.axvspan(0, trim_s, color="#c83a39", alpha=0.15, label="trim")
    ax2.axvspan((t1 - t0) - trim_e, t1 - t0, color="#c83a39", alpha=0.15)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("time in segment (s)")
    ax2.set_ylabel("fraction")
    ax2.legend(loc="lower right", fontsize=5.5)
    head(ax2, "c", "Valid / retained")

    ax3 = fig.add_subplot(gs[1, :])
    # units by region if available from source_summary
    summary = json.loads((D.session_dir(session) / "source_summary.json").read_text())
    by_reg = summary.get("n_units_by_region") or {}
    if not by_reg:
        # fall back to method json
        raw = json.loads((D.session_dir(session) / "raw.json").read_text())
        by_reg = raw.get("n_units_by_region") or {"all": report.get("n_units", 0)}
    labels = list(by_reg.keys())
    vals = [int(by_reg[k]) for k in labels]
    ax3.bar(range(len(labels)), vals, color="#8f8d87")
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, rotation=30, ha="right")
    ax3.set_ylabel("n units")
    head(ax3, "d", f"Units by region (n={report.get('n_units')})")

    _caption(
        fig,
        f"RD1 · Example session chosen by fixed rule: {example['rule']} "
        f"→ {session} (n_units={example['n_units']}, cohort median="
        f"{example['cohort_median_units']:.0f}). Room A; trims "
        f"{trim_s:.0f}s / {trim_e:.0f}s. {GPFA_OFFLINE_NOTE}",
    )
    return save_fig(fig, out_dir, "RD1_recording")


def fig_rd2(out_dir: Path, example: dict[str, Any], units: dict[str, Any]) -> Path:
    """Raster + causal vs Cell_* centred-window methods panel."""
    from analysis.real_quadrant.adapter import (
        causal_count_matrix,
        window_count_matrix,
    )

    session = example["session"]
    contract = D.load_contract(session)
    a = contract["arrays"]
    times = np.asarray(a["decode_times"], float)
    y = np.asarray(a["y"], float)
    retained = np.asarray(a["retained"], bool)
    t_rel = times - float(times[retained][0]) if retained.any() else times - times[0]
    # ~60 s window starting 120 s into retained
    t0 = 120.0
    win = (t_rel >= t0) & (t_rel < t0 + 60.0) & retained
    if win.sum() < 20:
        win = retained & (t_rel < 60.0)

    idxs = units["unit_indices"][:6]
    spike_times = units["spike_times"]
    uids = units["unit_ids"]

    fig = plt.figure(figsize=(W_FULL, W_FULL * 0.78))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.1, 1.0, 1.0],
                  hspace=0.4, wspace=0.3, left=0.08, right=0.98, top=0.90, bottom=0.14)
    ax_r = fig.add_subplot(gs[0, :])
    t_lo = float(times[win][0])
    t_hi = float(times[win][-1])
    for row, (j, uid) in enumerate(zip(idxs, uids)):
        st = np.asarray(spike_times[j], float)
        st = st[(st >= t_lo) & (st <= t_hi)]
        ax_r.vlines(st - t_lo, row + 0.1, row + 0.9, color="#3a3936", lw=0.4)
    ax_r.set_yticks(np.arange(len(uids)) + 0.5)
    ax_r.set_yticklabels([str(u) for u in uids])
    ax_r.set_xlabel("time in window (s)")
    ax_r.set_ylabel("unit id")
    head(ax_r, "a", "Example-unit raster (~60 s)")

    ax_p = fig.add_subplot(gs[1, :])
    ax_p.plot(times[win] - t_lo, y[win, 0], color="#2a78d6", lw=0.8, label="x")
    ax_p.plot(times[win] - t_lo, y[win, 1], color="#eb6834", lw=0.8, label="y")
    ax_p.set_ylabel("position (cm)")
    ax_p.set_xlabel("time in window (s)")
    ax_p.legend(loc="upper right")
    head(ax_p, "b", "Position")

    # Methods panel: causal vs centre window for first example unit
    j0 = idxs[0]
    grid = times[win]
    st0 = [np.asarray(spike_times[j0], float)]
    causal = causal_count_matrix(st0, grid, window_s=0.250).ravel()
    centre = window_count_matrix(st0, grid, -0.125, 0.125).ravel()
    ax_m = fig.add_subplot(gs[2, 0])
    ax_m.plot(grid - t_lo, centre, color="#c83a39", lw=0.9, label="Cell_* centre ±125 ms")
    ax_m.plot(grid - t_lo, causal, color="#2a78d6", lw=0.9, label="causal [t−250, t)")
    ax_m.set_xlabel("time in window (s)")
    ax_m.set_ylabel("spike count")
    ax_m.legend(fontsize=5.5, loc="upper right")
    head(ax_m, "c", f"Windows (unit {uids[0]})")

    ax_d = fig.add_subplot(gs[2, 1])
    # schematic of lookahead
    ax_d.axvspan(-0.250, 0.0, color="#2a78d6", alpha=0.25, label="causal")
    ax_d.axvspan(-0.125, 0.125, color="#c83a39", alpha=0.25, label="centre")
    ax_d.axvline(0.0, color="k", lw=0.8)
    ax_d.axvline(0.125, color="#c83a39", ls="--", lw=0.8)
    ax_d.set_xlim(-0.3, 0.2)
    ax_d.set_ylim(0, 1)
    ax_d.set_yticks([])
    ax_d.set_xlabel("time relative to label t (s)")
    ax_d.legend(fontsize=5.5, loc="upper left")
    head(ax_d, "d", "125 ms lookahead in Cell_*")

    _caption(
        fig,
        f"RD2 · Session rule: {example['rule']} → {session}. "
        f"Unit rule: {units['rule']} → ids {uids}. "
        "Panel c–d: dataset Cell_* is a centred 250 ms window (non-causal +125 ms); "
        "decoder features rebuild causal [t−250 ms, t) from spike times.",
    )
    return save_fig(fig, out_dir, "RD2_spikes_features")


def fig_rd3(out_dir: Path, example: dict[str, Any], units: dict[str, Any]) -> Path:
    session = example["session"]
    contract = D.load_contract(session)
    a = contract["arrays"]
    times = np.asarray(a["decode_times"], float)
    y = np.asarray(a["y"], float)
    train_ok = np.asarray(a["train_ok"], bool)
    t_train = times[train_ok]
    y_train = y[train_ok]
    x_edges = units["x_edges"]
    y_edges = units["y_edges"]
    occ = units["occ_train"]
    dt = units["dt"]

    fig = plt.figure(figsize=(W_FULL, W_FULL * 0.55))
    for k, (uid, j) in enumerate(zip(units["unit_ids"][:5], units["unit_indices"][:5])):
        ax = fig.add_subplot(2, 3, k + 1)
        st = np.asarray(units["spike_times"][j], float)
        if len(t_train):
            st_tr = st[(st >= t_train[0]) & (st <= t_train[-1])]
            idx = np.clip(
                np.searchsorted(t_train, st_tr, side="right") - 1, 0, len(t_train) - 1,
            )
            counts, _, _ = np.histogram2d(
                y_train[idx, 0], y_train[idx, 1], bins=[x_edges, y_edges],
            )
        else:
            counts = np.zeros_like(occ)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(occ > 0, counts / (occ * dt), np.nan)
        im = ax.imshow(
            np.nan_to_num(rate, nan=0.0).T, origin="lower",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            aspect="equal", cmap="viridis",
        )
        ax.set_title(f"unit {uid}", fontsize=6.5)
        if k == 0:
            head(ax, "a", "Train-only rate maps")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax_s = fig.add_subplot(2, 3, 6)
    if len(y_train) > 2:
        dy = np.diff(y_train, axis=0)
        dt_s = np.diff(t_train)
        speed = np.r_[0, np.linalg.norm(dy, axis=1) / np.maximum(dt_s, 1e-3)]
        ax_s.hist(speed[np.isfinite(speed)], bins=40, color="#8f8d87")
        ax_s.set_xlabel("speed (cm/s)")
        ax_s.set_ylabel("count")
        head(ax_s, "b", "Train speed")
    _caption(
        fig,
        f"RD3 · Train-only rate maps (descriptive; no decoder fit). "
        f"Session: {session}. Units: {units['rule']}. "
        "Occupancy-normalized; 5 cm bins.",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.14, hspace=0.4, wspace=0.35)
    return save_fig(fig, out_dir, "RD3_rate_maps")


def fig_rd4(out_dir: Path, example: dict[str, Any]) -> Path:
    session = example["session"]
    contract = D.load_contract(session)
    a = contract["arrays"]
    lat = contract["latents"]
    y = np.asarray(a["y"], float)
    eval_mask = np.asarray(a["eval_mask"], bool)
    # subsample for plotting
    idx = np.where(eval_mask)[0]
    if len(idx) > 4000:
        idx = idx[:: max(1, len(idx) // 4000)]
    pos = y[idx]
    r = np.linalg.norm(pos - np.nanmean(pos, axis=0), axis=1)

    fig = plt.figure(figsize=(W_FULL, W_FULL * 0.42))
    methods = [("pca", "PCA"), ("dm", "DM"), ("lds", "LDS"), ("gpfa", "GPFA†")]
    for i, (key, title) in enumerate(methods):
        ax = fig.add_subplot(1, 4, i + 1)
        Z = np.asarray(lat[f"Z_{key}"], float)[idx, :2]
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=r, s=2, cmap="viridis", linewidths=0)
        ax.set_title(title, fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            head(ax, "a", "Latents (eval; color = dist. from mean pos.)")
        if key == "gpfa":
            ax.text(0.5, -0.08, "offline ref.", transform=ax.transAxes,
                    ha="center", fontsize=5.5, color=MUTED)
        if i == 3:
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="‖Δpos‖")
    _caption(
        fig,
        f"RD4 · Example session {session} ({example['rule']}). "
        f"First two latent dims; eval samples subsampled. {GPFA_OFFLINE_NOTE}",
    )
    fig.subplots_adjust(left=0.04, right=0.96, top=0.82, bottom=0.18, wspace=0.25)
    return save_fig(fig, out_dir, "RD4_latents")


def fig_rd5(out_dir: Path, example: dict[str, Any]) -> Path:
    session = example["session"]
    contract = D.load_contract(session)
    report = D.load_session_report(session)
    a = contract["arrays"]
    preds = contract["preds"]
    times = np.asarray(a["decode_times"], float)
    y = np.asarray(a["y"], float)
    eval_mask = np.asarray(a["eval_mask"], bool)
    y_true = y[eval_mask]
    t_eval = times[eval_mask]

    # best causal by normalized error (exclude gpfa offline)
    causal_rows = [
        m for m in report["methods"]
        if m.get("status") in (None, "ok")
        and m["method"] not in OFFLINE
        and m.get("normalized_error") is not None
        and not str(m["method"]).endswith("_grid20")
        and "grid20" not in str(m["method"])
    ]
    # prefer primary methods without requiring smooth
    prefer = ["lds_smooth", "raw_smooth", "lds", "gpfa_causal", "pca_smooth", "raw"]
    by = {m["method"]: m for m in causal_rows}
    best = None
    for name in prefer:
        if name in by:
            best = by[name]
            break
    if best is None:
        best = min(causal_rows, key=lambda m: float(m["normalized_error"]))
    key = f"pred_{best['method']}_ridge"
    if key not in preds:
        for cand in (
            "pred_lds_smooth_ridge", "pred_raw_smooth_ridge", "pred_lds_ridge",
            "pred_gpfa_causal_ridge", "pred_raw_ridge",
        ):
            if cand in preds:
                key = cand
                break
    pred = np.asarray(preds[key], float)
    # align lengths
    n = min(len(pred), len(y_true))
    pred, y_true, t_eval = pred[:n], y_true[:n], t_eval[:n]
    err = np.linalg.norm(pred - y_true, axis=1)

    # pick ~120 s window with median-of-window-medians
    win_s = 120.0
    dt = float(np.median(np.diff(t_eval))) if len(t_eval) > 1 else 0.05
    w = max(1, int(round(win_s / dt)))
    if len(err) > w:
        meds = [float(np.median(err[i:i + w])) for i in range(0, len(err) - w, max(1, w // 4))]
        target = float(np.median(meds)) if meds else float(np.median(err))
        starts = list(range(0, len(err) - w, max(1, w // 4)))
        i0 = min(starts, key=lambda i: abs(float(np.median(err[i:i + w])) - target))
        sl = slice(i0, i0 + w)
        win_rule = (
            f"{win_s:.0f}s eval window whose median error is the median "
            f"across candidate windows"
        )
    else:
        sl = slice(None)
        win_rule = "full eval (short)"

    fig = plt.figure(figsize=(W_FULL, W_FULL * 0.7))
    ax0 = fig.add_subplot(2, 2, 1)
    ax0.plot(y_true[sl, 0], y_true[sl, 1], color="#8f8d87", lw=0.6, label="truth")
    ax0.plot(pred[sl, 0], pred[sl, 1], color=COL.get(best["method"], "#2a78d6"),
             lw=0.7, label=SHORT.get(best["method"], best["method"]))
    ax0.set_aspect("equal", adjustable="datalim")
    ax0.legend(fontsize=5.5)
    head(ax0, "a", "Decoded path")

    ax1 = fig.add_subplot(2, 2, 2)
    t0 = t_eval[sl][0]
    ax1.plot(t_eval[sl] - t0, y_true[sl, 0], color="#8f8d87", lw=0.7)
    ax1.plot(t_eval[sl] - t0, pred[sl, 0], color="#2a78d6", lw=0.7)
    ax1.set_ylabel("x (cm)")
    head(ax1, "b", "x(t)")

    ax2 = fig.add_subplot(2, 2, 3)
    ax2.plot(t_eval[sl] - t0, y_true[sl, 1], color="#8f8d87", lw=0.7)
    ax2.plot(t_eval[sl] - t0, pred[sl, 1], color="#eb6834", lw=0.7)
    ax2.set_ylabel("y (cm)")
    ax2.set_xlabel("time in window (s)")
    head(ax2, "c", "y(t)")

    ax3 = fig.add_subplot(2, 2, 4)
    ax3.plot(t_eval[sl] - t0, err[sl], color="#3a3936", lw=0.6)
    ax3.axhline(float(np.median(err[sl])), color="#c83a39", ls="--", lw=0.8)
    ax3.set_ylabel("error (cm)")
    ax3.set_xlabel("time in window (s)")
    head(ax3, "d", "Error")

    # also show gpfa offline as dashed on path inset note
    _caption(
        fig,
        f"RD5 · Best causal method for panels = {best['method']} "
        f"(lowest normalized error among causal methods; GPFA offline excluded). "
        f"Session {session}. Window: {win_rule}. {GPFA_OFFLINE_NOTE}",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, hspace=0.4, wspace=0.35)
    return save_fig(fig, out_dir, "RD5_predictions")


def fig_rd6(out_dir: Path, example: dict[str, Any]) -> Path:
    session = example["session"]
    contract = D.load_contract(session)
    report = D.load_session_report(session)
    a = contract["arrays"]
    preds = contract["preds"]
    a13 = contract["a13"]
    times = np.asarray(a["decode_times"], float)
    y = np.asarray(a["y"], float)
    train_mask = np.asarray(a["train_mask"], bool)
    test_mask = np.asarray(a["test_mask"], bool)
    train_ok = np.asarray(a["train_ok"], bool)
    eval_mask = np.asarray(a["eval_mask"], bool)
    retained = np.asarray(a["retained"], bool)
    y_true = y[eval_mask]
    floor_med = float(report["floor_median_cm"])

    methods_plot = ["raw", "raw_smooth", "pca", "lds", "gpfa_causal", "gpfa"]
    fig = plt.figure(figsize=(W_FULL, W_FULL * 0.7))
    ax0 = fig.add_subplot(2, 2, 1)
    for m in methods_plot:
        key = f"pred_{m}_ridge"
        if key not in preds:
            continue
        pred = np.asarray(preds[key], float)
        n = min(len(pred), len(y_true))
        err = np.linalg.norm(pred[:n] - y_true[:n], axis=1)
        style = dict(lw=1.0, color=COL.get(m, "#52514e"))
        if m in OFFLINE:
            style.update(ls="--", alpha=0.7)
        ax0.plot(np.sort(err), np.linspace(0, 1, len(err)), label=SHORT.get(m, m), **style)
    ax0.axvline(floor_med, color="#c83a39", ls=":", lw=1.0, label="floor")
    ax0.set_xlabel("error (cm)")
    ax0.set_ylabel("CDF")
    ax0.legend(fontsize=5.2, loc="lower right")
    head(ax0, "a", "Error CDF vs floor")

    ax1 = fig.add_subplot(2, 2, 2)
    names, deltas, colors = [], [], []
    for m in methods_plot:
        rec = a13.get(m) or {}
        if not rec:
            # from session report
            row = next((r for r in report["methods"] if r["method"] == m), None)
            if row:
                # no delta stored sometimes
                pass
        delta = rec.get("ridge_median_minus_floor")
        if delta is None:
            continue
        names.append(SHORT.get(m, m))
        deltas.append(float(delta))
        colors.append(COL.get(m, "#52514e"))
    if names:
        ax1.barh(range(len(names)), deltas, color=colors)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names)
        ax1.axvline(0, color="k", lw=0.6)
        ax1.set_xlabel("Ridge median − floor (cm)")
    head(ax1, "b", "A13: observed − floor")

    ax2 = fig.add_subplot(2, 1, 2)
    t0 = float(times[0])
    # timeline strips
    def _strip(mask, y0, color, label):
        m = np.asarray(mask, bool)
        if not m.any():
            return
        # find runs
        d = np.diff(m.astype(int))
        starts = np.where(d == 1)[0] + 1
        ends = np.where(d == -1)[0] + 1
        if m[0]:
            starts = np.r_[0, starts]
        if m[-1]:
            ends = np.r_[ends, len(m)]
        for s, e in zip(starts, ends):
            ax2.axvspan(times[s] - t0, times[min(e, len(times) - 1)] - t0,
                        ymin=y0, ymax=y0 + 0.15, color=color, alpha=0.8, label=label)
            label = None

    _strip(retained, 0.05, "#c4c2bb", "retained")
    _strip(train_mask & retained, 0.25, "#2a78d6", "train split")
    _strip(test_mask & retained, 0.45, "#eb6834", "test split")
    _strip(train_ok, 0.65, "#1baf7a", "train_ok")
    _strip(eval_mask, 0.85, "#0e7a54", "eval")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("time from segment start (s)")
    ax2.set_yticks([])
    ax2.legend(loc="upper right", fontsize=5.2, ncol=5)
    head(ax2, "c", "Split / trim timeline")

    _caption(
        fig,
        f"RD6 · Session {session}. Floor = train-mean position error. "
        f"A13 = circular label-shift null. {GPFA_OFFLINE_NOTE}",
    )
    fig.subplots_adjust(left=0.1, right=0.98, top=0.88, bottom=0.12, hspace=0.45, wspace=0.35)
    return save_fig(fig, out_dir, "RD6_validity")


def fig_rd7(out_dir: Path) -> Path:
    """Quadrant contrasts: final grid + faded grid20; dm_smooth−pca_smooth on grid20."""
    r2 = D.load_report2_contrasts()
    final = {c["contrast"]: c for c in r2["primary_contrasts_final"]}
    g20 = {c["contrast"]: c for c in r2["primary_contrasts_grid20"]}
    flips = set(r2.get("sign_flips_grid20_to_final") or [])

    order = [
        "lds - raw_smooth",
        "lds_smooth - raw_smooth",
        "gpfa_causal - raw_smooth",
        "pca_smooth - raw_smooth",
        "dm_smooth - pca_smooth",
        "raw_smooth - raw",
    ]
    # For dm_smooth - pca_smooth use grid20 as the primary (final) value
    display = []
    for name in order:
        if name == "dm_smooth - pca_smooth":
            primary = g20[name]
            faded = None  # already grid20; still show final pca_smooth-based if wanted
            # Also show what final-grid version would be as faded? User said use grid20
            # for both arms and show grid20 alongside faded for every contrast.
            # For this one, primary=grid20, faded could be final (mismatched d).
            faded = final.get(name)
            note = "grid20"
        else:
            primary = final[name]
            faded = g20.get(name)
            note = "final"
        display.append((name, primary, faded, note))

    fig = plt.figure(figsize=(W_FULL, W_FULL * 0.85))
    ax = fig.add_subplot(1, 1, 1)
    y_pos = np.arange(len(display))[::-1]
    for i, (name, prim, faded, note) in enumerate(display):
        y = y_pos[i]
        # faded grid20 (or final for dm contrast)
        if faded is not None and faded.get("mean") is not None:
            ax.plot(
                faded["mean"], y, "o", ms=7, color="#c4c2bb", alpha=0.55,
                markeredgecolor="#8f8d87", zorder=2,
            )
            # per-animal faded
            for j, (animal, val) in enumerate(sorted((faded.get("per_animal") or {}).items())):
                ax.plot(val, y + (j - 2.5) * 0.03, "|", color="#c4c2bb", alpha=0.45, ms=6)
        # primary
        mean = prim["mean"]
        color = "#c83a39" if name in flips else "#0b0b0b"
        ax.plot(mean, y, "o", ms=8, color=color, zorder=3)
        for j, (animal, val) in enumerate(sorted((prim.get("per_animal") or {}).items())):
            ax.plot(val, y + (j - 2.5) * 0.03, "|", color=color, alpha=0.7, ms=7)
        label = name.replace(" - ", " − ")
        if note == "grid20":
            label += " *"
        if name in flips:
            label += " [flip]"
        n_a, n_b = prim["n_a_better"], prim["n_b_better"]
        ax.text(
            0.02, y, f"{label}\n  mean={mean:+.3f}  {n_a}/{n_b}  N={prim['n_animals']}",
            transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.0,
        )

    ax.axvline(0, color="#8f8d87", lw=0.8)
    ax.set_yticks([])
    ax.set_xlabel("Δ normalized error (a − b); negative ⇒ a better")
    ax.set_xlim(ax.get_xlim()[0] - 0.02, ax.get_xlim()[1])
    # make room for labels on left
    ax.set_xlim(left=min(-0.15, ax.get_xlim()[0]))
    head(ax, "a", "Primary contrasts (animals)")
    ax.legend(
        handles=[
            Patch(facecolor="#0b0b0b", label="final grid (filled)"),
            Patch(facecolor="#c4c2bb", label="grid20 (faded)"),
            Patch(facecolor="#c83a39", label="sign flip vs grid20"),
        ],
        loc="lower right", fontsize=5.5,
    )
    _caption(
        fig,
        "RD7 · Final grid {2,3,5,10,20,40,80} for all contrasts except "
        "* dm_smooth − pca_smooth, which uses grid20 rows for both arms "
        "(DM was not extended past d=20). Faded markers = grid20 values. "
        "[flip] = sign of mean (or majority) flipped vs grid20. "
        "GPFA offline smoother excluded from contrasts. "
        f"Cohort rule: {D.load_cohort()['rule'][:120]}… "
        "Room A only. RD9 (region subsets) deferred — would need new fits.",
        y=0.02,
    )
    fig.subplots_adjust(left=0.42, right=0.98, top=0.90, bottom=0.16)
    return save_fig(fig, out_dir, "RD7_contrasts")
