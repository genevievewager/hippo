#!/usr/bin/env python3
"""Report 4a rebuild figures — Report 1/2 style (review cut: headline, Fig3–5).

Imports style helpers from agents.quadrant_n5.figures.figures (no fork).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.colors import Normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.quadrant_n5.figures.figures import (  # noqa: E402
    FAILC,
    GRID,
    INK,
    INK2,
    MM,
    MUTED,
    W_FULL,
    head,
    head_fig,
    save as _save,
)

FAM_COL = {
    "units": "#2a78d6",
    "sorting": "#eb6834",
    "spatial": "#1baf7a",
    "rhythm": "#7b5ea7",
    "population": "#0e7a54",
    "behaviour": "#c24e1e",
    "decoding": "#52514e",
    "information": "#8b3a4a",
    "other": "#a3a19b",
}
ARM_MRK = {"default": "s", "matched": "o"}
ARM_COL = {"default": "#6e6c66", "matched": "#0b0b0b"}


def save(fig, out, name):
    """Overflow check (Report 2 style) then save."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pad = 0.08
    offenders = []
    for i, ax in enumerate(fig.axes):
        try:
            bb = ax.get_tightbbox(renderer)
        except Exception:
            continue
        if bb is None:
            continue
        bb_fig = bb.transformed(fig.transFigure.inverted())
        if bb_fig.x0 < -pad or bb_fig.y0 < -pad or bb_fig.x1 > 1 + pad or bb_fig.y1 > 1 + pad:
            offenders.append(f"ax{i}")
    if offenders:
        # warn but still save for review; audit will catch
        print(json.dumps({"overflow_warn": name, "axes": offenders[:6]}), flush=True)
    _save(fig, out, name)


def _load(data: str) -> dict:
    D = {"meta": json.load(open(os.path.join(data, "data_meta.json")))}
    # Prefer Fig3 with information family if behaviour-info merge exists
    fig3_name = "data_fig3_divergence.csv"
    merged = os.path.join(
        os.path.dirname(data.rstrip("/")),
        "behavior_info",
        "data_fig3_divergence_with_info.csv",
    )
    if os.path.isfile(merged):
        D["fig3"] = pd.read_csv(merged)
        fig3_name = None
    for name, key in (
        ("data_fig3_divergence.csv", "fig3"),
        ("data_sim_gets_right.csv", "right"),
        ("data_session_metrics.csv", "sess"),
    ):
        if key == "fig3" and fig3_name is None:
            continue
        p = os.path.join(data, name)
        if os.path.isfile(p):
            D[key] = pd.read_csv(p)
    for name, key in (
        ("data_rate_maps.json", "rate_maps"),
        ("data_trajectories_60s.json", "traj"),
        ("data_isi_pool.json", "isi"),
        ("data_occupancy.json", "occ"),
        ("data_speed_turning_example.json", "spd"),
    ):
        p = os.path.join(data, name)
        if os.path.isfile(p):
            raw = json.load(open(p))
            if key == "rate_maps":
                for r in raw:
                    r["rate"] = np.asarray(r["rate"], float)
            D[key] = raw
    return D


def _caption(fig, text, y=0.012):
    fig.text(0.01, y, textwrap.fill(text, 145), fontsize=5.6, color=INK2, va="bottom", ha="left")


def _n_note(meta):
    return (
        f"n = {meta['n_animals']} animals, {meta['n_lab_sessions']} lab sessions; "
        f"{meta['n_matched_sessions']} matched sims; {meta['n_default_seeds']} default seeds"
    )


# ================================================================= FIG headline (for story PDF assembly uses reportlab; this is optional panel)
def fig_headline(D, out):
    """Single headline strip: refractory absolute + R overview thumbnail."""
    meta = D["meta"]
    fig3 = D["fig3"]
    fig = plt.figure(figsize=(W_FULL, 52 * MM))
    # Tight top: no blank gap above panel
    ax = fig.add_axes([0.34, 0.28, 0.38, 0.62])
    fig.text(0.01, 0.955, "a", fontsize=9, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.955, "Headline divergence (matched arm)", fontsize=7, color=INK, va="top")
    sub = fig3.dropna(subset=["R_matched"]).sort_values("R_matched", ascending=True).tail(8)
    y = np.arange(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        c = FAM_COL.get(r["family"], MUTED)
        ax.plot([r["R_matched_lo"], r["R_matched_hi"]], [i, i], color=c, lw=1.3)
        ax.plot(r["R_matched"], i, "o", color=c, ms=4.5)
    ax.axvline(1, color=FAILC, ls="--", lw=0.8)
    ax.set_yticks(y)
    # Full labels — wrap rather than truncate
    labels = []
    for l in sub["label"]:
        s = str(l)
        if len(s) > 42:
            s = s[:40] + "…"
        labels.append(s)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Divergence ratio R (log scale)", fontsize=6.5)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlim(0.5, max(30, float(sub["R_matched_hi"].max()) * 1.1))

    ax2 = fig.add_axes([0.78, 0.28, 0.20, 0.62])
    fig.text(0.78, 0.955, "b", fontsize=9, fontweight="bold", color=INK, va="top")
    fig.text(0.805, 0.955, "Refractory (absolute)", fontsize=7, color=INK, va="top")
    labs = ["lab", "matched\nsim", "default\nsim"]
    vals = [
        meta["refractory_pct"]["lab"],
        meta["refractory_pct"]["sim_matched"],
        meta["refractory_pct"]["sim_default"],
    ]
    ax2.bar([0, 1, 2], vals, color=[INK2, INK, MUTED], width=0.65)
    ax2.set_xticks([0, 1, 2], labs, fontsize=6)
    ax2.set_ylabel("% of ISIs < 1.5 ms", fontsize=6)
    for i, v in enumerate(vals):
        ax2.text(i, v + 0.05, f"{v:.2f}%", ha="center", fontsize=6)
    _caption(
        fig,
        f"{_n_note(meta)}. Dots = R; lines = 90% recording_day bootstrap. Bars = session-median refractory rate.",
        y=0.04,
    )
    save(fig, out, "Fig0_headline")


# ================================================================= FIG 3
def fig3(D, out):
    meta = D["meta"]
    df = D["fig3"].dropna(subset=["R_matched"]).copy()
    # order by family then R
    fam_order = [
        "units", "spatial", "behaviour", "information", "population",
        "decoding", "rhythm", "sorting", "other",
    ]
    df["_fo"] = df["family"].map({f: i for i, f in enumerate(fam_order)}).fillna(99)
    df = df.sort_values(["_fo", "R_matched"], ascending=[True, True]).reset_index(drop=True)

    fig = plt.figure(figsize=(W_FULL, 175 * MM))
    fig.suptitle(
        "Fig 3  Answer: where the simulation diverges from the lab (both arms)",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")

    ax = fig.add_axes([0.34, 0.10, 0.40, 0.84])
    y = np.arange(len(df))
    # Shade R ≤ 1 (sim within lab variability)
    lo_lim = 0.15
    ax.axvspan(lo_lim, 1.0, color="#eef6f1", lw=0, zorder=0)
    ax.axvline(1.0, color=FAILC, lw=0.9, ls="--", zorder=1)
    for i, r in df.iterrows():
        c = FAM_COL.get(r["family"], MUTED)
        lead = bool(r.get("lead_with_abs", False)) if "lead_with_abs" in df.columns else False
        if np.isfinite(r["R_matched"]):
            ax.plot([r["R_matched_lo"], r["R_matched_hi"]], [i, i], color=c, lw=1.4, zorder=2,
                    alpha=0.35 if lead else 1.0)
            ax.plot(r["R_matched"], i, "o", color=c, ms=5, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.3,
                    alpha=0.35 if lead else 1.0)
        # default arm: point only (5 seeds; no recording_day interval)
        if np.isfinite(r.get("R_default", np.nan)):
            ax.plot(r["R_default"], i, "s", color=c, ms=4, alpha=0.25 if lead else 0.55, zorder=3,
                    markeredgecolor=INK, markeredgewidth=0.4)
    ax.set_yticks(y)
    ylabels = []
    for _, r in df.iterrows():
        lab = str(r["label"])
        if "lead_with_abs" in df.columns and bool(r.get("lead_with_abs", False)):
            lab = lab + " *"
        ylabels.append(lab)
    ax.set_yticklabels(ylabels, fontsize=5.5)
    ax.set_xlabel("Divergence ratio R (log scale)")
    ax.set_xscale("log")
    lo = float(np.nanmin(np.r_[df["R_matched_lo"].to_numpy(float), df["R_default"].to_numpy(float)]))
    hi = float(np.nanmax(np.r_[df["R_matched_hi"].to_numpy(float), df["R_default"].to_numpy(float)]))
    ax.set_xlim(max(lo_lim, lo * 0.8), max(40, hi * 1.15))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.grid(axis="x", color=GRID, lw=0.4)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, ms=6, label="matched arm (±90% day bootstrap)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=MUTED, ms=5, label="default arm (5 seeds, no interval)"),
        Line2D([0], [0], color=FAILC, ls="--", lw=0.9, label="R = 1"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=5.4, frameon=False)

    # absolute side column
    ax2 = fig.add_axes([0.78, 0.08, 0.20, 0.86])
    ax2.set_ylim(-0.5, len(df) - 0.5)
    ax2.set_xlim(0, 1)
    ax2.axis("off")
    ax2.text(0.0, len(df) - 0.15, "lab  |  matched", fontsize=5.8, fontweight="bold", color=INK2)
    for i, r in df.iterrows():
        lab = r["lab_median"]
        sim = r["sim_matched_median"]
        # format by magnitude
        def fmt(v):
            if not np.isfinite(v):
                return "—"
            av = abs(v)
            if av >= 10:
                return f"{v:.1f}"
            if av >= 0.1:
                return f"{v:.2f}"
            return f"{v:.3f}"
        ax2.text(0.0, i, f"{fmt(lab)}  |  {fmt(sim)}", fontsize=5.5, va="center",
                 family="monospace", color=INK)

    # family colour key
    fams = [f for f in fam_order if f in set(df["family"])]
    for j, f in enumerate(fams):
        fig.text(0.01, 0.90 - j * 0.035, "█ " + f, fontsize=5.8, color=FAM_COL[f], family="sans-serif")

    _caption(
        fig,
        "One metric per concept (N = 20 where matched-N applies; W = 0.25 s for Fano). "
        "Circles = matched arm R with 90% recording_day bootstrap intervals; squares = default-arm R "
        "(5 seeds, no interval). Green shade = R ≤ 1 (sim within lab variability). "
        "Side column: absolute session-median values (lab | matched sim). "
        "* = lab MAD tiny: lead with absolute values; R is a footnote (information family, N = 20). "
        + _n_note(meta) + ".",
        y=0.008,
    )
    save(fig, out, "Fig3_answer")


# ================================================================= FIG 4
def _panel_head(fig, x, y, letter, title, title_dx=0.025):
    """Letter separated from title (no collision)."""
    fig.text(x, y, letter, fontsize=9, fontweight="bold", color=INK, va="top", ha="left")
    fig.text(x + title_dx, y, title, fontsize=6.2, color=INK, va="top", ha="left")


def fig4(D, out):
    meta = D["meta"]
    RM = D.get("rate_maps") or []
    isi = D.get("isi") or {}
    sess = D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 185 * MM))
    fig.suptitle(
        "Fig 4  Sim over-represents border / speed / HD cells; maps in Hz; grid at chance",
        fontsize=7.6, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, f"example {meta['example_session']}", fontsize=5.8, color=MUTED, ha="right", va="top")

    _panel_head(
        fig, 0.01, 0.968, "a",
        "Eligible-unit rate maps (20×20 Hz; SI p25/p50/p75 in MEC and SUB); grey = unvisited",
    )
    for row, ds, ylab in ((0, "lab", "lab"), (1, "sim_matched", "matched\nsim")):
        maps = [r for r in RM if r["dataset"] == ds]
        ordered = []
        for reg in ("MEC", "SUB"):
            sub = sorted([r for r in maps if r["region"] == reg], key=lambda r: r.get("percentile", 50))
            ordered.extend(sub[:3])
        for j, r in enumerate(ordered):
            ax = fig.add_axes([0.07 + j * 0.15, 0.72 - row * 0.22, 0.135, 0.17])
            rate = np.ma.masked_invalid(np.asarray(r["rate"], float))
            ext = r["extent"]
            ax.set_facecolor("#d8d6d0")
            ax.imshow(
                rate.T, origin="lower",
                extent=[ext[0], ext[1], ext[2], ext[3]],
                cmap="viridis", vmin=0,
                vmax=float(np.nanmax(r["rate"])) if np.isfinite(r["rate"]).any() else 1.0,
                aspect="equal", interpolation="nearest",
            )
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#6e6c66"); sp.set_linewidth(0.4)
            ax.set_title(
                f"{r['region']} p{int(r['percentile'])}\n"
                f"SI {r['spat_info']:.2f} · {r['peak_hz']:.1f} Hz",
                fontsize=5.0, pad=1.5, linespacing=1.05,
            )
            if j == 0:
                ax.set_ylabel(ylab, fontsize=6)
    rm = meta.get("rate_map") or {}
    fig.text(
        0.07, 0.48,
        f"20×20 train bbox; rate = spikes/(occ×dt) Hz; min occ {rm.get('min_occupancy_s', 0.15)} s; "
        f"σ = {rm.get('smooth_sigma_bins', 1.5)} bins; speed > {rm.get('speed_min_cm_s', 2)} cm/s; "
        f"eligible ≥{rm.get('min_train_spikes', 100)} spikes & ≥{rm.get('min_mean_rate_hz', 0.2)} Hz. "
        "Each panel scaled to its own peak.",
        fontsize=5.2, color=INK2,
    )

    # b — ISI
    ax = fig.add_axes([0.08, 0.10, 0.36, 0.32])
    _panel_head(fig, 0.08, 0.445, "b", "ISI (log time); refractory < 1.5 ms shaded")
    bins = np.logspace(-4, 0, 60)
    for key, col, lw, lab in (
        ("lab", INK2, 1.0, "lab"),
        ("sim_gt", "#2a78d6", 1.4, "sim GT"),
        ("sim_sorted", INK, 1.6, "sim sorted"),
    ):
        v = np.asarray(isi.get(key, []), float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size == 0:
            continue
        h, e = np.histogram(v, bins=bins, density=True)
        ax.plot(0.5 * (e[:-1] + e[1:]), h, color=col, lw=lw, label=lab)
    ax.axvspan(1e-4, isi.get("refractory_s", 0.0015), color="#fbe9e9", lw=0, zorder=0)
    ax.axvline(isi.get("refractory_s", 0.0015), color=FAILC, ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("ISI (s)")
    ax.set_ylabel("density per log-s bin")
    ax.legend(fontsize=5.5, loc="upper right", frameon=False)

    # c–e
    panels = [
        ("c", "Place-tuned frac", "frac_place_tuned", "frac. pass", 0.50, True),
        ("d", "Skaggs sparsity", "sparsity_median", "sparsity", 0.68, False),
        ("e", "Field size (place-tuned)", "place_field_size_median", "field / visited", 0.86, False),
    ]
    for k, (letter, title, metric, xlab, xpos, chance) in enumerate(panels):
        ax = fig.add_axes([xpos, 0.10, 0.125, 0.32])
        _panel_head(fig, xpos, 0.445, letter, title, title_dx=0.02)
        if sess is None:
            continue
        use = metric
        for ds, col, lw in (("lab", INK2, 0.7), ("sim_matched", INK, 1.6)):
            vals = sess[(sess.dataset == ds) & (sess.metric == use)]["value"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            s = np.sort(vals)
            ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=col, lw=lw)
        def_vals = sess[(sess.dataset == "sim_default") & (sess.metric == use)]["value"].to_numpy(float)
        def_vals = def_vals[np.isfinite(def_vals)]
        if def_vals.size:
            s = np.sort(def_vals)
            ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), "s", color=MUTED, ms=3.5, zorder=3)
        if chance:
            ax.axvline(0.05, color=FAILC, ls="--", lw=0.8, label="chance 0.05")
            ax.legend(fontsize=4.8, loc="lower right", frameon=False)
        ax.set_ylabel("session CDF" if k == 0 else "")
        ax.set_xlabel(xlab, fontsize=5.5)
        ax.set_ylim(0, 1)

    _caption(
        fig,
        "a: Hz rate maps (spikes / occupancy seconds); eligible SI percentiles; unvisited grey. "
        "b: ISI density on log-time bins. c: place-tuned = SI > shuffle 95% (dashed = chance 0.05). "
        "d: Skaggs sparsity (Σpr)²/Σpr². e: field size among place-tuned only "
        "(≥20% peak & ≥1 Hz, ≥4 contiguous bins). Thin lab, thick matched, squares = 5 default seeds. "
        + _n_note(meta) + ".",
        y=0.006,
    )
    save(fig, out, "Fig4_mechanism")


# ================================================================= FIG 5
def fig5(D, out):
    meta = D["meta"]
    traj = D.get("traj") or {}
    spd = D.get("spd") or {}
    occ = D.get("occ") or {}
    LAB_COL = "#2a78d6"
    SIM_COL = "#c24e1e"
    fig = plt.figure(figsize=(W_FULL, 155 * MM))
    fig.suptitle(
        "Fig 5  Simulated rat runs faster and turns less than lab rats",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, f"example {meta['example_session']} · 60 s", fontsize=5.8, color=MUTED, ha="right", va="top")

    def _outline(ax, ar):
        pts = ar.get("outline") if ar else None
        if pts:
            pts = np.asarray(pts, float)
            ax.plot(pts[:, 0], pts[:, 1], color="#8a8780", lw=0.8, zorder=0)

    for j, (key, title) in enumerate((("lab", "Lab"), ("sim_matched", "Matched sim"))):
        ax = fig.add_axes([0.06 + j * 0.24, 0.55, 0.20, 0.38])
        _panel_head(fig, 0.06 + j * 0.24, 0.955, "a" if j == 0 else "b", f"{title} trajectory (60 s)", 0.02)
        t = traj.get(key) or {}
        x, y = np.asarray(t.get("x", []), float), np.asarray(t.get("y", []), float)
        ar = (traj.get("arena") or {}).get("lab" if key == "lab" else "sim", {})
        _outline(ax, ar)
        if len(x):
            ax.plot(x, y, color=INK, lw=0.6, zorder=2)
            ax.plot(x[0], y[0], "o", color="#1baf7a", ms=4, zorder=3)
            ax.plot(x[-1], y[-1], "s", color=FAILC, ms=4, zorder=3)
        if ar:
            ax.set_xlim(ar["xlim"]); ax.set_ylim(ar["ylim"])
        ax.set_aspect("equal")
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)" if j == 0 else "")

    # Occupancy: colourbars between maps and CDFs with clear gap
    vmax = 0.0
    mats = {}
    for key in ("lab", "sim_matched"):
        o = occ.get(key) or {}
        if o:
            mats[key] = np.asarray(o["occ_s"], float)
            vmax = max(vmax, float(np.nanmax(mats[key])) if mats[key].size else 0.0)
    for j, key in enumerate(("lab", "sim_matched")):
        ax = fig.add_axes([0.06 + j * 0.24, 0.10, 0.18, 0.34])
        _panel_head(fig, 0.06 + j * 0.24, 0.475, "c" if j == 0 else "d",
                    f"Occupancy ({'lab' if j == 0 else 'matched'})", 0.02)
        o = occ.get(key) or {}
        ar = (occ.get("arena") or traj.get("arena") or {}).get("lab" if key == "lab" else "sim", {})
        if o and key in mats:
            mat = mats[key]
            ext = o["extent"]
            im = ax.imshow(
                mat.T, origin="lower", extent=[ext[0], ext[1], ext[2], ext[3]],
                cmap="magma", aspect="equal", interpolation="nearest",
                vmin=0, vmax=vmax if vmax > 0 else None,
            )
            _outline(ax, ar)
            if ar:
                ax.set_xlim(ar["xlim"]); ax.set_ylim(ar["ylim"])
            ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)" if j == 0 else "")
            # colourbar immediately right of map, left of CDF column
            cax = fig.add_axes([0.06 + j * 0.24 + 0.185, 0.10, 0.012, 0.34])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("s", fontsize=5.2)
            cb.ax.tick_params(labelsize=5)

    qs = np.asarray((spd.get("cohort") or {}).get("qs") or np.linspace(0, 1, 41), float)

    def _plot_cohort(ax, field, xlabel, letter, title, x0):
        _panel_head(fig, x0, 0.955 if letter == "e" else 0.475, letter, title, 0.02)
        cohort = spd.get("cohort") or {}
        for key, col, thin_a in (("lab", LAB_COL, 0.30), ("sim_matched", SIM_COL, 0.30)):
            rows = cohort.get(key) or []
            for row in rows:
                q = np.asarray(row.get(field, []), float)
                if q.size != qs.size:
                    continue
                if row.get("example"):
                    ax.plot(q, qs, color=col, lw=2.0, alpha=1.0, zorder=4,
                            label=f"example {key.replace('sim_matched','sim')}")
                else:
                    ax.plot(q, qs, color=col, lw=0.45, alpha=thin_a, zorder=1)
        if not (cohort.get("lab") or cohort.get("sim_matched")):
            for key, col in (("lab", LAB_COL), ("sim_matched", SIM_COL)):
                raw_key = "speed_cm_s" if field == "speed_q" else "turning_rad"
                v = np.asarray((spd.get(key) or {}).get(raw_key, []), float)
                if field == "turn_q":
                    v = np.abs(v)
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                s = np.sort(v)
                ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=col, lw=1.6,
                        label=f"example {key.replace('sim_matched','sim')}")
        ax.set_xlabel(xlabel, fontsize=5.8)
        ax.set_ylabel("CDF")
        ax.set_ylim(0, 1)
        handles = [
            Line2D([0], [0], color=LAB_COL, lw=0.6, label="lab sessions"),
            Line2D([0], [0], color=SIM_COL, lw=0.6, label="matched sessions"),
            Line2D([0], [0], color=LAB_COL, lw=2.0, label="example lab"),
            Line2D([0], [0], color=SIM_COL, lw=2.0, label="example sim"),
        ]
        ax.legend(handles=handles, fontsize=4.8, loc="lower right", frameon=False)

    ax = fig.add_axes([0.68, 0.55, 0.28, 0.38])
    _plot_cohort(ax, "speed_q", "speed (cm/s)", "e", "Speed CDF (per session)", 0.68)

    ax = fig.add_axes([0.68, 0.10, 0.28, 0.34])
    _plot_cohort(ax, "turn_q", "|Δ heading| while speed > 2 cm/s (rad)", "f", "Turning CDF (moving only)", 0.68)

    _caption(
        fig,
        "a–b: 60 s paths (green start, red end); arena outline; shared cm span. "
        "c–d: occupancy (s/bin); colourbars clear of CDFs. "
        "Note (c): some occupied bins fall outside the boundary polygon (path containment; "
        "to be handled in the report2-v1 step). "
        "e–f: blue = lab, orange = matched; thick = example session. "
        f"thigmotaxis = {meta['thigmotaxis']} (not fitted). {traj.get('selection', '')}",
        y=0.006,
    )
    save(fig, out, "Fig5_behaviour")


# ================================================================= FIG 1
def fig1(D, out):
    """Design: three arms, R definition, frozen knobs."""
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 120 * MM))
    fig.suptitle(
        "Fig 1  Design: lab vs default vs matched simulation arms",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")
    ax = fig.add_axes([0.04, 0.18, 0.92, 0.72])
    ax.set_axis_off()
    boxes = [
        (0.05, 0.55, 0.25, 0.35, "Lab\nroom-A sessions\nn=47 · 6 animals", "#e8f1fb"),
        (0.38, 0.55, 0.25, 0.35, "Default sim\n5 seeds · square arena\nthigmotaxis=0.8 (held)", "#f3f1ea"),
        (0.71, 0.55, 0.25, 0.35, "Matched sim\n1:1 per lab session\nyield-boosted MEC/SUB", "#eaf7f1"),
    ]
    for x, y, w, h, txt, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                    facecolor=fc, edgecolor=INK, lw=0.8, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=7,
                transform=ax.transAxes, color=INK)
    ax.annotate("", xy=(0.38, 0.72), xytext=(0.30, 0.72),
                arrowprops=dict(arrowstyle="->", color=INK2), transform=ax.transAxes)
    ax.annotate("", xy=(0.71, 0.72), xytext=(0.63, 0.72),
                arrowprops=dict(arrowstyle="->", color=INK2), transform=ax.transAxes)
    ax.text(0.5, 0.35, meta.get("R_definition", "R = d_sim / d_lab"),
            ha="center", va="center", fontsize=7, transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=GRID))
    ax.text(
        0.5, 0.12,
        "Same metric code path for lab and sim · dataset is a label only · "
        f"spatial rev {meta.get('spatial_revision', '4a_B')} · "
        "occ-matched joint bins dropped (coordinate frames differ)",
        ha="center", va="center", fontsize=6, color=INK2, transform=ax.transAxes,
        wrap=True,
    )
    _caption(
        fig,
        "Default arm: 5 independent seeds (point estimates, no intervals). "
        "Matched arm: recording-day bootstrap 90% intervals. "
        f"Thigmotaxis held at {meta['thigmotaxis']} (not fitted).",
        y=0.04,
    )
    save(fig, out, "Fig1_design")


# ================================================================= FIG 2
def fig2(D, out):
    """Validity: neutrality + path-containment Q3 table."""
    meta = D["meta"]
    q3 = meta.get("q3") or {}
    neu = meta.get("boost_neutrality") or {}
    fig = plt.figure(figsize=(W_FULL, 145 * MM))
    fig.suptitle(
        "Fig 2  Validity: neutrality gate and path-containment QC (Report 2 semantics)",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")

    ax = fig.add_axes([0.06, 0.58, 0.40, 0.32])
    status = str((neu.get("status") if isinstance(neu, dict) else None) or "see SUMMARY")
    ax.set_axis_off()
    ax.set_title("a  Neutrality / boost gate", loc="left", fontsize=7, fontweight="bold")
    ax.text(0.05, 0.7, f"Status: {status}", fontsize=8, transform=ax.transAxes, color=INK)
    ax.text(
        0.05, 0.35,
        "Matched-arm yield boost targets lab MEC/SUB counts.\n"
        "Discriminator nulls must sit near chance (group shuffle).",
        fontsize=6.5, color=INK2, transform=ax.transAxes, va="top",
    )

    ax2 = fig.add_axes([0.55, 0.58, 0.40, 0.30])
    ax2.set_title("b  Path containment (≥0.95), flag only", loc="left", fontsize=7, fontweight="bold")
    per = (q3.get("per_animal") or {})
    animals = sorted(per.keys())
    if animals:
        tot = [per[a]["total"] for a in animals]
        elig = [per[a]["eligible"] for a in animals]
        excl = [per[a]["excluded"] for a in animals]
        x = np.arange(len(animals))
        ax2.bar(x - 0.2, tot, 0.2, color=MUTED, label="total")
        ax2.bar(x, elig, 0.2, color="#2a78d6", label="eligible")
        ax2.bar(x + 0.2, excl, 0.2, color=FAILC, label="excluded")
        ax2.set_xticks(x)
        ax2.set_xticklabels(animals, fontsize=6)
        ax2.legend(fontsize=4.8, frameon=False, loc="upper right", borderaxespad=0.2)
        ax2.set_ylabel("sessions", fontsize=6)
        ax2.set_ylim(0, max(tot) * 1.25 if tot else 1)
    else:
        ax2.set_axis_off()
        ax2.text(0.5, 0.5, "Q3 table missing", ha="center", transform=ax2.transAxes)

    ax3 = fig.add_axes([0.08, 0.12, 0.84, 0.35])
    ax3.set_axis_off()
    ax3.set_title("c  Per-animal hard exclusions (session_select) · path failures stay in scores",
                  loc="left", fontsize=7, fontweight="bold")
    lines = ["animal  total  elig  excl  reasons  path_fail_elig"]
    for a in animals:
        v = per[a]
        lines.append(
            f"{a:6}  {v['total']:5}  {v['eligible']:4}  {v['excluded']:4}  "
            f"{v.get('exclude_reasons', {})}  {v.get('n_path_fail_eligible', 0)}"
        )
    fails = q3.get("path_fail_matched") or []
    lines.append("")
    lines.append(
        f"Matched set: eligible={q3.get('n_eligible')} · added={q3.get('added_vs_matched')} · "
        f"hard dropouts={q3.get('drop_hard_vs_matched')} · path_fail kept={len(fails)}"
    )
    if fails:
        lines.append(
            "Path-fail (flagged; path-visited mask): "
            + ", ".join(f"{r['session']}={float(r['path_containment_frac']):.3f}" for r in fails[:8])
        )
    ax3.text(0.0, 0.98, "\n".join(lines), fontsize=5.5, family="monospace",
             va="top", ha="left", transform=ax3.transAxes, color=INK)
    _caption(
        fig,
        "Hard exclude: min_units, no_room_A, empty_retained, min_valid_frac, ExtentError, "
        "ratemap_stability. Path containment ≥0.95 is flagged only (Report 2 extract_tidy); "
        "failures remain in scores with a path-visited bin mask for spatial metrics.",
        y=0.02,
    )
    save(fig, out, "Fig2_validity")


# ================================================================= FIG 6
def fig6(D, out):
    """Second answer panel: place-tuned + what sim gets right + top R."""
    meta = D["meta"]
    fig3 = D.get("fig3")
    right = D.get("right")
    sess = D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 140 * MM))
    fig.suptitle(
        "Fig 6  Place tuning and within-lab matches after Skaggs sparsity / field redefinition",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")

    ax = fig.add_axes([0.08, 0.55, 0.38, 0.35])
    ax.set_title("a  Place-tuned fraction (chance 0.05)", loc="left", fontsize=7, fontweight="bold")
    if sess is not None and "frac_place_tuned" in set(sess.metric):
        for ds, col, lw in (("lab", INK2, 0.8), ("sim_matched", INK, 1.6)):
            vals = sess[(sess.dataset == ds) & (sess.metric == "frac_place_tuned")]["value"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                s = np.sort(vals)
                ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=col, lw=lw, label=ds)
        ax.axvline(0.05, color=FAILC, ls="--", lw=0.8, label="chance")
        ax.legend(fontsize=5, frameon=False)
    ax.set_xlabel("frac place-tuned", fontsize=6)
    ax.set_ylabel("session CDF", fontsize=6)
    ax.set_ylim(0, 1)

    axb = fig.add_axes([0.55, 0.55, 0.40, 0.35])
    axb.set_title("b  Field size among place-tuned units", loc="left", fontsize=7, fontweight="bold")
    if sess is not None and "place_field_size_median" in set(sess.metric):
        for ds, col, lw in (("lab", INK2, 0.8), ("sim_matched", INK, 1.6)):
            vals = sess[(sess.dataset == ds) & (sess.metric == "place_field_size_median")]["value"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                s = np.sort(vals)
                axb.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=col, lw=lw, label=ds)
        axb.legend(fontsize=5, frameon=False)
    axb.set_xlabel("field / visited", fontsize=6)
    axb.set_ylim(0, 1)

    axc = fig.add_axes([0.08, 0.12, 0.84, 0.32])
    axc.set_title("c  Metrics with matched R ≤ 1 (within lab variability)", loc="left", fontsize=7, fontweight="bold")
    axc.set_axis_off()
    if right is not None and len(right):
        bits = [
            f"{r.label}: R={r.R_matched:.2f} (lab {r.lab_median:.3g} | matched {r.sim_matched_median:.3g})"
            for _, r in right.head(12).iterrows()
            if np.isfinite(r.get("R_matched", np.nan))
        ]
        axc.text(0.0, 0.95, "\n".join(bits) if bits else "none", fontsize=6, va="top",
                 transform=axc.transAxes, color=INK, family="monospace")
    _caption(
        fig,
        "Skaggs sparsity (Σpr)²/Σpr²; field = contiguous bins ≥20% peak & ≥1 Hz, ≥4 bins, "
        "among SI>shuffle95 units. Occupancy-matched rows dropped (lab polygon vs sim square).",
        y=0.02,
    )
    save(fig, out, "Fig6_answer")


# ================================================================= FIG 7
def fig7(D, out):
    """Trajectory gallery: lab vs matched (reuses Fig5 traj assets)."""
    meta = D["meta"]
    traj = D.get("traj") or {}
    fig = plt.figure(figsize=(W_FULL, 130 * MM))
    fig.suptitle(
        "Fig 7  Example trajectories (60 s) — lab polygon vs matched square",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, f"example {meta.get('example_session', '')}", fontsize=5.8, color=MUTED, ha="right", va="top")
    panels = [("a", "lab", traj.get("lab")), ("b", "sim_matched", traj.get("sim_matched"))]
    for i, (letter, title, tr) in enumerate(panels):
        ax = fig.add_axes([0.08 + i * 0.46, 0.22, 0.40, 0.65])
        ax.set_title(f"{letter}  {title}", loc="left", fontsize=7, fontweight="bold")
        if not tr:
            ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes)
            continue
        if "x" in tr and "y" in tr:
            xy = np.column_stack([np.asarray(tr["x"], float), np.asarray(tr["y"], float)])
        else:
            xy = np.asarray(tr.get("xy") or tr.get("positions") or [], float)
        if xy.ndim == 2 and xy.shape[1] >= 2 and len(xy):
            ax.plot(xy[:, 0], xy[:, 1], color=INK if title != "lab" else "#2a78d6", lw=0.6)
            ax.plot(xy[0, 0], xy[0, 1], "o", color="#1baf7a", ms=4)
            ax.plot(xy[-1, 0], xy[-1, 1], "s", color=FAILC, ms=4)
        arena = (D.get("traj") or {}).get("arena") or {}
        side = arena.get("lab") if title == "lab" else arena.get("sim")
        if isinstance(side, dict):
            poly = np.asarray(side.get("outline") or [], float)
        else:
            poly = np.asarray(tr.get("boundary") or tr.get("outline") or [], float)
        if poly.ndim == 2 and len(poly) >= 3:
            ax.plot(poly[:, 0], poly[:, 1], color=MUTED, lw=0.8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (cm)", fontsize=6)
        ax.set_ylabel("y (cm)", fontsize=6)
    _caption(
        fig,
        "Green = start, red = end. Behaviour confound remains: lab room polygon vs sim square; "
        "occupancy-matched joint bins are not used. Path-containment failures use path-visited masks.",
        y=0.04,
    )
    save(fig, out, "Fig7_trajectories")


def _supp_metric_panel(ax, sess, metric, title, xlab, chance=False):
    ax.set_title(title, loc="left", fontsize=7, fontweight="bold")
    if sess is None or metric not in set(sess.metric):
        ax.text(0.5, 0.5, f"{metric} n/a", ha="center", transform=ax.transAxes, fontsize=6)
        return
    for ds, col, lw in (("lab", INK2, 0.7), ("sim_matched", INK, 1.5)):
        vals = sess[(sess.dataset == ds) & (sess.metric == metric)]["value"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            s = np.sort(vals)
            ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=col, lw=lw, label=ds)
    if chance:
        ax.axvline(0.05, color=FAILC, ls="--", lw=0.7)
    ax.set_xlabel(xlab, fontsize=5.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=4.8, frameon=False, loc="lower right")


def figS1(D, out):
    meta, sess = D["meta"], D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    fig.suptitle("Fig S1  Units: refractory and rate statistics", fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")
    specs = [
        (0.08, "refractory_violation_rate", "a  Refractory viol. (ISI<1.5 ms)", "frac"),
        (0.38, "firing_rate_hz", "b  Firing rate", "Hz"),
        (0.68, "isi_cv", "c  ISI CV", "CV"),
    ]
    for x, m, t, xlab in specs:
        ax = fig.add_axes([x, 0.18, 0.26, 0.68])
        _supp_metric_panel(ax, sess, m, t, xlab)
    _caption(fig, "Thin = lab, thick = matched. Generator has no public refractory / min-ISI parameter.", y=0.03)
    save(fig, out, "FigS1_units")


def figS2(D, out):
    meta, sess = D["meta"], D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    fig.suptitle("Fig S2  Sorting realism proxies", fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")
    specs = [
        (0.08, "presence_ratio_median", "a  Presence ratio", "frac"),
        (0.38, "refractory_contam_proxy_median", "b  Refractory contam. proxy", "value"),
        (0.68, "sorted_over_gt_spike_count_median", "c  Sorted / GT spike count", "ratio"),
    ]
    for x, m, t, xlab in specs:
        ax = fig.add_axes([x, 0.18, 0.26, 0.68])
        _supp_metric_panel(ax, sess, m, t, xlab)
    _caption(fig, "Sorting metrics use the matched-arm simulated sorter path; GT-only metrics are diagnostic.", y=0.03)
    save(fig, out, "FigS2_sorting")


def figS3(D, out):
    meta, sess = D["meta"], D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    fig.suptitle("Fig S3  Spatial coding (revised)", fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")
    specs = [
        (0.08, "sparsity_median", "a  Skaggs sparsity", "sparsity"),
        (0.38, "spatial_info_median", "b  Spatial info", "bits/spk"),
        (0.68, "peak_rate_median", "c  Peak rate", "Hz"),
    ]
    for x, m, t, xlab in specs:
        ax = fig.add_axes([x, 0.18, 0.26, 0.68])
        _supp_metric_panel(ax, sess, m, t, xlab)
    _caption(
        fig,
        meta.get("sparsity_formula", "Skaggs sparsity") + ". "
        + meta.get("place_field_definition", ""),
        y=0.03,
    )
    save(fig, out, "FigS3_spatial")


def figS4(D, out):
    meta, sess = D["meta"], D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    fig.suptitle("Fig S4  Functional cell-type fractions (shuffle 95%)", fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")
    specs = [
        (0.06, "frac_border_pass", "a  Border", True),
        (0.30, "frac_speed_pass", "b  Speed", True),
        (0.54, "frac_hd_pass", "c  HD", True),
        (0.78, "frac_grid_pass", "d  Grid", True),
    ]
    for x, m, t, ch in specs:
        ax = fig.add_axes([x, 0.18, 0.20, 0.68])
        _supp_metric_panel(ax, sess, m, t, "frac", chance=ch)
    _caption(fig, "Dashed line = 0.05 chance floor from 95th-percentile shuffle.", y=0.03)
    save(fig, out, "FigS4_tuning_fracs")


def figS5(D, out):
    meta, sess = D["meta"], D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    fig.suptitle("Fig S5  Population structure", fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")
    specs = [
        (0.08, "participation_ratio", "a  Participation ratio", "PR"),
        (0.38, "mean_pairwise_corr", "b  Pairwise corr", "r"),
        (0.68, "noise_corr_mean", "c  Noise corr", "r"),
    ]
    for x, m, t, xlab in specs:
        ax = fig.add_axes([x, 0.18, 0.26, 0.68])
        _supp_metric_panel(ax, sess, m, t, xlab)
    _caption(fig, "Population metrics from the shared metric suite (matched-N forms preferred when present).", y=0.03)
    save(fig, out, "FigS5_population")


def figS6(D, out):
    meta = D["meta"]
    disc = meta.get("discriminator") or {}
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    fig.suptitle("Fig S6  Discriminator and decoding", fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")
    ax = fig.add_axes([0.10, 0.25, 0.38, 0.55])
    arms = ["default", "matched"]
    aucs, nulls = [], []
    for a in arms:
        d = disc.get(a) or {}
        aucs.append(float(d.get("auc_logit") or np.nan))
        nulls.append(float(d.get("auc_group_shuffle_null_stratified") or d.get("auc_group_shuffle_null") or np.nan))
    x = np.arange(len(arms))
    ax.bar(x - 0.15, aucs, 0.3, color=INK, label="AUC logit")
    ax.bar(x + 0.15, nulls, 0.3, color=MUTED, label="null StratGroupKF")
    ax.axhline(0.5, color=FAILC, ls="--", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5, frameon=False)
    ax.set_title("a  Lab-vs-sim discriminator", loc="left", fontsize=7, fontweight="bold")

    ax2 = fig.add_axes([0.58, 0.25, 0.35, 0.55])
    dec = meta.get("decode_error_over_floor_median_N20") or {}
    ax2.bar([0, 1, 2], [dec.get("lab", np.nan), dec.get("sim_matched", np.nan), dec.get("sim_default", np.nan)],
            color=["#2a78d6", INK, MUTED])
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["lab", "matched", "default"], fontsize=6)
    ax2.set_ylabel("error / floor", fontsize=6)
    ax2.set_title("b  Decode error / chance floor (N=20)", loc="left", fontsize=7, fontweight="bold")
    _caption(fig, "Null shuffles labels at session/group level. Calibration deferred to Report 4b.", y=0.04)
    save(fig, out, "FigS6_disc_decode")


def figS7(D, out):
    meta = D["meta"]
    q3 = meta.get("q3") or {}
    fig = plt.figure(figsize=(W_FULL, 130 * MM))
    fig.suptitle("Fig S7  Q3 eligibility table and caveats", fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    ax = fig.add_axes([0.06, 0.12, 0.88, 0.78])
    ax.set_axis_off()
    lines = [
        "Report 2 alignment: checkout report2-v1 · path containment flag-only · hard excludes via session_select",
        f"Sparsity: {meta.get('sparsity_formula')}",
        f"Fields: {meta.get('place_field_definition')}",
        f"Occ-matched: {meta.get('occmatched')}",
        "",
        "Per-animal (total / eligible / excluded / reasons / path_fail_elig):",
    ]
    for a, v in sorted((q3.get("per_animal") or {}).items()):
        lines.append(
            f"  {a}: {v['total']} / {v['eligible']} / {v['excluded']} / "
            f"{v.get('exclude_reasons')} / {v.get('n_path_fail_eligible')}"
        )
    fails = q3.get("path_fail_matched") or []
    lines.append("")
    lines.append(f"Path-fail matched sessions kept with path mask ({len(fails)}):")
    for r in fails:
        lines.append(f"  {r['session']}: frac={float(r['path_containment_frac']):.4f}")
    lines.append("")
    lines.append(meta.get("sub_caveat", ""))
    lines.append(meta.get("calibration_note", ""))
    ax.text(0.0, 1.0, "\n".join(lines), fontsize=5.8, family="monospace", va="top",
            transform=ax.transAxes, color=INK)
    save(fig, out, "FigS7_q3_caveats")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = _load(args.data)
    # Prefer tidy session metrics (broader) when present alongside review extract
    tidy_sess = os.path.join(args.data, "data_session_metrics.csv")
    if os.path.isfile(tidy_sess):
        D["sess"] = pd.read_csv(tidy_sess)
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    jobs = [
        ("Fig0", lambda: fig_headline(D, args.out)),
        ("Fig1", lambda: fig1(D, args.out)),
        ("Fig2", lambda: fig2(D, args.out)),
        ("Fig3", lambda: fig3(D, args.out)),
        ("Fig4", lambda: fig4(D, args.out)),
        ("Fig5", lambda: fig5(D, args.out)),
        ("Fig6", lambda: fig6(D, args.out)),
        ("Fig7", lambda: fig7(D, args.out)),
        ("FigS1", lambda: figS1(D, args.out)),
        ("FigS2", lambda: figS2(D, args.out)),
        ("FigS3", lambda: figS3(D, args.out)),
        ("FigS4", lambda: figS4(D, args.out)),
        ("FigS5", lambda: figS5(D, args.out)),
        ("FigS6", lambda: figS6(D, args.out)),
        ("FigS7", lambda: figS7(D, args.out)),
    ]
    for name, fn in jobs:
        if only and name not in only:
            continue
        fn()
        print(json.dumps({"fig": name}), flush=True)


if __name__ == "__main__":
    main()
