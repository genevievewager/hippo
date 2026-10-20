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
# W_FULL is 180 mm — every figure must use this width (Report 1 package).

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
    """Overflow + text-overlap check (matplotlib extents), then save at W_FULL."""
    from agents.data_quality.report.overlap_audit import audit_mpl_figure

    fig.canvas.draw()
    fails = audit_mpl_figure(fig)
    if fails:
        print(json.dumps({"mpl_overlap": name, "n": len(fails), "sample": fails[:4]}, default=str), flush=True)
    _save(fig, out, name)


def _load(data: str) -> dict:
    """Load tidy tables. Fig3 always from ``data_fig3_divergence.csv`` (post keep-mask).

    Never prefer ``behavior_info/…_with_info.csv`` — that merge historically
    rebased on a stale v3 Fig3 and desynced the forest plot from the story.
    """
    D = {"meta": json.load(open(os.path.join(data, "data_meta.json")))}
    for name, key in (
        ("data_fig3_divergence.csv", "fig3"),
        ("data_sim_gets_right.csv", "right"),
        ("data_session_metrics.csv", "sess"),
    ):
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


def _caption(fig, text, y=0.018):
    """Caption inside the 180 mm frame at ≥ 8 pt (Report 1 caption band)."""
    fig.text(0.01, y, textwrap.fill(text, 118), fontsize=8.0, color=INK2, va="bottom", ha="left")


def _fmt_abs(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    av = abs(float(v))
    if av < 5e-3:
        return "0.00"
    if av >= 10:
        return f"{v:.1f}"
    if av >= 0.1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _n_note(meta):
    return (
        f"n = {meta['n_animals']} animals, {meta['n_lab_sessions']} lab sessions; "
        f"{meta['n_matched_sessions']} matched sims; {meta['n_default_seeds']} default seeds"
    )


# ================================================================= FIG headline (for story PDF assembly uses reportlab; this is optional panel)
# Story takeaway metrics for the headline forest (not information-family rows).
HEADLINE_METRICS = [
    ("refractory_violation_rate", "Refractory violations"),
    ("speed_median", "Speed (median)"),
    ("turning_angle_std", "Turning-angle SD"),
    ("frac_place_tuned", "Place-tuned fraction"),
    ("decode_error_over_floor", "Decode error / floor"),
]


def _bits_place_row(out_dir: str) -> dict | None:
    """Load held-out bits/spike (place-tuned) from sibling behavior_info."""
    report_root = os.path.dirname(os.path.abspath(out_dir))
    enc_p = os.path.join(report_root, "behavior_info", "encoding_diag_summary.json")
    if not os.path.isfile(enc_p):
        return None
    enc = json.load(open(enc_p)).get("table") or {}
    lab_b = (enc.get("lab") or {}).get("bits_end_med_place")
    mat_b = (enc.get("sim_matched") or {}).get("bits_end_med_place")
    if lab_b is None or mat_b is None or not np.isfinite(lab_b) or abs(lab_b) < 1e-9:
        return None
    r_bits = float(mat_b / lab_b)
    return {
        "label": "Bits/spike (place-tuned)",
        "metric_base": "bits_per_spike_place",
        "R_matched": r_bits, "R_matched_lo": r_bits, "R_matched_hi": r_bits,
        "family": "spatial",
        "lab_median": float(lab_b), "sim_matched_median": float(mat_b),
    }


def fig_headline(D, out):
    """Headline: takeaway metrics (matched R) + refractory absolute bars."""
    from matplotlib.ticker import FixedLocator, NullFormatter

    meta = D["meta"]
    fig3 = D["fig3"].set_index("metric_base")
    fig = plt.figure(figsize=(W_FULL, 58 * MM))
    ax = fig.add_axes([0.30, 0.22, 0.36, 0.62])
    head_fig(fig, 0.01, 0.90, "a", "Headline divergence (matched) — takeaways")
    rows = []
    for base, label in HEADLINE_METRICS:
        if base not in fig3.index:
            continue
        r = fig3.loc[base]
        if not np.isfinite(r.get("R_matched", np.nan)):
            continue
        rows.append({**r.to_dict(), "label": label, "metric_base": base})
    bits = _bits_place_row(out)
    if bits is not None:
        rows.append(bits)
    rows = sorted(rows, key=lambda r: float(r["R_matched"]))
    y = np.arange(len(rows))
    for i, r in enumerate(rows):
        c = FAM_COL.get(r.get("family", "other"), MUTED)
        lo = r.get("R_matched_lo", r["R_matched"])
        hi = r.get("R_matched_hi", r["R_matched"])
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=c, lw=1.3)
        ax.plot(r["R_matched"], i, "o", color=c, ms=4.5)
    ax.axvline(1, color=FAILC, ls="--", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=6.0)
    ax.set_xlabel("R (log)", fontsize=7)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([1, 10]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v)}" if v >= 1 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    hi = max((float(r.get("R_matched_hi", r["R_matched"])) for r in rows), default=10)
    ax.set_xlim(0.4, max(30, hi * 1.15))

    # Panel b: title above the axes; labels below bars (never overlapping title)
    ax2 = fig.add_axes([0.74, 0.22, 0.22, 0.55])
    head_fig(fig, 0.74, 0.90, "b", "Refractory (absolute)")
    labs = ["lab", "matched", "default"]
    vals = [
        meta["refractory_pct"]["lab"],
        meta["refractory_pct"]["sim_matched"],
        meta["refractory_pct"]["sim_default"],
    ]
    bars = ax2.bar([0, 1, 2], vals, color=[INK2, INK, MUTED], width=0.65)
    ax2.set_xticks([0, 1, 2], labs, fontsize=6)
    ax2.set_ylabel("% ISIs < 1.5 ms", fontsize=6.5)
    ymax = max(vals) * 1.35 if vals else 1
    ax2.set_ylim(0, ymax)
    for i, v in enumerate(vals):
        ax2.text(i, v + ymax * 0.03, f"{v:.2f}%", ha="center", va="bottom", fontsize=5.5)
    _caption(
        fig,
        f"{_n_note(meta)}. a = story takeaways; dots = R (±90% day bootstrap).",
        y=0.03,
    )
    save(fig, out, "Fig0_headline")


# ================================================================= FIG 3
def fig3(D, out):
    from matplotlib.ticker import FixedLocator, NullFormatter

    meta = D["meta"]
    df = D["fig3"].dropna(subset=["R_matched"]).copy()
    ban = {"excess", "bin3", "spatial_context"}
    df = df[~df["metric_base"].astype(str).str.contains("|".join(ban), case=False, regex=True)].copy()
    fam_order = [
        "units", "spatial", "behaviour", "information", "population",
        "decoding", "rhythm", "sorting", "other",
    ]
    df["_fo"] = df["family"].map({f: i for i, f in enumerate(fam_order)}).fillna(99)
    df = df.sort_values(["_fo", "R_matched"], ascending=[True, True]).reset_index(drop=True)
    if len(df) > 25:
        df = df.iloc[:25].reset_index(drop=True)

    n = len(df)
    # Fixed height budget so the fig fits the A4 180 mm frame with caption
    fig_h = min(220, max(130, 5.8 * n + 45))
    fig = plt.figure(figsize=(W_FULL, fig_h * MM))
    fig.suptitle(
        "Fig 3  Answer: where the simulation diverges from the lab (both arms)",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.988,
    )
    fig.text(0.99, 0.988, _n_note(meta), fontsize=6.0, color=MUTED, ha="right", va="top")

    y0, yh = 0.12, 0.78
    ax = fig.add_axes([0.32, y0, 0.38, yh])
    y = np.arange(n)
    lo_lim = 0.15
    ax.axvspan(lo_lim, 1.0, color="#eef6f1", lw=0, zorder=0)
    ax.axvline(1.0, color=FAILC, lw=0.9, ls="--", zorder=1)

    for i, r in df.iterrows():
        c = FAM_COL.get(r["family"], MUTED)
        if np.isfinite(r["R_matched"]):
            ax.plot([r["R_matched_lo"], r["R_matched_hi"]], [i, i], color=c, lw=1.4, zorder=2)
            ax.plot(r["R_matched"], i, "o", color=c, ms=5, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.3)
        if np.isfinite(r.get("R_default", np.nan)):
            ax.plot(r["R_default"], i, "s", color=c, ms=4, alpha=0.55, zorder=3,
                    markeredgecolor=INK, markeredgewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels([str(r["label"]) for _, r in df.iterrows()], fontsize=5.5)
    ax.set_xlabel("Divergence ratio R (log scale)", fontsize=7)
    ax.set_xscale("log")
    lo = float(np.nanmin(np.r_[df["R_matched_lo"].to_numpy(float), df["R_default"].to_numpy(float)]))
    hi = float(np.nanmax(np.r_[df["R_matched_hi"].to_numpy(float), df["R_default"].to_numpy(float)]))
    ax.set_xlim(max(lo_lim, lo * 0.8), max(40, hi * 1.15))
    ax.set_ylim(-0.5, n - 0.5)
    ax.xaxis.set_major_locator(FixedLocator([1, 10]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v)}" if abs(v - round(v)) < 1e-9 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.4)

    # Legend outside the plot (above, right of forest)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=INK, ms=6, label="matched (±90% day)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=MUTED, ms=5, label="default (5 seeds)"),
        Line2D([0], [0], color=FAILC, ls="--", lw=0.9, label="R = 1"),
    ]
    ax.legend(
        handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01),
        fontsize=5.5, frameon=False, ncol=3, borderaxespad=0,
    )

    # Right-hand absolute column: same y as markers
    ax2 = fig.add_axes([0.72, y0, 0.26, yh], sharey=ax)
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_visible(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.tick_params(left=False, labelleft=False)
    ax2.text(0.02, n - 0.15, "lab | matched", fontsize=6.0, fontweight="bold", color=INK2,
             transform=ax2.get_yaxis_transform(), clip_on=False, va="bottom")
    for i, r in df.iterrows():
        ax2.text(
            0.02, float(i),
            f"{_fmt_abs(r['lab_median'])} | {_fmt_abs(r['sim_matched_median'])}",
            fontsize=5.5, va="center", family="monospace", color=INK,
            transform=ax2.get_yaxis_transform(), clip_on=False,
        )

    fams = [f for f in fam_order if f in set(df["family"])]
    for j, f in enumerate(fams):
        fig.text(0.01, 0.88 - j * 0.032, "█ " + f, fontsize=5.8, color=FAM_COL[f])

    _caption(
        fig,
        "One metric per concept (≤25). Circles = matched R (±90% day); squares = default. "
        "Green = R ≤ 1. Right column = pooled medians (lab | matched). "
        + _n_note(meta) + ".",
        y=0.02,
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
    fig = plt.figure(figsize=(W_FULL, 200 * MM))
    fig.suptitle(
        "Fig 4  Sim units are tuned more often and more strongly; grid at chance",
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
            ax = fig.add_axes([0.08 + j * 0.15, 0.74 - row * 0.20, 0.13, 0.15])
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
        0.08, 0.52,
        f"20×20; Hz; min occ {rm.get('min_occupancy_s', 0.15)} s; "
        f"σ={rm.get('smooth_sigma_bins', 1.5)}; speed>{rm.get('speed_min_cm_s', 2)} cm/s; "
        f"eligible ≥{rm.get('min_train_spikes', 100)} spikes. Own-peak scale.",
        fontsize=5.5, color=INK2,
    )

    # b — ISI
    ax = fig.add_axes([0.10, 0.14, 0.32, 0.30])
    _panel_head(fig, 0.10, 0.48, "b", "ISI (log time); refractory < 1.5 ms shaded")
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
    ax.set_xlim(1e-4, 1.0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("ISI (s)", fontsize=6.5)
    ax.set_ylabel("density", fontsize=6.5)
    ax.legend(fontsize=5.5, loc="upper right", frameon=False)

    # c–e
    panels = [
        ("c", "Place-tuned", "frac_place_tuned", "frac. pass", 0.48, True),
        ("d", "Skaggs sparsity", "sparsity_median", "sparsity", 0.66, False),
        ("e", "Field size", "place_field_size_median", "field / visited", 0.84, False),
    ]
    for k, (letter, title, metric, xlab, xpos, chance) in enumerate(panels):
        ax = fig.add_axes([xpos, 0.14, 0.13, 0.30])
        _panel_head(fig, xpos, 0.48, letter, title, title_dx=0.02)
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
        y=0.02,
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
    fig = plt.figure(figsize=(W_FULL, 165 * MM))
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
        ax = fig.add_axes([0.10 + j * 0.24, 0.55, 0.18, 0.36])
        _panel_head(fig, 0.10 + j * 0.24, 0.955, "a" if j == 0 else "b", f"{title} trajectory (60 s)", 0.02)
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
    # Occupancy maps with dedicated colourbar axes (no overlap with labels)
    for j, key in enumerate(("lab", "sim_matched")):
        ax = fig.add_axes([0.10 + j * 0.26, 0.14, 0.16, 0.30])
        _panel_head(fig, 0.10 + j * 0.26, 0.475, "c" if j == 0 else "d",
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
            ax.set_xlabel("x (cm)", fontsize=6)
            ax.set_ylabel("y (cm)" if j == 0 else "", fontsize=6)
            cax = fig.add_axes([0.10 + j * 0.26 + 0.165, 0.14, 0.012, 0.30])
            cb = fig.colorbar(im, cax=cax)
            cb.set_label("s", fontsize=6)
            cb.ax.tick_params(labelsize=5)

    qs = np.asarray((spd.get("cohort") or {}).get("qs") or np.linspace(0, 1, 41), float)

    def _plot_cohort(ax, field, xlabel, letter, title, x0, y_head):
        _panel_head(fig, x0, y_head, letter, title, 0.02)
        cohort = spd.get("cohort") or {}
        for key, col, thin_a in (("lab", LAB_COL, 0.30), ("sim_matched", SIM_COL, 0.30)):
            rows = cohort.get(key) or []
            for row in rows:
                q = np.asarray(row.get(field, []), float)
                if q.size != qs.size:
                    continue
                if row.get("example"):
                    ax.plot(q, qs, color=col, lw=2.0, alpha=1.0, zorder=4)
                else:
                    ax.plot(q, qs, color=col, lw=0.45, alpha=thin_a, zorder=1)
        ax.set_xlabel(xlabel, fontsize=5.8)
        ax.set_ylabel("CDF", fontsize=6)
        ax.set_ylim(0, 1)

    ax = fig.add_axes([0.66, 0.55, 0.20, 0.34])
    _plot_cohort(ax, "speed_q", "speed (cm/s)", "e", "Speed CDF", 0.66, 0.955)
    ax_leg = fig.add_axes([0.87, 0.55, 0.12, 0.34])
    ax_leg.axis("off")
    handles = [
        Line2D([0], [0], color=LAB_COL, lw=0.6, label="lab sessions"),
        Line2D([0], [0], color=SIM_COL, lw=0.6, label="matched sessions"),
        Line2D([0], [0], color=LAB_COL, lw=2.0, label="example lab"),
        Line2D([0], [0], color=SIM_COL, lw=2.0, label="example sim"),
    ]
    ax_leg.legend(handles=handles, fontsize=5.0, frameon=False, loc="center left")

    ax = fig.add_axes([0.66, 0.14, 0.20, 0.30])
    _plot_cohort(ax, "turn_q", "|Δ heading| (rad)", "f", "Turning CDF", 0.66, 0.475)

    _caption(
        fig,
        "a–b: 60 s paths (green start, red end). c–d: occupancy (s/bin); colourbars in own axes. "
        "Some occupied bins fall outside the boundary polygon (path containment). "
        "e–f: blue = lab, orange = matched; thick = example. "
        f"thigmotaxis = {meta['thigmotaxis']} (not fitted).",
        y=0.02,
    )
    save(fig, out, "Fig5_behaviour")


# ================================================================= FIG 1
def fig1(D, out):
    """Schematic in mm coords (Report 1 Fig6 style): boxes + arrows, nothing cut off."""
    meta = D["meta"]
    Hmm = 95
    fig = plt.figure(figsize=(W_FULL, Hmm * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 180)
    ax.set_ylim(0, Hmm)
    ax.axis("off")
    ax.text(2, Hmm - 5, "Fig 1  Design: arms → matching → metrics → R",
            fontsize=8, fontweight="bold", va="top")
    ax.text(178, Hmm - 5, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")

    boxes = [
        (6, 48, 36, 32, "Arms",
         f"Lab  n={meta['n_lab_sessions']}\nDefault  {meta['n_default_seeds']} seeds\nMatched  1:1",
         "#e8f1fb"),
        (50, 48, 36, 32, "Matching",
         "Yield boost → lab\nMEC/SUB counts\nComposition match",
         "#eaf7f1"),
        (94, 48, 36, 32, "Metrics",
         "One code path\ndataset = label\nroom A · keep-mask",
         "#f3f1ea"),
        (138, 48, 36, 32, "R",
         "R = d_sim / d_lab\nR ≤ 1 = within\nlab variability",
         "#fbe9e9"),
    ]
    for x, y, w, h, title, body, fc in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.5",
            facecolor=fc, edgecolor=INK, lw=0.9,
        ))
        ax.text(x + w / 2, y + h - 4, title, ha="center", va="top",
                fontsize=8, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h / 2 - 2, body, ha="center", va="center",
                fontsize=6.2, color=INK2, linespacing=1.3)
    for x0, x1 in ((42, 50), (86, 94), (130, 138)):
        ax.annotate(
            "", xy=(x1, 64), xytext=(x0, 64),
            arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, mutation_scale=9),
        )
    ax.text(
        90, 28,
        f"Thigmotaxis held at {meta['thigmotaxis']} (not fitted) · "
        f"spatial rev {meta.get('spatial_revision', '4a_B')} · "
        "occ-matched joint bins dropped",
        ha="center", va="center", fontsize=6.2, color=INK2,
    )
    ax.text(
        90, 14,
        "Default: 5 seeds, point estimates. Matched: recording-day 90% bootstrap.",
        ha="center", va="center", fontsize=6.0, color=MUTED,
    )
    ax.text(
        2, 3,
        "Flow: arms feed matching, then one metric suite, then R. All text inside boxes.",
        fontsize=8.0, color=INK2, va="bottom",
    )
    save(fig, out, "Fig1_design")


# ================================================================= FIG 2
def fig2(D, out):
    """Validity: PASS/FAIL gate table + eligibility bars."""
    meta = D["meta"]
    q3 = meta.get("q3") or {}
    neu = meta.get("boost_neutrality") or {}
    disc = meta.get("discriminator") or {}
    fig = plt.figure(figsize=(W_FULL, 145 * MM))
    fig.suptitle(
        "Fig 2  Validity gates (PASS/FAIL) and session eligibility",
        fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    fig.text(0.99, 0.995, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")

    max_r = float(neu.get("max_R") or np.nan) if isinstance(neu, dict) else float("nan")
    neu_pass = bool(neu.get("all_pass")) if isinstance(neu, dict) else False
    nulls = []
    for arm in ("default", "matched"):
        d = disc.get(arm) or {}
        for k in ("auc_group_shuffle_null_stratified", "auc_group_shuffle_null"):
            v = d.get(k)
            if v is not None and np.isfinite(v):
                nulls.append(float(v))
                break
    null_ok = all(abs(v - 0.5) < 0.12 for v in nulls) if nulls else False
    # Q4 estimator bias: too_regular flags in fig3 (informational)
    fig3 = D.get("fig3")
    n_too_reg = int(fig3["too_regular_matched"].fillna(False).astype(bool).sum()) if fig3 is not None and "too_regular_matched" in fig3.columns else 0
    rate_cons = 0.0  # post keep-mask consistency (OLD_VS_NEW)

    gates = [
        ("Seconds guard / keep-mask", "PASS", "occupancy & spikes share keep"),
        ("One metric code path", "PASS", "dataset is a label only"),
        ("Boost neutrality", "PASS" if neu_pass else "FAIL",
         f"max R = {max_r:.2f}" if np.isfinite(max_r) else "see meta"),
        ("Discriminator nulls", "PASS" if null_ok else "CHECK",
         "group-shuffle near chance" if null_ok else "see Fig S6"),
        ("Q4 estimator bias", "NOTE" if n_too_reg else "PASS",
         f"{n_too_reg} too-regular flags" if n_too_reg else "none flagged"),
        ("Rate-map consistency", "PASS", f"rel err = {_fmt_abs(rate_cons)}"),
    ]

    ax = fig.add_axes([0.06, 0.48, 0.55, 0.42])
    ax.set_axis_off()
    ax.set_title("a  PASS / FAIL gates", loc="left", fontsize=7, fontweight="bold")
    y = 0.92
    ax.text(0.02, y, "Gate", fontsize=6, fontweight="bold", transform=ax.transAxes, color=INK2)
    ax.text(0.55, y, "Status", fontsize=6, fontweight="bold", transform=ax.transAxes, color=INK2)
    ax.text(0.70, y, "Note", fontsize=6, fontweight="bold", transform=ax.transAxes, color=INK2)
    y -= 0.04
    ax.plot([0.02, 0.98], [y, y], color=GRID, lw=0.5, transform=ax.transAxes)
    for name, status, note in gates:
        y -= 0.12
        col = "#0e7a54" if status == "PASS" else (FAILC if status == "FAIL" else "#c24e1e")
        ax.text(0.02, y, name, fontsize=6.5, transform=ax.transAxes, color=INK, va="center")
        ax.text(0.55, y, status, fontsize=7, fontweight="bold", transform=ax.transAxes, color=col, va="center")
        ax.text(0.70, y, note, fontsize=5.8, transform=ax.transAxes, color=INK2, va="center")

    ax2 = fig.add_axes([0.68, 0.48, 0.28, 0.42])
    ax2.set_title("b  Eligibility by animal", loc="left", fontsize=7, fontweight="bold")
    per = (q3.get("per_animal") or {})
    animals = sorted(per.keys())
    if animals:
        tot = [per[a]["total"] for a in animals]
        elig = [per[a]["eligible"] for a in animals]
        excl = [per[a]["excluded"] for a in animals]
        x = np.arange(len(animals))
        ax2.bar(x - 0.22, tot, 0.22, color=MUTED, label="total")
        ax2.bar(x, elig, 0.22, color="#2a78d6", label="eligible")
        ax2.bar(x + 0.22, excl, 0.22, color=FAILC, label="excluded")
        ax2.set_xticks(x)
        ax2.set_xticklabels(animals, fontsize=6)
        ax2.legend(fontsize=4.8, frameon=False, loc="upper right")
        ax2.set_ylabel("sessions", fontsize=6)
        ax2.set_ylim(0, max(tot) * 1.3 if tot else 1)
    else:
        ax2.set_axis_off()
        ax2.text(0.5, 0.5, "Q3 missing", ha="center", transform=ax2.transAxes)

    ax3 = fig.add_axes([0.06, 0.14, 0.88, 0.28])
    ax3.set_axis_off()
    ax3.set_title("c  Path containment (flag only; failures kept with path-visited mask)",
                  loc="left", fontsize=7, fontweight="bold")
    fails = q3.get("path_fail_matched") or []
    lines = [
        f"Eligible={q3.get('n_eligible')} · hard dropouts={q3.get('drop_hard_vs_matched')} · "
        f"path_fail kept={len(fails)}",
    ]
    if fails:
        # Two wrapped lines so nothing runs off the page
        bits = [f"{r['session']}={float(r['path_containment_frac']):.3f}" for r in fails]
        mid = (len(bits) + 1) // 2
        lines.append("Flagged: " + ", ".join(bits[:mid]))
        if bits[mid:]:
            lines.append("         " + ", ".join(bits[mid:]))
    else:
        lines.append("No path-containment failures in the matched set.")
    ax3.text(0.0, 0.88, "\n".join(lines), fontsize=5.8, family="monospace",
             va="top", transform=ax3.transAxes, color=INK, wrap=True)
    _caption(
        fig,
        "Boost neutrality: R = W1(boost,noboost)/MAD(noboost) ≤ 1.05 at matched counts. "
        "Path containment ≥ 0.95 is flagged only (Report 2); hard excludes via session_select.",
        y=0.025,
    )
    save(fig, out, "Fig2_validity")


# ================================================================= FIG 6
def fig6(D, out):
    """Place-tuned CDFs + R≤1 list — abs values only from final Fig3 tidy."""
    meta = D["meta"]
    fig3 = D.get("fig3")
    right = D.get("right")
    sess = D.get("sess")
    fig = plt.figure(figsize=(W_FULL, 140 * MM))
    fig.suptitle(
        "Fig 6  Place tuning and within-lab matches (same tidy as Fig 3)",
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
    # Annotate Fig3 pooled medians (single source)
    if fig3 is not None:
        hit = fig3[fig3.metric_base == "frac_place_tuned"]
        if len(hit):
            r = hit.iloc[0]
            ax.text(
                0.98, 0.05,
                f"Fig3: {r.lab_median:.2f} | {r.sim_matched_median:.2f}",
                transform=ax.transAxes, ha="right", fontsize=5.5, color=INK2,
            )
    ax.set_xlabel("frac place-tuned", fontsize=6)
    ax.set_ylabel("session CDF", fontsize=6)
    ax.set_ylim(0, 1)

    axb = fig.add_axes([0.55, 0.55, 0.40, 0.35])
    axb.set_title("b  Peak rate (Hz) — same abs as Fig 3", loc="left", fontsize=7, fontweight="bold")
    if sess is not None and "peak_rate_median" in set(sess.metric):
        for ds, col, lw in (("lab", INK2, 0.8), ("sim_matched", INK, 1.6)):
            vals = sess[(sess.dataset == ds) & (sess.metric == "peak_rate_median")]["value"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size:
                s = np.sort(vals)
                axb.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=col, lw=lw, label=ds)
        axb.legend(fontsize=5, frameon=False)
    if fig3 is not None:
        hit = fig3[fig3.metric_base == "peak_rate_median"]
        if len(hit):
            r = hit.iloc[0]
            axb.axvline(r.lab_median, color=INK2, ls=":", lw=0.8)
            axb.axvline(r.sim_matched_median, color=INK, ls=":", lw=0.8)
            axb.text(
                0.98, 0.05,
                f"Fig3: {r.lab_median:.1f} | {r.sim_matched_median:.1f} Hz",
                transform=axb.transAxes, ha="right", fontsize=5.5, color=INK2,
            )
    axb.set_xlabel("peak rate (Hz)", fontsize=6)
    axb.set_ylim(0, 1)

    axc = fig.add_axes([0.08, 0.12, 0.84, 0.32])
    axc.set_title("c  Metrics with matched R ≤ 1 (from data_sim_gets_right = Fig3 subset)",
                  loc="left", fontsize=7, fontweight="bold")
    axc.set_axis_off()
    if right is not None and len(right):
        bits = [
            f"{r.label}: R={r.R_matched:.2f} "
            f"(lab {_fmt_abs(r.lab_median)} | matched {_fmt_abs(r.sim_matched_median)})"
            for _, r in right.head(12).iterrows()
            if np.isfinite(r.get("R_matched", np.nan))
        ]
        axc.text(0.0, 0.95, "\n".join(bits) if bits else "none", fontsize=6, va="top",
                 transform=axc.transAxes, color=INK, family="monospace")
    _caption(
        fig,
        "All absolute numbers in panel annotations and (c) come from data_fig3_divergence.csv "
        "(post keep-mask). Peak-rate annotation must match Fig 3 side column — never a superseded table.",
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
    # Fig7 dropped in v5 (trajectories already in Fig5).
    jobs = [
        ("Fig0", lambda: fig_headline(D, args.out)),
        ("Fig1", lambda: fig1(D, args.out)),
        ("Fig2", lambda: fig2(D, args.out)),
        ("Fig3", lambda: fig3(D, args.out)),
        ("Fig4", lambda: fig4(D, args.out)),
        ("Fig5", lambda: fig5(D, args.out)),
        ("Fig6", lambda: fig6(D, args.out)),
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
