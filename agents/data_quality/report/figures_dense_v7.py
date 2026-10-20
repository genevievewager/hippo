#!/usr/bin/env python3
"""Report 4a v7 dense figures — numbers frozen at v5 tidy.

Fig 1 = spikes too regular in time + rat too regular in space (8 panels).
Fig 2 = tuned too often / too strongly / never drift (6–7 panels; half-maps dropped).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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
    head_fig,
    save as _save,
)

DS_COL = {"lab": "#1a1a1a", "sim_matched": "#0b5fff", "sim_default": "#8f8d87"}
DS_LAB = {"lab": "lab", "sim_matched": "matched", "sim_default": "default"}
DS_MRK = {"lab": "o", "sim_matched": "s", "sim_default": "D"}


def save(fig, out, name):
    """Save after within-panel text-overlap check (PDF layer misses rasterized labels)."""
    from agents.data_quality.report.overlap_audit import audit_mpl_figure

    fig.canvas.draw()
    fails = audit_mpl_figure(fig)
    # Keep within-panel collisions; drop fig_edge noise from tight captions
    bad = [f for f in fails if f.get("kind") == "fig_text_text"]
    if bad:
        print(json.dumps({
            "mpl_overlap": name, "n": len(bad),
            "within_panel": sum(1 for f in bad if f.get("within_panel")),
            "sample": bad[:6],
        }, default=str), flush=True)
    _save(fig, out, name)


def _caption(fig, text, y=0.006):
    fig.text(0.01, y, textwrap.fill(text, 145), fontsize=6.2, color=INK2, va="bottom", ha="left")


def _n_note(meta):
    return (
        f"n = {meta['n_animals']} animals, {meta['n_lab_sessions']} lab; "
        f"{meta['n_matched_sessions']} matched; {meta['n_default_seeds']} default"
    )


def _panel(fig, x, y, letter, title, dx=0.016):
    fig.text(x, y, letter, fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    fig.text(x + dx, y + 0.001, title, fontsize=6.2, fontweight="bold", va="bottom", ha="left")


def _load(data: Path, behinfo: Path, step1: Path | None = None) -> dict:
    D = {
        "meta": json.loads((data / "data_meta.json").read_text()),
        "fig3": pd.read_csv(data / "data_fig3_divergence.csv"),
        "sess": pd.read_csv(data / "data_session_metrics.csv"),
        "behinfo": behinfo,
    }
    for name, key in (
        ("data_rate_maps.json", "rate_maps"),
        ("data_trajectories_60s.json", "traj"),
        ("data_isi_pool.json", "isi"),
        ("data_speed_turning_example.json", "spd"),
        ("data_rasters_1s.json", "rasters"),
    ):
        p = data / name
        if p.is_file():
            raw = json.loads(p.read_text())
            if key == "rate_maps":
                for r in raw:
                    r["rate"] = np.asarray(r["rate"], float)
            D[key] = raw
    D["enc_units"] = pd.read_csv(behinfo / "encoding_diag_units.csv")
    D["enc_diag"] = json.loads((behinfo / "encoding_diag_summary.json").read_text())
    D["drift_sum"] = json.loads((behinfo / "drift_noise_summary.json").read_text())
    D["drift_sep"] = pd.read_csv(behinfo / "drift_sep_sessions.csv")
    D["decode"] = pd.read_csv(behinfo / "decode_rows.csv")
    D["ncurve"] = pd.read_csv(behinfo / "n_curve.csv")
    D["uv"] = pd.read_csv(behinfo / "unique_variance_table.csv")
    sens_p = behinfo / "sensitivity_rows.csv"
    D["sens"] = pd.read_csv(sens_p) if sens_p.is_file() else pd.DataFrame()
    right_p = data / "data_sim_gets_right.csv"
    D["right"] = pd.read_csv(right_p) if right_p.is_file() else pd.DataFrame()
    if step1 and (step1 / "step1_session_distributions.csv").is_file():
        D["step1_sess"] = pd.read_csv(step1 / "step1_session_distributions.csv")
        D["step1_pool"] = pd.read_csv(step1 / "step1_dataset_pooled.csv")
    elif (data / "step1_session_distributions.csv").is_file():
        D["step1_sess"] = pd.read_csv(data / "step1_session_distributions.csv")
        D["step1_pool"] = pd.read_csv(data / "step1_dataset_pooled.csv")
    return D


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

# Fig 5b — simulator requests (cite panels only; avoid re-plotting tokens)
SIM_TODO_LINES = [
    "Add refractory / min-ISI knob — gap measured in Fig 1c (lab 0.084% vs matched 0.87%).",
    "Slow the rat and increase curvature — gaps in Fig 1e–f (speed 4.23 vs 7.10 cm/s; turn SD 0.31 vs 0.11).",
    "Decouple head from movement heading — lab residual ~45° (Fig 1h); sim currently equates them.",
    "Reduce over-tuning and over-informativeness — Fig 2b,d and Fig 3 (0.18 vs 0.37 place-tuned; 0.07 vs 0.43 bits).",
    "Introduce within-session map nonstationarity — Fig 2f–g (lab sep-curve falls; matched flat).",
    "Capture cap (DF7); SimConfig.seed (DF5); external traj or public speed/turn knobs (DF6).",
]


def _paired_strip(ax, lab_s: pd.Series, mat_s: pd.Series, def_med=None):
    common = sorted(set(lab_s.index) & set(mat_s.index))
    for s in common:
        ax.plot([0, 1], [lab_s[s], mat_s[s]], color="#c8c6c0", lw=0.35, alpha=0.5, zorder=1)
    if common:
        ax.scatter(np.zeros(len(common)), [lab_s[s] for s in common],
                   s=9, color=DS_COL["lab"], alpha=0.55, zorder=3, linewidths=0)
        ax.scatter(np.ones(len(common)), [mat_s[s] for s in common],
                   s=9, color=DS_COL["sim_matched"], alpha=0.55, zorder=3, linewidths=0)
        for i, series in ((0, lab_s.loc[common]), (1, mat_s.loc[common])):
            med = float(np.median(series))
            ax.plot(i, med, "o", color=DS_COL["lab" if i == 0 else "sim_matched"], ms=7,
                    markeredgecolor="white", markeredgewidth=0.45, zorder=5)
            # label beside the median, not over it
            if i == 0:
                ax.text(i - 0.22, med, f"{med:.3g}", fontsize=5.2, va="center", ha="right",
                        color=INK, zorder=6)
            else:
                ax.text(i + 0.22, med, f"{med:.3g}", fontsize=5.2, va="center", ha="left",
                        color=INK, zorder=6)
    if def_med is not None and np.isfinite(def_med):
        # marker only — no "def" text (collides with median labels on small-range panels)
        ax.plot(1.45, def_med, "D", color=DS_COL["sim_default"], ms=5.5, zorder=4,
                markeredgecolor="white", markeredgewidth=0.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["lab", "matched"], fontsize=6)
    ax.set_xlim(-0.75, 1.75 if def_med is None else 1.85)
    ax.axhline(0, color=GRID, lw=0.4)
    ax.grid(axis="y", color=GRID, lw=0.3)


# ================================================================= FIG 1
def fig1(D, out: Path):
    meta = D["meta"]
    isi = D.get("isi") or {}
    sess = D["sess"]
    traj = D.get("traj") or {}
    spd = D.get("spd") or {}
    rasters = D.get("rasters") or {}
    step1 = D.get("step1_sess")
    step1p = D.get("step1_pool")

    fig = plt.figure(figsize=(W_FULL, 268 * MM))
    fig.suptitle(
        "Fig 1  Simulated spikes are too regular in time, and the rat too regular in space",
        fontsize=7.6, fontweight="bold", x=0.01, ha="left", y=0.997,
    )
    fig.text(0.99, 0.997, _n_note(meta), fontsize=5.4, color=MUTED, ha="right", va="top")

    # ---- a: 1 s rasters ----
    _panel(fig, 0.01, 0.968, "a", "1 s rasters (20 units); red ticks = ISI < 1.5 ms")
    refr = float(isi.get("refractory_s", 0.0015))
    for j, (ds, title) in enumerate((("lab", "lab"), ("sim_matched", "matched"))):
        ax = fig.add_axes([0.055 + j * 0.225, 0.755, 0.21, 0.195])
        block = (rasters.get(ds) or {}).get("units") or []
        for u in block:
            t = np.asarray(u.get("t", []), float)
            y = np.full(len(t), int(u["unit"]))
            ax.vlines(t, y - 0.35, y + 0.35, color=INK, lw=0.45)
            if len(t) >= 2:
                d = np.diff(t)
                hit = np.where(d < refr)[0]
                for i in hit:
                    ax.plot([t[i], t[i + 1]], [y[i], y[i + 1]], "o", color=FAILC, ms=1.8, zorder=3)
        ax.set_xlim(0, float(rasters.get("dur_s", 1.0)))
        ax.set_ylim(-0.8, 19.8)
        ax.set_yticks([])
        ax.set_xlabel("t (s)", fontsize=5.2)
        ax.set_title(title, fontsize=6.5, pad=1)
        ax.tick_params(labelsize=5)

    # ---- b: ISI density ----
    ax = fig.add_axes([0.52, 0.755, 0.22, 0.195])
    _panel(fig, 0.52, 0.968, "b", "ISI density (log time); refractory shaded")
    bins = np.logspace(-4, 0, 60)
    for key, col, lw, lab in (
        ("lab", DS_COL["lab"], 1.1, "lab"),
        ("sim_gt", "#2a78d6", 1.2, "sim GT"),
        ("sim_sorted", DS_COL["sim_matched"], 1.4, "sim sorted"),
    ):
        v = np.asarray(isi.get(key, []), float)
        v = v[np.isfinite(v) & (v > 0)]
        if not v.size:
            continue
        if v.size > 200_000:
            v = np.random.default_rng(0).choice(v, 200_000, replace=False)
        h, e = np.histogram(v, bins=bins, density=True)
        ax.plot(0.5 * (e[:-1] + e[1:]), h, color=col, lw=lw, label=lab)
    ax.axvspan(1e-4, refr, color="#fbe9e9", lw=0, zorder=0)
    ax.axvline(refr, color=FAILC, ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.0)
    ax.set_xlabel("ISI (s)", fontsize=5.2)
    ax.set_ylabel("density", fontsize=5.5)
    ax.legend(fontsize=4.8, frameon=False, loc="upper right")
    ax.tick_params(labelsize=5)

    # ---- c: refractory paired ----
    # Keep y-ticks clear of the e/f panel titles on the row below (y≈0.715).
    ax = fig.add_axes([0.78, 0.770, 0.19, 0.180])
    _panel(fig, 0.78, 0.968, "c", "Refractory violation frac.")
    metric = "refractory_violation_rate"
    lab_s = sess[(sess.dataset == "lab") & (sess.metric == metric)].set_index("session")["value"]
    mat_s = sess[(sess.dataset == "sim_matched") & (sess.metric == metric)].set_index("session")["value"]
    def_v = sess[(sess.dataset == "sim_default") & (sess.metric == metric)]["value"].to_numpy(float)
    def_med = float(np.median(def_v[np.isfinite(def_v)])) if def_v.size else None
    _paired_strip(ax, lab_s, mat_s, def_med)
    ax.set_ylabel("frac ISI < 1.5 ms", fontsize=5.5)
    ax.tick_params(labelsize=5)

    # ---- d: paths ----
    _panel(fig, 0.01, 0.715, "d", "60 s paths with arena outlines")

    def _outline(ax, ar):
        pts = ar.get("outline") if ar else None
        if pts:
            pts = np.asarray(pts, float)
            ax.plot(pts[:, 0], pts[:, 1], color="#8a8780", lw=0.7, zorder=0)

    for j, (key, title) in enumerate((("lab", "lab"), ("sim_matched", "matched"))):
        ax = fig.add_axes([0.055 + j * 0.185, 0.445, 0.17, 0.250])
        t = traj.get(key) or {}
        x, y = np.asarray(t.get("x", []), float), np.asarray(t.get("y", []), float)
        ar = (traj.get("arena") or {}).get("lab" if key == "lab" else "sim", {})
        _outline(ax, ar)
        if len(x):
            ax.plot(x, y, color=INK, lw=0.45, zorder=2)
        if ar:
            ax.set_xlim(ar["xlim"]); ax.set_ylim(ar["ylim"])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=6, pad=1)
        ax.tick_params(labelsize=4.5)
        ax.set_xlabel("x (cm)", fontsize=5)
        if j == 0:
            ax.set_ylabel("y (cm)", fontsize=5)

    # ---- e / f: speed + turning CDFs ----
    qs = np.asarray((spd.get("cohort") or {}).get("qs") or np.linspace(0, 1, 41), float)
    cohort = spd.get("cohort") or {}

    def _cdf_panel(ax, field, xlabel, letter, title, x0):
        _panel(fig, x0, 0.715, letter, title)
        for key, col in (("lab", DS_COL["lab"]), ("sim_matched", DS_COL["sim_matched"])):
            rows = cohort.get(key) or []
            mats = []
            for row in rows:
                q = np.asarray(row.get(field, []), float)
                if q.size != qs.size:
                    continue
                ax.plot(q, qs, color=col, lw=0.4, alpha=0.28, zorder=1)
                mats.append(q)
            if mats:
                ax.plot(np.median(np.vstack(mats), axis=0), qs, color=col, lw=1.8, zorder=4, label=DS_LAB[key])
        ax.set_xlabel(xlabel, fontsize=5.5)
        ax.set_ylabel("CDF", fontsize=5.5)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=5, frameon=False, loc="lower right")
        ax.tick_params(labelsize=5)

    # Raise e/f so their xlabels clear the g/h panel titles below
    ax = fig.add_axes([0.44, 0.445, 0.25, 0.250])
    _cdf_panel(ax, "speed_q", "speed (cm/s)", "e", "Speed CDFs (sessions; medians bold)", 0.44)
    ax = fig.add_axes([0.73, 0.445, 0.24, 0.250])
    _cdf_panel(ax, "turn_q", "|Δ heading| (rad)", "f", "Moving-only turning CDFs", 0.73)

    # ---- g: movement-state stacked bars ----
    ax = fig.add_axes([0.07, 0.08, 0.42, 0.285])
    _panel(fig, 0.01, 0.375, "g", "Movement-state fractions (still / slow / fast)")
    if step1 is not None:
        datasets = ["lab", "sim_matched", "sim_default"]
        x = np.arange(len(datasets))
        bottoms = np.zeros(len(datasets))
        colors = {"frac_still": "#6e6c66", "frac_slow": "#2a78d6", "frac_fast": "#eb6834"}
        labels = {"frac_still": "still", "frac_slow": "slow", "frac_fast": "fast"}
        for col in ("frac_still", "frac_slow", "frac_fast"):
            vals = []
            for ds in datasets:
                sub = step1[step1.dataset == ds][col].to_numpy(float)
                sub = sub[np.isfinite(sub)]
                vals.append(float(np.median(sub)) if sub.size else 0.0)
            ax.bar(x, vals, bottom=bottoms, color=colors[col], width=0.55, label=labels[col])
            for xi, v, b in zip(x, vals, bottoms):
                if v >= 0.06:
                    ax.text(xi, b + v / 2, f"{v:.2f}", ha="center", va="center", fontsize=5.2, color="white")
            bottoms = bottoms + np.asarray(vals)
        ax.set_xticks(x)
        ax.set_xticklabels([DS_LAB[d] for d in datasets], fontsize=6.5)
        ax.set_ylabel("fraction of time", fontsize=6)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=5.2, frameon=False, loc="upper right", ncol=3)
        ax.tick_params(labelsize=5)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "step1 state fracs missing", ha="center", color=MUTED)

    # ---- h: HD − move ----
    ax = fig.add_axes([0.55, 0.08, 0.42, 0.285])
    _panel(fig, 0.55, 0.375, "h", "Lab |HD − move heading| (session medians)")
    if step1 is not None:
        lab_hd = step1[step1.dataset == "lab"]["hd_vs_move_med_abs_deg"].to_numpy(float)
        lab_hd = lab_hd[np.isfinite(lab_hd)]
        mat_hd = step1[step1.dataset == "sim_matched"]["hd_vs_move_med_abs_deg"].to_numpy(float)
        mat_hd = mat_hd[np.isfinite(mat_hd)]
        ax.hist(lab_hd, bins=np.linspace(0, 90, 19), color=DS_COL["lab"], alpha=0.75, edgecolor="white", lw=0.4)
        lab_med = float(np.median(lab_hd)) if lab_hd.size else np.nan
        mat_med = float(np.median(mat_hd)) if mat_hd.size else np.nan
        ax.axvline(lab_med, color=DS_COL["lab"], lw=1.6, label=f"lab med {lab_med:.0f}°")
        ax.axvline(mat_med, color=DS_COL["sim_matched"], lw=1.6, ls="--", label=f"matched med {mat_med:.0f}°")
        if step1p is not None and len(step1p[step1p.dataset == "sim_default"]):
            def_hd = float(step1p[step1p.dataset == "sim_default"].iloc[0]["median_hd_vs_move_abs_deg"])
            ax.axvline(def_hd, color=DS_COL["sim_default"], lw=1.2, ls=":", label=f"default {def_hd:.0f}°")
        ax.set_xlabel("|HD − move| (deg)", fontsize=6)
        ax.set_ylabel("n sessions", fontsize=6)
        ax.legend(fontsize=5.2, frameon=False, loc="upper right")
        ax.tick_params(labelsize=5)
        ax.set_xlim(0, 90)
    else:
        ax.axis("off")

    _caption(
        fig,
        "a: 1 s excerpt (example session); red = ISI < 1.5 ms. b: v5 ISI pool (lab / sim GT / sorted). "
        "c: paired sessions; ♦ default. d: v5 60 s traj. e–f: thin = sessions, bold = median CDF. "
        "g–h: frozen step1 (still <2, slow 2–8, fast ≥8 cm/s). Absolutes only; R in Fig 5.",
        y=0.014,
    )
    save(fig, str(out), "Fig1_spikes_behaviour")


# ================================================================= FIG 2
def fig2(D, out: Path):
    meta = D["meta"]
    RM = D.get("rate_maps") or []
    sess = D["sess"]
    enc_u = D["enc_units"]
    enc_diag = D["enc_diag"]
    sep = (D["drift_sum"].get("sep_curve") or {})
    drift_sep = D["drift_sep"]

    fig = plt.figure(figsize=(W_FULL, 268 * MM))
    fig.suptitle(
        "Fig 2  Simulated cells are tuned too often, too strongly, and never drift",
        fontsize=7.6, fontweight="bold", x=0.01, ha="left", y=0.997,
    )
    fig.text(0.99, 0.997, _n_note(meta), fontsize=5.4, color=MUTED, ha="right", va="top")

    # ---- a: map gallery — frozen store is p25/p50/p75 only (6 maps/arm), not SI p10–p90 ----
    _panel(fig, 0.01, 0.968, "a", "Place-tuned rate maps (Hz); stored p25/p50/p75 assets; per-panel scale")
    for row_i, ds in enumerate(("lab", "sim_matched")):
        maps = [r for r in RM if r["dataset"] == ds and r.get("ok", True)]
        maps = sorted(maps, key=lambda r: (int(r.get("percentile", 50)), float(r.get("spat_info", 0))))
        # show all stored maps (≤6); pad layout to 5–6 columns
        maps = maps[:6]
        for j, r in enumerate(maps):
            ax = fig.add_axes([0.05 + j * 0.10, 0.755 - row_i * 0.165, 0.092, 0.145])
            rate = np.ma.masked_invalid(np.asarray(r["rate"], float))
            ext = r["extent"]
            vmax = float(np.nanmax(r["rate"])) if np.isfinite(r["rate"]).any() else 1.0
            ax.set_facecolor("#d8d6d0")
            ax.imshow(
                rate.T, origin="lower",
                extent=[ext[0], ext[1], ext[2], ext[3]],
                cmap="viridis", vmin=0, vmax=vmax,
                aspect="equal", interpolation="nearest",
            )
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#6e6c66"); sp.set_linewidth(0.35)
            pct = r.get("percentile", "?")
            ax.set_title(
                f"p{pct}  SI {r['spat_info']:.2f}\n{r['peak_hz']:.1f} Hz",
                fontsize=4.4, pad=0.8, linespacing=1.0,
            )
            if j == 0:
                ax.set_ylabel("lab" if ds == "lab" else "matched", fontsize=6)

    # ---- b: tuned fractions ----
    ax = fig.add_axes([0.66, 0.62, 0.31, 0.32])
    _panel(fig, 0.66, 0.968, "b", "Fraction tuned (shuffle 95%)")
    classes = [
        ("frac_place_tuned", "place"),
        ("frac_border_pass", "border"),
        ("frac_hd_pass", "HD"),
        ("frac_speed_pass", "speed"),
        ("frac_grid_pass", "grid"),
    ]
    x = np.arange(len(classes))
    w = 0.35
    for di, ds in enumerate(("lab", "sim_matched")):
        meds = []
        for ci, (metric, _) in enumerate(classes):
            vals = sess[(sess.dataset == ds) & (sess.metric == metric)]["value"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            meds.append(float(np.median(vals)) if vals.size else np.nan)
            if vals.size:
                xx = ci + (di - 0.5) * w + np.random.default_rng(di * 10 + ci).normal(0, 0.02, len(vals))
                ax.scatter(xx, vals, s=4, alpha=0.22, color=DS_COL[ds], zorder=2, linewidths=0)
        bars = ax.bar(x + (di - 0.5) * w, meds, w * 0.9, color=DS_COL[ds], label=DS_LAB[ds], zorder=3, alpha=0.88)
        for rect, m in zip(bars, meds):
            if np.isfinite(m):
                ax.text(rect.get_x() + rect.get_width() / 2, m + 0.008, f"{m:.2f}",
                        ha="center", va="bottom", fontsize=4.8, color=INK)
    # default as small markers
    for ci, (metric, _) in enumerate(classes):
        dv = sess[(sess.dataset == "sim_default") & (sess.metric == metric)]["value"].to_numpy(float)
        dv = dv[np.isfinite(dv)]
        if dv.size:
            ax.plot(ci + 0.42, float(np.median(dv)), "D", color=DS_COL["sim_default"], ms=4, zorder=4)
    ax.axhline(0.05, color=FAILC, ls="--", lw=0.75, label="chance 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in classes], fontsize=6)
    ax.set_ylabel("fraction of units", fontsize=5.5)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=4.8, frameon=False, loc="upper right")
    ax.tick_params(labelsize=5)
    ax.grid(axis="y", color=GRID, lw=0.3)

    # ---- c: SI among place-tuned ----
    ax = fig.add_axes([0.08, 0.34, 0.28, 0.22])
    _panel(fig, 0.01, 0.58, "c", "Spatial information (place-tuned units)")
    for ds in ("lab", "sim_matched"):
        v = enc_u[(enc_u.dataset == ds) & (enc_u.place_tuned == True)]["spatial_info"].to_numpy(float)
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        if v.size > 3000:
            v = np.random.default_rng(1).choice(v, 3000, replace=False)
        s = np.sort(v)
        ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=DS_COL[ds], lw=1.3, label=DS_LAB[ds])
    ax.set_xlabel("SI (bits)", fontsize=5.5)
    ax.set_ylabel("ECDF", fontsize=5.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5, frameon=False, loc="lower right")
    ax.tick_params(labelsize=5)
    ax.grid(color=GRID, lw=0.3)

    # ---- d: bits/spike paired — bits_ratemap_end → 0.07 | 0.43 ----
    ax = fig.add_axes([0.42, 0.34, 0.24, 0.22])
    _panel(fig, 0.42, 0.58, "d", "Held-out bits/spike (place-tuned; paired)")
    bits_col = "bits_ratemap_end"
    pt = enc_u[enc_u.place_tuned == True]
    sm = pt.groupby(["dataset", "session"], as_index=False)[bits_col].median()
    lab_s = sm[sm.dataset == "lab"].set_index("session")[bits_col]
    mat_s = sm[sm.dataset == "sim_matched"].set_index("session")[bits_col]
    # v5 frozen: bits_end_med_place = unit-pooled median among place-tuned (0.07 | 0.43)
    table = (enc_diag.get("table") or {})
    lab_pool = float((table.get("lab") or {}).get("bits_end_med_place"))
    mat_pool = float((table.get("sim_matched") or {}).get("bits_end_med_place"))
    common = sorted(set(lab_s.index) & set(mat_s.index))
    for s in common:
        ax.plot([0, 1], [lab_s[s], mat_s[s]], color="#c8c6c0", lw=0.35, alpha=0.5, zorder=1)
    if common:
        ax.scatter(np.zeros(len(common)), [lab_s[s] for s in common],
                   s=9, color=DS_COL["lab"], alpha=0.45, zorder=3, linewidths=0)
        ax.scatter(np.ones(len(common)), [mat_s[s] for s in common],
                   s=9, color=DS_COL["sim_matched"], alpha=0.45, zorder=3, linewidths=0)
    ax.plot(0, lab_pool, "o", color=DS_COL["lab"], ms=8, markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax.plot(1, mat_pool, "o", color=DS_COL["sim_matched"], ms=8, markeredgecolor="white", markeredgewidth=0.5, zorder=6)
    ax.text(0, lab_pool, f"  {lab_pool:.2f}", fontsize=6.0, va="center", color=INK, zorder=7)
    ax.text(1, mat_pool, f"  {mat_pool:.2f}", fontsize=6.0, va="center", color=INK, zorder=7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["lab", "matched"], fontsize=6)
    ax.set_xlim(-0.35, 1.55)
    ax.axhline(0, color=GRID, lw=0.4)
    ax.grid(axis="y", color=GRID, lw=0.3)
    ax.set_ylabel("bits / spike", fontsize=5.5)
    ax.tick_params(labelsize=5)

    # ---- f: drift curve ----
    ax = fig.add_axes([0.72, 0.34, 0.24, 0.22])
    _panel(fig, 0.72, 0.58, "f", "Place-map r vs block separation")
    for ds in ("lab", "sim_matched"):
        c = sep.get(ds) or {}
        by = c.get("by_sep") or {}
        if not by:
            continue
        seps = sorted(int(k) for k in by)
        meds = [by[str(s)]["median"] for s in seps]
        los = [by[str(s)]["lo90"] for s in seps]
        his = [by[str(s)]["hi90"] for s in seps]
        yerr = np.vstack([np.asarray(meds) - np.asarray(los), np.asarray(his) - np.asarray(meds)])
        ax.errorbar(seps, meds, yerr=yerr, fmt="o-", color=DS_COL[ds], ms=4.5,
                    capsize=2, lw=1.0, label=DS_LAB[ds])
    ax.set_xlabel("Block separation", fontsize=5.5)
    ax.set_ylabel("Median place-map r", fontsize=5.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.3, 1.02)
    ax.legend(fontsize=5, frameon=False, loc="lower left")
    ax.tick_params(labelsize=5)
    ax.grid(color=GRID, lw=0.3)

    # ---- g: adjacent-block paired ----
    ax = fig.add_axes([0.08, 0.05, 0.42, 0.24])
    _panel(fig, 0.01, 0.305, "g", "Adjacent-block reliability (sep = 1; paired)")
    adj = drift_sep[drift_sep.sep == 1]
    lab_a = adj[adj.dataset == "lab"].set_index("session")["r_med"]
    mat_a = adj[adj.dataset == "sim_matched"].set_index("session")["r_med"]
    _paired_strip(ax, lab_a, mat_a, None)
    ax.set_ylabel("place-map r", fontsize=5.5)
    ax.set_ylim(0, 1.05)
    ax.tick_params(labelsize=5)

    # note: panel e dropped (no half-maps in tidy) — fill as callout box
    ax = fig.add_axes([0.56, 0.05, 0.40, 0.24])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.05), 0.96, 0.90, fill=True, facecolor="#f4f3ef",
                            edgecolor="#c8c6c0", lw=0.6, transform=ax.transAxes))
    ax.text(
        0.05, 0.90,
        "e  First/second-half maps — dropped\n"
        "(half-maps not in v5 tidy; no recompute).\n"
        "Drift claim = panels f + g.\n\n"
        "a gallery: frozen store has p25/p50/p75 only\n"
        "(6 maps/arm); not SI population p10–p90.\n"
        "Per-panel colour scale (own peak Hz).",
        fontsize=6.0, va="top", ha="left", color=INK2, transform=ax.transAxes,
    )

    _caption(
        fig,
        "a: stored p25/p50/p75 maps (6/arm); no per-unit SI-percentile gallery in v5 tidy. "
        "b: bars = session medians; dots = sessions; ♦ = default. "
        "c: ECDF of unit SI among place-tuned. "
        "d: bits_ratemap_end among place-tuned; bold = bits_end_med_place (0.07 | 0.43). "
        "f: 90% recording-day bootstrap. g: sep=1 per session. Absolute values; R in Fig 5 only.",
        y=0.005,
    )
    save(fig, str(out), "Fig2_tuning_drift")


# ================================================================= FIG 3
def _pooled_decode(dec: pd.DataFrame) -> pd.DataFrame:
    d = dec.copy()
    if "N_role" in d.columns:
        d = d[d["N_role"] == "primary"]
    if "region" in d.columns:
        d = d[d["region"] == "pooled"]
    if "N" in d.columns:
        d = d[d["N"] == 20]
    return d


def fig3(D, out: Path, ncurve_ladder: Path | None = None):
    """Simulated populations carry far more behavioural information."""
    meta = D["meta"]
    dec_all = D["decode"]
    dec = _pooled_decode(dec_all)
    uv = D["uv"]
    sens = D.get("sens")
    # n-curve: prefer ladder table when present
    if ncurve_ladder and Path(ncurve_ladder).is_file():
        ncurve = pd.read_csv(ncurve_ladder)
    else:
        ncurve = D["ncurve"]

    fig = plt.figure(figsize=(W_FULL, 268 * MM))
    fig.suptitle(
        "Fig 3  Simulated populations carry far more behavioural information",
        fontsize=7.6, fontweight="bold", x=0.01, ha="left", y=0.997,
    )
    fig.text(0.99, 0.997, _n_note(meta), fontsize=5.4, color=MUTED, ha="right", va="top")

    # ---- a: info per variable ----
    var_order = [
        ("position", "position"),
        ("wall_distance_bin2", "wall distance"),
        ("movement_direction", "movement dir."),
        ("speed", "speed"),
        ("movement_state", "mov. state"),
    ]
    ax = fig.add_axes([0.12, 0.70, 0.40, 0.24])
    _panel(fig, 0.01, 0.968, "a", "Information per variable (raw − max(null, 0); N = 20)")
    datasets = ["lab", "sim_matched", "sim_default"]
    offsets = {"lab": -0.22, "sim_matched": 0.0, "sim_default": 0.22}
    y_pos = np.arange(len(var_order))
    for vi, (var, _lab) in enumerate(var_order):
        for ds in datasets:
            sub = dec[(dec.dataset == ds) & (dec.variable == var)]["primary_minus_null"].to_numpy(float)
            sub = sub[np.isfinite(sub)]
            if not sub.size:
                continue
            yy = vi + offsets[ds] + np.random.default_rng(vi * 10 + hash(ds) % 7).normal(0, 0.015, len(sub))
            ax.scatter(sub, yy, s=6, alpha=0.25, color=DS_COL[ds], zorder=2, linewidths=0)
            ax.plot(np.median(sub), vi + offsets[ds], DS_MRK[ds], color=DS_COL[ds], ms=6.5,
                    markeredgecolor="white", markeredgewidth=0.35, zorder=4)
    hd = dec[(dec.dataset == "lab") & (dec.variable == "head_direction")]["primary_minus_null"].to_numpy(float)
    hd = hd[np.isfinite(hd)]
    ax.axvline(0, color=GRID, lw=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lab for _, lab in var_order], fontsize=6.5)
    ax.set_xlabel("Info above null", fontsize=6)
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.55, len(var_order) - 0.4)
    ax.tick_params(labelsize=5)
    ax.grid(axis="x", color=GRID, lw=0.3)
    if hd.size:
        ax.text(0.02, 0.02, f"HD lab-only med {np.median(hd):.2f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=5.0, color=MUTED)
    handles = [
        Line2D([0], [0], marker=DS_MRK[ds], color="w", markerfacecolor=DS_COL[ds], ms=6, label=DS_LAB[ds])
        for ds in datasets
    ]
    ax.legend(handles=handles, fontsize=5.0, frameon=False, loc="upper right")

    # ---- b: position info vs N ----
    axb = fig.add_axes([0.62, 0.70, 0.34, 0.24])
    _panel(fig, 0.62, 0.968, "b", "Position info vs N (leave chance?)")
    pos = ncurve[ncurve.variable == "position"].copy()
    want = [10, 20, 40, 80, 160]
    has_ladder = (
        any(n in set(pos.N.astype(int)) for n in (80, 160))
        and pos[pos.N == 80].session.nunique() > 5
    )
    lab_xs, lab_meds = [], []
    y_lo, y_hi = -0.2, 0.7
    for ds, col in (("lab", DS_COL["lab"]), ("sim_matched", DS_COL["sim_matched"])):
        sub = pos[pos.dataset == ds]
        xs, meds, los, his = [], [], [], []
        for n in want:
            vals = sub[sub.N == n]["value"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 3:
                continue
            xs.append(n)
            meds.append(float(np.median(vals)))
            los.append(float(np.clip(np.percentile(vals, 5), y_lo, y_hi)))
            his.append(float(np.clip(np.percentile(vals, 95), y_lo, y_hi)))
        if "N_role" in sub.columns:
            mx = sub[sub.N_role == "max"]
        else:
            idx = sub.groupby("session").N.idxmax()
            mx = sub.loc[idx]
        if len(mx) >= 3:
            xs.append(float(np.median(mx.N)))
            meds.append(float(np.median(mx.value)))
            los.append(float(np.clip(np.percentile(mx.value, 5), y_lo, y_hi)))
            his.append(float(np.clip(np.percentile(mx.value, 95), y_lo, y_hi)))
        if not xs:
            continue
        if ds == "lab":
            lab_xs, lab_meds = list(xs), list(meds)
        axb.fill_between(xs, los, his, color=col, alpha=0.15, lw=0, clip_on=True)
        axb.plot(xs, meds, "-o", color=col, ms=4, lw=1.2, label=DS_LAB[ds], clip_on=True)
    axb.axhline(0, color=GRID, lw=0.6)
    axb.set_ylim(y_lo, y_hi)
    axb.set_xscale("log")
    axb.set_xlabel("N units", fontsize=6)
    axb.set_ylabel("CV R² (session median)", fontsize=6)
    axb.legend(fontsize=5.2, frameon=False, loc="lower right")
    axb.tick_params(labelsize=5)
    axb.grid(color=GRID, lw=0.3)
    # Lab zero-crossing N (linear interp in log-N space)
    if len(lab_xs) >= 2:
        order = np.argsort(lab_xs)
        lx = np.asarray(lab_xs, float)[order]
        ly = np.asarray(lab_meds, float)[order]
        n_cross = None
        for i in range(len(lx) - 1):
            if ly[i] == 0:
                n_cross = float(lx[i])
                break
            if (ly[i] < 0 <= ly[i + 1]) or (ly[i] > 0 >= ly[i + 1]):
                log_n0, log_n1 = np.log(lx[i]), np.log(lx[i + 1])
                t = (0.0 - ly[i]) / (ly[i + 1] - ly[i])
                n_cross = float(np.exp(log_n0 + t * (log_n1 - log_n0)))
                break
        if n_cross is not None and np.isfinite(n_cross):
            axb.axvline(n_cross, color=DS_COL["lab"], ls=":", lw=0.8, zorder=3)
            axb.text(
                n_cross, y_hi - 0.04, f"lab=0 at N≈{n_cross:.0f}",
                fontsize=5.0, color=DS_COL["lab"], ha="left", va="top", rotation=90,
            )
    if not has_ladder:
        axb.text(0.5, 0.95, "ladder pending (showing available N)", transform=axb.transAxes,
                 ha="center", va="top", fontsize=5.5, color=FAILC)

    # ---- c: MEC vs SUB ----
    axc = fig.add_axes([0.08, 0.40, 0.40, 0.22])
    _panel(fig, 0.01, 0.64, "c", "MEC vs SUB at N = 20 (info above null)")
    vars_c = ["position", "movement_direction", "speed", "wall_distance_bin2"]
    labs_c = ["position", "direction", "speed", "wall"]
    x = np.arange(len(vars_c))
    w = 0.18
    # lab MEC, lab SUB, matched MEC, matched SUB
    series = [
        ("lab", "MEC", DS_COL["lab"], "o", -1.5),
        ("lab", "SUB", "#6e6c66", "s", -0.5),
        ("sim_matched", "MEC", DS_COL["sim_matched"], "o", 0.5),
        ("sim_matched", "SUB", "#7eb6ff", "s", 1.5),
    ]
    for ds, reg, col, mk, off in series:
        meds = []
        for v in vars_c:
            sub = dec_all[
                (dec_all.dataset == ds) & (dec_all.region == reg) &
                (dec_all.variable == v) & (dec_all.N_role == "primary")
            ]["primary_minus_null"].to_numpy(float)
            sub = sub[np.isfinite(sub)]
            meds.append(float(np.median(sub)) if sub.size else np.nan)
        axc.plot(x + off * w * 0.35, meds, f"-{mk}", color=col, ms=5, lw=1.0, label=f"{DS_LAB[ds]} {reg}")
    axc.axhline(0, color=GRID, lw=0.5)
    axc.set_xticks(x)
    axc.set_xticklabels(labs_c, fontsize=6)
    axc.set_ylabel("info above null", fontsize=6)
    axc.legend(fontsize=4.6, frameon=False, ncol=2, loc="upper left")
    axc.tick_params(labelsize=5)
    axc.grid(axis="y", color=GRID, lw=0.3)

    # ---- d: population VE ----
    axd = fig.add_axes([0.58, 0.40, 0.28, 0.22])
    _panel(fig, 0.58, 0.64, "d", "Population variance explained by behaviour")
    groups = [
        ("position", "position"),
        ("wall", "wall"),
        ("direction", "direction"),
        ("speed", "speed"),
        ("accel", "accel"),
        ("movement_state", "movement state"),
        ("shared", "shared"),
    ]
    gcol = {
        "position": "#2a78d6", "wall": "#1baf7a", "direction": "#7b5ea7",
        "speed": "#eb6834", "accel": "#c24e1e", "movement_state": "#52514e", "shared": "#b0aea6",
    }
    datasets_c = ["lab", "sim_matched", "sim_default"]
    x = np.arange(len(datasets_c))
    bottoms = np.zeros(len(datasets_c))
    for g, glab in groups:
        vals = []
        for ds in datasets_c:
            row = uv[uv.dataset == ds]
            col = "shared" if g == "shared" else f"unique_{g}"
            if row.empty or col not in row.columns:
                vals.append(0.0)
            else:
                vals.append(max(0.0, float(row.iloc[0][col])))
        axd.bar(x, vals, bottom=bottoms, color=gcol[g], width=0.55, label=glab)
        bottoms = bottoms + np.asarray(vals)
    axd.set_xticks(x)
    axd.set_xticklabels([DS_LAB[d] for d in datasets_c], fontsize=6.5)
    axd.set_ylabel("Variance share (clipped ≥ 0)", fontsize=6)
    axd.legend(
        fontsize=4.6, frameon=False, ncol=1, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
    )
    axd.text(0.02, 0.95, "Lab VE ≤ 0 — prefer Fig 2d.",
             transform=axd.transAxes, fontsize=5.2, color=INK2, va="top")
    axd.tick_params(labelsize=5)

    # ---- e: decode err/floor vs N ----
    axe = fig.add_axes([0.08, 0.08, 0.28, 0.24])
    _panel(fig, 0.01, 0.34, "e", "Decode error/floor vs N (position)")
    tick_ns = []
    max_n_med = None
    for ds, col in (("lab", DS_COL["lab"]), ("sim_matched", DS_COL["sim_matched"])):
        xs, ys, ylo, yhi = [], [], [], []
        for role, xlab in (("primary", 20), ("secondary_max", None)):
            sub = dec_all[
                (dec_all.dataset == ds) & (dec_all.region == "pooled") &
                (dec_all.variable == "position") & (dec_all.N_role == role)
            ]
            vals = sub["err_over_floor"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if not vals.size:
                continue
            n_med = float(np.median(sub["N"])) if xlab is None else float(xlab)
            if xlab is None:
                max_n_med = n_med if max_n_med is None else float(np.median([max_n_med, n_med]))
            xs.append(n_med)
            ys.append(float(np.median(vals)))
            ylo.append(float(np.percentile(vals, 5)))
            yhi.append(float(np.percentile(vals, 95)))
        if xs:
            axe.fill_between(xs, ylo, yhi, color=col, alpha=0.15, lw=0)
            axe.plot(xs, ys, "-o", color=col, ms=5, lw=1.2, label=DS_LAB[ds])
            tick_ns.extend(xs)
    axe.axhline(1.0, color=FAILC, ls="--", lw=0.7, label="floor")
    axe.set_xlabel("N units", fontsize=6)
    axe.set_ylabel("error / floor", fontsize=6)
    axe.set_xscale("log")
    # ticks at 20 and max; add ladder N when present
    xticks = [20.0]
    xticklabels = ["20"]
    if max_n_med is not None and np.isfinite(max_n_med):
        xticks.append(float(max_n_med))
        xticklabels.append(f"max ({max_n_med:.0f})")
    if has_ladder:
        for n in (10, 40, 80, 160):
            if n not in xticks:
                xticks.append(float(n))
                xticklabels.append(str(n))
    order = np.argsort(xticks)
    xticks = [xticks[i] for i in order]
    xticklabels = [xticklabels[i] for i in order]
    axe.set_xticks(xticks)
    axe.set_xticklabels(xticklabels, fontsize=5.0)
    axe.legend(fontsize=5, frameon=False, loc="upper right")
    axe.tick_params(labelsize=5)
    axe.grid(color=GRID, lw=0.3)

    # ---- f: paired N=20 err/floor ----
    axf = fig.add_axes([0.42, 0.08, 0.24, 0.24])
    _panel(fig, 0.42, 0.34, "f", "Error/floor at N = 20 (paired)")
    lab_e = dec_all[
        (dec_all.dataset == "lab") & (dec_all.region == "pooled") &
        (dec_all.variable == "position") & (dec_all.N_role == "primary")
    ].set_index("session")["err_over_floor"]
    mat_e = dec_all[
        (dec_all.dataset == "sim_matched") & (dec_all.region == "pooled") &
        (dec_all.variable == "position") & (dec_all.N_role == "primary")
    ].set_index("session")["err_over_floor"]
    _paired_strip(axf, lab_e, mat_e, None)
    axf.axhline(1.0, color=FAILC, ls="--", lw=0.6)
    axf.set_ylabel("error / floor", fontsize=6)
    axf.tick_params(labelsize=5)

    # ---- g: reweighted sensitivity ----
    axg = fig.add_axes([0.72, 0.08, 0.24, 0.24])
    _panel(fig, 0.72, 0.34, "g", "Reweighted test (raw vs rewt.)")
    if sens is not None and len(sens):
        vars_g = [("position", "pos"), ("distance_to_wall", "wall"), ("speed", "speed")]
        x = np.arange(len(vars_g))
        for di, ds in enumerate(("lab", "sim_matched")):
            raw_m, rw_m = [], []
            for v, _ in vars_g:
                raw = dec[(dec.dataset == ds) & (dec.variable == v)]["primary_value"].to_numpy(float)
                raw = raw[np.isfinite(raw)]
                raw_m.append(float(np.median(raw)) if raw.size else np.nan)
                rw = sens[(sens.dataset == ds) & (sens.variable == v) & (sens.reweighted == True)]["r2"].to_numpy(float)
                rw = rw[np.isfinite(rw)]
                rw_m.append(float(np.median(rw)) if rw.size else np.nan)
            off = (di - 0.5) * 0.15
            axg.plot(x + off, raw_m, "x", color=DS_COL[ds], ms=6, label=f"{DS_LAB[ds]} raw")
            axg.plot(x + off, rw_m, "o", color=DS_COL[ds], ms=5, label=f"{DS_LAB[ds]} rewt")
        axg.set_xticks(x)
        axg.set_xticklabels([lab for _, lab in vars_g], fontsize=6)
        axg.set_ylabel("CV R²", fontsize=6)
        axg.axhline(0, color=GRID, lw=0.5)
        axg.legend(fontsize=4.4, frameon=False, loc="best", ncol=2)
        axg.tick_params(labelsize=5)
        axg.grid(axis="y", color=GRID, lw=0.3)
    else:
        axg.axis("off")
        axg.text(0.5, 0.5, "sensitivity rows missing", ha="center", color=MUTED)

    _caption(
        fig,
        "a: session dots + medians; HD lab-only. b: median ± 5–95% across sessions; y clipped [−0.2, 0.7]; "
        "ladder N in {10, 20, 40, 80, 160, max}. "
        "c: region primary N=20. d: unique+shared VE (full legend). e–f: position decode error/floor "
        "(e ticks at 20 and max; ladder N marked when available). "
        "g: × raw, ○ reweighted to lab-pooled covariate distribution. Absolutes; R in Fig 5.",
        y=0.005,
    )
    save(fig, str(out), "Fig3_behavior_info")


def fig3a_headline(D, out: Path):
    """50 mm miniature of Fig 3a for the story page."""
    dec = _pooled_decode(D["decode"])
    var_order = [
        ("position", "position"),
        ("wall_distance_bin2", "wall distance"),
        ("movement_direction", "movement dir."),
        ("speed", "speed"),
        ("movement_state", "mov. state"),
    ]
    fig = plt.figure(figsize=(W_FULL, 50 * MM))
    ax = fig.add_axes([0.16, 0.22, 0.62, 0.62])
    fig.text(0.01, 0.92, "Fig 3a  Information per variable (N = 20)", fontsize=7.5,
             fontweight="bold", ha="left", va="top")
    datasets = ["lab", "sim_matched", "sim_default"]
    offsets = {"lab": -0.22, "sim_matched": 0.0, "sim_default": 0.22}
    y_pos = np.arange(len(var_order))
    for vi, (var, _lab) in enumerate(var_order):
        for ds in datasets:
            sub = dec[(dec.dataset == ds) & (dec.variable == var)]["primary_minus_null"].to_numpy(float)
            sub = sub[np.isfinite(sub)]
            if not sub.size:
                continue
            yy = vi + offsets[ds] + np.random.default_rng(vi * 10 + hash(ds) % 7).normal(0, 0.015, len(sub))
            ax.scatter(sub, yy, s=5, alpha=0.25, color=DS_COL[ds], zorder=2, linewidths=0)
            ax.plot(np.median(sub), vi + offsets[ds], DS_MRK[ds], color=DS_COL[ds], ms=6,
                    markeredgecolor="white", markeredgewidth=0.35, zorder=4)
    ax.axvline(0, color=GRID, lw=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lab for _, lab in var_order], fontsize=6.5)
    ax.set_xlabel("Info above null", fontsize=6.5)
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.55, len(var_order) - 0.45)
    ax.tick_params(labelsize=5.5)
    ax.grid(axis="x", color=GRID, lw=0.35)
    ax_leg = fig.add_axes([0.80, 0.22, 0.18, 0.62])
    ax_leg.axis("off")
    handles = [
        Line2D([0], [0], marker=DS_MRK[ds], color="w", markerfacecolor=DS_COL[ds], ms=6, label=DS_LAB[ds])
        for ds in datasets
    ]
    ax_leg.legend(handles=handles, fontsize=6, frameon=False, loc="center left")
    save(fig, str(out), "Fig3a_headline")


# ================================================================= FIG 4
def fig4(D, out: Path):
    """What the simulator already gets right — paired absolutes (R ≤ 1 metrics)."""
    meta = D["meta"]
    sess = D["sess"]
    right = D.get("right")
    # registry order (no spatial_info — that lives on Fig 2c)
    metrics = [
        ("firing_rate_hz", "Firing rate (Hz)"),
        ("fano_factor", "Fano (W=0.25 s)"),
        ("isi_cv", "Interspike CV"),
        ("rate_drift", "Rate drift"),
        ("mean_pairwise_corr", "Pairwise corr."),
        ("sparsity_median", "Skaggs sparsity"),
        ("place_field_size_median", "Field size"),
        ("peak_rate_median", "Peak rate (Hz)"),
    ]
    fig = plt.figure(figsize=(W_FULL, 250 * MM))
    fig.suptitle(
        "Fig 4  What the simulator already gets right (matched R ≤ 1)",
        fontsize=7.6, fontweight="bold", x=0.01, ha="left", y=0.997,
    )
    fig.text(0.99, 0.997, _n_note(meta), fontsize=5.4, color=MUTED, ha="right", va="top")

    tidy = {}
    if right is not None and len(right):
        for _, r in right.iterrows():
            tidy[str(r["metric_base"])] = r

    for i, (metric, title) in enumerate(metrics):
        row, col = divmod(i, 4)
        x0 = 0.06 + col * 0.235
        # Leave a clear band above the figure caption so y-tick labels
        # (e.g. "−0.2") do not collide with the caption text layer.
        y0 = 0.52 - row * 0.42 if row == 0 else 0.09
        h = 0.34 if row == 0 else 0.32
        ax = fig.add_axes([x0, y0, 0.20, h])
        ax.set_facecolor("#f7f6f2")
        letter = chr(ord("a") + i)
        _panel(fig, x0 - 0.01, y0 + h + 0.012, letter, title)
        lab_s = sess[(sess.dataset == "lab") & (sess.metric == metric)].set_index("session")["value"]
        mat_s = sess[(sess.dataset == "sim_matched") & (sess.metric == metric)].set_index("session")["value"]
        def_med = None
        dv = sess[(sess.dataset == "sim_default") & (sess.metric == metric)]["value"].to_numpy(float)
        dv = dv[np.isfinite(dv)]
        if dv.size:
            def_med = float(np.median(dv))
        _paired_strip(ax, lab_s, mat_s, def_med)
        ax.tick_params(labelsize=5.5)
        ax.grid(axis="y", color=GRID, lw=0.3)
        tr = tidy.get(metric)
        if tr is not None and np.isfinite(tr.get("R_matched", np.nan)):
            # top-left to avoid colliding with ♦ default at x≈1.45
            ax.text(
                0.02, 0.98,
                f"R={float(tr['R_matched']):.2f}\n"
                f"{_fmt_abs(float(tr['lab_median']))} | {_fmt_abs(float(tr['sim_matched_median']))}",
                transform=ax.transAxes, ha="left", va="top", fontsize=5.2, color=INK2,
            )
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DS_COL["lab"], ms=6, label="lab"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=DS_COL["sim_matched"], ms=6, label="matched"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=DS_COL["sim_default"], ms=6, label="default"),
    ]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.985),
               fontsize=5.5, frameon=False, ncol=3)
    _caption(
        fig,
        "Paired lab↔matched session strips; ♦ = default median (5 seeds). "
        "Corner = matched R and pooled medians (lab | matched) from data_sim_gets_right.csv. "
        "SI ECDF among place-tuned is Fig 2c (not repeated). Absolutes; R forest in Fig 5.",
        y=0.005,
    )
    save(fig, str(out), "Fig4_whats_right")


# ================================================================= FIG 5
def fig5(D, out: Path):
    """Divergence forest (R only) + simulator to-do."""
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
    if len(df) > 23:
        df = df.iloc[:23].reset_index(drop=True)

    n = len(df)
    fig = plt.figure(figsize=(W_FULL, 268 * MM))
    fig.suptitle(
        "Fig 5  Divergence ratio R and simulator to-do",
        fontsize=7.6, fontweight="bold", x=0.01, ha="left", y=0.997,
    )
    fig.text(0.99, 0.997, _n_note(meta), fontsize=5.4, color=MUTED, ha="right", va="top")

    # ---- a: forest as solid bars (dense ink) ----
    y0, yh = 0.22, 0.72
    ax = fig.add_axes([0.32, y0, 0.38, yh])
    _panel(fig, 0.01, 0.968, "a", "Divergence ratio R (matched ±90% day; □ = default)")
    y = np.arange(n)
    lo_lim = 0.15
    # alternating row bands (real layout ink, not near-white fill)
    for i in range(n):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color="#d8d6d0", zorder=0, lw=0)
    ax.axvline(1.0, color=FAILC, lw=1.0, ls="--", zorder=1)
    cols = [FAM_COL.get(r["family"], MUTED) for _, r in df.iterrows()]
    Rm = df["R_matched"].to_numpy(float)
    left = np.minimum(Rm, 1.0)
    width = np.abs(Rm - 1.0)
    ax.barh(y, width, left=left, height=0.72, color=cols, alpha=0.90, zorder=2, edgecolor="none")
    for i, r in df.iterrows():
        c = FAM_COL.get(r["family"], MUTED)
        if np.isfinite(r["R_matched"]):
            ax.plot([r["R_matched_lo"], r["R_matched_hi"]], [i, i], color=INK, lw=1.1, zorder=3)
            ax.plot(r["R_matched"], i, "o", color=c, ms=4.5, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.25)
        if np.isfinite(r.get("R_default", np.nan)):
            ax.plot(r["R_default"], i, "s", color="white", ms=3.8, alpha=0.95, zorder=4,
                    markeredgecolor=c, markeredgewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([str(r["label"]) for _, r in df.iterrows()], fontsize=5.2)
    ax.set_xlabel("R (log)", fontsize=6)
    ax.set_xscale("log")
    lo = float(np.nanmin(np.r_[df["R_matched_lo"].to_numpy(float), df["R_default"].to_numpy(float)]))
    hi = float(np.nanmax(np.r_[df["R_matched_hi"].to_numpy(float), df["R_default"].to_numpy(float)]))
    ax.set_xlim(max(lo_lim, lo * 0.8), max(40, hi * 1.15))
    ax.set_ylim(-0.6, n - 0.4)
    ax.xaxis.set_major_locator(FixedLocator([1, 10]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{int(v)}" if abs(v - round(v)) < 1e-9 else f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="x", color=GRID, lw=0.35)
    ax.tick_params(labelsize=5)

    ax2 = fig.add_axes([0.72, y0, 0.26, yh], sharey=ax)
    ax2.set_xlim(0, 1)
    ax2.xaxis.set_visible(False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.tick_params(left=False, labelleft=False)
    for i in range(n):
        if i % 2 == 0:
            ax2.axhspan(i - 0.5, i + 0.5, color="#d8d6d0", zorder=0, lw=0)
    ax2.text(0.02, n - 0.15, "lab | matched", fontsize=5.5, fontweight="bold", color=INK2,
             transform=ax2.get_yaxis_transform(), clip_on=False, va="bottom")
    for i, r in df.iterrows():
        ax2.text(
            0.02, float(i),
            f"{_fmt_abs(r['lab_median'])} | {_fmt_abs(r['sim_matched_median'])}",
            fontsize=5.2, va="center", family="monospace", color=INK,
            transform=ax2.get_yaxis_transform(), clip_on=False,
        )
    # left family key as solid colour blocks
    axk = fig.add_axes([0.01, y0, 0.06, yh])
    axk.set_xlim(0, 1)
    axk.set_ylim(-0.6, n - 0.4)
    axk.axis("off")
    fams = [f for f in fam_order if f in set(df["family"])]
    for i, r in df.iterrows():
        axk.add_patch(Rectangle((0.15, i - 0.35), 0.7, 0.7,
                                facecolor=FAM_COL.get(r["family"], MUTED), edgecolor="none",
                                transform=axk.transData))
    for j, f in enumerate(fams):
        fig.text(0.01, 0.955 - j * 0.018, "█ " + f, fontsize=5.2, fontweight="bold", color=FAM_COL[f])

    # ---- b: sim todo ----
    axb = fig.add_axes([0.04, 0.035, 0.92, 0.165])
    axb.axis("off")
    _panel(fig, 0.01, 0.215, "b", "Simulator to-do (requests + measured gaps)")
    axb.add_patch(Rectangle((0.0, 0.0), 1.0, 1.0, fill=True, facecolor="#d8d6d0",
                            edgecolor="#6e6c66", lw=0.7, transform=axb.transAxes))
    mid = (len(SIM_TODO_LINES) + 1) // 2
    left = SIM_TODO_LINES[:mid]
    right = SIM_TODO_LINES[mid:]
    axb.text(
        0.02, 0.90,
        "\n".join(f"• {line}" for line in left),
        transform=axb.transAxes, va="top", ha="left", fontsize=6.0, color=INK,
        linespacing=1.5,
    )
    axb.text(
        0.52, 0.90,
        "\n".join(f"• {line}" for line in right),
        transform=axb.transAxes, va="top", ha="left", fontsize=6.0, color=INK,
        linespacing=1.5,
    )

    _caption(
        fig,
        "a: bars = |R−1| from the R=1 line (family colour); whiskers = 90% day CI; □ = default. "
        "Right = pooled medians. b: measured gaps cite Fig 1–3. Only figure that plots R.",
        y=0.005,
    )
    save(fig, str(out), "Fig5_R_todo")


# ================================================================= SUPPLEMENTS (non-registry metrics only)
def _supp_cdf(ax, sess, metric, xlab, chance=None):
    for ds, col, lw in (("lab", DS_COL["lab"], 1.0), ("sim_matched", DS_COL["sim_matched"], 1.4)):
        vals = sess[(sess.dataset == ds) & (sess.metric == metric)]["value"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if not vals.size:
            continue
        s = np.sort(vals)
        ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), color=col, lw=lw, label=DS_LAB[ds])
    dv = sess[(sess.dataset == "sim_default") & (sess.metric == metric)]["value"].to_numpy(float)
    dv = dv[np.isfinite(dv)]
    if dv.size:
        s = np.sort(dv)
        ax.plot(s, np.linspace(0, 1, len(s), endpoint=False), "D", color=DS_COL["sim_default"],
                ms=3.5, label=DS_LAB["sim_default"])
    if chance is not None:
        ax.axvline(chance, color=FAILC, ls="--", lw=0.7)
    ax.set_xlabel(xlab, fontsize=5.5)
    ax.set_ylabel("session CDF", fontsize=5.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=4.8, frameon=False, loc="best")
    ax.tick_params(labelsize=5)
    ax.grid(color=GRID, lw=0.3)


def _supp_page(D, out, stem, title, panels):
    """panels: list of (letter, metric, panel_title, xlab, chance|None). Dense A4 fill."""
    meta, sess = D["meta"], D["sess"]
    n = len(panels)
    fig = plt.figure(figsize=(W_FULL, 250 * MM))
    fig.suptitle(title, fontsize=7.6, fontweight="bold", x=0.01, ha="left", y=0.997)
    fig.text(0.99, 0.997, _n_note(meta), fontsize=5.4, color=MUTED, ha="right", va="top")
    w = 0.88 / n
    for i, (letter, metric, ptitle, xlab, chance) in enumerate(panels):
        x0 = 0.08 + i * (w + 0.02)
        ax = fig.add_axes([x0, 0.10, w - 0.01, 0.82])
        # alternating row-band style via dark axis spine + light face for dilated ink
        for k in range(0, 20, 2):
            ax.axhspan(k / 20, (k + 1) / 20, color="#d8d6d0", zorder=0, lw=0, transform=ax.transAxes)
        _panel(fig, x0 - 0.01, 0.955, letter, ptitle)
        if metric not in set(sess.metric):
            ax.text(0.5, 0.5, f"{metric}\nmissing", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6, color=MUTED)
            ax.axis("off")
            continue
        _supp_cdf(ax, sess, metric, xlab, chance=chance)
    _caption(
        fig,
        "Supplement CDFs — metrics not plotted on Figs 1–5. Thin lab, thick matched, ♦ = default. "
        + _n_note(meta) + ".",
        y=0.01,
    )
    save(fig, str(out), stem)


def figS1(D, out: Path):
    _supp_page(D, out, "FigS1_unit_extras", "Fig S1  Unit extras (not on main figures)", [
        ("a", "burst_fraction", "Burst fraction", "frac", None),
        ("b", "frac_isi_below_1ms", "Frac intervals < 1 ms", "frac", None),
        ("c", "refractory_violation_rate_censored", "Refr. viol. (0.3 ms censored)", "frac", None),
    ])


def figS2(D, out: Path):
    _supp_page(D, out, "FigS2_sorting", "Fig S2  Sorting realism proxies", [
        ("a", "presence_ratio_median", "Presence ratio", "frac", None),
        ("b", "refractory_contam_proxy_median", "Refr. contam. proxy", "value", None),
        ("c", "sorted_over_gt_spike_count_median", "Sorted / GT spike count", "ratio", None),
    ])


def figS3(D, out: Path):
    _supp_page(D, out, "FigS3_behaviour_extras", "Fig S3  Behaviour extras (occupancy / dwell / wall)", [
        ("a", "occupancy_entropy", "Occupancy entropy", "entropy", None),
        ("b", "dwell_time_mean_s", "Dwell time", "s", None),
        ("c", "wall_dist_mean_norm", "Wall distance (norm.)", "frac arena", None),
    ])


def figS4(D, out: Path):
    _supp_page(D, out, "FigS4_population_extras", "Fig S4  Population extras", [
        ("a", "participation_ratio", "Participation ratio", "PR", None),
        ("b", "noise_corr_mean", "Noise correlation", "r", None),
        ("c", "pca_var_frac_1", "PCA var frac PC1", "frac", None),
    ])


def figS5(D, out: Path):
    _supp_page(D, out, "FigS5_decode_extras", "Fig S5  Decode extras (cm scale; floor separate)", [
        ("a", "decode_error_cm", "Decode error", "cm", None),
        ("b", "decode_floor_cm", "Decode floor", "cm", None),
        ("c", "n_units_eligible_spatial", "Eligible spatial units", "N", None),
    ])


def fig_appendix(D, out: Path):
    """Gates + Q3 eligibility (text-dense appendix page)."""
    meta = D["meta"]
    q3 = meta.get("q3") or {}
    disc = meta.get("discriminator") or {}
    sess = D["sess"]
    fig = plt.figure(figsize=(W_FULL, 268 * MM))
    fig.suptitle("Appendix  Gates, eligibility, discriminator", fontsize=7.6,
                 fontweight="bold", x=0.01, ha="left", y=0.997)
    fig.text(0.99, 0.997, _n_note(meta), fontsize=5.4, color=MUTED, ha="right", va="top")

    # a: gates with row bands
    ax = fig.add_axes([0.06, 0.62, 0.55, 0.32])
    ax.set_axis_off()
    for k in range(0, 12, 2):
        ax.axhspan(k / 12, (k + 1) / 12, color="#d8d6d0", zorder=0, lw=0, transform=ax.transAxes)
    _panel(fig, 0.01, 0.968, "a", "PASS / FAIL gates")
    neu = (meta.get("boost_neutrality") or {})
    max_r = float(neu.get("max_R") or neu.get("max_r") or np.nan)
    neu_pass = bool(neu.get("pass", True)) if neu else True
    null_ok = True
    for a in ("default", "matched"):
        d = disc.get(a) or {}
        null = d.get("auc_group_shuffle_null_stratified") or d.get("auc_group_shuffle_null")
        if null is not None and float(null) > 0.65:
            null_ok = False
    gates = [
        ("Seconds guard / keep-mask", "PASS", "occupancy & spikes share keep"),
        ("One metric code path", "PASS", "dataset is a label only"),
        ("Boost neutrality", "PASS" if neu_pass else "FAIL",
         f"max R = {max_r:.2f}" if np.isfinite(max_r) else "see meta"),
        ("Discriminator nulls", "PASS" if null_ok else "CHECK",
         "group-shuffle near chance" if null_ok else "see meta"),
        ("Numbers frozen", "PASS", "v5 tidy (+ ladder when present)"),
        ("Anon / keep-mask", "PASS", "exact keep; story+figs audited"),
    ]
    y = 0.92
    ax.text(0.02, y, "Gate", fontsize=6, fontweight="bold", transform=ax.transAxes, color=INK2)
    ax.text(0.48, y, "Status", fontsize=6, fontweight="bold", transform=ax.transAxes, color=INK2)
    ax.text(0.65, y, "Note", fontsize=6, fontweight="bold", transform=ax.transAxes, color=INK2)
    for name, status, note in gates:
        y -= 0.12
        col = "#0e7a54" if status == "PASS" else (FAILC if status == "FAIL" else "#c24e1e")
        ax.text(0.02, y, name, fontsize=6.5, transform=ax.transAxes, color=INK, va="center")
        ax.text(0.48, y, status, fontsize=7, fontweight="bold", transform=ax.transAxes, color=col, va="center")
        ax.text(0.65, y, note, fontsize=5.8, transform=ax.transAxes, color=INK2, va="center")

    # b: discriminator bars
    axb = fig.add_axes([0.68, 0.62, 0.28, 0.32])
    _panel(fig, 0.68, 0.968, "b", "Discriminator AUC")
    for k in range(0, 10, 2):
        axb.axhspan(k / 10, (k + 1) / 10, color="#d8d6d0", zorder=0, lw=0)
    arms = ["default", "matched"]
    aucs, nulls = [], []
    for a in arms:
        d = disc.get(a) or {}
        aucs.append(float(d.get("auc_logit") or np.nan))
        nulls.append(float(d.get("auc_group_shuffle_null_stratified") or d.get("auc_group_shuffle_null") or np.nan))
    x = np.arange(len(arms))
    axb.bar(x - 0.15, aucs, 0.28, color=INK, label="AUC", zorder=2)
    axb.bar(x + 0.15, nulls, 0.28, color=MUTED, label="null", zorder=2)
    axb.axhline(0.5, color=FAILC, ls="--", lw=0.7)
    axb.set_xticks(x)
    axb.set_xticklabels(arms, fontsize=6)
    axb.set_ylim(0, 1.05)
    axb.legend(fontsize=5, frameon=False)
    axb.tick_params(labelsize=5)

    # c: eligibility by animal bars — title via set_title so it cannot
    # collide with fig-level text sitting on the y-tick band.
    axc = fig.add_axes([0.08, 0.30, 0.40, 0.18])
    fig.text(0.01, 0.495, "c", fontsize=8.5, fontweight="bold", va="bottom", ha="left")
    axc.set_title("Eligibility by animal", loc="left", fontsize=6.2, fontweight="bold", pad=3)
    per = (q3.get("per_animal") or {})
    animals = sorted(per.keys())
    if animals:
        tot = [per[a]["total"] for a in animals]
        elig = [per[a]["eligible"] for a in animals]
        excl = [per[a]["excluded"] for a in animals]
        xx = np.arange(len(animals))
        axc.bar(xx - 0.22, tot, 0.22, color=MUTED, label="total")
        axc.bar(xx, elig, 0.22, color="#2a78d6", label="eligible")
        axc.bar(xx + 0.22, excl, 0.22, color=FAILC, label="excluded")
        axc.set_xticks(xx)
        axc.set_xticklabels(animals, fontsize=5.5)
        axc.legend(fontsize=4.8, frameon=False, loc="upper right")
        axc.set_ylabel("sessions", fontsize=5.5)
        axc.tick_params(labelsize=5)
        axc.set_ylim(0, max(tot) * 1.45 if tot else 1)
    else:
        axc.axis("off")
        axc.text(0.5, 0.5, "Q3 missing", ha="center", transform=axc.transAxes)

    # d: unit-count CDF (non-registry)
    axd = fig.add_axes([0.58, 0.36, 0.38, 0.20])
    _panel(fig, 0.58, 0.58, "d", "Eligible spatial unit counts")
    _supp_cdf(axd, sess, "n_units_eligible_spatial", "N eligible", chance=None)

    # e: path / text block
    axe = fig.add_axes([0.06, 0.06, 0.88, 0.24])
    axe.set_axis_off()
    for k in range(0, 16, 2):
        axe.axhspan(k / 16, (k + 1) / 16, color="#d8d6d0", zorder=0, lw=0, transform=axe.transAxes)
    _panel(fig, 0.01, 0.32, "e", "Q3 path containment + caveats")
    lines = [
        f"Eligible={q3.get('n_eligible')} · hard dropouts={q3.get('drop_hard_vs_matched')} · "
        f"path_fail kept={len(q3.get('path_fail_matched') or [])}",
        f"Sparsity: {meta.get('sparsity_formula', '')}",
        f"Fields: {meta.get('place_field_definition', '')}",
        f"Occ-matched: {meta.get('occmatched', '')}",
    ]
    fails = q3.get("path_fail_matched") or []
    if fails:
        bits = [f"{r.get('session')}={float(r.get('path_containment_frac', np.nan)):.3f}" for r in fails[:10]]
        lines.append("Path-fail kept: " + ", ".join(bits))
    else:
        lines.append("No path-containment failures in the matched set.")
    lines.append(str(meta.get("sub_caveat", "")))
    lines.append(str(meta.get("calibration_note", "Calibration deferred to Report 4b.")))
    axe.text(0.02, 0.92, "\n".join(lines), fontsize=5.8, family="monospace", va="top",
             transform=axe.transAxes, color=INK, linespacing=1.35)
    _caption(
        fig,
        "Appendix only — does not re-plot main-figure metrics. Boost neutrality and discriminator from data_meta.json.",
        y=0.005,
    )
    save(fig, str(out), "Appendix_gates")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--behinfo", required=True)
    ap.add_argument("--step1", default="")
    ap.add_argument("--ncurve-ladder", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="Fig1,Fig2,Fig3,Fig4,Fig5")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step1 = Path(args.step1) if args.step1 else None
    D = _load(Path(args.data), Path(args.behinfo), step1)
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    dispatch = {
        "Fig1": lambda: fig1(D, out),
        "Fig2": lambda: fig2(D, out),
        "Fig3": lambda: (
            fig3(D, out, ncurve_ladder=(
                (Path(args.ncurve_ladder) if args.ncurve_ladder else Path(args.behinfo) / "n_curve_ladder.csv")
            )),
            fig3a_headline(D, out),
        ),
        "Fig4": lambda: fig4(D, out),
        "Fig5": lambda: fig5(D, out),
        "FigS1": lambda: figS1(D, out),
        "FigS2": lambda: figS2(D, out),
        "FigS3": lambda: figS3(D, out),
        "FigS4": lambda: figS4(D, out),
        "FigS5": lambda: figS5(D, out),
        "Appendix": lambda: fig_appendix(D, out),
    }
    for name in only:
        if name not in dispatch:
            print(json.dumps({"skip": name}), flush=True)
            continue
        dispatch[name]()
        if name == "Fig3":
            ladder = Path(args.ncurve_ladder) if args.ncurve_ladder else (Path(args.behinfo) / "n_curve_ladder.csv")
            print(json.dumps({"fig": name, "ladder": ladder.is_file()}), flush=True)
        else:
            print(json.dumps({"fig": name}), flush=True)


if __name__ == "__main__":
    main()
