#!/usr/bin/env python3
"""Report 4a figures Q1–Q16 (calibration Q13–Q15 marked 4b).

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
from matplotlib.patches import FancyBboxPatch

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
    save,
)

FAM_COL = {
    "units": "#2a78d6",
    "sorting": "#eb6834",
    "spatial": "#1baf7a",
    "rhythm": "#8b5cf6",
    "population": "#0e7a54",
    "behaviour": "#c24e1e",
    "decoding": "#52514e",
    "other": "#a3a19b",
}
LAB_LW, SIM_LW = 0.7, 1.6


def _load(data: str) -> dict:
    D = {
        "div": pd.read_csv(os.path.join(data, "data_divergence.csv")),
        "sess": pd.read_csv(os.path.join(data, "data_session_metrics.csv")),
        "disc": pd.read_csv(os.path.join(data, "data_discriminator.csv")),
        "imp": pd.read_csv(os.path.join(data, "data_discriminator_importance.csv")),
        "abl": pd.read_csv(os.path.join(data, "data_discriminator_ablation.csv")),
        "find": pd.read_csv(os.path.join(data, "data_findings.csv")),
        "meta": json.loads(open(os.path.join(data, "data_meta.json")).read()),
    }
    return D


def _takeaway(fig, title: str, n_note: str):
    fig.suptitle(title, fontsize=8, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.99, 0.995, n_note, fontsize=6, color=MUTED, ha="right", va="top")


def _caption(fig, text: str, y: float = 0.02):
    fig.text(0.01, y, textwrap.fill(text, 130), fontsize=5.8, color=INK2, va="bottom", ha="left")


def fig_q1(D, out):
    """Design: matching protocol, R definition, thigmotaxis, audits."""
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    _takeaway(
        fig,
        "Q1  Matching protocol and divergence ratio (thigmotaxis held at 0.8)",
        f"n_lab={meta['n_lab_sessions']}  n_matched={meta['n_matched_sessions']}  n_default={meta['n_default_seeds']}",
    )
    ax = fig.add_axes([0.04, 0.12, 0.92, 0.78])
    ax.axis("off")
    boxes = [
        (0.02, 0.55, 0.30, 0.40, "A  Arms",
         f"Lab: {meta['n_lab_sessions']} room-A sessions\n"
         f"({meta['n_animals']} animals)\n"
         f"Default sim: {meta['n_default_seeds']} seeds\n"
         f"Matched sim: {meta['n_matched_sessions']} sessions\n"
         f"Region set: {meta['region_set']}"),
        (0.35, 0.55, 0.30, 0.40, "B  Chunking / N",
         f"Chunks: {meta['chunk_len_s']:.0f} s\n"
         f"Matched-N grid: {meta['matched_Ns']}\n"
         f"Primary N: {meta['matched_N_preferred']}\n"
         f"Subsamples: {meta['n_subsamples']}\n"
         f"Pos smooth σ: {meta['pos_smooth_sigma_s']} s"),
        (0.68, 0.55, 0.30, 0.40, "C  Divergence R",
         "R = d_sim / d_lab\n"
         "d = median |Δ| / MAD_lab\n"
         "Bootstrap by recording_day\n"
         "90% interval shown\n"
         "R=1: within lab var."),
        (0.02, 0.08, 0.45, 0.40, "D  Behaviour config",
         f"thigmotaxis = {meta['thigmotaxis']} (not fitted)\n"
         f"Speed / turning knobs: not public (DF6)\n"
         f"Behaviour divergences → simulator to-do\n"
         f"as structural requests."),
        (0.52, 0.08, 0.46, 0.40, "E  Refractory audit assumption",
         f"Lab-equivalent dup removal:\n"
         f"ISI floor ≈ {meta['censor_isi_ms']:.1f} ms\n"
         f"(lab median min ISI).\n"
         f"Public refractory param in hippo_sim:\n"
         f"{'yes' if meta['generator_has_public_refractory_param'] else 'none found'}."),
    ]
    for x, y, w, h, title, body in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#f7f6f2", edgecolor=GRID, transform=ax.transAxes, clip_on=False,
            )
        )
        ax.text(x + 0.015, y + h - 0.04, title, transform=ax.transAxes, fontsize=7, fontweight="bold", va="top")
        ax.text(x + 0.015, y + h - 0.10, body, transform=ax.transAxes, fontsize=6.2, va="top", color=INK2, family="monospace")
    _caption(
        fig,
        "Boxes summarise the frozen protocol. Ranking excludes tracking_jitter_rest_cm (Q8 diagnostic) and "
        "frac_isi_below_1ms_gt (sim-only structural). Calibration figures deferred to Report 4b.",
    )
    save(fig, out, "Q1_design")


def fig_q2(D, out, arm: str = "matched"):
    """Headline divergence overview."""
    div = D["div"]
    sub = div[(div.arm == arm) & (div.in_ranking)].copy()
    # prefer @N20 / plain already in ranking extract; drop duplicate bases keeping first by |R|
    sub = sub.sort_values("R", ascending=False)
    seen = set()
    keep = []
    for _, r in sub.iterrows():
        b = r["metric_base"]
        if b in seen:
            continue
        seen.add(b)
        keep.append(r)
    sub = pd.DataFrame(keep).head(25).iloc[::-1]
    meta = D["meta"]
    n_lab = int(meta["n_lab_sessions"])
    n_sim = int(meta["n_matched_sessions"] if arm == "matched" else meta["n_default_seeds"])
    fig = plt.figure(figsize=(W_FULL, 140 * MM))
    _takeaway(
        fig,
        f"Q2  Divergence overview ({arm}): refractory and turning dominate",
        f"n_lab={n_lab}  n_sim={n_sim}",
    )
    ax = fig.add_axes([0.38, 0.10, 0.58, 0.82])
    y = np.arange(len(sub))
    for i, (_, r) in enumerate(sub.iterrows()):
        c = FAM_COL.get(r["family"], MUTED)
        ax.plot([r["R_lo90"], r["R_hi90"]], [i, i], color=c, lw=1.2, solid_capstyle="round")
        ax.plot(r["R"], i, "o", color=c, ms=4.5, zorder=3)
    ax.axvline(1.0, color=FAILC, lw=0.9, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(sub["metric"].tolist(), fontsize=5.8)
    ax.set_xlabel("Divergence ratio R (90% bootstrap interval)")
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, lw=0.4)
    # legend
    handles = [
        Line2D([0], [0], color=FAM_COL[f], lw=2, label=f)
        for f in ("units", "sorting", "spatial", "behaviour", "population", "decoding", "rhythm")
        if f in set(sub["family"])
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6)
    _caption(
        fig,
        "Dots: point estimate of R = d_sim/d_lab. Horizontal lines: 90% recording_day bootstrap intervals. "
        "Dashed red: R=1 (sim within lab variability). Colour = metric family. Ranking excludes jitter and GT-only ISI.",
    )
    save(fig, out, "Q2_divergence")


def _rain(ax, sess: pd.DataFrame, metric: str, datasets=("lab", "sim_matched", "sim_default")):
    """Session medians as thin (lab) / thick (sim) vertical ticks + median line."""
    xpos = {d: i for i, d in enumerate(datasets)}
    for ds in datasets:
        vals = sess[(sess.dataset == ds) & (sess.metric == metric)]["value"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        x0 = xpos[ds]
        lw = LAB_LW if ds == "lab" else SIM_LW
        jitter = (np.linspace(-0.18, 0.18, vals.size) if vals.size < 80
                  else np.random.default_rng(0).uniform(-0.2, 0.2, vals.size))
        ax.vlines(x0 + jitter, vals * 0, vals, colors=INK2 if ds == "lab" else INK, lw=lw, alpha=0.55)
        ax.plot(x0, np.median(vals), "o", color=FAILC if ds.startswith("sim") else INK, ms=5, zorder=3)
    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(["lab", "matched", "default"], fontsize=6)
    ax.set_xlim(-0.5, len(datasets) - 0.5)


def fig_q3(D, out):
    """Units: rate, ISI, burst, refractory raw+censored, Fano, drift."""
    sess = D["sess"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 145 * MM))
    _takeaway(
        fig,
        "Q3  Units: refractory remains elevated after lab-equivalent 0.3 ms censor",
        f"n_lab={meta['n_lab_sessions']}  censor={meta['censor_isi_ms']:.1f} ms",
    )
    panels = [
        ("A", "Firing rate (Hz)", "firing_rate_hz"),
        ("B", "ISI CV", "isi_cv"),
        ("C", "Burst frac (ISI<10 ms)", "burst_fraction"),
        ("D", "Refractory raw (<1.5 ms)", "refractory_violation_rate"),
        ("E", "Refractory after 0.3 ms censor", "refractory_violation_rate_censored"),
        ("F", "Frac ISI < 1 ms (sorted)", "frac_isi_below_1ms"),
    ]
    for i, (letter, title, metric) in enumerate(panels):
        ax = fig.add_axes([0.07 + (i % 3) * 0.31, 0.55 - (i // 3) * 0.40, 0.26, 0.32])
        head(ax, letter, title)
        _rain(ax, sess, metric)
        ax.set_ylabel("session median" if i % 3 == 0 else "")
    # GT note strip
    gt = sess[sess.metric == "frac_isi_below_1ms_gt"]
    if not gt.empty:
        meds = gt.groupby("dataset")["value"].median()
        fig.text(
            0.01, 0.08,
            f"Structural (generator): sim GT frac ISI<1 ms median = "
            + ", ".join(f"{k} {v:.4f}" for k, v in meds.items())
            + f". No public refractory/min_isi parameter in hippo_sim (Poisson per bin).",
            fontsize=6, color=INK2,
        )
    _caption(
        fig,
        "Each thin vertical tick is one lab session median; thick ticks are sim sessions/seeds. "
        "Red circle: group median. Panel E applies lab-equivalent duplicate-spike removal (ISI floor 0.3 ms) to sim sorted spikes before scoring.",
        y=0.015,
    )
    save(fig, out, "Q3_units")


def fig_q4(D, out):
    """Sorting realism."""
    sess = D["sess"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 100 * MM))
    _takeaway(
        fig,
        "Q4  Sorting realism: contamination proxy tracks refractory; presence closer",
        f"n_lab={meta['n_lab_sessions']}",
    )
    panels = [
        ("A", "Refractory contam. proxy", "refractory_contam_proxy_median"),
        ("B", "Presence ratio", "presence_ratio_median"),
        ("C", "Sorted / GT spike count", "sorted_over_gt_spike_count_median"),
    ]
    for i, (letter, title, metric) in enumerate(panels):
        ax = fig.add_axes([0.08 + i * 0.31, 0.18, 0.26, 0.68])
        head(ax, letter, title)
        _rain(ax, sess, metric)
        if i == 0:
            ax.set_ylabel("session median")
    _caption(
        fig,
        "Lab has no Phy amplitude cutoffs; contamination uses refractory proxy. "
        "Panel C is sim-only (sorted vs ground-truth spike counts). Dots = group medians; ticks = sessions.",
    )
    save(fig, out, "Q4_sorting")


def fig_placeholder(out, name: str, title: str, note: str):
    fig = plt.figure(figsize=(W_FULL, 60 * MM))
    _takeaway(fig, title, "deferred")
    ax = fig.add_axes([0.08, 0.2, 0.84, 0.55])
    ax.axis("off")
    ax.text(0.5, 0.5, note, ha="center", va="center", fontsize=9, color=MUTED, transform=ax.transAxes)
    save(fig, out, name)


def fig_q5(D, out):
    sess = D["sess"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 100 * MM))
    _takeaway(fig, "Q5  Spatial coding: coverage and sparsity diverge; info closer", f"n_lab={meta['n_lab_sessions']}  N=20")
    panels = [
        ("A", "Spatial info (bits/spk)", "spatial_info_median"),
        ("B", "Sparsity", "sparsity_median"),
        ("C", "Coverage (bins)", "coverage_bins_mean"),
    ]
    for i, (letter, title, metric) in enumerate(panels):
        ax = fig.add_axes([0.08 + i * 0.31, 0.18, 0.26, 0.68])
        head(ax, letter, title)
        _rain(ax, sess, metric)
        if i == 0:
            ax.set_ylabel("session median")
    _caption(fig, "Matched-N=20 where applicable. Thin=lab sessions; thick=sim; red circle=group median.")
    save(fig, out, "Q5_spatial")


def fig_q6(D, out):
    sess = D["sess"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 80 * MM))
    _takeaway(fig, "Q6  Rhythm: spike-based theta modulation", f"n_lab={meta['n_lab_sessions']}")
    ax = fig.add_axes([0.12, 0.18, 0.5, 0.68])
    head(ax, "A", "Theta modulation index")
    _rain(ax, sess, "theta_mod_index_median")
    ax.set_ylabel("session median")
    _caption(fig, "If the simulator lacks theta structure, that is a finding (not suppressed).")
    save(fig, out, "Q6_rhythm")


def fig_q7(D, out):
    sess = D["sess"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 80 * MM))
    _takeaway(fig, "Q7  Population: participation ratio at matched N=20", f"n_lab={meta['n_lab_sessions']}")
    ax = fig.add_axes([0.12, 0.18, 0.5, 0.68])
    head(ax, "A", "Participation ratio")
    _rain(ax, sess, "participation_ratio")
    ax.set_ylabel("session median")
    _caption(fig, "N=20 preferred. Thin lab / thick sim ticks; red = group median.")
    save(fig, out, "Q7_population")


def fig_q8(D, out):
    sess = D["sess"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 120 * MM))
    _takeaway(
        fig,
        f"Q8  Behaviour: speed/turning structural (DF6); jitter diagnostic only; thig={meta['thigmotaxis']}",
        f"n_lab={meta['n_lab_sessions']}",
    )
    panels = [
        ("A", "Speed median", "speed_median"),
        ("B", "Turning angle std", "turning_angle_std"),
        ("C", "Occupancy entropy", "occupancy_entropy"),
        ("D", "Wall dist (norm)", "wall_dist_mean_norm"),
        ("E", "Dwell time (s)", "dwell_time_mean_s"),
        ("F", "Tracking jitter (diag.)", "tracking_jitter_rest_cm"),
    ]
    for i, (letter, title, metric) in enumerate(panels):
        ax = fig.add_axes([0.07 + (i % 3) * 0.31, 0.52 - (i // 3) * 0.40, 0.26, 0.32])
        head(ax, letter, title)
        _rain(ax, sess, metric)
    _caption(
        fig,
        "Panel F (tracking_jitter_rest_cm) is excluded from R ranking — measurement diagnostic only. "
        "Speed/turning divergences are structural: speed_* / rotational_velocity_* are not public (DF6).",
    )
    save(fig, out, "Q8_behaviour")


def fig_q9(D, out):
    sess = D["sess"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 80 * MM))
    _takeaway(fig, "Q9  Decoding: error/floor at N=20, W=0.25 (frozen)", f"n_lab={meta['n_lab_sessions']}")
    ax = fig.add_axes([0.12, 0.18, 0.5, 0.68])
    head(ax, "A", "Decode error / floor")
    _rain(ax, sess, "decode_error_over_floor")
    ax.set_ylabel("session median")
    _caption(fig, "Ridge on raw causal counts; W=0.25 frozen. Train-on-sim/test-on-lab not possible (no unit correspondence).")
    save(fig, out, "Q9_decoding")


def fig_q10(D, out):
    disc = D["disc"]
    imp = D["imp"]
    abl = D["abl"]
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 110 * MM))
    _takeaway(fig, "Q10  Discriminator: AUC≈1; both group-shuffle nulls near chance", f"n_lab={meta['n_lab_sessions']}")
    ax = fig.add_axes([0.08, 0.55, 0.35, 0.35])
    head(ax, "A", "AUC vs nulls")
    arms = list(disc["arm"])
    x = np.arange(len(arms))
    ax.bar(x - 0.2, disc["auc_logit"], width=0.2, color=INK, label="AUC logit")
    ax.bar(x, disc["auc_group_shuffle_null"], width=0.2, color=MUTED, label="null GroupKFold")
    ax.bar(x + 0.2, disc["auc_group_shuffle_null_stratified"], width=0.2, color=FAM_COL["units"], label="null StratGroupKF")
    ax.axhline(0.5, color=FAILC, ls="--", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(arms)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=5.5, loc="lower right")

    ax2 = fig.add_axes([0.52, 0.55, 0.42, 0.35])
    head(ax2, "B", "Giveaways (|coef| / drop)")
    sub = imp[imp.arm == "matched"].head(8).iloc[::-1]
    ax2.barh(np.arange(len(sub)), sub["abs_coef"].fillna(0), color=FAM_COL["sorting"])
    ax2.set_yticks(np.arange(len(sub)))
    ax2.set_yticklabels(sub["feature"], fontsize=6)

    ax3 = fig.add_axes([0.12, 0.12, 0.55, 0.32])
    head(ax3, "C", "Ablation (matched)")
    a = abl[abl.arm == "matched"]
    ax3.plot(a["n_features"], a["auc"], "-o", color=INK, ms=4)
    ax3.axhline(0.6, color=FAILC, ls="--", lw=0.8)
    ax3.set_xlabel("n features remaining")
    ax3.set_ylabel("CV AUC")
    _caption(
        fig,
        "Both nulls shuffle labels at group level. Null A=GroupKFold; Null B=StratifiedGroupKFold. "
        "Groups are dataset::session so matched lab/sim sharing a session name are not merged.",
    )
    save(fig, out, "Q10_discriminator")


def fig_q11(D, out):
    div = D["div"]
    sub = div[(div.arm == "matched") & (div.in_ranking)].copy()
    sub = sub.sort_values("R", ascending=False)
    seen = set()
    rows = []
    for _, r in sub.iterrows():
        if r["metric_base"] in seen:
            continue
        seen.add(r["metric_base"])
        rows.append(r)
    sub = pd.DataFrame(rows).head(20)
    fig = plt.figure(figsize=(W_FULL, 100 * MM))
    _takeaway(fig, "Q11  Variability: d_simsim vs d_lab (too-regular when d_simsim < 0.5 d_lab)", f"n={len(sub)} metrics")
    ax = fig.add_axes([0.12, 0.15, 0.78, 0.72])
    head(ax, "A", "d_simsim / d_lab")
    ratio = sub["d_simsim"] / sub["d_lab"]
    colors = [FAILC if bool(t) else FAM_COL.get(f, MUTED) for f, t in zip(sub["family"], sub["too_regular"])]
    ax.scatter(sub["d_lab"], sub["d_simsim"], c=colors, s=28)
    mx = max(float(np.nanmax(sub["d_lab"])), float(np.nanmax(sub["d_simsim"])), 1.0)
    ax.plot([0, mx], [0, mx], "--", color=MUTED, lw=0.8)
    ax.plot([0, mx], [0, 0.5 * mx], ":", color=FAILC, lw=0.8)
    ax.set_xlabel("d_lab")
    ax.set_ylabel("d_simsim")
    _caption(fig, "Points below the dotted line (d_simsim = 0.5 d_lab) are flagged too_regular. d_simsim is MAD-scaled distance, not a ratio to d_lab.")
    save(fig, out, "Q11_variability")


def fig_q12(D, out):
    meta = D["meta"]
    fig = plt.figure(figsize=(W_FULL, 70 * MM))
    _takeaway(fig, "Q12  Confounds / protocol notes", f"region={meta['region_set']}")
    ax = fig.add_axes([0.08, 0.15, 0.84, 0.7])
    ax.axis("off")
    ax.text(
        0.0, 0.9,
        "Primary region set {MEC, SUB}. Ratemap gate on for lab.\n"
        "Ranking excludes: " + ", ".join(meta["rank_exclude"]) + ".\n"
        f"Decoding: {meta['decoding']}.\n"
        "Report 3 §9 mismatch table: region-matched vs all-units comparison not re-run in 4a draft;\n"
        "primary set used throughout. Calibration / LOAO deferred to 4b.",
        fontsize=7, va="top", family="monospace", color=INK2,
    )
    save(fig, out, "Q12_confounds")


def fig_q16(D, out):
    meta = D["meta"]
    find = D["find"]
    fig = plt.figure(figsize=(W_FULL, 80 * MM))
    _takeaway(fig, "Q16  Validity / protocol audits (4a)", f"n_lab={meta['n_lab_sessions']}")
    ax = fig.add_axes([0.08, 0.15, 0.84, 0.7])
    ax.axis("off")
    n_flag = int(find[(find.arm == "matched") & (find.R > 1)].shape[0])
    ax.text(
        0.0, 0.95,
        f"Chunks={meta['chunk_len_s']} s; matched-N preferred={meta['matched_N_preferred']}; "
        f"pos smooth σ={meta['pos_smooth_sigma_s']} s.\n"
        f"thigmotaxis={meta['thigmotaxis']} fitted={meta['thigmotaxis_fitted']}.\n"
        f"Censor ISI={meta['censor_isi_ms']:.1f} ms (assumption: lab median min ISI).\n"
        f"Public refractory param: {meta['generator_has_public_refractory_param']}.\n"
        f"Matched metrics with R>1 in top findings table: {n_flag}.\n"
        "Proxy fidelity FAIL → calibration figures Q13–Q15 marked 4b.",
        fontsize=7, va="top", family="monospace", color=INK2,
    )
    save(fig, out, "Q16_validity")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma list e.g. Q1,Q2,Q3,Q4")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    D = _load(args.data)
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    jobs = [
        ("Q1", lambda: fig_q1(D, args.out)),
        ("Q2", lambda: fig_q2(D, args.out, arm="matched")),
        ("Q3", lambda: fig_q3(D, args.out)),
        ("Q4", lambda: fig_q4(D, args.out)),
        ("Q5", lambda: fig_q5(D, args.out)),
        ("Q6", lambda: fig_q6(D, args.out)),
        ("Q7", lambda: fig_q7(D, args.out)),
        ("Q8", lambda: fig_q8(D, args.out)),
        ("Q9", lambda: fig_q9(D, args.out)),
        ("Q10", lambda: fig_q10(D, args.out)),
        ("Q11", lambda: fig_q11(D, args.out)),
        ("Q12", lambda: fig_q12(D, args.out)),
        ("Q13", lambda: fig_placeholder(args.out, "Q13_calib_4b", "Q13  Calibration (4b)", "Deferred to Report 4b — proxy fidelity failed.")),
        ("Q14", lambda: fig_placeholder(args.out, "Q14_calib_4b", "Q14  Calibration sweep (4b)", "Deferred to Report 4b.")),
        ("Q15", lambda: fig_placeholder(args.out, "Q15_calib_4b", "Q15  Calibration validation (4b)", "Deferred to Report 4b.")),
        ("Q16", lambda: fig_q16(D, args.out)),
    ]
    for name, fn in jobs:
        if only and name not in only:
            continue
        fn()
        print(json.dumps({"fig": name, "out": args.out}), flush=True)


if __name__ == "__main__":
    main()
