#!/usr/bin/env python3
"""Report 4a narrative figures (v6 rebuild) — numbers frozen at v5 tidy.

Fig 1 = behavioural information claim.
Fig 2 = tuning / maps / drift claim.
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
    head_fig,
    save as _save,
)

DS_COL = {"lab": "#1a1a1a", "sim_matched": "#0b5fff", "sim_default": "#8f8d87"}
DS_LAB = {"lab": "lab", "sim_matched": "matched", "sim_default": "default"}
DS_MRK = {"lab": "o", "sim_matched": "s", "sim_default": "D"}


def save(fig, out, name):
    _save(fig, out, name)


def _caption(fig, text, y=0.012):
    fig.text(0.01, y, textwrap.fill(text, 125), fontsize=8.0, color=INK2, va="bottom", ha="left")


def _load(data: Path, behinfo: Path) -> dict:
    D = {
        "meta": json.loads((data / "data_meta.json").read_text()),
        "fig3": pd.read_csv(data / "data_fig3_divergence.csv"),
        "sess": pd.read_csv(data / "data_session_metrics.csv"),
        "behinfo": behinfo,
    }
    for name, key in (
        ("data_rate_maps.json", "rate_maps"),
        ("data_trajectories_60s.json", "traj"),
    ):
        p = data / name
        if p.is_file():
            raw = json.loads(p.read_text())
            if key == "rate_maps":
                for r in raw:
                    r["rate"] = np.asarray(r["rate"], float)
            D[key] = raw
    D["decode"] = pd.read_csv(behinfo / "decode_rows.csv")
    D["ncurve"] = pd.read_csv(behinfo / "n_curve.csv")
    D["enc_rows"] = pd.read_csv(behinfo / "encoding_rows.csv")
    D["enc_units"] = pd.read_csv(behinfo / "encoding_diag_units.csv")
    D["drift_units"] = pd.read_csv(behinfo / "drift_noise_units.csv")
    D["drift_sum"] = json.loads((behinfo / "drift_noise_summary.json").read_text())
    D["uv"] = pd.read_csv(behinfo / "unique_variance_table.csv")
    D["enc_diag"] = json.loads((behinfo / "encoding_diag_summary.json").read_text())
    return D


def _pooled_decode(dec: pd.DataFrame) -> pd.DataFrame:
    d = dec.copy()
    if "N_role" in d.columns:
        d = d[d["N_role"] == "primary"]
    if "region" in d.columns:
        d = d[d["region"] == "pooled"]
    if "N" in d.columns:
        d = d[d["N"] == 20]
    return d


# ================================================================= FIG 1
def fig1(D, out: Path):
    """Sim units carry far more behavioural information than real ones."""
    dec = _pooled_decode(D["decode"])
    ncurve = D["ncurve"]
    uv = D["uv"]
    enc_u = D["enc_units"]
    meta = D["meta"]

    # Variables for panel a (wall = wall_distance_bin2 in tidy)
    var_order = [
        ("position", "position"),
        ("wall_distance_bin2", "wall distance"),
        ("movement_direction", "movement dir."),
        ("speed", "speed"),
        ("movement_state", "movement state"),
    ]
    fig = plt.figure(figsize=(W_FULL, 200 * MM))
    fig.suptitle(
        "Fig 1  Sim units carry far more behavioural information than real ones",
        fontsize=8.0, fontweight="bold", x=0.01, ha="left", y=0.990,
    )
    fig.text(0.99, 0.990, f"N = 20 primary · {_n_note(meta)}", fontsize=5.8, color=MUTED, ha="right", va="top")

    # ---- a: horizontal info per variable ----
    ax = fig.add_axes([0.14, 0.56, 0.48, 0.36])
    head_fig(fig, 0.01, 0.945, "a", "Information per variable (raw − max(null, 0); N = 20)")
    y_pos = np.arange(len(var_order))
    datasets = ["lab", "sim_matched", "sim_default"]
    offsets = {"lab": -0.22, "sim_matched": 0.0, "sim_default": 0.22}
    for vi, (var, _lab) in enumerate(var_order):
        for ds in datasets:
            sub = dec[(dec.dataset == ds) & (dec.variable == var)]["primary_minus_null"].to_numpy(float)
            sub = sub[np.isfinite(sub)]
            if not sub.size:
                continue
            yy = vi + offsets[ds] + np.random.default_rng(vi * 10 + hash(ds) % 7).normal(0, 0.015, len(sub))
            ax.scatter(sub, yy, s=7, alpha=0.28, color=DS_COL[ds], zorder=2, linewidths=0)
            ax.plot(np.median(sub), vi + offsets[ds], DS_MRK[ds], color=DS_COL[ds], ms=7,
                    markeredgecolor="white", markeredgewidth=0.4, zorder=4)
    hd = dec[(dec.dataset == "lab") & (dec.variable == "head_direction")]["primary_minus_null"].to_numpy(float)
    hd = hd[np.isfinite(hd)]
    ax.axvline(0, color=GRID, lw=0.8, zorder=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([lab for _, lab in var_order], fontsize=7)
    ax.set_xlabel("Info above null", fontsize=7)
    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(-0.55, len(var_order) - 0.45)
    ax.grid(axis="x", color=GRID, lw=0.4)
    handles = [
        Line2D([0], [0], marker=DS_MRK[ds], color="w", markerfacecolor=DS_COL[ds], ms=7, label=DS_LAB[ds])
        for ds in datasets
    ]
    ax.legend(handles=handles, fontsize=6.0, frameon=False, loc="lower right",
              title="median", title_fontsize=6.0, borderpad=0.2, handletextpad=0.3)
    if hd.size:
        ax_hd = fig.add_axes([0.14, 0.52, 0.48, 0.035])
        ax_hd.set_xlim(-0.15, 1.05)
        ax_hd.set_ylim(0, 1)
        ax_hd.axvline(0, color=GRID, lw=0.6)
        ax_hd.scatter(hd, np.full(len(hd), 0.5), s=6, alpha=0.3, color=DS_COL["lab"], linewidths=0)
        ax_hd.plot(np.median(hd), 0.5, "o", color=DS_COL["lab"], ms=6, markeredgecolor="white", markeredgewidth=0.4)
        ax_hd.set_yticks([0.5])
        ax_hd.set_yticklabels(["head dir. (lab only)"], fontsize=6.5)
        ax_hd.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax_hd.spines[sp].set_visible(False)

    # ---- b: info vs N (position), 10 → max ----
    axb = fig.add_axes([0.72, 0.56, 0.24, 0.36])
    head_fig(fig, 0.72, 0.945, "b", "Position info vs N")
    pos = ncurve[ncurve.variable == "position"].copy()
    # Prefer fixed ladder; fall back to every N with ≥5 sessions in both arms.
    want = [10, 20, 40, 60, 80, 100, 150, 200, 300]
    lab_Ns = set(pos[pos.dataset == "lab"].N.astype(int))
    mat_Ns = set(pos[pos.dataset == "sim_matched"].N.astype(int))
    ladder = [n for n in want if n in lab_Ns and n in mat_Ns]
    if len(ladder) < 3:
        common = sorted(lab_Ns & mat_Ns)
        ladder = [n for n in common if
                  (pos[(pos.dataset == "lab") & (pos.N == n)].shape[0] >= 5) and
                  (pos[(pos.dataset == "sim_matched") & (pos.N == n)].shape[0] >= 5)]
    for ds, col in (("lab", DS_COL["lab"]), ("sim_matched", DS_COL["sim_matched"])):
        sub = pos[pos.dataset == ds]
        xs, meds, los, his = [], [], [], []
        for n in ladder:
            vals = sub[sub.N == n]["value"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size < 3:
                continue
            xs.append(n)
            meds.append(float(np.median(vals)))
            los.append(float(np.percentile(vals, 5)))
            his.append(float(np.percentile(vals, 95)))
        if not xs:
            continue
        axb.fill_between(xs, los, his, color=col, alpha=0.15, lw=0)
        axb.plot(xs, meds, "-o", color=col, ms=4.0, lw=1.2, label=DS_LAB[ds])
    axb.axhline(0, color=GRID, lw=0.6)
    axb.set_xlabel("N units", fontsize=6.5)
    axb.set_ylabel("info", fontsize=6.5)
    axb.set_xscale("log")
    axb.legend(fontsize=5.5, frameon=False, loc="upper left", borderpad=0.2)
    axb.tick_params(labelsize=5.5)
    axb.grid(color=GRID, lw=0.35)

    # ---- c: population VE stacked ----
    axc = fig.add_axes([0.08, 0.16, 0.36, 0.28])
    head_fig(fig, 0.01, 0.48, "c", "Population variance explained by behaviour")
    groups = ["position", "wall", "direction", "speed", "accel", "movement_state"]
    gcol = {
        "position": "#2a78d6", "wall": "#1baf7a", "direction": "#7b5ea7",
        "speed": "#eb6834", "accel": "#c24e1e", "movement_state": "#52514e", "shared": "#b0aea6",
    }
    datasets_c = ["lab", "sim_matched", "sim_default"]
    x = np.arange(len(datasets_c))
    bottoms = np.zeros(len(datasets_c))
    for g in groups + ["shared"]:
        vals = []
        for ds in datasets_c:
            row = uv[uv.dataset == ds]
            col = "shared" if g == "shared" else f"unique_{g}"
            if row.empty or col not in row.columns:
                vals.append(0.0)
            else:
                vals.append(max(0.0, float(row.iloc[0][col])))
        axc.bar(x, vals, bottom=bottoms, color=gcol[g], width=0.55, label=g)
        bottoms = bottoms + np.asarray(vals)
    lab_ve = float(uv[uv.dataset == "lab"].iloc[0]["ve_full_minus_null"]) if len(uv[uv.dataset == "lab"]) else 0
    axc.plot(0, max(0, lab_ve), "D", color=INK, ms=6, zorder=5)
    axc.set_xticks(x)
    axc.set_xticklabels([DS_LAB[d] for d in datasets_c], fontsize=7)
    axc.set_ylabel("Variance share (clipped ≥ 0)", fontsize=6.5)
    axc.axhline(0, color=GRID, lw=0.5)
    axc.legend(fontsize=4.6, frameon=False, ncol=1, loc="upper left",
               bbox_to_anchor=(1.02, 1.0), borderpad=0.15, labelspacing=0.2)
    fig.text(
        0.08, 0.125,
        "Lab VE ≤ 0 even with per-unit CV — not a valid absolute encoding measure; prefer bits/spike.",
        fontsize=5.6, color=INK2, va="top", ha="left",
    )

    # ---- d: paired session strip of held-out bits/spike (place-tuned) ----
    axd = fig.add_axes([0.66, 0.16, 0.28, 0.28])
    head_fig(fig, 0.66, 0.48, "d", "Held-out bits/spike (place-tuned; paired sessions)")
    rng = np.random.default_rng(0)
    bits_col = "bits_spk_end" if "bits_spk_end" in enc_u.columns else "bits_ratemap_end"
    for i, ds in enumerate(("lab", "sim_matched")):
        sub = enc_u[(enc_u.dataset == ds) & (enc_u.place_tuned == True)][bits_col].to_numpy(float)
        sub = sub[np.isfinite(sub)]
        if not sub.size:
            continue
        if sub.size > 600:
            sub = rng.choice(sub, 600, replace=False)
        xx = i + rng.normal(0, 0.05, len(sub))
        axd.scatter(xx, sub, s=3, alpha=0.12, color="#c24e1e", linewidths=0, zorder=1)
    pt = enc_u[enc_u.place_tuned == True]
    sm = pt.groupby(["dataset", "session"], as_index=False)[bits_col].median()
    lab_s = sm[sm.dataset == "lab"].set_index("session")[bits_col]
    mat_s = sm[sm.dataset == "sim_matched"].set_index("session")[bits_col]
    common = sorted(set(lab_s.index) & set(mat_s.index))
    for s in common:
        axd.plot([0, 1], [lab_s[s], mat_s[s]], color="#b0aea6", lw=0.4, alpha=0.45, zorder=2)
    if common:
        axd.scatter(np.zeros(len(common)), [lab_s[s] for s in common],
                    s=10, color=DS_COL["lab"], alpha=0.55, zorder=3, linewidths=0)
        axd.scatter(np.ones(len(common)), [mat_s[s] for s in common],
                    s=10, color=DS_COL["sim_matched"], alpha=0.55, zorder=3, linewidths=0)
        for i, series in ((0, lab_s.loc[common]), (1, mat_s.loc[common])):
            med = float(np.median(series))
            axd.plot(i, med, "o", color=DS_COL["lab" if i == 0 else "sim_matched"], ms=8,
                     markeredgecolor="white", markeredgewidth=0.5, zorder=5)
            axd.text(i, med + 0.03, f"{med:.2f}", ha="center", fontsize=6.5, color=INK)
    axd.set_xticks([0, 1])
    axd.set_xticklabels(["lab", "matched"], fontsize=7)
    axd.set_ylabel("bits / spike", fontsize=7)
    axd.set_xlim(-0.45, 1.45)
    axd.axhline(0, color=GRID, lw=0.5)
    axd.grid(axis="y", color=GRID, lw=0.35)

    _caption(
        fig,
        "a: faint dots = sessions; bold markers = medians. Head direction is lab-only (sim HD = movement heading). "
        "b: session median ± 5–95% band; N ladder 10→max shared. "
        "c: unique + shared population VE (lab near zero — see note). "
        "d: grey lines = paired sessions; orange cloud = place-tuned units; bold = session medians. Frozen v5.",
        y=0.015,
    )
    save(fig, str(out), "Fig1_behavior_info")


def fig1a_headline(D, out: Path):
    """Miniature of Fig 1a for the story page."""
    dec = _pooled_decode(D["decode"])
    var_order = [
        ("position", "position"),
        ("wall_distance_bin2", "wall distance"),
        ("movement_direction", "movement dir."),
        ("speed", "speed"),
        ("movement_state", "movement state"),
    ]
    fig = plt.figure(figsize=(W_FULL, 52 * MM))
    ax = fig.add_axes([0.16, 0.22, 0.62, 0.62])
    fig.text(0.01, 0.92, "Fig 1a  Information per variable (N = 20)", fontsize=7.5,
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
    save(fig, str(out), "Fig1a_headline")


def _n_note(meta):
    return (
        f"n = {meta['n_animals']} animals, {meta['n_lab_sessions']} lab; "
        f"{meta['n_matched_sessions']} matched; {meta['n_default_seeds']} default"
    )


# ================================================================= FIG 2
def fig2(D, out: Path):
    """Sim units are tuned more often, more strongly, and their maps never drift."""
    meta = D["meta"]
    RM = D.get("rate_maps") or []
    sess = D["sess"]
    sep = (D["drift_sum"].get("sep_curve") or {})
    enc_u = D["enc_units"]
    drift_u = D["drift_units"]

    fig = plt.figure(figsize=(W_FULL, 205 * MM))
    fig.suptitle(
        "Fig 2  Sim units are tuned more often, more strongly, and their maps never drift",
        fontsize=7.8, fontweight="bold", x=0.01, ha="left", y=0.990,
    )
    fig.text(0.99, 0.990, _n_note(meta), fontsize=5.8, color=MUTED, ha="right", va="top")

    # ---- a: rate-map gallery 2×5 ----
    head_fig(fig, 0.01, 0.955, "a", "Place-tuned rate maps (Hz); SI percentiles among stored eligible units")
    for row_i, ds in enumerate(("lab", "sim_matched")):
        maps = [r for r in RM if r["dataset"] == ds and r.get("ok", True)]
        # order by spat_info; take up to 5
        maps = sorted(maps, key=lambda r: float(r.get("spat_info", 0)))
        if len(maps) >= 5:
            # pick indices spanning the range ≈ 10/30/50/70/90 of available
            idxs = [int(round(p * (len(maps) - 1))) for p in (0.10, 0.30, 0.50, 0.70, 0.90)]
            maps = [maps[i] for i in idxs]
        # shared colour scale per row
        vmax = max((float(np.nanmax(m["rate"])) for m in maps if np.isfinite(m["rate"]).any()), default=1.0)
        for j, r in enumerate(maps[:5]):
            ax = fig.add_axes([0.06 + j * 0.12, 0.72 - row_i * 0.18, 0.105, 0.14])
            rate = np.ma.masked_invalid(np.asarray(r["rate"], float))
            ext = r["extent"]
            ax.set_facecolor("#d8d6d0")
            ax.imshow(
                rate.T, origin="lower",
                extent=[ext[0], ext[1], ext[2], ext[3]],
                cmap="viridis", vmin=0, vmax=vmax,
                aspect="equal", interpolation="nearest",
            )
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color("#6e6c66"); sp.set_linewidth(0.4)
            pct = r.get("percentile", "?")
            ax.set_title(
                f"p{pct}  SI {r['spat_info']:.2f} · {r['peak_hz']:.1f} Hz",
                fontsize=4.8, pad=1.0,
            )
            if j == 0:
                ax.set_ylabel("lab" if ds == "lab" else "matched", fontsize=6.5)

    # ---- b: tuned-cell fractions, grouped bars + session dots ----
    axb = fig.add_axes([0.68, 0.58, 0.28, 0.32])
    head_fig(fig, 0.68, 0.955, "b", "Fraction tuned (shuffle 95%)")
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
                axb.scatter(xx, vals, s=5, alpha=0.25, color=DS_COL[ds], zorder=2, linewidths=0)
        bars = axb.bar(x + (di - 0.5) * w, meds, w * 0.9, color=DS_COL[ds], label=DS_LAB[ds], zorder=3, alpha=0.85)
        for rect, m in zip(bars, meds):
            if np.isfinite(m):
                axb.text(rect.get_x() + rect.get_width() / 2, m + 0.01, f"{m:.2f}",
                         ha="center", va="bottom", fontsize=5.0, color=INK)
    axb.axhline(0.05, color=FAILC, ls="--", lw=0.8, label="chance 0.05")
    axb.set_xticks(x)
    axb.set_xticklabels([lab for _, lab in classes], fontsize=6.5)
    axb.set_ylabel("fraction of units", fontsize=6.5)
    axb.set_ylim(0, 0.55)
    axb.legend(fontsize=5.2, frameon=False, loc="upper right")
    axb.grid(axis="y", color=GRID, lw=0.35)

    # ---- c: drift curve ----
    axc = fig.add_axes([0.08, 0.12, 0.42, 0.32])
    head_fig(fig, 0.01, 0.48, "c", "Place-map r vs block separation")
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
        axc.errorbar(seps, meds, yerr=yerr, fmt="o-", color=DS_COL[ds], ms=5.5,
                     capsize=2.5, lw=1.1, label=DS_LAB[ds])
    axc.set_xlabel("Block separation", fontsize=7)
    axc.set_ylabel("Median place-map r", fontsize=7)
    axc.set_xticks([1, 2, 3, 4, 5])
    axc.set_ylim(0.3, 1.02)
    axc.legend(fontsize=6, frameon=False, loc="lower left")
    axc.grid(color=GRID, lw=0.35)
    # adjacent-block reliability in-panel
    try:
        lab_adj = sep["lab"]["by_sep"]["1"]["median"]
        mat_adj = sep["sim_matched"]["by_sep"]["1"]["median"]
        axc.text(
            0.98, 0.08,
            f"Adjacent-block r: lab {lab_adj:.2f} · matched {mat_adj:.2f}",
            transform=axc.transAxes, ha="right", fontsize=6.2, color=INK2,
        )
    except Exception:
        pass

    # ---- d: first vs second half — unit callout from tidy correlations ----
    axd = fig.add_axes([0.58, 0.12, 0.38, 0.32])
    head_fig(fig, 0.58, 0.48, "d", "First-half vs second-half reliability (example units)")
    axd.set_xlim(0, 1)
    axd.set_ylim(0, 1)
    axd.axis("off")
    # Pick one place-tuned lab and one sim unit with finite r_first_last
    def _pick_unit(ds: str):
        place = enc_u[(enc_u.dataset == ds) & (enc_u.place_tuned == True)][
            ["session", "unit", "spatial_info", "bits_spk_end"]
        ].copy()
        du = drift_u[drift_u.dataset == ds][["session", "unit", "r_first_last"]].copy()
        m = place.merge(du, on=["session", "unit"], how="inner")
        m = m[np.isfinite(m["r_first_last"])]
        if m.empty:
            return None
        # high SI example
        return m.sort_values("spatial_info", ascending=False).iloc[0]

    for i, ds in enumerate(("lab", "sim_matched")):
        u = _pick_unit(ds)
        x0 = 0.05 + i * 0.48
        # two half boxes
        for hi, label in enumerate(("1st half", "2nd half")):
            axd.add_patch(FancyBboxPatch(
                (x0 + hi * 0.20, 0.35), 0.18, 0.45,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="#eef6f1" if ds == "lab" else "#e8f1fb",
                edgecolor=INK, lw=0.7, transform=axd.transAxes,
            ))
            axd.text(x0 + hi * 0.20 + 0.09, 0.58, label, ha="center", va="center",
                     fontsize=6, transform=axd.transAxes, color=INK2)
        title = "lab" if ds == "lab" else "matched sim"
        if u is not None:
            axd.text(
                x0 + 0.19, 0.22,
                f"{title}\nSI {float(u.spatial_info):.2f} · r(½,½) = {float(u.r_first_last):.2f}",
                ha="center", va="top", fontsize=6.5, transform=axd.transAxes, color=INK,
            )
        else:
            axd.text(x0 + 0.19, 0.22, f"{title}\n(no unit)", ha="center", fontsize=6.5,
                     transform=axd.transAxes, color=MUTED)
        # arrow between halves
        axd.annotate(
            "", xy=(x0 + 0.38, 0.58), xytext=(x0 + 0.23, 0.58),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.0),
            transform=axd.transAxes,
        )
    axd.text(
        0.5, 0.05,
        "Maps not re-extracted for preview; r from drift tidy (same keep-mask). Full half-maps in next pass if approved.",
        ha="center", fontsize=5.5, color=MUTED, transform=axd.transAxes,
    )

    _caption(
        fig,
        "a: shared colour scale per row; labels = SI and peak Hz (stored gallery, SI-rank ≈ p10–p90 of available maps). "
        "b: bars = session medians; dots = paired sessions; dashed = chance 0.05. "
        "c: 90% recording-day bootstrap of session medians. "
        "d: example place-tuned units; first/last half map correlation from v5 drift tidy.",
        y=0.012,
    )
    save(fig, str(out), "Fig2_tuning_drift")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--behinfo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="Fig1,Fig2")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    D = _load(Path(args.data), Path(args.behinfo))
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if "Fig1" in only:
        fig1(D, out)
        print(json.dumps({"fig": "Fig1"}), flush=True)
    if "Fig1a" in only or "Fig1" in only:
        fig1a_headline(D, out)
        print(json.dumps({"fig": "Fig1a_headline"}), flush=True)
    if "Fig2" in only:
        fig2(D, out)
        print(json.dumps({"fig": "Fig2"}), flush=True)


if __name__ == "__main__":
    main()
