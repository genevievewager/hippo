#!/usr/bin/env python3
"""Fig 8 + S8 + Fig3 information family (N=20 primary; null = max(null,0))."""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from agents.quadrant_n5.figures.figures import GRID, INK, INK2, MM, MUTED, W_FULL, FAILC  # noqa: E402
from agents.data_quality.report.figures_r4 import save  # noqa: E402
from agents.data_quality.phase3.divergence import _recording_day  # noqa: E402

OUT_DATA = _REPO / "outputs/data_quality/reports/report4a_v6/behavior_info"
OUT_FIGS = _REPO / "outputs/data_quality/reports/report4a_v6/figs"
DS_COL = {"lab": "#1a1a1a", "sim_matched": "#0b5fff", "sim_default": "#6e6c66"}
DS_LAB = {"lab": "lab", "sim_matched": "matched", "sim_default": "default"}
PRIMARY_VARS = [
    "position", "distance_to_wall", "speed", "acceleration",
    "movement_direction", "wall_distance_bin2", "spatial_context", "movement_state",
]
N_PRIMARY = 20


def _load():
    return dict(
        decode=pd.read_csv(OUT_DATA / "decode_rows.csv"),
        ncurve=pd.read_csv(OUT_DATA / "n_curve.csv"),
        enc=pd.read_csv(OUT_DATA / "encoding_rows.csv"),
        sens=pd.read_csv(OUT_DATA / "sensitivity_rows.csv") if (OUT_DATA / "sensitivity_rows.csv").is_file() else pd.DataFrame(),
        fig3=pd.read_csv(OUT_DATA / "fig3_information.csv") if (OUT_DATA / "fig3_information.csv").is_file() else pd.DataFrame(),
        hd=pd.read_csv(OUT_DATA / "lab_hd_vs_movement.csv") if (OUT_DATA / "lab_hd_vs_movement.csv").is_file() else pd.DataFrame(),
        decisions=json.loads((OUT_DATA / "decisions.json").read_text()),
    )


def _pooled_primary(dec: pd.DataFrame) -> pd.DataFrame:
    d = dec.copy()
    if "N_role" in d.columns:
        d = d[d["N_role"] == "primary"]
    if "region" in d.columns:
        d = d[d["region"] == "pooled"]
    if "N" in d.columns:
        d = d[d["N"] == N_PRIMARY]
    return d[d["primary_minus_null"].notna()].copy()


def _day_bootstrap_ci(session_vals: pd.DataFrame, col: str, n_boot: int = 200, rng=None):
    """90% CI of session-median, resampling by recording_day."""
    rng = rng or np.random.default_rng(0)
    df = session_vals.copy()
    df["recording_day"] = df["session"].astype(str).map(_recording_day)
    days = sorted(df["recording_day"].unique())
    if len(days) < 2:
        med = float(np.nanmedian(df[col]))
        return med, med, med
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(days, size=len(days), replace=True)
        sub = df[df["recording_day"].isin(draw)]
        boots.append(float(np.nanmedian(sub[col])))
    a = np.asarray(boots, float)
    return float(np.nanmedian(df[col])), float(np.nanquantile(a, 0.05)), float(np.nanquantile(a, 0.95))


def _load_sep_curve():
    p = OUT_DATA / "drift_noise_summary.json"
    if not p.is_file():
        return None
    return json.loads(p.read_text()).get("sep_curve") or {}


def fig8(D, out_dir: Path):
    """Fig 7 (renumbered): behaviour information — L→R reading order; e full width."""
    from agents.quadrant_n5.figures.figures import head_fig, save as _save

    dec = _pooled_primary(D["decode"])
    enc = D["enc"]
    if "N_role" in enc.columns:
        enc = enc[enc["N_role"] == "primary"].copy()
    ncurve = D["ncurve"]
    sep_curve = _load_sep_curve()
    # Fit A4 180 mm frame (~220 mm tall max with caption)
    fig = plt.figure(figsize=(W_FULL, 220 * MM))
    fig.suptitle(
        "Fig 7  How much behaviour the population carries (N = 20)",
        fontsize=8.0, fontweight="bold", x=0.01, ha="left", y=0.988,
    )

    # Row 1: a | b  (left → right)
    ax = fig.add_axes([0.07, 0.68, 0.36, 0.22])
    head_fig(fig, 0.07, 0.925, "a", "Info per variable (raw − max(null,0); ≤1)")
    vars_ = [v for v in PRIMARY_VARS if v in set(dec["variable"])]
    if "head_direction" in set(dec["variable"]):
        vars_ = vars_ + ["head_direction"]
    xpos = np.arange(len(vars_))
    width = 0.25
    for i, ds in enumerate(("lab", "sim_matched", "sim_default")):
        vals_med, vals_all = [], []
        for v in vars_:
            sub = dec[(dec.dataset == ds) & (dec.variable == v)]["primary_minus_null"].to_numpy(float)
            sub = sub[np.isfinite(sub)]
            vals_med.append(float(np.median(sub)) if sub.size else np.nan)
            vals_all.append(sub)
        xs, ys = [], []
        for j, a in enumerate(vals_all):
            if len(a):
                xs.append(np.full(len(a), xpos[j] + (i - 1) * width))
                ys.append(a)
        if xs:
            ax.scatter(np.concatenate(xs), np.concatenate(ys), s=8, alpha=0.35, color=DS_COL[ds], zorder=2)
        ax.plot(xpos + (i - 1) * width, vals_med, "o", ms=5.5, color=DS_COL[ds],
                label=DS_LAB[ds], zorder=3)
    ax.axhline(0, color=GRID, lw=0.6)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(xpos)
    ax.set_xticklabels([v.replace("_", "\n") for v in vars_], fontsize=5.2)
    ax.set_ylabel("Info above null", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.grid(axis="y", color=GRID, lw=0.4)
    # legend outside a
    ax_la = fig.add_axes([0.44, 0.68, 0.08, 0.22])
    ax_la.axis("off")
    ax_la.legend(*ax.get_legend_handles_labels(), fontsize=5.2, frameon=False, loc="center left")

    ax = fig.add_axes([0.56, 0.68, 0.30, 0.22])
    head_fig(fig, 0.56, 0.925, "b", "Encoding variance shares (90% day bootstrap)")
    groups = ["position", "wall", "direction", "speed", "accel", "movement_state"]
    gcol = {
        "position": "#2a78d6", "wall": "#1baf7a", "direction": "#7b5ea7",
        "speed": "#eb6834", "accel": "#c24e1e", "movement_state": "#52514e", "shared": "#b0aea6",
    }
    datasets = ["lab", "sim_matched", "sim_default"]
    x = np.arange(len(datasets))
    bottoms = np.zeros(len(datasets))
    rng = np.random.default_rng(0)
    ve_cis = []
    for g in groups + ["shared"]:
        meds = []
        for ds in datasets:
            sub = enc[enc.dataset == ds]
            col = "shared" if g == "shared" else f"unique_{g}"
            if sub.empty or col not in sub.columns:
                meds.append(0.0)
                continue
            med, lo, hi = _day_bootstrap_ci(sub, col, rng=rng)
            meds.append(med)
        show = np.clip(np.asarray(meds, float), 0, None)
        ax.bar(x, show, bottom=bottoms, color=gcol[g], width=0.55, label=g)
        bottoms = bottoms + show
    for i, ds in enumerate(datasets):
        sub = enc[enc.dataset == ds]
        if sub.empty:
            continue
        med, lo, hi = _day_bootstrap_ci(sub, "ve_full_minus_null", rng=rng)
        ax.errorbar(i, med, yerr=[[max(0, med - lo)], [max(0, hi - med)]],
                    fmt="D", color=INK, ms=5, capsize=2.5, zorder=5)
        ve_cis.append((ds, med, lo, hi))
    ax.set_xticks(x)
    ax.set_xticklabels([DS_LAB[d] for d in datasets], fontsize=6.5)
    ax.set_ylabel("Variance share", fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.axhline(0, color=GRID, lw=0.5)
    ax_lb = fig.add_axes([0.87, 0.68, 0.12, 0.22])
    ax_lb.axis("off")
    ax_lb.legend(*ax.get_legend_handles_labels(), fontsize=4.6, frameon=False, loc="center left")

    # Row 2: c | d
    ax = fig.add_axes([0.07, 0.38, 0.36, 0.22])
    head_fig(fig, 0.07, 0.625, "c", "Information vs N (raw CV R²)")
    markers = {"position": "o", "movement_direction": "s", "speed": "^"}
    for var in ("position", "movement_direction", "speed"):
        for ds in ("lab", "sim_matched", "sim_default"):
            sub = ncurve[(ncurve.variable == var) & (ncurve.dataset == ds)]
            if sub.empty:
                continue
            g = sub.groupby("N")["value"].median()
            ax.plot(g.index, g.values, markers[var], ms=4.5, color=DS_COL[ds], alpha=0.9)
            ax.plot(g.index, g.values, "-", color=DS_COL[ds], lw=0.7, alpha=0.5)
    ax.axvline(N_PRIMARY, color=FAILC, ls=":", lw=0.8)
    ax.set_xlabel("N units", fontsize=6.5)
    ax.set_ylabel("CV R²", fontsize=6.5)
    ax.set_xscale("log")
    ax.grid(color=GRID, lw=0.4)
    ax.tick_params(labelsize=5.5)
    handles_c = [
        Line2D([0], [0], marker="o", color=INK, ms=5, lw=0, label="position"),
        Line2D([0], [0], marker="s", color=INK, ms=5, lw=0, label="mov. dir."),
        Line2D([0], [0], marker="^", color=INK, ms=5, lw=0, label="speed"),
        Line2D([0], [0], marker="o", color=DS_COL["lab"], ms=5, lw=0, label="lab"),
        Line2D([0], [0], marker="o", color=DS_COL["sim_matched"], ms=5, lw=0, label="matched"),
        Line2D([0], [0], marker="o", color=DS_COL["sim_default"], ms=5, lw=0, label="default"),
        Line2D([0], [0], color=FAILC, ls=":", lw=0.8, label="N=20"),
    ]
    ax_lc = fig.add_axes([0.44, 0.38, 0.08, 0.22])
    ax_lc.axis("off")
    ax_lc.legend(handles=handles_c, fontsize=4.6, frameon=False, loc="center left")

    ax = fig.add_axes([0.56, 0.38, 0.30, 0.22])
    head_fig(fig, 0.56, 0.625, "d", "MEC vs SUB (N=20; ≥10 units)")
    reg = D["decode"]
    reg = reg[reg["region"].isin(["MEC", "SUB"])].copy()
    vars_d = ["position", "movement_direction", "speed", "wall_distance_bin2"]
    xpos = np.arange(len(vars_d))
    for i, ds in enumerate(("lab", "sim_matched")):
        for j, v in enumerate(vars_d):
            for reg_name, marker in (("MEC", "o"), ("SUB", "s")):
                sub = reg[(reg.dataset == ds) & (reg.variable == v) & (reg.region == reg_name)]
                if "primary_minus_null" in sub.columns and sub["primary_minus_null"].notna().any():
                    vals = sub["primary_minus_null"].to_numpy(float)
                elif "r2" in sub.columns and sub["r2"].notna().any():
                    vals = sub["r2"].to_numpy(float)
                else:
                    vals = sub["info"].to_numpy(float) if "info" in sub.columns else np.array([])
                vals = vals[np.isfinite(vals)]
                if not vals.size:
                    continue
                xx = j + (i - 0.5) * 0.35 + (0 if reg_name == "MEC" else 0.12)
                ax.scatter(np.full(len(vals), xx), vals, s=10, alpha=0.4,
                           color=DS_COL[ds], marker=marker)
                ax.plot(xx, np.median(vals), marker, color=DS_COL[ds], ms=6, mec="white", mew=0.4)
    ax.set_xticks(xpos + 0.05)
    ax.set_xticklabels([v.replace("_", "\n") for v in vars_d], fontsize=5.2)
    ax.set_ylabel("Info", fontsize=6.5)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0, color=GRID, lw=0.5)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.tick_params(labelsize=5.5)
    handles_d = [
        Line2D([0], [0], marker="o", color=INK, ms=5, lw=0, label="MEC"),
        Line2D([0], [0], marker="s", color=INK, ms=5, lw=0, label="SUB"),
        Line2D([0], [0], marker="o", color=DS_COL["lab"], ms=5, lw=0, label="lab"),
        Line2D([0], [0], marker="o", color=DS_COL["sim_matched"], ms=5, lw=0, label="matched"),
    ]
    ax_ld = fig.add_axes([0.87, 0.38, 0.12, 0.22])
    ax_ld.axis("off")
    ax_ld.legend(handles=handles_d, fontsize=5.0, frameon=False, loc="center left")

    # (e) full width at bottom
    ax = fig.add_axes([0.07, 0.10, 0.78, 0.20])
    head_fig(fig, 0.07, 0.325, "e", "Place-map r vs block separation (6 train blocks)")
    slope_txt = []
    if sep_curve:
        for ds in ("lab", "sim_matched"):
            c = sep_curve.get(ds) or {}
            by = c.get("by_sep") or {}
            if not by:
                continue
            seps = sorted(int(k) for k in by)
            meds = [by[str(s)]["median"] for s in seps]
            los = [by[str(s)]["lo90"] for s in seps]
            his = [by[str(s)]["hi90"] for s in seps]
            yerr = np.vstack([
                np.asarray(meds) - np.asarray(los),
                np.asarray(his) - np.asarray(meds),
            ])
            ax.errorbar(
                seps, meds, yerr=yerr, fmt="o-", color=DS_COL[ds], ms=5.5,
                capsize=2.5, lw=1.0, label=DS_LAB[ds],
            )
            sl = c.get("slope")
            if sl is not None:
                slope_txt.append(
                    f"{DS_LAB[ds]} slope={sl:.3f}/block "
                    f"[{c.get('slope_lo90', float('nan')):.3f},"
                    f"{c.get('slope_hi90', float('nan')):.3f}]"
                )
    else:
        ax.text(0.5, 0.5, "no sep-curve summary", ha="center", va="center",
                color=MUTED, transform=ax.transAxes)
    ax.set_xlabel("Block separation", fontsize=6.5)
    ax.set_ylabel("Median place-map r", fontsize=6.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_ylim(0.3, 1.02)
    ax.grid(color=GRID, lw=0.4)
    ax.tick_params(labelsize=5.5)
    ax_le = fig.add_axes([0.87, 0.10, 0.12, 0.20])
    ax_le.axis("off")
    ax_le.legend(*ax.get_legend_handles_labels(), fontsize=5.2, frameon=False, loc="center left")
    if slope_txt:
        ax.text(0.01, 0.05, "  ·  ".join(slope_txt), transform=ax.transAxes,
                fontsize=5.2, ha="left", color=INK2)

    sha = (D.get("decisions") or {}).get("hippo_r2v2_commit", "?")[:8]
    fig.text(
        0.01, 0.015,
        textwrap.fill(
            f"Primary N=20. Net info = raw CV − max(null,0). Report 2 v2 speed @{sha}. "
            "Diamond in b = full VE − null ±90% day bootstrap. Prefer held-out bits/spike over ridge VE on lab. "
            "e: 90% day bootstrap; lab r declines with separation, matched flat.",
            130,
        ),
        fontsize=8.0, color=INK2, va="bottom",
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, str(out_dir), "Fig7_behavior_info")
    # Keep legacy filename alias for any stale callers
    for ext in ("pdf", "png"):
        src = out_dir / f"Fig7_behavior_info.{ext}"
        dst = out_dir / f"Fig8_behavior_info.{ext}"
        if src.is_file():
            import shutil
            shutil.copy2(src, dst)
    plt.close(fig)
    pd.DataFrame([{"dataset": d, "ve_full_minus_null": m, "lo90": lo, "hi90": hi} for d, m, lo, hi in ve_cis]).to_csv(
        OUT_DATA / "encoding_ve_bootstrap.csv", index=False,
    )


def fig_s8(D, out_dir: Path):
    sens = D["sens"]
    dec = _pooled_primary(D["decode"])
    fig = plt.figure(figsize=(W_FULL, 170 * MM))
    fig.suptitle(
        "S8  Variable distributions, null diagnostics, and reweighted sensitivity (N = 20)",
        fontsize=7.8, fontweight="bold", x=0.01, ha="left", y=0.995,
    )
    # a: raw / null / net for a few vars
    ax = fig.add_axes([0.08, 0.55, 0.40, 0.38])
    ax.text(0.0, 1.02, "a  Raw CV, null mean / p95, and net (matched arm)", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", va="bottom")
    vars_a = ["position", "speed", "movement_direction", "wall_distance_bin2"]
    x = np.arange(len(vars_a))
    for i, ds in enumerate(("lab", "sim_matched")):
        raws, nets, nmeans, np95s = [], [], [], []
        for v in vars_a:
            sub = dec[(dec.dataset == ds) & (dec.variable == v)]
            raws.append(float(sub["primary_value"].median()) if len(sub) else np.nan)
            nets.append(float(sub["primary_minus_null"].median()) if len(sub) else np.nan)
            nmeans.append(float(sub["null_mean"].median()) if len(sub) and "null_mean" in sub else np.nan)
            np95s.append(float(sub["null_p95"].median()) if len(sub) and "null_p95" in sub else np.nan)
        off = (i - 0.5) * 0.15
        ax.plot(x + off, raws, "o", color=DS_COL[ds], ms=5, label=f"{DS_LAB[ds]} raw")
        ax.plot(x + off, nets, "s", color=DS_COL[ds], ms=5, label=f"{DS_LAB[ds]} net")
        ax.vlines(x + off, nmeans, np95s, color=DS_COL[ds], lw=1.2, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace("_", "\n") for v in vars_a], fontsize=5.5)
    ax.set_ylabel("Score", fontsize=6.5)
    ax.set_ylim(-0.2, 1.05)
    ax.axhline(0, color=GRID, lw=0.5)
    ax.legend(fontsize=4.8, frameon=False, ncol=2)
    ax.grid(axis="y", color=GRID, lw=0.4)

    # b: distributions reminder
    ax = fig.add_axes([0.55, 0.55, 0.40, 0.38])
    ax.text(0.0, 1.02, "b  Class / continuous summary (Step 1; exploratory spatial_context)", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", va="bottom")
    step1 = _REPO / "outputs/data_quality/reports/report4a_v3/_step1_behavior_info/step1_dataset_pooled.csv"
    ax.axis("off")
    if step1.is_file():
        s1 = pd.read_csv(step1)
        lines = []
        for _, r in s1.iterrows():
            lines.append(
                f"{r['dataset']}: wall {r['wall_med']:.1f} cm · speed(step1) {r['speed_med']:.1f} · "
                f"still/slow/fast {r['frac_state_still']*100:.0f}/{r['frac_state_slow']*100:.0f}/{r['frac_state_fast']*100:.0f}"
            )
        ax.text(0.0, 0.95, "\n".join(lines), fontsize=5.8, va="top", family="monospace",
                transform=ax.transAxes, color=INK)
        ax.text(0.0, 0.45,
                "Primary speed now = 250 ms backward |Δpos|/Δt (Report 2 v2).\n"
                "Lab V is a sensitivity row only.\n"
                "spatial_context: bbox 15%/25%; ~50% of lab bbox corner\n"
                "region lies outside the polygon (phantom corners).",
                fontsize=5.8, va="top", color=INK2, transform=ax.transAxes)

    # c: reweighted
    ax = fig.add_axes([0.08, 0.08, 0.40, 0.38])
    ax.text(0.0, 1.02, "c  Test reweighted to lab-pooled distribution (N = 20)", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", va="bottom")
    if sens is None or sens.empty:
        ax.axis("off")
        ax.text(0.1, 0.5, "no sensitivity rows", color=MUTED)
    else:
        vars_ = ["position", "distance_to_wall", "speed", "acceleration"]
        xpos = np.arange(len(vars_))
        for i, ds in enumerate(("lab", "sim_matched", "sim_default")):
            meds = []
            for v in vars_:
                sub = sens[(sens.dataset == ds) & (sens.variable == v)]["r2"].to_numpy(float)
                sub = sub[np.isfinite(sub)]
                meds.append(float(np.median(sub)) if sub.size else np.nan)
            ax.plot(xpos + (i - 1) * 0.08, meds, "o-", color=DS_COL[ds], ms=5, label=DS_LAB[ds])
            meds0 = []
            for v in vars_:
                sub = dec[(dec.dataset == ds) & (dec.variable == v)]["primary_value"].to_numpy(float)
                sub = sub[np.isfinite(sub)]
                meds0.append(float(np.median(sub)) if sub.size else np.nan)
            ax.plot(xpos + (i - 1) * 0.08, meds0, "x", color=DS_COL[ds], ms=5, alpha=0.7)
        ax.set_xticks(xpos)
        ax.set_xticklabels([v.replace("_", "\n") for v in vars_], fontsize=5.5)
        ax.set_ylabel("CV R²", fontsize=6.5)
        ax.legend(fontsize=5.5, frameon=False)
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax.text(0.99, 0.02, "○ reweighted  × original raw", transform=ax.transAxes,
                fontsize=5.0, ha="right", color=MUTED)

    # d: speed_table_V sensitivity
    ax = fig.add_axes([0.55, 0.08, 0.40, 0.38])
    ax.text(0.0, 1.02, "d  Speed sensitivity: table V vs 250 ms pos-speed (N = 20)", transform=ax.transAxes,
            fontsize=7.0, fontweight="bold", va="bottom")
    rows = []
    for ds in ("lab", "sim_matched", "sim_default"):
        prim = dec[(dec.dataset == ds) & (dec.variable == "speed")]["primary_value"]
        tab = sens[(sens.dataset == ds) & (sens.variable == "speed_table_V")]["r2"] if not sens.empty else pd.Series(dtype=float)
        rows.append((ds, float(prim.median()) if len(prim) else np.nan,
                     float(tab.median()) if len(tab) else np.nan))
    ax.bar(np.arange(3) - 0.15, [r[1] for r in rows], width=0.3, color="#2a78d6", label="250 ms pos-speed")
    ax.bar(np.arange(3) + 0.15, [r[2] for r in rows], width=0.3, color="#eb6834", label="table V / speed_cm_s")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([DS_LAB[r[0]] for r in rows], fontsize=6.5)
    ax.set_ylabel("CV R²", fontsize=6.5)
    ax.legend(fontsize=5.2, frameon=False)
    ax.grid(axis="y", color=GRID, lw=0.4)

    fig.text(
        0.01, 0.005,
        "Null bars in a span null mean → p95 (session medians). Net never uses a negative null baseline. "
        "Table V is sensitivity only (Report 2 v2).",
        fontsize=5.0, color=INK2,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    save(fig, str(out_dir), "S8_behavior_distributions_sensitivity")
    plt.close(fig)


def unique_variance_table(D) -> pd.DataFrame:
    enc = D["enc"]
    if "N_role" in enc.columns:
        enc = enc[enc["N_role"] == "primary"]
    rows = []
    groups = ["position", "wall", "direction", "speed", "accel", "movement_state"]
    rng = np.random.default_rng(0)
    for ds in ("lab", "sim_matched", "sim_default"):
        sub = enc[enc.dataset == ds]
        if sub.empty:
            continue
        med, lo, hi = _day_bootstrap_ci(sub, "ve_full_minus_null", rng=rng)
        row = {
            "dataset": ds, "n_sessions": len(sub), "N": N_PRIMARY,
            "ve_full": float(np.median(sub["ve_full"])),
            "ve_full_minus_null": med, "ve_lo90": lo, "ve_hi90": hi,
            "shared": float(np.median(sub["shared"])),
        }
        for g in groups:
            row[f"unique_{g}"] = float(np.median(sub[f"unique_{g}"]))
        rows.append(row)
    return pd.DataFrame(rows)


def merge_fig3(D, report_data: Path):
    fig3_new = D["fig3"]
    if fig3_new.empty:
        return None
    rows = []
    matched = fig3_new[fig3_new.arm == "sim_matched"].set_index("metric")
    default = fig3_new[fig3_new.arm == "sim_default"].set_index("metric")
    for m in matched.index:
        lead = bool(matched.loc[m].get("lead_with_abs", False)) if "lead_with_abs" in matched.columns else False
        rows.append({
            "metric_base": m,
            "metric": m,
            "label": matched.loc[m, "label"],
            "family": "information",
            "R_matched": matched.loc[m, "R"],
            "R_matched_lo": matched.loc[m, "R_lo90"],
            "R_matched_hi": matched.loc[m, "R_hi90"],
            "R_default": default.loc[m, "R"] if m in default.index else np.nan,
            "R_default_lo": np.nan,
            "R_default_hi": np.nan,
            "lab_median": matched.loc[m, "lab_median"],
            "sim_matched_median": matched.loc[m, "sim_median"],
            "sim_default_median": default.loc[m, "sim_median"] if m in default.index else np.nan,
            "lab_mad": matched.loc[m].get("lab_mad", np.nan) if "lab_mad" in matched.columns else np.nan,
            "lead_with_abs": lead,
            "N": N_PRIMARY,
            "direction_matched": np.nan,
            "too_regular_matched": False,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DATA / "fig3_information_merged_rows.csv", index=False)
    base = report_data / "data_fig3_divergence.csv"
    if base.is_file():
        old = pd.read_csv(base)
        old = old[old["family"] != "information"].copy()
        if "lead_with_abs" not in old.columns:
            old["lead_with_abs"] = False
        if "lab_mad" not in old.columns:
            old["lab_mad"] = np.nan
        if "N" not in old.columns:
            old["N"] = np.nan
        if "metric" not in old.columns and "metric_base" in old.columns:
            old["metric"] = old["metric_base"]
        merged = pd.concat([old, out], ignore_index=True, sort=False)
        merged.to_csv(OUT_DATA / "data_fig3_divergence_with_info.csv", index=False)
    return out


def fig3_info_panel(D, out_dir: Path):
    df = pd.read_csv(OUT_DATA / "fig3_information_merged_rows.csv")
    df = df.sort_values("R_matched", ascending=True)
    fig = plt.figure(figsize=(W_FULL, 120 * MM))
    fig.suptitle(
        "Fig 3 addition  Information family (N = 20; net ≤ 1)",
        fontsize=7.8, fontweight="bold", x=0.01, ha="left", y=0.98,
    )
    ax = fig.add_axes([0.28, 0.10, 0.48, 0.82])
    y = np.arange(len(df))
    ax.axvline(1, color=FAILC, ls="--", lw=0.9)
    for i, r in enumerate(df.itertuples()):
        ax.plot(r.R_matched, i, "o", color=INK, ms=6)
        if np.isfinite(r.R_matched_lo) and np.isfinite(r.R_matched_hi):
            ax.plot([r.R_matched_lo, r.R_matched_hi], [i, i], "-", color=INK, lw=1.0)
        if np.isfinite(getattr(r, "R_default", np.nan)):
            ax.plot(r.R_default, i, "s", color=MUTED, ms=5)
    ax.set_yticks(y)
    labels = []
    for r in df.itertuples():
        lab = r.label
        if getattr(r, "lead_with_abs", False):
            lab = lab + " *"
        labels.append(lab)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Divergence ratio R (log)", fontsize=6.5)
    ax.set_xscale("log")
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax2 = fig.add_axes([0.78, 0.10, 0.20, 0.82])
    ax2.axis("off")
    ax2.set_ylim(-0.5, len(df) - 0.5)
    ax2.text(0, len(df) - 0.2, "lab | matched", fontsize=5.8, fontweight="bold", color=INK2)
    for i, r in enumerate(df.itertuples()):
        ax2.text(0, i, f"{r.lab_median:.2f} | {r.sim_matched_median:.2f}", fontsize=5.5,
                 va="center", family="monospace")
    fig.text(
        0.01, 0.01,
        "Circles = matched R (±90% day bootstrap); squares = default. Values = session-median net info "
        "(raw − max(null,0), ≤1) at N = 20. * = lab MAD tiny: lead with absolute values; R is a footnote.",
        fontsize=5.0, color=INK2,
    )
    save(fig, str(out_dir), "Fig3_information_family")
    plt.close(fig)


def main():
    D = _load()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    fig8(D, OUT_FIGS)
    fig_s8(D, OUT_FIGS)
    uvt = unique_variance_table(D)
    uvt.to_csv(OUT_DATA / "unique_variance_table.csv", index=False)
    merge_fig3(D, _REPO / "outputs/data_quality/reports/report4a_v3/data")
    fig3_info_panel(D, OUT_FIGS)
    print(json.dumps({
        "fig7": str(OUT_FIGS / "Fig7_behavior_info.png"),
        "s8": str(OUT_FIGS / "S8_behavior_distributions_sensitivity.png"),
        "fig3": str(OUT_FIGS / "Fig3_information_family.png"),
        "unique_variance": uvt.to_dict(orient="records"),
    }, indent=2))


if __name__ == "__main__":
    main()
