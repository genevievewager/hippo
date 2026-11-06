"""Report 6 figures — rebuilt for easy interpretation of pilot results."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

from agents.report6.paths import OUT_ROOT
from agents.report6.prose import (
    BOARD_CRITERIA,
    FIG61B_ROWS,
    FIG61_ONE_LINE_RULES,
    PAGE1_BOARD_NOTE,
)

MM = 1 / 25.4
W_FULL = 180 * MM
FIG_DIR = OUT_ROOT / "figures"
DATA_DIR = OUT_ROOT

PASS_C = "#2ca02c"
FAIL_C = "#d62728"
NA_C = "#bdbdbd"
ACCENT = "#4C78A8"
ACCENT2 = "#F58518"


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    # 400 dpi so draft pages (300 PPI embed) never upsample a soft source bitmap
    fig.savefig(png, dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png


def _load(path: Path) -> Any:
    return json.loads(path.read_text()) if path.is_file() else None


def _idx(rows: list[dict], **kw) -> dict | None:
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    return None


def _yn(v) -> str:
    if v is None:
        return "—"
    return "PASS" if v else "FAIL"


def _pca_level_table() -> dict[str, Any] | None:
    """Prefer d-grid / N1=50 tables when present; else the default table."""
    for name in (
        "pca_level_table_dgrid.json",
        "pca_level_table_n1_50.json",
        "pca_level_table.json",
    ):
        blob = _load(DATA_DIR / "location" / name)
        if blob and blob.get("rows"):
            return blob
    return None


def _loc_row(
    tab: dict[str, Any] | None,
    alias: str,
    *,
    split: str = "S1",
    d: int | None = None,
    primary: bool = False,
) -> dict[str, Any] | None:
    if not tab:
        return None
    for r in tab["rows"]:
        if r.get("alias") != alias or r.get("arm") != "real":
            continue
        if r.get("split") != split or r.get("drop_u") is not True:
            continue
        if primary and r.get("is_primary_d"):
            return r
        if not primary and d is not None and r.get("d") == d:
            return r
    return None


def _hard_split_flags(alias: str, tab: dict[str, Any] | None) -> dict[str, Any]:
    """Pre-registered conjunction at primary d (S1, u out); PCA defines targets."""
    out: dict[str, Any] = {
        "same_subspace": None,
        "same_tiles": None,
        "resolvable": None,
        "n_stable": None,
        "n_stable_d3": None,
        "n_stable_n1_p95": None,
        "n_stable_d3_n1_p95": None,
        "primary_d": None,
        "e_rigid_d3": None,
        "r_sigma_250": None,
        "r_sigma_1s": None,
        "ari_s1_d3": None,
        "ari_s1_n1_p95": None,
        "ari_s2_d3": None,
        "session_pass": None,
    }
    if not tab:
        return out
    r_pri = _loc_row(tab, alias, primary=True)
    r_d3 = _loc_row(tab, alias, d=3)
    r_s2 = _loc_row(tab, alias, split="S2", d=3)
    w1 = _load(DATA_DIR / "location" / "pca_level_table_w1s.json")
    r_w1 = _loc_row(w1, alias, d=3)
    if r_pri:
        out["same_subspace"] = r_pri.get("pass_n_stable")
        out["same_tiles"] = r_pri.get("pass_ari")
        out["resolvable"] = r_pri.get("pass_tiles")
        out["n_stable"] = r_pri.get("n_stable")
        out["n_stable_n1_p95"] = r_pri.get("n_stable_n1_p95")
        out["primary_d"] = r_pri.get("primary_d")
        flags = [out["same_subspace"], out["same_tiles"], out["resolvable"]]
        if all(f is not None for f in flags):
            out["session_pass"] = all(bool(f) for f in flags)
    if r_d3:
        out["n_stable_d3"] = r_d3.get("n_stable")
        out["n_stable_d3_n1_p95"] = r_d3.get("n_stable_n1_p95")
        out["e_rigid_d3"] = r_d3.get("e_rigid_median_all")
        out["r_sigma_250"] = r_d3.get("median_r_over_sigma_noise")
        out["ari_s1_d3"] = r_d3.get("ari")
        out["ari_s1_n1_p95"] = r_d3.get("ari_n1_p95")
    if r_s2:
        out["ari_s2_d3"] = r_s2.get("ari")
    if r_w1:
        out["r_sigma_1s"] = r_w1.get("median_r_over_sigma_noise")
    return out


def collect_session_table() -> list[dict[str, Any]]:
    """One row per real session; PCA primary for surrogate / hard-split board."""
    m0 = _load(DATA_DIR / "m0_m2" / "all.json") or []
    na = _load(DATA_DIR / "null_a" / "summary.json") or []
    nb = _load(DATA_DIR / "null_b" / "summary.json") or []
    sp = _load(DATA_DIR / "split" / "summary.json") or []
    bt = _load(DATA_DIR / "boot" / "summary.json") or []
    loc = _pca_level_table()

    rows = []
    for sess in m0:
        if sess.get("arm") != "real":
            continue
        alias = sess["alias"]
        # Board / page-1: PCA targets are primary for qualify/detect numbers too
        # when present; fall back to LDS for legacy occupancy strip.
        pca = (sess.get("methods") or {}).get("pca") or {}
        lds = (sess.get("methods") or {}).get("lds") or {}
        te = (pca.get("targets_eval") or lds.get("targets_eval") or [])
        top = te[0] if te else None
        na_r = _idx(na, alias=alias, method="lds")
        nb_lds = _idx(nb, alias=alias, method="lds")
        nb_pca = _idx(nb, alias=alias, method="pca")
        sp_lds = _idx(sp, alias=alias, method="lds")
        sp_pca = _idx(sp, alias=alias, method="pca")
        bt_lds = _idx(bt, alias=alias, method="lds")
        bt_pca = _idx(bt, alias=alias, method="pca")
        hs = _hard_split_flags(alias, loc)
        rows.append({
            "alias": alias,
            "animal": sess.get("methods", {}).get("lds", {}).get("animal")
                      or (lds.get("animal")),
            "qualifies": bool(te),
            "occupancy": None if not top else top["target"]["occupancy"],
            "visits": None if not top else top["target"]["n_visits"],
            "dwell_s": None if not top else top["target"]["median_dwell_s"],
            "delta_p": None if not top else top.get("delta_p"),
            "brier": None if not top else top.get("brier"),
            "distinguishable": None if not top else bool(top.get("distinguishable_pass")),
            "null_a": None if na_r is None else bool(na_r.get("null_a_pass")),
            "null_a_npass": None if na_r is None else na_r.get("n_pass"),
            "null_b_pca": None if nb_pca is None else bool(nb_pca.get("null_b_pass")),
            "null_b_lds": None if nb_lds is None else bool(nb_lds.get("null_b_pass")),
            # Legacy key: PCA primary
            "null_b": None if nb_pca is None else bool(nb_pca.get("null_b_pass")),
            "null_b_method": "pca",
            "null_b_npass": None if nb_pca is None else nb_pca.get("n_pass"),
            "split": None if (sp_pca or sp_lds) is None else bool(
                (sp_pca or sp_lds).get("pass")
            ),
            "split_kappa": None if (sp_pca or sp_lds) is None else (
                sp_pca or sp_lds
            ).get("kappa"),
            "boot": None if (bt_pca or bt_lds) is None else bool(
                (bt_pca or bt_lds).get("pass")
            ),
            "boot_jaccard": None if (bt_pca or bt_lds) is None else (
                bt_pca or bt_lds
            ).get("median_jaccard"),
            "same_subspace": hs["same_subspace"],
            "same_tiles": hs["same_tiles"],
            "resolvable": hs["resolvable"],
            "n_stable": hs["n_stable"],
            "n_stable_d3": hs["n_stable_d3"],
            "n_stable_n1_p95": hs["n_stable_n1_p95"],
            "n_stable_d3_n1_p95": hs["n_stable_d3_n1_p95"],
            "primary_d": hs["primary_d"],
            "e_rigid_d3": hs["e_rigid_d3"],
            "r_sigma_250": hs["r_sigma_250"],
            "r_sigma_1s": hs["r_sigma_1s"],
            "ari_s1_d3": hs["ari_s1_d3"],
            "ari_s1_n1_p95": hs["ari_s1_n1_p95"],
            "ari_s2_d3": hs["ari_s2_d3"],
            "hard_split_pass": hs["session_pass"],
            "null_b_npass_pca": None if nb_pca is None else nb_pca.get("n_pass"),
            "null_b_npass_lds": None if nb_lds is None else nb_lds.get("n_pass"),
            "null_b_nsurr_pca": None if nb_pca is None else nb_pca.get("n_surr"),
            "null_b_nsurr_lds": None if nb_lds is None else nb_lds.get("n_surr"),
        })
    cohort = _load(DATA_DIR / "cohort_public.json") or []
    by_a = {r["alias"]: r for r in cohort}
    for r in rows:
        if not r.get("animal") and r["alias"] in by_a:
            r["animal"] = by_a[r["alias"]].get("animal")
    return rows


# ---------------------------------------------------------------------------
# F6.0 — results board (the figure to open first)
# ---------------------------------------------------------------------------

def fig60_results_board() -> Path:
    rows = collect_session_table()
    pl = _load(DATA_DIR / "plant" / "summary.json") or []
    plant_pca = _idx(pl, method="pca") or (pl[0] if pl else None)

    criteria = list(BOARD_CRITERIA)
    aliases = [r["alias"] for r in rows]
    n_s, n_c = len(aliases), len(criteria)
    M = np.full((n_s, n_c), -1.0)
    for i, r in enumerate(rows):
        for j, (key, _) in enumerate(criteria):
            v = r.get(key)
            if v is None:
                M[i, j] = -1
            else:
                M[i, j] = 1 if v else 0

    fig = plt.figure(figsize=(W_FULL, 170 * MM), facecolor="white")
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.12, 1.35, 0.28, 0.85], hspace=0.35,
        left=0.22, right=0.98, top=0.90, bottom=0.05,
    )

    ax_leg = fig.add_subplot(gs[0, 0])
    ax_leg.set_axis_off()
    ax_leg.legend(
        handles=[
            Patch(facecolor=PASS_C, edgecolor="#333", label="PASS"),
            Patch(facecolor=FAIL_C, edgecolor="#333", label="FAIL"),
            Patch(facecolor=NA_C, edgecolor="#333", label="not run"),
        ],
        loc="center left", fontsize=8, frameon=False, ncol=3,
        handlelength=1.4,
    )
    ax_leg.set_title(
        "a  Per-session pass/fail — PCA targets; LDS shown for the surrogate row only",
        loc="left", fontsize=8, fontweight="bold", pad=2,
    )

    ax = fig.add_subplot(gs[1, 0])
    cmap = ListedColormap([FAIL_C, NA_C, PASS_C])
    Z = np.where(M < 0, 1, np.where(M > 0, 2, 0))
    ax.imshow(Z.T, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(n_s))
    ax.set_xticklabels(aliases, fontsize=9)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels([c[1] for c in criteria], fontsize=8)
    ax.tick_params(axis="y", pad=4)
    for i in range(n_s):
        for j in range(n_c):
            if M[i, j] < 0:
                lab, col = "—", "#555"
            elif M[i, j] > 0:
                lab, col = "PASS", "white"
            else:
                lab, col = "FAIL", "white"
            ax.text(i, j, lab, ha="center", va="center", fontsize=7,
                    color=col, fontweight="bold")

    ax_note = fig.add_subplot(gs[2, 0])
    ax_note.set_axis_off()
    note = textwrap.fill(PAGE1_BOARD_NOTE, width=108)
    ax_note.text(
        0.0, 0.85, note, transform=ax_note.transAxes, va="top", ha="left",
        fontsize=7.0, color="#333",
    )

    ax2 = fig.add_subplot(gs[3, 0])
    ax2.set_axis_off()
    n = max(len(rows), 1)
    frac_q = sum(1 for r in rows if r["qualifies"]) / n
    frac_d = sum(1 for r in rows if r["qualifies"] and r["distinguishable"]) / n
    frac_surr = sum(
        1 for r in rows
        if r["qualifies"] and r["distinguishable"] and r.get("null_b") is True
    ) / n
    plant_hit = None if plant_pca is None else plant_pca.get("hit_rate")
    plant_fa = None if plant_pca is None else plant_pca.get("fa_per_min")

    boxes = [
        (0.02, f"{frac_q*100:.0f}%", "Recurring\ntarget (n=10)"),
        (0.27, f"{frac_d*100:.0f}%", "Also held-out\ndistinguishable"),
        (0.52, f"{frac_surr*100:.0f}%", "Also survives\nsurrogate"),
        (0.77,
         "—" if plant_hit is None else f"{plant_hit*100:.0f}%",
         f"Plant hit rate\n(FA/min={plant_fa})"),
    ]
    for x, big, small in boxes:
        ax2.add_patch(FancyBboxPatch(
            (x, 0.12), 0.22, 0.72, transform=ax2.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#f7f7f7", edgecolor="#333", lw=1.0,
            mutation_aspect=0.3,
        ))
        ax2.text(x + 0.11, 0.58, big, transform=ax2.transAxes,
                 ha="center", va="center", fontsize=18, fontweight="bold")
        ax2.text(x + 0.11, 0.28, small, transform=ax2.transAxes,
                 ha="center", va="center", fontsize=7.5, color="#333")
    ax2.set_title("b  Three numbers for the deck (+ plant)", loc="left",
                  fontsize=9, fontweight="bold", pad=6)

    fig.suptitle(
        "PCA targets; LDS shown for the surrogate row only",
        fontsize=9, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_0_board")


# ---------------------------------------------------------------------------
# F6.1 schematic
# ---------------------------------------------------------------------------

def fig61_schematic_and_table() -> Path:
    """Pipeline (a) + one-line pass rules (b). Full table on the appendix page."""
    fig = plt.figure(figsize=(W_FULL, 190 * MM), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.85, 1.55], hspace=0.18,
        left=0.04, right=0.98, top=0.93, bottom=0.03,
    )
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    boxes = [
        (0.3, 3.6, "M0  Causal latents\n(LDS / PCA cache)"),
        (3.4, 3.6, "M1  Target rule\n(train density)"),
        (6.5, 3.6, "M2  Detector\n(held-out only)"),
        (0.3, 0.6, "Surrogate\n(null b)"),
        (3.4, 0.6, "Split-pop +\nhard-split"),
        (6.5, 0.6, "Plant &\nrecover"),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), 2.9, 2.0, boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor="#f3f3f3", edgecolor="#333", lw=1.0,
        ))
        ax.text(x + 1.45, y + 1.0, t, ha="center", va="center", fontsize=8)
    ax.set_title("a  Pipeline", loc="left", fontsize=10, fontweight="bold", pad=4)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_axis_off()
    ax2.set_title(
        "b  Pass rules (one line each; full question/how/fail table in appendix)",
        loc="left", fontsize=9, fontweight="bold", pad=4,
    )
    headers = ["test", "pass if"]
    cell = [headers]
    for name, pass_if in FIG61_ONE_LINE_RULES:
        cell.append([name, pass_if])
    table = ax2.table(
        cellText=cell, loc="upper center", cellLoc="left",
        colWidths=[0.28, 0.72],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.85)
    for j in range(2):
        table[0, j].set_facecolor("#e8e8e8")
        table[0, j].set_text_props(fontweight="bold")
    for i in range(len(cell)):
        for j in range(2):
            table[i, j].PAD = 0.03
            table[i, j].set_text_props(va="center")

    fig.suptitle(
        "A neural-state target counts only if it recurs, separates, belongs to "
        "the population, and reproduces on an independent half",
        fontsize=9, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_1")


# ---------------------------------------------------------------------------
# F6.2 targets — separate occupancy / visits (no dual axis)
# ---------------------------------------------------------------------------

def fig62_targets() -> Path | None:
    data = _load(DATA_DIR / "m0_m2" / "all.json")
    if not data:
        return None
    occ_cmp = _load(DATA_DIR / "null_b" / "occupancy_compare.json") or []

    def top_stats(arm: str, method: str):
        aliases, occ, vis, inv_m, surr_occ = [], [], [], [], []
        for sess in data:
            if sess.get("arm") != arm:
                continue
            blob = (sess.get("methods") or {}).get(method) or {}
            te = blob.get("targets_eval") or []
            m1 = blob.get("m1") or {}
            k = int(m1.get("k") or 0)
            alias = sess["alias"]
            aliases.append(alias.replace("SIM", "S") if arm == "sim" else alias)
            if te:
                t = te[0]["target"]
                occ.append(100 * t["occupancy"])
                vis.append(t["n_visits"])
            else:
                occ.append(0.0)
                vis.append(0)
            inv_m.append(100.0 / k if k > 0 else np.nan)
            oc = _idx(occ_cmp, alias=alias, method=method)
            if oc and oc.get("median_surrogate_occupancy") is not None:
                surr_occ.append(100.0 * float(oc["median_surrogate_occupancy"]))
            else:
                surr_occ.append(np.nan)
        return aliases, occ, vis, inv_m, surr_occ

    panels = [
        (0, 0, "real", "lds", "a  Real · LDS"),
        (0, 1, "real", "pca", "b  Real · PCA"),
        (1, 0, "sim", "lds", "c  Sim (k=0) · LDS"),
        (1, 1, "sim", "pca", "d  Sim (k=0) · PCA"),
    ]
    packed = []
    all_occ, all_vis = [], []
    for spec in panels:
        aliases, occ, vis, inv_m, surr_occ = top_stats(spec[2], spec[3])
        packed.append((spec, aliases, occ, vis, inv_m, surr_occ))
        all_occ.extend(occ)
        all_vis.extend(vis)

    occ_max = max(5.0, max(all_occ) if all_occ else 5.0) * 1.12
    vis_max = max(10.0, max(all_vis) if all_vis else 10.0) * 1.12

    fig = plt.figure(figsize=(W_FULL, 140 * MM), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, hspace=0.55, wspace=0.60,
        left=0.09, right=0.88, top=0.86, bottom=0.14,
    )

    legend_handles = None
    for (r, c, _arm, _method, title), aliases, occ, vis, inv_m, surr_occ in packed:
        ax = fig.add_subplot(gs[r, c])
        x = np.arange(len(aliases))
        w = 0.38
        b = ax.bar(x - w / 2, occ, width=w, color=ACCENT, label="occupancy %")
        h5 = ax.axhline(5, color="gray", ls="--", lw=0.8, label="5% rule")
        (m1ln,) = ax.plot(
            x, inv_m, "s", color="#666", ms=3.5, label="1/M %", zorder=5,
        )
        (surrln,) = ax.plot(
            x, surr_occ, "x", color="#888", ms=4, label="surr. med occ %", zorder=5,
        )
        ax.set_ylim(0, occ_max)
        ax.tick_params(axis="y", labelcolor=ACCENT, labelsize=7)
        if c == 0:
            ax.set_ylabel("occupancy %", fontsize=8, color=ACCENT)
        ax2 = ax.twinx()
        (ln,) = ax2.plot(x, vis, "o-", color=ACCENT2, ms=4, lw=1.2, label="visits")
        h10 = ax2.axhline(10, color=ACCENT2, ls=":", lw=0.8, alpha=0.7, label="10 visits")
        ax2.set_ylim(0, vis_max)
        ax2.tick_params(axis="y", labelcolor=ACCENT2, labelsize=7)
        if c == 1:
            ax2.set_ylabel("visits", fontsize=8, color=ACCENT2)
        else:
            ax2.set_ylabel("")
        ax.set_xticks(x)
        ax.set_xticklabels(aliases, fontsize=6, rotation=45, ha="right")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=4)
        if legend_handles is None:
            legend_handles = [b, h5, m1ln, surrln, ln, h10]

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["occupancy %", "5% rule", "1/M %", "surr. med occ %", "visits", "10 visits"],
            loc="lower center", ncol=6, fontsize=6.5, frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

    n_real = sum(1 for s in data if s["arm"] == "real")
    n_sim = sum(1 for s in data if s["arm"] == "sim")
    fig.suptitle(
        f"Top qualifying target per session (n_real={n_real}, n_sim={n_sim}). "
        "Shared y-limits across panels; dashed = 5% occ.; dotted = 10 visits.",
        fontsize=9, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_2")


# ---------------------------------------------------------------------------
# F6.3 detection + nulls
# ---------------------------------------------------------------------------

def fig63_detection_nulls() -> Path | None:
    m0 = _load(DATA_DIR / "m0_m2" / "all.json")
    na = _load(DATA_DIR / "null_a" / "summary.json") or []
    nb = _load(DATA_DIR / "null_b" / "summary.json") or []
    if not m0:
        return None

    fig = plt.figure(figsize=(W_FULL, 145 * MM), facecolor="white")
    gs = fig.add_gridspec(
        2, 2, hspace=0.65, wspace=0.40,
        left=0.10, right=0.98, top=0.88, bottom=0.10,
    )

    # a ΔP — PCA primary (filled), LDS hollow markers
    ax = fig.add_subplot(gs[0, 0])
    aliases = [s["alias"] for s in m0 if s.get("arm") == "real"]
    x = np.arange(len(aliases))
    pca_y, lds_y = [], []
    for alias in aliases:
        sess = next(s for s in m0 if s["alias"] == alias)
        te_p = ((sess.get("methods") or {}).get("pca") or {}).get("targets_eval") or []
        te_l = ((sess.get("methods") or {}).get("lds") or {}).get("targets_eval") or []
        pca_y.append((te_p[0].get("delta_p") or 0) if te_p else np.nan)
        lds_y.append((te_l[0].get("delta_p") or 0) if te_l else np.nan)
    ax.bar(x - 0.18, pca_y, width=0.36, color=ACCENT, label="PCA")
    ax.scatter(x + 0.18, lds_y, s=28, facecolors="none", edgecolors="#333",
               linewidths=1.1, zorder=3, label="LDS")
    ax.axhline(0.10, color="gray", ls="--", lw=0.9)
    y_hi = max(0.15, np.nanmax(pca_y + lds_y)) * 1.18
    ax.set_ylim(0, y_hi)
    ax.set_xticks(x)
    ax.set_xticklabels(aliases, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("ΔP vs best neighbour", fontsize=8)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.set_title("a  Held-out separability (PCA primary; LDS hollow)",
                 loc="left", fontsize=8, fontweight="bold", pad=4)

    # b calibration — PCA solid, LDS hollow
    ax = fig.add_subplot(gs[0, 1])
    for sess in m0:
        if sess["arm"] != "real":
            continue
        for method, style in (("pca", dict(marker="o", ls="-", alpha=0.7)),
                              ("lds", dict(marker="o", ls="--", alpha=0.55,
                                           fillstyle="none"))):
            te = ((sess.get("methods") or {}).get(method) or {}).get("targets_eval") or []
            if not te:
                continue
            rel = te[0].get("reliability") or []
            px = [r["p_mean"] for r in rel if r.get("p_mean") is not None]
            py = [r["y_mean"] for r in rel if r.get("y_mean") is not None]
            if px:
                ax.plot(px, py, ms=3, lw=0.8, color=ACCENT if method == "pca" else "#555",
                        **style)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted P", fontsize=8)
    ax.set_ylabel("observed fraction", fontsize=8)
    ax.set_title("b  Calibration (PCA solid; LDS hollow)", loc="left",
                 fontsize=8, fontweight="bold", pad=4)
    ax.set_aspect("equal", adjustable="box")

    # c / d: surrogate (null b) — PCA and LDS separately (revision C1/C2).
    # Threshold: PCA at most 20% of N; LDS at most 20% of N (N=20 PCA, N=5 LDS).
    def _surrogate_panel(ax, method: str, n_expected: int, title_prefix: str) -> None:
        rows = [
            r for r in nb
            if isinstance(r, dict)
            and isinstance(r.get("alias"), str)
            and r["alias"].startswith("S")
            and not r["alias"].startswith("SIM")
            and r.get("method") == method
            and "error" not in r
        ]
        rows = sorted(rows, key=lambda r: r["alias"])
        thresh = max(1, int(round(0.20 * n_expected)))
        if rows:
            aliases = [r["alias"] for r in rows]
            npass = [r.get("n_pass", 0) for r in rows]
            # Prefer explicit n_surrogate if recorded; else n_expected.
            cols = []
            for r in rows:
                n_s = int(r.get("n_surrogate") or n_expected)
                lim = max(1, int(round(0.20 * n_s)))
                cols.append(PASS_C if (r.get("n_pass", 0) <= lim) else FAIL_C)
            ax.bar(aliases, npass, color=cols)
            ax.axhline(thresh, color="black", ls="--", lw=0.9)
            ax.set_ylim(0, max(n_expected, max(npass) if npass else 0) * 1.08 + 0.2)
            ax.tick_params(axis="x", labelrotation=45, labelsize=7)
            n_ok = sum(1 for c in cols if c == PASS_C)
            ax.set_title(
                f"{title_prefix}  ·  PASS {n_ok}/{len(rows)} (need ≤{thresh}/{n_expected})",
                loc="left", fontsize=8, fontweight="bold", pad=4,
            )
        else:
            ax.set_title(
                f"{title_prefix}  ·  pending",
                loc="left", fontsize=8, fontweight="bold", pad=4,
            )
        ax.set_ylabel(f"# surrogate passes / {n_expected}", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    _surrogate_panel(ax, "pca", 20, "c  Surrogate: PCA")
    ax = fig.add_subplot(gs[1, 1])
    _surrogate_panel(ax, "lds", 5, "d  Surrogate: LDS")

    fig.suptitle(
        "Detection on held-out data; the surrogate asks whether coordination "
        "between units is required",
        fontsize=9, fontweight="bold", y=0.98,
    )
    # silence unused null_a load until supplement consumes it
    _ = na
    return _save(fig, "F6_3")


# ---------------------------------------------------------------------------
# F6.4 split + boot
# ---------------------------------------------------------------------------

def fig64_split_boot() -> Path | None:
    """Split-pop (a) + hard-split d-grid panels (b-f) + W and per-axis (g-h)."""
    sp = _load(DATA_DIR / "split" / "summary.json") or []
    loc = _pca_level_table()
    w1 = _load(DATA_DIR / "location" / "pca_level_table_w1s.json")
    w05 = _load(DATA_DIR / "location" / "pca_level_table_w500ms.json")
    w025 = _load(DATA_DIR / "location" / "pca_level_table.json")
    if not sp and not loc:
        return None

    fig = plt.figure(figsize=(W_FULL, 240 * MM), facecolor="white")
    gs = fig.add_gridspec(
        4, 2, hspace=0.55, wspace=0.35,
        left=0.10, right=0.98, top=0.94, bottom=0.04,
        height_ratios=[0.9, 1.0, 1.0, 0.9],
    )

    # (a) split-population
    ax = fig.add_subplot(gs[0, 0])
    rows = [r for r in sp if r.get("kappa") is not None]
    rows = sorted(rows, key=lambda r: (r["alias"], r.get("method", "")))
    if rows:
        xs = [f"{r['alias']}:{r.get('method','')}" for r in rows]
        ys = [r["kappa"] for r in rows]
        cols = [PASS_C if r.get("pass") else FAIL_C for r in rows]
        ax.barh(range(len(xs)), ys, color=cols)
        ax.set_yticks(range(len(xs)))
        ax.set_yticklabels(xs, fontsize=6)
        ax.axvline(0.2, color="black", ls="--", lw=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Cohen kappa", fontsize=7)
        n_ok = sum(1 for r in rows if r.get("pass"))
        ax.set_title(
            f"a  Split-population ({n_ok}/{len(rows)} PASS)",
            loc="left", fontsize=8, fontweight="bold", pad=3,
        )
        ax.invert_yaxis()

    def _s1_u(tab, arm="real"):
        if not tab:
            return []
        return [
            r for r in tab["rows"]
            if r.get("arm") == arm and r.get("split") == "S1" and r.get("drop_u")
        ]

    s1 = _s1_u(loc)

    # (b) n_stable vs d with N1 p95 floor
    ax = fig.add_subplot(gs[0, 1])
    if s1:
        aliases = sorted({r["alias"] for r in s1})
        for alias in aliases:
            sub = sorted(
                [r for r in s1 if r["alias"] == alias],
                key=lambda r: r["d"],
            )
            ax.plot(
                [r["d"] for r in sub], [r["n_stable"] for r in sub],
                "-o", ms=2.5, lw=0.8, alpha=0.75,
            )
            floors = [
                (r["d"], r.get("n_stable_n1_p95"))
                for r in sub if r.get("n_stable_n1_p95") is not None
            ]
            if floors:
                ax.plot(
                    [d for d, _ in floors], [f for _, f in floors],
                    ":", color="#666", lw=0.9, alpha=0.7,
                )
            prim = next((r for r in sub if r.get("is_primary_d")), None)
            if prim:
                ax.axvline(prim["d"], color="#aaa", lw=0.4, alpha=0.3)
        ax.axhline(2, color="black", ls="--", lw=0.8)
        ax.set_xlabel("d", fontsize=7)
        ax.set_ylabel("n_stable", fontsize=7)
        ax.set_title(
            "b  n_stable vs d (dotted = N1 p95; dashed = 2)",
            loc="left", fontsize=8, fontweight="bold", pad=3,
        )
        ax.tick_params(labelsize=6)

    # (c) three pass criteria vs d (fraction of sessions)
    ax = fig.add_subplot(gs[1, 0])
    if s1:
        ds = sorted({r["d"] for r in s1 if r["d"] <= 20 or r.get("is_primary_d")})
        # use grid ds preferentially
        ds_plot = sorted({r["d"] for r in s1 if r["d"] in (2, 3, 5, 10, 20)})
        if not ds_plot:
            ds_plot = sorted({r["d"] for r in s1})
        xs, fr_ns, fr_ari, fr_tile = [], [], [], []
        for d in ds_plot:
            sub = [r for r in s1 if r["d"] == d]
            if not sub:
                continue
            xs.append(d)
            fr_ns.append(np.mean([bool(r.get("pass_n_stable")) for r in sub]))
            fr_ari.append(np.mean([
                bool(r.get("pass_ari")) for r in sub if r.get("pass_ari") is not None
            ] if any(r.get("pass_ari") is not None for r in sub) else [0]))
            fr_tile.append(np.mean([
                bool(r.get("pass_tiles")) for r in sub if r.get("pass_tiles") is not None
            ] if any(r.get("pass_tiles") is not None for r in sub) else [0]))
        ax.plot(xs, fr_ns, "-o", label="same subsp.", ms=4)
        ax.plot(xs, fr_ari, "-s", label="same tiles", ms=4)
        ax.plot(xs, fr_tile, "-^", label="resolvable", ms=4)
        # Report 2 primary ds as ticks
        prims = sorted({r["primary_d"] for r in s1})
        for pd in prims[:3]:
            ax.axvline(pd, color="#bbb", ls=":", lw=0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("d", fontsize=7)
        ax.set_ylabel("fraction PASS", fontsize=7)
        ax.legend(fontsize=6, frameon=False, loc="best")
        ax.set_title(
            "c  Criteria vs d (Report 2 primary d dotted)",
            loc="left", fontsize=8, fontweight="bold", pad=3,
        )

    # (d) per-axis rigid error for example session at d=3
    ax = fig.add_subplot(gs[1, 1])
    ex = next((r for r in s1 if r["alias"] == "S03" and r["d"] == 3), None)
    if ex is None and s1:
        ex = next((r for r in s1 if r["d"] == 3), s1[0])
    if ex and ex.get("e_rigid_per_axis_frac"):
        e_ax = ex["e_rigid_per_axis_frac"]
        ax.bar(range(1, len(e_ax) + 1), e_ax, color=ACCENT)
        ax.set_xlabel("aligned axis", fontsize=7)
        ax.set_ylabel("median |e_k| / scale_k", fontsize=7)
        ax.set_title(
            f"d  Per-axis rigid error ({ex['alias']}, d={ex['d']})",
            loc="left", fontsize=8, fontweight="bold", pad=3,
        )
        # S0 ceiling from same alias
        s0 = next(
            (r for r in (loc or {}).get("rows", [])
             if r.get("alias") == ex["alias"] and r.get("split") == "S0"
             and r.get("drop_u") and r.get("d") == ex["d"]),
            None,
        )
        if s0 and s0.get("e_rigid_median_all") is not None:
            ax.axhline(s0["e_rigid_median_all"], color="gray", ls="--", lw=0.8,
                       label="S0 e_med")
            ax.legend(fontsize=6, frameon=False)

    # (e) ARI by session at d=3 for S1/S2/S0 + N1 p95
    ax = fig.add_subplot(gs[2, 0])
    if loc:
        aliases = sorted({
            r["alias"] for r in loc["rows"]
            if r.get("arm") == "real"
        })
        width = 0.25
        x = np.arange(len(aliases))
        for i, split in enumerate(("S1", "S2", "S0")):
            ys, yerr_lo = [], []
            for a in aliases:
                r = next(
                    (rr for rr in loc["rows"]
                     if rr.get("alias") == a and rr.get("split") == split
                     and rr.get("drop_u") and rr.get("d") == 3),
                    None,
                )
                ys.append(np.nan if r is None else r.get("ari"))
            ax.bar(x + (i - 1) * width, ys, width=width, label=split)
        # N1 p95 markers for S1
        for j, a in enumerate(aliases):
            r = next(
                (rr for rr in loc["rows"]
                 if rr.get("alias") == a and rr.get("split") == "S1"
                 and rr.get("drop_u") and rr.get("d") == 3),
                None,
            )
            if r and r.get("ari_n1_p95") is not None:
                ax.plot(j, r["ari_n1_p95"], "k_", ms=8)
        ax.set_xticks(x)
        ax.set_xticklabels(aliases, fontsize=6, rotation=45, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=6, frameon=False, ncol=3)
        ax.set_title(
            "e  ARI at d=3 (N1 p95 as black ticks on S1)",
            loc="left", fontsize=8, fontweight="bold", pad=3,
        )

    # (f) hard-split conjunction pass counts (real vs sim)
    ax = fig.add_subplot(gs[2, 1])
    def _conj_pass(tab, arm: str, *, primary: bool = False, d: int | None = None) -> tuple[int, int]:
        if not tab:
            return 0, 0
        rows_ = [
            r for r in tab["rows"]
            if r.get("arm") == arm and r.get("split") == "S1" and r.get("drop_u")
            and ((primary and r.get("is_primary_d")) or (d is not None and r.get("d") == d))
        ]
        n_ok = sum(
            1 for r in rows_
            if r.get("pass_n_stable") and r.get("pass_ari") and r.get("pass_tiles")
        )
        return n_ok, len(rows_)

    labels = ["real@Rd", "real@d3", "sim@Rd"]
    vals = [
        _conj_pass(loc, "real", primary=True),
        _conj_pass(loc, "real", d=3),
        _conj_pass(loc, "sim", primary=True),
    ]
    ys = [n / m if m else 0 for n, m in vals]
    bars = ax.bar(labels, ys, color=[ACCENT, ACCENT, "#999"])
    for b, (n, m) in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{n}/{m}", ha="center", fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("fraction PASS", fontsize=7)
    ax.set_title(
        "f  Hard-split conjunction (real vs sim)",
        loc="left", fontsize=8, fontweight="bold", pad=3,
    )

    # (g) r/sigma vs W — result stated in title
    ax = fig.add_subplot(gs[3, 0])
    w_tabs = []
    for W, tab in ((0.25, w025 or loc), (0.5, w05), (1.0, w1)):
        if not tab:
            continue
        w_tabs.append((W, tab))
    ratios = []
    if w_tabs:
        aliases = sorted({
            r["alias"] for _, tab in w_tabs for r in tab["rows"]
            if r.get("arm") == "real" and r.get("split") == "S1"
            and r.get("drop_u") and r.get("d") == 3
        })
        by_w = {W: {} for W, _ in w_tabs}
        for a in aliases:
            xs, ys = [], []
            for W, tab in w_tabs:
                r = next(
                    (rr for rr in tab["rows"]
                     if rr.get("alias") == a and rr.get("split") == "S1"
                     and rr.get("drop_u") and rr.get("d") == 3),
                    None,
                )
                if r and r.get("median_r_over_sigma_noise") is not None:
                    xs.append(W)
                    ys.append(r["median_r_over_sigma_noise"])
                    by_w[W][a] = r["median_r_over_sigma_noise"]
            if xs:
                ax.plot(xs, ys, "-o", ms=3, lw=0.8, alpha=0.8)
        if 0.25 in by_w and 1.0 in by_w:
            for a in aliases:
                if a in by_w[0.25] and a in by_w[1.0] and by_w[0.25][a]:
                    ratios.append(by_w[1.0][a] / by_w[0.25][a])
        ax.axhline(2, color="black", ls="--", lw=0.9)
        ax.set_xlabel("W (s)", fontsize=7)
        ax.set_ylabel("median r / sigma", fontsize=7)
        med = float(np.median(ratios)) if ratios else float("nan")
        n15 = sum(1 for x in ratios if x >= 1.5)
        ax.set_title(
            f"g  r/sigma vs W (median 1 s / 250 ms = {med:.2f}; "
            f"{n15}/{len(ratios)} ≥ 1.5)",
            loc="left", fontsize=7.5, fontweight="bold", pad=3,
        )

    # (h) example-session scatter: first two aligned axes
    ax = fig.add_subplot(gs[3, 1])
    _plot_example_tile_scatter(ax, alias="S07", d=3)

    fig.suptitle(
        "Population agreement and hard-split reproducibility "
        "(PCA; pre-registered conjunction at Report 2 d)",
        fontsize=9, fontweight="bold", y=0.985,
    )
    return _save(fig, "F6_4")


def _plot_example_tile_scatter(ax, *, alias: str = "S07", d: int = 3) -> None:
    """A-tiles colored; aligned B centroids overlaid; matched pairs joined."""
    try:
        from sklearn.cluster import KMeans
        from scipy.optimize import linear_sum_assignment
        from agents.report6.io_cohort import build_alias_map, iter_sessions
        from agents.report6.location.pca_core import fit_pca_half, rigid_R
        from agents.report6.location.split_summary import (
            load_cfg, purge_g_seconds, s1_masks, tau_ac_seconds,
        )
        from agents.report6.paths import DT_S
        from agents.report6.target_rule import choose_k
    except Exception as e:
        ax.set_axis_off()
        ax.text(0.5, 0.5, f"scatter unavailable\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=7)
        return

    build_alias_map(force=False)
    cfg = load_cfg()
    sess = next((s for s in iter_sessions(arm="real") if s.alias == alias), None)
    if sess is None or sess.X_counts is None:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "session missing", ha="center", transform=ax.transAxes)
        return
    X = np.asarray(sess.X_counts, float)
    retained = np.asarray(sess.retained, bool)
    dt = float(cfg.get("dt_s", DT_S))
    tau = tau_ac_seconds(X, retained, dt_s=dt)
    g = purge_g_seconds(tau, window_s=0.25, purge_cap_s=float(cfg.get("purge_cap_s", 5)))
    masks = s1_masks(X.shape[0], retained, block_s=60.0, g_s=g, dt_s=dt)
    ma, mb = masks["mask_A"], masks["mask_B"]
    fa = fit_pca_half(X, ma, d=d, drop_u=True)
    fb = fit_pca_half(X, mb, d=d, drop_u=True)
    R = rigid_R(fa["V"], fb["V"])
    Ca = fa["C"]
    Cb = fb["C"] @ R
    M, _ = choose_k(Ca[ma], seed=0)
    M = max(3, min(int(M), 8))
    km_a = KMeans(n_clusters=M, n_init=10, random_state=0).fit(Ca[ma])
    km_b = KMeans(n_clusters=M, n_init=10, random_state=1).fit(Cb[mb])
    mu_a, mu_b = km_a.cluster_centers_, km_b.cluster_centers_
    cost = np.linalg.norm(mu_a[:, None, :] - mu_b[None, :, :], axis=2)
    ra, rb = linear_sum_assignment(cost)
    pts = Ca[ma]
    labs = km_a.labels_
    cmap = plt.colormaps["tab10"].resampled(M)
    for k in range(M):
        sel = labs == k
        ax.scatter(pts[sel, 0], pts[sel, 1], s=4, color=cmap(k), alpha=0.35, linewidths=0)
    for i, j in zip(ra, rb):
        ax.plot([mu_a[i, 0], mu_b[j, 0]], [mu_a[i, 1], mu_b[j, 1]],
                color="#333", lw=0.8, alpha=0.8)
        ax.scatter(mu_a[i, 0], mu_a[i, 1], s=36, marker="s",
                   facecolors=cmap(i), edgecolors="black", linewidths=0.6, zorder=4)
        ax.scatter(mu_b[j, 0], mu_b[j, 1], s=36, marker="o",
                   facecolors="white", edgecolors=cmap(i), linewidths=1.2, zorder=5)
    ax.set_xlabel("aligned axis 1", fontsize=7)
    ax.set_ylabel("aligned axis 2", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(
        f"h  {alias} d={d}: A-tiles (sq) + aligned B centroids (o)",
        loc="left", fontsize=7.5, fontweight="bold", pad=3,
    )


# ---------------------------------------------------------------------------
# F6.5 plant
# ---------------------------------------------------------------------------

def fig65_plant() -> Path | None:
    pl = _load(DATA_DIR / "plant" / "summary.json")
    if not pl:
        return None
    fig = plt.figure(figsize=(W_FULL, 130 * MM), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, wspace=0.45,
        left=0.10, right=0.96, top=0.86, bottom=0.12,
    )
    ax = fig.add_subplot(gs[0, 0])
    methods = [r["method"] for r in pl]
    hits = [r.get("hit_rate") or 0 for r in pl]
    cols = [PASS_C if (r.get("hit_rate") or 0) >= 0.5 else FAIL_C for r in pl]
    ax.bar(methods, hits, color=cols)
    ax.axhline(0.5, color="black", ls="--", lw=0.9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("hit rate", fontsize=8)
    ax.set_title("a  Planted-window hit rate", loc="left", fontsize=8, fontweight="bold", pad=4)
    for m, h in zip(methods, hits):
        ax.text(m, min(h + 0.04, 1.10), f"{h:.2f}", ha="center", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    fas = [r.get("fa_per_min") if r.get("fa_per_min") is not None else 0 for r in pl]
    cols = [PASS_C if f <= 2.0 else FAIL_C for f in fas]
    ax.bar(methods, fas, color=cols)
    ax.axhline(2.0, color="black", ls="--", lw=0.9)
    ax.set_ylabel("false alarms / min", fontsize=8)
    ax.set_title("b  False-alarm rate", loc="left", fontsize=8, fontweight="bold", pad=4)
    ymax = max(2.4, max(fas) * 1.25 if fas else 2.4)
    ax.set_ylim(0, ymax)
    for m, f in zip(methods, fas):
        ax.text(m, f + 0.04 * ymax, f"{f:.2f}", ha="center", fontsize=8)

    alias = pl[0].get("alias", "SIM")
    both = all(r.get("pass") for r in pl)
    badge = "PASS" if both else "mixed"
    fig.suptitle(
        f"Plant-and-recover on matched sim ({alias}): overall {badge}",
        fontsize=9, fontweight="bold", y=0.96,
    )
    return _save(fig, "F6_5")


# ---------------------------------------------------------------------------
# F6.metric — numbers strip for one-glance reading
# ---------------------------------------------------------------------------

def fig66_metric_cards() -> Path:
    """Fig 6.0b: numbers (not PASS/FAIL) for hard-split columns."""
    rows = collect_session_table()
    fig = plt.figure(figsize=(W_FULL, 160 * MM), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()

    def fmt(x, nd=2):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    cols = [
        "S", "occ%", "vis", "ΔP",
        "surrP", "surrL", "κ",
        "ns d3", "ns Rd", "e_d3",
        "r/σ.25", "r/σ1s", "ARI1", "ARI2",
    ]
    cell = [cols]
    for r in rows:
        ns_d3 = r.get("n_stable_d3")
        ns_d3_p = r.get("n_stable_d3_n1_p95")
        ns_p = r.get("n_stable")
        ns_pp = r.get("n_stable_n1_p95")
        ari1 = r.get("ari_s1_d3")
        ari1p = r.get("ari_s1_n1_p95")
        cell.append([
            r["alias"],
            "—" if r["occupancy"] is None else f"{100 * r['occupancy']:.0f}",
            "—" if r["visits"] is None else str(r["visits"]),
            fmt(r.get("delta_p")),
            "—" if r.get("null_b_npass_pca") is None else f"{r['null_b_npass_pca']}/{r.get('null_b_nsurr_pca') or 20}",
            "—" if r.get("null_b_npass_lds") is None else f"{r['null_b_npass_lds']}/{r.get('null_b_nsurr_lds') or 5}",
            fmt(r.get("split_kappa")),
            "—" if ns_d3 is None else f"{ns_d3} ({fmt(ns_d3_p, 1)})",
            "—" if ns_p is None else f"{ns_p} ({fmt(ns_pp, 1)})",
            fmt(r.get("e_rigid_d3")),
            fmt(r.get("r_sigma_250")),
            fmt(r.get("r_sigma_1s")),
            "—" if ari1 is None else f"{fmt(ari1)} ({fmt(ari1p)})",
            fmt(r.get("ari_s2_d3")),
        ])
    table = ax.table(cellText=cell, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.85)
    for j in range(len(cols)):
        table[0, j].set_facecolor("#e8e8e8")
        table[0, j].set_text_props(fontweight="bold")
    fig.suptitle(
        "Fig 6.0b  Session numbers (PCA targets; ns = n_stable with N1 p95; "
        "Rd = Report 2 d; ARI at d = 3)",
        fontsize=8, fontweight="bold", y=0.97,
    )
    return _save(fig, "F6_numbers")


def fig_s_retired() -> Path:
    """Supplement: archived null-(a) and bootstrap panels only (no text block)."""
    na = _load(DATA_DIR / "null_a" / "summary.json") or []
    bt = _load(DATA_DIR / "boot" / "summary.json") or []

    fig = plt.figure(figsize=(W_FULL, 140 * MM), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.0], hspace=0.35,
        left=0.10, right=0.98, top=0.90, bottom=0.08,
    )

    ax = fig.add_subplot(gs[0, 0])
    rows = [
        r for r in na
        if isinstance(r, dict)
        and r.get("method") == "lds"
        and isinstance(r.get("alias"), str)
        and r["alias"].startswith("S")
        and not r["alias"].startswith("SIM")
    ]
    rows = sorted(rows, key=lambda r: r["alias"])
    if rows:
        aliases = [r["alias"] for r in rows]
        npass = [r.get("n_pass", 0) for r in rows]
        cols = [PASS_C if r.get("null_a_pass") else FAIL_C for r in rows]
        ax.bar(aliases, npass, color=cols)
        ax.axhline(4, color="black", ls="--", lw=0.9)
        ax.set_ylim(0, 21)
        ax.set_ylabel("# shift passes / 20", fontsize=8)
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.set_title("a  Null (a) whole-series time shift (retired)",
                 loc="left", fontsize=8, fontweight="bold", pad=4)

    ax = fig.add_subplot(gs[1, 0])
    brows = [r for r in bt if isinstance(r, dict) and r.get("median_jaccard") is not None]
    brows = sorted(brows, key=lambda r: (r["alias"], r.get("method", "")))
    if brows:
        xs = [f"{r['alias']}:{r.get('method','')}" for r in brows]
        ys = [r["median_jaccard"] for r in brows]
        cols = [PASS_C if r.get("pass") else FAIL_C for r in brows]
        ax.barh(range(len(xs)), ys, color=cols)
        ax.set_yticks(range(len(xs)))
        ax.set_yticklabels(xs, fontsize=6)
        ax.axvline(0.3, color="black", ls="--", lw=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("median Jaccard", fontsize=8)
        ax.invert_yaxis()
    ax.set_title("b  Bootstrap refit stability (retired)",
                 loc="left", fontsize=8, fontweight="bold", pad=4)

    fig.suptitle(
        "Supplement: retired pilot tests (panels only; prose in captions)",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_S_retired")


def _s04_half_occupancy() -> tuple[float, float] | None:
    """First vs second half train occupancy of the top LDS target (null-a outlier)."""
    try:
        from agents.report6.io_cohort import build_alias_map, iter_sessions
        from agents.report6.pipeline import latent_for_method
        from agents.report6.target_rule import choose_k, qualify_targets
        from sklearn.cluster import KMeans
    except Exception:
        return None
    build_alias_map(force=False)
    sess = next((s for s in iter_sessions(arm="real") if s.alias == "S04"), None)
    if sess is None:
        return None
    Z, d = latent_for_method(sess, "lds")
    Z3 = Z[:, :3]
    train = np.asarray(sess.train_ok, bool)
    idx = np.where(train)[0]
    if idx.size < 100:
        return None
    mid = idx[idx.size // 2]
    first = train & (np.arange(train.size) < mid)
    second = train & (np.arange(train.size) >= mid)
    M, _ = choose_k(Z3[train], seed=0)
    km = KMeans(n_clusters=int(M), n_init=20, random_state=0).fit(Z3[train])
    labs = km.predict(Z3)
    # top qualifying by occupancy on full train
    from collections import Counter
    counts = Counter(labs[train])
    top = max(counts, key=counts.get)
    occ1 = float((labs[first] == top).mean()) if first.any() else float("nan")
    occ2 = float((labs[second] == top).mean()) if second.any() else float("nan")
    return occ1, occ2


def fig_appendix() -> Path:
    """Appendix: drift check, d-grid hard-split numbers, LDS split-pop, full test table."""
    loc = _pca_level_table()
    sp = _load(DATA_DIR / "split" / "summary.json") or []
    fig = plt.figure(figsize=(W_FULL, 240 * MM), facecolor="white")
    gs = fig.add_gridspec(
        4, 1, height_ratios=[0.55, 1.1, 0.55, 1.6], hspace=0.45,
        left=0.06, right=0.98, top=0.94, bottom=0.03,
    )

    # a drift
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    halves = _s04_half_occupancy()
    lines = [
        "a  Drift check for the null-(a) outlier (S04, LDS top target)",
        "    First-half vs second-half training occupancy (fractions).",
    ]
    if halves:
        lines.append(f"    first half = {halves[0]:.3f}; second half = {halves[1]:.3f}")
    else:
        lines.append("    (could not recompute; see caption)")
    ax.text(0.0, 0.95, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=8, family="monospace")

    # b d-grid table
    ax = fig.add_subplot(gs[1, 0])
    ax.set_axis_off()
    ax.set_title("b  Hard-split numbers across the d grid (S1, u out, real)",
                 loc="left", fontsize=8, fontweight="bold", pad=2)
    if loc:
        ds = [2, 3, 5, 10, 20]
        headers = ["S"] + [f"ns@{d}" for d in ds] + [f"ARI@{d}" for d in (3,)] + ["ns@Rd"]
        cell = [headers]
        for i in range(1, 11):
            a = f"S{i:02d}"
            row = [a]
            for d in ds:
                r = _loc_row(loc, a, d=d)
                row.append("—" if r is None else str(r.get("n_stable")))
            r3 = _loc_row(loc, a, d=3)
            row.append("—" if r3 is None or r3.get("ari") is None else f"{r3['ari']:.2f}")
            rp = _loc_row(loc, a, primary=True)
            row.append("—" if rp is None else str(rp.get("n_stable")))
            cell.append(row)
        table = ax.table(cellText=cell, loc="upper center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(6.5)
        table.scale(1.0, 1.15)

    # c LDS split-pop
    ax = fig.add_subplot(gs[2, 0])
    lds_sp = [r for r in sp if isinstance(r, dict) and r.get("method") == "lds"]
    lds_sp = sorted(lds_sp, key=lambda r: r.get("alias", ""))
    if lds_sp:
        xs = [r["alias"] for r in lds_sp]
        ys = [r.get("kappa") or 0 for r in lds_sp]
        cols = [PASS_C if r.get("pass") else FAIL_C for r in lds_sp]
        ax.bar(xs, ys, color=cols)
        ax.axhline(0.2, color="black", ls="--", lw=0.9)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Cohen kappa", fontsize=8)
    ax.set_title(
        f"c  LDS split-population ({len(lds_sp)} sessions)",
        loc="left", fontsize=8, fontweight="bold", pad=3,
    )

    # d full test table at >=7pt
    ax = fig.add_subplot(gs[3, 0])
    ax.set_axis_off()
    ax.set_title("d  Full pass-rule table (question / how / pass if / fail)",
                 loc="left", fontsize=8, fontweight="bold", pad=2)
    headers = ["test", "question", "pass if", "a fail means"]
    cell = [headers]
    for name, q, _how, pif, fail in FIG61B_ROWS:
        cell.append([
            name,
            textwrap.fill(q, width=36),
            textwrap.fill(pif, width=32),
            textwrap.fill(fail, width=32),
        ])
    table = ax.table(
        cellText=cell, loc="upper center", cellLoc="left",
        colWidths=[0.18, 0.28, 0.27, 0.27],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1.0, 1.55)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#e8e8e8")
        table[0, j].set_text_props(fontweight="bold")

    fig.suptitle("Appendix", fontsize=11, fontweight="bold", y=0.985)
    return _save(fig, "F6_appendix")


def build_all_available() -> dict[str, str]:
    out = {}
    out["F6.0"] = str(fig60_results_board())
    out["F6.1"] = str(fig61_schematic_and_table())
    p = fig62_targets()
    if p:
        out["F6.2"] = str(p)
    p = fig63_detection_nulls()
    if p:
        out["F6.3"] = str(p)
    p = fig64_split_boot()
    if p:
        out["F6.4"] = str(p)
    p = fig65_plant()
    if p:
        out["F6.5"] = str(p)
    out["F6.numbers"] = str(fig66_metric_cards())
    out["F6.S"] = str(fig_s_retired())
    out["F6.appendix"] = str(fig_appendix())
    return out


if __name__ == "__main__":
    print(json.dumps(build_all_available(), indent=2))
