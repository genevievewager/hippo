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
    SUPPLEMENT_BOOT,
    SUPPLEMENT_NULL_A,
    SUPPLEMENT_PANELS_NOTE,
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


def collect_session_table() -> list[dict[str, Any]]:
    """One row per real session with all criterion outcomes (LDS primary)."""
    m0 = _load(DATA_DIR / "m0_m2" / "all.json") or []
    na = _load(DATA_DIR / "null_a" / "summary.json") or []
    nb = _load(DATA_DIR / "null_b" / "summary.json") or []
    sp = _load(DATA_DIR / "split" / "summary.json") or []
    bt = _load(DATA_DIR / "boot" / "summary.json") or []

    rows = []
    for sess in m0:
        if sess.get("arm") != "real":
            continue
        alias = sess["alias"]
        lds = (sess.get("methods") or {}).get("lds") or {}
        te = (lds.get("targets_eval") or [])
        top = te[0] if te else None
        na_r = _idx(na, alias=alias, method="lds")
        nb_lds = _idx(nb, alias=alias, method="lds")
        nb_pca = _idx(nb, alias=alias, method="pca")
        sp_lds = _idx(sp, alias=alias, method="lds")
        sp_pca = _idx(sp, alias=alias, method="pca")
        bt_lds = _idx(bt, alias=alias, method="lds")
        bt_pca = _idx(bt, alias=alias, method="pca")
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
            "null_b": None if (nb_lds or nb_pca) is None else bool(
                (nb_lds or nb_pca).get("null_b_pass")
            ),
            "null_b_method": "lds" if nb_lds is not None else ("pca" if nb_pca else None),
            "null_b_npass": None if (nb_lds or nb_pca) is None else (nb_lds or nb_pca).get("n_pass"),
            "split": None if (sp_lds or sp_pca) is None else bool(
                (sp_lds or sp_pca).get("pass")
            ),
            "split_kappa": None if (sp_lds or sp_pca) is None else (sp_lds or sp_pca).get("kappa"),
            "boot": None if (bt_lds or bt_pca) is None else bool(
                (bt_lds or bt_pca).get("pass")
            ),
            "boot_jaccard": None if (bt_lds or bt_pca) is None else (
                bt_lds or bt_pca
            ).get("median_jaccard"),
            # Hard-split amendment (filled at stop point 2+); None until then.
            "same_subspace": None,
            "same_tiles": None,
        })
    # fill animal from public cohort if missing
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

    fig = plt.figure(figsize=(W_FULL, 155 * MM), facecolor="white")
    gs = fig.add_gridspec(
        3, 1, height_ratios=[0.12, 1.35, 0.9], hspace=0.45,
        left=0.22, right=0.98, top=0.90, bottom=0.06,
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
        "a  Per-session pass/fail (counts per run; surrogate / split / "
        "reproducibility use PCA until LDS runs finish)",
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

    ax2 = fig.add_subplot(gs[2, 0])
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
        "Report 6 pilot — recurring, separable neural targets; surrogate and "
        "hard-split reproducibility carry the \"real state\" claim",
        fontsize=9, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_0_board")


# ---------------------------------------------------------------------------
# F6.1 schematic
# ---------------------------------------------------------------------------

def fig61_schematic_and_table() -> Path:
    """Pipeline (a) + revision B2 pass-rule table (b). Layout fit only."""
    fig = plt.figure(figsize=(W_FULL, 220 * MM), facecolor="white")
    gs = fig.add_gridspec(
        2, 1, height_ratios=[0.55, 2.4], hspace=0.18,
        left=0.03, right=0.99, top=0.93, bottom=0.02,
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
        ax.text(x + 1.45, y + 1.0, t, ha="center", va="center", fontsize=7.5)
    ax.set_title("a  Pipeline", loc="left", fontsize=10, fontweight="bold", pad=4)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_axis_off()
    ax2.set_title(
        "b  Every test: the question, how it is done, pass if, what a fail means",
        loc="left", fontsize=9, fontweight="bold", pad=4,
    )
    headers = ["test", "question", "how", "pass if", "a fail means"]
    cell = [headers]
    for row in FIG61B_ROWS:
        cell.append(list(row))
    table = ax2.table(
        cellText=cell, loc="upper center", cellLoc="left",
        colWidths=[0.13, 0.18, 0.30, 0.20, 0.19],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(5.2)
    table.scale(1.0, 2.35)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#e8e8e8")
        table[0, j].set_text_props(fontweight="bold")
    for i in range(len(cell)):
        for j in range(len(headers)):
            table[i, j].PAD = 0.04
            table[i, j].set_text_props(va="top")

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

    def top_stats(arm: str, method: str):
        aliases, occ, vis, dwell = [], [], [], []
        for sess in data:
            if sess.get("arm") != arm:
                continue
            blob = (sess.get("methods") or {}).get(method) or {}
            te = blob.get("targets_eval") or []
            aliases.append(sess["alias"].replace("SIM", "S") if arm == "sim" else sess["alias"])
            if te:
                t = te[0]["target"]
                occ.append(100 * t["occupancy"])
                vis.append(t["n_visits"])
                dwell.append(t["median_dwell_s"])
            else:
                occ.append(0.0)
                vis.append(0)
                dwell.append(np.nan)
        return aliases, occ, vis, dwell

    panels = [
        (0, 0, "real", "lds", "a  Real · LDS"),
        (0, 1, "real", "pca", "b  Real · PCA"),
        (1, 0, "sim", "lds", "c  Sim (k=0) · LDS"),
        (1, 1, "sim", "pca", "d  Sim (k=0) · PCA"),
    ]
    packed = []
    all_occ, all_vis = [], []
    for spec in panels:
        aliases, occ, vis, _ = top_stats(spec[2], spec[3])
        packed.append((spec, aliases, occ, vis))
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
    for (r, c, _arm, _method, title), aliases, occ, vis in packed:
        ax = fig.add_subplot(gs[r, c])
        x = np.arange(len(aliases))
        w = 0.38
        b = ax.bar(x - w / 2, occ, width=w, color=ACCENT, label="occupancy %")
        h5 = ax.axhline(5, color="gray", ls="--", lw=0.8, label="5% rule")
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
            legend_handles = [b, h5, ln, h10]

    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["occupancy %", "5% rule", "visits", "10 visits"],
            loc="lower center", ncol=4, fontsize=7.5, frameon=False,
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

    # a ΔP
    ax = fig.add_subplot(gs[0, 0])
    xs, ys, cols = [], [], []
    for sess in m0:
        if sess["arm"] != "real":
            continue
        te = ((sess.get("methods") or {}).get("lds") or {}).get("targets_eval") or []
        if not te:
            continue
        xs.append(sess["alias"])
        ys.append(te[0].get("delta_p") or 0)
        cols.append(PASS_C if te[0].get("distinguishable_pass") else FAIL_C)
    ax.bar(xs, ys, color=cols)
    ax.axhline(0.10, color="gray", ls="--", lw=0.9, label="ΔP = 0.10")
    y_hi = max(0.15, max(ys) if ys else 0.15) * 1.18
    ax.set_ylim(0, y_hi)
    ax.set_ylabel("ΔP vs best neighbour", fontsize=8)
    ax.set_title("a  Held-out separability (LDS)  ·  green = PASS",
                 loc="left", fontsize=8, fontweight="bold", pad=4)
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)

    # b calibration
    ax = fig.add_subplot(gs[0, 1])
    for sess in m0:
        if sess["arm"] != "real":
            continue
        te = ((sess.get("methods") or {}).get("lds") or {}).get("targets_eval") or []
        if not te:
            continue
        rel = te[0].get("reliability") or []
        px = [r["p_mean"] for r in rel if r.get("p_mean") is not None]
        py = [r["y_mean"] for r in rel if r.get("y_mean") is not None]
        if px:
            ax.plot(px, py, "o-", ms=3, lw=0.8, alpha=0.55)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("predicted P", fontsize=8)
    ax.set_ylabel("observed fraction", fontsize=8)
    ax.set_title("b  Calibration (real LDS, 1 line / session)", loc="left",
                 fontsize=8, fontweight="bold", pad=4)
    ax.set_aspect("equal", adjustable="box")

    # c / d: surrogate (null b) — PCA and LDS separately (revision C1/C2).
    # Threshold: PCA at most 20% of N; LDS at most 20% of N (N=20 PCA, N=5 LDS).
    def _surrogate_panel(ax, method: str, n_expected: int, title_prefix: str) -> None:
        rows = [
            r for r in nb
            if isinstance(r.get("alias"), str)
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
    _surrogate_panel(ax, "pca", 20, "c  Surrogate — PCA")
    ax = fig.add_subplot(gs[1, 1])
    _surrogate_panel(ax, "lds", 5, "d  Surrogate — LDS")

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
    """Panel (a) split-population only; hard-split panels fill in after amendment."""
    sp = _load(DATA_DIR / "split" / "summary.json") or []
    if not sp:
        return None

    fig = plt.figure(figsize=(W_FULL, 95 * MM), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, wspace=0.40,
        left=0.16, right=0.98, top=0.84, bottom=0.12,
    )

    ax = fig.add_subplot(gs[0, 0])
    rows = [r for r in sp if r.get("kappa") is not None]
    rows = sorted(rows, key=lambda r: (r["alias"], r.get("method", "")))
    if rows:
        xs = [f"{r['alias']}:{r.get('method','')}" for r in rows]
        ys = [r["kappa"] for r in rows]
        cols = [PASS_C if r.get("pass") else FAIL_C for r in rows]
        ax.barh(range(len(xs)), ys, color=cols)
        ax.set_yticks(range(len(xs)))
        ax.set_yticklabels(xs, fontsize=7)
        ax.axvline(0.2, color="black", ls="--", lw=0.9)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Cohen kappa", fontsize=8)
        n_ok = sum(1 for r in rows if r.get("pass"))
        ax.set_title(
            f"a  Split-population  ({n_ok}/{len(rows)} PASS; per run)",
            loc="left", fontsize=8, fontweight="bold", pad=4,
        )
        ax.invert_yaxis()

    ax = fig.add_subplot(gs[0, 1])
    ax.set_axis_off()
    ax.text(
        0.02, 0.95,
        "b–f  Hard-split reproducibility\n(pending amendment stop point 2)",
        transform=ax.transAxes, va="top", fontsize=9, fontweight="bold",
    )
    ax.text(
        0.02, 0.72,
        "Alternating 60 s blocks (S1), first vs second half (S2),\n"
        "principal angles, coordinate error, ARI, matched tiles.\n"
        "Bootstrap refit panel retired — see supplement.",
        transform=ax.transAxes, va="top", fontsize=8, color="#333",
    )

    fig.suptitle(
        "Population agreement (split-pop) and hard-split reproducibility "
        "(amendment)",
        fontsize=9, fontweight="bold", y=0.97,
    )
    return _save(fig, "F6_4")


# ---------------------------------------------------------------------------
# F6.5 plant
# ---------------------------------------------------------------------------

def fig65_plant() -> Path | None:
    pl = _load(DATA_DIR / "plant" / "summary.json")
    if not pl:
        return None
    fig = plt.figure(figsize=(W_FULL, 85 * MM), facecolor="white")
    gs = fig.add_gridspec(
        1, 2, wspace=0.45,
        left=0.10, right=0.96, top=0.82, bottom=0.14,
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
        f"Plant-and-recover on matched sim ({alias}) — overall {badge}",
        fontsize=9, fontweight="bold", y=0.96,
    )
    return _save(fig, "F6_5")


# ---------------------------------------------------------------------------
# F6.metric — numbers strip for one-glance reading
# ---------------------------------------------------------------------------

def fig66_metric_cards() -> Path:
    rows = collect_session_table()
    fig = plt.figure(figsize=(W_FULL, 90 * MM), facecolor="white")
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()

    cols = [
        "Session", "occ%", "visits", "delta P",
        "surrogate", "split", "subspace", "tiles",
    ]
    cell = [cols]
    for r in rows:
        cell.append([
            r["alias"],
            "—" if r["occupancy"] is None else f"{100*r['occupancy']:.0f}",
            "—" if r["visits"] is None else str(r["visits"]),
            "—" if r["delta_p"] is None else f"{r['delta_p']:.2f}",
            _yn(r["null_b"]),
            _yn(r["split"]),
            _yn(r.get("same_subspace")),
            _yn(r.get("same_tiles")),
        ])
    table = ax.table(cellText=cell, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    for i in range(1, len(cell)):
        for j in (4, 5, 6, 7):
            val = cell[i][j]
            if val == "PASS":
                table[i, j].set_facecolor("#c6efce")
            elif val == "FAIL":
                table[i, j].set_facecolor("#ffc7ce")
    fig.suptitle(
        "Report 6 numbers (real cohort; counts per run; subspace/tiles pending "
        "hard-split)",
        fontsize=9, fontweight="bold", y=0.96,
    )
    return _save(fig, "F6_numbers")


def fig_s_retired() -> Path:
    """Supplement: B4 text + archived null-(a) and bootstrap panels."""
    na = _load(DATA_DIR / "null_a" / "summary.json") or []
    bt = _load(DATA_DIR / "boot" / "summary.json") or []

    fig = plt.figure(figsize=(W_FULL, 200 * MM), facecolor="white")
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.1, 1.0, 1.0], hspace=0.35,
        left=0.08, right=0.98, top=0.92, bottom=0.04,
    )

    ax_t = fig.add_subplot(gs[0, 0])
    ax_t.set_axis_off()
    ax_t.text(0.0, 0.98, "Retired tests (pilot)", fontsize=11, fontweight="bold",
              transform=ax_t.transAxes, va="top")
    y = 0.88
    for para in (SUPPLEMENT_NULL_A, SUPPLEMENT_BOOT, SUPPLEMENT_PANELS_NOTE):
        lines = textwrap.wrap(para, width=108)
        for line in lines:
            ax_t.text(0.0, y, line, fontsize=6.8, transform=ax_t.transAxes, va="top")
            y -= 0.055
        y -= 0.04

    # Old Fig 6.3c — null (a)
    ax = fig.add_subplot(gs[1, 0])
    rows = [
        r for r in na
        if r.get("method") == "lds"
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
    ax.set_title("Old Fig 6.3c  Null (a) whole-series time shift (retired)",
                 loc="left", fontsize=8, fontweight="bold", pad=4)

    # Old Fig 6.4b — bootstrap
    ax = fig.add_subplot(gs[2, 0])
    brows = [r for r in bt if r.get("median_jaccard") is not None]
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
    ax.set_title("Old Fig 6.4b  Bootstrap refit stability (retired)",
                 loc="left", fontsize=8, fontweight="bold", pad=4)

    fig.suptitle(
        "Supplement — retired pilot tests (kept for the record)",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_S_retired")


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
    return out


if __name__ == "__main__":
    print(json.dumps(build_all_available(), indent=2))
