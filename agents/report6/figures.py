"""Report 6 figures — rebuilt for easy interpretation of pilot results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, Patch

from agents.report6.paths import OUT_ROOT

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
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
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

    criteria = [
        ("qualifies", "Recurs\n(qual target)"),
        ("distinguishable", "Distinguishable\n(held-out ΔP)"),
        ("null_a", "Fails null (a)\ntime shift"),
        ("null_b", "Fails null (b)\nsurrogate"),
        ("split", "Split-pop κ"),
        ("boot", "Refit Jaccard"),
    ]
    aliases = [r["alias"] for r in rows]
    n_s, n_c = len(aliases), len(criteria)
    # 1=PASS, 0=FAIL, -1=NA
    M = np.full((n_s, n_c), -1.0)
    for i, r in enumerate(rows):
        for j, (key, _) in enumerate(criteria):
            v = r.get(key)
            if v is None:
                M[i, j] = -1
            else:
                M[i, j] = 1 if v else 0

    fig = plt.figure(figsize=(W_FULL, 145 * MM), facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.35, 0.85], hspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    cmap = ListedColormap([FAIL_C, NA_C, PASS_C])
    # map -1,0,1 → 0,1,2
    show = M.copy()
    show[M < 0] = 0.5  # will remap
    Z = np.where(M < 0, 1, np.where(M > 0, 2, 0))
    ax.imshow(Z.T, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(n_s))
    ax.set_xticklabels(aliases, fontsize=9)
    ax.set_yticks(range(n_c))
    ax.set_yticklabels([c[1] for c in criteria], fontsize=8)
    for i in range(n_s):
        for j in range(n_c):
            lab = {1: "PASS", 0: "FAIL", -1: "—"}.get(int(M[i, j]) if M[i, j] >= 0 else -1, "—")
            if M[i, j] < 0:
                lab = "—"
            elif M[i, j] > 0:
                lab = "PASS"
            else:
                lab = "FAIL"
            ax.text(i, j, lab, ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] >= 0 else "#555", fontweight="bold")
    ax.set_title(
        "a  Per-session pass/fail (LDS primary; null_b/split/boot use PCA when LDS not run)",
        loc="left", fontsize=9, fontweight="bold",
    )
    ax.legend(
        handles=[
            Patch(facecolor=PASS_C, label="PASS"),
            Patch(facecolor=FAIL_C, label="FAIL"),
            Patch(facecolor=NA_C, label="not run"),
        ],
        loc="upper right", fontsize=7, frameon=False,
        bbox_to_anchor=(1.0, 1.18), ncol=3,
    )

    # headline numbers
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_axis_off()
    n = max(len(rows), 1)
    frac_q = sum(1 for r in rows if r["qualifies"]) / n
    frac_d = sum(1 for r in rows if r["qualifies"] and r["distinguishable"]) / n
    frac_a = sum(
        1 for r in rows
        if r["qualifies"] and r["distinguishable"] and r.get("null_a") is True
    ) / n
    plant_hit = None if plant_pca is None else plant_pca.get("hit_rate")
    plant_fa = None if plant_pca is None else plant_pca.get("fa_per_min")

    boxes = [
        (0.02, f"{frac_q*100:.0f}%", "Qualifying LDS\ntarget (n=10)"),
        (0.27, f"{frac_d*100:.0f}%", "Also held-out\ndistinguishable"),
        (0.52, f"{frac_a*100:.0f}%", "Also fails\nnull (a)"),
        (0.77,
         "—" if plant_hit is None else f"{plant_hit*100:.0f}%",
         f"Plant hit rate\n(FA/min={plant_fa})"),
    ]
    for x, big, small in boxes:
        ax2.add_patch(FancyBboxPatch(
            (x, 0.15), 0.22, 0.7, transform=ax2.transAxes,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#f7f7f7", edgecolor="#333", lw=1.0,
            mutation_aspect=0.3,
        ))
        ax2.text(x + 0.11, 0.58, big, transform=ax2.transAxes,
                 ha="center", va="center", fontsize=18, fontweight="bold")
        ax2.text(x + 0.11, 0.28, small, transform=ax2.transAxes,
                 ha="center", va="center", fontsize=7.5, color="#333")
    ax2.set_title("b  Three numbers for the deck (+ plant)", loc="left",
                  fontsize=9, fontweight="bold")

    fig.suptitle(
        "Report 6 pilot — what passed? Density targets recur and separate, "
        "but circular time-shift rarely kills them",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_0_board")


# ---------------------------------------------------------------------------
# F6.1 schematic
# ---------------------------------------------------------------------------

def fig61_schematic_and_table() -> Path:
    fig = plt.figure(figsize=(W_FULL, 110 * MM), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.25], wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    boxes = [
        (0.4, 7.4, "M0  Causal latents\n(LDS / PCA cache)"),
        (0.4, 4.7, "M1  Target rule\n(train density peaks)"),
        (0.4, 2.0, "M2  Detector\n(held-out only)"),
        (5.4, 7.4, "Nulls (a)/(b)"),
        (5.4, 4.7, "Split κ + refit"),
        (5.4, 2.0, "Plant & recover"),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), 3.9, 2.0, boxstyle="round,pad=0.05,rounding_size=0.15",
            facecolor="#f3f3f3", edgecolor="#333", lw=1.0,
        ))
        ax.text(x + 1.95, y + 1.0, t, ha="center", va="center", fontsize=8)
    ax.set_title("a  Pipeline", loc="left", fontsize=10, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    rows = [
        ("Recurs", "≥5% occupancy, ≥10 visits"),
        ("Distinguishable", "ΔP ≥ 0.10, Brier ≤ 0.25"),
        ("Population", "κ ≥ 0.2 and > chance"),
        ("Refit-stable", "median Jaccard ≥ 0.3"),
        ("Fails null (a)", "≤ 4 / 20 time-shift passes"),
        ("Fails null (b)", "≤ 1 / 5 surrogate passes"),
        ("Plant/recover", "hit ≥ 0.5, FA ≤ 2 / min"),
    ]
    ax2.set_title("b  Pre-registered pass rules", loc="left", fontsize=10, fontweight="bold")
    y = 0.92
    ax2.text(0.0, y, "Criterion", fontsize=8, fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.38, y, "Pass if…", fontsize=8, fontweight="bold", transform=ax2.transAxes)
    y -= 0.09
    for c, r in rows:
        ax2.text(0.0, y, c, fontsize=7.5, transform=ax2.transAxes, va="top")
        ax2.text(0.38, y, r, fontsize=7.5, transform=ax2.transAxes, va="top")
        y -= 0.11
    fig.suptitle(
        "A neural-state target counts only if it recurs, separates, and fails nulls",
        fontsize=10, fontweight="bold", y=0.98,
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

    fig = plt.figure(figsize=(W_FULL, 130 * MM), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.3)

    panels = [
        (0, 0, "real", "lds", "a  Real · LDS occupancy & visits"),
        (0, 1, "real", "pca", "b  Real · PCA"),
        (1, 0, "sim", "lds", "c  Sim (k=0) · LDS"),
        (1, 1, "sim", "pca", "d  Sim (k=0) · PCA"),
    ]
    for r, c, arm, method, title in panels:
        ax = fig.add_subplot(gs[r, c])
        aliases, occ, vis, _ = top_stats(arm, method)
        x = np.arange(len(aliases))
        w = 0.38
        ax.bar(x - w / 2, occ, width=w, color=ACCENT, label="occupancy %")
        ax.axhline(5, color="gray", ls="--", lw=0.8, label="5% rule")
        ax.set_ylabel("occupancy %", fontsize=7, color=ACCENT)
        ax.tick_params(axis="y", labelcolor=ACCENT, labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(x, vis, "o-", color=ACCENT2, ms=4, lw=1.2, label="visits")
        ax2.set_ylabel("visits", fontsize=7, color=ACCENT2)
        ax2.tick_params(axis="y", labelcolor=ACCENT2, labelsize=7)
        ax2.axhline(10, color=ACCENT2, ls=":", lw=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(aliases, fontsize=6, rotation=45, ha="right")
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        if r == 0 and c == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right", frameon=False)

    n_real = sum(1 for s in data if s["arm"] == "real")
    n_sim = sum(1 for s in data if s["arm"] == "sim")
    fig.suptitle(
        f"Top qualifying target per session (n_real={n_real}, n_sim={n_sim}). "
        "Dashed = 5% occ.; dotted = 10 visits.",
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

    fig = plt.figure(figsize=(W_FULL, 125 * MM), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)

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
    ax.axhline(0.10, color="gray", ls="--", lw=0.9)
    ax.set_ylabel("ΔP vs best neighbour")
    ax.set_title("a  Held-out separability (LDS)", loc="left", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
    ax.text(0.98, 0.95, "green = PASS", transform=ax.transAxes, ha="right",
            va="top", fontsize=7, color=PASS_C)

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
    ax.set_xlabel("predicted P")
    ax.set_ylabel("observed fraction")
    ax.set_title("b  Calibration (real LDS, 1 line / session)", loc="left",
                 fontsize=8, fontweight="bold")

    # c null a
    ax = fig.add_subplot(gs[1, 0])
    rows = [r for r in na if r.get("method") == "lds" and str(r.get("alias", "")).startswith("S")]
    rows = sorted(rows, key=lambda r: r["alias"])
    if rows:
        aliases = [r["alias"] for r in rows]
        npass = [r.get("n_pass", 0) for r in rows]
        cols = [PASS_C if r.get("null_a_pass") else FAIL_C for r in rows]
        ax.bar(aliases, npass, color=cols)
        ax.axhline(4, color="black", ls="--", lw=0.9)
        ax.set_ylabel("# shift passes / 20")
        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
        n_ok = sum(1 for r in rows if r.get("null_a_pass"))
        ax.text(0.98, 0.95, f"PASS {n_ok}/{len(rows)}  (need ≤4)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_title("c  Null (a) circular time shift — LDS", loc="left",
                 fontsize=8, fontweight="bold")

    # d null b
    ax = fig.add_subplot(gs[1, 1])
    rows = [r for r in nb if str(r.get("alias", "")).startswith("S") and "error" not in r]
    rows = sorted(rows, key=lambda r: (r["alias"], r.get("method", "")))
    if rows:
        labels = [f"{r['alias']}\n{r.get('method','')}" for r in rows]
        npass = [r.get("n_pass", 0) for r in rows]
        cols = [PASS_C if r.get("null_b_pass") else FAIL_C for r in rows]
        ax.bar(range(len(labels)), npass, color=cols)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.axhline(1, color="black", ls="--", lw=0.9)
        ax.set_ylabel("# surrogate passes / 5")
        n_ok = sum(1 for r in rows if r.get("null_b_pass"))
        ax.text(0.98, 0.95, f"PASS {n_ok}/{len(rows)}  (need ≤1)",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set_title("d  Null (b) correlation-break + refit", loc="left",
                 fontsize=8, fontweight="bold")

    fig.suptitle(
        "Detection works on held-out data; null (a) rarely kills density targets; "
        "null (b) usually does (PCA)",
        fontsize=9, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_3")


# ---------------------------------------------------------------------------
# F6.4 split + boot
# ---------------------------------------------------------------------------

def fig64_split_boot() -> Path | None:
    sp = _load(DATA_DIR / "split" / "summary.json") or []
    bt = _load(DATA_DIR / "boot" / "summary.json") or []
    if not sp and not bt:
        return None

    fig = plt.figure(figsize=(W_FULL, 95 * MM), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.35)

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
        ax.set_xlabel("Cohen κ")
        n_ok = sum(1 for r in rows if r.get("pass"))
        ax.set_title(f"a  Split-population  ({n_ok}/{len(rows)} PASS)",
                     loc="left", fontsize=8, fontweight="bold")
        ax.invert_yaxis()

    ax = fig.add_subplot(gs[0, 1])
    rows = [r for r in bt if r.get("median_jaccard") is not None]
    rows = sorted(rows, key=lambda r: (r["alias"], r.get("method", "")))
    if rows:
        xs = [f"{r['alias']}:{r.get('method','')}" for r in rows]
        ys = [r["median_jaccard"] for r in rows]
        cols = [PASS_C if r.get("pass") else FAIL_C for r in rows]
        ax.barh(range(len(xs)), ys, color=cols)
        ax.set_yticks(range(len(xs)))
        ax.set_yticklabels(xs, fontsize=6)
        ax.axvline(0.3, color="black", ls="--", lw=0.9)
        ax.set_xlabel("median Jaccard")
        n_ok = sum(1 for r in rows if r.get("pass"))
        ax.set_title(f"b  Bootstrap refit stability  ({n_ok}/{len(rows)} PASS)",
                     loc="left", fontsize=8, fontweight="bold")
        ax.invert_yaxis()

    fig.suptitle(
        "Same region across unit halves (κ) and across bootstrap refits (Jaccard)",
        fontsize=9, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_4")


# ---------------------------------------------------------------------------
# F6.5 plant
# ---------------------------------------------------------------------------

def fig65_plant() -> Path | None:
    pl = _load(DATA_DIR / "plant" / "summary.json")
    if not pl:
        return None
    fig = plt.figure(figsize=(W_FULL, 75 * MM), facecolor="white")
    ax = fig.add_subplot(1, 2, 1)
    methods = [r["method"] for r in pl]
    hits = [r.get("hit_rate") or 0 for r in pl]
    cols = [PASS_C if (r.get("hit_rate") or 0) >= 0.5 else FAIL_C for r in pl]
    ax.bar(methods, hits, color=cols)
    ax.axhline(0.5, color="black", ls="--", lw=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit rate")
    ax.set_title("a  Planted-window hit rate", loc="left", fontsize=8, fontweight="bold")
    for m, h in zip(methods, hits):
        ax.text(m, h + 0.03, f"{h:.2f}", ha="center", fontsize=8)

    ax = fig.add_subplot(1, 2, 2)
    fas = [r.get("fa_per_min") if r.get("fa_per_min") is not None else 0 for r in pl]
    cols = [PASS_C if f <= 2.0 else FAIL_C for f in fas]
    ax.bar(methods, fas, color=cols)
    ax.axhline(2.0, color="black", ls="--", lw=0.9)
    ax.set_ylabel("false alarms / min")
    ax.set_title("b  False-alarm rate", loc="left", fontsize=8, fontweight="bold")
    ymax = max(2.2, max(fas) * 1.2 if fas else 2.2)
    ax.set_ylim(0, ymax)
    for m, f in zip(methods, fas):
        ax.text(m, f + 0.05 * ymax, f"{f:.2f}", ha="center", fontsize=8)

    alias = pl[0].get("alias", "SIM")
    both = all(r.get("pass") for r in pl)
    badge = "PASS" if both else "mixed"
    fig.suptitle(
        f"Plant-and-recover on matched sim ({alias}) — overall {badge}",
        fontsize=9, fontweight="bold", y=0.98,
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

    # table
    cols = ["Session", "occ%", "visits", "ΔP", "null_a", "null_b", "split", "boot"]
    cell = [cols]
    for r in rows:
        cell.append([
            r["alias"],
            "—" if r["occupancy"] is None else f"{100*r['occupancy']:.0f}",
            "—" if r["visits"] is None else str(r["visits"]),
            "—" if r["delta_p"] is None else f"{r['delta_p']:.2f}",
            _yn(r["null_a"]),
            _yn(r["null_b"]),
            _yn(r["split"]),
            _yn(r["boot"]),
        ])
    table = ax.table(cellText=cell, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.55)
    # colour PASS/FAIL cells
    for i in range(1, len(cell)):
        for j in (4, 5, 6, 7):
            val = cell[i][j]
            if val == "PASS":
                table[i, j].set_facecolor("#c6efce")
            elif val == "FAIL":
                table[i, j].set_facecolor("#ffc7ce")
    fig.suptitle(
        "Report 6 pilot numbers (real cohort, LDS targets; null_b/split/boot as run)",
        fontsize=10, fontweight="bold", y=0.96,
    )
    return _save(fig, "F6_numbers")


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
    return out


if __name__ == "__main__":
    print(json.dumps(build_all_available(), indent=2))
