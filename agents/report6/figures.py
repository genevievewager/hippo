"""F6.1–F6.5 builders (Report 1 style: mm widths, white face, takeaway titles)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from agents.report6.paths import OUT_ROOT

MM = 1 / 25.4
W_FULL = 180 * MM
FIG_DIR = OUT_ROOT / "figures"
DATA_DIR = OUT_ROOT


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


def fig61_schematic_and_table() -> Path:
    """Idea schematic + validation criteria table (data-free)."""
    fig = plt.figure(figsize=(W_FULL, 120 * MM), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.2], wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    boxes = [
        (0.5, 7.2, "M0 Manifold\n(LDS / PCA cache)"),
        (0.5, 4.6, "M1 Target rule\n(train density)"),
        (0.5, 2.0, "M2 Causal detector\n(held-out only)"),
        (5.2, 7.2, "Nulls\n(shift / surrogate)"),
        (5.2, 4.6, "Split-pop κ\n+ refit Jaccard"),
        (5.2, 2.0, "Plant & recover\n(sim)"),
    ]
    for x, y, t in boxes:
        ax.add_patch(FancyBboxPatch(
            (x, y), 3.8, 2.0, boxstyle="round,pad=0.05,rounding_size=0.2",
            facecolor="#f3f3f3", edgecolor="#333", lw=1.0,
        ))
        ax.text(x + 1.9, y + 1.0, t, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(2.4, 6.6), xytext=(2.4, 7.2),
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("", xy=(2.4, 4.0), xytext=(2.4, 4.6),
                arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_title("a  Pipeline", loc="left", fontsize=10, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_axis_off()
    rows = [
        ("Recurs", "≥5% occ., ≥10 visits"),
        ("Distinguishable", "ΔP≥0.10, Brier≤0.25"),
        ("Population", "κ≥0.2 > chance"),
        ("Refit-stable", "median Jaccard≥0.3"),
        ("Null (a)", "≤4/20 shift passes"),
        ("Null (b)", "≤1/5 surrogate passes"),
        ("Plant/recover", "hit≥0.5, FA≤2/min"),
    ]
    ax2.set_title("b  Validation (pre-registered)", loc="left", fontsize=10, fontweight="bold")
    y = 0.92
    ax2.text(0.02, y, "Criterion", fontsize=8, fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.42, y, "Pass rule", fontsize=8, fontweight="bold", transform=ax2.transAxes)
    y -= 0.08
    for c, r in rows:
        ax2.text(0.02, y, c, fontsize=7.5, transform=ax2.transAxes, va="top")
        ax2.text(0.42, y, r, fontsize=7.5, transform=ax2.transAxes, va="top")
        y -= 0.11
    fig.suptitle(
        "A neural-state target is real only if it recurs, separates, and fails nulls",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_1")


def fig62_targets() -> Path | None:
    all_path = DATA_DIR / "m0_m2" / "all.json"
    data = _load(all_path)
    if not data:
        return None
    fig = plt.figure(figsize=(W_FULL, 110 * MM), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
    # gather
    rows = []
    for sess in data:
        for method, blob in (sess.get("methods") or {}).items():
            for t in blob.get("targets_eval") or []:
                rows.append({
                    "alias": sess["alias"],
                    "arm": sess["arm"],
                    "method": method,
                    "occ": t["target"]["occupancy"],
                    "visits": t["target"]["n_visits"],
                    "dwell": t["target"]["median_dwell_s"],
                })
            if blob.get("status") == "no_qualifying_target":
                rows.append({
                    "alias": sess["alias"], "arm": sess["arm"], "method": method,
                    "occ": 0.0, "visits": 0, "dwell": np.nan,
                })
    def _panel(ax, arm, method, title):
        sub = [r for r in rows if r["arm"] == arm and r["method"] == method]
        # one point per session: max visits among targets else 0
        by = {}
        for r in sub:
            a = r["alias"]
            if a not in by or r["visits"] > by[a]["visits"]:
                by[a] = r
        aliases = sorted(by)
        occ = [by[a]["occ"] * 100 for a in aliases]
        vis = [by[a]["visits"] for a in aliases]
        x = np.arange(len(aliases))
        ax.bar(x - 0.15, occ, width=0.3, label="occupancy %", color="#4C78A8")
        ax2 = ax.twinx()
        ax2.bar(x + 0.15, vis, width=0.3, label="visits", color="#F58518")
        ax.set_xticks(x)
        ax.set_xticklabels(aliases, fontsize=6, rotation=45, ha="right")
        ax.set_ylabel("occupancy %", fontsize=7)
        ax2.set_ylabel("visits", fontsize=7)
        ax.set_title(title, loc="left", fontsize=8, fontweight="bold")
        ax.axhline(5, color="gray", ls="--", lw=0.6)

    ax = fig.add_subplot(gs[0, 0]); _panel(ax, "real", "lds", "a  Real LDS")
    ax = fig.add_subplot(gs[0, 1]); _panel(ax, "real", "pca", "b  Real PCA")
    ax = fig.add_subplot(gs[1, 0]); _panel(ax, "sim", "lds", "c  Sim LDS (k=0)")
    ax = fig.add_subplot(gs[1, 1]); _panel(ax, "sim", "pca", "d  Sim PCA (k=0)")
    n_real = sum(1 for s in data if s["arm"] == "real")
    n_sim = sum(1 for s in data if s["arm"] == "sim")
    fig.suptitle(
        f"Qualified targets: occupancy and visits (n_real={n_real}, n_sim={n_sim})",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_2")


def fig63_detection_nulls() -> Path | None:
    m0 = _load(DATA_DIR / "m0_m2" / "all.json")
    na = _load(DATA_DIR / "null_a" / "summary.json")
    nb = _load(DATA_DIR / "null_b" / "summary.json")
    if not m0:
        return None
    fig = plt.figure(figsize=(W_FULL, 115 * MM), facecolor="white")
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.35)

    # a: delta P real LDS
    ax = fig.add_subplot(gs[0, 0])
    xs, ys, cols = [], [], []
    for sess in m0:
        if sess["arm"] != "real":
            continue
        blob = sess["methods"].get("lds", {})
        te = blob.get("targets_eval") or []
        if not te:
            continue
        xs.append(sess["alias"])
        ys.append(te[0].get("delta_p") or 0)
        cols.append("#2ca02c" if te[0].get("distinguishable_pass") else "#d62728")
    ax.bar(xs, ys, color=cols)
    ax.axhline(0.10, color="gray", ls="--", lw=0.7)
    ax.set_ylabel("ΔP vs best neighbour")
    ax.set_title("a  Held-out separability (LDS)", loc="left", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelrotation=45, labelsize=6)

    # b: calibration pooled
    ax = fig.add_subplot(gs[0, 1])
    for sess in m0:
        if sess["arm"] != "real":
            continue
        te = (sess["methods"].get("lds") or {}).get("targets_eval") or []
        if not te:
            continue
        rel = te[0].get("reliability") or []
        px = [r["p_mean"] for r in rel if r.get("p_mean") is not None]
        py = [r["y_mean"] for r in rel if r.get("y_mean") is not None]
        if px:
            ax.plot(px, py, "o-", ms=3, alpha=0.5)
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax.set_xlabel("predicted P"); ax.set_ylabel("observed fraction")
    ax.set_title("b  Calibration (real LDS)", loc="left", fontsize=8, fontweight="bold")

    # c: null a
    ax = fig.add_subplot(gs[1, 0])
    if na:
        aliases = [r["alias"] for r in na if r["method"] == "lds" and not str(r["alias"]).startswith("SIM")]
        npass = [r["n_pass"] for r in na if r["method"] == "lds" and not str(r["alias"]).startswith("SIM")]
        ax.bar(aliases, npass, color="#4C78A8")
        ax.axhline(4, color="red", ls="--", lw=0.8)
        ax.set_ylabel("# shift passes / 20")
        ax.tick_params(axis="x", labelrotation=45, labelsize=6)
    ax.set_title("c  Null (a) time shift (LDS)", loc="left", fontsize=8, fontweight="bold")

    # d: null b
    ax = fig.add_subplot(gs[1, 1])
    if nb:
        lab = [f"{r['alias']}\n{r['method']}" for r in nb if not str(r["alias"]).startswith("SIM")]
        npass = [r.get("n_pass", 0) for r in nb if not str(r["alias"]).startswith("SIM")]
        ax.bar(range(len(lab)), npass, color="#F58518")
        ax.set_xticks(range(len(lab)))
        ax.set_xticklabels(lab, fontsize=5)
        ax.axhline(1, color="red", ls="--", lw=0.8)
        ax.set_ylabel("# surrogate passes / 5")
    ax.set_title("d  Null (b) correlation-break", loc="left", fontsize=8, fontweight="bold")

    fig.suptitle(
        "Held-out detection must beat neighbours and fail structured nulls",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_3")


def fig64_split_boot() -> Path | None:
    sp = _load(DATA_DIR / "split" / "summary.json")
    bt = _load(DATA_DIR / "boot" / "summary.json")
    if not sp and not bt:
        return None
    fig = plt.figure(figsize=(W_FULL, 90 * MM), facecolor="white")
    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax = fig.add_subplot(gs[0, 0])
    if sp:
        xs = [r["alias"] + ":" + r.get("method", "") for r in sp if r.get("kappa") is not None]
        ys = [r["kappa"] for r in sp if r.get("kappa") is not None]
        cols = ["#2ca02c" if r.get("pass") else "#d62728" for r in sp if r.get("kappa") is not None]
        ax.bar(range(len(xs)), ys, color=cols)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=5, rotation=45, ha="right")
        ax.axhline(0.2, color="gray", ls="--", lw=0.7)
    ax.set_ylabel("Cohen κ")
    ax.set_title("a  Split-population agreement", loc="left", fontsize=8, fontweight="bold")

    ax = fig.add_subplot(gs[0, 1])
    if bt:
        xs = [r["alias"] + ":" + r.get("method", "") for r in bt if r.get("median_jaccard") is not None]
        ys = [r["median_jaccard"] for r in bt if r.get("median_jaccard") is not None]
        cols = ["#2ca02c" if r.get("pass") else "#d62728" for r in bt if r.get("median_jaccard") is not None]
        ax.bar(range(len(xs)), ys, color=cols)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, fontsize=5, rotation=45, ha="right")
        ax.axhline(0.3, color="gray", ls="--", lw=0.7)
    ax.set_ylabel("median Jaccard")
    ax.set_title("b  Bootstrap refit stability", loc="left", fontsize=8, fontweight="bold")
    fig.suptitle(
        "Same region across unit halves and refits (where computed)",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_4")


def fig65_plant() -> Path | None:
    pl = _load(DATA_DIR / "plant" / "summary.json")
    if not pl:
        return None
    fig = plt.figure(figsize=(W_FULL, 70 * MM), facecolor="white")
    ax = fig.add_subplot(1, 2, 1)
    methods = [r["method"] for r in pl]
    hits = [r.get("hit_rate") or 0 for r in pl]
    ax.bar(methods, hits, color="#4C78A8")
    ax.axhline(0.5, color="red", ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("hit rate")
    ax.set_title("a  Planted-window hit rate", loc="left", fontsize=8, fontweight="bold")
    ax = fig.add_subplot(1, 2, 2)
    fas = [r.get("fa_per_min") or 0 for r in pl]
    ax.bar(methods, fas, color="#F58518")
    ax.axhline(2.0, color="red", ls="--")
    ax.set_ylabel("false alarms / min")
    ax.set_title("b  False-alarm rate", loc="left", fontsize=8, fontweight="bold")
    alias = pl[0].get("alias", "SIM")
    fig.suptitle(
        f"Plant-and-recover on matched sim ({alias}, n_events=40)",
        fontsize=10, fontweight="bold", y=0.98,
    )
    return _save(fig, "F6_5")


def build_all_available() -> dict[str, str]:
    out = {}
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
    return out


if __name__ == "__main__":
    print(json.dumps(build_all_available(), indent=2))
