#!/usr/bin/env python3
"""One-page summary + multi-page figure PDF; SHA256SUMS; ID grep; number audit."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.report6.figures import build_all_available, FIG_DIR  # noqa: E402
from agents.report6.paths import OUT_ROOT  # noqa: E402

MM = 1 / 25.4
W = 180 * MM

# Real-ID patterns (do not list concrete IDs in source beyond digit-shapes)
ID_PAT = re.compile(
    r"\b(?:\d{3,4}_\d{6,8}\w*|\b(?:1120|1130|1140|1150|1170|7394)_\w+)\b"
)


def _load(p: Path):
    return json.loads(p.read_text()) if p.is_file() else None


def summarize() -> dict:
    m0 = _load(OUT_ROOT / "m0_m2" / "all.json") or []
    na = _load(OUT_ROOT / "null_a" / "summary.json") or []
    nb = _load(OUT_ROOT / "null_b" / "summary.json") or []
    sp = _load(OUT_ROOT / "split" / "summary.json") or []
    bt = _load(OUT_ROOT / "boot" / "summary.json") or []
    pl = _load(OUT_ROOT / "plant" / "summary.json") or []

    rows = []
    real = [s for s in m0 if s.get("arm") == "real"]
    for s in real:
        alias = s["alias"]
        lds = (s.get("methods") or {}).get("lds") or {}
        te = (lds.get("targets_eval") or [])
        qual = len(te) > 0
        dist = bool(te[0]["distinguishable_pass"]) if te else False
        na_row = next((r for r in na if r.get("alias") == alias and r.get("method") == "lds"), None)
        nb_row = next((r for r in nb if r.get("alias") == alias and r.get("method") == "lds"), None)
        if nb_row is None:
            nb_row = next((r for r in nb if r.get("alias") == alias and r.get("method") == "pca"), None)
        # Prefer LDS row when present (matches F6.0 board); else PCA.
        sp_row = next((r for r in sp if r.get("alias") == alias and r.get("method") == "lds"), None)
        if sp_row is None:
            sp_row = next((r for r in sp if r.get("alias") == alias and r.get("method") == "pca"), None)
        if sp_row is None:
            sp_row = next((r for r in sp if r.get("alias") == alias), None)
        bt_row = next((r for r in bt if r.get("alias") == alias and r.get("method") == "lds"), None)
        if bt_row is None:
            bt_row = next((r for r in bt if r.get("alias") == alias and r.get("method") == "pca"), None)
        if bt_row is None:
            bt_row = next((r for r in bt if r.get("alias") == alias), None)
        rows.append({
            "alias": alias,
            "qualifies": qual,
            "distinguishable": dist,
            "null_a": None if na_row is None else bool(na_row.get("null_a_pass")),
            "null_b": None if nb_row is None else bool(nb_row.get("null_b_pass")),
            "split": None if sp_row is None else bool(sp_row.get("pass")),
            "boot": None if bt_row is None else bool(bt_row.get("pass")),
            "delta_p": te[0].get("delta_p") if te else None,
            "visits": te[0]["target"]["n_visits"] if te else 0,
        })

    n = len(rows) or 1
    frac_qual = sum(1 for r in rows if r["qualifies"]) / n
    frac_dist = sum(1 for r in rows if r["qualifies"] and r["distinguishable"]) / n
    null_a_done = [r for r in rows if r.get("null_a") is not None]
    if null_a_done:
        frac_strict = sum(
            1 for r in rows
            if r["qualifies"] and r["distinguishable"] and r.get("null_a") is True
        ) / n
        strict_label = "qual+distinguishable+null_a"
    else:
        frac_strict = frac_dist
        strict_label = "qual+distinguishable (null_a pending)"
    plant_hit = None
    if pl:
        pref = next((r for r in pl if r.get("method") == "pca"), pl[0])
        plant_hit = pref.get("hit_rate")

    three = {
        "frac_real_lds_qualifying": round(frac_qual, 3),
        "frac_qual_and_distinguishable_and_null_a": round(frac_strict, 3),
        "strict_label": strict_label,
        "plant_hit_rate": plant_hit,
        "n_null_a_done": len(null_a_done),
        "note": "rebuild PDF after null_a/plant/null_b complete",
    }
    return {"rows": rows, "three_numbers": three, "n_real": len(rows)}


def write_summary_page(pdf: PdfPages, summary: dict) -> None:
    fig = plt.figure(figsize=(W, 260 * MM), facecolor="white")
    fig.text(0.05, 0.96, "Report 6 pilot — latent-target clamp (one-page summary)",
             fontsize=12, fontweight="bold", va="top")
    fig.text(
        0.05, 0.92,
        "Within-session neural targets from causal LDS/PCA caches. "
        "No behaviour in target definition. Anonymised S01–S10 / A–F.",
        fontsize=8, va="top", wrap=True,
    )
    three = summary["three_numbers"]
    fig.text(0.05, 0.86, "Say aloud:", fontsize=9, fontweight="bold")
    fig.text(
        0.05, 0.83,
        f"1. Fraction real sessions with ≥1 qualifying LDS target: "
        f"{three['frac_real_lds_qualifying']*100:.0f}% "
        f"(n={summary['n_real']})\n"
        f"2. Fraction that also pass held-out distinguishable + null (a): "
        f"{three['frac_qual_and_distinguishable_and_null_a']*100:.0f}%\n"
        f"3. Plant-and-recover hit rate (sim): {three['plant_hit_rate']}",
        fontsize=8, va="top", family="monospace",
    )
    # table
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.65])
    ax.set_axis_off()
    cols = ["S", "qual", "dist", "null_a", "null_b", "split", "boot", "ΔP", "visits"]
    cell = [cols]
    for r in summary["rows"]:
        def yn(v):
            if v is None:
                return "—"
            return "PASS" if v else "FAIL"
        cell.append([
            r["alias"], yn(r["qualifies"]), yn(r["distinguishable"]),
            yn(r["null_a"]), yn(r["null_b"]), yn(r["split"]), yn(r["boot"]),
            "—" if r["delta_p"] is None else f"{r['delta_p']:.2f}",
            str(r["visits"]),
        ])
    table = ax.table(cellText=cell, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.35)
    fig.text(0.05, 0.05, "PASS/FAIL per pre-registered PILOT_PLAN.md criteria. Dash = not run (runtime cut).",
             fontsize=7)
    pdf.savefig(fig, facecolor="white")
    plt.close(fig)


def append_figure_pages(pdf: PdfPages) -> list[str]:
    used = []
    for name in [
        "F6_0_board", "F6_numbers", "F6_1", "F6_2", "F6_3", "F6_4", "F6_5",
    ]:
        png = FIG_DIR / f"{name}.png"
        if not png.is_file():
            continue
        img = plt.imread(png)
        fig = plt.figure(figsize=(W, W * img.shape[0] / img.shape[1]), facecolor="white")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        ax.set_axis_off()
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)
        used.append(name)
    return used


CAPTIONS = {
    "F6_0_board": (
        "Results board. Per-session PASS/FAIL for each pre-registered criterion "
        "(LDS primary). Headline fractions: qualifying targets, distinguishable, "
        "and those that also fail null (a); plant hit rate on sim."
    ),
    "F6_numbers": (
        "Numeric table for the real cohort: occupancy, visits, held-out ΔP, and "
        "PASS/FAIL for nulls, split-population κ, and bootstrap Jaccard."
    ),
    "F6_1": (
        "Idea and validation table. A latent-space target is defined from neural "
        "activity alone (M0–M2) and must pass recurrence, held-out separability, "
        "split-population agreement, refit stability, structured nulls, and "
        "plant-and-recover."
    ),
    "F6_2": (
        "Top qualifying target per session for real HF cohort and matched sim "
        "(k=0). Bars: training occupancy (%); orange: visit count. Dashed = 5% "
        "occupancy; dotted = 10 visits."
    ),
    "F6_3": (
        "Held-out ΔP vs neighbours, calibration, null (a) circular time shifts "
        "(20; PASS if ≤4), and null (b) correlation-breaking surrogates (5; "
        "PASS if ≤1) with manifold refit where run. Green = PASS, red = FAIL."
    ),
    "F6_4": (
        "Split-population Cohen κ (PASS if ≥0.2 and above chance) and bootstrap "
        "refit Jaccard after Procrustes (PASS if median ≥0.3). PCA on all real "
        "sessions; LDS on the three smallest-n_units sessions."
    ),
    "F6_5": (
        "Plant-and-recover on one matched-sim session: hit rate on planted test "
        "windows and false alarms per minute after refit (PCA and LDS)."
    ),
}


def number_audit(text: str) -> list[str]:
    """Very light audit: flag bare integers that look like session fragments."""
    fails = []
    for m in ID_PAT.finditer(text):
        fails.append(m.group(0))
    return fails


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_all_available()
    summary = summarize()
    (OUT_ROOT / "summary_counts.json").write_text(json.dumps(summary, indent=2) + "\n")

    pdf_path = OUT_ROOT / "report6_pilot_anon.pdf"
    with PdfPages(pdf_path) as pdf:
        write_summary_page(pdf, summary)
        used = append_figure_pages(pdf)
        # captions page
        fig = plt.figure(figsize=(W, 200 * MM), facecolor="white")
        fig.text(0.05, 0.95, "Figure captions", fontsize=11, fontweight="bold")
        y = 0.88
        for name in used:
            cap = CAPTIONS.get(name, "")
            fig.text(0.05, y, f"{name.replace('_', '.')}. {cap}", fontsize=7.5,
                     va="top", wrap=True)
            y -= 0.14
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

    # SHA256
    lines = []
    for p in sorted(OUT_ROOT.rglob("*")):
        if p.is_file() and p.suffix in {".png", ".pdf", ".json"} and "alias_map" not in p.name:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(OUT_ROOT)}")
    (OUT_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    # pdftotext ID grep
    txt_path = OUT_ROOT / "report6_pilot_anon.txt"
    try:
        subprocess.run(
            ["pdftotext", str(pdf_path), str(txt_path)],
            check=True, capture_output=True,
        )
        text = txt_path.read_text()
    except Exception as e:
        text = ""
        (OUT_ROOT / "pdftotext_error.txt").write_text(str(e))
    hits = number_audit(text)
    # also scan captions
    hits += number_audit(" ".join(CAPTIONS.values()))
    audit = {
        "pdf": str(pdf_path),
        "id_grep_hits": hits,
        "id_grep_n": len(hits),
        "pass": len(hits) == 0,
        "figures": used,
        "three_numbers": summary["three_numbers"],
    }
    (OUT_ROOT / "anon_leak_check.json").write_text(json.dumps(audit, indent=2) + "\n")

    # Ship to canonical reports/report6/ (hippo + hippo-r6), same pattern as Report 1.
    ship_dirs = [
        Path("/home/gmwager/projects/hippo/reports/report6"),
        ROOT / "reports" / "report6",
    ]
    for dest in ship_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "figures").mkdir(parents=True, exist_ok=True)
        for src in [pdf_path, OUT_ROOT / "summary_counts.json",
                    OUT_ROOT / "anon_leak_check.json", OUT_ROOT / "SHA256SUMS"]:
            if src.is_file():
                (dest / src.name).write_bytes(src.read_bytes())
        for fig in sorted((OUT_ROOT / "figures").glob("F6_*")):
            (dest / "figures" / fig.name).write_bytes(fig.read_bytes())
        audit.setdefault("shipped_to", []).append(str(dest))

    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
