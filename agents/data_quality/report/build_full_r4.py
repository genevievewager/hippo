#!/usr/bin/env python3
"""Build full Report 4a (v6): formatting-only — A4 frames, overlap audit, Fig 1–7."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfWriter

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.report.anon_util import anonymize_data_dir, leak_check
from agents.data_quality.report.build_pdf_4a import findings_page, todo_page
from agents.data_quality.report.build_story_r4 import write_story
from agents.data_quality.report.number_audit import audit
from agents.data_quality.report.overlap_audit import audit_pdf
from agents.data_quality.report.page_geometry import place_figure_on_a4

ROOT = Path(__file__).resolve().parents[3]
OUT = assert_dq_path(DATA_QUALITY_ROOT / "reports" / "report4a_v6")
V5 = DATA_QUALITY_ROOT / "reports" / "report4a_v5"
DATA = OUT / "data"
DATA_ANON = OUT / "data_anon"
FIGS = OUT / "figs"
FIGS_ANON = OUT / "figs_anon"
PAGES = OUT / "pages"
PAGES_ANON = OUT / "pages_anon"
BEHINFO = OUT / "behavior_info"
DELIVER = Path("/home/gmwager/projects/hippo/tmp_r4_review/v6")

# Continuous Fig 1–7 (former Fig 8 → Fig 7). Supplements after.
FIG_ORDER = [
    "Fig1_design",
    "Fig2_validity",
    "Fig3_answer",
    "Fig4_mechanism",
    "Fig5_behaviour",
    "Fig6_answer",
    "Fig7_behavior_info",
    "FigS1_units",
    "FigS2_sorting",
    "FigS3_spatial",
    "FigS4_tuning_fracs",
    "FigS5_population",
    "FigS6_disc_decode",
    "FigS7_q3_caveats",
    "S8_behavior_distributions_sensitivity",
    "Appendix_OLD_VS_NEW_spike_keep",
]


def _run(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    pip_local = str(ROOT / ".pip_local")
    env = {**os.environ}
    env["PYTHONPATH"] = f"{ROOT}:{pip_local}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def _reuse_v5_data() -> None:
    """Formatting-only: copy approved v5 tidy tables (no re-extract / re-score)."""
    if not (V5 / "data").is_dir():
        raise SystemExit(f"v5 data missing at {V5 / 'data'}")
    if DATA.exists():
        shutil.rmtree(DATA)
    shutil.copytree(V5 / "data", DATA)
    if (V5 / "behavior_info").is_dir():
        if BEHINFO.exists():
            shutil.rmtree(BEHINFO)
        shutil.copytree(V5 / "behavior_info", BEHINFO)
    for name in ("OLD_VS_NEW_spike_keep.csv",):
        src = V5 / name
        if src.is_file():
            shutil.copy2(src, OUT / name)
    if DATA_ANON.exists():
        shutil.rmtree(DATA_ANON)
    anonymize_data_dir(DATA, DATA_ANON)


def _render_figs(data: Path, figs: Path) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, "-m", "agents.data_quality.report.figures_r4",
        "--data", str(data), "--out", str(figs),
    ])


def _appendix_old_vs_new(figs: Path, csv_path: Path) -> None:
    import numpy as np
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    # Portrait A4 (no landscape)
    path = figs / "Appendix_OLD_VS_NEW_spike_keep.pdf"
    df = pd.read_csv(csv_path)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path), pagesize=A4,
        leftMargin=14 * mm, rightMargin=14 * mm, topMargin=14 * mm, bottomMargin=14 * mm,
    )
    story = [
        Paragraph("Appendix — OLD vs NEW after exact keep-mask", styles["Heading2"]),
        Paragraph(
            "Old = pre-fix; New = post keep-mask. R = d_sim / d_lab.",
            styles["Normal"],
        ),
        Spacer(1, 3 * mm),
    ]
    cols = [c for c in (
        "family", "metric", "lab_old", "lab_new", "matched_old", "matched_new",
        "R_matched_old", "R_matched_new",
    ) if c in df.columns]
    data = [cols]
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                row.append(f"{v:.3g}" if abs(v) >= 5e-3 else "0.00")
            else:
                row.append("" if (isinstance(v, float) and not np.isfinite(v)) else str(v)[:18])
        data.append(row)
    tbl = Table(data, repeatRows=1, colWidths=[18 * mm] * len(cols))
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0aea6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(tbl)
    doc.build(story)


def _text_pages(data: Path, tmp: Path, figs: Path) -> dict[str, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    story = tmp / "_story.pdf"
    headline = figs / "Fig0_headline.png"
    write_story(
        data, story,
        headline_png=headline if headline.is_file() else None,
        behinfo=BEHINFO if BEHINFO.is_dir() else None,
    )
    findings_page(data, tmp)
    todo_page(data, tmp)
    return {
        "story": story,
        "findings": tmp / "_findings.pdf",
        "todo": tmp / "_todo.pdf",
    }


def _assemble_a4(parts: dict[str, Path], figs: Path, out_pdf: Path) -> Path:
    """Every page A4 portrait; figures placed at fixed 180 mm width."""
    w = PdfWriter()
    for key in ("story", "findings", "todo"):
        p = parts[key]
        if p.is_file():
            w.append(str(p))
    for name in FIG_ORDER:
        p = figs / f"{name}.pdf"
        if not p.is_file():
            continue
        placed, _meta = place_figure_on_a4(p)
        w.add_page(placed)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as f:
        w.write(f)
    return out_pdf


def main() -> int:
    print(json.dumps({"build_full_start": True, "out": str(OUT), "version": "v6"}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    _reuse_v5_data()

    if FIGS.exists():
        shutil.rmtree(FIGS)
    _render_figs(DATA, FIGS)
    # Behaviour-info Fig7 into same figs dir
    _run([sys.executable, "-m", "agents.data_quality.report.figures_behavior_info"])
    old_new = OUT / "OLD_VS_NEW_spike_keep.csv"
    if old_new.is_file():
        _appendix_old_vs_new(FIGS, old_new)

    if FIGS_ANON.exists():
        shutil.rmtree(FIGS_ANON)
    _render_figs(DATA_ANON, FIGS_ANON)
    for name in (
        "Fig7_behavior_info", "Fig8_behavior_info",
        "S8_behavior_distributions_sensitivity", "Appendix_OLD_VS_NEW_spike_keep",
        "Fig3_information_family",
    ):
        for ext in (".pdf", ".png"):
            src = FIGS / f"{name}{ext}"
            if src.is_file():
                shutil.copy2(src, FIGS_ANON / f"{name}{ext}")

    tmp = OUT / "_pdf_parts"
    parts = _text_pages(DATA, tmp, FIGS)
    pdf = _assemble_a4(parts, FIGS, OUT / "report4a_full.pdf")

    tmp_a = OUT / "_pdf_parts_anon"
    parts_a = _text_pages(DATA_ANON, tmp_a, FIGS_ANON)
    pdf_anon = _assemble_a4(parts_a, FIGS_ANON, OUT / "report4a_full_anon.pdf")

    # Audits
    if PAGES.exists():
        shutil.rmtree(PAGES)
    overlap = audit_pdf(pdf, PAGES)
    (OUT / "overlap_audit.json").write_text(json.dumps(overlap, indent=2, default=str) + "\n")

    audit_res = audit(DATA, DATA / "number_claims.json", pdf)
    (OUT / "number_audit.json").write_text(json.dumps(audit_res, indent=2, default=str) + "\n")

    leak = leak_check([
        pdf_anon,
        DATA_ANON / "data_meta.json",
        *FIGS_ANON.glob("*.pdf"),
        *list(tmp_a.glob("*.pdf")),
    ])
    (OUT / "anon_leak_check.json").write_text(json.dumps(leak, indent=2) + "\n")

    if PAGES_ANON.exists():
        shutil.rmtree(PAGES_ANON)
    audit_pdf(pdf_anon, PAGES_ANON)  # render anon PNGs; overlap result secondary

    # Deliver only if overlap passes
    DELIVER.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "v6",
        "pdf": str(pdf),
        "pdf_anon": str(pdf_anon),
        "deliver": str(DELIVER),
        "number_audit_ok": audit_res.get("ok"),
        "number_audit_failures": audit_res.get("failures"),
        "anon_leak_ok": leak.get("ok"),
        "anon_leak_hits": leak.get("hits"),
        "overlap_ok": overlap.get("ok"),
        "overlap_n_failures": overlap.get("n_failures"),
        "overlap_failures": overlap.get("failures"),
        "n_pages": overlap.get("n_pages"),
        "delivered": False,
    }
    if overlap.get("ok") and audit_res.get("ok") and leak.get("ok"):
        for src in (
            pdf, pdf_anon,
            OUT / "number_audit.json",
            OUT / "anon_leak_check.json",
            OUT / "overlap_audit.json",
        ):
            if src.is_file():
                shutil.copy2(src, DELIVER / src.name)
        # PNG pages
        for p in sorted(PAGES.glob("page*.png")):
            shutil.copy2(p, DELIVER / p.name)
        summary["delivered"] = True
    else:
        summary["deliver_blocked"] = "overlap/number/leak audit failed — PDF not copied to deliver"
        # Still write audit JSONs for inspection
        for src in (OUT / "number_audit.json", OUT / "anon_leak_check.json", OUT / "overlap_audit.json"):
            if src.is_file():
                shutil.copy2(src, DELIVER / src.name)

    (OUT / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    (DELIVER / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    return 0 if summary.get("delivered") else 1


if __name__ == "__main__":
    raise SystemExit(main())
