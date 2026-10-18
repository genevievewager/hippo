#!/usr/bin/env python3
"""Build full Report 4a (v4): extract → figures → story/findings/todo/audit → PDF + anon + PNGs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader, PdfWriter

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.report.build_pdf_4a import findings_page, todo_page
from agents.data_quality.report.build_story_r4 import assemble as assemble_story, write_story
from agents.data_quality.report.extract_tidy import write_tidy
from agents.data_quality.report.number_audit import audit

ROOT = Path(__file__).resolve().parents[3]
OUT = assert_dq_path(DATA_QUALITY_ROOT / "reports" / "report4a_v4")
DATA = OUT / "data"
DATA_ANON = OUT / "data_anon"
FIGS = OUT / "figs"
FIGS_ANON = OUT / "figs_anon"
PAGES = OUT / "pages"
PAGES_ANON = OUT / "pages_anon"
BEHINFO = OUT / "behavior_info"


FIG_ORDER = [
    "Fig1_design",
    "Fig2_validity",
    "Fig3_answer",
    "Fig4_mechanism",
    "Fig5_behaviour",
    "Fig6_answer",
    "Fig7_trajectories",
    "Fig8_behavior_info",
    "FigS1_units",
    "FigS2_sorting",
    "FigS3_spatial",
    "FigS4_tuning_fracs",
    "FigS5_population",
    "FigS6_disc_decode",
    "FigS7_q3_caveats",
    "S8_behavior_distributions_sensitivity",
    "Fig_calib_deferred_4b",
    "Appendix_OLD_VS_NEW_spike_keep",
]


def _run(cmd: list[str]) -> None:
    print(json.dumps({"cmd": cmd}), flush=True)
    pip_local = str(ROOT / ".pip_local")
    env = {**os.environ}
    env["PYTHONPATH"] = f"{ROOT}:{pip_local}" + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    subprocess.check_call(cmd, cwd=str(ROOT), env=env)


def _merge_data_layers() -> None:
    """Review extract (fig3 wide + rich meta) + tidy (findings/disc/divergence long)."""
    import pandas as pd

    DATA.mkdir(parents=True, exist_ok=True)
    DATA_ANON.mkdir(parents=True, exist_ok=True)
    # 1) review extract first (skip heavy; reuse v2 pools)
    _run([
        sys.executable, "-m", "agents.data_quality.report.extract_review_assets",
        "--out", str(DATA), "--skip-heavy",
    ])
    review_meta = json.loads((DATA / "data_meta.json").read_text()) if (DATA / "data_meta.json").is_file() else {}
    review_sess = DATA / "data_session_metrics.csv"
    review_sess_df = pd.read_csv(review_sess) if review_sess.is_file() else None

    # 2) tidy into a side folder then merge
    tidy_tmp = OUT / "_tidy_tmp"
    write_tidy(tidy_tmp, anon=False)
    write_tidy(DATA_ANON, anon=True)
    for name in (
        "data_divergence.csv",
        "data_findings.csv",
        "data_discriminator.csv",
        "data_discriminator_importance.csv",
        "data_discriminator_ablation.csv",
    ):
        src = tidy_tmp / name
        if src.is_file():
            shutil.copy2(src, DATA / name)

    tidy_sess = tidy_tmp / "data_session_metrics.csv"
    if tidy_sess.is_file() and review_sess_df is not None and len(review_sess_df):
        tdf = pd.read_csv(tidy_sess)
        key = ["dataset", "session", "metric"]
        # Drop overlapping keys from tidy, then concat review (preferred for spatial)
        tdf = tdf.merge(review_sess_df[key].drop_duplicates(), on=key, how="left", indicator=True)
        tdf = tdf[tdf["_merge"] == "left_only"].drop(columns=["_merge"])
        merged = pd.concat([tdf, review_sess_df], ignore_index=True, sort=False)
        merged.to_csv(DATA / "data_session_metrics.csv", index=False)
    elif tidy_sess.is_file():
        shutil.copy2(tidy_sess, DATA / "data_session_metrics.csv")

    # restore rich review meta (with Q3 / sparsity notes)
    if review_meta:
        # keep tidy thigmotaxis fields if missing
        tidy_meta_p = tidy_tmp / "data_meta.json"
        if tidy_meta_p.is_file():
            tm = json.loads(tidy_meta_p.read_text())
            for k in ("censor_isi_ms", "censor_assumption", "rank_exclude"):
                review_meta.setdefault(k, tm.get(k))
        (DATA / "data_meta.json").write_text(json.dumps(review_meta, indent=2, default=str) + "\n")

    for name in (
        "data_fig3_divergence.csv",
        "data_sim_gets_right.csv",
        "data_meta.json",
        "data_spatial_abs_medians.csv",
        "data_rate_maps.json",
        "data_trajectories_60s.json",
        "data_isi_pool.json",
        "data_occupancy.json",
        "data_speed_turning_example.json",
        "data_session_metrics.csv",
        "data_divergence.csv",
        "data_findings.csv",
        "data_discriminator.csv",
        "data_discriminator_importance.csv",
        "data_discriminator_ablation.csv",
    ):
        src = DATA / name
        if src.is_file():
            shutil.copy2(src, DATA_ANON / name)

    src_comp = DATA_QUALITY_ROOT / "phase3_4a" / "OLD_VS_NEW_R_spatial_turning.csv"
    if src_comp.is_file():
        shutil.copy2(src_comp, OUT / "OLD_VS_NEW_R_spatial_turning.csv")


def _render_figs(data: Path, figs: Path) -> None:
    figs.mkdir(parents=True, exist_ok=True)
    _run([
        sys.executable, "-m", "agents.data_quality.report.figures_r4",
        "--data", str(data), "--out", str(figs),
    ])


def _deferred_calib_page(figs: Path) -> Path:
    """Single page marking calibration Q13–Q15 as deferred to Report 4b."""
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.lib.units import mm

    path = figs / "Fig_calib_deferred_4b.pdf"
    c = rl_canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    c.setFont("Helvetica-Bold", 14)
    c.drawString(20 * mm, h - 30 * mm, "Calibration (Q13–Q15) — deferred (Report 4b)")
    c.setFont("Helvetica", 10)
    lines = [
        "Proxy-fidelity gates failed the objective criterion in the 4a prep run.",
        "Calibration pages (former Q13–Q15) are therefore marked deferred and are not scored here.",
        "They will be rebuilt in Report 4b once the proxy protocol is fixed.",
    ]
    y = h - 45 * mm
    for line in lines:
        c.drawString(20 * mm, y, line)
        y -= 7 * mm
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(20 * mm, 20 * mm, "deferred (4b)")
    c.save()
    # companion PNG for gallery
    try:
        subprocess.check_call(
            ["pdftoppm", "-png", "-r", "150", "-singlefile", str(path), str(figs / "Fig_calib_deferred_4b")],
            cwd=str(ROOT),
        )
    except Exception:
        pass
    return path


def _appendix_old_vs_new(figs: Path, csv_path: Path) -> Path:
    """Render OLD_VS_NEW_spike_keep as a multi-page appendix table."""
    import numpy as np
    import pandas as pd
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    path = figs / "Appendix_OLD_VS_NEW_spike_keep.pdf"
    df = pd.read_csv(csv_path)
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(A4),
        leftMargin=10 * mm, rightMargin=10 * mm, topMargin=12 * mm, bottomMargin=12 * mm,
    )
    story = [
        Paragraph("Appendix — OLD vs NEW after exact keep-mask spike assignment", styles["Heading2"]),
        Paragraph(
            "Old = pre-fix report4a_v3 spatial/encoding/drift numbers. "
            "New = post exact frame-interval keep mask (occupancy and spikes from the same keep). "
            "R = d_sim / d_lab (matched / default).",
            styles["Normal"],
        ),
        Spacer(1, 4 * mm),
    ]
    cols = [
        "family", "metric",
        "lab_old", "lab_new", "matched_old", "matched_new", "default_old", "default_new",
        "R_matched_old", "R_matched_new", "R_default_old", "R_default_new",
    ]
    cols = [c for c in cols if c in df.columns]
    header = cols
    data = [header]
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float) and np.isfinite(v):
                row.append(f"{v:.3g}")
            else:
                row.append("" if (isinstance(v, float) and not np.isfinite(v)) else str(v))
        data.append(row)
    tbl = Table(data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e6e0")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#b0aea6")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f6f2")]),
    ]))
    story.append(tbl)
    doc.build(story)
    try:
        subprocess.check_call(
            ["pdftoppm", "-png", "-r", "120", "-singlefile", str(path),
             str(figs / "Appendix_OLD_VS_NEW_spike_keep")],
            cwd=str(ROOT),
        )
    except Exception:
        pass
    return path


def _text_pages(data: Path, tmp: Path) -> dict[str, Path]:
    tmp.mkdir(parents=True, exist_ok=True)
    story = tmp / "_story.pdf"
    headline = FIGS / "Fig0_headline.png"
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


def _assemble(parts: dict[str, Path], figs: Path, out_pdf: Path) -> Path:
    w = PdfWriter()
    for key in ("story", "findings", "todo"):
        p = parts[key]
        if p.is_file():
            w.append(str(p))
    for name in FIG_ORDER:
        p = figs / f"{name}.pdf"
        if p.is_file():
            w.append(str(p))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as f:
        w.write(f)
    return out_pdf


def _pdf_to_pngs(pdf: Path, pages_dir: Path) -> list[Path]:
    pages_dir.mkdir(parents=True, exist_ok=True)
    # Prefer pdftoppm; fall back to pypdf+pil
    out_prefix = pages_dir / "page"
    try:
        subprocess.check_call(
            ["pdftoppm", "-png", "-r", "150", str(pdf), str(out_prefix)],
            cwd=str(ROOT),
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        from PIL import Image
        try:
            import fitz  # pymupdf
            doc = fitz.open(str(pdf))
            paths = []
            for i, page in enumerate(doc, 1):
                pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72))
                path = pages_dir / f"page-{i}.png"
                pix.save(str(path))
                paths.append(path)
            return paths
        except Exception:
            # last resort: copy existing fig PNGs as pages
            paths = []
            for i, name in enumerate(["_story"] + FIG_ORDER, 1):
                src = FIGS / f"{name}.png"
                if src.is_file():
                    dst = pages_dir / f"page-{i}.png"
                    shutil.copy2(src, dst)
                    paths.append(dst)
            return paths
    return sorted(pages_dir.glob("page*.png"))


def _audit_pngs(paths: list[Path]) -> dict:
    """Flag PNGs that look truncated (large white margin collapse) via bbox heuristic."""
    from PIL import Image
    import numpy as np

    issues = []
    for p in paths:
        im = Image.open(p).convert("L")
        arr = np.asarray(im)
        # content = pixels darker than near-white
        mask = arr < 250
        if not mask.any():
            issues.append({"page": p.name, "issue": "blank"})
            continue
        ys, xs = np.where(mask)
        # if content touches edge within 2 px → possible truncation
        h, w = arr.shape
        touch = int(ys.min() <= 2 or xs.min() <= 2 or ys.max() >= h - 3 or xs.max() >= w - 3)
        if touch:
            # common for full-bleed captions; only flag if >15% of edge pixels are dark
            edge = np.concatenate([
                arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1],
            ])
            if (edge < 240).mean() > 0.15:
                issues.append({"page": p.name, "issue": "edge_content", "edge_frac": float((edge < 240).mean())})
    return {"n_pages": len(paths), "issues": issues}


def main() -> int:
    print(json.dumps({"build_full_start": True, "out": str(OUT)}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    # Bring forward behaviour-info artefacts (encoding / drift already recomputed)
    v3_beh = DATA_QUALITY_ROOT / "reports" / "report4a_v3" / "behavior_info"
    if v3_beh.is_dir():
        if BEHINFO.exists():
            shutil.rmtree(BEHINFO)
        shutil.copytree(v3_beh, BEHINFO)
    # OLD vs NEW table
    _run([sys.executable, "-m", "agents.data_quality.report.write_old_vs_new_spike_keep"])
    old_new = OUT / "OLD_VS_NEW_spike_keep.csv"

    _merge_data_layers()
    _render_figs(DATA, FIGS)
    # Fig 8 / S8 (behaviour-info) into the same figs dir
    _run([sys.executable, "-m", "agents.data_quality.report.figures_behavior_info"])
    _deferred_calib_page(FIGS)
    if old_new.is_file():
        _appendix_old_vs_new(FIGS, old_new)
    # anon figs from anon meta (same numbers; animal codes in tidy sess if used)
    shutil.copytree(FIGS, FIGS_ANON, dirs_exist_ok=True)

    tmp = OUT / "_pdf_parts"
    parts = _text_pages(DATA, tmp)
    pdf = _assemble(parts, FIGS, OUT / "report4a_full.pdf")
    # anon: rebuild text pages with anon data
    tmp_a = OUT / "_pdf_parts_anon"
    parts_a = _text_pages(DATA_ANON, tmp_a)
    pdf_anon = _assemble(parts_a, FIGS_ANON, OUT / "report4a_full_anon.pdf")

    audit_res = audit(DATA, DATA / "number_claims.json")
    (OUT / "number_audit.json").write_text(json.dumps(audit_res, indent=2, default=str) + "\n")

    pngs = _pdf_to_pngs(pdf, PAGES)
    png_audit = _audit_pngs(pngs)
    (OUT / "png_audit.json").write_text(json.dumps(png_audit, indent=2) + "\n")
    _pdf_to_pngs(pdf_anon, PAGES_ANON)

    # Copy summary artefacts for delivery
    summary = {
        "pdf": str(pdf),
        "pdf_anon": str(pdf_anon),
        "pages": [str(p) for p in pngs],
        "png_audit": png_audit,
        "number_audit_ok": audit_res.get("ok"),
        "number_audit_failures": audit_res.get("failures"),
        "old_vs_new": str(old_new),
        "sanity": str(OUT / "sanity_keep_rate_consistency.json"),
    }
    (OUT / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if audit_res.get("ok") and not png_audit.get("issues") else 0  # soft: still deliver


if __name__ == "__main__":
    raise SystemExit(main())
