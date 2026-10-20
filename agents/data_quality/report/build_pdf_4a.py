#!/usr/bin/env python3
"""Assemble Report 4a PDF from rendered Q*.pdf figures + findings pages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE.parents[1] / "quadrant_n5" / "figures" / "fonts"


def _fonts():
    candidates = [
        (FONT_DIR / "LiberationSans-Regular.ttf", FONT_DIR / "LiberationSans-Bold.ttf"),
        (
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ),
    ]
    for reg, bold in candidates:
        if reg.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("Sans", str(reg)))
            pdfmetrics.registerFont(TTFont("Sans-Bold", str(bold)))
            return "Sans", "Sans-Bold"
    raise RuntimeError("Liberation Sans not found")


FONT, BOLD = _fonts()
W, H = A4
M = 15 * mm
BODY = ParagraphStyle("b", fontName=FONT, fontSize=8.6, leading=11.6, spaceAfter=4)
H1 = ParagraphStyle("h1", fontName=BOLD, fontSize=14, leading=17, spaceAfter=8)
H2 = ParagraphStyle("h2", fontName=BOLD, fontSize=10, leading=13, spaceBefore=6, spaceAfter=3)
SM = ParagraphStyle("s", fontName=FONT, fontSize=7, leading=9, textColor=colors.HexColor("#52514e"))


def _page_from_story(path: Path, story) -> Path:
    c = canvas.Canvas(str(path), pagesize=A4)
    f = Frame(M, M, W - 2 * M, H - 2 * M, showBoundary=0)
    f.addFromList(story, c)
    c.save()
    return path


def _fig3_row(fig3: pd.DataFrame, base: str):
    hit = fig3[fig3.metric_base == base]
    return None if hit.empty else hit.iloc[0]


def findings_page(data: Path, out: Path) -> Path:
    """Prose summary from final Fig3 tidy (story is authority; no raw dump)."""
    meta = json.loads((data / "data_meta.json").read_text())
    fig3 = pd.read_csv(data / "data_fig3_divergence.csv")
    disc = pd.read_csv(data / "data_discriminator.csv") if (data / "data_discriminator.csv").is_file() else None

    def fmt_pair(base: str) -> str:
        r = _fig3_row(fig3, base)
        if r is None:
            return "—"
        return (
            f"{float(r.lab_median):.2g} lab | {float(r.sim_matched_median):.2g} matched "
            f"(R = {float(r.R_matched):.2f})"
        )

    top = fig3.dropna(subset=["R_matched"]).sort_values("R_matched", ascending=False).head(5)
    top_txt = "; ".join(
        f"{r.label} R = {float(r.R_matched):.2f}" for _, r in top.iterrows()
    )
    story = [
        Paragraph("Report 4a — Findings (from final tidy)", H1),
        Paragraph(
            f"Lab n = {meta['n_lab_sessions']} sessions ({meta['n_animals']} animals); "
            f"matched n = {meta['n_matched_sessions']}; default seeds = {meta['n_default_seeds']}. "
            f"thigmotaxis = {meta['thigmotaxis']} (not fitted). "
            "Numbers below are the same post keep-mask tables as the story page and Fig 3.",
            BODY,
        ),
        Paragraph("Largest matched-arm divergences", H2),
        Paragraph(top_txt + ".", BODY),
        Paragraph("Spatial / behaviour absolutes (lab | matched)", H2),
        Paragraph(
            f"Place-tuned fraction: {fmt_pair('frac_place_tuned')}. "
            f"Peak rate: {fmt_pair('peak_rate_median')}. "
            f"Spatial information: {fmt_pair('spatial_info_median')}. "
            f"Field size: {fmt_pair('place_field_size_median')}. "
            f"Speed cells: {fmt_pair('frac_speed_pass')}. "
            f"Speed median: {fmt_pair('speed_median')}. "
            f"Turning-angle SD: {fmt_pair('turning_angle_std')}. "
            f"Refractory violations: {fmt_pair('refractory_violation_rate')}.",
            BODY,
        ),
        Paragraph("Discriminator (nulls near chance)", H2),
    ]
    if disc is not None and len(disc):
        for _, r in disc.iterrows():
            null = r.get("auc_group_shuffle_null_stratified", r.get("auc_group_shuffle_null"))
            null_s = f"{float(null):.2f}" if null is not None and np.isfinite(null) else "—"
            story.append(
                Paragraph(
                    f"{r['arm']}: AUC(logit) = {float(r['auc_logit']):.2f}; "
                    f"group-shuffle null ≈ {null_s}.",
                    BODY,
                )
            )
    story.append(
        Paragraph(
            "Calibration (Q13–Q15) deferred to Report 4b. See story page for takeaways and simulator to-do.",
            SM,
        )
    )
    return _page_from_story(out / "_findings.pdf", story)


def todo_page(data: Path, out: Path) -> Path:
    """One numbered request per mechanism (1..n); no duplicate border rows."""
    meta = json.loads((data / "data_meta.json").read_text())
    fig3 = pd.read_csv(data / "data_fig3_divergence.csv")

    def R(base: str) -> str:
        r = _fig3_row(fig3, base)
        return f"{float(r.R_matched):.1f}" if r is not None and np.isfinite(r.R_matched) else "—"

    items = [
        (
            "Refractory period",
            f"Add a public refractory / min-ISI parameter to spike generation "
            f"(inhomogeneous Poisson today; no public knob). Target: bring refractory "
            f"violation R toward 1 (now R = {R('refractory_violation_rate')}). Structural.",
        ),
        (
            "Speed / turning knobs or external trajectories",
            f"Expose speed_* / rotational_velocity_* (DF6), or accept external trajectories "
            f"so lab behaviour can be replayed. Speed R = {R('speed_median')}; "
            f"turning-angle SD R = {R('turning_angle_std')}; "
            f"thigmotaxis held at {meta['thigmotaxis']} (not fitted). Structural.",
        ),
        (
            "Independent head direction",
            "Lab head direction differs from movement direction (median ~45°); "
            "sim HD = movement heading. Add an independent HD drive.",
        ),
        (
            "Tuned-cell prevalence and per-spike information",
            f"Reduce excess border / HD prevalence and over-strong behavioural encoding "
            f"(place-tuned R = {R('frac_place_tuned')}; border R = {R('frac_border_pass')}; "
            f"HD R = {R('frac_hd_pass')}). Speed-cell excess is near the reliability floor "
            f"after the keep-mask fix (R = {R('frac_speed_pass')}) and is not a primary "
            f"cell-type request. Raise held-out bits/spike realism among place-tuned units.",
        ),
        (
            "Within-session drift",
            "Lab place maps are less reliable (adjacent-block r ≈ 0.48 vs 0.89) and drift "
            "slowly (slope ≈ −0.024/block vs ~0). Add a public nonstationarity / "
            "representational-drift mechanism so simulated units are not frozen across the session.",
        ),
        (
            "Expose the capture cap (DF7)",
            "Lab-format cell capture caps each n_* at baseline RatInABox size, so raising "
            "unit_density_per_mm cannot grow yield past ~100 MEC units. Allow n_* to scale "
            "above baseline with density (DF7).",
        ),
        (
            "Honour SimConfig.seed (DF5)",
            "Wire config.seed into the Agent loop (today an unused RNG is built; Agent is "
            "unseeded). Document public seeding (DF5).",
        ),
    ]
    story = [
        Paragraph("Simulator to-do (requests to the other lab)", H1),
        Paragraph(
            "One request per mechanism. Numbers from the final post keep-mask Fig 3 tidy.",
            BODY,
        ),
    ]
    for i, (title, body) in enumerate(items, 1):
        story.append(Paragraph(f"<b>{i}. {title}.</b> {body}", BODY))
    return _page_from_story(out / "_todo.pdf", story)


def assemble(figs: Path, data: Path, out_pdf: Path, *, draft_q1_q4: bool = False) -> Path:
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_pdf.parent / "_pdf_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    findings_page(data, tmp)
    todo_page(data, tmp)
    order = [
        tmp / "_findings.pdf",
        tmp / "_todo.pdf",
    ]
    names = ["Q1_design", "Q2_divergence", "Q3_units", "Q4_sorting"]
    if not draft_q1_q4:
        names += [
            "Q5_spatial", "Q6_rhythm", "Q7_population", "Q8_behaviour",
            "Q9_decoding", "Q10_discriminator", "Q11_variability", "Q12_confounds",
            "Q13_calib_4b", "Q14_calib_4b", "Q15_calib_4b", "Q16_validity",
        ]
    writer = PdfWriter()
    for p in order:
        if p.is_file():
            writer.append(str(p))
    for name in names:
        pdf = figs / f"{name}.pdf"
        if not pdf.is_file():
            continue
        reader = PdfReader(str(pdf))
        # wrap on A4 with small margin via reportlab blank + merge scale
        page = reader.pages[0]
        writer.add_page(page)
    with out_pdf.open("wb") as f:
        writer.write(f)
    return out_pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--draft-q1-q4", action="store_true")
    args = ap.parse_args()
    path = assemble(Path(args.figs), Path(args.data), Path(args.out), draft_q1_q4=args.draft_q1_q4)
    print(json.dumps({"pdf": str(path)}, indent=2))


if __name__ == "__main__":
    main()
