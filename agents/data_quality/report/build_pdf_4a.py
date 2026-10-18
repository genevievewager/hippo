#!/usr/bin/env python3
"""Assemble Report 4a PDF from rendered Q*.pdf figures + findings pages."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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


def findings_page(data: Path, out: Path) -> Path:
    find = pd.read_csv(data / "data_findings.csv")
    meta = json.loads((data / "data_meta.json").read_text())
    disc = pd.read_csv(data / "data_discriminator.csv")
    story = [
        Paragraph("Report 4a — Findings (measured)", H1),
        Paragraph(
            f"Lab n={meta['n_lab_sessions']} sessions ({meta['n_animals']} animals); "
            f"matched n={meta['n_matched_sessions']}; default seeds={meta['n_default_seeds']}. "
            f"thigmotaxis={meta['thigmotaxis']} (not fitted). "
            f"Censor assumption: {meta['censor_isi_ms']:.1f} ms lab-equivalent duplicate removal.",
            BODY,
        ),
        Paragraph("Top divergences (matched arm)", H2),
    ]
    matched = find[find.arm == "matched"].head(8)
    for _, r in matched.iterrows():
        story.append(
            Paragraph(
                f"<b>Q2</b> {r['metric']}: R={r['R']:.2f} "
                f"[{r['R_lo90']:.2f}, {r['R_hi90']:.2f}]; "
                f"direction={'sim higher' if r['direction'] > 0 else 'sim lower'}; family={r['family']}.",
                BODY,
            )
        )
    story.append(Paragraph("Discriminator", H2))
    for _, r in disc.iterrows():
        story.append(
            Paragraph(
                f"<b>Q10</b> {r['arm']}: AUC(logit)={r['auc_logit']:.3f}; "
                f"null GroupKFold={r['auc_group_shuffle_null']}; "
                f"null StratifiedGroupKFold={r['auc_group_shuffle_null_stratified']}.",
                BODY,
            )
        )
    story.append(
        Paragraph(
            "Generator has no public refractory parameter; sim GT sub-ms ISIs are a structural candidate (Q3).",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "Calibration (Q13–Q15) deferred to Report 4b (proxy fidelity failed objective gate).",
            SM,
        )
    )
    return _page_from_story(out / "_findings.pdf", story)


def todo_page(data: Path, out: Path) -> Path:
    find = pd.read_csv(data / "data_findings.csv")
    meta = json.loads((data / "data_meta.json").read_text())
    matched = find[find.arm == "matched"].head(10)
    story = [
        Paragraph("Simulator to-do (requests to the other lab)", H1),
        Paragraph(
            "Each item is a measured divergence phrased as a change request. "
            "Hypotheses about which knob is responsible are labelled as such.",
            BODY,
        ),
    ]
    requests = []
    for _, r in matched.iterrows():
        base = str(r["metric"]).split("@", 1)[0]
        if base.startswith("refractory") or "isi" in base:
            req = (
                f"Add a public refractory / min-ISI parameter to spike generation "
                f"(none found; Poisson per bin today). Target: bring refractory violation "
                f"R toward 1 (now R={r['R']:.1f}). Structural."
            )
        elif base in {"turning_angle_std", "speed_median", "speed_mean", "accel_std"}:
            req = (
                f"Expose speed_* / rotational_velocity_* (DF6). Behaviour R for {base} "
                f"is {r['R']:.1f}; thigmotaxis held at {meta['thigmotaxis']} and was not fitted. Structural."
            )
        elif "coverage" in base or "sparsity" in base or "grid" in base:
            req = (
                f"Review spatial-field / coverage generation for {base} "
                f"(R={r['R']:.1f}). Hypothesis: place-field / grid parameters."
            )
        elif "participation" in base or "pca_var" in base:
            req = (
                f"Review population covariance / dimensionality for {base} (R={r['R']:.1f})."
            )
        else:
            req = f"Investigate {base} (R={r['R']:.1f}, family={r['family']})."
        requests.append(req)
    # dedupe similar
    seen = set()
    for i, req in enumerate(requests, 1):
        key = req.split("(")[0][:60]
        if key in seen:
            continue
        seen.add(key)
        story.append(Paragraph(f"<b>{i}.</b> {req}", BODY))
    story.append(
        Paragraph(
            "Also: accept external trajectories (DF6) so lab behaviour can be replayed exactly when knobs are insufficient.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>HD dissociation:</b> lab head direction differs from movement direction "
            "(median ~45°); the sim has no such dissociation (sim HD = movement heading). "
            "Add an independent HD drive.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>Within-session drift:</b> lab place maps are less reliable "
            "(adjacent-block r ≈ 0.48 vs 0.89) and drift slowly (slope ≈ −0.024/block vs ~0). "
            "Add a public nonstationarity / representational-drift mechanism. "
            "Speed-cell excess is near the reliability floor after the keep-mask fix (R ≈ 1.26) "
            "and is not a primary cell-type request; border and HD remain.",
            BODY,
        )
    )
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
