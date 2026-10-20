#!/usr/bin/env python3
"""Report 4a v8 — page 1 prose only (no figure, table, or code-style names)."""

from __future__ import annotations

import json
import re
from pathlib import Path

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
    for reg, bold in (
        (FONT_DIR / "LiberationSans-Regular.ttf", FONT_DIR / "LiberationSans-Bold.ttf"),
        (
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ),
    ):
        if reg.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("Sans", str(reg)))
            pdfmetrics.registerFont(TTFont("Sans-Bold", str(bold)))
            return "Sans", "Sans-Bold"
    raise RuntimeError("Liberation Sans not found")


FONT, BOLD = _fonts()
W, H = A4
M = 12 * mm

H1 = ParagraphStyle("h1", fontName=BOLD, fontSize=12.0, leading=14.5, spaceAfter=5.0)
H2 = ParagraphStyle("h2", fontName=BOLD, fontSize=9.8, leading=12.2, spaceBefore=6.0, spaceAfter=3.0)
BODY = ParagraphStyle("b", fontName=FONT, fontSize=9.4, leading=12.8, spaceAfter=5.0)
FIND = ParagraphStyle("f", fontName=FONT, fontSize=9.4, leading=12.8, spaceAfter=5.5)
SM = ParagraphStyle(
    "s", fontName=FONT, fontSize=7.6, leading=9.8,
    textColor=colors.HexColor("#52514e"), spaceAfter=5.0,
)


def write_story_v8(data: Path, out_pdf: Path) -> Path:
    meta = json.loads((data / "data_meta.json").read_text())
    n_an = meta["n_animals"]
    n_lab = meta["n_lab_sessions"]
    n_mat = meta["n_matched_sessions"]
    thig = meta["thigmotaxis"]

    flow = [
        Paragraph(
            "How far is the simulator from the lab? A paired data-quality audit of "
            "default and condition-matched arms against hippocampal recordings",
            H1,
        ),
        Paragraph(
            "Report 4a v8, tag report4a-v8",
            SM,
        ),
        Paragraph(
            f"{n_an} animals, {n_lab} room-A sessions · "
            f"condition-matched {n_mat} · thigmotaxis default {thig} · "
            "internal draft · numbers frozen (v5 tidy plus N-ladder)",
            SM,
        ),
        Paragraph("The question", H2),
        Paragraph(
            "The simulator is the training and testing ground for hippocampal brain–computer "
            "interfaces before they reach real animals. If its spikes, place fields, and "
            "behaviour already diverge from the lab on basic statistics, then offline decoding "
            "wins and closed-loop designs built on those spikes will not transfer. This report "
            "asks which absolute statistics already match hippocampal recordings and which "
            "still diverge, so that calibration work has a concrete punch list.",
            BODY,
        ),
        Paragraph("What was done", H2),
        Paragraph(
            f"We paired {n_lab} room-A lab sessions from {n_an} animals with one "
            "condition-matched simulation per session that copies each lab session's recording "
            "conditions (segment duration, per-region unit counts, arena area). It does not "
            "copy behaviour or physiology: the simulator cannot take an external trajectory "
            "and does not expose speed or turning parameters, so the simulated rat moves by "
            f"the simulator's own model (thigmotaxis at its default of {thig}). The same "
            "metric code path scores every arm; the dataset label is the only switch. Absolute "
            "values lead the story (Figures 1–4); the divergence ratio appears only in "
            "Figure 5. Numbers are frozen at the v5 tidy tables, with the N-ladder amendment "
            "for population curves.",
            BODY,
        ),
        Paragraph("Three findings", H2),
        Paragraph(
            "<b>1. Simulated populations carry far more behavioural information (Figure 1).</b> "
            "Among place-tuned units, held-out bits per spike are 0.07 in the lab versus 0.43 "
            "in the condition-matched sim. At twenty units, position is decodable above chance "
            "in the sim but not in the lab. Lab position information leaves chance only near "
            "forty-two units and reaches 0.23 at the session maximum, whereas the sim already "
            "sits at 0.38 by forty units. A decoder trained and tested in the sim therefore "
            "looks optimistic at every population size.",
            FIND,
        ),
        Paragraph(
            "<b>2. Simulated cells are tuned too often, too strongly, and never drift (Figure 2).</b> "
            "The place-tuned fraction is 0.18 in the lab versus 0.37 in the condition-matched "
            "sim (chance would be about one in twenty). Adjacent-block place-map correlation "
            "is 0.48 versus 0.89; lab maps drift within a session while condition-matched maps "
            "stay pinned. Rate maps, spatial information, and bits per spike all point the "
            "same way: the sim is a sharper, more stable place code than the recordings it "
            "stands in for.",
            FIND,
        ),
        Paragraph(
            "<b>3. No refractory period; the rat is too fast, too straight, and never still "
            "(Figure 3).</b> "
            "Condition-matched simulations violate a 1.5-millisecond refractory window far "
            "more often than lab units. Median speed is higher, turning is tighter, and still "
            "periods are scarce. In the sim, head direction equals movement heading—so any "
            "decoder that leans on that residual in the lab will not find it here. The tuning "
            "and information gaps in Figures 1–2 are measured on top of this behaviour "
            "mismatch; the reweighting test in Figure 1g shows that behaviour alone does not "
            "account for them.",
            FIND,
        ),
        Paragraph("What matches", H2),
        Paragraph(
            "Several unit-level and population summaries already sit close under the "
            "condition-matched protocol (Figure 4): mean firing rate, Fano factor, interval "
            "coefficient of variation, within-session rate drift, pairwise correlation, "
            "sparsity, field size, and peak rate. Those are the levers that do not need to "
            "move first. The gaps that remain are information, tuning prevalence, temporal "
            "regularity, and the structure of behaviour.",
            BODY,
        ),
        Paragraph("What this means for deployment", H2),
        Paragraph(
            "Sim decoding is optimistic at every population size. A lab decoder needs on the "
            "order of forty or more units even to beat chance on position, and well over a "
            "hundred for accuracy that would feel useful in a closed loop. Real place maps "
            "also drift within a session, so a static encoder fitted on the first block will "
            "quietly degrade. The information and tuning gaps (Figures 1–2) sit on top of the "
            "behaviour mismatch, and Figure 1g's reweighting test shows behaviour alone does "
            "not explain them. Designs that only pass in the condition-matched sim should be "
            "treated as upper bounds, not as predictions of lab performance.",
            BODY,
        ),
        Paragraph("Caveats", H2),
        Paragraph(
            "All claims are room A only. Subiculum coverage is near-absent, so region "
            "splits lean on MEC. Thigmotaxis is at its default, not swept. Population variance "
            "explained is not a valid absolute on the lab arm (it sits at or below zero); "
            "we cite bits per spike and decode information above null instead. Occ-matched "
            "joint bins are not available because the lab arena is a polygon and the sim "
            "arena is a square. The default arm (five seeds) is shown on figures for "
            "orientation; primary claims compare lab against condition-matched. Calibration "
            "of the remaining gaps is deferred to Report 4b.",
            BODY,
        ),
        Paragraph(
            "How to read the figures. Figure 1 leads with behavioural information and "
            "population decode curves. Figure 2 shows tuning prevalence, bits per spike, and "
            "within-session map drift. Figure 3 covers spike timing and locomotion. Figure 4 "
            "collects the statistics that already match. Figure 5 is the only place the "
            "divergence ratio appears. Supplement S1 holds extra session distributions; "
            "the appendix holds gates, eligibility, and path containment.",
            BODY,
        ),
    ]

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    frame = Frame(
        M, M, W - 2 * M, H - 2 * M, showBoundary=0,
        leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
    )
    rem = frame.addFromList(list(flow), c) or []
    if rem:
        c.setFont(FONT, 7)
        c.setFillColor(colors.HexColor("#c83a39"))
        c.drawString(M, M + 1 * mm, f"[story overflow: {len(rem)} blocks]")
    c.save()
    return out_pdf


_UNDERSCORE = re.compile(r"_")
_DIGITS_ONLY_TABLE = re.compile(
    r"(?m)^(?:\s*[\d.eE+\-]+\s+){2,}[\d.eE+\-]+\s*$"
)


def audit_story_prose(pdf: Path) -> dict:
    """Page-1 prose rules: no underscores; no digits-only table rows."""
    from pypdf import PdfReader

    text = PdfReader(str(pdf)).pages[0].extract_text() or ""
    unders = sorted(set(_UNDERSCORE.findall(text)))
    # also flag identifier-like tokens with underscores
    unders_tok = sorted(set(re.findall(r"\b\w*_\w*\b", text)))
    digit_rows = _DIGITS_ONLY_TABLE.findall(text)
    ok = (len(unders_tok) == 0) and (len(digit_rows) == 0)
    return {
        "ok": ok,
        "underscore_tokens": unders_tok[:20],
        "digits_only_rows": digit_rows[:10],
        "n_chars": len(text),
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = write_story_v8(Path(args.data), Path(args.out))
    print({"story": str(path), "prose_audit": audit_story_prose(path)})
