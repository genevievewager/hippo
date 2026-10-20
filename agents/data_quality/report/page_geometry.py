#!/usr/bin/env python3
"""Fixed A4 portrait page geometry for Report 4a (Report 1 margins + 180 mm figures)."""

from __future__ import annotations

import io
from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

# Report 1 story margins
MARGIN = 14 * mm
FIG_WIDTH = 180 * mm  # matches W_FULL = 180 * MM in quadrant_n5.figures
PAGE_W, PAGE_H = A4
CONTENT_W = PAGE_W - 2 * MARGIN
# Figure sits in the upper content band; caption reserve below when present
CAPTION_RESERVE = 0  # captions live inside the matplotlib figure
FIG_TOP = PAGE_H - MARGIN
FIG_MAX_H = PAGE_H - 2 * MARGIN - CAPTION_RESERVE


def place_figure_on_a4(fig_pdf: Path, out_page_pdf: Path | None = None):
    """Place a 180 mm-wide figure PDF onto one A4 portrait page at fixed width.

    Scale is determined only by FIG_WIDTH / native_width — never by content height.
    Tall figures that exceed FIG_MAX_H are a layout bug (overlap audit will catch edge overflow).
    """
    fig = PdfReader(str(fig_pdf)).pages[0]
    fw = float(fig.mediabox.width)
    fh = float(fig.mediabox.height)
    # Fixed width scale (points)
    s = FIG_WIDTH / fw
    placed_h = fh * s
    # Centre horizontally in content band; pin top to FIG_TOP
    x = MARGIN + (CONTENT_W - FIG_WIDTH) / 2.0
    y = FIG_TOP - placed_h

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    # Force a blank A4 page into the buffer (empty save can yield 0 pages).
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    c.showPage()
    c.save()
    buf.seek(0)
    page = PdfReader(buf).pages[0]
    page.merge_transformed_page(fig, Transformation().scale(s).translate(x, y))

    if out_page_pdf is not None:
        w = PdfWriter()
        w.add_page(page)
        with out_page_pdf.open("wb") as f:
            w.write(f)
    return page, {"scale": s, "placed_h_mm": placed_h / mm, "native_w_pt": fw, "native_h_pt": fh}


def wrap_pdf_pages_a4(src_pdf: Path, dst_pdf: Path) -> dict:
    """Re-wrap every page of src onto A4 with the same frame (for already-A4 text pages, copy)."""
    reader = PdfReader(str(src_pdf))
    w = PdfWriter()
    notes = []
    for i, page in enumerate(reader.pages):
        pw = float(page.mediabox.width)
        ph = float(page.mediabox.height)
        # Already A4 (within 2 pt)? keep
        if abs(pw - PAGE_W) < 2 and abs(ph - PAGE_H) < 2:
            w.add_page(page)
            notes.append({"page": i + 1, "action": "keep_a4"})
            continue
        # Treat as a figure page: write temp and place
        tmp = dst_pdf.parent / f"_wrap_tmp_{i}.pdf"
        tw = PdfWriter()
        tw.add_page(page)
        with tmp.open("wb") as f:
            tw.write(f)
        placed, meta = place_figure_on_a4(tmp)
        w.add_page(placed)
        notes.append({"page": i + 1, "action": "placed_180mm", **meta})
        tmp.unlink(missing_ok=True)
    dst_pdf.parent.mkdir(parents=True, exist_ok=True)
    with dst_pdf.open("wb") as f:
        w.write(f)
    return {"n_pages": len(w.pages), "notes": notes}
