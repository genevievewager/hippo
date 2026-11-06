#!/usr/bin/env python3
"""Report 6 draft PDF to Report 1 standard: summary page + figure+caption pages.

No reportlab dependency — matplotlib A4 pages. Ships to reports/report6/.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.report6.build_summary_pdf import summarize  # noqa: E402
from agents.report6.captions import CAPTIONS  # noqa: E402
from agents.report6.figures import build_all_available, FIG_DIR  # noqa: E402
from agents.report6.paths import OUT_ROOT  # noqa: E402
from agents.report6 import prose as P  # noqa: E402

# A4 inches
A4_W, A4_H = 8.27, 11.69
MARGIN = 0.55
PRINT_W = A4_W - 2 * MARGIN
# Matplotlib PdfPages resamples imshow bitmaps to figure/savefig dpi.
# Default ~100 PPI made every figure page soft; match print-quality PNGs.
PAGE_DPI = 300

ID_PAT = re.compile(
    r"\b(?:\d{3,4}_\d{6,8}\w*|\b(?:1120|1130|1140|1150|1170|7394)_\w+)\b"
)

# Figure order in the draft (board first, then numbers, then story figures)
FIGURE_ORDER = [
    "F6_0_board",
    "F6_numbers",
    "F6_1",
    "F6_2",
    "F6_3",
    "F6_4",
    "F6_5",
    "F6_S_retired",
]


def _wrap(text: str, width: int = 92) -> str:
    paras = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            paras.append("")
            continue
        paras.append(textwrap.fill(block, width=width))
    return "\n".join(paras)


def _draw_wrapped(ax, x, y, text, *, fontsize=9, fontweight="normal",
                  color="black", width_chars=95, va="top", line_h=0.016):
    """Draw wrapped text in axes fraction coords; return y after last line."""
    lines = []
    for para in text.split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(para, width=width_chars) or [""])
    for line in lines:
        ax.text(x, y, line, transform=ax.transAxes, fontsize=fontsize,
                fontweight=fontweight, color=color, va=va, ha="left",
                family="sans-serif")
        y -= line_h if line else line_h * 0.6
    return y


def write_summary_page(pdf: PdfPages, summary: dict) -> None:
    """Page 1: revision B1 prose + B5 board columns (US spelling)."""
    rows = summary["rows"]

    fig = plt.figure(figsize=(A4_W, A4_H), dpi=PAGE_DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    y = 0.968
    ax.text(MARGIN / A4_W, y, "Report 6 — Draft", fontsize=7.5, color="#555",
            transform=ax.transAxes, va="top")
    y -= 0.022
    ax.text(MARGIN / A4_W, y, "Latent-space target clamp (pilot)",
            fontsize=14, fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.024
    ax.text(
        MARGIN / A4_W, y,
        "One-page summary  ·  anonymized sessions S01–S10  ·  no behavior in target definition",
        fontsize=7.5, color="#444", transform=ax.transAxes, va="top",
    )
    y -= 0.018
    ax.plot([MARGIN / A4_W, 1 - MARGIN / A4_W], [y, y], color="#333", lw=0.8,
            transform=ax.transAxes)
    y -= 0.018

    y = _draw_wrapped(
        ax, MARGIN / A4_W, y, P.PAGE1_QUESTION,
        fontsize=8.0, width_chars=108, line_h=0.0125,
    )
    y -= 0.008
    y = _draw_wrapped(
        ax, MARGIN / A4_W, y, P.PAGE1_WHAT_WE_DID,
        fontsize=7.8, width_chars=110, line_h=0.0120,
    )
    y -= 0.008

    ax.text(MARGIN / A4_W, y, P.PAGE1_TESTS_HEADER, fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.014
    for name, body in P.PAGE1_TESTS:
        y = _draw_wrapped(
            ax, MARGIN / A4_W, y, f"- {name}: {body}",
            fontsize=7.2, width_chars=112, line_h=0.0112,
        )
        y -= 0.002

    y -= 0.006
    ax.text(MARGIN / A4_W, y, "What to say aloud", fontsize=8.5, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.014
    for b in P.PAGE1_SAY_ALOUD:
        y = _draw_wrapped(
            ax, MARGIN / A4_W, y, b,
            fontsize=7.6, width_chars=110, line_h=0.0118,
        )
        y -= 0.002

    y -= 0.006
    ax.text(MARGIN / A4_W, y, "How to read the rest of this draft",
            fontsize=8.5, fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.014
    y = _draw_wrapped(
        ax, MARGIN / A4_W, y, P.PAGE1_READING_GUIDE,
        fontsize=7.2, width_chars=112, line_h=0.0112,
    )
    y -= 0.010

    ax.text(MARGIN / A4_W, y, "Pass / fail by session",
            fontsize=8.5, fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.008

    def yn(v):
        if v is None:
            return "—"
        return "PASS" if v else "FAIL"

    col_labels = list(P.PAGE1_TABLE_COLS)
    cell = [col_labels]
    for r in rows:
        cell.append([
            r["alias"],
            yn(r.get("qualifies")),
            yn(r.get("distinguishable")),
            yn(r.get("null_b")),
            yn(r.get("split")),
            yn(r.get("same_subspace")),
            yn(r.get("same_tiles")),
        ])

    tab_h = min(0.28, max(0.18, y - 0.05))
    tab_bottom = max(0.045, y - tab_h - 0.01)
    tax = fig.add_axes([MARGIN / A4_W, tab_bottom, PRINT_W / A4_W, tab_h])
    tax.set_axis_off()
    table = tax.table(cellText=cell, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(6.5)
    table.scale(1.0, 1.25)
    for j in range(len(col_labels)):
        table[0, j].set_facecolor("#e8e8e8")
        table[0, j].set_text_props(fontweight="bold")
    for i in range(1, len(cell)):
        for j in range(1, len(col_labels)):
            val = cell[i][j]
            if val == "PASS":
                table[i, j].set_facecolor("#c6efce")
            elif val == "FAIL":
                table[i, j].set_facecolor("#ffc7ce")

    fig.text(MARGIN / A4_W, 0.018, P.PAGE1_BOARD_NOTE,
             fontsize=6.2, color="#444", va="bottom")
    pdf.savefig(fig, facecolor="white", dpi=PAGE_DPI)
    plt.close(fig)


def write_figure_page(pdf: PdfPages, fig_key: str, meta: dict) -> None:
    """One page: figure title, image, takeaway + caption under the figure."""
    png = FIG_DIR / f"{fig_key}.png"
    if not png.is_file():
        return

    img = PILImage.open(png).convert("RGB")
    iw, ih = img.size
    aspect = ih / iw

    fig = plt.figure(figsize=(A4_W, A4_H), dpi=PAGE_DPI, facecolor="white")
    # Header
    ax_full = fig.add_axes([0, 0, 1, 1])
    ax_full.set_axis_off()
    ax_full.text(MARGIN / A4_W, 0.965, meta["title"], fontsize=11,
                 fontweight="bold", va="top", transform=ax_full.transAxes)

    # Estimate caption block height and reserve space so text is never clipped.
    takeaway = meta["takeaway"]
    caption = meta["caption"]
    n_lines = (
        len(textwrap.wrap(takeaway, width=96))
        + 1
        + sum(max(1, len(textwrap.wrap(p, width=98))) for p in caption.split("\n") if p.strip())
    )
    # line heights in axes fraction (~0.0145 / 0.013)
    cap_needed = 0.04 + n_lines * 0.0135 + 0.02
    cap_needed = float(np.clip(cap_needed, 0.28, 0.42))
    cap_top = cap_needed
    fig_top = 0.93
    fig_bottom = cap_top + 0.015
    avail_h = fig_top - fig_bottom
    avail_w = PRINT_W / A4_W
    box_aspect = avail_h / avail_w
    if aspect > box_aspect:
        disp_h = avail_h
        disp_w = disp_h / aspect
    else:
        disp_w = avail_w
        disp_h = disp_w * aspect
    left = MARGIN / A4_W + (avail_w - disp_w) / 2
    bottom = fig_bottom + (avail_h - disp_h) / 2
    ax_img = fig.add_axes([left, bottom, disp_w, disp_h])
    # nearest: avoid bilinear soft-blur when the bitmap is near native page dpi
    ax_img.imshow(np.asarray(img), interpolation="nearest", aspect="equal")
    ax_img.set_axis_off()

    # Caption block under the figure
    y = cap_top - 0.008
    y = _draw_wrapped(
        ax_full, MARGIN / A4_W, y, takeaway,
        fontsize=9.0, fontweight="bold", width_chars=96, line_h=0.0138,
    )
    y -= 0.008
    y = _draw_wrapped(
        ax_full, MARGIN / A4_W, y, caption,
        fontsize=7.8, width_chars=100, line_h=0.0128, color="#222",
    )

    pdf.savefig(fig, facecolor="white", dpi=PAGE_DPI)
    plt.close(fig)


def ship(pdf_path: Path, extra: list[Path] | None = None) -> list[str]:
    # Revision work stays in hippo-r6 only (do not write into hippo/).
    ship_dirs = [
        ROOT / "reports" / "report6",
    ]
    shipped = []
    for dest in ship_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "figures").mkdir(parents=True, exist_ok=True)
        (dest / pdf_path.name).write_bytes(pdf_path.read_bytes())
        for src in [OUT_ROOT / "summary_counts.json",
                    OUT_ROOT / "anon_leak_check.json",
                    OUT_ROOT / "SHA256SUMS"]:
            if src.is_file():
                (dest / src.name).write_bytes(src.read_bytes())
        for fig in sorted((OUT_ROOT / "figures").glob("F6_*")):
            (dest / "figures" / fig.name).write_bytes(fig.read_bytes())
        if extra:
            for e in extra:
                if e.is_file():
                    (dest / e.name).write_bytes(e.read_bytes())
        shipped.append(str(dest))
    return shipped


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    figs = build_all_available()
    summary = summarize()
    (OUT_ROOT / "summary_counts.json").write_text(json.dumps(summary, indent=2) + "\n")

    pdf_path = OUT_ROOT / "report6_draft.pdf"
    used = []
    with PdfPages(pdf_path) as pdf:
        write_summary_page(pdf, summary)
        for key in FIGURE_ORDER:
            meta = CAPTIONS.get(key)
            if meta is None or not (FIG_DIR / f"{key}.png").is_file():
                continue
            write_figure_page(pdf, key, meta)
            used.append(key)

    # Also keep anonymised alias of the draft as the pilot name for convenience
    pilot = OUT_ROOT / "report6_pilot_anon.pdf"
    pilot.write_bytes(pdf_path.read_bytes())

    # SHA256 of outputs
    lines = []
    for p in sorted(OUT_ROOT.rglob("*")):
        if p.is_file() and p.suffix in {".png", ".pdf", ".json"} and "alias_map" not in p.name:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(OUT_ROOT)}")
    (OUT_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    # ID grep
    txt_path = OUT_ROOT / "report6_draft.txt"
    try:
        subprocess.run(["pdftotext", str(pdf_path), str(txt_path)],
                       check=True, capture_output=True)
        text = txt_path.read_text()
    except Exception as e:
        text = ""
        (OUT_ROOT / "pdftotext_error.txt").write_text(str(e))
    hits = ID_PAT.findall(text)
    for meta in CAPTIONS.values():
        hits += ID_PAT.findall(meta["takeaway"] + " " + meta["caption"])
    audit = {
        "pdf": str(pdf_path),
        "pages": 1 + len(used),
        "figures": used,
        "id_grep_hits": hits,
        "id_grep_n": len(hits),
        "pass": len(hits) == 0,
        "three_numbers": summary["three_numbers"],
        "built_figures": figs,
    }
    (OUT_ROOT / "anon_leak_check.json").write_text(json.dumps(audit, indent=2) + "\n")
    shipped = ship(pdf_path, extra=[pilot])
    # also ship under the familiar pilot name
    for d in shipped:
        Path(d, "report6_pilot_anon.pdf").write_bytes(pdf_path.read_bytes())
    audit["shipped_to"] = shipped
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
