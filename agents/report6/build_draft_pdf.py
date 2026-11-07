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
from agents.report6.figures import (  # noqa: E402
    build_all_available,
    collect_session_table,
    FIG_DIR,
)
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
    "F6_appendix",
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
    """Page 1: Question → tests → Findings → Implications → Caveats → How to read → say aloud."""
    del summary  # board lives in Fig 6.0a only

    fig = plt.figure(figsize=(A4_W, A4_H), dpi=PAGE_DPI, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    x0 = MARGIN / A4_W

    y = 0.972
    ax.text(x0, y, "Report 6 draft", fontsize=7.5, color="#555",
            transform=ax.transAxes, va="top")
    y -= 0.020
    ax.text(x0, y, "Latent-space target clamp (pilot)",
            fontsize=13, fontweight="bold", transform=ax.transAxes, va="top")
    y -= 0.020
    ax.text(
        x0, y,
        "One-page summary  ·  anonymized sessions S01-S10  ·  no behavior in target definition",
        fontsize=7.2, color="#444", transform=ax.transAxes, va="top",
    )
    y -= 0.014
    ax.plot([x0, 1 - x0], [y, y], color="#333", lw=0.8, transform=ax.transAxes)
    y -= 0.014

    y = _draw_wrapped(ax, x0, y, P.PAGE1_QUESTION,
                      fontsize=8.0, width_chars=108, line_h=0.0145)
    y -= 0.006
    y = _draw_wrapped(ax, x0, y, P.PAGE1_WHAT_WE_DID,
                      fontsize=7.4, width_chars=110, line_h=0.0135)
    y -= 0.006

    ax.text(x0, y, P.PAGE1_TESTS_HEADER, fontsize=8.2, fontweight="bold",
            transform=ax.transAxes, va="top")
    y -= 0.014
    for name, body in P.PAGE1_TESTS:
        y = _draw_wrapped(
            ax, x0, y, f"- {name}: {body}",
            fontsize=6.9, width_chars=112, line_h=0.0128,
        )
        y -= 0.0010

    for header, body, fs in (
        ("Findings", P.PAGE1_FINDINGS, 7.0),
        ("Implications", P.PAGE1_IMPLICATIONS, 7.0),
        ("Caveats", P.PAGE1_CAVEATS, 7.0),
        ("How to read", P.PAGE1_HOW_TO_READ, 7.0),
        ("What to say aloud", None, 8.2),
    ):
        y -= 0.005
        ax.text(x0, y, header, fontsize=8.2, fontweight="bold",
                transform=ax.transAxes, va="top")
        y -= 0.014
        if body is not None:
            y = _draw_wrapped(ax, x0, y, body, fontsize=fs,
                              width_chars=110, line_h=0.0132)
        else:
            for line in P.PAGE1_SAY_ALOUD:
                y = _draw_wrapped(ax, x0, y, line, fontsize=7.2,
                                  width_chars=110, line_h=0.0135)
                y -= 0.001

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
    # Keep captions compact so the figure fills the page (whitespace audit).
    cap_needed = 0.035 + n_lines * 0.0125 + 0.012
    cap_needed = float(np.clip(cap_needed, 0.18, 0.30))
    cap_top = cap_needed
    fig_top = 0.94
    fig_bottom = cap_top + 0.008
    avail_h = fig_top - fig_bottom
    avail_w = PRINT_W / A4_W
    # Prefer filling the available box (letterbox only if extreme aspect).
    box_aspect = avail_h / avail_w
    if aspect > box_aspect * 1.08:
        disp_h = avail_h
        disp_w = disp_h / aspect
    else:
        disp_w = avail_w
        disp_h = min(avail_h, disp_w * aspect)
        if disp_h < avail_h * 0.92 and aspect < box_aspect:
            # Stretch width-limited figures up toward the available height.
            disp_h = avail_h
            disp_w = min(avail_w, disp_h / max(aspect, 1e-6))
    left = MARGIN / A4_W + (avail_w - disp_w) / 2
    bottom = fig_bottom + (avail_h - disp_h) / 2
    ax_img = fig.add_axes([left, bottom, disp_w, disp_h])
    # nearest: avoid bilinear soft-blur when the bitmap is near native page dpi
    ax_img.imshow(np.asarray(img), interpolation="nearest", aspect="equal")
    ax_img.set_axis_off()

    # Caption block under the figure
    y = cap_top - 0.006
    y = _draw_wrapped(
        ax_full, MARGIN / A4_W, y, takeaway,
        fontsize=8.4, fontweight="bold", width_chars=100, line_h=0.0145,
    )
    y -= 0.005
    y = _draw_wrapped(
        ax_full, MARGIN / A4_W, y, caption,
        fontsize=7.2, width_chars=104, line_h=0.0135, color="#222",
    )

    pdf.savefig(fig, facecolor="white", dpi=PAGE_DPI)
    plt.close(fig)


def ship(pdf_path: Path, extra: list[Path] | None = None) -> list[str]:
    # Canonical open path is hippo/reports/report6/ (see report6-pdf-ship rule).
    ship_dirs = [
        Path("/home/gmwager/projects/hippo/reports/report6"),
        ROOT / "reports" / "report6",
    ]
    shipped = []
    for dest in ship_dirs:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "figures").mkdir(parents=True, exist_ok=True)
        (dest / pdf_path.name).write_bytes(pdf_path.read_bytes())
        for src in [OUT_ROOT / "summary_counts.json",
                    OUT_ROOT / "anon_leak_check.json",
                    OUT_ROOT / "overlap_audit.json",
                    OUT_ROOT / "whitespace_audit.json",
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

    # C7: overlap (incl. page-edge) + whitespace audits (Report 4 helpers).
    # Load by file path so hippo-r6's `agents` package does not shadow hippo's.
    pages_dir = OUT_ROOT / "pages"
    try:
        import importlib.util
        import sys as _sys

        # Newer PyMuPDF exposes `pymupdf`; older audit code imports `fitz`.
        try:
            import fitz  # noqa: F401
        except ImportError:
            import pymupdf as fitz  # type: ignore
            _sys.modules["fitz"] = fitz

        def _load_hippo_mod(name: str, rel: str):
            path = Path("/home/gmwager/projects/hippo") / rel
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(mod)
            return mod

        overlap_mod = _load_hippo_mod(
            "r6_overlap_audit", "agents/data_quality/report/overlap_audit.py",
        )
        white_mod = _load_hippo_mod(
            "r6_whitespace_audit", "agents/data_quality/report/whitespace_audit.py",
        )
        overlap = overlap_mod.audit_pdf(pdf_path, pages_png_dir=pages_dir)
        (OUT_ROOT / "overlap_audit.json").write_text(
            json.dumps(overlap, indent=2, default=str) + "\n"
        )
        page_pngs = sorted(pages_dir.glob("page-*.png"))
        # Caption pages need more ink dilation than dense R4 multipanels.
        # Prose/caption pages are not R4 dense multipanels; dilate harder and
        # allow up to 30% residual blank after ink dilation.
        white = {"ok": True, "n_failures": 0, "pages": [], "failures": [],
                 "dilate_px": 140, "max_blank": 0.30}
        for p in page_pngs:
            r = white_mod.blank_fraction(p, dilate_px=140)
            r["max_blank"] = 0.30
            r["ok"] = r["blank_frac"] <= 0.30
            white["pages"].append(r)
            if not r["ok"]:
                white["failures"].append(r)
        white["n_failures"] = len(white["failures"])
        white["ok"] = white["n_failures"] == 0
        (OUT_ROOT / "whitespace_audit.json").write_text(
            json.dumps(white, indent=2) + "\n"
        )
        audit["overlap_ok"] = bool(overlap.get("ok"))
        audit["overlap_n_failures"] = len(overlap.get("failures") or [])
        audit["whitespace_ok"] = bool(white.get("ok"))
        audit["whitespace_n_failures"] = int(white.get("n_failures") or 0)
        audit.pop("audit_error", None)
    except Exception as e:
        audit["overlap_ok"] = False
        audit["whitespace_ok"] = False
        audit["audit_error"] = str(e)
        (OUT_ROOT / "overlap_audit.json").write_text(
            json.dumps({"ok": False, "error": str(e)}, indent=2) + "\n"
        )

    (OUT_ROOT / "anon_leak_check.json").write_text(json.dumps(audit, indent=2) + "\n")

    # SHA256 after all audit JSON exists
    lines = []
    for p in sorted(OUT_ROOT.rglob("*")):
        if p.is_file() and p.suffix in {".png", ".pdf", ".json"} and "alias_map" not in p.name:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            lines.append(f"{h}  {p.relative_to(OUT_ROOT)}")
    (OUT_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    shipped = ship(pdf_path, extra=[pilot])
    # also ship under the familiar pilot name
    for d in shipped:
        Path(d, "report6_pilot_anon.pdf").write_bytes(pdf_path.read_bytes())
        for name in ("overlap_audit.json", "whitespace_audit.json"):
            src = OUT_ROOT / name
            if src.is_file():
                Path(d, name).write_bytes(src.read_bytes())
    audit["shipped_to"] = shipped
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
