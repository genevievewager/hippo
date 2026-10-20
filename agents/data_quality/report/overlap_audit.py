#!/usr/bin/env python3
"""Real overlap audit: render PDF pages to PNG and detect text bbox collisions.

Uses PyMuPDF text spans (PDF text layer). Fails on:
  - text overlapping text
  - text overlapping the page edge (within margin)
  - text whose bbox extends outside the page

Also accepts matplotlib figure PNGs and checks ink touching the PNG edge.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


MARGIN_PT = 14 * 72 / 25.4  # 14 mm in PDF points ≈ reportlab MARGIN
MIN_OVERLAP_AREA = 4.0  # pt² — ignore tiny grazing contacts
EDGE_PAD_PX = 2  # PNG edge content pad


def _iou_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return float((ix1 - ix0) * (iy1 - iy0))


def _smaller_area(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))


def _spans_from_page(page) -> list[dict]:
    """Extract visible text spans with bboxes from a PyMuPDF page."""
    spans = []
    d = page.get_text("dict")
    for block in d.get("blocks") or []:
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            for sp in line.get("spans") or []:
                text = (sp.get("text") or "").strip()
                if not text:
                    continue
                bbox = sp.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x0, y0, x1, y1 = map(float, bbox)
                if (x1 - x0) < 0.5 or (y1 - y0) < 0.5:
                    continue
                spans.append({"text": text[:60], "bbox": (x0, y0, x1, y1)})
    return spans


def audit_pdf(pdf: Path, pages_png_dir: Path | None = None) -> dict:
    try:
        import fitz
    except ImportError as e:
        return {"ok": False, "failures": [f"pymupdf required: {e}"], "n_pages": 0}

    doc = fitz.open(str(pdf))
    failures = []
    page_reports = []

    if pages_png_dir is not None:
        pages_png_dir.mkdir(parents=True, exist_ok=True)
        # Render via pdftoppm for delivery PNGs (also used for edge-ink check)
        try:
            subprocess.check_call(
                ["pdftoppm", "-png", "-r", "150", str(pdf), str(pages_png_dir / "page")],
            )
        except Exception as exc:
            failures.append(f"pdftoppm failed: {exc}")

    for i, page in enumerate(doc, 1):
        pw, ph = float(page.rect.width), float(page.rect.height)
        spans = _spans_from_page(page)
        page_fail = []

        # Edge collisions
        for sp in spans:
            x0, y0, x1, y1 = sp["bbox"]
            if x0 < MARGIN_PT * 0.5 or y0 < MARGIN_PT * 0.5 or x1 > pw - MARGIN_PT * 0.5 or y1 > ph - MARGIN_PT * 0.5:
                # Allow content that intentionally sits in the content band but flag true edge hits
                if x0 < 2 or y0 < 2 or x1 > pw - 2 or y1 > ph - 2:
                    page_fail.append({
                        "kind": "page_edge",
                        "text": sp["text"],
                        "bbox": sp["bbox"],
                    })

        # Pairwise text overlaps (same page)
        for a_i in range(len(spans)):
            for b_i in range(a_i + 1, len(spans)):
                a, b = spans[a_i], spans[b_i]
                # Skip if one is substring visually stacked as same line (identical bbox)
                if a["bbox"] == b["bbox"]:
                    continue
                area = _iou_area(a["bbox"], b["bbox"])
                small = _smaller_area(a["bbox"], b["bbox"])
                # Require ≥25% of the smaller span covered (ignore grazing contacts)
                if area >= MIN_OVERLAP_AREA and small > 0 and (area / small) >= 0.25:
                    if len(a["text"]) <= 1 or len(b["text"]) <= 1:
                        continue
                    # Skip pure tick-label collisions (short numeric / mathtext)
                    if _tickish(a["text"]) and _tickish(b["text"]):
                        continue
                    page_fail.append({
                        "kind": "text_text",
                        "a": a["text"],
                        "b": b["text"],
                        "area": area,
                        "frac": area / small,
                        "bbox_a": a["bbox"],
                        "bbox_b": b["bbox"],
                    })

        # PNG edge-ink check
        if pages_png_dir is not None:
            png = pages_png_dir / f"page-{i:02d}.png"
            if not png.is_file():
                # pdftoppm may use page-1.png without zero pad
                cands = list(pages_png_dir.glob(f"page-{i}.png")) + list(pages_png_dir.glob(f"page-{i:02d}.png"))
                png = cands[0] if cands else png
            if png.is_file():
                from PIL import Image
                arr = np.asarray(Image.open(png).convert("L"))
                h, w = arr.shape
                edge = np.concatenate([
                    arr[:EDGE_PAD_PX, :].ravel(),
                    arr[-EDGE_PAD_PX:, :].ravel(),
                    arr[:, :EDGE_PAD_PX].ravel(),
                    arr[:, -EDGE_PAD_PX:].ravel(),
                ])
                dark_frac = float((edge < 240).mean())
                if dark_frac > 0.05:
                    page_fail.append({
                        "kind": "png_edge_ink",
                        "edge_frac": dark_frac,
                        "file": png.name,
                    })

        if page_fail:
            failures.extend([{"page": i, **f} for f in page_fail])
        page_reports.append({"page": i, "n_spans": len(spans), "n_issues": len(page_fail)})

    return {
        "ok": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures[:80],
        "n_pages": len(doc),
        "pages": page_reports,
    }


def _tickish(t: str) -> bool:
    t = t.strip().replace("−", "-").replace("–", "-")
    if len(t) <= 6 and all(ch in "0123456789.-+eE%$\\mathdefault{} " for ch in t):
        return True
    return t.startswith("$")


def audit_mpl_figure(fig) -> list[dict]:
    """Check matplotlib text artists for pairwise overlaps and figure-edge overflow.

    Catches text-on-text *within* a panel (including y-tick labels vs annotations).
    Rasterized figure text never reaches the PDF text layer, so this must run at
    figure-save time — PDF ``text_text`` alone is not enough for embedded PNGs.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches() * fig.dpi
    boxes = []
    for artist in fig.findobj(match=lambda a: hasattr(a, "get_window_extent")):
        try:
            from matplotlib.text import Text
            if not isinstance(artist, Text):
                continue
            if not artist.get_visible():
                continue
            txt = artist.get_text()
            if not str(txt).strip():
                continue
            bb = artist.get_window_extent(renderer).expanded(1.02, 1.02)
            boxes.append({
                "text": str(txt)[:80].replace("\n", " "),
                "bbox": (bb.x0, bb.y0, bb.x1, bb.y1),
                "artist": artist,
            })
        except Exception:
            continue
    fails = []
    pad = 1.0
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        if x0 < -pad or y0 < -pad or x1 > fw + pad or y1 > fh + pad:
            fails.append({"kind": "fig_edge", "text": b["text"], "bbox": b["bbox"]})
        # Longer labels that run past the parent panel edge (not short ticks)
        try:
            ax = b["artist"].axes
            if ax is not None and not _tickish(b["text"]) and len(b["text"]) >= 8:
                ab = ax.get_window_extent(renderer)
                edge_pad = 4.0
                overflow_x = max(0.0, ab.x0 - edge_pad - x0, x1 - (ab.x1 + edge_pad))
                overflow_y = max(0.0, ab.y0 - edge_pad - y0, y1 - (ab.y1 + edge_pad))
                if overflow_x > 6.0 or overflow_y > 6.0:
                    fails.append({
                        "kind": "panel_edge",
                        "text": b["text"],
                        "bbox": b["bbox"],
                        "overflow_x": overflow_x,
                        "overflow_y": overflow_y,
                    })
        except Exception:
            pass
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a["bbox"] == b["bbox"]:
                continue
            # Same parent axes = within-panel; still flag cross-axes overlaps too
            area = _iou_area(a["bbox"], b["bbox"])
            small = _smaller_area(a["bbox"], b["bbox"])
            if area < MIN_OVERLAP_AREA or small <= 0:
                continue
            frac = area / small
            # Stricter than PDF pass: catch within-panel label collisions (≥20%)
            if frac < 0.20:
                continue
            if len(a["text"]) <= 1 or len(b["text"]) <= 1:
                continue
            # Skip tick-vs-tick only; label-vs-tick and label-vs-label always fail
            if _tickish(a["text"]) and _tickish(b["text"]):
                continue
            same_ax = False
            try:
                same_ax = a["artist"].axes is not None and a["artist"].axes is b["artist"].axes
            except Exception:
                pass
            # fig-level panel titles have axes=None but still sit inside a panel;
            # treat as within-panel when either bbox centre falls inside the
            # other's parent axes (catches title-vs-tick collisions).
            if not same_ax:
                for left, right in ((a, b), (b, a)):
                    ax = None
                    try:
                        ax = right["artist"].axes
                    except Exception:
                        ax = None
                    if ax is None:
                        continue
                    try:
                        ab = ax.get_window_extent(renderer)
                        cx = 0.5 * (left["bbox"][0] + left["bbox"][2])
                        cy = 0.5 * (left["bbox"][1] + left["bbox"][3])
                        if ab.x0 <= cx <= ab.x1 and ab.y0 <= cy <= ab.y1:
                            same_ax = True
                            break
                    except Exception:
                        pass
            fails.append({
                "kind": "fig_text_text",
                "within_panel": bool(same_ax),
                "a": a["text"],
                "b": b["text"],
                "area": area,
                "frac": frac,
            })
    return fails


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--pages-dir", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    pages = Path(args.pages_dir) if args.pages_dir else None
    res = audit_pdf(Path(args.pdf), pages)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2, default=str) + "\n")
    print(json.dumps({k: res[k] for k in ("ok", "n_failures", "n_pages", "failures")}, indent=2, default=str))
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
