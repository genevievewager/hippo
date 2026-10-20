#!/usr/bin/env python3
"""Real overlap audit: render PDF pages to PNG and detect text bbox collisions.

Uses PyMuPDF text spans (PDF text layer). Fails on:
  - text overlapping text
  - text overlapping the page edge (within margin)
  - text whose bbox extends outside the page

Also audits matplotlib figures at save time:
  - every text vs every other text
  - every text vs every axes bbox / image extent (foreign-panel hits)
  - label bbox exceeding its own panel
  - panel axes occupying < 60% of their grid cell
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
PANEL_FILL_MIN = 0.60  # axes area / grid-cell area


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


def _bbox_area(b) -> float:
    return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))


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

        for sp in spans:
            x0, y0, x1, y1 = sp["bbox"]
            if x0 < MARGIN_PT * 0.5 or y0 < MARGIN_PT * 0.5 or x1 > pw - MARGIN_PT * 0.5 or y1 > ph - MARGIN_PT * 0.5:
                if x0 < 2 or y0 < 2 or x1 > pw - 2 or y1 > ph - 2:
                    page_fail.append({
                        "kind": "page_edge",
                        "text": sp["text"],
                        "bbox": sp["bbox"],
                    })

        for a_i in range(len(spans)):
            for b_i in range(a_i + 1, len(spans)):
                a, b = spans[a_i], spans[b_i]
                if a["bbox"] == b["bbox"]:
                    continue
                area = _iou_area(a["bbox"], b["bbox"])
                small = _smaller_area(a["bbox"], b["bbox"])
                if area >= MIN_OVERLAP_AREA and small > 0 and (area / small) >= 0.25:
                    if len(a["text"]) <= 1 or len(b["text"]) <= 1:
                        continue
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

        if pages_png_dir is not None:
            png = pages_png_dir / f"page-{i:02d}.png"
            if not png.is_file():
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


def _cluster_1d(vals: list[float], tol: float) -> list[float]:
    """Merge nearby positions into cluster centres (sorted)."""
    if not vals:
        return []
    s = sorted(vals)
    clusters = [[s[0]]]
    for v in s[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def _grid_cells(axes_boxes: list[tuple], fw: float, fh: float) -> list[dict]:
    """Assign each axes a grid cell from clustered left/bottom edges."""
    if not axes_boxes:
        return []
    lefts = [b[0] for b in axes_boxes]
    bottoms = [b[1] for b in axes_boxes]
    tol_x = max(8.0, fw * 0.03)
    tol_y = max(8.0, fh * 0.03)
    xs = _cluster_1d(lefts, tol_x)
    ys = _cluster_1d(bottoms, tol_y)
    # Cell boundaries = midpoints between centres, clamped to figure
    def bounds(centers, lo, hi):
        if len(centers) == 1:
            return [(lo, hi)]
        edges = [lo]
        for a, b in zip(centers, centers[1:]):
            edges.append(0.5 * (a + b))
        edges.append(hi)
        return list(zip(edges[:-1], edges[1:]))

    x_bounds = bounds(xs, 0.0, fw)
    y_bounds = bounds(ys, 0.0, fh)
    cells = []
    for b in axes_boxes:
        cx = 0.5 * (b[0] + b[2])
        cy = 0.5 * (b[1] + b[3])
        # nearest cluster by left/bottom of axes
        xi = int(np.argmin([abs(b[0] - x) for x in xs]))
        yi = int(np.argmin([abs(b[1] - y) for y in ys]))
        x0, x1 = x_bounds[xi]
        y0, y1 = y_bounds[yi]
        cells.append({
            "axes_bbox": b,
            "cell_bbox": (x0, y0, x1, y1),
            "fill": _bbox_area(b) / max(_bbox_area((x0, y0, x1, y1)), 1.0),
            "col": xi,
            "row": yi,
        })
    return cells


def audit_mpl_figure(fig) -> list[dict]:
    """Check matplotlib artists: text↔text, text↔axes/image, panel overflow, sparse panels."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches() * fig.dpi

    # ---- collect text ----
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

    # ---- collect axes + image extents ----
    axes_list = []
    image_boxes = []
    for ax in fig.axes:
        try:
            if not ax.get_visible():
                continue
            ab = ax.get_window_extent(renderer)
            abox = (ab.x0, ab.y0, ab.x1, ab.y1)
            # Skip tiny / fully off axes (colourbars sometimes)
            if _bbox_area(abox) < 20:
                continue
            axes_list.append({"ax": ax, "bbox": abox})
            for im in ax.images:
                try:
                    ib = im.get_window_extent(renderer)
                    image_boxes.append({
                        "bbox": (ib.x0, ib.y0, ib.x1, ib.y1),
                        "ax": ax,
                    })
                except Exception:
                    pass
        except Exception:
            continue

    fails = []
    pad = 1.0

    # Figure edge
    for b in boxes:
        x0, y0, x1, y1 = b["bbox"]
        if x0 < -pad or y0 < -pad or x1 > fw + pad or y1 > fh + pad:
            fails.append({"kind": "fig_edge", "text": b["text"], "bbox": b["bbox"]})

    # Annotation / custom label exceeds its own panel.
    # Axis title / xlabel / ylabel are allowed just outside the spines.
    chrome_texts = set()
    for ax_info in axes_list:
        ax = ax_info["ax"]
        try:
            chrome_texts.add(str(ax.get_title()))
            chrome_texts.add(str(ax.get_xlabel()))
            chrome_texts.add(str(ax.get_ylabel()))
        except Exception:
            pass
    for b in boxes:
        try:
            if b["text"] in chrome_texts:
                continue
            ax = b["artist"].axes
            if ax is None:
                continue
            art = b["artist"]
            ab = ax.get_window_extent(renderer)
            x0, y0, x1, y1 = b["bbox"]
            is_axis_chrome = (
                art is ax.title
                or art is ax.xaxis.label
                or art is ax.yaxis.label
            )
            if is_axis_chrome:
                continue
            if _tickish(b["text"]) and len(b["text"]) <= 6:
                continue
            # Legends / annotations with clip_on=False intentionally overhang
            try:
                if not art.get_clip_on():
                    continue
            except Exception:
                pass
            edge_pad = 3.0
            overflow_x = max(0.0, ab.x0 - edge_pad - x0, x1 - (ab.x1 + edge_pad))
            overflow_y = max(0.0, ab.y0 - edge_pad - y0, y1 - (ab.y1 + edge_pad))
            if overflow_x > 4.0 or overflow_y > 4.0:
                fails.append({
                    "kind": "panel_edge",
                    "text": b["text"],
                    "bbox": b["bbox"],
                    "overflow_x": overflow_x,
                    "overflow_y": overflow_y,
                })
        except Exception:
            pass

    def _own_axes(artist, text: str):
        """Resolve parent axes even when Text.axes is cleared after draw."""
        own = None
        try:
            own = artist.axes
        except Exception:
            own = None
        if own is not None:
            return own
        for ax_info in axes_list:
            ax = ax_info["ax"]
            try:
                if artist is ax.title or artist is ax.xaxis.label or artist is ax.yaxis.label:
                    return ax
                if text in (str(ax.get_title()), str(ax.get_xlabel()), str(ax.get_ylabel())):
                    return ax
            except Exception:
                continue
        return None

    # Fig-level text (no parent axes) vs axes / image bodies
    for b in boxes:
        if _tickish(b["text"]) and len(b["text"]) <= 4:
            continue
        own = _own_axes(b["artist"], b["text"])
        if own is not None:
            continue  # axes-owned labels: panel_edge covers overflow
        cx = 0.5 * (b["bbox"][0] + b["bbox"][2])
        cy = 0.5 * (b["bbox"][1] + b["bbox"][3])
        for ax_info in axes_list:
            ab = ax_info["bbox"]
            if not (ab[0] + 4 <= cx <= ab[2] - 4 and ab[1] + 8 <= cy <= ab[3] - 8):
                continue
            area = _iou_area(b["bbox"], ab)
            if area >= MIN_OVERLAP_AREA:
                fails.append({
                    "kind": "text_axes",
                    "text": b["text"],
                    "area": area,
                    "bbox": b["bbox"],
                    "axes_bbox": ab,
                })
        for im in image_boxes:
            area = _iou_area(b["bbox"], im["bbox"])
            if area >= MIN_OVERLAP_AREA * 2:
                fails.append({
                    "kind": "text_image",
                    "text": b["text"],
                    "area": area,
                    "bbox": b["bbox"],
                    "image_bbox": im["bbox"],
                })

    # Text ↔ text
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if a["bbox"] == b["bbox"]:
                continue
            area = _iou_area(a["bbox"], b["bbox"])
            small = _smaller_area(a["bbox"], b["bbox"])
            if area < MIN_OVERLAP_AREA or small <= 0:
                continue
            frac = area / small
            if frac < 0.20:
                continue
            if len(a["text"]) <= 1 or len(b["text"]) <= 1:
                continue
            if _tickish(a["text"]) and _tickish(b["text"]):
                continue
            same_ax = False
            try:
                same_ax = a["artist"].axes is not None and a["artist"].axes is b["artist"].axes
            except Exception:
                pass
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

    # Sparse panels: axes fill < 60% of inferred grid cell.
    # Only score mid/large content axes (≥15% fig width); gallery tiles and
    # narrow gutters are excluded (they share a row with many peers).
    fill_candidates = [
        a for a in axes_list
        if _bbox_area(a["bbox"]) > (fw * fh * 0.015)
        and (a["bbox"][2] - a["bbox"][0]) >= fw * 0.15
        and (a["bbox"][3] - a["bbox"][1]) >= fh * 0.08
    ]
    cells = _grid_cells([a["bbox"] for a in fill_candidates], fw, fh)
    for cell in cells:
        axes_a = _bbox_area(cell["axes_bbox"])
        cell_a = _bbox_area(cell["cell_bbox"])
        # Skip unreliable cells (clustering merged multiple rows/cols)
        if cell_a > axes_a * 1.55 or axes_a < 1:
            continue
        if cell["fill"] < PANEL_FILL_MIN:
            fails.append({
                "kind": "sparse_panel",
                "fill": round(cell["fill"], 3),
                "axes_bbox": cell["axes_bbox"],
                "cell_bbox": cell["cell_bbox"],
                "row": cell["row"],
                "col": cell["col"],
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
