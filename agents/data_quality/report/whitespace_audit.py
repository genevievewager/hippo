#!/usr/bin/env python3
"""Blank-pixel fraction outside margins for dense-figure pages (≤ 15% target)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


MARGIN_FRAC = 14 / 297  # ~14 mm / A4 height — also applied to width via 14/210
MAX_BLANK = 0.15
# Layout blank after ink dilation; ≥85% of the content band must be occupied.


def blank_fraction(
    png: Path,
    margin_frac: float = MARGIN_FRAC,
    ink_thresh: int = 245,
    dilate_px: int = 56,
) -> dict:
    """Layout whitespace after dilating ink so plot interiors count as filled.

    Raw white-pixel fraction inside axes is ~70–85% for any scatter/line plot;
    that is not the layout gap the ≤15% rule targets. Dilate ink by ~dilate_px
    so each panel's rectangle is treated as occupied, then measure remaining
    blank in the content band.
    """
    from PIL import Image, ImageFilter

    im = Image.open(png).convert("L")
    arr = np.asarray(im)
    h, w = arr.shape
    my = max(1, int(round(h * margin_frac)))
    mx = max(1, int(round(w * (14 / 210))))
    content = arr[my: h - my, mx: w - mx]
    if content.size == 0:
        return {"ok": False, "blank_frac": 1.0, "error": "empty content"}
    raw_blank = float((content >= ink_thresh).mean())
    # Binary ink mask → dilate so panel bodies fill
    ink = (content < ink_thresh).astype(np.uint8) * 255
    ink_im = Image.fromarray(ink, mode="L")
    # Iterative MaxFilter (PIL size capped at 49) to approximate large dilation
    remaining = dilate_px
    while remaining > 0:
        step = min(remaining, 24)
        k = step * 2 + 1
        ink_im = ink_im.filter(ImageFilter.MaxFilter(size=k))
        remaining -= step
    ink_d = np.asarray(ink_im)
    layout_blank = float((ink_d == 0).mean())
    return {
        "file": png.name,
        "blank_frac": layout_blank,
        "raw_blank_frac": raw_blank,
        "ink_frac": 1.0 - layout_blank,
        "ok": layout_blank <= MAX_BLANK,
        "max_blank": MAX_BLANK,
        "dilate_px": dilate_px,
        "content_shape": list(content.shape),
    }


def audit_pngs(paths: list[Path], max_blank: float = MAX_BLANK) -> dict:
    reports = []
    failures = []
    for p in paths:
        if not p.is_file():
            failures.append({"file": str(p), "error": "missing"})
            continue
        r = blank_fraction(p)
        r["max_blank"] = max_blank
        r["ok"] = r["blank_frac"] <= max_blank
        reports.append(r)
        if not r["ok"]:
            failures.append(r)
    return {
        "ok": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
        "pages": reports,
    }


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("pngs", nargs="+")
    ap.add_argument("--write", default="")
    args = ap.parse_args()
    res = audit_pngs([Path(p) for p in args.pngs])
    print(json.dumps(res, indent=2))
    if args.write:
        Path(args.write).write_text(json.dumps(res, indent=2) + "\n")
    raise SystemExit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
