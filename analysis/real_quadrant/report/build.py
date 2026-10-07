"""Build M4 / Report 2 PDF from saved artifacts only.

    HIPPO_DATA_ROOT=... python -m analysis.real_quadrant.report.build
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

from analysis.real_quadrant.report import data as D
from analysis.real_quadrant.report import figures as F
from analysis.real_quadrant.report import pages as P
from analysis.real_quadrant.report.style import apply_rc

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "outputs" / "real_quadrant" / "report"


def _pngs_to_pdf(png_paths: list[Path], pdf_path: Path) -> None:
    """Stack PNGs into one multi-page PDF (each page = one figure)."""
    with PdfPages(pdf_path) as pdf:
        for png in png_paths:
            img = Image.open(png)
            # size in inches at 100 dpi display; PDF embeds raster
            w_in, h_in = img.size[0] / 100.0, img.size[1] / 100.0
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(min(w_in, 11), min(h_in, 14)))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(img)
            ax.axis("off")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)


def main() -> int:
    if not os.environ.get("HIPPO_DATA_ROOT"):
        raise SystemExit("HIPPO_DATA_ROOT must be set (needed for RD2/RD3 spike loads)")
    apply_rc()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    cohort = D.load_cohort()
    example = D.pick_example_session(cohort)
    print(f"example session: {example['session']} (n_units={example['n_units']})", flush=True)
    units = D.pick_example_units(example["session"])
    print(f"example units: {units['unit_ids']}", flush=True)

    paths: list[Path] = []
    print("story / methods / limitations …", flush=True)
    paths.append(P.page_story(OUT_DIR))
    paths.append(P.page_methods(OUT_DIR))
    paths.append(P.page_limitations(OUT_DIR))

    print("RD1 …", flush=True)
    paths.append(F.fig_rd1(OUT_DIR, example))
    print("RD2 …", flush=True)
    paths.append(F.fig_rd2(OUT_DIR, example, units))
    print("RD3 …", flush=True)
    paths.append(F.fig_rd3(OUT_DIR, example, units))
    print("RD4 …", flush=True)
    paths.append(F.fig_rd4(OUT_DIR, example))
    print("RD5 …", flush=True)
    paths.append(F.fig_rd5(OUT_DIR, example))
    print("RD6 …", flush=True)
    paths.append(F.fig_rd6(OUT_DIR, example))
    print("RD7 …", flush=True)
    paths.append(F.fig_rd7(OUT_DIR))

    pdf_path = OUT_DIR / "report2_real_data.pdf"
    print(f"assembling {pdf_path} …", flush=True)
    _pngs_to_pdf(paths, pdf_path)

    provenance = {
        "report": "Report 2 / M4 real-data-only",
        "pdf": str(pdf_path),
        "pngs": [str(p) for p in paths],
        "example_session": example,
        "example_units": {
            k: units[k] for k in (
                "rule", "session", "unit_ids", "peak_rates_hz", "n_candidates",
            )
        },
        "cohort_rule": cohort["rule"],
        "notes": [
            "RD8 moved to Report 1 revision",
            "RD9 deferred (would need new fitting)",
            "No model fitting in this build; train-only rate maps are descriptive",
            "GPFA offline smoother marked as reference; excluded from contrasts",
        ],
        "wall_s": time.perf_counter() - t0,
    }
    (OUT_DIR / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    print(f"done in {provenance['wall_s']:.1f}s → {pdf_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
