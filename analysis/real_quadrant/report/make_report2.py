"""Build Report 2 (real data) PDFs from frozen M3 artifacts via Report 1 builders.

    python -m analysis.real_quadrant.report.make_report2

Writes (gitignored):
  outputs/real_quadrant/report2/data/          tidy tables
  outputs/real_quadrant/report2/figures/       Fig*.pdf/png
  outputs/real_quadrant/report2/report2_real_data.pdf
  outputs/real_quadrant/report2/report2_real_data_anon.pdf
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "outputs" / "real_quadrant" / "report2"


def run(cmd: list[str], **kw) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def build_one(*, anon: bool, out_root: Path) -> Path:
    tag = "anon" if anon else "named"
    data = out_root / f"data_{tag}"
    figs = out_root / f"figures_{tag}"
    data.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "-m", "analysis.real_quadrant.report.extract_tidy",
           "--out", str(data)]
    if anon:
        cmd.append("--anon")
    run(cmd, cwd=str(REPO), env={**os.environ})

    # render figures via figures_real CLI (domain=real path)
    run([
        sys.executable, str(REPO / "agents/quadrant_n5/figures/figures_real.py"),
        "--data", str(data), "--out", str(figs),
    ], cwd=str(REPO), env={**os.environ})

    # number audit
    claims = data / "number_claims.json"
    run([
        sys.executable, "-m", "analysis.real_quadrant.report.number_audit",
        "--data", str(data), "--write-claims", str(claims),
    ], cwd=str(REPO))

    pdf_name = "report2_real_data_anon.pdf" if anon else "report2_real_data.pdf"
    pdf = out_root / pdf_name
    run([
        sys.executable, str(REPO / "agents/quadrant_n5/figures/build_story_real.py"),
        "--data", str(data), "--figs", str(figs), "--out", str(pdf),
    ], cwd=str(REPO))
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--named-only", action="store_true")
    ap.add_argument("--anon-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    paths = []
    if not a.anon_only:
        paths.append(build_one(anon=False, out_root=a.out))
    if not a.named_only:
        paths.append(build_one(anon=True, out_root=a.out))
    print("DONE")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
