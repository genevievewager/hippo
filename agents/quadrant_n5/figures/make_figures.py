"""
One-command figure build for the quadrant N=5 experiment (pipeline Phase 6).

    python agents/quadrant_n5/figures/make_figures.py \
        --results outputs/quadrant_n5 --out outputs/quadrant_n5/figures

Reads only result JSONs + sim/behavior.csv under --results. Writes:
    <out>/data/data_*.csv            tidy source data behind every panel
    <out>/Fig1..Fig6 .pdf/.png        vector PDF + 600-dpi PNG, 180 mm wide
    <out>/quadrant_n5_figure_set.pdf  story page + each figure with its legend
No network access; nothing outside --results is read.
"""
import argparse, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(*args):
    subprocess.run([sys.executable, *args], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="outputs/quadrant_n5")
    ap.add_argument("--out", default="outputs/quadrant_n5/figures")
    a = ap.parse_args()
    data = os.path.join(a.out, "data")
    run(os.path.join(HERE, "extract_json.py"), a.results, data)
    run(os.path.join(HERE, "figures.py"), "--data", data, "--out", a.out)
    run(os.path.join(HERE, "build_story.py"), "--data", data, "--figs", a.out,
        "--out", os.path.join(a.out, "quadrant_n5_figure_set.pdf"))
