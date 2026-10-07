"""Fail the build if tidy contrasts disagree with animal-mean errors.

Every story number is produced by ``build_story_real.numbers`` from the same
tables; this audit checks internal consistency of those tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from agents.quadrant_n5.figures.build_story_real import numbers


def _round_eq(a: float, b: float, tol: float = 1e-3) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(float(a) - float(b)) <= tol


def audit(data_dir: Path) -> list[str]:
    n = numbers(str(data_dir))
    failures: list[str] = []
    if n["n_ani"] < 1:
        failures.append("n_animals < 1")
    for key, block in n["contrasts"].items():
        a, sep, b = key.partition(" - ")
        if not sep or a not in n["mean_rn"] or b not in n["mean_rn"]:
            continue
        expected = n["mean_rn"][a] - n["mean_rn"][b]
        if not _round_eq(expected, block["mean"], tol=1e-3):
            failures.append(
                f"contrast {key}: table mean {block['mean']:.4f} != "
                f"animal-mean delta {expected:.4f}"
            )
        if block["n"] != n["n_ani"] and block["n"] < n["n_ani"] - 1:
            # allow missing animal for rare method failures
            failures.append(
                f"contrast {key}: n={block['n']} animals (expected ~{n['n_ani']})"
            )
    # Ridge means finite for core methods
    for rep in ("raw", "raw_smooth", "pca", "lds"):
        if rep not in n["mean_rn"] or not np.isfinite(n["mean_rn"][rep]):
            failures.append(f"missing ridge_norm mean for {rep}")
    return failures


def write_claims(data_dir: Path, out: Path) -> Path:
    n = numbers(str(data_dir))
    claims = {
        "n_ani": n["n_ani"],
        "n_sess": n["n_sess"],
        "floor_lo": round(n["floor_lo"], 3),
        "floor_hi": round(n["floor_hi"], 3),
        "contrasts": {
            k: {"mean": round(v["mean"], 4), "k": v["k"], "n": v["n"]}
            for k, v in n["contrasts"].items()
        },
        "mean_rn": {k: round(v, 4) for k, v in n["mean_rn"].items()},
        "mean_cm": {k: round(v, 2) for k, v in n["mean_cm"].items()},
    }
    out.write_text(json.dumps(claims, indent=2) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--write-claims", type=Path, default=None)
    a = ap.parse_args()
    if a.write_claims:
        write_claims(a.data, a.write_claims)
        print("wrote", a.write_claims)
    fails = audit(a.data)
    if fails:
        print("NUMBER AUDIT FAILED:", file=sys.stderr)
        for f in fails:
            print(" -", f, file=sys.stderr)
        sys.exit(1)
    print("number audit OK")
