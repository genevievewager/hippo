"""Fail the build if tidy contrasts disagree with frozen grid20 / animal means.

Especially: dm_smooth - pca_smooth must match report2_contrasts.json
primary_contrasts_grid20 (matched d<=20 arms), not the mismatched final value.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from agents.quadrant_n5.figures.build_story_real import numbers

REPO = Path(__file__).resolve().parents[3]
R2_JSON = REPO / "outputs" / "real_quadrant" / "m3" / "report2_contrasts.json"


def _round_eq(a: float, b: float, tol: float = 1e-3) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(float(a) - float(b)) <= tol


def audit(data_dir: Path) -> list[str]:
    n = numbers(str(data_dir))
    failures: list[str] = []
    if n["n_ani"] < 1:
        failures.append("n_animals < 1")

    # Internal: contrast mean == animal-mean delta for same grid_mode arms
    # (already baked into data_contrasts.csv by extract).
    for key, block in n["contrasts"].items():
        if block["n"] < n["n_ani"] - 1:
            failures.append(
                f"contrast {key}: n={block['n']} animals (expected ~{n['n_ani']})"
            )

    # Grid20 gold: dm_smooth - pca_smooth must match frozen report2_contrasts
    if R2_JSON.is_file():
        r2 = json.loads(R2_JSON.read_text())
        gold = None
        for c in r2.get("primary_contrasts_grid20") or []:
            if c.get("contrast") == "dm_smooth - pca_smooth":
                gold = c
                break
        got = n["contrasts"].get("dm_smooth - pca_smooth")
        if gold is None:
            failures.append("frozen report2_contrasts.json missing dm_smooth - pca_smooth grid20")
        elif got is None:
            failures.append("tidy contrasts missing dm_smooth - pca_smooth")
        else:
            if not _round_eq(got["mean"], float(gold["mean"]), tol=1e-3):
                failures.append(
                    f"dm_smooth - pca_smooth mean {got['mean']:.4f} != "
                    f"grid20 gold {gold['mean']:.4f} "
                    f"(mismatched final would be ~0.052 — audit must use grid20)"
                )
            if int(got["k"]) != int(gold.get("n_a_better", -1)):
                # n_a_better = first term lower = delta < 0
                # our k = neg(delta) = first lower
                if int(got["k"]) != int(gold.get("n_a_better", got["k"])):
                    failures.append(
                        f"dm_smooth - pca_smooth k={got['k']} != "
                        f"grid20 n_a_better={gold.get('n_a_better')}"
                    )
            # Guard against the known wrong value
            if abs(got["mean"] - 0.0517) < 0.002 and abs(got["mean"] - float(gold["mean"])) > 0.01:
                failures.append(
                    "dm_smooth - pca_smooth still looks like the mismatched final-grid value"
                )

    for rep in ("raw", "raw_smooth", "pca", "lds"):
        if rep not in n["mean_rn"] or not np.isfinite(n["mean_rn"][rep]):
            failures.append(f"missing ridge_norm mean for {rep}")

    # Path-containment QC: record failures (do not fail the build for the
    # frozen cohort — list them). Require the table to exist.
    cont_path = data_dir / "data_path_containment.csv"
    if not cont_path.is_file():
        failures.append("missing data_path_containment.csv (path-in-polygon QC)")
    else:
        import pandas as pd
        C = pd.read_csv(cont_path)
        if "pass_qc" not in C.columns or "path_containment_frac" not in C.columns:
            failures.append("data_path_containment.csv missing pass_qc / path_containment_frac")
        else:
            bad = C[~C.pass_qc.astype(bool)]
            if len(bad):
                print(
                    "path containment QC failures (< "
                    f"{float(C.path_containment_min.iloc[0]):.2f}): "
                    + "; ".join(
                        f"{r.session}={float(r.path_containment_frac):.3f}"
                        for _, r in bad.iterrows()
                    ),
                    flush=True,
                )
    return failures


def write_claims(data_dir: Path, out: Path) -> Path:
    n = numbers(str(data_dir))
    claims = {
        "n_ani": n["n_ani"],
        "n_sess": n["n_sess"],
        "floor_lo": round(n["floor_lo"], 3),
        "floor_hi": round(n["floor_hi"], 3),
        "floor_mean": round(n["floor_mean"], 3),
        "mean_cm": {k: round(v, 1) for k, v in n["mean_cm"].items()},
        "mean_rn": {k: round(v, 3) for k, v in n["mean_rn"].items()},
        "contrasts": {
            k: {"mean": round(v["mean"], 3), "k": v["k"], "n": v["n"], "grid": v.get("grid")}
            for k, v in n["contrasts"].items()
        },
        "aggregation": n["aggregation"],
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
