#!/usr/bin/env python3
"""Check every numeric claim in findings/todo text against tidy tables."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NUM = re.compile(r"(?<![A-Za-z_])(-?\d+\.\d+|\d+)(?![A-Za-z_])")


def _close(a: float, b: float, tol: float = 0.015) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    if abs(b) < 1e-9:
        return abs(a - b) <= tol
    return abs(a - b) <= tol * max(1.0, abs(b)) or abs(a - b) <= 0.02


def audit(data: Path, claims_path: Path | None = None) -> dict:
    find = pd.read_csv(data / "data_findings.csv")
    disc = pd.read_csv(data / "data_discriminator.csv")
    meta = json.loads((data / "data_meta.json").read_text())
    div = pd.read_csv(data / "data_divergence.csv")

    claims = {
        "n_lab_sessions": int(meta["n_lab_sessions"]),
        "n_animals": int(meta["n_animals"]),
        "n_matched": int(meta["n_matched_sessions"]),
        "n_default": int(meta["n_default_seeds"]),
        "thigmotaxis": float(meta["thigmotaxis"]),
        "censor_isi_ms": float(meta["censor_isi_ms"]),
        "top_matched": [],
        "discriminator": [],
    }
    for _, r in find[find.arm == "matched"].head(10).iterrows():
        claims["top_matched"].append(
            {
                "metric": r["metric"],
                "R": float(r["R"]),
                "R_lo90": float(r["R_lo90"]),
                "R_hi90": float(r["R_hi90"]),
            }
        )
    for _, r in disc.iterrows():
        claims["discriminator"].append(
            {
                "arm": r["arm"],
                "auc_logit": float(r["auc_logit"]),
                "auc_group_shuffle_null": float(r["auc_group_shuffle_null"])
                if np.isfinite(r["auc_group_shuffle_null"])
                else None,
                "auc_group_shuffle_null_stratified": float(r["auc_group_shuffle_null_stratified"])
                if np.isfinite(r["auc_group_shuffle_null_stratified"])
                else None,
            }
        )

    failures = []
    # Cross-check findings vs divergence table
    for item in claims["top_matched"]:
        hit = div[(div.arm == "matched") & (div.metric == item["metric"])]
        if hit.empty:
            failures.append(f"missing divergence row for {item['metric']}")
            continue
        row = hit.iloc[0]
        for k in ("R", "R_lo90", "R_hi90"):
            if not _close(item[k], float(row[k])):
                failures.append(f"{item['metric']} {k}: claim {item[k]} != tidy {row[k]}")

    # jitter excluded
    jit = div[(div.arm == "matched") & (div.metric_base == "tracking_jitter_rest_cm")]
    if not jit.empty and bool(jit.iloc[0]["in_ranking"]):
        failures.append("tracking_jitter_rest_cm still in_ranking")

    # thigmotaxis
    if abs(claims["thigmotaxis"] - 0.8) > 1e-9:
        failures.append(f"thigmotaxis {claims['thigmotaxis']} != 0.8")

    if claims_path is not None:
        claims_path.write_text(json.dumps({"claims": claims, "failures": failures}, indent=2) + "\n")

    return {"ok": len(failures) == 0, "n_failures": len(failures), "failures": failures, "claims": claims}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--write-claims", default="")
    args = ap.parse_args()
    out = Path(args.write_claims) if args.write_claims else Path(args.data) / "number_claims.json"
    res = audit(Path(args.data), out)
    print(json.dumps({k: res[k] for k in ("ok", "n_failures", "failures")}, indent=2))
    if not res["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
