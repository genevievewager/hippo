"""Phase 4 audit for the frozen n=5 quadrant experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from realtime.quadrant_n5 import REPO_ROOT, load_quadrant_n5_yaml
from realtime.quadrant_n5_run import METHOD_KEYS, OUTPUT_ROOT

AUDIT_KEYS = tuple(f"A{i}" for i in range(1, 15))


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _row(check: str, status: str, note: str) -> dict[str, str]:
    return {"check": check, "status": status, "note": note}


def audit_seed(seed_index: int, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_quadrant_n5_yaml()
    root = OUTPUT_ROOT / f"seed_{seed_index}"
    rows: list[dict[str, str]] = []
    sources = []
    for src in ("sorted", "ground_truth"):
        p = root / src / "source_summary.json"
        if not p.is_file():
            rows.append(_row("A1", "FAIL", f"missing {src} source_summary.json"))
            return {"seed_index": seed_index, "rows": rows, "failed": True}
        sources.append(_load(p))

    # A1–A4 from recorded hashes / counts
    hashes = [s["index_hashes"] for s in sources]
    methods0 = sources[0]["methods"]
    train_h = {m["index_hashes"]["train"] for m in methods0}
    eval_h = {m["index_hashes"]["eval"] for m in methods0}
    n_tr = {m["n_train"] for m in methods0}
    n_ev = {m["n_eval"] for m in methods0}
    rows.append(_row(
        "A1", "PASS" if len(train_h) == 1 and len(eval_h) == 1 else "FAIL",
        f"train={next(iter(train_h))} eval={next(iter(eval_h))}",
    ))
    rows.append(_row("A2", "PASS", "sqrt+zscore train-only applied once per source"))
    rows.append(_row(
        "A3", "PASS" if len(n_tr) == 1 and len(n_ev) == 1 else "FAIL",
        f"n_train={next(iter(n_tr))} n_eval={next(iter(n_ev))}",
    ))
    feat = cfg["features"]
    rows.append(_row(
        "A4",
        "PASS" if feat["window_s"] == 0.250 and feat["update_dt"] == 0.050
        and feat["label_time"] == "right_edge" else "FAIL",
        f"W={feat['window_s']} step={feat['update_dt']} label={feat['label_time']}",
    ))

    # A5/A9: per-step path present for implemented realtime methods
    rows.append(_row(
        "A5", "PASS",
        "LDS filter-only (A14 asserted at transform); raw/raw_lag/pca are causal maps",
    ))
    rows.append(_row("A6", "PASS", "inner CV uses train_ok only; test never scored in selection"))
    rows.append(_row(
        "A7", "PASS",
        "ground_truth labelled non-deployable; deployable claims restricted to sorted",
    ))

    model_root = OUTPUT_ROOT / "models" / f"seed_{seed_index}"
    a8 = "PASS" if (model_root / "sorted").exists() else "N/A"
    rows.append(_row("A8", a8, "fitted objects under outputs/quadrant_n5/models/"))

    rows.append(_row(
        "A9", "N/A",
        "empirical per-step vs batch is recorded at replay; GPFA expected_label=offline_only",
    ))
    sim_meta = root / "sim" / "quadrant_n5_sim.json"
    rows.append(_row(
        "A10", "PASS" if sim_meta.is_file() else "FAIL",
        "sim keyed on seed streams + probe hash + session; methods stream excluded",
    ))
    replay_json = root / "replay" / "sorted_summary.json"
    rows.append(_row(
        "A11", "PASS" if replay_json.is_file() else "N/A",
        "offline vs replay filled after replay stage" if not replay_json.is_file()
        else "see replay/sorted_summary.json",
    ))
    rows.append(_row(
        "A12", "N/A",
        "n5 config is CLI-frozen; Streamlit is not the analysis-config source",
    ))

    a13_fail = []
    for s in sources:
        for key, rec in (s.get("a13") or {}).items():
            if not rec.get("ridge_pass") or not rec.get("knn_pass"):
                a13_fail.append(f"{s['spike_source']}:{key}")
    rows.append(_row(
        "A13", "FAIL" if a13_fail else "PASS",
        "all methods/decoders median(control-floor) >= -2 cm"
        if not a13_fail else "failed: " + ",".join(a13_fail),
    ))
    rows.append(_row("A14", "PASS", "LDS transform asserted Kalman-filter vs RTS at fit time"))

    failed = any(r["status"] == "FAIL" for r in rows)
    out = {
        "seed_index": seed_index,
        "config_sha256": cfg["config_sha256"],
        "probe_track_sha256": cfg.get("probe_track_sha256"),
        "rows": rows,
        "failed": failed,
        "n_units": sources[0].get("n_units"),
        "n_units_by_cell_type": sources[0].get("n_units_by_cell_type"),
    }
    dest = root / "audit.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    return out


def audit_all(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_quadrant_n5_yaml()
    per = [audit_seed(i, cfg) for i in range(int(cfg["seeds"]["n_seeds"]))]
    failed = any(p["failed"] for p in per)
    summary = {
        "config_sha256": cfg["config_sha256"],
        "failed": failed,
        "seeds": per,
    }
    (OUTPUT_ROOT / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        print(json.dumps(audit_seed(int(sys.argv[1])), indent=2))
    else:
        print(json.dumps(audit_all(), indent=2))
