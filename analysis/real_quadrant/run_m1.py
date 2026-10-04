"""M1: raw method on room A of one 2-room session via the analyze_source seam."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from analysis.real_quadrant.adapter import build_segment_bundle
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import analyze_source

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "real_quadrant" / "m1"


def _pick_2room_session(data_root: Path) -> str:
    sessions = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and "2rooms" in p.name
    )
    if not sessions:
        raise RuntimeError("no 2rooms sessions under HIPPO_DATA_ROOT")
    return sessions[0]


def main() -> int:
    data_root = Path(os.environ["HIPPO_DATA_ROOT"])
    session = _pick_2room_session(data_root)
    t0 = time.perf_counter()
    bundle = build_segment_bundle(session, room="A", data_root=data_root)
    cfg = dict(load_quadrant_n5_yaml())
    cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
    # Record trims in a hashable overlay note (frozen YAML hash stays the sim one;
    # real-data provenance is in bundle meta + segment block).
    streams = {
        "methods": 0,
        "data_seed": 0,
        "master_seed": 0,
        "seed_index": 0,
        "trajectory": 0,
        "neural": 0,
        "recording_noise": 0,
        "sorting_errors": 0,
    }
    out_dir = OUT_ROOT / "room_A" / "raw"
    summary = analyze_source(
        cfg,
        None,
        "real",
        streams,
        0,
        bundle=bundle,
        output_dir=out_dir,
        method_keys=("raw",),
    )
    wall = time.perf_counter() - t0
    raw = next(m for m in summary["methods"] if m["method"] == "raw")
    floor = summary["floor"]
    med = float(raw["ridge"]["median"])
    floor_med = float(floor["median"])
    report = {
        "n_units": int(summary["n_units"]),
        "n_train": int(summary["n_train"]),
        "n_eval": int(summary["n_eval"]),
        "n_retained": int(summary["segment"]["n_retained"]),
        "dropped_valid_fraction": float(
            summary["segment"]["dropped_valid_fraction"]
        ),
        "ridge_median_cm": med,
        "floor_median_cm": floor_med,
        "normalized_error": med / floor_med if floor_med > 0 else None,
        "integrity_match_fraction": float(
            bundle["meta"]["integrity_match_fraction"]
        ),
        "wall_time_s": wall,
        "method_elapsed_s": float(raw["elapsed_s"]),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "m1_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    # Print aggregates only (no session name).
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
