"""Q3: per-animal eligibility + path-containment QC (Report 2 semantics).

Exclusions (session_select / M2): min_units, no_room_A, empty_retained,
min_valid_frac, path extent (ExtentError), ratemap_stability — EXCLUDE.
Path containment ≥ 0.95: FLAG only (Report 2 extract_tidy); sessions stay
eligible. Failures that stay in use a path-based reachable-bin mask for
spatial metrics.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from matplotlib.path import Path as MplPath

from analysis.real_quadrant.adapter import (
    PATH_CONTAINMENT_MIN,
    _room_polygon_global,
    path_polygon_containment,
)
from analysis.real_quadrant.session_select import (
    _passes_room_a_exclusions,
    animal_id_from_session,
    list_2room_sessions,
)
from agents.data_quality.io.paths import DATA_QUALITY_ROOT
from realtime.quadrant_n5_run import segment_retained_mask

OUT = DATA_QUALITY_ROOT / "reports" / "report4a_v3"
MANIFEST = DATA_QUALITY_ROOT / "matched" / "MANIFEST.json"


def _path_containment_session(session_dir: Path, room: str = "A") -> dict[str, Any]:
    with open(session_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    rooms = cfg["preprocessing"]["map_rooms"]["rooms"]
    if room not in rooms:
        return dict(
            session=session_dir.name, room=room,
            path_containment_frac=float("nan"),
            path_containment_min=PATH_CONTAINMENT_MIN,
            pass_qc=False, reason="no_room",
        )
    t0, t1 = [float(x) for x in rooms[room]["range"]]
    pos = pd.read_csv(
        session_dir / "postions_dataset.csv",
        usecols=["timestamp", "X", "Y", "valid"],
    )
    seg = (pos.timestamp >= t0) & (pos.timestamp < t1)
    g = pos.loc[seg]
    times = g.timestamp.to_numpy(dtype=float)
    retained = segment_retained_mask(times, t0, t1)
    vv = g.valid.to_numpy()
    if vv.dtype != bool:
        vv = pd.to_numeric(g.valid, errors="coerce").fillna(0).to_numpy() != 0
    x = g.X.to_numpy(dtype=float)
    y = g.Y.to_numpy(dtype=float)
    m = retained & vv & np.isfinite(x) & np.isfinite(y)
    poly = _room_polygon_global(cfg, room)
    frac = path_polygon_containment(x[m], y[m], poly) if m.any() else float("nan")
    return dict(
        session=session_dir.name,
        room=room,
        path_containment_frac=float(frac),
        path_containment_min=float(PATH_CONTAINMENT_MIN),
        pass_qc=bool(np.isfinite(frac) and frac >= PATH_CONTAINMENT_MIN),
        n_path=int(m.sum()),
        n_poly_verts=int(len(poly) - 1),
        animal=animal_id_from_session(session_dir.name),
    )


def main() -> int:
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    man = json.loads(MANIFEST.read_text())
    matched = {s["session"] for s in man["sessions"]}
    sessions = list_2room_sessions(root)

    by_animal: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "total": 0, "eligible": 0, "excluded": 0,
        "exclude_reasons": Counter(),
        "eligible_sessions": [],
        "path_fail_eligible": [],
    })
    eligible_all: set[str] = set()
    rows = []

    for name in sessions:
        animal = animal_id_from_session(name)
        by_animal[animal]["total"] += 1
        ok, n_u, reason = _passes_room_a_exclusions(root / name)
        if ok and n_u is not None:
            by_animal[animal]["eligible"] += 1
            by_animal[animal]["eligible_sessions"].append(name)
            eligible_all.add(name)
            pc = _path_containment_session(root / name)
            pc["n_units"] = int(n_u)
            pc["in_matched"] = name in matched
            rows.append(pc)
            if not pc["pass_qc"]:
                by_animal[animal]["path_fail_eligible"].append(name)
        else:
            by_animal[animal]["excluded"] += 1
            by_animal[animal]["exclude_reasons"][reason or "unknown"] += 1

    added = sorted(eligible_all - matched)
    # Matched sessions that fail hard exclusions now (would drop out)
    drop_hard = []
    path_fail_matched = []
    for s in sorted(matched):
        ok, n_u, reason = _passes_room_a_exclusions(root / s)
        if not (ok and n_u is not None):
            drop_hard.append({"session": s, "reason": reason or "unknown"})
            continue
        pc = _path_containment_session(root / s)
        if not pc["pass_qc"]:
            path_fail_matched.append(pc)

    OUT.mkdir(parents=True, exist_ok=True)
    table = {
        "path_containment_min": PATH_CONTAINMENT_MIN,
        "semantics": {
            "hard_exclude": [
                "min_units", "no_room_A", "empty_retained", "min_valid_frac",
                "ExtentError", "ratemap_stability",
            ],
            "path_containment": "flag_only_keep_in_scores_with_path_mask",
        },
        "n_2rooms": len(sessions),
        "n_eligible": len(eligible_all),
        "n_matched": len(matched),
        "added_vs_matched": added,
        "drop_hard_vs_matched": drop_hard,
        "path_fail_matched": path_fail_matched,
        "per_animal": {
            a: {
                "total": v["total"],
                "eligible": v["eligible"],
                "excluded": v["excluded"],
                "exclude_reasons": dict(v["exclude_reasons"]),
                "n_path_fail_eligible": len(v["path_fail_eligible"]),
                "path_fail_eligible": v["path_fail_eligible"],
            }
            for a, v in sorted(by_animal.items())
        },
        "containment_rows": rows,
    }
    (OUT / "q3_path_containment.json").write_text(json.dumps(table, indent=2, default=str))
    pd.DataFrame(rows).to_csv(OUT / "q3_path_containment.csv", index=False)

    print("=== Q3 per-animal ===")
    for a, v in sorted(by_animal.items()):
        print(
            f"  {a}: total={v['total']} eligible={v['eligible']} "
            f"excluded={v['excluded']} reasons={dict(v['exclude_reasons'])} "
            f"path_fail_elig={len(v['path_fail_eligible'])}"
        )
    print(f"added vs matched ({len(added)}): {added[:20]}{'...' if len(added)>20 else ''}")
    print(f"drop hard ({len(drop_hard)}): {drop_hard}")
    fail_txt = ", ".join(
        f"{r['session']}={r['path_containment_frac']:.3f}" for r in path_fail_matched
    )
    print(f"path_fail matched ({len(path_fail_matched)}): {fail_txt}")
    if added:
        print("STOP: new eligible sessions not in matched MANIFEST — need approval to generate.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
