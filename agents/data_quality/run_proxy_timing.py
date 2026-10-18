#!/usr/bin/env python3
"""Time one cheap-calibration proxy session; project total; hard-cap 3 h."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.derived_config import (
    build_matched_capture_yaml,
    lab_segment_match_spec,
)
from agents.data_quality.matching.session_discovery import eligibility_by_animal
from agents.data_quality.matching.yield_boost import boost_ratinabox_for_required
from hippo_sim.config import SimConfig
from hippo_sim.pipeline import apply_trajectory_to_config
from run_simulation import generate_dataset


def main() -> None:
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    elig = eligibility_by_animal(root)
    # one eligible session per animal (first listed)
    sessions = []
    for a in elig["animals"]:
        es = a.get("eligible_sessions") or []
        if not es:
            continue
        sessions.append(es[0]["session"] if isinstance(es[0], dict) else es[0])
    if not sessions:
        raise RuntimeError("no eligible sessions for proxy timing")
    session = sessions[0]
    duration = float(os.environ.get("DQ_PROXY_DURATION_S", "300"))
    n_units = int(os.environ.get("DQ_PROXY_N_UNITS", "50"))

    out = assert_dq_path(DATA_QUALITY_ROOT / "calibration" / "proxy_timing" / session)
    out.mkdir(parents=True, exist_ok=True)
    spec = lab_segment_match_spec(root / session, "A")
    # shrink required to ~50 units keeping proportions
    req = dict(spec["required_counts"])
    tot = max(1, int(req.get("MEC", 0)) + int(req.get("SUB", 0)))
    scale = n_units / tot
    req = {k: max(1, int(round(v * scale))) if int(v) > 0 else 0 for k, v in req.items()}
    spec = dict(spec)
    spec["required_counts"] = req
    spec["session_duration_s"] = duration
    capture = build_matched_capture_yaml(spec, out / "cell_capture_proxy.yaml")
    (out / "match_spec.json").write_text(json.dumps(spec, indent=2, default=str) + "\n")

    sim_dir = out / "sim"
    if sim_dir.exists():
        import shutil
        shutil.rmtree(sim_dir)
    sim_dir.mkdir()
    cfg = SimConfig(
        output_dir=sim_dir,
        seed=20261007,
        session_duration_s=duration,
        behavior_dt=0.05,
        arena_size_cm=float(spec["arena_size_cm"]),
        thigmotaxis=0.8,
    )
    apply_trajectory_to_config(
        cfg,
        trajectory_config="configs/trajectories/lab_npx2_default.yaml",
        cell_capture_config=str(capture),
    )
    boost_ratinabox_for_required(cfg, req, safety=1.3)
    t0 = time.perf_counter()
    summary = generate_dataset(cfg)
    wall = time.perf_counter() - t0

    # memory-based n_jobs
    mem_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            mem_kib = int(line.split()[1])
            break
    mem_gib = mem_kib / (1024 ** 2)
    n_jobs = int(os.environ.get("DQ_N_JOBS") or max(1, min(48, int((mem_gib - 100) / 5))))

    # budget from cheap yaml
    n_animals = len(sessions)
    morris, lhs, gp, val = 28, 60, 90, 18
    n_theta = morris + lhs + gp  # validation separate (full-size; not in proxy budget)
    # each θ → n_animals proxy gens
    n_proxy_gens = n_theta * n_animals
    proj_serial_s = n_proxy_gens * wall
    proj_wall_h = proj_serial_s / n_jobs / 3600.0
    # validation full-size estimate: use matched pilot ratio if available
    matched_est = DATA_QUALITY_ROOT / "_phase0" / "pilot_longest_summary.json"
    full_gen_s = wall * (1100 / duration)  # rough scale
    if matched_est.is_file():
        full_gen_s = float(json.loads(matched_est.read_text()).get("generation_wall_s") or full_gen_s)
    val_wall_h = (val * full_gen_s) / n_jobs / 3600.0
    total_h = proj_wall_h + val_wall_h

    report = {
        "session": session,
        "duration_s": duration,
        "required_counts": req,
        "n_units_summary": summary.get("n_units"),
        "proxy_gen_wall_s": wall,
        "n_animals": n_animals,
        "n_jobs": n_jobs,
        "mem_avail_gib": mem_gib,
        "budget_theta": {"morris": morris, "lhs": lhs, "gp": gp, "validation": val},
        "n_proxy_gens": n_proxy_gens,
        "projected_proxy_wall_h": proj_wall_h,
        "projected_validation_wall_h": val_wall_h,
        "projected_total_wall_h": total_h,
        "hard_cap_h": 3.0,
        "under_cap": bool(total_h <= 3.0),
    }
    path = assert_dq_path(DATA_QUALITY_ROOT / "calibration" / "proxy_timing.json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if total_h > 3.0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
