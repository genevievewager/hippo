#!/usr/bin/env python3
"""Pilot one matched sim session (duration/density/arena; no behaviour fit)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.derived_config import (
    build_matched_capture_yaml,
    lab_segment_match_spec,
    write_condition_keys,
)
from agents.data_quality.matching.regions import add_mapped_lab_region, map_sim_region
from hippo_sim.config import SimConfig
from hippo_sim.pipeline import apply_trajectory_to_config
from run_simulation import generate_dataset
from agents.data_quality.matching.yield_boost import boost_ratinabox_for_required


def main() -> None:
    data_root = Path(os.environ["HIPPO_DATA_ROOT"])
    session = os.environ.get("DQ_PILOT_SESSION", "1150_27102024_2rooms")
    room = "A"
    out = assert_dq_path(DATA_QUALITY_ROOT / "matched" / f"{session}_{room}")
    out.mkdir(parents=True, exist_ok=True)
    write_condition_keys()

    spec = lab_segment_match_spec(data_root / session, room)
    capture_path = build_matched_capture_yaml(spec, out / "cell_capture_matched.yaml")
    (out / "match_spec.json").write_text(json.dumps(spec, indent=2, default=str) + "\n")

    sim_dir = out / "sim"
    if sim_dir.exists():
        import shutil
        shutil.rmtree(sim_dir)
    sim_dir.mkdir(parents=True)

    cfg = SimConfig(
        output_dir=sim_dir,
        seed=20261005,
        session_duration_s=float(spec["session_duration_s"]),
        behavior_dt=0.05,
        arena_size_cm=float(spec["arena_size_cm"]),
        thigmotaxis=0.8,  # behaviour fit pending — keep Report 1 default
    )
    apply_trajectory_to_config(
        cfg,
        trajectory_config="configs/trajectories/lab_npx2_default.yaml",
        cell_capture_config=str(capture_path),
    )
    boost_report = boost_ratinabox_for_required(
        cfg, spec["required_counts"], safety=1.3,
    )

    def _run_gen():
        t0 = time.perf_counter()
        summary = generate_dataset(cfg)
        return summary, time.perf_counter() - t0

    summary, gen_s = _run_gen()

    units = pd.read_csv(sim_dir / "units.csv")
    # never use include_in_decoder
    units = add_mapped_lab_region(units, region_col="region")
    yield_counts = units["lab_region"].value_counts().to_dict()
    required = spec["required_counts"]
    regen = 0
    while regen < 2 and any(
        int(yield_counts.get(k, 0)) < int(required.get(k, 0)) for k in ("MEC", "SUB")
    ):
        regen += 1
        # escalate: boost again with higher safety; keep FINAL report
        boost_report = boost_ratinabox_for_required(
            cfg, required, safety=1.3 * (1.5 ** regen),
        )
        import shutil
        if sim_dir.exists():
            shutil.rmtree(sim_dir)
        sim_dir.mkdir(parents=True)
        cfg.output_dir = sim_dir
        summary, gen_extra = _run_gen()
        gen_s += gen_extra
        units = pd.read_csv(sim_dir / "units.csv")
        units = add_mapped_lab_region(units, region_col="region")
        yield_counts = units["lab_region"].value_counts().to_dict()

    # quick unit-metric cost on GT spikes for one 300 s chunk
    from agents.data_quality.metrics.units import unit_metric_rows
    from agents.data_quality.metrics.guards import seconds_guard

    gt = pd.read_csv(sim_dir / "spikes_ground_truth.csv")
    spike_times = [
        gt.loc[gt["unit_id"] == uid, "spike_time_s"].to_numpy(dtype=float)
        for uid in units["unit_id"].to_numpy()
    ]
    seconds_guard(spike_times, name="pilot_sim")
    t_m0 = time.perf_counter()
    rows = unit_metric_rows(
        spike_times,
        dataset="sim",
        animal_or_seed="pilot",
        session=session,
        chunk=0,
        region_set="primary",
        t0=0.0,
        t1=min(300.0, float(spec["session_duration_s"])),
        regions=units["lab_region"].astype(str).tolist(),
    )
    metric_s = time.perf_counter() - t_m0

    report = {
        "session": session,
        "room": room,
        "generation_wall_s": gen_s,
        "unit_metrics_one_chunk_wall_s": metric_s,
        "n_metric_rows": len(rows),
        "session_duration_s": spec["session_duration_s"],
        "arena_size_cm": spec["arena_size_cm"],
        "shape_confound": spec["shape_confound"],
        "required_counts": required,
        "expected_yield": spec["expected_yield"],
        "sim_yield": {k: int(yield_counts.get(k, 0)) for k in ("MEC", "SUB", "OTHER")},
        "yield_ok": {
            k: int(yield_counts.get(k, 0)) >= int(required.get(k, 0))
            for k in ("MEC", "SUB")
        },
        "summary_n_units": summary.get("n_units"),
        "density_scales": spec["density_scales"],
        "ratinabox_boost": boost_report,
        "n_regenerations": regen,
        "behavior_fit": "not_run",
    }
    (out / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
