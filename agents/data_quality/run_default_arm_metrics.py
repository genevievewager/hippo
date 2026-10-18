#!/usr/bin/env python3
"""Compute unit metrics on default sim arm (copied) + one lab segment."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.real_quadrant.adapter import _parse_spike_times
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.chunks import chunk_bounds
from agents.data_quality.matching.regions import (
    add_mapped_lab_region,
    stratify_lab_units,
)
from agents.data_quality.metrics.guards import seconds_guard
from agents.data_quality.metrics.units import unit_metric_rows


def _animal_from_session(name: str) -> str:
    return name.split("_")[0]


def main() -> None:
    out = assert_dq_path(DATA_QUALITY_ROOT / "metrics_rows" / "default_arm_units.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # --- sim default arm (seed_0, all units, region-stratified labels) ---
    sim_dir = DATA_QUALITY_ROOT / "sim_inputs" / "seed_0" / "sim"
    units = add_mapped_lab_region(pd.read_csv(sim_dir / "units.csv"))
    gt = pd.read_csv(sim_dir / "spikes_ground_truth.csv")
    sorted_sp = pd.read_csv(sim_dir / "spikes_sorted.csv")
    for arm, spdf in (("sim_gt", gt), ("sim_sorted", sorted_sp)):
        spike_times = [
            spdf.loc[spdf["unit_id"] == uid, "spike_time_s"].to_numpy(float)
            for uid in units["unit_id"].to_numpy()
        ]
        seconds_guard(spike_times, name=arm)
        # 600 s → 2 chunks of 300
        for ci, (t0, t1) in enumerate([(0.0, 300.0), (300.0, 600.0)]):
            rows.extend(
                unit_metric_rows(
                    spike_times,
                    dataset=arm,
                    animal_or_seed="seed_0",
                    session="seed_0",
                    chunk=ci,
                    region_set="primary",
                    t0=t0,
                    t1=t1,
                    regions=units["lab_region"].astype(str).tolist(),
                )
            )

    # --- one lab segment via wrapper ---
    session = os.environ.get("DQ_PILOT_SESSION", "1150_27102024_2rooms")
    bundle = build_segment_bundle_dq(session, "A")
    # Rebuild per-unit spike times from clusters (bundle has counts only)
    root = Path(os.environ["HIPPO_DATA_ROOT"]) / session
    meta = pd.read_csv(root / "clusters_dataset.csv", usecols=lambda c: c in ("cell", "Region", "BClabel", "timestamp"))
    meta = meta[meta["BClabel"].astype(str) != "NON-SOMA"].reset_index(drop=True)
    # align to bundle unit_ids
    id_to_row = {int(c): i for i, c in enumerate(meta["cell"].to_numpy())}
    spike_times = []
    regions = []
    for uid in bundle["unit_ids"]:
        i = id_to_row[int(uid)]
        spike_times.append(_parse_spike_times(meta.loc[i, "timestamp"]))
        regions.append(str(meta.loc[i, "Region"]))
    seconds_guard(spike_times, name="lab")
    t0 = float(bundle["segment_t0"])
    t1 = float(bundle["segment_t1"])
    bounds = chunk_bounds(t0, t1)
    # primary mask
    mask = stratify_lab_units(regions, set_name="primary")
    st_p = [spike_times[i] for i in np.flatnonzero(mask)]
    reg_p = [regions[i] for i in np.flatnonzero(mask)]
    for ci, (lo, hi) in enumerate(bounds):
        rows.extend(
            unit_metric_rows(
                st_p,
                dataset="lab",
                animal_or_seed=_animal_from_session(session),
                session=session,
                chunk=ci,
                region_set="primary",
                t0=lo,
                t1=hi,
                regions=reg_p,
            )
        )

    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({"n_rows": len(rows), "path": str(out), "n_lab_chunks": len(bounds), "n_primary_units": int(mask.sum())}))


if __name__ == "__main__":
    main()
