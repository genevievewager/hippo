#!/usr/bin/env python3
"""Extract a 1 s multi-unit raster excerpt (display asset; does not change metrics)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from agents.data_quality.io.paths import DATA_QUALITY_ROOT


def extract_rasters_1s(out: Path, session: str, n_units: int = 20, dur_s: float = 1.0) -> Path:
    from analysis.real_quadrant.adapter import _parse_spike_times
    from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
    from agents.data_quality.matching.regions import add_mapped_lab_region, stratify_lab_units

    root = Path(os.environ["HIPPO_DATA_ROOT"])
    bundle = build_segment_bundle_dq(session, "A", require_ratemap_stability=True)
    times = np.asarray(bundle["decode_times"], float)
    # start 120 s into segment (same convention as traj extract)
    t0 = float(times[0] + 120.0)
    t1 = t0 + dur_s

    meta = pd.read_csv(
        root / session / "clusters_dataset.csv",
        usecols=lambda c: c in ("cell", "Region", "BClabel", "timestamp"),
    )
    if "BClabel" in meta.columns:
        meta = meta[meta["BClabel"].astype(str) != "NON-SOMA"].reset_index(drop=True)
    id_to = {int(c): i for i, c in enumerate(meta["cell"].to_numpy())}
    spikes, regions = [], []
    for uid in bundle["unit_ids"]:
        i = id_to[int(uid)]
        spikes.append(_parse_spike_times(meta.loc[i, "timestamp"]))
        regions.append(str(meta.loc[i, "Region"]))
    mask = stratify_lab_units(regions, set_name="primary")
    spikes = [spikes[i] for i in np.flatnonzero(mask)]

    # pick n_units with most spikes in window
    counts = [int(((st >= t0) & (st < t1)).sum()) for st in spikes]
    order = np.argsort(counts)[::-1][:n_units]
    lab_rows = []
    for rank, ui in enumerate(order):
        st = spikes[int(ui)]
        in_win = st[(st >= t0) & (st < t1)] - t0
        lab_rows.append({"unit": int(rank), "t": in_win.tolist()})

    sim_dir = DATA_QUALITY_ROOT / "matched" / f"{session}_A" / "sim"
    units = add_mapped_lab_region(pd.read_csv(sim_dir / "units.csv"))
    units = units[units["lab_region"].isin(["MEC", "SUB"])]
    so = pd.read_csv(sim_dir / "spikes_sorted.csv")
    sspikes = [so.loc[so["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
    # align window to sim time origin similarly
    beh = pd.read_csv(sim_dir / "behavior.csv")
    tcol = "time_s" if "time_s" in beh.columns else beh.columns[0]
    stimes = beh[tcol].to_numpy(float)
    st0 = float(stimes[0] + 120.0)
    st1 = st0 + dur_s
    scounts = [int(((st >= st0) & (st < st1)).sum()) for st in sspikes]
    sorder = np.argsort(scounts)[::-1][:n_units]
    sim_rows = []
    for rank, ui in enumerate(sorder):
        st = sspikes[int(ui)]
        in_win = st[(st >= st0) & (st < st1)] - st0
        sim_rows.append({"unit": int(rank), "t": in_win.tolist()})

    payload = {
        "session": session,
        "dur_s": dur_s,
        "n_units": n_units,
        "refractory_s": 0.0015,
        "lab": {"t0": t0, "units": lab_rows},
        "sim_matched": {"t0": st0, "units": sim_rows},
        "note": "Display-only 1 s excerpt; metrics unchanged (v5 tidy).",
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / "data_rasters_1s.json"
    path.write_text(json.dumps(payload) + "\n")
    return path


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--session", default="1150_09102024_2rooms")
    args = ap.parse_args()
    p = extract_rasters_1s(Path(args.out), args.session)
    print(json.dumps({"wrote": str(p)}))
