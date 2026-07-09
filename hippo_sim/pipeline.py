"""End-to-end simulation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hippo_sim.anatomy import build_anatomy
from hippo_sim.behavior import simulate_behavior
from hippo_sim.config import SimConfig
from hippo_sim.drift import DriftState
from hippo_sim.features import compute_global_features
from hippo_sim.rate_equations import integrate_rates
from hippo_sim.recording import build_unit_templates, simulate_recording
from hippo_sim.sorting import kilosort_like_sort
from hippo_sim.spikes import generate_spike_trains


def run_pipeline(config: SimConfig) -> dict:
    """Run full simulation and save outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    print("[1/7] Simulating behavior (RatInABox)...", flush=True)
    behavior = simulate_behavior(config)
    behavior_df = pd.DataFrame({
        "time_s": behavior.time_s,
        "x_cm": behavior.position_cm[:, 0],
        "y_cm": behavior.position_cm[:, 1],
        "speed_cm_s": behavior.speed_cm_s,
        "head_direction_rad": behavior.head_direction_rad,
    })
    behavior_df.to_csv(config.output_dir / "behavior.csv", index=False)

    print("[2/7] Building anatomy and unit assignments...", flush=True)
    anatomy = build_anatomy(config, rng)
    pd.DataFrame(anatomy.region_table).to_csv(config.output_dir / "anatomy_regions.csv", index=False)

    units_meta = [{
        "unit_id": u.unit_id,
        "cell_type": u.cell_type,
        "region": u.region,
        "layer": u.layer,
        "channel": u.channel + 1,
        "depth_um": u.depth_um,
        "place_x_cm": u.place_center_cm[0],
        "place_y_cm": u.place_center_cm[1],
        "hd_pref_rad": u.hd_pref_rad,
    } for u in anatomy.units]
    pd.DataFrame(units_meta).to_csv(config.output_dir / "units.csv", index=False)

    print(f"[3/7] Computing features and integrating rates ({len(anatomy.units)} units)...", flush=True)
    global_features = compute_global_features(behavior, config)
    drift_state = DriftState(anatomy.units, config, rng)
    rates = integrate_rates(anatomy.units, global_features, drift_state, config)

    print("[4/7] Generating ground-truth spike trains...", flush=True)
    spike_trains = generate_spike_trains(rates, config, rng)

    gt_rows = []
    for train in spike_trains:
        for t in train.spike_times_s:
            gt_rows.append({"unit_id": train.unit_id, "spike_time_s": t})
    pd.DataFrame(gt_rows).to_csv(config.output_dir / "spikes_ground_truth.csv", index=False)

    print("[5/7] Building templates and simulating Neuropixels recording...", flush=True)
    templates = build_unit_templates(anatomy, config, rng)
    events = simulate_recording(spike_trains, templates, config, rng)

    print("[6/7] Kilosort-like re-extraction...", flush=True)
    sorted_spikes = kilosort_like_sort(events, templates, spike_trains, config, rng)

    sorted_rows = [{
        "unit_id": s.unit_id,
        "spike_time_s": s.time_s,
        "channel": s.channel + 1,
        "confidence": s.confidence,
    } for s in sorted_spikes]
    pd.DataFrame(sorted_rows).to_csv(config.output_dir / "spikes_sorted.csv", index=False)

    print("[7/7] Saving summary...", flush=True)
    summary = {
        "n_units": len(anatomy.units),
        "n_ground_truth_spikes": len(gt_rows),
        "n_sorted_spikes": len(sorted_rows),
        "session_duration_s": config.session_duration_s,
        "n_channels": config.n_channels,
        "seed": config.seed,
    }
    with open(config.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
