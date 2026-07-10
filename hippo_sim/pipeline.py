"""End-to-end simulation pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hippo_sim.anatomy import AnatomyMap, build_anatomy
from hippo_sim.behavior import simulate_behavior
from hippo_sim.config import SimConfig
from hippo_sim.neural_backend import simulate_neural_activity
from hippo_sim.recording import build_unit_templates, simulate_recording
from hippo_sim.sorting import kilosort_like_sort
from hippo_sim.spikes import generate_spike_trains


def _build_region_table(config: SimConfig) -> list[dict]:
    from hippo_sim.anatomy import channels_for_segment, REGION_TO_CELL_TYPE

    table = []
    for seg in config.region_segments:
        channels = channels_for_segment(seg["z_start"], seg["z_end"])
        n_channels = len(channels)
        n_units = max(1, int(seg["density"] * n_channels))
        ch_start = int(channels[0]) + 1 if len(channels) else 0
        ch_end = int(channels[-1]) + 1 if len(channels) else 0
        cell_type = REGION_TO_CELL_TYPE[(seg["region"], seg["layer"])]
        table.append({
            "region": seg["region"],
            "layer": seg["layer"],
            "depth_start_um": seg["z_start"],
            "depth_end_um": seg["z_end"],
            "channels": f"{ch_start}-{ch_end}",
            "n_channels": n_channels,
            "n_units": n_units,
            "cell_types": cell_type,
        })
    return table


def _units_to_dataframe(units: list) -> pd.DataFrame:
    rows = []
    for u in units:
        row = {
            "unit_id": u.unit_id,
            "cell_type": u.cell_type,
            "region": u.region,
            "layer": u.layer,
            "channel": u.channel + 1,
            "depth_um": u.depth_um,
            "place_x_cm": u.place_center_cm[0],
            "place_y_cm": u.place_center_cm[1],
            "hd_pref_rad": u.hd_pref_rad,
            "rate_model": u.rate_model or f"custom_{u.cell_type}_rate_equation",
        }
        if u.ratinabox_class is not None:
            row["ratinabox_class"] = u.ratinabox_class
        rows.append(row)
    return pd.DataFrame(rows)


def _spikes_to_dataframe(
    spike_trains,
    units_df: pd.DataFrame,
) -> pd.DataFrame:
    unit_lookup = units_df.set_index("unit_id")
    rows = []
    for train in spike_trains:
        uid = train.unit_id
        meta = unit_lookup.loc[uid]
        for t in train.spike_times_s:
            row = {
                "unit_id": uid,
                "spike_time_s": t,
                "cell_type": meta["cell_type"],
                "region": meta["region"],
                "rate_model": meta.get("rate_model", meta["cell_type"]),
            }
            if "ratinabox_class" in meta.index:
                row["ratinabox_class"] = meta["ratinabox_class"]
            rows.append(row)
    return pd.DataFrame(rows)


def run_pipeline(config: SimConfig) -> dict:
    """Run full simulation and save outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)

    print("[1/7] Simulating behavior (RatInABox)...", flush=True)
    behavior_result = simulate_behavior(config)
    behavior = behavior_result.trace
    behavior_df = pd.DataFrame({
        "time_s": behavior.time_s,
        "x_cm": behavior.position_cm[:, 0],
        "y_cm": behavior.position_cm[:, 1],
        "speed_cm_s": behavior.speed_cm_s,
        "head_direction_rad": behavior.head_direction_rad,
        "distance_to_wall_cm": behavior.distance_to_wall_cm,
        "acceleration_cm_s2": behavior.acceleration_cm_s2,
    })
    behavior_df.to_csv(config.output_dir / "behavior.csv", index=False)

    print("[2/7] Building anatomy and unit assignments...", flush=True)
    region_table = _build_region_table(config)
    pd.DataFrame(region_table).to_csv(config.output_dir / "anatomy_regions.csv", index=False)

    anatomy: AnatomyMap | None = None
    if config.neural_backend == "custom_rate_equations":
        anatomy = build_anatomy(config, rng)

    print(f"[3/7] Generating neural activity ({config.neural_backend})...", flush=True)
    units, rates, neural_metadata = simulate_neural_activity(
        config=config,
        behavior=behavior,
        rng=rng,
        anatomy=anatomy,
        behavior_result=behavior_result,
    )

    np.save(config.output_dir / "rates.npy", rates)
    with open(config.output_dir / "neural_backend_metadata.json", "w") as f:
        json.dump(neural_metadata, f, indent=2)

    if config.neural_backend == "ratinabox_neurons":
        _save_ratinabox_group_outputs(config, units, rates, neural_metadata)

    units_df = _units_to_dataframe(units)
    units_df.to_csv(config.output_dir / "units.csv", index=False)

    anatomy = AnatomyMap(units=units, region_table=region_table, channel_to_region={})

    print("[4/7] Generating ground-truth spike trains...", flush=True)
    spike_trains = generate_spike_trains(rates, config, rng, time_axis=behavior.time_s)
    gt_df = _spikes_to_dataframe(spike_trains, units_df)
    gt_df.to_csv(config.output_dir / "spikes_ground_truth.csv", index=False)

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
        "neural_backend": config.neural_backend,
        "n_units": len(units),
        "n_behavior_steps": config.n_behavior_steps,
        "behavior_dt": config.behavior_dt,
        "rates_shape": list(rates.shape),
        "n_ground_truth_spikes": len(gt_df),
        "n_sorted_spikes": len(sorted_rows),
        "session_duration_s": config.session_duration_s,
        "n_channels": config.n_channels,
        "seed": config.seed,
        "arena_size_cm": config.arena_size_cm,
    }
    if "ratinabox_cell_groups" in neural_metadata:
        summary["ratinabox_cell_groups"] = neural_metadata["ratinabox_cell_groups"]

    with open(config.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _save_ratinabox_group_outputs(
    config: SimConfig,
    units: list,
    rates: np.ndarray,
    metadata: dict,
) -> None:
    """Save per-group RatInABox rate arrays and metadata."""
    groups = metadata.get("ratinabox_cell_groups", {})
    if not groups:
        return

    group_meta_rows = []
    offset = 0
    arrays = {}
    for rb_class, n_group in groups.items():
        sl = rates[offset:offset + n_group]
        arrays[rb_class] = sl
        sample_unit = units[offset]
        group_meta_rows.append({
            "group_name": rb_class,
            "ratinabox_class": rb_class,
            "cell_type": sample_unit.cell_type,
            "rate_model": sample_unit.rate_model,
            "n_units": n_group,
        })
        offset += n_group

    np.savez(config.output_dir / "ratinabox_rates_by_group.npz", **arrays)
    pd.DataFrame(group_meta_rows).to_csv(
        config.output_dir / "ratinabox_group_metadata.csv", index=False,
    )
