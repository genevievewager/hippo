"""Probe geometry and unit placement along hippocampal regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.config import (
    REGION_TO_CELL_TYPE,
    SITE_PITCH_UM,
    SimConfig,
    channels_for_segment,
)


@dataclass
class Unit:
    unit_id: int
    cell_type: str
    region: str
    layer: str
    channel: int
    depth_um: float
    place_center_cm: np.ndarray  # (2,)
    hd_pref_rad: float
    gain: float = 1.0


@dataclass
class AnatomyMap:
    units: list[Unit]
    region_table: list[dict]
    channel_to_region: dict[int, str]


def build_anatomy(config: SimConfig, rng: np.random.Generator) -> AnatomyMap:
    """Assign units to regions/channels based on segment density."""
    units: list[Unit] = []
    region_table: list[dict] = []
    channel_to_region: dict[int, str] = {}
    unit_id = 0

    for seg in config.region_segments:
        channels = channels_for_segment(seg["z_start"], seg["z_end"])
        n_channels = len(channels)
        n_units = max(1, int(seg["density"] * n_channels))

        region_key = (seg["region"], seg["layer"])
        cell_type = REGION_TO_CELL_TYPE[region_key]

        ch_start = int(channels[0]) + 1 if len(channels) else 0
        ch_end = int(channels[-1]) + 1 if len(channels) else 0

        region_table.append({
            "region": seg["region"],
            "layer": seg["layer"],
            "depth_start_um": seg["z_start"],
            "depth_end_um": seg["z_end"],
            "channels": f"{ch_start}-{ch_end}",
            "n_channels": n_channels,
            "n_units": n_units,
            "cell_types": cell_type,
        })

        for ch in channels:
            channel_to_region[int(ch)] = f"{seg['region']}/{seg['layer']}"

        assigned_channels = rng.choice(channels, size=n_units, replace=True)
        for ch in assigned_channels:
            depth = float(ch) * SITE_PITCH_UM
            center = rng.uniform(10, config.arena_size_cm - 10, size=2)
            hd_pref = rng.uniform(-np.pi, np.pi)
            units.append(Unit(
                unit_id=unit_id,
                cell_type=cell_type,
                region=seg["region"],
                layer=seg["layer"],
                channel=int(ch),
                depth_um=depth,
                place_center_cm=center,
                hd_pref_rad=hd_pref,
                gain=rng.uniform(0.8, 1.2),
            ))
            unit_id += 1

    return AnatomyMap(units=units, region_table=region_table, channel_to_region=channel_to_region)
