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
    rate_model: str = ""
    ratinabox_class: str | None = None


@dataclass
class AnatomyMap:
    units: list[Unit]
    region_table: list[dict]
    channel_to_region: dict[int, str]


CELL_TYPE_TO_PROBE = {
    "CA1_pyr": ("CA1", "pyramidal"),
    "CA1_int": ("CA1", "oriens"),
    "CA2_pyr": ("CA2", "pyramidal"),
    "CA3_pyr": ("CA3", "pyramidal"),
    "DG_granule": ("DG", "granule"),
    "MEC_grid": ("MEC", "layer2"),
    "MEC_hd": ("MEC", "layer3"),
    "MEC_speed": ("MEC", "layer2"),
    "Sub_bvc": ("Subiculum", "pyramidal"),
}


def assign_probe_channels(
    units: list[Unit],
    config: SimConfig,
    rng: np.random.Generator,
) -> list[Unit]:
    """Assign probe channel/depth to units using hippocampal region segments."""
    channel_pools: dict[str, np.ndarray] = {}
    for seg in config.region_segments:
        region_key = (seg["region"], seg["layer"])
        cell_type = REGION_TO_CELL_TYPE.get(region_key)
        if cell_type is None:
            continue
        channels = channels_for_segment(seg["z_start"], seg["z_end"])
        if len(channels):
            channel_pools.setdefault(cell_type, [])
            channel_pools[cell_type].extend(channels.tolist())

    for pool in channel_pools.values():
        pool.sort()

    assigned: list[Unit] = []
    for unit in units:
        pool = channel_pools.get(unit.cell_type)
        if not pool:
            region, layer = CELL_TYPE_TO_PROBE.get(unit.cell_type, ("CA1", "pyramidal"))
            for seg in config.region_segments:
                if seg["region"] == region and seg["layer"] == layer:
                    pool = channels_for_segment(seg["z_start"], seg["z_end"]).tolist()
                    break
            if not pool:
                for seg in config.region_segments:
                    if seg["region"] == region:
                        pool = channels_for_segment(seg["z_start"], seg["z_end"]).tolist()
                        break
        if not pool:
            pool = [0]

        ch = int(rng.choice(pool))
        region, layer = CELL_TYPE_TO_PROBE.get(unit.cell_type, ("CA1", "pyramidal"))
        assigned.append(Unit(
            unit_id=unit.unit_id,
            cell_type=unit.cell_type,
            region=region,
            layer=layer,
            channel=ch,
            depth_um=float(ch) * SITE_PITCH_UM,
            place_center_cm=unit.place_center_cm,
            hd_pref_rad=unit.hd_pref_rad,
            gain=unit.gain,
            rate_model=unit.rate_model,
            ratinabox_class=unit.ratinabox_class,
        ))
    return assigned
