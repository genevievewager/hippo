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
    "INT_CA1": ("CA1", "oriens"),
    "INT_CA2": ("CA2", "pyramidal"),
    "INT_CA3": ("CA3", "pyramidal"),
    "INT_DG": ("DG", "hilus"),
    "INT_SUB": ("Subiculum", "pyramidal"),
    "interneuron": ("CA1", "oriens"),  # legacy → CA1-local
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
    pitch = float(getattr(config, "site_pitch_um", SITE_PITCH_UM))
    n_channels = int(getattr(config, "n_channels", 384))
    channel_pools: dict[str, list[int]] = {}
    # Prefer capture-aware multi-type pools when cell_capture_config is present.
    capture_cfg = getattr(config, "cell_capture_config", None) or {}
    use_capture_probs = bool(capture_cfg.get("region_layer_probabilities"))

    if use_capture_probs:
        from hippo.anatomy.cell_capture import _probs_for_band

        for seg in config.region_segments:
            channels = channels_for_segment(
                seg["z_start"], seg["z_end"], pitch_um=pitch, n_channels=n_channels,
            )
            if not len(channels):
                continue
            probs = _probs_for_band(capture_cfg, seg["region"], seg["layer"])
            if not probs:
                cell_type = REGION_TO_CELL_TYPE.get((seg["region"], seg["layer"]))
                if cell_type:
                    channel_pools.setdefault(cell_type, []).extend(channels.tolist())
                continue
            for cell_type, weight in probs.items():
                if weight <= 0:
                    continue
                channel_pools.setdefault(cell_type, []).extend(channels.tolist())
    else:
        for seg in config.region_segments:
            region_key = (seg["region"], seg["layer"])
            cell_type = REGION_TO_CELL_TYPE.get(region_key)
            if cell_type is None:
                continue
            channels = channels_for_segment(
                seg["z_start"], seg["z_end"], pitch_um=pitch, n_channels=n_channels,
            )
            if len(channels):
                channel_pools.setdefault(cell_type, []).extend(channels.tolist())

    for pool in channel_pools.values():
        pool.sort()

    assigned: list[Unit] = []
    for unit in units:
        pool = channel_pools.get(unit.cell_type)
        if not pool:
            region, layer = CELL_TYPE_TO_PROBE.get(unit.cell_type, ("CA1", "pyramidal"))
            for seg in config.region_segments:
                if seg["region"] == region and seg["layer"] == layer:
                    pool = channels_for_segment(
                        seg["z_start"], seg["z_end"], pitch_um=pitch, n_channels=n_channels,
                    ).tolist()
                    break
            if not pool:
                for seg in config.region_segments:
                    if seg["region"] == region:
                        pool = channels_for_segment(
                            seg["z_start"], seg["z_end"], pitch_um=pitch, n_channels=n_channels,
                        ).tolist()
                        break
        if not pool:
            # Unit cell type has no crossed band — drop by parking on invalid sentinel
            # only if no segments at all; otherwise skip assignment to ch 0 of first band.
            if config.region_segments:
                seg = config.region_segments[0]
                pool = channels_for_segment(
                    seg["z_start"], seg["z_end"], pitch_um=pitch, n_channels=n_channels,
                ).tolist() or [0]
            else:
                pool = [0]

        ch = int(rng.choice(pool))
        # Prefer region/layer from the segment that owns this channel.
        region, layer = CELL_TYPE_TO_PROBE.get(unit.cell_type, ("CA1", "pyramidal"))
        depth = float(ch) * pitch
        for seg in config.region_segments:
            if seg["z_start"] <= depth < seg["z_end"]:
                region, layer = seg["region"], seg["layer"]
                break
        assigned.append(Unit(
            unit_id=unit.unit_id,
            cell_type=unit.cell_type,
            region=region,
            layer=layer,
            channel=ch,
            depth_um=depth,
            place_center_cm=unit.place_center_cm,
            hd_pref_rad=unit.hd_pref_rad,
            gain=unit.gain,
            rate_model=unit.rate_model,
            ratinabox_class=unit.ratinabox_class,
        ))
    return assigned
