"""Neuropixels-like extracellular recording degradation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.anatomy import AnatomyMap, Unit
from hippo_sim.config import SimConfig, WAVEFORM_POST_SAMPLES, WAVEFORM_PRE_SAMPLES
from hippo_sim.spikes import SpikeTrain


@dataclass
class UnitTemplate:
    unit_id: int
    best_channel: int
    channel_ids: np.ndarray
    amplitudes_uv: np.ndarray
    waveform: np.ndarray  # (n_samples,)
    peak_amplitude_uv: float


@dataclass
class RecordedEvent:
    time_s: float
    channel: int
    amplitude_uv: float
    unit_id: int
    waveform: np.ndarray
    is_collision: bool = False


def _make_waveform(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic extracellular action potential template."""
    t = np.linspace(-1, 1, n_samples)
    w = -0.8 * np.exp(-((t + 0.3) ** 2) / 0.02) + np.exp(-((t - 0.1) ** 2) / 0.005)
    w += 0.05 * rng.standard_normal(n_samples)
    w /= np.max(np.abs(w)) + 1e-9
    return w


def build_unit_templates(
    anatomy: AnatomyMap,
    config: SimConfig,
    rng: np.random.Generator,
) -> list[UnitTemplate]:
    """Assign each unit a local multi-channel template footprint."""
    params = config.recording_params
    span_min, span_max = params["template_span_channels"]
    amp_lo, amp_hi = params["amplitude_range_uv"]
    n_samples = WAVEFORM_PRE_SAMPLES + WAVEFORM_POST_SAMPLES
    templates: list[UnitTemplate] = []

    for unit in anatomy.units:
        span = int(rng.integers(span_min, span_max + 1))
        half = span // 2
        ch_lo = max(0, unit.channel - half)
        ch_hi = min(config.n_channels - 1, unit.channel + half)
        ch_ids = np.arange(ch_lo, ch_hi + 1)

        dist = np.abs(ch_ids - unit.channel).astype(float)
        amps = (amp_lo + (amp_hi - amp_lo) * rng.random()) * np.exp(-dist / 2.0)
        amps = amps * rng.uniform(0.7, 1.0, size=len(amps))

        waveform = _make_waveform(n_samples, rng)
        templates.append(UnitTemplate(
            unit_id=unit.unit_id,
            best_channel=unit.channel,
            channel_ids=ch_ids,
            amplitudes_uv=amps,
            waveform=waveform,
            peak_amplitude_uv=float(np.max(amps)),
        ))

    return templates


def simulate_recording(
    spike_trains: list[SpikeTrain],
    templates: list[UnitTemplate],
    config: SimConfig,
    rng: np.random.Generator,
) -> list[RecordedEvent]:
    """
    Event-based recording simulation with motion drift, collisions, and noise.
    Returns detected-threshold candidate events (pre-sorting).
    """
    params = config.recording_params
    n_samples = WAVEFORM_PRE_SAMPLES + WAVEFORM_POST_SAMPLES
    events: list[RecordedEvent] = []
    template_by_unit = {t.unit_id: t for t in templates}

    motion_scale = np.ones(len(templates))
    last_motion_update = 0.0
    motion_interval = 60.0

    all_spikes: list[tuple[float, int]] = []
    for train in spike_trains:
        for t_sp in train.spike_times_s:
            all_spikes.append((t_sp, train.unit_id))
    all_spikes.sort(key=lambda x: x[0])

    recent_times: dict[int, list[float]] = {}

    for t_sp, uid in all_spikes:
        if t_sp - last_motion_update > motion_interval:
            drift = 1.0 + params["motion_amplitude_drift_per_min"] * rng.standard_normal(len(templates))
            motion_scale = np.clip(drift, 0.5, 1.5)
            last_motion_update = t_sp

        tmpl = template_by_unit[uid]
        scale = motion_scale[uid]

        is_collision = False
        ch = tmpl.best_channel
        if ch in recent_times:
            for rt in recent_times[ch]:
                if abs(t_sp - rt) < 0.001:
                    is_collision = True
                    break
        if is_collision and rng.random() < params["overlap_collision_prob"]:
            continue

        amp = tmpl.peak_amplitude_uv * scale * rng.uniform(0.85, 1.15)
        noise = rng.normal(0, params["noise_std_uv"], n_samples)
        if rng.random() < params["burst_noise_prob"]:
            noise += params["burst_noise_amplitude_uv"] * rng.standard_normal(n_samples)

        waveform = tmpl.waveform * amp + noise

        events.append(RecordedEvent(
            time_s=t_sp,
            channel=ch,
            amplitude_uv=amp,
            unit_id=uid,
            waveform=waveform,
            is_collision=is_collision,
        ))

        recent_times.setdefault(ch, []).append(t_sp)
        recent_times[ch] = [x for x in recent_times[ch] if t_sp - x < 0.01]

    return events
