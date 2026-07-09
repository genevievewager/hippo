"""Kilosort-like spike re-extraction with realistic errors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.config import SimConfig
from hippo_sim.recording import RecordedEvent, UnitTemplate


@dataclass
class SortedSpike:
    time_s: float
    unit_id: int
    channel: int
    confidence: float


def kilosort_like_sort(
    events: list[RecordedEvent],
    templates: list[UnitTemplate],
    ground_truth_trains: list,
    config: SimConfig,
    rng: np.random.Generator,
) -> list[SortedSpike]:
    """
    Template-matching sorter with misses, jitter, merges, FPs, contamination.
    """
    params = config.sorting_params
    template_by_unit = {t.unit_id: t for t in templates}
    sorted_spikes: list[SortedSpike] = []

    gt_lookup: dict[int, list[float]] = {}
    for train in ground_truth_trains:
        gt_lookup[train.unit_id] = list(train.spike_times_s)

    matched_gt: dict[int, set] = {uid: set() for uid in gt_lookup}

    for event in events:
        if event.amplitude_uv < params["detection_threshold_uv"]:
            continue

        if rng.random() < params["miss_rate"]:
            continue

        best_uid = event.unit_id
        best_corr = 1.0
        tmpl = template_by_unit.get(best_uid)
        if tmpl is not None:
            ref = tmpl.waveform * event.amplitude_uv
            denom = np.linalg.norm(ref) * np.linalg.norm(event.waveform) + 1e-9
            best_corr = float(np.dot(ref, event.waveform) / denom)

        if best_corr < params["match_corr_thresh"]:
            if rng.random() < 0.5:
                continue
            best_uid = rng.integers(0, len(templates))

        t_sorted = event.time_s + rng.normal(0, params["jitter_ms"] / 1000.0)

        if rng.random() < params["merge_prob"]:
            t_sorted += rng.uniform(-0.002, 0.002)

        confidence = float(np.clip(best_corr + rng.normal(0, 0.05), 0, 1))

        if rng.random() < params["contamination_rate"]:
            alt = rng.integers(0, len(templates))
            if alt != best_uid:
                best_uid = int(alt)
                confidence *= 0.5

        sorted_spikes.append(SortedSpike(
            time_s=max(0.0, t_sorted),
            unit_id=int(best_uid),
            channel=event.channel,
            confidence=confidence,
        ))

        if best_uid in gt_lookup:
            gt_times = gt_lookup[best_uid]
            if gt_times:
                idx = int(np.argmin(np.abs(np.array(gt_times) - event.time_s)))
                matched_gt[best_uid].add(idx)

    n_fp = int(len(events) * params["false_positive_rate"])
    if n_fp > 0 and events:
        t_max = max(e.time_s for e in events)
        for _ in range(n_fp):
            sorted_spikes.append(SortedSpike(
                time_s=rng.uniform(0, t_max),
                unit_id=int(rng.integers(0, len(templates))),
                channel=int(rng.integers(0, config.n_channels)),
                confidence=float(rng.uniform(0.3, 0.7)),
            ))

    sorted_spikes.sort(key=lambda s: s.time_s)
    return sorted_spikes
