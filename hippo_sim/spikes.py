"""Poisson spike generation from firing rates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.config import SimConfig


@dataclass
class SpikeTrain:
    unit_id: int
    spike_times_s: np.ndarray


def generate_spike_trains(
    rates: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator,
) -> list[SpikeTrain]:
    """
    Generate ground-truth spike times from rate matrix (n_units, n_steps).
    Uses inhomogeneous Poisson with sub-bin jitter for continuous times.
    """
    n_units, n_steps = rates.shape
    dt = config.behavior_dt
    trains: list[SpikeTrain] = []

    for u in range(n_units):
        prob = rates[u] * dt
        prob = np.clip(prob, 0, 1)
        spike_bins = np.where(rng.random(n_steps) < prob)[0]
        times = spike_bins * dt + rng.uniform(0, dt, size=len(spike_bins))
        trains.append(SpikeTrain(unit_id=u, spike_times_s=np.sort(times)))

    return trains
