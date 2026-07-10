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
    time_axis: np.ndarray | None = None,
) -> list[SpikeTrain]:
    """
    Generate ground-truth spike times from rate matrix (n_units, n_steps).

    Uses Poisson spike counts per behavior bin so multiple spikes per bin
    are allowed when rates are high.
    """
    n_units, n_steps = rates.shape
    dt = config.behavior_dt
    if time_axis is None:
        time_axis = config.time_axis
    trains: list[SpikeTrain] = []

    for u in range(n_units):
        spike_times: list[float] = []
        for t in range(n_steps):
            lam = float(rates[u, t]) * dt
            n_spikes = int(rng.poisson(lam))
            if n_spikes > 0:
                offsets = rng.uniform(0.0, dt, size=n_spikes)
                spike_times.extend((time_axis[t] + offsets).tolist())
        trains.append(SpikeTrain(
            unit_id=u,
            spike_times_s=np.sort(np.asarray(spike_times, dtype=float)),
        ))

    return trains
