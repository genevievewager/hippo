"""Drift processes: place-field, state, and gain."""

from __future__ import annotations

import numpy as np

from hippo_sim.anatomy import Unit
from hippo_sim.config import SimConfig


class DriftState:
    """Within-session drift state for all units."""

    def __init__(self, units: list[Unit], config: SimConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.place_centers = np.array([u.place_center_cm.copy() for u in units])
        self.gains = np.array([u.gain for u in units])
        self.state_ou = 0.0
        self._steps_since_place_update = 0
        self._place_update_steps = int(
            config.drift_params["place_drift_update_s"] / config.behavior_dt
        )

    def step(self) -> tuple[np.ndarray, float, np.ndarray]:
        """Advance drift one behavior timestep. Returns (place_centers, state_mod, gains)."""
        dt = self.config.behavior_dt
        dp = self.config.drift_params

        tau_state = dp["state_drift_tau_s"]
        self.state_ou += (-self.state_ou / tau_state) * dt + dp["state_drift_sigma"] * np.sqrt(dt) * self.rng.standard_normal()
        state_mod = float(np.exp(self.state_ou))

        tau_gain = dp["gain_drift_tau_s"]
        gain_noise = (-np.log(self.gains) / tau_gain) * dt + dp["gain_drift_sigma"] * np.sqrt(dt) * self.rng.standard_normal(len(self.gains))
        self.gains = np.clip(self.gains * np.exp(gain_noise), 0.5, 2.0)

        self._steps_since_place_update += 1
        if self._steps_since_place_update >= self._place_update_steps:
            self._steps_since_place_update = 0
            sd = dp["place_drift_sd_cm_per_min"] * (dp["place_drift_update_s"] / 60.0)
            self.place_centers += self.rng.normal(0, sd, self.place_centers.shape)
            self.place_centers = np.clip(self.place_centers, 5, self.config.arena_size_cm - 5)

        return self.place_centers, state_mod, self.gains
