"""Neural activity backend — RatInABox only."""

from __future__ import annotations

import numpy as np

from hippo_sim.anatomy import Unit
from hippo_sim.behavior import BehaviorSimulationResult, BehaviorTrace
from hippo_sim.config import SimConfig
from hippo_sim.ratinabox_neural_backend import simulate_ratinabox_neural_activity


def simulate_neural_activity(
    config: SimConfig,
    behavior: BehaviorTrace,
    rng: np.random.Generator,
    behavior_result: BehaviorSimulationResult | None = None,
) -> tuple[list[Unit], np.ndarray, dict]:
    """
    Generate neural rates from RatInABox neuron classes + hippocampal overlays.

    Returns units and rates_hz with shape (n_units, n_behavior_steps).
    """
    env = behavior_result.env if behavior_result else None
    agent = behavior_result.agent if behavior_result else None
    return simulate_ratinabox_neural_activity(
        config, behavior, env=env, agent=agent, rng=rng,
    )
