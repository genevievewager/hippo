"""Neural activity backend selector."""

from __future__ import annotations

import numpy as np

from hippo_sim.anatomy import AnatomyMap, Unit
from hippo_sim.behavior import BehaviorSimulationResult, BehaviorTrace
from hippo_sim.config import SimConfig
from hippo_sim.drift import DriftState
from hippo_sim.features import compute_global_features
from hippo_sim.rate_equations import integrate_rates
from hippo_sim.ratinabox_neural_backend import simulate_ratinabox_neural_activity


def simulate_neural_activity(
    config: SimConfig,
    behavior: BehaviorTrace,
    rng: np.random.Generator,
    anatomy: AnatomyMap | None = None,
    behavior_result: BehaviorSimulationResult | None = None,
) -> tuple[list[Unit], np.ndarray, dict]:
    """
    Dispatch to the configured neural activity backend.

    Both backends return units and rates_hz with shape (n_units, n_behavior_steps).
    """
    if config.neural_backend == "custom_rate_equations":
        if anatomy is None:
            raise ValueError("custom_rate_equations backend requires anatomy with units.")
        global_features = compute_global_features(behavior, config)
        drift_state = DriftState(anatomy.units, config, rng)
        rates = integrate_rates(anatomy.units, global_features, drift_state, config)
        metadata = {
            "neural_backend": "custom_rate_equations",
            "n_units": len(anatomy.units),
            "rate_units": "Hz",
            "poisson_spike_method": "poisson_count_per_bin",
        }
        return anatomy.units, rates, metadata

    if config.neural_backend == "ratinabox_neurons":
        env = behavior_result.env if behavior_result else None
        agent = behavior_result.agent if behavior_result else None
        units, rates, metadata = simulate_ratinabox_neural_activity(
            config, behavior, env=env, agent=agent, rng=rng,
        )
        return units, rates, metadata

    raise ValueError(
        f"Unknown neural_backend {config.neural_backend!r}. "
        "Use 'custom_rate_equations' or 'ratinabox_neurons'."
    )
