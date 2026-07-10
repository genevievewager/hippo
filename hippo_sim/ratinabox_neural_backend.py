"""RatInABox neural-class backend for rate-based hippocampal activity.

RatInABox neurons are fundamentally rate-based. Their firing rates are sampled
into ground-truth Poisson spike trains so downstream Neuropixels processing
remains unchanged.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

from hippo_sim.anatomy import Unit, assign_probe_channels
from hippo_sim.behavior import BehaviorTrace
from hippo_sim.config import SimConfig

logger = logging.getLogger(__name__)

RATINABOX_TO_CELL_TYPE = {
    "PlaceCells": "CA1_pyr",
    "HeadDirectionCells": "CA2_pyr",
    "BoundaryVectorCells": "CA3_pyr",
    "GridCells": "CA1_pyr",
    "SpeedCells": "DG_granule",
    "SpeedCells_fallback": "DG_granule",
}


@dataclass
class RiaBGroup:
    name: str
    ratinabox_class: str
    cell_type: str
    rate_model: str
    neurons: object | None
    rates_hz: np.ndarray  # (n_cells, n_steps)


def extract_rates_from_ratinabox_neurons(neurons, n_steps: int) -> np.ndarray:
    """
    Return rates with shape (n_cells, n_steps).

    Detects common RatInABox rate storage formats and transposes if needed.
    """
    if neurons is None:
        raise ValueError("Cannot extract rates from None neurons object.")

    rates = None
    if hasattr(neurons, "get_history_arrays"):
        history = neurons.get_history_arrays()
        for key in ("firingrate", "firingrates", "rate", "rates"):
            if key in history:
                rates = np.asarray(history[key], dtype=float)
                break
    if rates is None and hasattr(neurons, "history"):
        history = neurons.history
        for key in ("firingrate", "firingrates", "rate", "rates"):
            if key in history and len(history[key]) > 0:
                rates = np.asarray(history[key], dtype=float)
                break
    if rates is None:
        for attr in ("firingrate", "firingrates", "rate", "rates"):
            if hasattr(neurons, attr):
                val = getattr(neurons, attr)
                if val is not None:
                    rates = np.asarray(val, dtype=float)
                    break

    if rates is None:
        raise ValueError(
            f"Could not extract firing rates from {type(neurons).__name__}. "
            "Expected history['firingrate'] or neurons.firingrate."
        )

    if rates.ndim == 1:
        rates = rates.reshape(-1, 1)
    elif rates.ndim == 2:
        if rates.shape[0] == n_steps and rates.shape[1] != n_steps:
            rates = rates.T
        elif rates.shape[0] != n_steps and rates.shape[1] == n_steps:
            pass
        else:
            rates = rates[:n_steps].T if rates.shape[0] >= n_steps else rates.T[:, :n_steps]
    else:
        raise ValueError(f"Unexpected rate array shape {rates.shape}")

    n_cells = getattr(neurons, "n", rates.shape[0])
    if rates.shape[0] != n_cells:
        rates = rates[:n_cells]

    if rates.shape[1] < n_steps:
        pad = np.zeros((rates.shape[0], n_steps - rates.shape[1]))
        rates = np.hstack([rates, pad])
    elif rates.shape[1] > n_steps:
        rates = rates[:, :n_steps]

    return rates


def _compute_speed_fallback_rates(
    behavior: BehaviorTrace,
    n_cells: int,
    config: SimConfig,
) -> np.ndarray:
    """Speed-modulated fallback rates when RatInABox SpeedCell supports only one unit."""
    rp = config.ratinabox_params
    baseline = float(rp["speed_baseline_hz"])
    amplitude = float(rp["speed_amplitude_hz"])
    thresh = float(rp["speed_threshold_cm_s"])
    scale = float(rp["speed_scale_cm_s"])

    speed = behavior.speed_cm_s
    drive = np.clip(speed - thresh, 0, None) / max(scale, 1e-6)
    base_rate = baseline + amplitude * drive

    rng = np.random.default_rng(config.seed + 99)
    n_steps = len(speed)
    rates = np.zeros((n_cells, n_steps), dtype=float)
    for i in range(n_cells):
        gain = rng.uniform(0.7, 1.3)
        rates[i] = base_rate * gain
    return rates


def _sync_agent_to_behavior_step(agent, behavior: BehaviorTrace, step: int) -> None:
    """Set RatInABox Agent state from saved behavior (same trajectory as behavior.csv)."""
    pos_m = behavior.position_cm[step] / 100.0
    vel_m_s = behavior.velocity_cm_s[step] / 100.0
    agent.pos = np.asarray(pos_m, dtype=float)
    agent.t = float(behavior.time_s[step])
    if hasattr(agent, "velocity"):
        agent.velocity = np.asarray(vel_m_s, dtype=float)
    if hasattr(agent, "head_direction"):
        hd = behavior.head_direction_rad[step]
        agent.head_direction = np.array([np.cos(hd), np.sin(hd)], dtype=float)
    if hasattr(agent, "measured_velocity"):
        agent.measured_velocity = np.asarray(vel_m_s, dtype=float)


def _try_create_neurons(agent, class_name: str, params: dict):
    try:
        from ratinabox.Neurons import (
            BoundaryVectorCells,
            GridCells,
            HeadDirectionCells,
            PlaceCells,
            SpeedCell,
        )
    except ImportError as exc:
        raise ImportError(
            "RatInABox is required for neural_backend='ratinabox_neurons'."
        ) from exc

    mapping = {
        "PlaceCells": PlaceCells,
        "HeadDirectionCells": HeadDirectionCells,
        "BoundaryVectorCells": BoundaryVectorCells,
        "GridCells": GridCells,
        "SpeedCell": SpeedCell,
    }
    cls = mapping.get(class_name)
    if cls is None:
        return None
    return cls(agent, params=params)


def simulate_ratinabox_neural_activity(
    config: SimConfig,
    behavior: BehaviorTrace,
    env=None,
    agent=None,
    rng: np.random.Generator | None = None,
) -> tuple[list[Unit], np.ndarray, dict]:
    """
    Use RatInABox neural classes to generate rate-based neural activity
    from the RatInABox environment and agent trajectory.

    Returns:
        units: list[Unit]
        rates_hz: np.ndarray with shape (n_units, n_steps)
        metadata: dict with backend details
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    if env is None or agent is None:
        from hippo_sim.behavior import simulate_behavior
        result = simulate_behavior(config)
        env, agent = result.env, result.agent
        behavior = result.trace

    n_steps = config.n_behavior_steps
    rp = config.ratinabox_params
    rate_scale = float(rp["rate_scale_hz"])

    if hasattr(agent, "reset_history"):
        agent.reset_history()

    groups: list[RiaBGroup] = []
    neuron_objects: list[tuple[str, object]] = []

    group_specs = [
        ("PlaceCells", {
            "n": int(rp["n_place_cells"]),
            "max_fr": rate_scale,
            "min_fr": 0.0,
        }),
        ("HeadDirectionCells", {
            "n": int(rp["n_head_direction_cells"]),
            "max_fr": rate_scale,
            "min_fr": 0.0,
        }),
        ("BoundaryVectorCells", {
            "n": int(rp["n_boundary_vector_cells"]),
            "max_fr": rate_scale,
            "min_fr": 0.0,
        }),
    ]

    if int(rp.get("n_grid_cells", 0)) > 0:
        group_specs.append(("GridCells", {
            "n": int(rp["n_grid_cells"]),
            "max_fr": rate_scale,
            "min_fr": 0.0,
        }))

    for class_name, params in group_specs:
        try:
            neurons = _try_create_neurons(agent, class_name, params)
        except Exception as exc:
            warnings.warn(f"Skipping RatInABox {class_name}: {exc}")
            logger.warning("Skipping RatInABox %s: %s", class_name, exc)
            continue
        if neurons is None:
            continue
        if hasattr(neurons, "reset_history"):
            neurons.reset_history()
        neuron_objects.append((class_name, neurons))

    dt = config.behavior_dt
    for step in range(n_steps):
        _sync_agent_to_behavior_step(agent, behavior, step)
        for _, neurons in neuron_objects:
            neurons.update()

    for class_name, neurons in neuron_objects:
        rates_hz = extract_rates_from_ratinabox_neurons(neurons, n_steps)
        rates_hz = rates_hz * 1.0  # already scaled via max_fr
        groups.append(RiaBGroup(
            name=class_name,
            ratinabox_class=class_name,
            cell_type=RATINABOX_TO_CELL_TYPE[class_name],
            rate_model=f"ratinabox_{class_name}",
            neurons=neurons,
            rates_hz=rates_hz,
        ))

    n_speed = int(rp["n_speed_cells"])
    if n_speed > 0:
        speed_rates = _compute_speed_fallback_rates(behavior, n_speed, config)
        groups.append(RiaBGroup(
            name="SpeedCells_fallback",
            ratinabox_class="SpeedCells_fallback",
            cell_type="DG_granule",
            rate_model="ratinabox_SpeedCells_fallback",
            neurons=None,
            rates_hz=speed_rates,
        ))

    if not groups:
        raise RuntimeError("No RatInABox neural groups could be created.")

    units: list[Unit] = []
    rate_blocks: list[np.ndarray] = []
    unit_id = 0

    for group in groups:
        n_cells = group.rates_hz.shape[0]
        for i in range(n_cells):
            place_center = np.array([np.nan, np.nan])
            hd_pref = np.nan

            if group.neurons is not None:
                if group.ratinabox_class == "PlaceCells" and hasattr(group.neurons, "place_cell_centres"):
                    ctr = np.asarray(group.neurons.place_cell_centres[i]) * 100.0
                    place_center = ctr[:2]
                if group.ratinabox_class == "HeadDirectionCells" and hasattr(group.neurons, "preferred_angles"):
                    hd_pref = float(group.neurons.preferred_angles[i])

            units.append(Unit(
                unit_id=unit_id,
                cell_type=group.cell_type,
                region="",
                layer="",
                channel=0,
                depth_um=0.0,
                place_center_cm=place_center,
                hd_pref_rad=float(hd_pref) if not np.isnan(hd_pref) else 0.0,
                gain=1.0,
                rate_model=group.rate_model,
                ratinabox_class=group.ratinabox_class,
            ))
            unit_id += 1
        rate_blocks.append(group.rates_hz)

    units = assign_probe_channels(units, config, rng)
    rates_hz = np.vstack(rate_blocks)
    rates_hz = np.clip(rates_hz, 0.0, None)

    group_counts = {g.ratinabox_class: g.rates_hz.shape[0] for g in groups}
    metadata = {
        "neural_backend": "ratinabox_neurons",
        "ratinabox_cell_groups": group_counts,
        "n_units": int(rates_hz.shape[0]),
        "rate_units": "Hz",
        "poisson_spike_method": "poisson_count_per_bin",
        "rate_scale_hz": rate_scale,
    }
    return units, rates_hz, metadata
