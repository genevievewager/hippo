"""RatInABox neural-class backend for rate-based hippocampal activity.

Uses the maximally hippocampal population table in
``hippo_sim.hippocampal_populations``: anatomically mapped RiaB classes,
post-hoc theta / ripple / DG-sparsity / CA3-recurrent dynamics, and
trisynaptic / entorhinal feedforward coupling (``hippo_sim.feedforward``).
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np

from hippo_sim.anatomy import Unit, assign_probe_channels
from hippo_sim.behavior import BehaviorTrace
from hippo_sim.config import SimConfig
from hippo_sim.features import compute_global_features
from hippo_sim.feedforward import (
    apply_int_to_ca1_inhibition,
    apply_trisynaptic_feedforward,
)
from hippo_sim.hippocampal_populations import (
    HIPPOCAMPAL_RIA_B_POPULATIONS,
    population_count,
    population_table_summary,
)

logger = logging.getLogger(__name__)


@dataclass
class RiaBGroup:
    name: str
    ratinabox_class: str
    cell_type: str
    rate_model: str
    neurons: object | None
    rates_hz: np.ndarray  # (n_cells, n_steps)
    dynamics: dict


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
    """Speed-modulated population rates (RiaB SpeedCell is always n=1)."""
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


def _compute_interneuron_rates(
    behavior: BehaviorTrace,
    n_cells: int,
    config: SimConfig,
    ca1_pyr_rates: np.ndarray | None,
) -> np.ndarray:
    """Synthetic CA1 oriens interneurons: high tonic + theta, anti-CA1-pyr."""
    rp = config.ratinabox_params
    rate_params = config.rate_params.get("CA1_int", {})
    baseline = float(rp.get("int_baseline_hz", rate_params.get("baseline_hz", 18.0)))
    anti_w = float(rp.get("int_anti_ca1_weight", 0.35))
    w_theta = float(rate_params.get("w_theta", 0.4))
    theta_freq = float(rate_params.get("theta_freq_hz", 8.0))

    t = behavior.time_s
    theta = 1.0 + w_theta * np.cos(2 * np.pi * theta_freq * t)
    n_steps = len(t)

    if ca1_pyr_rates is not None and ca1_pyr_rates.size:
        ca1_mean = ca1_pyr_rates.mean(axis=0)
        ca1_norm = ca1_mean / max(float(ca1_mean.max()), 1e-6)
    else:
        ca1_norm = np.zeros(n_steps)

    rng = np.random.default_rng(config.seed + 77)
    rates = np.zeros((n_cells, n_steps), dtype=float)
    for i in range(n_cells):
        gain = rng.uniform(0.8, 1.2)
        rates[i] = gain * baseline * theta * (1.0 - anti_w * ca1_norm)
    return np.maximum(rates, 0.0)


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
            "RatInABox is required for neural activity simulation."
        ) from exc

    mapping = {
        "PlaceCells": PlaceCells,
        "HeadDirectionCells": HeadDirectionCells,
        "BoundaryVectorCells": BoundaryVectorCells,
        "GridCells": GridCells,
        "SpeedCell": SpeedCell,
    }

    if class_name == "PhasePrecessingPlaceCells":
        try:
            from ratinabox.contribs.PhasePrecessingPlaceCells import (
                PhasePrecessingPlaceCells,
            )
        except ImportError as exc:
            raise ImportError(
                "PhasePrecessingPlaceCells contrib is required for the "
                "maximally hippocampal RiaB config."
            ) from exc
        mapping["PhasePrecessingPlaceCells"] = PhasePrecessingPlaceCells

    cls = mapping.get(class_name)
    if cls is None:
        return None
    return cls(agent, params=params)


def apply_hippocampal_dynamics(
    rates_hz: np.ndarray,
    *,
    cell_type: str,
    dynamics: dict,
    behavior: BehaviorTrace,
    config: SimConfig,
    global_features: dict,
) -> np.ndarray:
    """Overlay theta / ripple / sparsity / recurrent / speed on RiaB rates."""
    if not config.ratinabox_params.get("apply_hippocampal_dynamics", True):
        return rates_hz

    out = np.asarray(rates_hz, dtype=float).copy()
    rp = config.rate_params.get(cell_type, {})
    theta_phase = global_features["theta_phase"]
    ripple = global_features["ripple"]
    speed = global_features["speed"]

    if dynamics.get("theta"):
        w_theta = float(rp.get("w_theta", 0.1))
        out *= 1.0 + w_theta * np.cos(theta_phase)[None, :]

    if dynamics.get("speed_gain"):
        w_speed = float(rp.get("w_speed", 0.2))
        thresh = float(rp.get("speed_thresh_cm_s", 2.0))
        speed_drive = np.clip(speed - thresh, 0, None) / max(1.0, 30.0 - thresh)
        out *= 1.0 + w_speed * speed_drive[None, :]

    if dynamics.get("ripple"):
        w_ripple = float(rp.get("w_ripple", 0.0))
        amp = float(rp.get("amplitude_hz", 10.0))
        out += w_ripple * amp * ripple[None, :]

    if dynamics.get("sparsity"):
        thresh = float(rp.get("sparsity_thresh", 0.3))
        baseline = float(rp.get("baseline_hz", 0.05))
        peak = np.maximum(out.max(axis=1, keepdims=True), 1e-6)
        gate = out / peak
        sparse = gate < thresh
        out = np.where(sparse, baseline * 0.1, out)

    if dynamics.get("recurrent"):
        w_rec = float(rp.get("w_recurrent", 0.15))
        pop_mean = out.mean(axis=0, keepdims=True)
        out += w_rec * pop_mean

    return np.maximum(out, 0.0)


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
    global_features = compute_global_features(behavior, config)

    if hasattr(agent, "reset_history"):
        agent.reset_history()

    groups: list[RiaBGroup] = []
    neuron_objects: list[tuple[dict, object]] = []

    for spec in HIPPOCAMPAL_RIA_B_POPULATIONS:
        n = population_count(spec, rp)
        if n <= 0:
            continue

        class_name = spec["ratinabox_class"]

        # Synthetic / fallback groups — no RiaB neuron object.
        if class_name in ("SpeedCells_fallback", "CA1_Interneurons_synthetic"):
            continue

        params = {
            "n": n,
            "max_fr": rate_scale,
            "min_fr": 0.0,
            **dict(spec.get("riab_params") or {}),
        }
        try:
            neurons = _try_create_neurons(agent, class_name, params)
        except Exception as exc:
            warnings.warn(f"Skipping RatInABox {spec['name']} ({class_name}): {exc}")
            logger.warning(
                "Skipping RatInABox %s (%s): %s", spec["name"], class_name, exc
            )
            continue
        if neurons is None:
            warnings.warn(f"Unknown RatInABox class {class_name} for {spec['name']}")
            continue
        if hasattr(neurons, "reset_history"):
            neurons.reset_history()
        neuron_objects.append((spec, neurons))

    for step in range(n_steps):
        _sync_agent_to_behavior_step(agent, behavior, step)
        for _, neurons in neuron_objects:
            neurons.update()

    ca1_pyr_rates: np.ndarray | None = None

    for spec, neurons in neuron_objects:
        rates_hz = extract_rates_from_ratinabox_neurons(neurons, n_steps)
        rates_hz = apply_hippocampal_dynamics(
            rates_hz,
            cell_type=spec["cell_type"],
            dynamics=spec["dynamics"],
            behavior=behavior,
            config=config,
            global_features=global_features,
        )
        group = RiaBGroup(
            name=spec["name"],
            ratinabox_class=spec["ratinabox_class"],
            cell_type=spec["cell_type"],
            rate_model=f"ratinabox_{spec['name']}",
            neurons=neurons,
            rates_hz=rates_hz,
            dynamics=spec["dynamics"],
        )
        groups.append(group)

    # Speed population (fallback; RiaB SpeedCell is n=1).
    for spec in HIPPOCAMPAL_RIA_B_POPULATIONS:
        if spec["ratinabox_class"] != "SpeedCells_fallback":
            continue
        n = population_count(spec, rp)
        if n <= 0:
            continue
        speed_rates = _compute_speed_fallback_rates(behavior, n, config)
        speed_rates = apply_hippocampal_dynamics(
            speed_rates,
            cell_type=spec["cell_type"],
            dynamics=spec["dynamics"],
            behavior=behavior,
            config=config,
            global_features=global_features,
        )
        groups.append(RiaBGroup(
            name=spec["name"],
            ratinabox_class=spec["ratinabox_class"],
            cell_type=spec["cell_type"],
            rate_model=f"ratinabox_{spec['name']}",
            neurons=None,
            rates_hz=speed_rates,
            dynamics=spec["dynamics"],
        ))

    # Excitatory trisynaptic / EC feedforward (before interneurons).
    ff_meta = apply_trisynaptic_feedforward(groups, ratinabox_params=rp)

    for g in groups:
        if g.name == "CA1_place_pp":
            ca1_pyr_rates = g.rates_hz
            break

    # CA1 interneurons track post-feedforward CA1, then inhibit CA1.
    for spec in HIPPOCAMPAL_RIA_B_POPULATIONS:
        if spec["ratinabox_class"] != "CA1_Interneurons_synthetic":
            continue
        n = population_count(spec, rp)
        if n <= 0:
            continue
        int_rates = _compute_interneuron_rates(behavior, n, config, ca1_pyr_rates)
        if spec["dynamics"].get("ripple"):
            w_ripple = float(config.rate_params.get("CA1_int", {}).get("w_ripple", 0.3))
            amp = float(config.rate_params.get("CA1_int", {}).get("amplitude_hz", 4.0))
            int_rates = int_rates + w_ripple * amp * global_features["ripple"][None, :]
            int_rates = np.maximum(int_rates, 0.0)
        groups.append(RiaBGroup(
            name=spec["name"],
            ratinabox_class=spec["ratinabox_class"],
            cell_type=spec["cell_type"],
            rate_model=f"ratinabox_{spec['name']}",
            neurons=None,
            rates_hz=int_rates,
            dynamics=spec["dynamics"],
        ))

    # Second pass: INT→CA1 inhibition after interneurons exist.
    int_meta = apply_int_to_ca1_inhibition(groups, ratinabox_params=rp)
    ff_meta["int_to_ca1"] = int_meta

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
                if hasattr(group.neurons, "place_cell_centres"):
                    ctr = np.asarray(group.neurons.place_cell_centres[i]) * 100.0
                    place_center = ctr[:2]
                if hasattr(group.neurons, "preferred_angles"):
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

    group_counts = {g.name: g.rates_hz.shape[0] for g in groups}
    metadata = {
        "neural_backend": "ratinabox_neurons",
        "ratinabox_cell_groups": group_counts,
        "ratinabox_population_table": population_table_summary(rp),
        "apply_hippocampal_dynamics": bool(rp.get("apply_hippocampal_dynamics", True)),
        "feedforward": ff_meta,
        "n_units": int(rates_hz.shape[0]),
        "rate_units": "Hz",
        "poisson_spike_method": "poisson_count_per_bin",
        "rate_scale_hz": rate_scale,
    }
    return units, rates_hz, metadata
