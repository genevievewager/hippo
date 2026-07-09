"""Cell-type rate equations driven by behavioral features."""

from __future__ import annotations

import numpy as np

from hippo_sim.anatomy import Unit
from hippo_sim.config import SimConfig
from hippo_sim.features import (
    acceleration_feature,
    boundary_feature,
    head_direction_feature,
    place_feature,
    speed_feature,
    theta_modulation,
)


def compute_target_rate(
    unit: Unit,
    t_idx: int,
    global_features: dict,
    place_center: np.ndarray,
    state_mod: float,
    gain: float,
    config: SimConfig,
    population_mean_rate: float = 0.0,
) -> float:
    """Compute instantaneous target firing rate (Hz) for one unit at one timestep."""
    p = config.rate_params[unit.cell_type]
    pos = global_features["position"][t_idx]
    hd = global_features["head_direction"][t_idx]
    spd = global_features["speed"][t_idx]
    acc = global_features["acceleration"][t_idx]
    dist_wall = global_features["distance_to_wall"][t_idx]
    theta_ph = global_features["theta_phase"][t_idx]
    ripple = global_features["ripple"][t_idx]

    f_place = place_feature(pos, place_center, p["sigma_place_cm"])
    f_hd = float(head_direction_feature(np.array([hd]), unit.hd_pref_rad, p["kappa_hd"])[0])
    f_spd = float(speed_feature(np.array([spd]), p["speed_thresh_cm_s"])[0])
    f_acc = float(acceleration_feature(np.array([acc]))[0])
    f_bnd = float(boundary_feature(np.array([dist_wall]))[0])
    f_theta = float(theta_modulation(np.array([theta_ph]), p["w_theta"])[0])

    drive = (
        p["baseline_hz"]
        + p["amplitude_hz"] * f_place * (1.0 + p["w_hd"] * f_hd)
        * (1.0 + p["w_speed"] * f_spd)
        * (1.0 + 0.2 * f_acc)
        * f_theta
        + p["w_boundary"] * p["amplitude_hz"] * f_bnd
        + p.get("w_ripple", 0) * p["amplitude_hz"] * ripple
    )

    if unit.cell_type == "CA3_pyr":
        drive += p.get("w_recurrent", 0) * population_mean_rate

    if unit.cell_type == "DG_granule":
        gate = f_place * (1.0 + p["w_speed"] * f_spd)
        if gate < p["sparsity_thresh"]:
            drive = p["baseline_hz"] * 0.1

    drive *= state_mod * gain
    return max(0.0, float(drive))


def integrate_rates(
    units: list[Unit],
    global_features: dict,
    drift_state,
    config: SimConfig,
) -> np.ndarray:
    """
    Euler integration of tau dR/dt = -R + target_rate.
    Returns rates array (n_units, n_steps).
    """
    n_units = len(units)
    n_steps = config.n_behavior_steps
    rates = np.zeros((n_units, n_steps))
    dt = config.behavior_dt

    r = np.array([config.rate_params[u.cell_type]["baseline_hz"] for u in units])

    for t in range(n_steps):
        place_centers, state_mod, gains = drift_state.step()
        pop_mean = float(np.mean(r))

        targets = np.array([
            compute_target_rate(
                units[i], t, global_features,
                place_centers[i], state_mod, gains[i],
                config, pop_mean,
            )
            for i in range(n_units)
        ])

        for i, unit in enumerate(units):
            tau = config.rate_params[unit.cell_type]["tau_s"]
            r[i] += (dt / tau) * (-r[i] + targets[i])

        rates[:, t] = r

    return rates
