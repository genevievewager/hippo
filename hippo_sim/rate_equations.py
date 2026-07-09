"""Cell-type rate equations driven by behavioral features."""

from __future__ import annotations

import numpy as np
from scipy.special import i0
from tqdm import tqdm

from hippo_sim.anatomy import Unit
from hippo_sim.config import SimConfig


def _build_unit_param_arrays(units: list[Unit], config: SimConfig) -> dict[str, np.ndarray]:
    """Pack per-unit rate parameters into arrays for vectorized computation."""
    n_units = len(units)
    arrays: dict[str, np.ndarray] = {
        "baseline_hz": np.zeros(n_units),
        "amplitude_hz": np.zeros(n_units),
        "sigma_place_cm": np.zeros(n_units),
        "w_hd": np.zeros(n_units),
        "kappa_hd": np.zeros(n_units),
        "hd_pref_rad": np.zeros(n_units),
        "w_speed": np.zeros(n_units),
        "speed_thresh_cm_s": np.zeros(n_units),
        "w_theta": np.zeros(n_units),
        "w_boundary": np.zeros(n_units),
        "w_ripple": np.zeros(n_units),
        "w_recurrent": np.zeros(n_units),
        "sparsity_thresh": np.full(n_units, np.inf),
        "tau_s": np.zeros(n_units),
        "is_ca3": np.zeros(n_units, dtype=bool),
        "is_dg": np.zeros(n_units, dtype=bool),
    }

    for i, unit in enumerate(units):
        p = config.rate_params[unit.cell_type]
        arrays["baseline_hz"][i] = p["baseline_hz"]
        arrays["amplitude_hz"][i] = p["amplitude_hz"]
        arrays["sigma_place_cm"][i] = p["sigma_place_cm"]
        arrays["w_hd"][i] = p["w_hd"]
        arrays["kappa_hd"][i] = p["kappa_hd"]
        arrays["hd_pref_rad"][i] = unit.hd_pref_rad
        arrays["w_speed"][i] = p["w_speed"]
        arrays["speed_thresh_cm_s"][i] = p["speed_thresh_cm_s"]
        arrays["w_theta"][i] = p["w_theta"]
        arrays["w_boundary"][i] = p["w_boundary"]
        arrays["w_ripple"][i] = p.get("w_ripple", 0.0)
        arrays["w_recurrent"][i] = p.get("w_recurrent", 0.0)
        arrays["tau_s"][i] = p["tau_s"]
        arrays["is_ca3"][i] = unit.cell_type == "CA3_pyr"
        arrays["is_dg"][i] = unit.cell_type == "DG_granule"
        if unit.cell_type == "DG_granule":
            arrays["sparsity_thresh"][i] = p["sparsity_thresh"]

    return arrays


def compute_target_rates(
    t_idx: int,
    place_centers: np.ndarray,
    state_mod: float,
    gains: np.ndarray,
    pop_mean: float,
    global_features: dict,
    params: dict[str, np.ndarray],
) -> np.ndarray:
    """Vectorized target firing rates (Hz) for all units at one timestep."""
    pos = global_features["position"][t_idx]
    hd = global_features["head_direction"][t_idx]
    spd = global_features["speed"][t_idx]
    acc = global_features["acceleration"][t_idx]
    dist_wall = global_features["distance_to_wall"][t_idx]
    theta_ph = global_features["theta_phase"][t_idx]
    ripple = global_features["ripple"][t_idx]

    diff = place_centers - pos
    dist_sq = np.sum(diff * diff, axis=1)
    sigma_sq = params["sigma_place_cm"] ** 2
    f_place = np.exp(-dist_sq / (2.0 * sigma_sq))

    kappa = params["kappa_hd"]
    f_hd = np.exp(kappa * np.cos(hd - params["hd_pref_rad"]))
    f_hd /= np.exp(kappa) / i0(kappa)

    speed_denom = np.maximum(1.0, 30.0 - params["speed_thresh_cm_s"])
    f_spd = np.clip(spd - params["speed_thresh_cm_s"], 0, None) / speed_denom
    f_acc = np.clip(abs(acc) / 50.0, 0, 1)
    f_bnd = np.exp(-(dist_wall ** 2) / (2.0 * 15.0 ** 2))
    f_theta = 1.0 + params["w_theta"] * np.cos(theta_ph)

    drive = (
        params["baseline_hz"]
        + params["amplitude_hz"] * f_place * (1.0 + params["w_hd"] * f_hd)
        * (1.0 + params["w_speed"] * f_spd)
        * (1.0 + 0.2 * f_acc)
        * f_theta
        + params["w_boundary"] * params["amplitude_hz"] * f_bnd
        + params["w_ripple"] * params["amplitude_hz"] * ripple
    )

    if np.any(params["is_ca3"]):
        drive[params["is_ca3"]] += params["w_recurrent"][params["is_ca3"]] * pop_mean

    if np.any(params["is_dg"]):
        gate = f_place * (1.0 + params["w_speed"] * f_spd)
        sparse = params["is_dg"] & (gate < params["sparsity_thresh"])
        drive[sparse] = params["baseline_hz"][sparse] * 0.1

    drive *= state_mod * gains
    return np.maximum(drive, 0.0)


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

    params = _build_unit_param_arrays(units, config)
    r = params["baseline_hz"].copy()
    tau = params["tau_s"]
    dt_over_tau = dt / tau

    for t in tqdm(range(n_steps), desc="Integrating rates", unit="step"):
        place_centers, state_mod, gains = drift_state.step()
        pop_mean = float(np.mean(r))

        targets = compute_target_rates(
            t, place_centers, state_mod, gains, pop_mean, global_features, params
        )

        r += dt_over_tau * (-r + targets)
        rates[:, t] = r

    return rates
