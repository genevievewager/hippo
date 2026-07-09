"""Environmental and proprioceptive feature computations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import i0

from hippo_sim.behavior import BehaviorTrace
from hippo_sim.config import SimConfig, RIPPLE_PARAMS


@dataclass
class FeatureTrace:
    place: np.ndarray  # (T, N_units) computed per-unit externally; placeholder
    head_direction: np.ndarray  # (T, N_units)
    speed: np.ndarray  # (T, N_units)
    acceleration: np.ndarray  # (T, N_units)
    boundary: np.ndarray  # (T, N_units)
    theta_phase: np.ndarray  # (T,)
    ripple: np.ndarray  # (T,)


def compute_global_features(behavior: BehaviorTrace, config: SimConfig) -> dict:
    """Compute session-wide driver features shared across units."""
    t = behavior.time_s
    theta_freq = config.rate_params["CA1_pyr"]["theta_freq_hz"]
    theta_phase = (2 * np.pi * theta_freq * t) % (2 * np.pi)
    ripple = _generate_ripple_envelope(t, config.ripple_params)

    return {
        "theta_phase": theta_phase,
        "ripple": ripple,
        "speed": behavior.speed_cm_s,
        "acceleration": behavior.acceleration_cm_s2,
        "head_direction": behavior.head_direction_rad,
        "distance_to_wall": behavior.distance_to_wall_cm,
        "boundary_angle": behavior.allocentric_boundary_angle_rad,
        "position": behavior.position_cm,
    }


def place_feature(position: np.ndarray, center: np.ndarray, sigma_cm: float) -> float:
    """Gaussian place field: exp(-||p - p0||^2 / (2 sigma^2))."""
    pos = np.asarray(position, dtype=float).reshape(-1)
    ctr = np.asarray(center, dtype=float).reshape(-1)
    dist_sq = float(np.sum((pos - ctr) ** 2))
    return float(np.exp(-dist_sq / (2 * sigma_cm ** 2)))


def head_direction_feature(theta: np.ndarray, theta_pref: float, kappa: float) -> np.ndarray:
    """von Mises head-direction tuning, normalized to [0, 1]."""
    raw = np.exp(kappa * np.cos(theta - theta_pref))
    norm = np.exp(kappa) / i0(kappa)
    return raw / norm


def speed_feature(speed: np.ndarray, thresh: float) -> np.ndarray:
    """Thresholded speed gate."""
    return np.clip(speed - thresh, 0, None) / max(1.0, 30.0 - thresh)


def acceleration_feature(accel: np.ndarray) -> np.ndarray:
    """Normalized acceleration magnitude."""
    return np.clip(np.abs(accel) / 50.0, 0, 1)


def boundary_feature(dist_to_wall: np.ndarray, sigma_cm: float = 15.0) -> np.ndarray:
    """Boundary proximity: high near walls."""
    return np.exp(-dist_to_wall ** 2 / (2 * sigma_cm ** 2))


def theta_modulation(theta_phase: np.ndarray, weight: float) -> np.ndarray:
    """Multiplicative theta phase modulation centered at 1."""
    return 1.0 + weight * np.cos(theta_phase)


def _generate_ripple_envelope(time_s: np.ndarray, params: dict) -> np.ndarray:
    """Generate sparse CA1 ripple bursts as a rate multiplier envelope."""
    ripple = np.zeros_like(time_s)
    duration = params["duration_s"]
    n_expected = int(params["rate_per_min"] * (time_s[-1] / 60.0))
    rng = np.random.default_rng(int(time_s[0] * 1000) % (2**31))

    for _ in range(n_expected):
        t0 = rng.uniform(0, time_s[-1] - duration)
        mask = (time_s >= t0) & (time_s < t0 + duration)
        phase = (time_s[mask] - t0) / duration
        ripple[mask] = np.maximum(ripple[mask], np.sin(np.pi * phase))

    return ripple
