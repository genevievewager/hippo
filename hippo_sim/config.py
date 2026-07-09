"""Simulation configuration and parameter definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np

SESSION_DURATION_S = 600.0
BEHAVIOR_DT = 0.05  # 20 Hz rate updates
SAMPLE_RATE_HZ = 30_000
WAVEFORM_PRE_SAMPLES = 20
WAVEFORM_POST_SAMPLES = 60

ARENA_SIZE_CM = 100.0
THIGMOTAXIS = 0.8

N_CHANNELS = 384
SITE_PITCH_UM = 20.0

REGION_SEGMENTS: List[Dict] = [
    {"region": "CA1", "layer": "oriens", "z_start": 0, "z_end": 200, "density": 2.0},
    {"region": "CA1", "layer": "pyramidal", "z_start": 200, "z_end": 400, "density": 8.0},
    {"region": "CA1", "layer": "radiatum", "z_start": 400, "z_end": 600, "density": 5.0},
    {"region": "CA2", "layer": "pyramidal", "z_start": 600, "z_end": 800, "density": 6.0},
    {"region": "CA3", "layer": "pyramidal", "z_start": 800, "z_end": 1400, "density": 7.0},
    {"region": "DG", "layer": "granule", "z_start": 1400, "z_end": 1800, "density": 10.0},
    {"region": "DG", "layer": "hilus", "z_start": 1800, "z_end": 2000, "density": 3.0},
]

CELL_TYPES = ["CA1_pyr", "CA2_pyr", "CA3_pyr", "DG_granule"]

REGION_TO_CELL_TYPE = {
    ("CA1", "oriens"): "CA1_pyr",
    ("CA1", "pyramidal"): "CA1_pyr",
    ("CA1", "radiatum"): "CA1_pyr",
    ("CA2", "pyramidal"): "CA2_pyr",
    ("CA3", "pyramidal"): "CA3_pyr",
    ("DG", "granule"): "DG_granule",
    ("DG", "hilus"): "DG_granule",
}

RATE_PARAMS: Dict[str, Dict] = {
    "CA1_pyr": {
        "tau_s": 0.05,
        "baseline_hz": 0.5,
        "amplitude_hz": 12.0,
        "sigma_place_cm": 10.0,
        "w_hd": 0.4,
        "kappa_hd": 2.0,
        "w_speed": 0.3,
        "speed_thresh_cm_s": 2.0,
        "w_theta": 0.25,
        "theta_freq_hz": 8.0,
        "w_ripple": 0.5,
        "w_boundary": 0.2,
    },
    "CA2_pyr": {
        "tau_s": 0.05,
        "baseline_hz": 0.4,
        "amplitude_hz": 10.0,
        "sigma_place_cm": 8.0,
        "w_hd": 0.35,
        "kappa_hd": 2.5,
        "w_speed": 0.25,
        "speed_thresh_cm_s": 2.0,
        "w_theta": 0.15,
        "theta_freq_hz": 8.0,
        "w_ripple": 0.1,
        "w_boundary": 0.15,
    },
    "CA3_pyr": {
        "tau_s": 0.06,
        "baseline_hz": 0.3,
        "amplitude_hz": 8.0,
        "sigma_place_cm": 12.0,
        "w_hd": 0.2,
        "kappa_hd": 1.5,
        "w_speed": 0.2,
        "speed_thresh_cm_s": 2.0,
        "w_theta": 0.1,
        "theta_freq_hz": 8.0,
        "w_ripple": 0.6,
        "w_boundary": 0.1,
        "w_recurrent": 0.15,
    },
    "DG_granule": {
        "tau_s": 0.08,
        "baseline_hz": 0.05,
        "amplitude_hz": 15.0,
        "sigma_place_cm": 6.0,
        "w_hd": 0.1,
        "kappa_hd": 1.0,
        "w_speed": 0.5,
        "speed_thresh_cm_s": 3.0,
        "w_theta": 0.05,
        "theta_freq_hz": 8.0,
        "w_ripple": 0.0,
        "w_boundary": 0.3,
        "sparsity_thresh": 0.3,
    },
}

DRIFT_PARAMS = {
    "place_drift_sd_cm_per_min": 0.1,
    "place_drift_update_s": 30.0,
    "state_drift_tau_s": 120.0,
    "state_drift_sigma": 0.15,
    "gain_drift_tau_s": 180.0,
    "gain_drift_sigma": 0.1,
}

RIPPLE_PARAMS = {
    "rate_per_min": 2.0,
    "duration_s": 0.08,
    "amplitude_hz": 80.0,
}

RECORDING_PARAMS = {
    "template_span_channels": (3, 10),
    "amplitude_range_uv": (20.0, 200.0),
    "noise_std_uv": 15.0,
    "noise_correlation": 0.3,
    "motion_amplitude_drift_per_min": 0.15,
    "overlap_collision_prob": 0.08,
    "burst_noise_prob": 0.002,
    "burst_noise_amplitude_uv": 80.0,
}

SORTING_PARAMS = {
    "detection_threshold_uv": 25.0,
    "match_corr_thresh": 0.85,
    "miss_rate": 0.12,
    "false_positive_rate": 0.005,
    "jitter_ms": 0.3,
    "merge_prob": 0.04,
    "contamination_rate": 0.08,
}


@dataclass
class SimConfig:
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    seed: int = 42
    session_duration_s: float = SESSION_DURATION_S
    behavior_dt: float = BEHAVIOR_DT
    sample_rate_hz: int = SAMPLE_RATE_HZ
    n_channels: int = N_CHANNELS
    arena_size_cm: float = ARENA_SIZE_CM
    thigmotaxis: float = THIGMOTAXIS
    region_segments: List[Dict] = field(default_factory=lambda: list(REGION_SEGMENTS))
    rate_params: Dict[str, Dict] = field(default_factory=lambda: dict(RATE_PARAMS))
    drift_params: Dict = field(default_factory=lambda: dict(DRIFT_PARAMS))
    ripple_params: Dict = field(default_factory=lambda: dict(RIPPLE_PARAMS))
    recording_params: Dict = field(default_factory=lambda: dict(RECORDING_PARAMS))
    sorting_params: Dict = field(default_factory=lambda: dict(SORTING_PARAMS))

    @property
    def n_behavior_steps(self) -> int:
        return int(self.session_duration_s / self.behavior_dt)

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(self.n_behavior_steps) * self.behavior_dt


def channels_for_segment(z_start_um: float, z_end_um: float, pitch_um: float = SITE_PITCH_UM) -> np.ndarray:
    depths = np.arange(N_CHANNELS) * pitch_um
    mask = (depths >= z_start_um) & (depths < z_end_um)
    return np.where(mask)[0]
