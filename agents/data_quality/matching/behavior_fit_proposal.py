"""Behaviour matching proposal — list params and fit targets; do not fit yet."""

from __future__ import annotations

# What ``hippo_sim.behavior.simulate_behavior`` currently wires from SimConfig.
SIMULATE_BEHAVIOR_WIRED = {
    "config.arena_size_cm": {
        "default": 100.0,
        "maps_to": "Environment.params['scale'] = arena_m; aspect fixed 1.0",
        "controls": "square arena side (cm)",
    },
    "config.behavior_dt": {
        "default": 0.05,
        "maps_to": "Agent.params['dt']",
        "controls": "integration step (s)",
    },
    "config.thigmotaxis": {
        "default": 0.8,
        "maps_to": "Agent.params['thigmotaxis'] (RatInABox default 0.5)",
        "controls": "wall-following tendency",
    },
    "config.session_duration_s": {
        "default": 600.0,
        "maps_to": "n_behavior_steps = duration / behavior_dt",
        "controls": "trace length",
    },
    "config.seed": {
        "default": 42,
        "maps_to": "np.random.default_rng(seed) — currently UNUSED in the update loop",
        "controls": "intended reproducibility (Agent uses its own RNG today)",
    },
    "Environment fixed": {
        "default": "dimensionality=2D, boundary_conditions=solid, aspect=1.0",
        "maps_to": "Environment(params=...)",
        "controls": "square solid box only",
    },
}

# RatInABox Agent.default_params NOT currently passed by simulate_behavior.
# Fitting these requires a new wrapper (allowed) that passes them into Agent(...).
AGENT_PUBLIC_UNWIRED = {
    "speed_mean": {"default": 0.08, "units": "m/s", "controls": "mean locomotion speed"},
    "speed_std": {"default": 0.08, "units": "m/s", "controls": "speed variability"},
    "speed_coherence_time": {"default": 0.7, "units": "s", "controls": "speed OU timescale"},
    "rotational_velocity_std": {
        "default": 2.094...,
        "units": "rad/s",
        "controls": "turning amplitude",
    },
    "rotational_velocity_coherence_time": {
        "default": 0.08,
        "units": "s",
        "controls": "turning OU timescale",
    },
    "head_direction_smoothing_timescale": {
        "default": 0.15,
        "units": "s",
        "controls": "HD smoothing",
    },
    "wall_repel_distance": {"default": 0.1, "units": "m", "controls": "wall repulsion range"},
    "wall_repel_strength": {"default": 1.0, "units": "au", "controls": "wall repulsion gain"},
}

# Lab statistics to match (behaviour only — no neural params).
PROPOSED_FIT_TARGETS = [
    {
        "lab_stat": "speed distribution (cm/s)",
        "how": "histogram / CDF of speed_cm_s on retained valid moving samples (>2 cm/s)",
        "fit_params": ["speed_mean", "speed_std", "speed_coherence_time"],
        "notes": "convert cm/s ↔ Agent m/s by /100",
    },
    {
        "lab_stat": "wall-distance distribution (cm)",
        "how": "distance to boundary polygon (lab) vs square walls (sim); compare CDFs",
        "fit_params": ["thigmotaxis", "wall_repel_distance", "wall_repel_strength"],
        "notes": "shape confound remains (polygon vs square)",
    },
    {
        "lab_stat": "occupancy entropy",
        "how": "entropy of 2D occupancy on equal-area square / lab bbox grids",
        "fit_params": ["thigmotaxis", "rotational_velocity_std", "speed_mean"],
        "notes": "secondary objective after speed + wall-distance",
    },
]
