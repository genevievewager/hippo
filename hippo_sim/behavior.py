"""RatInABox locomotion in a square open field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.config import SimConfig


@dataclass
class BehaviorTrace: # store the behavior of the rat
    time_s: np.ndarray
    position_cm: np.ndarray  # (T, 2)
    velocity_cm_s: np.ndarray  # (T, 2)
    speed_cm_s: np.ndarray  # (T,)
    acceleration_cm_s2: np.ndarray  # (T,)
    head_direction_rad: np.ndarray  # (T,)
    distance_to_wall_cm: np.ndarray  # (T,)
    allocentric_boundary_angle_rad: np.ndarray  # (T,)


@dataclass
class BehaviorSimulationResult:
    """RatInABox behavior trace plus Environment/Agent for neural backends."""

    trace: BehaviorTrace
    env: object | None = None
    agent: object | None = None


def simulate_behavior(config: SimConfig) -> BehaviorSimulationResult:
    """
    Simulate open-field locomotion with RatInABox.

    RatInABox uses a 2D box with coordinates in environment units.
    Here we use a 1.0 x 1.0 RatInABox box and convert outputs to cm
    so the rest of the simulation can keep using ARENA_SIZE_CM = 100.
    """
    from ratinabox.Environment import Environment
    from ratinabox.Agent import Agent

    rng = np.random.default_rng(config.seed)

    arena_size_cm = float(config.arena_size_cm)
    arena_size_m = arena_size_cm / 100.0

    # RatInABox environment in meters.
    # For ARENA_SIZE_CM = 100, this is a 1 m x 1 m open field.
    env_params = {
        "dimensionality": "2D",
        "boundary_conditions": "solid",
        "scale": arena_size_m,
        "aspect": 1.0,
    }

    env = Environment(params=env_params)

    # RatInABox Agent parameters.
    # thigmotaxis controls wall-following tendency.
    agent_params = {
        "dt": config.behavior_dt,
        "thigmotaxis": config.thigmotaxis,
    }

    agent = Agent(env, params=agent_params)

    # Optional: set reproducible initial position manually.
    # Keep it away from exact boundaries.
    if hasattr(agent, "pos"):
        agent.pos = np.array([arena_size_m / 2.0, arena_size_m / 2.0])

    n_steps = config.n_behavior_steps
    dt = config.behavior_dt
    time_s = config.time_axis

    position_cm = np.zeros((n_steps, 2), dtype=float)
    velocity_cm_s = np.zeros((n_steps, 2), dtype=float)
    speed_cm_s = np.zeros(n_steps, dtype=float)
    acceleration_cm_s2 = np.zeros(n_steps, dtype=float)
    head_direction_rad = np.zeros(n_steps, dtype=float)
    distance_to_wall_cm = np.zeros(n_steps, dtype=float)
    boundary_angle_rad = np.zeros(n_steps, dtype=float)

    prev_speed = 0.0

    for t in range(n_steps):
        agent.update(dt=dt)

        # RatInABox position/velocity are in meters if scale is given in meters.
        pos_m = np.asarray(agent.pos, dtype=float).reshape(-1)[:2]

        if hasattr(agent, "velocity"):
            vel_m_s = np.asarray(agent.velocity, dtype=float).reshape(-1)[:2]
        else:
            # Fallback: finite difference if velocity is not exposed.
            if t == 0:
                vel_m_s = np.zeros(2, dtype=float)
            else:
                prev_pos_m = position_cm[t - 1] / 100.0
                vel_m_s = (pos_m - prev_pos_m) / dt

        pos_cm = pos_m * 100.0
        vel_cm_s = vel_m_s * 100.0
        spd = float(np.linalg.norm(vel_cm_s))

        position_cm[t] = pos_cm
        velocity_cm_s[t] = vel_cm_s
        speed_cm_s[t] = spd
        acceleration_cm_s2[t] = (spd - prev_speed) / dt
        prev_speed = spd

        if spd > 0.1:
            head_direction_rad[t] = np.arctan2(vel_cm_s[1], vel_cm_s[0])
        elif t > 0:
            head_direction_rad[t] = head_direction_rad[t - 1]
        else:
            head_direction_rad[t] = 0.0

        distance_to_wall_cm[t], boundary_angle_rad[t] = (
            _distance_and_angle_to_nearest_wall(pos_cm, arena_size_cm)
        )

    return BehaviorSimulationResult(trace=BehaviorTrace(
        time_s=time_s,
        position_cm=position_cm,
        velocity_cm_s=velocity_cm_s,
        speed_cm_s=speed_cm_s,
        acceleration_cm_s2=acceleration_cm_s2,
        head_direction_rad=head_direction_rad,
        distance_to_wall_cm=distance_to_wall_cm,
        allocentric_boundary_angle_rad=boundary_angle_rad,
    ), env=env, agent=agent)


def _distance_and_angle_to_nearest_wall(
    pos_cm: np.ndarray,
    arena_size_cm: float,
) -> tuple[float, float]:
    """
    Return distance and allocentric angle to the nearest wall.

    Angles point from the animal toward the nearest boundary:
    - right wall: 0
    - top wall: pi/2
    - left wall: pi
    - bottom wall: 3pi/2
    """
    x, y = np.asarray(pos_cm, dtype=float).reshape(-1)[:2]

    dist_left = x
    dist_right = arena_size_cm - x
    dist_bottom = y
    dist_top = arena_size_cm - y

    dists = np.array([dist_left, dist_right, dist_bottom, dist_top], dtype=float)
    angles = np.array([np.pi, 0.0, 3.0 * np.pi / 2.0, np.pi / 2.0], dtype=float)

    idx = int(np.argmin(dists))
    return float(dists[idx]), float(angles[idx])