"""RatInABox locomotion in a square open field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hippo_sim.config import SimConfig


@dataclass
class BehaviorTrace:
    time_s: np.ndarray
    position_cm: np.ndarray  # (T, 2)
    velocity_cm_s: np.ndarray  # (T, 2)
    speed_cm_s: np.ndarray  # (T,)
    acceleration_cm_s2: np.ndarray  # (T,)
    head_direction_rad: np.ndarray  # (T,)
    distance_to_wall_cm: np.ndarray  # (T,)
    allocentric_boundary_angle_rad: np.ndarray  # (T,)


def simulate_behavior(config: SimConfig) -> BehaviorTrace:
    """Simulate 10-minute open-field locomotion with RatInABox."""
    from ratinabox.Environment import Environment
    from ratinabox.Agent import Agent

    params = {
        "boundary_conditions": "contained",
        "walls": [],
        "aspect": 1.0,
    }
    Env = Environment(params=params)
    Env.add_wall(np.array([[0, 0], [config.arena_size_cm, 0]]))
    Env.add_wall(np.array([[config.arena_size_cm, 0], [config.arena_size_cm, config.arena_size_cm]]))
    Env.add_wall(np.array([[config.arena_size_cm, config.arena_size_cm], [0, config.arena_size_cm]]))
    Env.add_wall(np.array([[0, config.arena_size_cm], [0, 0]]))

    Ag = Agent(Env, params={"thigmotaxis": config.thigmotaxis})

    n_steps = config.n_behavior_steps
    dt = config.behavior_dt
    time_s = config.time_axis

    position = np.zeros((n_steps, 2))
    velocity = np.zeros((n_steps, 2))
    speed = np.zeros(n_steps)
    acceleration = np.zeros(n_steps)
    head_direction = np.zeros(n_steps)
    dist_wall = np.zeros(n_steps)
    boundary_angle = np.zeros(n_steps)

    prev_speed = 0.0
    for t in range(n_steps):
        Ag.update(dt=dt)
        pos = np.asarray(Ag.pos, dtype=float).flatten()[:2]
        vel = np.asarray(Ag.velocity, dtype=float).flatten()[:2]
        spd = float(np.linalg.norm(vel))

        position[t] = pos
        velocity[t] = vel
        speed[t] = spd
        acceleration[t] = (spd - prev_speed) / dt
        prev_speed = spd

        if spd > 0.1:
            head_direction[t] = np.arctan2(vel[1], vel[0])
        elif t > 0:
            head_direction[t] = head_direction[t - 1]

        dist_wall[t], boundary_angle[t] = _distance_and_angle_to_nearest_wall(pos, config.arena_size_cm)

    return BehaviorTrace(
        time_s=time_s,
        position_cm=position,
        velocity_cm_s=velocity,
        speed_cm_s=speed,
        acceleration_cm_s2=acceleration,
        head_direction_rad=head_direction,
        distance_to_wall_cm=dist_wall,
        allocentric_boundary_angle_rad=boundary_angle,
    )


def _distance_and_angle_to_nearest_wall(pos: np.ndarray, arena_size: float) -> tuple[float, float]:
    x, y = pos
    dist_left = x
    dist_right = arena_size - x
    dist_bottom = y
    dist_top = arena_size - y
    dists = [dist_left, dist_right, dist_bottom, dist_top]
    angles = [np.pi, 0.0, 3 * np.pi / 2, np.pi / 2]
    idx = int(np.argmin(dists))
    return float(dists[idx]), float(angles[idx])
