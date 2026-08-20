"""Extended decoding targets for decoder comparison experiments.

Behavioral variables are latent from the Neuropixels recording perspective.
Labels come from simulator ground truth (offline evaluation only).

Exteroceptive/spatial: position, spatial context, distance to wall, wall bin.
Proprioceptive: speed, acceleration, head direction (circular), movement state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime.train_decoder import (
    MOVEMENT_THRESHOLDS,
    _resolve_behavior_columns,
    classify_movement_state,
    classify_spatial_context,
    infer_arena_bounds,
)

WALL_DISTANCE_NEAR_CM = 10.0
WALL_DISTANCE_MIDDLE_CM = 30.0


def distance_to_wall(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Minimum distance from (x, y) to any arena wall (cm)."""
    return np.minimum.reduce([
        x - x_min,
        x_max - x,
        y - y_min,
        y_max - y,
    ])


def classify_wall_distance_bin(
    dist: np.ndarray,
    near_cm: float = WALL_DISTANCE_NEAR_CM,
    middle_cm: float = WALL_DISTANCE_MIDDLE_CM,
) -> np.ndarray:
    """Assign near_wall / middle / center from distance to nearest wall."""
    labels = np.full(len(dist), "center", dtype=object)
    labels[dist < near_cm] = "near_wall"
    labels[(dist >= near_cm) & (dist < middle_cm)] = "middle"
    return labels


def align_extended_behavior_to_decoder_times(
    behavior_df: pd.DataFrame,
    decode_times: np.ndarray,
    summary: dict | None = None,
) -> pd.DataFrame:
    """Align behavior and derive exteroceptive/proprioceptive decoding targets."""
    cols = _resolve_behavior_columns(behavior_df)
    beh_times = behavior_df[cols["time"]].to_numpy()

    nearest_idx = np.searchsorted(beh_times, decode_times, side="left")
    nearest_idx = np.clip(nearest_idx, 0, len(beh_times) - 1)
    prev_idx = np.clip(nearest_idx - 1, 0, len(beh_times) - 1)
    choose_prev = np.abs(beh_times[prev_idx] - decode_times) < np.abs(
        beh_times[nearest_idx] - decode_times
    )
    idx = np.where(choose_prev, prev_idx, nearest_idx)

    x = behavior_df[cols["x"]].to_numpy()[idx]
    y = behavior_df[cols["y"]].to_numpy()[idx]
    speed = behavior_df[cols["speed"]].to_numpy()[idx]
    hd = behavior_df[cols["head_direction"]].to_numpy()[idx]

    x_min, x_max, y_min, y_max = infer_arena_bounds(behavior_df, summary)
    spatial_context = classify_spatial_context(x, y, x_min, x_max, y_min, y_max)
    movement_state = classify_movement_state(speed)
    dist_wall = distance_to_wall(x, y, x_min, x_max, y_min, y_max)
    wall_bin = classify_wall_distance_bin(dist_wall)

    # Acceleration from speed differences on the decoder time grid.
    acceleration = np.zeros_like(speed)
    if len(decode_times) > 1:
        dt = np.diff(decode_times)
        dt = np.maximum(dt, 1e-9)
        acceleration[1:] = np.diff(speed) / dt

    return pd.DataFrame({
        "time": decode_times,
        "x": x,
        "y": y,
        "speed": speed,
        "acceleration": acceleration,
        "head_direction": hd,
        "head_direction_sin": np.sin(hd),
        "head_direction_cos": np.cos(hd),
        "distance_to_wall": dist_wall,
        "spatial_context": spatial_context,
        "movement_state": movement_state,
        "wall_distance_bin": wall_bin,
    })


def circular_error_deg(true_angles: np.ndarray, pred_angles: np.ndarray) -> np.ndarray:
    """Absolute circular error in degrees.

    ``true_angles`` and ``pred_angles`` must be in **radians**. For degree
    inputs use :func:`circular_error_from_degrees`.
    """
    diff = np.arctan2(
        np.sin(pred_angles - true_angles),
        np.cos(pred_angles - true_angles),
    )
    return np.degrees(np.abs(diff))


def circular_error_from_degrees(
    true_deg: np.ndarray,
    pred_deg: np.ndarray,
) -> np.ndarray:
    """Absolute circular error (degrees) from degree-valued angles.

    Uses the shortest arc; 359° vs 1° is 2°, not 358°.
    """
    true_rad = np.deg2rad(np.asarray(true_deg, dtype=float))
    pred_rad = np.deg2rad(np.asarray(pred_deg, dtype=float))
    return circular_error_deg(true_rad, pred_rad)


def angles_from_sin_cos(sin_vals: np.ndarray, cos_vals: np.ndarray) -> np.ndarray:
    return np.arctan2(sin_vals, cos_vals)
