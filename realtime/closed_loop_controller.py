"""Closed-loop controller driven by decoded latent behavioral state.

The decoder infers behavioral variables from spikes only. Closed-loop events
fire when decoded state crosses configurable thresholds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime.decoding_targets import (
    classify_wall_distance_bin,
    distance_to_wall,
)
from realtime.train_decoder import classify_spatial_context


EVENT_COLUMNS = [
    "time",
    "event_type",
    "closed_loop_target",
    "decoded_spatial_context",
    "spatial_context_confidence",
    "decoded_movement_state",
    "movement_state_confidence",
    "true_spatial_context",
    "true_movement_state",
    "correct_trigger",
]


def evaluate_closed_loop(
    decoded_df: pd.DataFrame,
    closed_loop_target: str | None = "position",
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_movement: str | None = None,
    trigger_wall_bin: str | None = "near_wall",
    trigger_distance_lt_cm: float | None = 10.0,
    trigger_speed_gt_cm_s: float | None = 10.0,
    trigger_zone: str | None = "wall",
    trigger_hd_center_deg: float | None = 90.0,
    trigger_hd_width_deg: float | None = 30.0,
    arena_bounds: tuple[float, float, float, float] | None = None,
) -> pd.DataFrame:
    """
    Generate closed-loop trigger events from decoded state for a target.

    A trigger fires when decoded conditions are met. It is correct when the
    true behavioral state matches the target condition.
    """
    target = closed_loop_target or "position"
    events = []

    for _, row in decoded_df.iterrows():
        fired, event_type, correct = _evaluate_row(
            row,
            target=target,
            trigger_context=trigger_context,
            trigger_confidence=trigger_confidence,
            trigger_movement=trigger_movement,
            trigger_wall_bin=trigger_wall_bin,
            trigger_distance_lt_cm=trigger_distance_lt_cm,
            trigger_speed_gt_cm_s=trigger_speed_gt_cm_s,
            trigger_zone=trigger_zone,
            trigger_hd_center_deg=trigger_hd_center_deg,
            trigger_hd_width_deg=trigger_hd_width_deg,
            arena_bounds=arena_bounds,
        )
        if not fired:
            continue
        events.append({
            "time": row["time"],
            "event_type": event_type,
            "closed_loop_target": target,
            "decoded_spatial_context": row.get("decoded_spatial_context"),
            "spatial_context_confidence": row.get("spatial_context_confidence", float("nan")),
            "decoded_movement_state": row.get("decoded_movement_state"),
            "movement_state_confidence": row.get("movement_state_confidence", float("nan")),
            "true_spatial_context": row.get("true_spatial_context"),
            "true_movement_state": row.get("true_movement_state"),
            "correct_trigger": bool(correct),
        })

    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.DataFrame(events)


def _evaluate_row(
    row: pd.Series,
    target: str,
    trigger_context: str | None,
    trigger_confidence: float,
    trigger_movement: str | None,
    trigger_wall_bin: str | None,
    trigger_distance_lt_cm: float | None,
    trigger_speed_gt_cm_s: float | None,
    trigger_zone: str | None,
    trigger_hd_center_deg: float | None,
    trigger_hd_width_deg: float | None,
    arena_bounds: tuple[float, float, float, float] | None,
) -> tuple[bool, str, bool]:
    if target == "spatial_context":
        return _trigger_spatial_context(row, trigger_context, trigger_confidence)
    if target == "movement_state":
        return _trigger_movement(row, trigger_movement, trigger_confidence)
    if target == "wall_distance_bin":
        return _trigger_wall_bin(row, trigger_wall_bin, trigger_confidence)
    if target == "distance_to_wall":
        return _trigger_distance(row, trigger_distance_lt_cm)
    if target == "speed":
        return _trigger_speed(row, trigger_speed_gt_cm_s)
    if target == "position":
        return _trigger_position_zone(row, trigger_zone, arena_bounds)
    if target == "head_direction":
        return _trigger_head_direction(row, trigger_hd_center_deg, trigger_hd_width_deg)
    # Legacy combined path when target unset and both context/movement provided
    return _trigger_spatial_context(row, trigger_context, trigger_confidence)


def _trigger_spatial_context(
    row: pd.Series,
    trigger_context: str | None,
    trigger_confidence: float,
) -> tuple[bool, str, bool]:
    if trigger_context is None:
        return False, "none", False
    contexts = {trigger_context}
    if trigger_context == "wall":
        contexts.add("corner")
    conf = float(row.get("spatial_context_confidence", 1.0))
    if np.isnan(conf):
        conf = 1.0
    fired = row["decoded_spatial_context"] in contexts and conf >= trigger_confidence
    if trigger_context == "wall":
        correct = row["true_spatial_context"] in {"wall", "corner"}
    else:
        correct = row["true_spatial_context"] == trigger_context
    return fired, f"context_{trigger_context}", correct


def _trigger_movement(
    row: pd.Series,
    trigger_movement: str | None,
    trigger_confidence: float,
) -> tuple[bool, str, bool]:
    if trigger_movement is None or str(trigger_movement).lower() == "none":
        return False, "none", False
    conf = float(row.get("movement_state_confidence", 1.0))
    if np.isnan(conf):
        conf = 1.0
    fired = row["decoded_movement_state"] == trigger_movement and conf >= trigger_confidence
    correct = row["true_movement_state"] == trigger_movement
    return fired, f"movement_{trigger_movement}", correct


def _trigger_wall_bin(
    row: pd.Series,
    trigger_wall_bin: str | None,
    trigger_confidence: float,
) -> tuple[bool, str, bool]:
    if trigger_wall_bin is None:
        return False, "none", False
    decoded = row.get("decoded_wall_distance_bin")
    true = row.get("true_wall_distance_bin")
    conf = float(row.get("wall_distance_bin_confidence", 1.0))
    if np.isnan(conf):
        conf = 1.0
    fired = decoded == trigger_wall_bin and conf >= trigger_confidence
    correct = true == trigger_wall_bin
    return fired, f"wall_bin_{trigger_wall_bin}", correct


def _trigger_distance(
    row: pd.Series,
    trigger_distance_lt_cm: float | None,
) -> tuple[bool, str, bool]:
    if trigger_distance_lt_cm is None:
        return False, "none", False
    decoded = float(row.get("decoded_distance_to_wall", np.nan))
    true = float(row.get("true_distance_to_wall", np.nan))
    if np.isnan(decoded):
        return False, "none", False
    fired = decoded < trigger_distance_lt_cm
    correct = (not np.isnan(true)) and true < trigger_distance_lt_cm
    return fired, f"distance_lt_{trigger_distance_lt_cm}", correct


def _trigger_speed(
    row: pd.Series,
    trigger_speed_gt_cm_s: float | None,
) -> tuple[bool, str, bool]:
    if trigger_speed_gt_cm_s is None:
        return False, "none", False
    decoded = float(row["decoded_speed"])
    true = float(row["true_speed"])
    fired = decoded > trigger_speed_gt_cm_s
    correct = true > trigger_speed_gt_cm_s
    return fired, f"speed_gt_{trigger_speed_gt_cm_s}", correct


def _trigger_position_zone(
    row: pd.Series,
    trigger_zone: str | None,
    arena_bounds: tuple[float, float, float, float] | None,
) -> tuple[bool, str, bool]:
    if trigger_zone is None or arena_bounds is None:
        return False, "none", False
    x_min, x_max, y_min, y_max = arena_bounds
    decoded_zone = classify_spatial_context(
        np.array([row["decoded_x"]]),
        np.array([row["decoded_y"]]),
        x_min, x_max, y_min, y_max,
    )[0]
    true_zone = classify_spatial_context(
        np.array([row["true_x"]]),
        np.array([row["true_y"]]),
        x_min, x_max, y_min, y_max,
    )[0]
    zones = {trigger_zone}
    if trigger_zone == "wall":
        zones.add("corner")
    fired = decoded_zone in zones
    correct = true_zone in zones
    return fired, f"zone_{trigger_zone}", correct


def _trigger_head_direction(
    row: pd.Series,
    center_deg: float | None,
    width_deg: float | None,
) -> tuple[bool, str, bool]:
    if center_deg is None or width_deg is None:
        return False, "none", False
    decoded = float(row.get("decoded_head_direction_deg", np.nan))
    true = float(row.get("true_head_direction_deg", np.nan))
    if np.isnan(decoded):
        return False, "none", False
    fired = _circular_within(decoded, center_deg, width_deg)
    correct = (not np.isnan(true)) and _circular_within(true, center_deg, width_deg)
    return fired, f"hd_{center_deg}_w{width_deg}", correct


def _circular_within(angle_deg: float, center_deg: float, width_deg: float) -> bool:
    diff = (angle_deg - center_deg + 180.0) % 360.0 - 180.0
    return abs(diff) <= width_deg / 2.0
