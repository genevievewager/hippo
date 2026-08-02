"""Closed-loop trigger / control rule search (dimension C)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from realtime.closed_loop_controller import evaluate_closed_loop


@dataclass(frozen=True)
class TriggerRule:
    """One closed-loop trigger configuration."""

    name: str
    closed_loop_target: str
    trigger_context: str | None = None
    trigger_confidence: float | None = None
    trigger_wall_bin: str | None = None
    trigger_distance_lt_cm: float | None = None
    trigger_speed_gt_cm_s: float | None = None
    trigger_movement: str | None = None
    trigger_hd_center_deg: float | None = None
    trigger_hd_width_deg: float | None = None
    trigger_zone: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger_rule": self.name,
            "closed_loop_target": self.closed_loop_target,
            "trigger_context": self.trigger_context,
            "trigger_confidence": self.trigger_confidence,
            "trigger_wall_bin": self.trigger_wall_bin,
            "trigger_distance_lt_cm": self.trigger_distance_lt_cm,
            "trigger_speed_gt_cm_s": self.trigger_speed_gt_cm_s,
            "trigger_movement": self.trigger_movement,
            "trigger_hd_center_deg": self.trigger_hd_center_deg,
            "trigger_hd_width_deg": self.trigger_hd_width_deg,
            "trigger_zone": self.trigger_zone,
        }


def default_trigger_rules(
    *,
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_wall_bin: str | None = "near_wall",
    trigger_distance_lt_cm: float | None = 10.0,
    trigger_speed_gt_cm_s: float | None = 10.0,
    trigger_movement: str | None = "fast",
    trigger_hd_center_deg: float | None = 90.0,
    trigger_hd_width_deg: float | None = 30.0,
) -> list[TriggerRule]:
    """Build the default C search set from CLI-like parameters."""
    rules: list[TriggerRule] = []
    if trigger_context is not None:
        rules.append(TriggerRule(
            name=f"spatial_context=={trigger_context}@conf{trigger_confidence:.2f}",
            closed_loop_target="spatial_context",
            trigger_context=trigger_context,
            trigger_confidence=trigger_confidence,
        ))
    if trigger_wall_bin is not None:
        rules.append(TriggerRule(
            name=f"wall_distance_bin=={trigger_wall_bin}@conf{trigger_confidence:.2f}",
            closed_loop_target="wall_distance_bin",
            trigger_wall_bin=trigger_wall_bin,
            trigger_confidence=trigger_confidence,
        ))
    if trigger_distance_lt_cm is not None:
        rules.append(TriggerRule(
            name=f"distance_to_wall<{trigger_distance_lt_cm:g}cm",
            closed_loop_target="distance_to_wall",
            trigger_distance_lt_cm=trigger_distance_lt_cm,
        ))
    if trigger_speed_gt_cm_s is not None:
        rules.append(TriggerRule(
            name=f"speed>{trigger_speed_gt_cm_s:g}cm_s",
            closed_loop_target="speed",
            trigger_speed_gt_cm_s=trigger_speed_gt_cm_s,
        ))
    if trigger_movement is not None:
        rules.append(TriggerRule(
            name=f"movement_state=={trigger_movement}@conf{trigger_confidence:.2f}",
            closed_loop_target="movement_state",
            trigger_movement=trigger_movement,
            trigger_confidence=trigger_confidence,
        ))
    if trigger_hd_center_deg is not None and trigger_hd_width_deg is not None:
        rules.append(TriggerRule(
            name=f"head_direction@{trigger_hd_center_deg:g}±{trigger_hd_width_deg:g}deg",
            closed_loop_target="head_direction",
            trigger_hd_center_deg=trigger_hd_center_deg,
            trigger_hd_width_deg=trigger_hd_width_deg,
        ))
    # Position-in-zone uses the same zone label as spatial context wall/corner/center.
    if trigger_context is not None:
        rules.append(TriggerRule(
            name=f"position_zone=={trigger_context}",
            closed_loop_target="position",
            trigger_zone=trigger_context,
        ))
    return rules


def build_decoded_frame_for_triggers(
    *,
    target: str,
    decode_times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    behavior_test: pd.DataFrame,
    confidence: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Assemble the column schema expected by ``evaluate_closed_loop``."""
    n = len(decode_times)
    conf = (
        np.asarray(confidence, dtype=float)
        if confidence is not None
        else np.ones(n, dtype=float)
    )
    df = pd.DataFrame({"time": np.asarray(decode_times, dtype=float)})

    if target == "spatial_context":
        df["decoded_spatial_context"] = y_pred
        df["true_spatial_context"] = y_true
        df["spatial_context_confidence"] = conf
    elif target == "movement_state":
        df["decoded_movement_state"] = y_pred
        df["true_movement_state"] = y_true
        df["movement_state_confidence"] = conf
    elif target == "wall_distance_bin":
        df["decoded_wall_distance_bin"] = y_pred
        df["true_wall_distance_bin"] = y_true
        df["wall_distance_bin_confidence"] = conf
    elif target == "distance_to_wall":
        df["decoded_distance_to_wall"] = np.asarray(y_pred, dtype=float).ravel()
        df["true_distance_to_wall"] = np.asarray(y_true, dtype=float).ravel()
    elif target == "speed":
        df["decoded_speed"] = np.asarray(y_pred, dtype=float).ravel()
        df["true_speed"] = np.asarray(y_true, dtype=float).ravel()
    elif target == "position":
        yp = np.asarray(y_pred, dtype=float)
        yt = np.asarray(y_true, dtype=float)
        df["decoded_x"] = yp[:, 0]
        df["decoded_y"] = yp[:, 1]
        df["true_x"] = yt[:, 0]
        df["true_y"] = yt[:, 1]
    elif target == "head_direction":
        from realtime.decoding_targets import angles_from_sin_cos

        yp = np.asarray(y_pred, dtype=float)
        yt = np.asarray(y_true, dtype=float)
        if yp.ndim == 2 and yp.shape[1] >= 2:
            df["decoded_head_direction_deg"] = angles_from_sin_cos(yp[:, 0], yp[:, 1])
            df["true_head_direction_deg"] = angles_from_sin_cos(yt[:, 0], yt[:, 1])
        else:
            df["decoded_head_direction_deg"] = yp.ravel()
            df["true_head_direction_deg"] = yt.ravel()
    else:
        # Unused target for this rule set.
        pass

    # Fill optional true columns from behavior when present.
    for col, alias in (
        ("spatial_context", "true_spatial_context"),
        ("movement_state", "true_movement_state"),
        ("wall_distance_bin", "true_wall_distance_bin"),
        ("distance_to_wall", "true_distance_to_wall"),
        ("speed", "true_speed"),
    ):
        if alias not in df.columns and col in behavior_test.columns:
            df[alias] = behavior_test[col].to_numpy()
    if "true_x" not in df.columns and "x" in behavior_test.columns:
        df["true_x"] = behavior_test["x"].to_numpy()
        df["true_y"] = behavior_test["y"].to_numpy()
    return df


def summarize_trigger_events(
    events: pd.DataFrame,
    *,
    decode_times: np.ndarray,
    rule: TriggerRule,
    true_positive_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute precision/recall/FPR/FNR/rate/latency for one trigger rule."""
    times = np.asarray(decode_times, dtype=float)
    duration_min = max((times[-1] - times[0]) / 60.0, 1e-9) if len(times) else 1e-9
    n = len(times)

    fired_times = (
        events["time"].to_numpy(dtype=float) if not events.empty else np.asarray([], dtype=float)
    )
    n_triggers = int(len(fired_times))
    correct = (
        events["correct_trigger"].to_numpy(dtype=bool)
        if not events.empty and "correct_trigger" in events.columns
        else np.asarray([], dtype=bool)
    )
    tp = int(correct.sum()) if len(correct) else 0
    fp = int((~correct).sum()) if len(correct) else 0

    # Approximate ground-truth positive frames for recall / FNR.
    if true_positive_mask is None:
        # Fall back: treat correct triggers as the only TP evidence.
        n_true_pos = max(tp, 1) if n_triggers else 0
    else:
        n_true_pos = int(np.asarray(true_positive_mask, dtype=bool).sum())

    fn = max(n_true_pos - tp, 0)
    tn = max(n - tp - fp - fn, 0)

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    fnr = fn / (fn + tp) if (fn + tp) else float("nan")

    # Mean latency: delay from first true-positive frame in each bout — approximate
    # as mean gap between consecutive triggers when available.
    if n_triggers >= 2:
        mean_latency = float(np.mean(np.diff(np.sort(fired_times))))
    elif n_triggers == 1:
        mean_latency = 0.0
    else:
        mean_latency = float("nan")

    return {
        **rule.to_dict(),
        "trigger_precision": float(precision) if precision == precision else float("nan"),
        "trigger_recall": float(recall) if recall == recall else float("nan"),
        "trigger_false_positive_rate": float(fpr) if fpr == fpr else float("nan"),
        "trigger_false_negative_rate": float(fnr) if fnr == fnr else float("nan"),
        "trigger_rate_per_min": float(n_triggers / duration_min),
        "mean_trigger_latency_s": mean_latency,
        "n_triggers": n_triggers,
    }


def evaluate_trigger_rule_on_predictions(
    rule: TriggerRule,
    *,
    decode_times: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    behavior_test: pd.DataFrame,
    confidence: np.ndarray | None = None,
    arena_bounds: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    if rule.closed_loop_target not in {
        "spatial_context", "movement_state", "wall_distance_bin",
        "distance_to_wall", "speed", "position", "head_direction",
    }:
        return {**rule.to_dict(), "n_triggers": 0}

    decoded = build_decoded_frame_for_triggers(
        target=rule.closed_loop_target,
        decode_times=decode_times,
        y_true=y_true,
        y_pred=y_pred,
        behavior_test=behavior_test,
        confidence=confidence,
    )
    events = evaluate_closed_loop(
        decoded,
        closed_loop_target=rule.closed_loop_target,
        trigger_context=rule.trigger_context,
        trigger_confidence=rule.trigger_confidence if rule.trigger_confidence is not None else 0.0,
        trigger_movement=rule.trigger_movement,
        trigger_wall_bin=rule.trigger_wall_bin,
        trigger_distance_lt_cm=rule.trigger_distance_lt_cm,
        trigger_speed_gt_cm_s=rule.trigger_speed_gt_cm_s,
        trigger_zone=rule.trigger_zone,
        trigger_hd_center_deg=rule.trigger_hd_center_deg,
        trigger_hd_width_deg=rule.trigger_hd_width_deg,
        arena_bounds=arena_bounds,
    )
    true_mask = _true_positive_mask(rule, behavior_test, y_true)
    return summarize_trigger_events(
        events, decode_times=decode_times, rule=rule, true_positive_mask=true_mask,
    )


def _true_positive_mask(
    rule: TriggerRule,
    behavior_test: pd.DataFrame,
    y_true: np.ndarray,
) -> np.ndarray:
    y = np.asarray(y_true)
    if rule.closed_loop_target == "spatial_context" and rule.trigger_context is not None:
        if rule.trigger_context == "wall":
            return np.isin(y, ["wall", "corner"])
        return y == rule.trigger_context
    if rule.closed_loop_target == "wall_distance_bin" and rule.trigger_wall_bin is not None:
        return y == rule.trigger_wall_bin
    if rule.closed_loop_target == "movement_state" and rule.trigger_movement is not None:
        return y == rule.trigger_movement
    if rule.closed_loop_target == "distance_to_wall" and rule.trigger_distance_lt_cm is not None:
        return np.asarray(y, dtype=float).ravel() < rule.trigger_distance_lt_cm
    if rule.closed_loop_target == "speed" and rule.trigger_speed_gt_cm_s is not None:
        return np.asarray(y, dtype=float).ravel() > rule.trigger_speed_gt_cm_s
    if rule.closed_loop_target == "position" and rule.trigger_zone is not None:
        if "spatial_context" in behavior_test.columns:
            sc = behavior_test["spatial_context"].to_numpy()
            if rule.trigger_zone == "wall":
                return np.isin(sc, ["wall", "corner"])
            return sc == rule.trigger_zone
    return np.zeros(len(behavior_test), dtype=bool)
