"""Closed-loop controller driven by decoded latent behavioral state.

The decoder infers behavioral variables from spikes only. Closed-loop events
fire when decoded state crosses configurable thresholds.
"""

from __future__ import annotations

import pandas as pd


def evaluate_closed_loop(
    decoded_df: pd.DataFrame,
    trigger_context: str | None = "wall",
    trigger_confidence: float = 0.80,
    trigger_movement: str | None = None,
) -> pd.DataFrame:
    """
    Generate closed-loop trigger events from decoded state.

    A trigger fires when decoded conditions are met. It is correct when the
    true behavioral state matches the target condition.
    """
    events = []

    for _, row in decoded_df.iterrows():
        context_trigger = False
        movement_trigger = False

        if trigger_context is not None:
            contexts = {trigger_context}
            if trigger_context == "wall":
                contexts.add("corner")
            context_trigger = (
                row["decoded_spatial_context"] in contexts
                and row["spatial_context_confidence"] >= trigger_confidence
            )

        if trigger_movement is not None and trigger_movement.lower() != "none":
            movement_trigger = row["decoded_movement_state"] == trigger_movement

        if trigger_context is not None and trigger_movement is not None and trigger_movement.lower() != "none":
            trigger = context_trigger and movement_trigger
            event_type = f"{trigger_context}_{trigger_movement}"
            correct = (
                (row["true_spatial_context"] in {trigger_context, "corner"} if trigger_context == "wall"
                 else row["true_spatial_context"] == trigger_context)
                and row["true_movement_state"] == trigger_movement
            )
        elif trigger_movement is not None and trigger_movement.lower() != "none":
            trigger = movement_trigger
            event_type = f"movement_{trigger_movement}"
            correct = row["true_movement_state"] == trigger_movement
        elif trigger_context is not None:
            trigger = context_trigger
            event_type = f"context_{trigger_context}"
            if trigger_context == "wall":
                correct = row["true_spatial_context"] in {"wall", "corner"}
            else:
                correct = row["true_spatial_context"] == trigger_context
        else:
            trigger = False
            event_type = "none"
            correct = False

        if not trigger:
            continue

        events.append({
            "time": row["time"],
            "event_type": event_type,
            "decoded_spatial_context": row["decoded_spatial_context"],
            "spatial_context_confidence": row["spatial_context_confidence"],
            "decoded_movement_state": row["decoded_movement_state"],
            "movement_state_confidence": row["movement_state_confidence"],
            "true_spatial_context": row["true_spatial_context"],
            "true_movement_state": row["true_movement_state"],
            "correct_trigger": bool(correct),
        })

    return pd.DataFrame(events)
