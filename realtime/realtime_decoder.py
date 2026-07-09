"""Causal real-time decoder replay.

Real-time means causal replay: the decoder uses spikes from the recent past
[t - window, t) but never future spikes or future behavior during decoding.
True behavior is retained only for offline evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime.spike_binner import count_spikes_in_window
from realtime.train_decoder import TrainedDecoders


class RealTimeDecoder:
    """Replay a trained decoder causally at fixed update intervals."""

    def __init__(
        self,
        models: TrainedDecoders,
        unit_ids: list[int] | np.ndarray,
        decode_window: float = 0.250,
        update_dt: float = 0.025,
    ):
        self.models = models
        self.unit_ids = np.asarray(unit_ids, dtype=int)
        self.decode_window = decode_window
        self.update_dt = update_dt

    def decode_at_time(self, spikes_df: pd.DataFrame, t: float) -> dict:
        """
        Use only spikes from [t - decode_window, t).

        Returns decoded position, context, movement state, speed, and confidences.
        """
        counts = count_spikes_in_window(
            spikes_df,
            self.unit_ids,
            t - self.decode_window,
            t,
        ).reshape(1, -1)

        decoded_xy = self.models.position.predict(counts)[0]
        decoded_speed = float(self.models.speed.predict(counts).ravel()[0])

        decoded_context = self.models.spatial_context.predict(counts)[0]
        decoded_movement = self.models.movement_state.predict(counts)[0]

        context_conf = self._classifier_confidence(
            self.models.spatial_context, counts, decoded_context
        )
        movement_conf = self._classifier_confidence(
            self.models.movement_state, counts, decoded_movement
        )

        return {
            "time": t,
            "decoded_x": float(decoded_xy[0]),
            "decoded_y": float(decoded_xy[1]),
            "decoded_spatial_context": str(decoded_context),
            "spatial_context_confidence": context_conf,
            "decoded_movement_state": str(decoded_movement),
            "movement_state_confidence": movement_conf,
            "decoded_speed": decoded_speed,
        }

    @staticmethod
    def _classifier_confidence(pipeline, X: np.ndarray, predicted_label: str) -> float:
        if not hasattr(pipeline, "predict_proba"):
            return float("nan")
        proba = pipeline.predict_proba(X)[0]
        classes = list(pipeline.named_steps["clf"].classes_)
        if predicted_label not in classes:
            return float("nan")
        return float(proba[classes.index(predicted_label)])

    def replay(
        self,
        spikes_df: pd.DataFrame,
        aligned_behavior: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Causally decode at each time in aligned_behavior and compare to true state.

        aligned_behavior must contain decoder times and ground-truth labels.
        """
        rows = []
        for _, beh in aligned_behavior.iterrows():
            t = float(beh["time"])
            decoded = self.decode_at_time(spikes_df, t)

            true_x = float(beh["x"])
            true_y = float(beh["y"])
            pos_err = float(np.hypot(decoded["decoded_x"] - true_x, decoded["decoded_y"] - true_y))

            rows.append({
                "time": t,
                "decoded_x": decoded["decoded_x"],
                "decoded_y": decoded["decoded_y"],
                "true_x": true_x,
                "true_y": true_y,
                "position_error_cm": pos_err,
                "decoded_spatial_context": decoded["decoded_spatial_context"],
                "true_spatial_context": beh["spatial_context"],
                "spatial_context_confidence": decoded["spatial_context_confidence"],
                "decoded_movement_state": decoded["decoded_movement_state"],
                "true_movement_state": beh["movement_state"],
                "movement_state_confidence": decoded["movement_state_confidence"],
                "decoded_speed": decoded["decoded_speed"],
                "true_speed": float(beh["speed"]),
                "speed_error": float(decoded["decoded_speed"] - beh["speed"]),
            })

        return pd.DataFrame(rows)
