"""Causal real-time decoder replay.

Real-time means causal replay: the decoder uses spikes from the recent past
[t - window, t) but never future spikes or future behavior during decoding.
True behavior is retained only for offline evaluation.
"""

from __future__ import annotations

from typing import Any

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
        update_dt: float = 0.050,
        feature_type: str = "counts",
        primary_model: Any | None = None,
        primary_target: str | None = None,
        feature_transformer: Any | None = None,
    ):
        self.models = models
        self.unit_ids = np.asarray(unit_ids, dtype=int)
        self.decode_window = decode_window
        self.update_dt = update_dt
        self.feature_type = feature_type
        self.primary_model = primary_model
        self.primary_target = primary_target
        # Optional fitted manifold / identity transform (frozen at replay time).
        self.feature_transformer = feature_transformer

    def _features(self, spikes_df: pd.DataFrame, t: float) -> np.ndarray:
        counts = count_spikes_in_window(
            spikes_df,
            self.unit_ids,
            t - self.decode_window,
            t,
        ).reshape(1, -1)
        if self.feature_transformer is not None:
            return np.asarray(self.feature_transformer.transform(counts), dtype=float)
        if self.feature_type == "rates":
            return counts / self.decode_window
        return counts

    def decode_at_time(self, spikes_df: pd.DataFrame, t: float) -> dict:
        """
        Use only spikes from [t - decode_window, t).

        Returns decoded position, context, movement state, speed, and confidences.
        """
        feats = self._features(spikes_df, t)

        decoded_xy = self.models.position.predict(feats)[0]
        decoded_speed = float(self.models.speed.predict(feats).ravel()[0])

        decoded_context = self.models.spatial_context.predict(feats)[0]
        decoded_movement = self.models.movement_state.predict(feats)[0]

        context_conf = self._classifier_confidence(
            self.models.spatial_context, feats, decoded_context
        )
        movement_conf = self._classifier_confidence(
            self.models.movement_state, feats, decoded_movement
        )

        result = {
            "time": t,
            "decode_time": t,
            "window_start": t - self.decode_window,
            "window_end": t,
            "decode_window_s": self.decode_window,
            "decoded_x": float(decoded_xy[0]),
            "decoded_y": float(decoded_xy[1]),
            "decoded_spatial_context": str(decoded_context),
            "spatial_context_confidence": context_conf,
            "decoded_movement_state": str(decoded_movement),
            "movement_state_confidence": movement_conf,
            "decoded_speed": decoded_speed,
        }

        # Optional primary-model override for closed-loop target
        if self.primary_model is not None and self.primary_target is not None:
            result.update(self._apply_primary(feats))

        return result

    def _apply_primary(self, feats: np.ndarray) -> dict:
        target = self.primary_target
        model = self.primary_model
        out: dict = {}
        if target == "position":
            xy = np.asarray(model.predict(feats)).reshape(-1)
            out["decoded_x"] = float(xy[0])
            out["decoded_y"] = float(xy[1])
        elif target == "speed":
            out["decoded_speed"] = float(np.asarray(model.predict(feats)).ravel()[0])
        elif target == "spatial_context":
            lab = model.predict(feats)[0]
            out["decoded_spatial_context"] = str(lab)
            out["spatial_context_confidence"] = self._classifier_confidence(model, feats, lab)
        elif target == "movement_state":
            lab = model.predict(feats)[0]
            out["decoded_movement_state"] = str(lab)
            out["movement_state_confidence"] = self._classifier_confidence(model, feats, lab)
        elif target == "wall_distance_bin":
            lab = model.predict(feats)[0]
            out["decoded_wall_distance_bin"] = str(lab)
            out["wall_distance_bin_confidence"] = self._classifier_confidence(model, feats, lab)
        elif target == "distance_to_wall":
            out["decoded_distance_to_wall"] = float(np.asarray(model.predict(feats)).ravel()[0])
        elif target == "head_direction":
            sc = np.asarray(model.predict(feats)).reshape(-1)
            angle = float(np.degrees(np.arctan2(sc[0], sc[1])))
            out["decoded_head_direction_deg"] = angle
        elif target == "acceleration":
            out["decoded_acceleration"] = float(np.asarray(model.predict(feats)).ravel()[0])
        return out

    @staticmethod
    def _classifier_confidence(pipeline, X: np.ndarray, predicted_label: str) -> float:
        model = pipeline
        if hasattr(pipeline, "named_steps"):
            if "clf" in pipeline.named_steps:
                model = pipeline
            elif "model" in pipeline.named_steps:
                model = pipeline
        if not hasattr(pipeline, "predict_proba"):
            return float("nan")
        proba = pipeline.predict_proba(X)[0]
        if hasattr(pipeline, "named_steps"):
            step = pipeline.named_steps.get("clf") or pipeline.named_steps.get("model")
            classes = list(step.classes_) if hasattr(step, "classes_") else list(getattr(pipeline, "classes_", []))
        else:
            classes = list(getattr(pipeline, "classes_", []))
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
        # searchsorted-based window counts require monotonically increasing times.
        time_col = "time" if "time" in spikes_df.columns else "spike_time_s"
        times = spikes_df[time_col].to_numpy()
        if times.size and np.any(times[:-1] > times[1:]):
            spikes_df = spikes_df.sort_values(time_col, kind="mergesort")

        rows = []
        for _, beh in aligned_behavior.iterrows():
            t = float(beh["time"])
            decoded = self.decode_at_time(spikes_df, t)

            true_x = float(beh["x"])
            true_y = float(beh["y"])
            pos_err = float(np.hypot(decoded["decoded_x"] - true_x, decoded["decoded_y"] - true_y))

            row = {
                "time": t,
                "decode_time": decoded["decode_time"],
                "window_start": decoded["window_start"],
                "window_end": decoded["window_end"],
                "decode_window_s": decoded["decode_window_s"],
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
            }
            if "decoded_wall_distance_bin" in decoded:
                row["decoded_wall_distance_bin"] = decoded["decoded_wall_distance_bin"]
                row["wall_distance_bin_confidence"] = decoded.get(
                    "wall_distance_bin_confidence", float("nan")
                )
            if "true_wall_distance_bin" in beh:
                row["true_wall_distance_bin"] = beh["true_wall_distance_bin"]
            if "decoded_distance_to_wall" in decoded:
                row["decoded_distance_to_wall"] = decoded["decoded_distance_to_wall"]
            if "distance_to_wall" in beh:
                row["true_distance_to_wall"] = float(beh["distance_to_wall"])
            if "decoded_head_direction_deg" in decoded:
                row["decoded_head_direction_deg"] = decoded["decoded_head_direction_deg"]
            if "head_direction" in beh:
                row["true_head_direction_deg"] = float(np.degrees(beh["head_direction"]))
            if "decoded_acceleration" in decoded:
                row["decoded_acceleration"] = decoded["decoded_acceleration"]
            if "acceleration" in beh:
                row["true_acceleration"] = float(beh["acceleration"])
            rows.append(row)

        return pd.DataFrame(rows)
