"""Causal real-time decoder replay.

Real-time means causal replay: the decoder uses spikes from the recent past
[t - window, t) but never future spikes or future behavior during decoding.
True behavior is retained only for offline evaluation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from realtime.latency_profiler import LatencySample
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
        neural_feature_extractor: Any | None = None,
    ):
        self.models = models
        self.unit_ids = np.asarray(unit_ids, dtype=int)
        self.decode_window = decode_window
        self.update_dt = update_dt
        self.feature_type = feature_type
        self.primary_model = primary_model
        self.primary_target = primary_target
        # Optional fitted manifold / identity / dynamic-latent transform (frozen at replay).
        self.feature_transformer = feature_transformer
        # Optional neural feature extractor (upstream of manifold).
        self.neural_feature_extractor = neural_feature_extractor
        self._prev_counts: np.ndarray | None = None
        # Dynamic latent models expose step()/reset_state(); track for latency split.
        self._is_dynamic = bool(
            feature_transformer is not None
            and hasattr(feature_transformer, "step")
            and hasattr(feature_transformer, "reset_state")
            and getattr(feature_transformer, "supports_realtime", True)
        )

    def _count_window(self, spikes_df: pd.DataFrame, t: float) -> np.ndarray:
        """Causal half-open counts from [t - W, t)."""
        return count_spikes_in_window(
            spikes_df,
            self.unit_ids,
            t - self.decode_window,
            t,
        ).reshape(1, -1)

    def _window_stats(self, counts: np.ndarray) -> dict[str, float | int]:
        c = np.asarray(counts, dtype=float).ravel()
        return {
            "update_dt_s": float(self.update_dt),
            "n_spikes_in_window": int(np.sum(c)),
            "n_active_units_in_window": int(np.sum(c > 0)),
        }

    def _apply_embedding(self, base: np.ndarray) -> np.ndarray:
        """Apply frozen embedding / dynamic latent step to a feature row [1, n]."""
        if self.feature_transformer is None:
            return base
        if self._is_dynamic:
            z = self.feature_transformer.step(np.asarray(base, dtype=float).ravel())
            return np.asarray(z, dtype=float).reshape(1, -1)
        tr = self.feature_transformer
        if hasattr(tr, "transform_one") and np.asarray(base).ndim == 2 and base.shape[0] == 1:
            z = tr.transform_one(np.asarray(base).ravel())
            return np.asarray(z, dtype=float).reshape(1, -1)
        return np.asarray(tr.transform(base), dtype=float)

    def _features(self, spikes_df: pd.DataFrame, t: float) -> tuple[np.ndarray, np.ndarray]:
        if self.neural_feature_extractor is not None:
            result = self.neural_feature_extractor.extract_at(
                spikes_df, t, prev_counts=self._prev_counts,
            )
            base = np.asarray(result.feature_vector, dtype=float)
            # Track count slice when present for window stats.
            if result.feature_names and result.feature_names[0].startswith("count_u"):
                n_units = len(self.unit_ids)
                counts = base[:, :n_units] if base.shape[1] >= n_units else base
            else:
                counts = self._count_window(spikes_df, t)
            self._prev_counts = counts.copy()
            feats = self._apply_embedding(base)
            return feats, counts

        counts = self._count_window(spikes_df, t)
        if self.feature_transformer is not None:
            feats = self._apply_embedding(counts)
        elif self.feature_type == "rates":
            feats = counts / self.decode_window
        else:
            feats = counts
        return feats, counts

    def _features_profiled(
        self, spikes_df: pd.DataFrame, t: float
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        stages: dict[str, float] = {}
        t0 = time.perf_counter()
        counts = self._count_window(spikes_df, t)
        stages["spike_binning"] = (time.perf_counter() - t0) * 1000.0
        stages["feature_construction"] = stages["spike_binning"]

        # Split identity/rate feature prep vs frozen manifold / dynamic latent transform.
        t1 = time.perf_counter()
        if self.feature_type == "rates":
            base = counts / self.decode_window
        else:
            base = counts
        stages["feature_transform"] = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter_ns()
        if self.feature_transformer is not None:
            feats = self._apply_embedding(base)
            elapsed = (time.perf_counter_ns() - t2) / 1e6
            stages["manifold_transform"] = elapsed
            lat = getattr(self.feature_transformer, "last_stage_latencies_ms_", None)
            if isinstance(lat, dict):
                stages["feature_scaling"] = float(lat.get("feature_scaling_ms", 0.0))
                stages["diffusion_nystrom_transform"] = float(
                    lat.get("diffusion_nystrom_transform_ms", elapsed)
                )
            if self._is_dynamic:
                stages["latent_state_update"] = elapsed
        else:
            feats = base
            stages["manifold_transform"] = 0.0
            stages["latent_state_update"] = 0.0
            stages["feature_scaling"] = 0.0
        return feats, counts, stages

    def decode_at_time(self, spikes_df: pd.DataFrame, t: float) -> dict:
        """
        Use only spikes from [t - decode_window, t).

        Returns decoded position, context, movement state, speed, and confidences.
        """
        feats, counts = self._features(spikes_df, t)
        result = self._decode_from_features(feats, t)
        result.update(self._window_stats(counts))
        ood = getattr(self.feature_transformer, "last_ood_", None)
        if isinstance(ood, dict):
            result["ood_flag"] = bool(ood.get("ood_flag", False))
            result["nearest_landmark_distance"] = ood.get("nearest_landmark_distance")
            result["sigma_x"] = ood.get("sigma_x")
            result["max_kernel_weight"] = ood.get("max_kernel_weight")
            result["kernel_entropy"] = ood.get("kernel_entropy")
            result["effective_n_landmarks"] = ood.get("effective_n_landmarks")
        return result

    def decode_at_time_profiled(
        self, spikes_df: pd.DataFrame, t: float
    ) -> tuple[dict, LatencySample]:
        """Like ``decode_at_time`` but also return per-stage latency (ms)."""
        t_total0 = time.perf_counter()
        feats, counts, stages = self._features_profiled(spikes_df, t)

        t0 = time.perf_counter()
        decoded_xy = self.models.position.predict(feats)[0]
        stages["decode_position"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        decoded_speed = float(self.models.speed.predict(feats).ravel()[0])
        stages["decode_speed"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        decoded_context = self.models.spatial_context.predict(feats)[0]
        context_conf = self._classifier_confidence(
            self.models.spatial_context, feats, decoded_context
        )
        stages["decode_spatial_context"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        decoded_movement = self.models.movement_state.predict(feats)[0]
        movement_conf = self._classifier_confidence(
            self.models.movement_state, feats, decoded_movement
        )
        stages["decode_movement_state"] = (time.perf_counter() - t0) * 1000.0

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
        result.update(self._window_stats(counts))
        ood = getattr(self.feature_transformer, "last_ood_", None)
        if isinstance(ood, dict):
            result["ood_flag"] = bool(ood.get("ood_flag", False))
            result["nearest_landmark_distance"] = ood.get("nearest_landmark_distance")
            result["sigma_x"] = ood.get("sigma_x")
            result["max_kernel_weight"] = ood.get("max_kernel_weight")
            result["kernel_entropy"] = ood.get("kernel_entropy")
            result["effective_n_landmarks"] = ood.get("effective_n_landmarks")

        if self.primary_model is not None and self.primary_target is not None:
            t0 = time.perf_counter()
            result.update(self._apply_primary(feats))
            stages["decode_primary"] = (time.perf_counter() - t0) * 1000.0
        else:
            stages["decode_primary"] = 0.0

        stages["decoder_inference"] = float(
            stages.get("decode_position", 0.0)
            + stages.get("decode_speed", 0.0)
            + stages.get("decode_spatial_context", 0.0)
            + stages.get("decode_movement_state", 0.0)
            + stages.get("decode_primary", 0.0)
        )
        stages["closed_loop_policy"] = stages.get("closed_loop_policy", 0.0)
        stages["trigger_decision"] = stages["closed_loop_policy"]
        stages["total_update"] = (time.perf_counter() - t_total0) * 1000.0
        stages["total_operation"] = stages["total_update"]
        return result, LatencySample(time_s=t, stages_ms=stages)

    def _decode_from_features(self, feats: np.ndarray, t: float) -> dict:
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
        # Window stats are filled by decode_at_time; keep keys present for helpers.
        result.setdefault("update_dt_s", float(self.update_dt))
        result.setdefault("n_spikes_in_window", None)
        result.setdefault("n_active_units_in_window", None)

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
        *,
        profile_latency: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, list[LatencySample]]:
        """
        Causally decode at each time in aligned_behavior and compare to true state.

        aligned_behavior must contain decoder times and ground-truth labels.
        When ``profile_latency`` is True, also return per-update latency samples.
        """
        # searchsorted-based window counts require monotonically increasing times.
        time_col = "time" if "time" in spikes_df.columns else "spike_time_s"
        times = spikes_df[time_col].to_numpy()
        if times.size and np.any(times[:-1] > times[1:]):
            spikes_df = spikes_df.sort_values(time_col, kind="mergesort")

        # Dynamic latent filters must start from the initial belief at replay start.
        if self._is_dynamic:
            self.feature_transformer.reset_state()
        self._prev_counts = None

        # Dense per-frame profiling around RF predicts is expensive; subsample.
        max_profile_updates = 400
        if profile_latency and len(aligned_behavior) > max_profile_updates:
            profile_idx = set(
                np.linspace(0, len(aligned_behavior) - 1, max_profile_updates).astype(int)
            )
        else:
            profile_idx = None

        rows = []
        samples: list[LatencySample] = []
        for i, (_, beh) in enumerate(aligned_behavior.iterrows()):
            t = float(beh["time"])
            do_profile = profile_latency and (profile_idx is None or i in profile_idx)
            if do_profile:
                decoded, sample = self.decode_at_time_profiled(spikes_df, t)
                samples.append(sample)
            else:
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
                "update_dt_s": decoded.get("update_dt_s", float(self.update_dt)),
                "n_spikes_in_window": decoded.get("n_spikes_in_window"),
                "n_active_units_in_window": decoded.get("n_active_units_in_window"),
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
            # Persist dynamic latent state when available (from last feature call).
            if self._is_dynamic:
                z = getattr(self.feature_transformer, "model_", None)
                if z is not None and getattr(z, "_mu", None) is not None:
                    for di, val in enumerate(np.asarray(z._mu).ravel()):
                        row[f"z{di + 1}"] = float(val)
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
            if "ood_flag" in decoded:
                row["ood_flag"] = decoded["ood_flag"]
                row["nearest_landmark_distance"] = decoded.get("nearest_landmark_distance")
                row["sigma_x"] = decoded.get("sigma_x")
                row["max_kernel_weight"] = decoded.get("max_kernel_weight")
                row["kernel_entropy"] = decoded.get("kernel_entropy")
                row["effective_n_landmarks"] = decoded.get("effective_n_landmarks")
            rows.append(row)

        decoded_df = pd.DataFrame(rows)
        if profile_latency:
            return decoded_df, samples
        return decoded_df
