"""Single-target live / replay decoder runtime.

Loads a frozen deployment bundle and runs causal F → E → D inference on a
``SpikeStream``. Acquisition, inference, and UI refresh are intentionally
decoupled — call ``step()`` at the neural update cadence; the UI may poll
``latest_prediction`` / history more slowly.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from realtime.deployment_bundle import DeploymentBundle, load_deployment_bundle
from realtime.live.config import RuntimeState
from realtime.live.session_logger import SessionLogger
from realtime.live.spike_buffer import CausalSpikeBuffer
from realtime.live.spike_stream import ReplaySpikeStream, SpikeStream
from realtime.live.unit_mapping import UnitMappingReport, map_units
from realtime.spike_binner import count_spikes_in_window


class DeploymentMode(str, Enum):
    """Runtime interpretation of predictions."""

    PIPELINE_TEST = "pipeline_test"
    VALIDATED = "validated"


@dataclass
class PredictionRecord:
    timestamp: float
    target: str
    prediction: Any
    model_id: str
    inference_latency_ms: float
    loop_latency_ms: float
    overrun: bool
    n_spikes_in_window: int
    flags: dict[str, Any] = field(default_factory=dict)

    def to_log_row(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "timestamp": self.timestamp,
            "target": self.target,
            "model_id": self.model_id,
            "inference_latency_ms": self.inference_latency_ms,
            "loop_latency_ms": self.loop_latency_ms,
            "overrun": self.overrun,
            "n_spikes_in_window": self.n_spikes_in_window,
        }
        pred = self.prediction
        if isinstance(pred, dict):
            row.update(pred)
        else:
            row["prediction"] = pred
        for k, v in self.flags.items():
            row[f"flag_{k}"] = v
        return row


class LiveDecoder:
    """Causal single-target inference from a deployment bundle + spike stream."""

    def __init__(self, bundle: DeploymentBundle):
        self.bundle = bundle
        self.config = bundle.config
        self.target = bundle.config.target
        self.decode_window_s = float(bundle.decode_window_s)
        self.update_dt_s = float(bundle.update_dt_s)
        self.expected_unit_ids = list(bundle.unit_ids)
        self.decoder = bundle.decoder
        self.embedding = bundle.embedding
        self.neural_extractor = bundle.neural_extractor
        self.feature_type = str(
            (bundle.feature_config or {}).get("feature_mode")
            or bundle.config.extras.get("feature_mode")
            or bundle.config.feature_set
            or "counts"
        )
        self.model_id = (
            bundle.metadata.get("training_run_id")
            or bundle.config.training_run_id
            or bundle.path.name
        )

        self._stream: SpikeStream | None = None
        self._buffer: CausalSpikeBuffer | None = None
        self._mapping: UnitMappingReport | None = None
        self._state = RuntimeState.DISCONNECTED
        self._error: str | None = None
        self._mode = DeploymentMode.PIPELINE_TEST
        self._pipeline_test_override = False
        self._allow_missing_units = False

        self._t_cursor: float | None = None
        self._n_updates = 0
        self._n_dropped = 0
        self._latencies_ms: deque[float] = deque(maxlen=500)
        self._loop_latencies_ms: deque[float] = deque(maxlen=500)
        self._latest: PredictionRecord | None = None
        self._history: deque[PredictionRecord] = deque(maxlen=5000)
        self._logger: SessionLogger | None = None
        self._prev_counts: np.ndarray | None = None
        self._is_dynamic = bool(
            self.embedding is not None
            and hasattr(self.embedding, "step")
            and hasattr(self.embedding, "reset_state")
            and getattr(self.embedding, "supports_realtime", True)
        )
        self._spike_rate_hz: float = 0.0
        self._last_ingest_n: int = 0

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_bundle(cls, path: Path | str) -> "LiveDecoder":
        return cls(load_deployment_bundle(path))

    # ------------------------------------------------------------------ properties
    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def error(self) -> str | None:
        return self._error

    @property
    def mode(self) -> DeploymentMode:
        return self._mode

    @property
    def unit_mapping(self) -> UnitMappingReport | None:
        return self._mapping

    @property
    def latest_prediction(self) -> PredictionRecord | None:
        return self._latest

    @property
    def history(self) -> list[PredictionRecord]:
        return list(self._history)

    @property
    def is_simulation_trained(self) -> bool:
        return self.bundle.is_simulation_trained()

    @property
    def samples_processed(self) -> int:
        return self._n_updates

    @property
    def dropped_updates(self) -> int:
        return self._n_dropped

    def mean_inference_latency_ms(self) -> float | None:
        if not self._latencies_ms:
            return None
        return float(np.mean(self._latencies_ms))

    def mean_loop_latency_ms(self) -> float | None:
        if not self._loop_latencies_ms:
            return None
        return float(np.mean(self._loop_latencies_ms))

    # ------------------------------------------------------------------ modes
    def set_pipeline_test_override(self, enabled: bool) -> None:
        """Allow start despite imperfect unit mapping (engineering tests only)."""
        self._pipeline_test_override = bool(enabled)
        self._allow_missing_units = bool(enabled)
        self._refresh_ready_state()

    def resolve_deployment_mode(self) -> DeploymentMode:
        """Validated only for non-simulation bundles with compatible units."""
        meta = self.bundle.metadata
        validated_ok = (
            not meta.get("simulation_trained", True)
            and meta.get("validated_for_session")
            and self._mapping is not None
            and self._mapping.exact_match
        )
        self._mode = (
            DeploymentMode.VALIDATED if validated_ok else DeploymentMode.PIPELINE_TEST
        )
        return self._mode

    # ------------------------------------------------------------------ connect
    def connect(self, stream: SpikeStream) -> None:
        self._error = None
        try:
            if not stream.connected:
                stream.connect()
            self._stream = stream
            self._state = RuntimeState.CONNECTED
            if self.decoder is not None:
                self._state = RuntimeState.MODEL_LOADED
            self._validate_units()
            self._refresh_ready_state()
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._state = RuntimeState.ERROR
            raise

    def disconnect(self) -> None:
        if self._stream is not None:
            try:
                self._stream.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._stream = None
        self._buffer = None
        self._mapping = None
        if self._state == RuntimeState.RUNNING:
            self._state = RuntimeState.STOPPED
        else:
            self._state = RuntimeState.DISCONNECTED

    def _validate_units(self) -> None:
        if self._stream is None:
            return
        live_ids = self._stream.list_unit_ids()
        self._mapping = map_units(self.expected_unit_ids, live_ids)
        # Feature buffer always uses training expected order.
        history = max(self.decode_window_s * 4.0, 2.0)
        self._buffer = CausalSpikeBuffer(self.expected_unit_ids, history_s=history)
        self.resolve_deployment_mode()

    def _refresh_ready_state(self) -> None:
        if self._state in (RuntimeState.RUNNING, RuntimeState.ERROR):
            return
        if self._stream is None or not self._stream.connected:
            self._state = RuntimeState.DISCONNECTED
            return
        if self.decoder is None:
            self._state = RuntimeState.CONNECTED
            return
        mapping = self._mapping
        if mapping is None:
            self._state = RuntimeState.MODEL_LOADED
            return
        if mapping.n_missing and not self._allow_missing_units:
            self._state = RuntimeState.INVALID_INPUT
            return
        self._state = RuntimeState.READY

    def can_start(self) -> bool:
        return self._state in (RuntimeState.READY, RuntimeState.STOPPED) or (
            self._state == RuntimeState.INVALID_INPUT and self._pipeline_test_override
        )

    # ------------------------------------------------------------------ session
    def start(
        self,
        *,
        session_logger: SessionLogger | None = None,
        t0: float | None = None,
    ) -> None:
        if not self.can_start():
            raise RuntimeError(
                f"Cannot start from state {self._state.value}; "
                "connect, load model, and validate unit mapping first "
                "(or enable Pipeline Test override)."
            )
        if self._pipeline_test_override and self._state == RuntimeState.INVALID_INPUT:
            self._state = RuntimeState.READY
        self._logger = session_logger
        if self._logger is not None:
            self._logger.write_json("deployment_config.json", self.config.to_dict())
            if self._mapping is not None:
                self._logger.write_json("unit_mapping.json", self._mapping.to_dict())
            self._logger.write_json(
                "metadata.json",
                {
                    "mode": self._mode.value,
                    "model_id": self.model_id,
                    "bundle_path": str(self.bundle.path),
                    "simulation_trained": self.is_simulation_trained,
                    "pipeline_test_override": self._pipeline_test_override,
                },
            )
            self._logger.log_event("start", mode=self._mode.value)

        if self._is_dynamic and self.embedding is not None:
            self.embedding.reset_state()
        self._prev_counts = None
        self._n_updates = 0
        self._n_dropped = 0
        self._latencies_ms.clear()
        self._loop_latencies_ms.clear()
        self._history.clear()
        self._latest = None

        if isinstance(self._stream, ReplaySpikeStream):
            if t0 is None:
                t0 = float(self._stream.t_start) + self.decode_window_s
            self._stream.seek(max(0.0, t0 - self.decode_window_s))
        if t0 is None:
            t0 = self.decode_window_s
        self._t_cursor = float(t0)
        self._state = RuntimeState.RUNNING

    def stop(self) -> None:
        if self._logger is not None:
            self._logger.log_event("stop", n_updates=self._n_updates, dropped=self._n_dropped)
            self._logger.close()
            self._logger = None
        if self._state == RuntimeState.RUNNING:
            self._state = RuntimeState.STOPPED

    # ------------------------------------------------------------------ features
    def _features_from_counts(self, counts_1d: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts_1d, dtype=float).reshape(1, -1)
        if self.neural_extractor is not None:
            # Prefer extractor when present (matches offline neural-feature path).
            spikes_df = self._buffer.as_dataframe() if self._buffer is not None else pd.DataFrame()
            t = float(self._t_cursor or 0.0)
            result = self.neural_extractor.extract_at(
                spikes_df, t, prev_counts=self._prev_counts,
            )
            base = np.asarray(result.feature_vector, dtype=float)
            self._prev_counts = counts.copy()
        elif self.feature_type in ("rates",):
            base = counts / max(self.decode_window_s, 1e-9)
        else:
            base = counts

        if self.embedding is None:
            return base
        if self._is_dynamic:
            z = self.embedding.step(np.asarray(base, dtype=float).ravel())
            return np.asarray(z, dtype=float).reshape(1, -1)
        if hasattr(self.embedding, "transform_one"):
            z = self.embedding.transform_one(np.asarray(base, dtype=float).ravel())
            return np.asarray(z, dtype=float).reshape(1, -1)
        return np.asarray(self.embedding.transform(base), dtype=float)

    def _predict_from_features(self, feats: np.ndarray) -> dict[str, Any]:
        model = self.decoder
        target = self.target
        out: dict[str, Any] = {"target": target}
        if target == "position":
            xy = np.asarray(model.predict(feats)).reshape(-1)
            out["decoded_x"] = float(xy[0])
            out["decoded_y"] = float(xy[1] if xy.size > 1 else float("nan"))
            out["prediction"] = {"x": out["decoded_x"], "y": out["decoded_y"]}
        elif target == "speed":
            out["decoded_speed"] = float(np.asarray(model.predict(feats)).ravel()[0])
            out["prediction"] = out["decoded_speed"]
        elif target == "acceleration":
            out["decoded_acceleration"] = float(np.asarray(model.predict(feats)).ravel()[0])
            out["prediction"] = out["decoded_acceleration"]
        elif target == "distance_to_wall":
            out["decoded_distance_to_wall"] = float(
                np.asarray(model.predict(feats)).ravel()[0]
            )
            out["prediction"] = out["decoded_distance_to_wall"]
        elif target == "head_direction":
            sc = np.asarray(model.predict(feats)).reshape(-1)
            if sc.size >= 2:
                angle = float(np.degrees(np.arctan2(sc[0], sc[1])))
            else:
                angle = float(sc[0])
            out["decoded_head_direction_deg"] = angle
            out["prediction"] = angle
        elif target in ("spatial_context", "movement_state", "wall_distance_bin"):
            lab = model.predict(feats)[0]
            key = f"decoded_{target}"
            out[key] = str(lab)
            out["prediction"] = str(lab)
            out["predicted_class"] = str(lab)
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(feats)[0]
                    classes = list(
                        getattr(
                            model.named_steps.get("clf", model)
                            if hasattr(model, "named_steps")
                            else model,
                            "classes_",
                            [],
                        )
                    )
                    if not classes and hasattr(model, "classes_"):
                        classes = list(model.classes_)
                    out["class_probabilities"] = {
                        str(c): float(p) for c, p in zip(classes, proba)
                    }
                    if str(lab) in out["class_probabilities"]:
                        out["confidence"] = out["class_probabilities"][str(lab)]
                except Exception:  # noqa: BLE001
                    pass
        else:
            pred = model.predict(feats)
            arr = np.asarray(pred).ravel()
            out["prediction"] = float(arr[0]) if arr.size == 1 else arr.tolist()
        return out

    # ------------------------------------------------------------------ step
    def step(self, *, up_to_time: float | None = None) -> PredictionRecord | None:
        """Advance one decoder update. Returns prediction or None if not running."""
        if self._state != RuntimeState.RUNNING:
            return None
        if self._stream is None or self._buffer is None or self._t_cursor is None:
            self._state = RuntimeState.ERROR
            self._error = "Stream/buffer not initialized"
            return None

        loop0 = time.perf_counter()
        t = float(up_to_time) if up_to_time is not None else float(self._t_cursor)

        # Ingest spikes that arrive before t (causal).
        try:
            new_spikes = self._stream.get_new_spikes(up_to_time=t)
        except Exception as exc:  # noqa: BLE001
            self._error = str(exc)
            self._state = RuntimeState.ERROR
            if self._logger:
                self._logger.log_event("error", message=str(exc))
            return None

        self._last_ingest_n = 0 if new_spikes is None else int(len(new_spikes))
        if new_spikes is not None and not new_spikes.empty:
            self._buffer.extend_dataframe(new_spikes)
            span = max(self.update_dt_s, 1e-6)
            self._spike_rate_hz = self._last_ingest_n / span

        t_inf0 = time.perf_counter()
        counts = self._buffer.counts_at(t, self.decode_window_s)
        feats = self._features_from_counts(counts)
        pred_dict = self._predict_from_features(feats)
        inference_ms = (time.perf_counter() - t_inf0) * 1000.0
        loop_ms = (time.perf_counter() - loop0) * 1000.0
        budget_ms = self.update_dt_s * 1000.0
        overrun = loop_ms > budget_ms
        if overrun:
            self._n_dropped += 1

        record = PredictionRecord(
            timestamp=t,
            target=self.target,
            prediction=pred_dict.get("prediction", pred_dict),
            model_id=str(self.model_id),
            inference_latency_ms=float(inference_ms),
            loop_latency_ms=float(loop_ms),
            overrun=overrun,
            n_spikes_in_window=int(np.sum(counts)),
            flags={
                "mode": self._mode.value,
                "pipeline_test_override": self._pipeline_test_override,
                "overrun": overrun,
                **{k: v for k, v in pred_dict.items() if k != "prediction"},
            },
        )
        # Flatten common decoded_* into flags for logging convenience already in pred_dict.
        self._latest = record
        self._history.append(record)
        self._latencies_ms.append(inference_ms)
        self._loop_latencies_ms.append(loop_ms)
        self._n_updates += 1

        if self._logger is not None:
            self._logger.log_prediction(record.to_log_row())
            self._logger.log_runtime(
                {
                    "timestamp": t,
                    "inference_latency_ms": inference_ms,
                    "loop_latency_ms": loop_ms,
                    "overrun": overrun,
                    "n_spikes_in_window": int(np.sum(counts)),
                    "ingest_n": self._last_ingest_n,
                    "spike_rate_hz": self._spike_rate_hz,
                    "n_updates": self._n_updates,
                    "dropped_updates": self._n_dropped,
                }
            )
            if overrun:
                self._logger.log_event(
                    "overrun",
                    loop_ms=loop_ms,
                    budget_ms=budget_ms,
                    t=t,
                )

        # Advance virtual clock for next step when caller does not supply times.
        if up_to_time is None:
            self._t_cursor = t + self.update_dt_s
        else:
            self._t_cursor = t + self.update_dt_s
        return record

    def run_replay_for(
        self,
        duration_s: float | None = None,
        *,
        max_steps: int | None = None,
        flush_every: int = 50,
    ) -> list[PredictionRecord]:
        """Drive replay stream for a duration / step budget (offline validation)."""
        if self._state != RuntimeState.RUNNING:
            raise RuntimeError("start() before run_replay_for()")
        out: list[PredictionRecord] = []
        t_end = None
        if isinstance(self._stream, ReplaySpikeStream):
            t_end = float(self._stream.t_end)
        if duration_s is not None and self._t_cursor is not None:
            t_end = (
                min(t_end, self._t_cursor + duration_s)
                if t_end is not None
                else self._t_cursor + duration_s
            )
        steps = 0
        while True:
            if max_steps is not None and steps >= max_steps:
                break
            if self._t_cursor is None:
                break
            if t_end is not None and self._t_cursor > t_end:
                break
            rec = self.step(up_to_time=self._t_cursor)
            if rec is not None:
                out.append(rec)
            steps += 1
            if self._logger is not None and steps % flush_every == 0:
                self._logger.flush()
            if self._state != RuntimeState.RUNNING:
                break
        if self._logger is not None:
            self._logger.flush()
        return out

    def status_dict(self) -> dict[str, Any]:
        m = self._mapping
        return {
            "state": self._state.value,
            "mode": self._mode.value,
            "target": self.target,
            "update_dt_s": self.update_dt_s,
            "decode_window_s": self.decode_window_s,
            "model_id": self.model_id,
            "simulation_trained": self.is_simulation_trained,
            "samples_processed": self._n_updates,
            "dropped_updates": self._n_dropped,
            "inference_latency_ms": self.mean_inference_latency_ms(),
            "mean_loop_latency_ms": self.mean_loop_latency_ms(),
            "spike_rate_hz": self._spike_rate_hz,
            "source": self._stream.source_name if self._stream else None,
            "units": {
                "expected": m.n_expected if m else len(self.expected_unit_ids),
                "mapped": m.n_mapped if m else 0,
                "missing": m.n_missing if m else 0,
                "unexpected": m.n_unexpected if m else 0,
            },
            "error": self._error,
            "pipeline_test_override": self._pipeline_test_override,
        }

    # ------------------------------------------------------------------ helpers for tests
    def counts_at_time_from_dataframe(
        self, spikes_df: pd.DataFrame, t: float
    ) -> np.ndarray:
        """Reference path: same half-open window via spike_binner (no buffer)."""
        return count_spikes_in_window(
            spikes_df, self.expected_unit_ids, t - self.decode_window_s, t,
        )


def default_ui_refresh_hz() -> float:
    """UI refresh is slower than neural inference cadence."""
    return 5.0


InputSource = Literal["replay", "open_ephys"]
