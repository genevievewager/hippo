"""F × E × D × W × C configuration objects for live deployment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class RuntimeState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTED = "CONNECTED"
    MODEL_LOADED = "MODEL_LOADED"
    INVALID_INPUT = "INVALID_INPUT"
    READY = "READY"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


@dataclass
class DeployableConfiguration:
    """Complete deployable F × E × D × W × C selection for one behavioral target."""

    target: str
    feature_set: str  # F (neural feature family / mode)
    embedding_type: str  # E
    decoder_name: str  # D
    decode_window_s: float  # W
    extras: dict[str, Any] = field(default_factory=dict)  # C

    # Paths / artifacts (may be relative to experiment or absolute)
    model_path: str | None = None
    manifold_transform_path: str | None = None
    neural_extractor_path: str | None = None
    comparison_dir: str | None = None
    experiment_dir: str | None = None

    # Selection / scoring metadata
    metric_name: str | None = None
    metric_value: float | None = None
    metric_direction: str | None = None  # "lower" | "higher"
    selection_policy: str = "shortest_near_optimal"
    spike_source: str = "sorted"
    realtime_compatible: bool = True
    deployable: bool = True
    remapped_from_offline: bool = False
    update_dt_s: float = 0.025
    manifold_n_components: int | None = None
    training_run_id: str | None = None

    @property
    def F(self) -> str:
        return self.feature_set

    @property
    def E(self) -> str:
        return self.embedding_type

    @property
    def D(self) -> str:
        return self.decoder_name

    @property
    def W(self) -> float:
        return float(self.decode_window_s)

    @property
    def C(self) -> dict[str, Any]:
        base = {
            "update_dt_s": self.update_dt_s,
            "manifold_n_components": self.manifold_n_components,
            "spike_source": self.spike_source,
            "selection_policy": self.selection_policy,
            "realtime_compatible": self.realtime_compatible,
            "model_path": self.model_path,
            "manifold_transform_path": self.manifold_transform_path,
            "neural_extractor_path": self.neural_extractor_path,
            "training_run_id": self.training_run_id,
        }
        return {**base, **(self.extras or {})}

    def summary_lines(self) -> list[str]:
        c_bits = []
        if self.manifold_n_components is not None:
            c_bits.append(f"k={self.manifold_n_components}")
        c_bits.append(f"update_dt={self.update_dt_s:.3f}s")
        c_bits.append(f"spike_source={self.spike_source}")
        if self.metric_name is not None and self.metric_value is not None:
            c_bits.append(f"{self.metric_name}={self.metric_value:.4g}")
        return [
            f"F: {self.F}",
            f"E: {self.E}",
            f"D: {self.D}",
            f"W: {self.W:.3f} s",
            f"C: {', '.join(c_bits)}",
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DeployableConfiguration":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})
