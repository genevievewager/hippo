"""Typed pipeline artifacts, provenance, and configuration hashing.

Scientific state lives here — not in Streamlit session state. Downstream
stages inherit the observation window ``W`` from a ``FeatureDataset`` rather
than choosing a new one.

Cache keys include train/test identity whenever a transform is fitted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
LEGACY_SCHEMA_VERSION = 0

# Canonical pipeline stages (dependency order).
STAGE_SIMULATION = "simulation"
STAGE_FEATURES = "features"
STAGE_REPRESENTATION = "representation"
STAGE_DECODER = "decoder"
STAGE_REPLAY = "replay"
STAGE_DEPLOYMENT = "deployment"

PIPELINE_STAGES: tuple[str, ...] = (
    STAGE_SIMULATION,
    STAGE_FEATURES,
    STAGE_REPRESENTATION,
    STAGE_DECODER,
    STAGE_REPLAY,
    STAGE_DEPLOYMENT,
)

# Changing an upstream stage invalidates these downstream stages.
INVALIDATION: dict[str, tuple[str, ...]] = {
    STAGE_SIMULATION: (
        STAGE_FEATURES,
        STAGE_REPRESENTATION,
        STAGE_DECODER,
        STAGE_REPLAY,
        STAGE_DEPLOYMENT,
    ),
    STAGE_FEATURES: (
        STAGE_REPRESENTATION,
        STAGE_DECODER,
        STAGE_REPLAY,
        STAGE_DEPLOYMENT,
    ),
    STAGE_REPRESENTATION: (STAGE_DECODER, STAGE_REPLAY, STAGE_DEPLOYMENT),
    STAGE_DECODER: (STAGE_REPLAY, STAGE_DEPLOYMENT),
    STAGE_REPLAY: (),
    STAGE_DEPLOYMENT: (),
}


class ArtifactOrigin(str, Enum):
    COMPUTED = "computed"
    CACHE = "cache"
    MIGRATED = "migrated"
    UNKNOWN = "unknown"


class ArtifactStatus(str, Enum):
    FRESH = "fresh"
    CACHED = "cached"
    STALE = "stale"
    INCOMPATIBLE = "incompatible"
    MISSING = "missing"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashing (sorted keys, default=str)."""
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))


def config_hash(payload: Any, *, n: int = 16) -> str:
    """Short hex digest of a configuration mapping."""
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    return digest[: max(int(n), 8)]


def window_ms(window_s: float) -> int:
    return int(round(float(window_s) * 1000.0))


def windows_close(a: float, b: float, *, atol: float = 1e-9) -> bool:
    return abs(float(a) - float(b)) <= float(atol)


@dataclass(frozen=True)
class ObservationConfig:
    """Neural observation O(W, F): spikes in [t-W, t) plus feature construction.

    ``W`` is owned here. Downstream representation / decoder / replay inherit
    this configuration; they do not choose a different window in pipeline mode.
    """

    window_s: float
    update_dt: float = 0.050
    feature_set: str = "counts"
    feature_type: str = "counts"
    source_spikes: str = "sorted"
    simulation_run_id: str = ""
    feature_mode: str | None = None
    seed: int | None = None
    train_frac: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "window_s", float(self.window_s))
        object.__setattr__(self, "update_dt", float(self.update_dt))
        if self.window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {self.window_s}")
        if self.update_dt <= 0:
            raise ValueError(f"update_dt must be > 0, got {self.update_dt}")
        src = str(self.source_spikes or "sorted").strip() or "sorted"
        object.__setattr__(self, "source_spikes", src)
        mode = self.feature_mode or str(self.feature_set)
        object.__setattr__(self, "feature_mode", str(mode))

    @property
    def window_ms(self) -> int:
        return window_ms(self.window_s)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        payload = {
            "window_s": round(self.window_s, 9),
            "update_dt": round(self.update_dt, 9),
            "feature_set": self.feature_set,
            "feature_type": self.feature_type,
            "source_spikes": self.source_spikes,
            "simulation_run_id": self.simulation_run_id,
            "feature_mode": self.feature_mode,
        }
        return config_hash(payload)

    def fit_hash(self) -> str:
        """Hash including train/test identity (for fitted F transforms)."""
        payload = {
            **json.loads(canonical_json(self.to_dict())),
            "seed": self.seed,
            "train_frac": self.train_frac,
        }
        return config_hash(payload)

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ObservationConfig":
        data = dict(raw or {})
        migrate_legacy_observation(data)
        window = data.get("window_s", data.get("decode_window_s", data.get("W")))
        if window is None:
            raise ValueError(
                "ObservationConfig requires window_s "
                "(legacy decode_window_s / selected_causal_window_s also accepted)."
            )
        return cls(
            window_s=float(window),
            update_dt=float(data.get("update_dt", data.get("update_dt_s", 0.050))),
            feature_set=str(data.get("feature_set") or data.get("feature_mode") or "counts"),
            feature_type=str(data.get("feature_type") or "counts"),
            source_spikes=str(
                data.get("source_spikes") or data.get("spike_source") or "sorted"
            ),
            simulation_run_id=str(
                data.get("simulation_run_id") or data.get("run_id") or ""
            ),
            feature_mode=data.get("feature_mode"),
            seed=data.get("seed"),
            train_frac=data.get("train_frac"),
        )


def migrate_legacy_observation(data: dict[str, Any]) -> dict[str, Any]:
    """Fill missing observation fields from older decoder / registry schemas."""
    if data.get("window_s") is None:
        for key in (
            "decode_window_s",
            "selected_causal_window_s",
            "best_decode_window_s",
            "recommended_realtime_window_s",
            "W",
        ):
            if data.get(key) is not None:
                data["window_s"] = float(data[key])
                data.setdefault("_migrated_window_from", key)
                break
    if data.get("update_dt") is None and data.get("update_dt_s") is not None:
        data["update_dt"] = float(data["update_dt_s"])
    if data.get("source_spikes") is None and data.get("spike_source") is not None:
        data["source_spikes"] = str(data["spike_source"])
    if data.get("feature_set") is None:
        mode = data.get("selected_feature_mode") or data.get("feature_mode")
        if mode:
            data["feature_set"] = str(mode)
    return data


@dataclass
class ArtifactProvenance:
    run_id: str
    config_hash: str
    origin: ArtifactOrigin = ArtifactOrigin.COMPUTED
    status: ArtifactStatus = ArtifactStatus.FRESH
    created_at: str = field(default_factory=_now_iso)
    source_run_id: str | None = None
    source_config_hash: str | None = None
    schema_version: int = SCHEMA_VERSION
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "origin": self.origin.value if isinstance(self.origin, ArtifactOrigin) else str(self.origin),
            "status": self.status.value if isinstance(self.status, ArtifactStatus) else str(self.status),
            "created_at": self.created_at,
            "source_run_id": self.source_run_id,
            "source_config_hash": self.source_config_hash,
            "schema_version": int(self.schema_version),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ArtifactProvenance":
        data = dict(raw or {})
        origin = data.get("origin") or ArtifactOrigin.UNKNOWN
        status = data.get("status") or ArtifactStatus.FRESH
        try:
            origin_e = ArtifactOrigin(str(origin))
        except ValueError:
            origin_e = ArtifactOrigin.UNKNOWN
        try:
            status_e = ArtifactStatus(str(status))
        except ValueError:
            status_e = ArtifactStatus.FRESH
        schema = data.get("schema_version")
        if schema is None:
            origin_e = ArtifactOrigin.MIGRATED
            schema = LEGACY_SCHEMA_VERSION
        return cls(
            run_id=str(data.get("run_id") or ""),
            config_hash=str(data.get("config_hash") or ""),
            origin=origin_e,
            status=status_e,
            created_at=str(data.get("created_at") or _now_iso()),
            source_run_id=data.get("source_run_id"),
            source_config_hash=data.get("source_config_hash"),
            schema_version=int(schema),
            notes=str(data.get("notes") or ""),
        )


@dataclass
class SimulationResult:
    """On-disk simulation / acquisition artifact."""

    experiment_dir: Path
    run_id: str
    spike_source: str = "sorted"
    session_duration_s: float | None = None
    n_units: int | None = None
    seed: int | None = None
    provenance: ArtifactProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_dir": str(self.experiment_dir),
            "run_id": self.run_id,
            "spike_source": self.spike_source,
            "session_duration_s": self.session_duration_s,
            "n_units": self.n_units,
            "seed": self.seed,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class FeatureDataset:
    """Neural observation matrix ``X`` with inherited window metadata.

    ``X`` may be omitted when the artifact is a disk reference only.
    """

    observation: ObservationConfig
    timestamps: np.ndarray | None = None
    X: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    path: Path | None = None
    provenance: ArtifactProvenance | None = None
    train_mask: np.ndarray | None = None
    test_mask: np.ndarray | None = None

    @property
    def window_s(self) -> float:
        return float(self.observation.window_s)

    @property
    def update_dt(self) -> float:
        return float(self.observation.update_dt)

    @property
    def n_samples(self) -> int:
        if self.X is not None:
            return int(self.X.shape[0])
        if self.timestamps is not None:
            return int(len(self.timestamps))
        return 0

    @property
    def n_features(self) -> int:
        if self.X is not None:
            return int(self.X.shape[1])
        return len(self.feature_names)

    def to_meta(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "FeatureDataset",
            "observation": self.observation.to_dict(),
            "config_hash": self.observation.hash(),
            "fit_hash": self.observation.fit_hash(),
            "feature_names": list(self.feature_names),
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "path": str(self.path) if self.path else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class RepresentationResult:
    """Fitted latent / manifold transform ``E`` applied to a FeatureDataset."""

    representation_name: str
    source_feature_hash: str
    source_feature_run_id: str = ""
    window_s: float = 0.250
    update_dt: float = 0.050
    n_components: int | None = None
    timestamps: np.ndarray | None = None
    Z: np.ndarray | None = None
    fitted_transform: Any = None
    path: Path | None = None
    provenance: ArtifactProvenance | None = None
    target_name: str | None = None  # supervised embeddings only

    def to_meta(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "RepresentationResult",
            "representation_name": self.representation_name,
            "source_feature_hash": self.source_feature_hash,
            "source_feature_run_id": self.source_feature_run_id,
            "window_s": float(self.window_s),
            "update_dt": float(self.update_dt),
            "n_components": self.n_components,
            "target_name": self.target_name,
            "path": str(self.path) if self.path else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class DecoderResult:
    """Fitted decoder plus enough provenance to replay the same observation."""

    target: str
    decoder_name: str
    observation: ObservationConfig
    source_feature_hash: str
    source_representation_hash: str | None = None
    representation_name: str = "identity"
    train_frac: float = 0.70
    seed: int = 42
    metrics: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    fitted_decoder: Any = None
    path: Path | None = None
    provenance: ArtifactProvenance | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "DecoderResult",
            "target": self.target,
            "decoder_name": self.decoder_name,
            "observation": self.observation.to_dict(),
            "window_s": float(self.observation.window_s),
            "update_dt": float(self.observation.update_dt),
            "source_feature_hash": self.source_feature_hash,
            "source_representation_hash": self.source_representation_hash,
            "representation_name": self.representation_name,
            "train_frac": float(self.train_frac),
            "seed": int(self.seed),
            "metrics": dict(self.metrics),
            "hyperparameters": dict(self.hyperparameters),
            "path": str(self.path) if self.path else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


@dataclass
class ReplayResult:
    observation: ObservationConfig
    source_decoder_hash: str | None = None
    path: Path | None = None
    provenance: ArtifactProvenance | None = None

    def to_meta(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "ReplayResult",
            "observation": self.observation.to_dict(),
            "source_decoder_hash": self.source_decoder_hash,
            "path": str(self.path) if self.path else None,
            "provenance": self.provenance.to_dict() if self.provenance else None,
        }


def write_provenance(path: Path, payload: dict[str, Any]) -> Path:
    """Write ``provenance.json`` next to an existing transform directory."""
    loc = Path(path)
    loc.mkdir(parents=True, exist_ok=True)
    dest = loc / "provenance.json"
    dest.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    return dest


def read_provenance(path: Path) -> dict[str, Any]:
    loc = Path(path)
    dest = loc / "provenance.json" if loc.is_dir() else loc
    if not dest.exists():
        return {}
    try:
        raw = json.loads(dest.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def migrate_legacy_decoder_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Normalize an older registry / metrics row without mutating the original."""
    out = dict(entry or {})
    migrate_legacy_observation(out)
    if out.get("window_s") is None:
        nested = out.get("decoder_config")
        if isinstance(nested, dict):
            migrate_legacy_observation(nested)
            if nested.get("window_s") is not None:
                out["window_s"] = nested["window_s"]
    out.setdefault("schema_version", LEGACY_SCHEMA_VERSION)
    if out.get("config_hash") is None and out.get("window_s") is not None:
        try:
            out["config_hash"] = ObservationConfig.from_dict(out).hash()
            out["_migrated"] = True
        except (TypeError, ValueError):
            pass
    return out


def representation_cache_key(
    *,
    feature_hash: str,
    representation_name: str,
    n_components: int | None,
    n_neighbors: int | None,
    seed: int | None,
    target: str | None = None,
) -> str:
    return config_hash(
        {
            "feature_hash": feature_hash,
            "representation_name": representation_name,
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            "seed": seed,
            "target": target,
        }
    )


def with_origin(prov: ArtifactProvenance, origin: ArtifactOrigin) -> ArtifactProvenance:
    return replace(prov, origin=origin)
