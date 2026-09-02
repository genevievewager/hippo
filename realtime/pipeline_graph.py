"""Active pipeline run: dependency graph, stale invalidation, disk persistence.

The Streamlit UI reads/writes this object; it is not the source of scientific
state. Persist under ``<experiment>/pipeline_run.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realtime.pipeline_artifacts import (
    INVALIDATION,
    PIPELINE_STAGES,
    SCHEMA_VERSION,
    STAGE_DECODER,
    STAGE_DEPLOYMENT,
    STAGE_FEATURES,
    STAGE_REPLAY,
    STAGE_REPRESENTATION,
    STAGE_SIMULATION,
    ArtifactOrigin,
    ArtifactStatus,
    ObservationConfig,
    config_hash,
    migrate_legacy_observation,
    read_provenance,
    windows_close,
)

PIPELINE_FILENAME = "pipeline_run.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def simulation_run_id_from_dir(experiment_dir: Path) -> str:
    return Path(experiment_dir).name


@dataclass
class StageRecord:
    """One pipeline stage as committed on disk."""

    name: str
    status: ArtifactStatus = ArtifactStatus.MISSING
    run_id: str | None = None
    config_hash: str | None = None
    source_hash: str | None = None
    origin: ArtifactOrigin = ArtifactOrigin.UNKNOWN
    updated_at: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value if isinstance(self.status, ArtifactStatus) else str(self.status),
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "source_hash": self.source_hash,
            "origin": self.origin.value if isinstance(self.origin, ArtifactOrigin) else str(self.origin),
            "updated_at": self.updated_at,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None, *, name: str) -> "StageRecord":
        data = dict(raw or {})
        try:
            status = ArtifactStatus(str(data.get("status") or ArtifactStatus.MISSING.value))
        except ValueError:
            status = ArtifactStatus.MISSING
        try:
            origin = ArtifactOrigin(str(data.get("origin") or ArtifactOrigin.UNKNOWN.value))
        except ValueError:
            origin = ArtifactOrigin.UNKNOWN
        extra = data.get("extra") if isinstance(data.get("extra"), dict) else {}
        # Promote known observation fields into extra for older files.
        for key in ("window_s", "update_dt", "feature_set", "feature_type", "source_spikes"):
            if key in data and key not in extra:
                extra[key] = data[key]
        return cls(
            name=str(data.get("name") or name),
            status=status,
            run_id=data.get("run_id"),
            config_hash=data.get("config_hash"),
            source_hash=data.get("source_hash"),
            origin=origin,
            updated_at=data.get("updated_at"),
            extra=dict(extra),
        )


@dataclass
class PipelineRun:
    """Committed scientific pipeline for one experiment directory."""

    experiment_dir: Path
    simulation_run_id: str
    stages: dict[str, StageRecord] = field(default_factory=dict)
    observation: ObservationConfig | None = None
    schema_version: int = SCHEMA_VERSION
    updated_at: str = field(default_factory=_now_iso)

    def __post_init__(self) -> None:
        for name in PIPELINE_STAGES:
            if name not in self.stages:
                self.stages[name] = StageRecord(name=name)

    def stage(self, name: str) -> StageRecord:
        if name not in self.stages:
            self.stages[name] = StageRecord(name=name)
        return self.stages[name]

    def mark_committed(
        self,
        name: str,
        *,
        run_id: str | None = None,
        config_hash: str | None = None,
        source_hash: str | None = None,
        origin: ArtifactOrigin = ArtifactOrigin.COMPUTED,
        extra: dict[str, Any] | None = None,
        invalidate: bool | None = None,
    ) -> None:
        rec = self.stage(name)
        prev_hash = rec.config_hash
        rec.status = ArtifactStatus.CACHED if origin == ArtifactOrigin.CACHE else ArtifactStatus.FRESH
        rec.run_id = run_id
        rec.config_hash = config_hash
        rec.source_hash = source_hash
        rec.origin = origin
        rec.updated_at = _now_iso()
        if extra:
            rec.extra.update(extra)
        self.updated_at = rec.updated_at
        if invalidate is None:
            invalidate = bool(prev_hash) and config_hash is not None and prev_hash != config_hash
        if invalidate:
            self.invalidate_downstream(name)

    def invalidate_downstream(self, name: str) -> list[str]:
        """Mark downstream stages stale. Does not delete artifacts."""
        touched: list[str] = []
        for ds in INVALIDATION.get(name, ()):
            rec = self.stage(ds)
            if rec.status in (ArtifactStatus.MISSING, ArtifactStatus.INCOMPATIBLE):
                continue
            rec.status = ArtifactStatus.STALE
            rec.updated_at = _now_iso()
            touched.append(ds)
        if touched:
            self.updated_at = _now_iso()
        return touched

    def mark_incompatible(self, name: str, reason: str) -> None:
        rec = self.stage(name)
        rec.status = ArtifactStatus.INCOMPATIBLE
        rec.extra["incompatible_reason"] = reason
        rec.updated_at = _now_iso()
        self.updated_at = rec.updated_at

    def set_observation(self, observation: ObservationConfig, *, run_id: str | None = None) -> list[str]:
        """Commit the active pipeline observation; stale-mark representation+."""
        prev = self.observation
        self.observation = observation
        extra = observation.to_dict()
        extra["window_s"] = float(observation.window_s)
        changed = prev is not None and prev.hash() != observation.hash()
        self.mark_committed(
            STAGE_FEATURES,
            run_id=run_id,
            config_hash=observation.hash(),
            source_hash=self.stage(STAGE_SIMULATION).config_hash,
            extra=extra,
            invalidate=changed,
        )
        return list(INVALIDATION[STAGE_FEATURES]) if changed else []

    def inherited_window_s(self) -> float | None:
        if self.observation is not None:
            return float(self.observation.window_s)
        extra = self.stage(STAGE_FEATURES).extra
        if extra.get("window_s") is not None:
            return float(extra["window_s"])
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "experiment_dir": str(self.experiment_dir),
            "simulation_run_id": self.simulation_run_id,
            "updated_at": self.updated_at,
            "observation": self.observation.to_dict() if self.observation else None,
            "stages": {name: self.stage(name).to_dict() for name in PIPELINE_STAGES},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, experiment_dir: Path | None = None) -> "PipelineRun":
        data = dict(raw or {})
        exp = Path(experiment_dir or data.get("experiment_dir") or ".")
        obs_raw = data.get("observation")
        observation = None
        if isinstance(obs_raw, dict):
            try:
                observation = ObservationConfig.from_dict(obs_raw)
            except (TypeError, ValueError):
                observation = None
        stages: dict[str, StageRecord] = {}
        raw_stages = data.get("stages") or {}
        for name in PIPELINE_STAGES:
            stages[name] = StageRecord.from_dict(raw_stages.get(name), name=name)
        schema = data.get("schema_version")
        if schema is None:
            schema = 0
        return cls(
            experiment_dir=exp,
            simulation_run_id=str(
                data.get("simulation_run_id") or simulation_run_id_from_dir(exp)
            ),
            stages=stages,
            observation=observation,
            schema_version=int(schema),
            updated_at=str(data.get("updated_at") or _now_iso()),
        )


def pipeline_path(experiment_dir: Path) -> Path:
    return Path(experiment_dir) / PIPELINE_FILENAME


def save_pipeline_run(run: PipelineRun) -> Path:
    path = pipeline_path(run.experiment_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    run.updated_at = _now_iso()
    path.write_text(json.dumps(run.to_dict(), indent=2, default=str) + "\n")
    return path


def load_pipeline_run(experiment_dir: Path) -> PipelineRun | None:
    path = pipeline_path(experiment_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return PipelineRun.from_dict(raw, experiment_dir=Path(experiment_dir))


def infer_pipeline_from_disk(experiment_dir: Path) -> PipelineRun:
    """Build a pipeline view from existing files when ``pipeline_run.json`` is missing."""
    exp = Path(experiment_dir)
    run = PipelineRun(
        experiment_dir=exp,
        simulation_run_id=simulation_run_id_from_dir(exp),
    )
    summary_path = exp / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except (OSError, json.JSONDecodeError):
            summary = {}
        seed = summary.get("seed")
        sim_hash = config_hash(
            {
                "run_id": run.simulation_run_id,
                "seed": seed,
                "session_duration_s": summary.get("session_duration_s"),
                "n_units": summary.get("n_units"),
            }
        )
        run.mark_committed(
            STAGE_SIMULATION,
            run_id=run.simulation_run_id,
            config_hash=sim_hash,
            origin=ArtifactOrigin.MIGRATED,
            extra={
                "seed": seed,
                "session_duration_s": summary.get("session_duration_s"),
                "n_units": summary.get("n_units"),
            },
        )
        # mark_committed invalidates downstream; restore missing until we infer them.
        for name in (
            STAGE_FEATURES,
            STAGE_REPRESENTATION,
            STAGE_DECODER,
            STAGE_REPLAY,
            STAGE_DEPLOYMENT,
        ):
            run.stage(name).status = ArtifactStatus.MISSING

    _infer_features(run)
    _infer_representation(run)
    _infer_decoder(run)
    _infer_replay(run)
    _infer_deployment(run)
    return run


def load_or_infer_pipeline(experiment_dir: Path) -> PipelineRun:
    existing = load_pipeline_run(experiment_dir)
    if existing is not None:
        return existing
    return infer_pipeline_from_disk(experiment_dir)


def _infer_features(run: PipelineRun) -> None:
    exp = run.experiment_dir
    feat_root = exp / "decoder_comparison" / "sorted" / "models" / "feature_transforms"
    if not feat_root.exists():
        return
    metas: list[dict[str, Any]] = []
    for meta_path in sorted(feat_root.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(meta, dict):
            continue
        migrate_legacy_observation(meta)
        prov = read_provenance(meta_path.parent)
        meta["_path"] = str(meta_path.parent)
        if prov:
            meta["_provenance"] = prov
        metas.append(meta)
    if not metas:
        return
    # Prefer 250 ms counts if present, else the first cache.
    chosen = None
    for meta in metas:
        w = meta.get("window_s", meta.get("decode_window_s"))
        fs = str(meta.get("feature_set") or "counts")
        if w is not None and windows_close(float(w), 0.250) and fs == "counts":
            chosen = meta
            break
    if chosen is None:
        chosen = metas[0]
    try:
        obs = ObservationConfig.from_dict(
            {
                **chosen,
                "simulation_run_id": run.simulation_run_id,
                "source_spikes": chosen.get("source_spikes")
                or chosen.get("spike_source")
                or "sorted",
            }
        )
    except (TypeError, ValueError):
        return
    origin = ArtifactOrigin.MIGRATED
    run.set_observation(obs, run_id=Path(chosen["_path"]).name)
    rec = run.stage(STAGE_FEATURES)
    rec.origin = origin
    rec.status = ArtifactStatus.CACHED


def _infer_representation(run: PipelineRun) -> None:
    man = (
        run.experiment_dir
        / "decoder_comparison"
        / "sorted"
        / "models"
        / "manifold_transforms"
    )
    if not man.exists() or not any(man.iterdir()):
        return
    rec = run.stage(STAGE_REPRESENTATION)
    rec.status = ArtifactStatus.CACHED
    rec.origin = ArtifactOrigin.MIGRATED
    rec.updated_at = _now_iso()
    if run.observation is not None:
        rec.source_hash = run.observation.hash()


def _infer_decoder(run: PipelineRun) -> None:
    metrics = (
        run.experiment_dir
        / "decoder_comparison"
        / "sorted"
        / "decoder_comparison_metrics.csv"
    )
    registry = run.experiment_dir / "models" / "best_realtime_decoders.json"
    if not metrics.exists() and not registry.exists():
        return
    rec = run.stage(STAGE_DECODER)
    rec.status = ArtifactStatus.CACHED
    rec.origin = ArtifactOrigin.MIGRATED
    rec.updated_at = _now_iso()
    if run.observation is not None:
        rec.source_hash = run.observation.hash()


def _infer_replay(run: PipelineRun) -> None:
    rt = run.experiment_dir / "realtime_decoding" / "sorted"
    if not rt.exists() or not any(rt.rglob("decoded_realtime.csv")):
        return
    rec = run.stage(STAGE_REPLAY)
    rec.status = ArtifactStatus.CACHED
    rec.origin = ArtifactOrigin.MIGRATED
    rec.updated_at = _now_iso()


def _infer_deployment(run: PipelineRun) -> None:
    bundles = run.experiment_dir / "deployment_bundles"
    if not bundles.exists() or not any(bundles.iterdir()):
        return
    rec = run.stage(STAGE_DEPLOYMENT)
    rec.status = ArtifactStatus.CACHED
    rec.origin = ArtifactOrigin.MIGRATED
    rec.updated_at = _now_iso()


def commit_simulation(experiment_dir: Path, *, summary: dict[str, Any] | None = None) -> PipelineRun:
    run = load_or_infer_pipeline(experiment_dir)
    summary = summary or {}
    if not summary:
        sp = Path(experiment_dir) / "summary.json"
        if sp.exists():
            try:
                summary = json.loads(sp.read_text())
            except (OSError, json.JSONDecodeError):
                summary = {}
    sim_hash = config_hash(
        {
            "run_id": run.simulation_run_id,
            "seed": summary.get("seed"),
            "session_duration_s": summary.get("session_duration_s"),
            "n_units": summary.get("n_units"),
        }
    )
    run.mark_committed(
        STAGE_SIMULATION,
        run_id=run.simulation_run_id,
        config_hash=sim_hash,
        extra={
            "seed": summary.get("seed"),
            "session_duration_s": summary.get("session_duration_s"),
            "n_units": summary.get("n_units"),
        },
    )
    save_pipeline_run(run)
    return run


def commit_observation(
    experiment_dir: Path,
    observation: ObservationConfig,
    *,
    run_id: str | None = None,
    origin: ArtifactOrigin = ArtifactOrigin.COMPUTED,
) -> PipelineRun:
    run = load_or_infer_pipeline(experiment_dir)
    run.set_observation(observation, run_id=run_id)
    rec = run.stage(STAGE_FEATURES)
    rec.origin = origin
    save_pipeline_run(run)
    return run


def commit_stage(
    experiment_dir: Path,
    name: str,
    *,
    run_id: str | None = None,
    config_hash: str | None = None,
    source_hash: str | None = None,
    origin: ArtifactOrigin = ArtifactOrigin.COMPUTED,
    extra: dict[str, Any] | None = None,
) -> PipelineRun:
    run = load_or_infer_pipeline(experiment_dir)
    run.mark_committed(
        name,
        run_id=run_id,
        config_hash=config_hash,
        source_hash=source_hash,
        origin=origin,
        extra=extra,
    )
    save_pipeline_run(run)
    return run


def inspect_stage(run: PipelineRun, name: str) -> dict[str, Any]:
    rec = run.stage(name)
    return {
        "stage": name,
        "status": rec.status.value,
        "origin": rec.origin.value,
        "run_id": rec.run_id,
        "config_hash": rec.config_hash,
        "source_hash": rec.source_hash,
        "stale": rec.status == ArtifactStatus.STALE,
        "incompatible": rec.status == ArtifactStatus.INCOMPATIBLE,
        "from_cache": rec.origin == ArtifactOrigin.CACHE,
        "newly_computed": rec.origin == ArtifactOrigin.COMPUTED
        and rec.status == ArtifactStatus.FRESH,
        "extra": dict(rec.extra),
    }
