"""Lightweight on-disk experiment run registry (JSON, no database)."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REGISTRY_FILENAME = "hippo_run_metadata.json"


@dataclass
class RunMetadata:
    """Persisted benchmark / experiment run descriptor."""

    run_id: str
    timestamp: str
    input_dataset: str
    output_directory: str
    targets: list[str] = field(default_factory=list)
    feature_sets: list[str] = field(default_factory=list)
    manifolds: list[str] = field(default_factory=list)
    decode_windows: list[float] = field(default_factory=list)
    degradation_levels: list[float] = field(default_factory=list)
    feature_ablation: bool = False
    compare_sources: bool = False
    spike_source: str = "sorted"
    git_commit: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    error: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunMetadata":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def new_run_id(prefix: str = "bench") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def try_git_commit(repo_root: Path | None = None) -> str | None:
    root = repo_root or Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def metadata_path(output_dir: Path) -> Path:
    return Path(output_dir) / REGISTRY_FILENAME


def save_run_metadata(meta: RunMetadata, output_dir: Path | None = None) -> Path:
    """Write metadata JSON into the run output directory."""
    out = Path(output_dir) if output_dir is not None else Path(meta.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    path = metadata_path(out)
    path.write_text(json.dumps(meta.to_dict(), indent=2, sort_keys=True) + "\n")
    return path


def load_run_metadata(path: Path) -> RunMetadata:
    """Load metadata from a file or from a directory containing the file."""
    path = Path(path)
    if path.is_dir():
        path = metadata_path(path)
    with open(path) as f:
        return RunMetadata.from_dict(json.load(f))


def discover_runs(
    search_roots: list[Path] | None = None,
    *,
    outputs_root: Path | None = None,
) -> list[RunMetadata]:
    """Find ``hippo_run_metadata.json`` files under experiment trees."""
    roots = list(search_roots or [])
    if outputs_root is None:
        outputs_root = Path(__file__).resolve().parents[2] / "outputs"
    if not roots:
        roots = [outputs_root]
    found: list[RunMetadata] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for meta_file in root.rglob(REGISTRY_FILENAME):
            try:
                found.append(load_run_metadata(meta_file))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                continue
    found.sort(key=lambda m: m.timestamp, reverse=True)
    return found


def create_pending_metadata(
    *,
    input_dataset: Path | str,
    output_directory: Path | str,
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...],
    feature_ablation: bool = False,
    compare_sources: bool = False,
    spike_source: str = "sorted",
    configuration: dict[str, Any] | None = None,
    targets: list[str] | None = None,
    run_id: str | None = None,
) -> RunMetadata:
    """Construct a pending registry entry (caller should save after)."""
    return RunMetadata(
        run_id=run_id or new_run_id(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        input_dataset=str(input_dataset),
        output_directory=str(output_directory),
        targets=list(targets or []),
        feature_sets=list(feature_sets),
        manifolds=list(manifolds),
        decode_windows=[float(w) for w in decode_windows],
        degradation_levels=[],
        feature_ablation=bool(feature_ablation),
        compare_sources=bool(compare_sources),
        spike_source=spike_source,
        git_commit=try_git_commit(),
        configuration=dict(configuration or {}),
        status="pending",
    )
