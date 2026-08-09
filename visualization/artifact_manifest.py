"""Optional on-disk artifact manifests written alongside saved figures.

Backward compatible: old experiments without ``artifacts.json`` still work via
filesystem discovery in ``ui.artifacts.discovery``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "artifacts.json"


def manifest_path_for(path: Path) -> Path:
    """Prefer ``figures/artifacts.json`` when saving under a figures tree."""
    path = Path(path).resolve()
    parts = path.parts
    if "figures" in parts:
        idx = parts.index("figures")
        return Path(*parts[: idx + 1]) / MANIFEST_FILENAME
    # Sidecar next to the file's parent experiment-ish directory
    return path.parent / MANIFEST_FILENAME


def load_manifest(manifest_file: Path) -> list[dict[str, Any]]:
    path = Path(manifest_file)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read artifact manifest %s: %s", path, exc)
        return []
    if isinstance(data, dict):
        arts = data.get("artifacts", [])
        return list(arts) if isinstance(arts, list) else []
    if isinstance(data, list):
        return data
    return []


def save_manifest(manifest_file: Path, artifacts: list[dict[str, Any]]) -> None:
    path = Path(manifest_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def register_artifact(
    path: Path | str,
    *,
    category: str | None = None,
    title: str | None = None,
    target: str | None = None,
    feature_set: str | None = None,
    manifold: str | None = None,
    decoder: str | None = None,
    decode_window: float | None = None,
    degradation_level: float | None = None,
    run_id: str | None = None,
    description: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Append / update one artifact entry in the nearest ``artifacts.json``.

    Safe no-op on I/O errors so figure generation never fails due to registry.
    """
    path = Path(path)
    try:
        path = path.resolve()
    except OSError:
        path = Path(path)

    entry: dict[str, Any] = {
        "path": str(path),
        "relative_path": path.name,
        "category": category,
        "title": title,
        "target": target,
        "feature_set": feature_set,
        "manifold": manifold,
        "decoder": decoder,
        "decode_window": decode_window,
        "degradation_level": degradation_level,
        "run_id": run_id,
        "description": description,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        entry.update(extra)

    # Prefer path relative to figures/ root when possible
    manifest_file = manifest_path_for(path)
    figures_root = manifest_file.parent
    try:
        entry["relative_path"] = path.relative_to(figures_root).as_posix()
    except ValueError:
        entry["relative_path"] = path.name

    try:
        existing = load_manifest(manifest_file)
        # Upsert by relative_path / absolute path
        key = entry["relative_path"]
        kept = [
            a for a in existing
            if str(a.get("relative_path") or a.get("path")) != key
            and str(a.get("path")) != str(path)
        ]
        kept.append({k: v for k, v in entry.items() if v is not None})
        save_manifest(manifest_file, kept)
    except OSError as exc:
        logger.warning("Could not register artifact %s: %s", path, exc)

    return path
