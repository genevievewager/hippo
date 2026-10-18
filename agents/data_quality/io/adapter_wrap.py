"""In-process wrapper around analysis.real_quadrant.adapter (no module edits)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

import analysis.real_quadrant.adapter as adapter

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path

CACHE_ROOT = DATA_QUALITY_ROOT / "cache"
MANIFEST_PATH = DATA_QUALITY_ROOT / "manifest.jsonl"


def _append_manifest_dq(record: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    path = assert_dq_path(path, kind="manifest")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def _assert_path_inside_polygon_flag_only(
    x, y, polygon_xy, *, min_frac: float = adapter.PATH_CONTAINMENT_MIN,
) -> dict[str, float]:
    """Report 2 extract_tidy semantics: flag path containment, do not exclude.

    Adapter default raises PathContainmentError (hard exclude at load). For
    Report 4 DQ we keep the session and record the flag so spatial metrics can
    use a path-visited bin mask.
    """
    frac = adapter.path_polygon_containment(x, y, polygon_xy)
    stats = {
        "path_containment_frac": float(frac),
        "path_containment_min": float(min_frac),
        "path_containment_flagged": bool(
            (not np.isfinite(frac)) or frac < float(min_frac)
        ),
    }
    if stats["path_containment_flagged"]:
        _append_manifest_dq({
            "status": "path_containment_flag",
            "path_containment_frac": stats["path_containment_frac"],
            "path_containment_min": stats["path_containment_min"],
            "note": "flagged only; kept for DQ scores with path-visited mask",
        })
    return stats


@contextmanager
def _redirect_adapter_io() -> Iterator[None]:
    """Monkeypatch adapter defaults and manifest writer for this process only."""
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    assert_dq_path(CACHE_ROOT, kind="cache_root")
    assert_dq_path(MANIFEST_PATH, kind="manifest")

    old_cache = adapter.DEFAULT_CACHE_ROOT
    old_manifest = adapter.DEFAULT_MANIFEST_PATH
    old_append = adapter._append_manifest
    old_assert_pc = adapter.assert_path_inside_polygon
    adapter.DEFAULT_CACHE_ROOT = CACHE_ROOT
    adapter.DEFAULT_MANIFEST_PATH = MANIFEST_PATH
    adapter._append_manifest = _append_manifest_dq  # type: ignore[assignment]
    adapter.assert_path_inside_polygon = _assert_path_inside_polygon_flag_only  # type: ignore[assignment]
    try:
        yield
    finally:
        adapter.DEFAULT_CACHE_ROOT = old_cache
        adapter.DEFAULT_MANIFEST_PATH = old_manifest
        adapter._append_manifest = old_append  # type: ignore[assignment]
        adapter.assert_path_inside_polygon = old_assert_pc  # type: ignore[assignment]


def build_segment_bundle_dq(
    session_name: str,
    room: str = "A",
    **kwargs: Any,
) -> dict[str, Any]:
    """Call ``build_segment_bundle`` with caches/manifest under outputs/data_quality/."""
    kwargs = dict(kwargs)
    kwargs.setdefault("cache_root", CACHE_ROOT)
    with _redirect_adapter_io():
        return adapter.build_segment_bundle(session_name, room, **kwargs)
