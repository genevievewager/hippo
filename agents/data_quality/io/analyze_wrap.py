"""In-process OUTPUT_ROOT redirect for realtime.quadrant_n5_run (no module edits)."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import realtime.quadrant_n5_run as qrun

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path

ANALYZE_OUTPUT_ROOT = DATA_QUALITY_ROOT / "quadrant_n5_run"


@contextmanager
def patch_output_root(root: Path | None = None) -> Iterator[Path]:
    """Temporarily point ``quadrant_n5_run.OUTPUT_ROOT`` under data_quality."""
    dest = assert_dq_path(root or ANALYZE_OUTPUT_ROOT, kind="OUTPUT_ROOT")
    dest.mkdir(parents=True, exist_ok=True)
    old = qrun.OUTPUT_ROOT
    qrun.OUTPUT_ROOT = dest
    try:
        yield dest
    finally:
        qrun.OUTPUT_ROOT = old
