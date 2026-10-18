"""IO guards and in-process wrappers (no edits to existing modules)."""

from agents.data_quality.io.paths import (
    DATA_QUALITY_ROOT,
    WORKTREE_ROOT,
    assert_dq_path,
    refuse_live_paths,
)
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.analyze_wrap import patch_output_root

__all__ = [
    "DATA_QUALITY_ROOT",
    "WORKTREE_ROOT",
    "assert_dq_path",
    "refuse_live_paths",
    "build_segment_bundle_dq",
    "patch_output_root",
]
