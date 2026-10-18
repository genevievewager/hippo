"""Output-path guards for Report 4."""

from __future__ import annotations

from pathlib import Path

WORKTREE_ROOT = Path("/home/gmwager/projects/hippo-dq")
LIVE_CHECKOUT = Path("/home/gmwager/projects/hippo")
DATA_QUALITY_ROOT = WORKTREE_ROOT / "outputs" / "data_quality"

# Paths listed in LIVE_RUN.md / isolation exclusions (never write).
FORBIDDEN_PREFIXES: tuple[Path, ...] = (
    LIVE_CHECKOUT,
    WORKTREE_ROOT / "outputs" / "real_quadrant",
    WORKTREE_ROOT / "outputs" / "quadrant_n5",
    LIVE_CHECKOUT / "outputs" / "real_quadrant",
    LIVE_CHECKOUT / "outputs" / "quadrant_n5",
)


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_dq_path(path: Path | str, *, kind: str = "path") -> Path:
    """Require ``path`` to resolve under ``outputs/data_quality/``."""
    p = Path(path).resolve()
    if not _is_under(p, DATA_QUALITY_ROOT):
        raise RuntimeError(
            f"refusing {kind} outside outputs/data_quality/: {p}"
        )
    return p


def refuse_live_paths(path: Path | str, *, kind: str = "path") -> Path:
    """Refuse any path under the live checkout or shared Report 1/2 output trees."""
    p = Path(path).resolve()
    for bad in FORBIDDEN_PREFIXES:
        if _is_under(p, bad) and not (
            bad == LIVE_CHECKOUT and _is_under(p, DATA_QUALITY_ROOT)
        ):
            # LIVE_CHECKOUT is a parent of nothing in worktree; worktree is sibling.
            if _is_under(p, bad):
                raise RuntimeError(f"refusing {kind} under forbidden tree {bad}: {p}")
    # Explicit: must be under data_quality for writes
    return assert_dq_path(p, kind=kind)
