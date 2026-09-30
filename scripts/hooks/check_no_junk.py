#!/usr/bin/env python3
"""Reject staged junk before commit (public-repo hygiene).

Blocks:
  - files larger than 5 MB
  - data extensions: .npy .npz .parquet .h5 .nwb .mat .pkl
  - paths under root-only data/, outputs/, quadrant_n5/, reports/
    (same anchoring as .gitignore `/quadrant_n5/` — not agents/quadrant_n5/),
    analysis/**/outputs/, or HIPPO_DATA_ROOT (when set)

Allowlist (fixtures):
  - configs/trajectories/*_regions.csv
  - data/probe_trajectories/lab_insertion_001.csv

Usage:
  scripts/hooks/check_no_junk.py              # check staged files
  scripts/hooks/check_no_junk.py PATH...      # check explicit paths (tests)
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
DATA_EXTS = {".npy", ".npz", ".parquet", ".h5", ".nwb", ".mat", ".pkl"}
# Root-anchored only (path must begin with these segments). Nested names like
# agents/quadrant_n5/ are allowed — mirrors .gitignore `/quadrant_n5/`.
BLOCKED_ROOT_DIRS = frozenset({
    "data",
    "outputs",
    "quadrant_n5",
    "reports",
})
ALLOWLIST = {
    "configs/trajectories/hpc_optimal_regions.csv",
    "configs/trajectories/lab_npx2_default_regions.csv",
    "data/probe_trajectories/lab_insertion_001.csv",
}


def _repo_root() -> Path:
    out = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(out)


def staged_paths() -> list[str]:
    out = subprocess.check_output(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_under_analysis_outputs(path: str) -> bool:
    parts = Path(path).parts
    if "analysis" not in parts:
        return False
    try:
        i = parts.index("analysis")
    except ValueError:
        return False
    return "outputs" in parts[i + 1 :]


def is_under_data_root(path: str, root: Path) -> bool:
    data_root = os.environ.get("HIPPO_DATA_ROOT", "").strip()
    if not data_root:
        return False
    try:
        resolved = (root / path).resolve()
        return resolved.is_relative_to(Path(data_root).resolve())
    except (OSError, ValueError):
        return False


def violations_for(path: str, *, root: Path, size: int | None) -> list[str]:
    norm = path.replace("\\", "/").lstrip("./")
    if norm in ALLOWLIST:
        return []
    if any(Path(norm).match(pat) for pat in (
        "configs/trajectories/*_regions.csv",
    )):
        # keep allowlist explicit; match pattern for future fixtures
        if Path(norm).name.endswith("_regions.csv") and norm.startswith(
            "configs/trajectories/"
        ):
            return []

    bad: list[str] = []
    if size is not None and size > MAX_BYTES:
        bad.append(f">{MAX_BYTES} bytes")
    ext = Path(norm).suffix.lower()
    if ext in DATA_EXTS:
        bad.append(f"extension {ext}")
    top = Path(norm).parts[0] if Path(norm).parts else ""
    if top in BLOCKED_ROOT_DIRS:
        bad.append("blocked path prefix")
    if is_under_analysis_outputs(norm):
        bad.append("under analysis/**/outputs/")
    if is_under_data_root(norm, root):
        bad.append("under HIPPO_DATA_ROOT")
    return bad


def check_paths(paths: list[str], *, root: Path | None = None) -> list[str]:
    root = root or _repo_root()
    errors: list[str] = []
    for path in paths:
        norm = path.replace("\\", "/").lstrip("./")
        full = root / norm
        size = full.stat().st_size if full.is_file() else None
        bad = violations_for(norm, root=root, size=size)
        if bad:
            errors.append(f"{norm}: " + ", ".join(bad))
    return errors


def main(argv: list[str]) -> int:
    root = _repo_root()
    paths = argv[1:] if len(argv) > 1 else staged_paths()
    if not paths:
        return 0
    errors = check_paths(paths, root=root)
    if errors:
        sys.stderr.write(
            "pre-commit: refusing junk / oversized / data paths:\n"
            + "\n".join(f"  - {e}" for e in errors)
            + "\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
