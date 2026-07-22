"""Adaptive integration-window grids for decoder comparison and temporal runs.

Coarse → refine keeps flanks (too short / too long) while densifying near
optima discovered in a first pass. Temporal W grids can inherit Step-1 picks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# Full candidate pool used for refinement / flank selection.
WINDOW_CANDIDATE_POOL: tuple[float, ...] = (0.050, 0.100, 0.250, 0.500, 1.000)

# Default coarse grid for quick/standard profiles (skips 0.1 until refine).
COARSE_DECODE_WINDOWS: tuple[float, ...] = (0.050, 0.250, 0.500, 1.000)


def _closest_pool_index(window_s: float, pool: tuple[float, ...] = WINDOW_CANDIDATE_POOL) -> int:
    return int(min(range(len(pool)), key=lambda i: abs(pool[i] - float(window_s))))


def propose_refined_windows(
    tested_windows: Iterable[float],
    best_windows: Iterable[float],
    *,
    pool: tuple[float, ...] = WINDOW_CANDIDATE_POOL,
) -> tuple[float, ...]:
    """Return untested pool neighbors of each best window (near-optimal densify).

    For every best ``W``, adds immediate left/right neighbors in ``pool`` that
    were not already tested. Does not drop the original coarse flanks.
    """
    tested = {round(float(w), 6) for w in tested_windows}
    extras: set[float] = set()
    for w in best_windows:
        idx = _closest_pool_index(float(w), pool)
        for j in (idx - 1, idx + 1):
            if 0 <= j < len(pool):
                cand = float(pool[j])
                if round(cand, 6) not in tested:
                    extras.add(cand)
    return tuple(sorted(extras))


def windows_from_comparison_dir(
    comparison_dir: Path,
    sources: tuple[str, ...],
    *,
    fallback: tuple[float, ...] = COARSE_DECODE_WINDOWS,
    pool: tuple[float, ...] = WINDOW_CANDIDATE_POOL,
    include_flanks: bool = True,
) -> tuple[float, ...]:
    """Build a temporal W grid from Step-1 best / recommended windows.

    Unions ``best_decode_window_s`` and ``recommended_realtime_window_s`` across
    sources, optionally adding pool min/max flanks so the sweep still shows
    both sides of the operating region.
    """
    comparison_dir = Path(comparison_dir)
    collected: set[float] = set()
    for source in sources:
        for path in (
            comparison_dir / source / "best_decoder_by_target.csv",
            comparison_dir / "best_decoder_by_target.csv",
        ):
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if "spike_source" in df.columns:
                df = df[df["spike_source"].astype(str) == source]
            for col in ("best_decode_window_s", "recommended_realtime_window_s"):
                if col in df.columns:
                    for v in df[col].dropna().tolist():
                        collected.add(float(v))
            break

    if not collected:
        return tuple(float(w) for w in fallback)

    if include_flanks and pool:
        collected.add(float(pool[0]))
        collected.add(float(pool[-1]))

    # Snap to pool values when very close (avoid 0.5000001 duplicates).
    snapped: set[float] = set()
    for w in collected:
        idx = _closest_pool_index(w, pool)
        snapped.add(float(pool[idx]) if abs(pool[idx] - w) < 1e-6 else round(w, 6))
    return tuple(sorted(snapped))
