"""Matched-length chunking (default 300 s)."""

from __future__ import annotations

import numpy as np


def chunk_bounds(
    t0: float,
    t1: float,
    *,
    chunk_s: float = 300.0,
    trim_start_s: float = 60.0,
    trim_end_s: float = 10.0,
) -> list[tuple[float, float]]:
    """Return half-open [lo, hi) chunk windows on the retained segment."""
    lo = float(t0) + float(trim_start_s)
    hi = float(t1) - float(trim_end_s)
    if hi <= lo:
        # short sessions (e.g. 300 s proxy): one residual window if any usable length
        lo2 = float(t0)
        hi2 = float(t1)
        if hi2 - lo2 < 30.0:
            return []
        return [(lo2, hi2)]
    out: list[tuple[float, float]] = []
    t = lo
    while t + chunk_s <= hi + 1e-9:
        out.append((t, t + chunk_s))
        t += chunk_s
    if not out and hi - lo >= 30.0:
        # duration too short for a full chunk_s after trim — one trimmed window
        out.append((lo, hi))
    return out


def assign_chunk_index(times: np.ndarray, bounds: list[tuple[float, float]]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    idx = np.full(times.shape, -1, dtype=int)
    for i, (lo, hi) in enumerate(bounds):
        idx[(times >= lo) & (times < hi)] = i
    return idx
