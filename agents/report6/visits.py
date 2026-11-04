"""Visit counting for latent-target occupancy (pre-registered rule)."""

from __future__ import annotations

import numpy as np


def count_visits(
    in_target: np.ndarray,
    *,
    times_s: np.ndarray | None = None,
    dt_s: float = 0.050,
    min_outside_s: float = 1.0,
) -> dict[str, float | int | list[float]]:
    """Count separate visits to a binary in-target series.

    A visit is an entry into True after a contiguous outside period of at least
    ``min_outside_s`` (pre-registered: 1.0 s). The first True run in the series
    counts as a visit.

    Returns occupancy fraction, visit count, and per-visit dwell times (s).
    """
    m = np.asarray(in_target, dtype=bool).ravel()
    n = int(m.size)
    if n == 0:
        return {
            "occupancy": 0.0,
            "n_visits": 0,
            "dwell_s": [],
            "mean_dwell_s": float("nan"),
            "median_dwell_s": float("nan"),
        }

    if times_s is None:
        t = np.arange(n, dtype=float) * float(dt_s)
    else:
        t = np.asarray(times_s, dtype=float).ravel()
        if t.size != n:
            raise ValueError("times_s length must match in_target")

    # Pad with False so edge runs close cleanly.
    padded = np.concatenate([[False], m, [False]])
    diff = np.diff(padded.astype(np.int8))
    entries = np.where(diff == 1)[0]  # index into m
    exits = np.where(diff == -1)[0]

    dwells: list[float] = []
    n_visits = 0
    last_exit_t = None
    for ent, ex in zip(entries, exits):
        # dwell from ent inclusive to ex exclusive in m-index
        dwell = float(t[ex - 1] - t[ent]) + float(dt_s) if ex > ent else float(dt_s)
        if last_exit_t is None:
            ok = True
        else:
            gap = float(t[ent] - last_exit_t)
            ok = gap >= float(min_outside_s)
        if ok:
            n_visits += 1
            dwells.append(max(dwell, 0.0))
        last_exit_t = float(t[ex - 1]) + float(dt_s) if ex > 0 else float(t[ent])

    occ = float(m.mean()) if n else 0.0
    return {
        "occupancy": occ,
        "n_visits": int(n_visits),
        "dwell_s": dwells,
        "mean_dwell_s": float(np.mean(dwells)) if dwells else float("nan"),
        "median_dwell_s": float(np.median(dwells)) if dwells else float("nan"),
    }


def binary_runs_from_labels(
    labels: np.ndarray,
    target_label: int,
) -> np.ndarray:
    """Boolean membership for one cluster label."""
    return np.asarray(labels) == int(target_label)
