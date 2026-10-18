"""Unit-level metrics (rate, ISI, burst, refractory, Fano, drift)."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.data_quality.metrics.schema import tidy_row

W_GRID = (0.05, 0.1, 0.25, 0.5, 1.0)


def _isi(st: np.ndarray) -> np.ndarray:
    st = np.sort(np.asarray(st, dtype=float))
    d = np.diff(st)
    return d[d > 0]


def firing_rate(st: np.ndarray, t0: float, t1: float) -> float:
    st = np.asarray(st, dtype=float)
    st = st[(st >= t0) & (st < t1)]
    return float(st.size / max(t1 - t0, 1e-9))


def isi_cv(st: np.ndarray) -> float:
    d = _isi(st)
    if d.size < 2:
        return float("nan")
    return float(np.std(d) / max(np.mean(d), 1e-12))


def isi_cv2(st: np.ndarray) -> float:
    d = _isi(st)
    if d.size < 2:
        return float("nan")
    pairs = 2.0 * np.abs(d[1:] - d[:-1]) / np.maximum(d[1:] + d[:-1], 1e-12)
    return float(np.mean(pairs))


def burst_fraction(st: np.ndarray, thresh_s: float = 0.010) -> float:
    d = _isi(st)
    if d.size == 0:
        return float("nan")
    return float(np.mean(d < thresh_s))


def refractory_violation_rate(st: np.ndarray, ref_s: float = 0.0015) -> float:
    d = _isi(st)
    if d.size == 0:
        return float("nan")
    return float(np.mean(d < ref_s))


# Lab median min-ISI ≈ 0.3 ms (duplicate-spike removal assumption; see Report 4a Q3).
LAB_DUP_REMOVAL_ISI_S = 0.0003


def remove_spikes_below_isi(st: np.ndarray, min_isi_s: float = LAB_DUP_REMOVAL_ISI_S) -> np.ndarray:
    """Keep first spike of each run closer than ``min_isi_s`` (lab-equivalent dup removal)."""
    st = np.sort(np.asarray(st, dtype=float))
    if st.size <= 1:
        return st
    keep = [float(st[0])]
    for t in st[1:]:
        if float(t) - keep[-1] >= float(min_isi_s):
            keep.append(float(t))
    return np.asarray(keep, dtype=float)


def frac_isi_below(st: np.ndarray, thresh_s: float) -> float:
    d = _isi(st)
    if d.size == 0:
        return float("nan")
    return float(np.mean(d < float(thresh_s)))


def fano_factor(st: np.ndarray, t0: float, t1: float, W: float) -> float:
    st = np.asarray(st, dtype=float)
    st = st[(st >= t0) & (st < t1)]
    if t1 - t0 < W:
        return float("nan")
    edges = np.arange(t0, t1 + 1e-9, W)
    if edges.size < 3:
        return float("nan")
    counts, _ = np.histogram(st, bins=edges)
    if counts.size < 2:
        return float("nan")
    mu = float(np.mean(counts))
    if mu <= 0:
        return float("nan")
    return float(np.var(counts) / mu)


def rate_drift(st: np.ndarray, t0: float, t1: float) -> float:
    """Slope of rates across chunk thirds, divided by mean rate."""
    st = np.asarray(st, dtype=float)
    st = st[(st >= t0) & (st < t1)]
    thirds = np.linspace(t0, t1, 4)
    rates = []
    for a, b in zip(thirds[:-1], thirds[1:]):
        rates.append(np.sum((st >= a) & (st < b)) / max(b - a, 1e-9))
    rates = np.asarray(rates, dtype=float)
    mean_r = float(np.mean(rates))
    if mean_r <= 0:
        return float("nan")
    x = np.array([0.0, 1.0, 2.0])
    slope = float(np.polyfit(x, rates, 1)[0])
    return slope / mean_r


def unit_metric_rows(
    spike_times: list[np.ndarray],
    *,
    dataset: str,
    animal_or_seed: str,
    session: str,
    chunk: int | str,
    region_set: str,
    t0: float,
    t1: float,
    regions: list[str] | None = None,
    W_grid: tuple[float, ...] = W_GRID,
) -> list[dict[str, Any]]:
    """One code path for sim and lab — ``dataset`` is a label only."""
    rows: list[dict[str, Any]] = []
    for j, st in enumerate(spike_times):
        reg = None if regions is None else regions[j]
        base = dict(
            dataset=dataset,
            animal_or_seed=animal_or_seed,
            session=session,
            chunk=chunk,
            region_set=region_set if reg is None else f"{region_set}:{reg}",
        )
        fr = firing_rate(st, t0, t1)
        rows.append(tidy_row(**base, metric="firing_rate_hz", W=None, N=None, value=fr))
        rows.append(tidy_row(**base, metric="isi_cv", W=None, N=None, value=isi_cv(st)))
        rows.append(tidy_row(**base, metric="isi_cv2", W=None, N=None, value=isi_cv2(st)))
        rows.append(tidy_row(**base, metric="burst_fraction", W=None, N=None, value=burst_fraction(st)))
        rows.append(
            tidy_row(**base, metric="refractory_violation_rate", W=None, N=None,
                     value=refractory_violation_rate(st))
        )
        rows.append(tidy_row(**base, metric="rate_drift", W=None, N=None, value=rate_drift(st, t0, t1)))
        for W in W_grid:
            rows.append(
                tidy_row(**base, metric="fano_factor", W=float(W), N=None,
                         value=fano_factor(st, t0, t1, float(W)))
            )
    return rows
