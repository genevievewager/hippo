"""Causal internal binning utilities for coactivity features.

Subdivide the decode window ``[t - W, t)`` into ``B = floor(W / bin_dt)``
half-open bins.  All bins lie strictly before ``t`` (no future spikes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from realtime.spike_binner import _resolve_spike_columns


def n_coactivity_bins(decode_window: float, coactivity_bin_dt: float) -> int:
    """Number of internal bins covering [t-W, t) with width ``coactivity_bin_dt``."""
    w = float(decode_window)
    dt = float(coactivity_bin_dt)
    if dt <= 0:
        raise ValueError("coactivity_bin_dt must be > 0")
    n = int(np.floor(w / dt + 1e-12))
    return max(1, n)


def build_causal_unit_bin_tensor(
    spikes_df: pd.DataFrame,
    unit_ids: list[int] | np.ndarray,
    decode_times: np.ndarray,
    decode_window: float,
    coactivity_bin_dt: float,
) -> np.ndarray:
    """
    Build unit × time-bin activity inside each causal decode window.

    Returns
    -------
    X : ndarray, shape [n_times, n_units, n_bins]
        Spike counts in each half-open internal bin of ``[t - W, t)``.
        Bins are ordered oldest → newest; the last bin ends at ``t`` (exclusive).
    """
    unit_ids = np.asarray(unit_ids, dtype=int)
    decode_times = np.asarray(decode_times, dtype=float)
    n_units = len(unit_ids)
    n_times = len(decode_times)
    n_bins = n_coactivity_bins(decode_window, coactivity_bin_dt)
    effective_w = n_bins * float(coactivity_bin_dt)
    X = np.zeros((n_times, n_units, n_bins), dtype=np.float64)

    if spikes_df.empty or n_times == 0 or n_units == 0:
        return X

    time_col, unit_col = _resolve_spike_columns(spikes_df)
    spikes = spikes_df[[time_col, unit_col]].copy()
    spikes.columns = ["time", "unit_id"]
    spikes = spikes.sort_values("time", kind="mergesort")
    spikes = spikes[spikes["unit_id"].isin(unit_ids)]
    if spikes.empty:
        return X

    unit_to_idx = {int(u): i for i, u in enumerate(unit_ids)}
    times = spikes["time"].to_numpy(dtype=float)
    units = spikes["unit_id"].to_numpy(dtype=int)
    bin_dt = float(coactivity_bin_dt)

    for i, t in enumerate(decode_times):
        t_end = float(t)
        t_start = t_end - effective_w
        left = int(np.searchsorted(times, t_start, side="left"))
        right = int(np.searchsorted(times, t_end, side="left"))
        if left >= right:
            continue
        win_times = times[left:right]
        win_units = units[left:right]
        # Bin index 0 = oldest. Clamp to [0, n_bins).
        bin_idx = np.floor((win_times - t_start) / bin_dt).astype(np.int64)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        for u, b in zip(win_units, bin_idx, strict=False):
            ui = unit_to_idx.get(int(u), -1)
            if ui >= 0:
                X[i, ui, int(b)] += 1.0
    return X


def counts_from_bin_tensor(X_bins: np.ndarray) -> np.ndarray:
    """Sum internal bins → causal spike-count matrix [n_times, n_units]."""
    return np.asarray(X_bins, dtype=float).sum(axis=2)


def regional_population_traces(
    X_bins: np.ndarray,
    region_labels: list[str] | np.ndarray,
    region_order: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Collapse unit×bin activity to region×bin population traces.

    Returns
    -------
    traces : ndarray [n_times, n_regions, n_bins]
    region_order : list of region names matching axis 1
    """
    X = np.asarray(X_bins, dtype=float)
    labels = np.asarray(region_labels, dtype=object)
    if region_order is None:
        region_order = sorted({str(r) for r in labels.tolist()})
    n_times, _, n_bins = X.shape
    n_regions = len(region_order)
    traces = np.zeros((n_times, n_regions, n_bins), dtype=np.float64)
    for ri, region in enumerate(region_order):
        idx = np.where(labels == region)[0]
        if idx.size:
            traces[:, ri, :] = X[:, idx, :].sum(axis=1)
    return traces, list(region_order)
