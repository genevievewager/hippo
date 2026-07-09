"""Causal spike counting for real-time decoding windows.

The simulator knows true behavior, but the decoder only sees spikes recorded
up to the current time. Each decode step uses spikes from [t - window, t) only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _resolve_spike_columns(spikes_df: pd.DataFrame) -> tuple[str, str]:
    time_candidates = ["time", "spike_time", "spike_time_s", "timestamp", "times"]
    unit_candidates = ["unit_id", "unit", "cluster_id"]

    time_col = next((c for c in time_candidates if c in spikes_df.columns), None)
    unit_col = next((c for c in unit_candidates if c in spikes_df.columns), None)

    if time_col is None:
        raise ValueError(
            f"No spike time column found. Expected one of {time_candidates}; "
            f"got columns: {list(spikes_df.columns)}"
        )
    if unit_col is None:
        raise ValueError(
            f"No unit id column found. Expected one of {unit_candidates}; "
            f"got columns: {list(spikes_df.columns)}"
        )
    return time_col, unit_col


def count_spikes_in_window(
    spikes_df: pd.DataFrame,
    unit_ids: list[int] | np.ndarray,
    t_start: float,
    t_end: float,
) -> np.ndarray:
    """
    Return a 1D numpy array of spike counts for all units in unit_ids
    using only spikes in [t_start, t_end).

    Units appear in the same order as unit_ids.
    """
    unit_ids = np.asarray(unit_ids, dtype=int)
    unit_to_idx = {int(u): i for i, u in enumerate(unit_ids)}
    counts = np.zeros(len(unit_ids), dtype=np.float64)

    if spikes_df.empty:
        return counts

    time_col, unit_col = _resolve_spike_columns(spikes_df)
    times = spikes_df[time_col].to_numpy()
    units = spikes_df[unit_col].to_numpy(dtype=int)

    left = int(np.searchsorted(times, t_start, side="left"))
    right = int(np.searchsorted(times, t_end, side="left"))
    if left >= right:
        return counts

    window_units = units[left:right]
    mapped = np.fromiter(
        (unit_to_idx.get(int(u), -1) for u in window_units),
        dtype=np.int64,
        count=len(window_units),
    )
    valid = mapped >= 0
    if np.any(valid):
        counts += np.bincount(mapped[valid], minlength=len(unit_ids))

    return counts


def build_causal_spike_matrix(
    spikes_df: pd.DataFrame,
    unit_ids: list[int] | np.ndarray,
    decode_times: np.ndarray,
    decode_window: float,
) -> np.ndarray:
    """
    For each decoder time t, count spikes in [t - decode_window, t).

    Returns X with shape [n_decode_times, n_units].
    """
    unit_ids = np.asarray(unit_ids, dtype=int)
    n_units = len(unit_ids)
    n_times = len(decode_times)
    X = np.zeros((n_times, n_units), dtype=np.float64)

    if spikes_df.empty or n_times == 0:
        return X

    time_col, unit_col = _resolve_spike_columns(spikes_df)
    spikes = spikes_df[[time_col, unit_col]].copy()
    spikes.columns = ["time", "unit_id"]
    spikes = spikes.sort_values("time", kind="mergesort")
    spikes = spikes[spikes["unit_id"].isin(unit_ids)]

    if spikes.empty:
        return X

    unit_to_idx = {int(u): i for i, u in enumerate(unit_ids)}
    times = spikes["time"].to_numpy()
    units = spikes["unit_id"].to_numpy(dtype=int)

    for i, t in enumerate(decode_times):
        t_start = t - decode_window
        left = int(np.searchsorted(times, t_start, side="left"))
        right = int(np.searchsorted(times, t, side="left"))
        if left >= right:
            continue

        window_units = units[left:right]
        mapped = np.fromiter(
            (unit_to_idx.get(int(u), -1) for u in window_units),
            dtype=np.int64,
            count=len(window_units),
        )
        valid = mapped >= 0
        if np.any(valid):
            X[i] = np.bincount(mapped[valid], minlength=n_units)

    return X
