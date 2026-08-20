"""Causal rolling spike buffer for live / replay inference."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np
import pandas as pd


class CausalSpikeBuffer:
    """Retain spikes and build count vectors for ``[t - W, t)``.

    Uses the same half-open window convention as offline
    ``count_spikes_in_window`` / ``build_causal_spike_matrix``.
    """

    def __init__(self, unit_ids: Iterable[int], *, history_s: float = 2.0):
        self.unit_ids = np.asarray(list(unit_ids), dtype=int)
        self.history_s = float(history_s)
        self._times: deque[float] = deque()
        self._units: deque[int] = deque()
        self._unit_to_col = {int(u): i for i, u in enumerate(self.unit_ids)}

    def clear(self) -> None:
        self._times.clear()
        self._units.clear()

    def extend(self, times: np.ndarray | list[float], unit_ids: np.ndarray | list[int]) -> None:
        times_a = np.asarray(times, dtype=float).ravel()
        units_a = np.asarray(unit_ids, dtype=int).ravel()
        if times_a.size != units_a.size:
            raise ValueError("times and unit_ids must have the same length")
        for t, u in zip(times_a.tolist(), units_a.tolist()):
            self._times.append(float(t))
            self._units.append(int(u))
        self._prune(max(times_a) if times_a.size else None)

    def extend_dataframe(self, spikes: pd.DataFrame, *, time_col: str = "time", unit_col: str = "unit_id") -> None:
        if spikes is None or spikes.empty:
            return
        # Accept common aliases.
        tcol = time_col
        if tcol not in spikes.columns:
            for c in ("time", "time_s", "spike_time", "t"):
                if c in spikes.columns:
                    tcol = c
                    break
        ucol = unit_col
        if ucol not in spikes.columns:
            for c in ("unit_id", "unit", "cluster_id"):
                if c in spikes.columns:
                    ucol = c
                    break
        self.extend(spikes[tcol].to_numpy(), spikes[ucol].to_numpy())

    def _prune(self, latest_t: float | None) -> None:
        if latest_t is None or not self._times:
            return
        cutoff = float(latest_t) - self.history_s
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._units.popleft()

    def counts_at(self, t: float, decode_window_s: float) -> np.ndarray:
        """Return length-|units| count vector for spikes in ``[t - W, t)``."""
        t = float(t)
        t0 = t - float(decode_window_s)
        out = np.zeros(len(self.unit_ids), dtype=float)
        # Linear scan is fine for small buffers; spikes are pruned to history_s.
        for ts, uid in zip(self._times, self._units):
            if ts < t0:
                continue
            if ts >= t:
                break
            col = self._unit_to_col.get(int(uid))
            if col is not None:
                out[col] += 1.0
        return out

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({"time": list(self._times), "unit_id": list(self._units)})

    @property
    def n_spikes(self) -> int:
        return len(self._times)
