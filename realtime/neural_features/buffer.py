"""Bounded rolling spike buffer for realtime neural feature extraction.

Stores only the recent spike history required for the decode window (plus a
small margin for lagged / dynamics features). Does not retain the full session.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd

from realtime.neural_features.extractor import NeuralFeatureExtractor
from realtime.neural_features.types import FeatureExtractionResult


class CausalSpikeBuffer:
    """Ring-buffer of (time, unit_id) spikes for online decoding."""

    def __init__(
        self,
        *,
        history_seconds: float,
        unit_ids: list[int] | np.ndarray,
    ):
        self.history_seconds = float(history_seconds)
        self.unit_ids = np.asarray(unit_ids, dtype=int)
        self._times: deque[float] = deque()
        self._units: deque[int] = deque()

    def __len__(self) -> int:
        return len(self._times)

    def clear(self) -> None:
        self._times.clear()
        self._units.clear()

    def push(self, time: float, unit_id: int) -> None:
        t = float(time)
        self._times.append(t)
        self._units.append(int(unit_id))
        self._evict(t)

    def push_many(self, times: np.ndarray, unit_ids: np.ndarray) -> None:
        for t, u in zip(np.asarray(times, dtype=float), np.asarray(unit_ids, dtype=int), strict=False):
            self.push(float(t), int(u))

    def _evict(self, now: float) -> None:
        cutoff = now - self.history_seconds
        while self._times and self._times[0] < cutoff:
            self._times.popleft()
            self._units.popleft()

    def as_dataframe(self) -> pd.DataFrame:
        if not self._times:
            return pd.DataFrame({"time": [], "unit_id": []})
        return pd.DataFrame({
            "time": np.fromiter(self._times, dtype=float, count=len(self._times)),
            "unit_id": np.fromiter(self._units, dtype=int, count=len(self._units)),
        })

    def extract(
        self,
        extractor: NeuralFeatureExtractor,
        t: float,
        *,
        prev_counts: np.ndarray | None = None,
    ) -> FeatureExtractionResult:
        """Evict stale spikes then extract features at ``t``."""
        self._evict(float(t))
        return extractor.extract_at(self.as_dataframe(), t, prev_counts=prev_counts)


def required_buffer_seconds(extractor: NeuralFeatureExtractor) -> float:
    """Bounded history length needed for an extractor configuration."""
    base = float(extractor.decode_window)
    extra = 0.0
    for spec in extractor.feature_metadata_ or []:
        extra = max(extra, float(spec.required_history_seconds))
    # Dynamics need one previous update; also keep margin for lag bins.
    if "count_dynamics" in extractor.families:
        extra = max(extra, float(extractor.update_dt))
    if "lagged_coupling" in extractor.families and extractor.lagged_coupling_lags:
        max_lag = max(extractor.lagged_coupling_lags)
        extra = max(extra, max_lag * float(extractor.coactivity_bin_dt))
    return base + extra + 1e-3
