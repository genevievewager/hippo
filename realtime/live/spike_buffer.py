"""Causal rolling spike buffer for live / replay inference.

Backed by preallocated numpy arrays kept in ascending time order, so
``counts_at`` is a binary search rather than a linear scan and cost depends on
the window, not on how much history is retained.

Three invariants, each of which was a silent wrong-answer bug when it was only
an assumption:

  1. Non-finite timestamps are rejected on insert. NaN compares False against
     every ordering test, so a single NaN that gets in is counted into every
     window forever and never prunes.
  2. Storage is sorted by time regardless of arrival order. Live sorters
     deliver events slightly out of order; a scan that assumes monotonic
     arrival silently undercounts.
  3. Pruning is driven by the decoder's clock, not by incoming data. Using the
     max of an incoming chunk lets one glitched sample evict all real history.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# One alias table, shared with realtime.spike_binner via _resolve_spike_columns.
# Three separate tables is how the same frame came to pass one module and fail
# another.
TIME_ALIASES = ("time", "time_s", "spike_time", "spike_time_s", "timestamp", "times", "t")
UNIT_ALIASES = ("unit_id", "unit", "cluster_id", "clusters", "cluster")


def resolve_spike_columns(
    spikes: pd.DataFrame,
    *,
    time_col: str | None = None,
    unit_col: str | None = None,
) -> tuple[str, str]:
    """Return (time_column, unit_column), raising with the columns actually seen."""
    tcol = time_col if time_col in spikes.columns else None
    if tcol is None:
        tcol = next((c for c in TIME_ALIASES if c in spikes.columns), None)
    ucol = unit_col if unit_col in spikes.columns else None
    if ucol is None:
        ucol = next((c for c in UNIT_ALIASES if c in spikes.columns), None)
    if tcol is None:
        raise ValueError(
            f"No spike time column found. Expected one of {TIME_ALIASES}; "
            f"got columns: {list(spikes.columns)}"
        )
    if ucol is None:
        raise ValueError(
            f"No unit id column found. Expected one of {UNIT_ALIASES}; "
            f"got columns: {list(spikes.columns)}"
        )
    return tcol, ucol


class CausalSpikeBuffer:
    """Retain spikes and build count vectors for ``[t - W, t)``.

    Same half-open convention as offline ``count_spikes_in_window`` /
    ``build_causal_spike_matrix``.
    """

    def __init__(
        self,
        unit_ids: Iterable[int],
        *,
        history_s: float = 2.0,
        max_future_skew_s: float = 1.0,
    ):
        self.unit_ids = np.asarray(list(unit_ids), dtype=int)
        self.history_s = float(history_s)
        # A sample this far beyond the decoder clock is a clock glitch, not data.
        self.max_future_skew_s = float(max_future_skew_s)
        self._times = np.empty(0, dtype=np.float64)
        self._cols = np.empty(0, dtype=np.int64)
        self._unit_to_col = {int(u): i for i, u in enumerate(self.unit_ids)}
        self._clock: float | None = None
        self.n_rejected_nonfinite = 0
        self.n_rejected_unknown_unit = 0

    # -- ingest ---------------------------------------------------------

    def clear(self) -> None:
        self._times = np.empty(0, dtype=np.float64)
        self._cols = np.empty(0, dtype=np.int64)
        self._clock = None

    def extend(
        self,
        times: np.ndarray | list[float],
        unit_ids: np.ndarray | list[int],
    ) -> None:
        t = np.asarray(times, dtype=float).ravel()
        u = np.asarray(unit_ids).ravel()
        if t.size != u.size:
            raise ValueError("times and unit_ids must have the same length")
        if t.size == 0:
            return

        finite = np.isfinite(t)
        if not finite.all():
            self.n_rejected_nonfinite += int((~finite).sum())
            raise ValueError(
                f"{int((~finite).sum())} non-finite spike timestamp(s) rejected. "
                "NaN/inf in a spike train is corrupt acquisition, not data: it "
                "would be counted into every decode window and would disable "
                "history pruning."
            )

        cols = np.fromiter(
            (self._unit_to_col.get(int(x), -1) for x in u),
            dtype=np.int64,
            count=u.size,
        )
        known = cols >= 0
        self.n_rejected_unknown_unit += int((~known).sum())
        t, cols = t[known], cols[known]
        if t.size == 0:
            return

        self._times = np.concatenate((self._times, t))
        self._cols = np.concatenate((self._cols, cols))
        # Arrival order is not guaranteed monotonic; storage order is.
        if self._times.size > 1 and not np.all(self._times[:-1] <= self._times[1:]):
            order = np.argsort(self._times, kind="stable")
            self._times = self._times[order]
            self._cols = self._cols[order]

    def extend_dataframe(
        self,
        spikes: pd.DataFrame,
        *,
        time_col: str = "time",
        unit_col: str = "unit_id",
    ) -> None:
        if spikes is None or spikes.empty:
            return
        tcol, ucol = resolve_spike_columns(spikes, time_col=time_col, unit_col=unit_col)
        self.extend(spikes[tcol].to_numpy(), spikes[ucol].to_numpy())

    # -- retention ------------------------------------------------------

    def _prune(self) -> None:
        """Drop everything older than ``clock - history_s``.

        Driven by the decoder clock set in ``counts_at``, never by incoming
        data, so a bad sample cannot evict real history.
        """
        if self._clock is None or self._times.size == 0:
            return
        cutoff = self._clock - self.history_s
        keep_from = int(np.searchsorted(self._times, cutoff, side="left"))
        if keep_from > 0:
            self._times = self._times[keep_from:]
            self._cols = self._cols[keep_from:]

    # -- query ----------------------------------------------------------

    def counts_at(self, t: float, decode_window_s: float) -> np.ndarray:
        """Length-|units| count vector for spikes in ``[t - W, t)``."""
        t = float(t)
        if not np.isfinite(t):
            raise ValueError(f"decode time must be finite, got {t}")
        self._clock = t if self._clock is None else max(self._clock, t)

        out = np.zeros(len(self.unit_ids), dtype=float)
        if self._times.size:
            t0 = t - float(decode_window_s)
            lo = int(np.searchsorted(self._times, t0, side="left"))
            hi = int(np.searchsorted(self._times, t, side="left"))
            if hi > lo:
                np.add.at(out, self._cols[lo:hi], 1.0)

        self._prune()
        return out

    # -- introspection --------------------------------------------------

    def as_dataframe(self) -> pd.DataFrame:
        """Retained spikes, always sorted ascending by time."""
        return pd.DataFrame(
            {
                "time": self._times,
                "unit_id": self.unit_ids[self._cols] if self._cols.size
                else np.empty(0, dtype=int),
            }
        )

    @property
    def n_spikes(self) -> int:
        return int(self._times.size)

    @property
    def clock(self) -> float | None:
        """Latest decode time seen, which drives retention."""
        return self._clock
