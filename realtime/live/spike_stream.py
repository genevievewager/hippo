"""Spike stream abstractions: replay and Open Ephys adapters."""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class SpikeStream(abc.ABC):
    """Source of sorted spike events for live / replay inference."""

    @abc.abstractmethod
    def connect(self) -> None:
        ...

    @abc.abstractmethod
    def disconnect(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def connected(self) -> bool:
        ...

    @abc.abstractmethod
    def get_new_spikes(self, *, up_to_time: float | None = None) -> pd.DataFrame:
        """Return new spike events since the last poll.

        Columns: ``time`` (s), ``unit_id`` (int). Empty frame if none.
        """

    @abc.abstractmethod
    def list_unit_ids(self) -> list[int]:
        ...

    @property
    def source_name(self) -> str:
        return type(self).__name__


class ReplaySpikeStream(SpikeStream):
    """Replay stored sorted spikes as if they were arriving online.

    Advancing is driven by calling ``get_new_spikes(up_to_time=t)`` — the
    client (LiveDecoder loop) controls the virtual clock.
    """

    def __init__(
        self,
        spikes_df: pd.DataFrame | None = None,
        *,
        experiment_dir: Path | str | None = None,
        spike_source: str = "sorted",
    ):
        self.experiment_dir = Path(experiment_dir) if experiment_dir is not None else None
        self.spike_source = spike_source
        self._spikes: pd.DataFrame | None = spikes_df
        self._cursor = 0
        self._connected = False
        self._unit_ids: list[int] = []

    def connect(self) -> None:
        if self._spikes is None:
            if self.experiment_dir is None:
                raise ValueError("ReplaySpikeStream needs spikes_df or experiment_dir")
            from realtime.data_loading import load_simulation_data

            data = load_simulation_data(self.experiment_dir, self.spike_source)
            self._spikes = data["spikes_df"].copy()
            self._unit_ids = [int(u) for u in data["unit_ids"]]
        else:
            u = self._spikes.get("unit_id", self._spikes.get("unit"))
            self._unit_ids = sorted({int(x) for x in np.asarray(u).ravel().tolist()})
        # Normalize columns
        df = self._spikes
        if "time" not in df.columns:
            for c in ("time_s", "spike_time", "t"):
                if c in df.columns:
                    df = df.rename(columns={c: "time"})
                    break
        if "unit_id" not in df.columns:
            for c in ("unit", "cluster_id"):
                if c in df.columns:
                    df = df.rename(columns={c: "unit_id"})
                    break
        self._spikes = df.sort_values("time").reset_index(drop=True)
        self._cursor = 0
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def list_unit_ids(self) -> list[int]:
        return list(self._unit_ids)

    def seek(self, t: float) -> None:
        """Move the replay cursor so the next poll starts near time ``t``."""
        if self._spikes is None or self._spikes.empty:
            self._cursor = 0
            return
        times = self._spikes["time"].to_numpy(dtype=float)
        self._cursor = int(np.searchsorted(times, float(t), side="left"))

    def get_new_spikes(self, *, up_to_time: float | None = None) -> pd.DataFrame:
        if not self._connected or self._spikes is None or self._spikes.empty:
            return pd.DataFrame(columns=["time", "unit_id"])
        if up_to_time is None:
            # Deliver everything remaining (batch drain) — unusual for live loop.
            chunk = self._spikes.iloc[self._cursor :].copy()
            self._cursor = len(self._spikes)
            return chunk[["time", "unit_id"]]
        times = self._spikes["time"].to_numpy(dtype=float)
        end = int(np.searchsorted(times, float(up_to_time), side="left"))
        if end <= self._cursor:
            return pd.DataFrame(columns=["time", "unit_id"])
        chunk = self._spikes.iloc[self._cursor:end][["time", "unit_id"]].copy()
        self._cursor = end
        return chunk

    @property
    def t_start(self) -> float:
        if self._spikes is None or self._spikes.empty:
            return 0.0
        return float(self._spikes["time"].iloc[0])

    @property
    def t_end(self) -> float:
        if self._spikes is None or self._spikes.empty:
            return 0.0
        return float(self._spikes["time"].iloc[-1])

    @property
    def source_name(self) -> str:
        return f"Replay({self.experiment_dir.name if self.experiment_dir else 'memory'})"


class OpenEphysSpikeStream(SpikeStream):
    """Adapter stub for live Open Ephys sorted-spike streaming.

    TODO: Wire to the laboratory Open Ephys / SpikeInterface endpoint once the
    concrete streaming interface is available. This stub keeps acquisition
    decoupled from ``LiveDecoder`` so the rest of the live pipeline can be
    developed and tested via ``ReplaySpikeStream``.

    Expected eventual responsibilities:
    - connect to the OE ZMQ / SpikeInterface live sorter feed
    - normalize events to columns ``time``, ``unit_id``
    - expose currently active sorted cluster IDs via ``list_unit_ids``
    """

    def __init__(self, *, endpoint: str | None = None, config: dict[str, Any] | None = None):
        self.endpoint = endpoint
        self.config = config or {}
        self._connected = False
        self._unit_ids: list[int] = []

    def connect(self) -> None:
        # Intentionally not implemented for hardware — fail loudly rather than
        # silently pretending acquisition works.
        raise NotImplementedError(
            "OpenEphysSpikeStream is a stub. Use ReplaySpikeStream for pipeline "
            "tests, or implement the laboratory Open Ephys streaming connector "
            f"(endpoint={self.endpoint!r})."
        )

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def list_unit_ids(self) -> list[int]:
        return list(self._unit_ids)

    def get_new_spikes(self, *, up_to_time: float | None = None) -> pd.DataFrame:
        if not self._connected:
            return pd.DataFrame(columns=["time", "unit_id"])
        raise NotImplementedError("OpenEphysSpikeStream.get_new_spikes not connected to hardware")

    @property
    def source_name(self) -> str:
        return f"OpenEphys(stub:{self.endpoint or 'unset'})"
