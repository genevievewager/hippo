"""Kilosort / Phy sorted-spike ingest.

Phy writes spike times as **sample indices**, not seconds. At 30 kHz that is a
30,000x error, and it does not announce itself: indices interpreted as seconds
put every spike hours past the end of the session, so the causal windows come
back empty and the decoder trains on silence. Nothing downstream can tell that
apart from a quiet recording.

So this module refuses to guess. It reads the sampling rate from `params.py`,
or takes one explicitly, and validates that the resulting session duration is
physically plausible before handing anything on.

Directory layout it expects (standard Kilosort/Phy output)::

    spike_times.npy       int64 sample indices, ascending
    spike_clusters.npy    int32 cluster id per spike
    params.py             contains `sample_rate = 30000.`
    cluster_group.tsv     optional: cluster_id -> good / mua / noise
    cluster_KSLabel.tsv   optional: Kilosort's own labels
    cluster_info.tsv      optional: depth, channel, firing rate per cluster
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realtime.live.spike_stream import SpikeStream

# A session outside this range means the time units are wrong, not that the
# recording was unusual.
MIN_PLAUSIBLE_SESSION_S = 1.0
MAX_PLAUSIBLE_SESSION_S = 24 * 3600.0

# Kilosort/Phy quality labels that may enter analysis.
DEFAULT_ACCEPTED_GROUPS: tuple[str, ...] = ("good",)


class PhyIngestError(ValueError):
    """Raised when a Phy directory cannot be read unambiguously."""


def read_sample_rate(phy_dir: Path) -> float:
    """Parse `sample_rate` out of Phy's params.py.

    params.py is executable Python, but executing a file from a data directory
    to read one number is not a trade worth making. Parse it.
    """
    params = Path(phy_dir) / "params.py"
    if not params.exists():
        raise PhyIngestError(
            f"No params.py in {phy_dir}. Pass fs= explicitly, but be certain of "
            "it: spike_times.npy holds sample indices and the wrong rate scales "
            "every timestamp without any error."
        )
    text = params.read_text(errors="ignore")
    m = re.search(r"^\s*sample_rate\s*=\s*([0-9]+(?:\.[0-9]*)?)", text, re.M)
    if not m:
        raise PhyIngestError(f"No sample_rate found in {params}. Pass fs= explicitly.")
    fs = float(m.group(1))
    if not (1_000.0 <= fs <= 100_000.0):
        raise PhyIngestError(
            f"sample_rate={fs} in {params} is outside any plausible ephys range."
        )
    return fs


def read_cluster_groups(phy_dir: Path) -> pd.DataFrame:
    """Cluster quality labels, from cluster_group.tsv or cluster_KSLabel.tsv."""
    phy_dir = Path(phy_dir)
    for name, col in (
        ("cluster_group.tsv", "group"),        # curated in Phy, takes precedence
        ("cluster_KSLabel.tsv", "KSLabel"),    # Kilosort's own
    ):
        path = phy_dir / name
        if not path.exists():
            continue
        df = pd.read_csv(path, sep="\t")
        idcol = next((c for c in ("cluster_id", "id") if c in df.columns), None)
        labcol = next((c for c in (col, "group", "KSLabel") if c in df.columns), None)
        if idcol is None or labcol is None:
            continue
        return pd.DataFrame(
            {
                "unit_id": df[idcol].astype(int),
                "quality": df[labcol].astype(str).str.strip().str.lower(),
                "quality_source": name,
            }
        )
    return pd.DataFrame(columns=["unit_id", "quality", "quality_source"])


def load_phy_spikes(
    phy_dir: Path | str,
    *,
    fs: float | None = None,
    accepted_groups: tuple[str, ...] | None = DEFAULT_ACCEPTED_GROUPS,
    t0_s: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a Phy directory into a canonical spikes frame plus metadata.

    Returns ``(spikes_df, meta)`` where spikes_df has columns ``time`` (seconds)
    and ``unit_id``, sorted ascending by time.

    `accepted_groups=None` keeps every cluster including noise — diagnostic
    only; it is not a decoding configuration.
    """
    phy_dir = Path(phy_dir)
    times_path = phy_dir / "spike_times.npy"
    clusters_path = phy_dir / "spike_clusters.npy"
    for p in (times_path, clusters_path):
        if not p.exists():
            raise PhyIngestError(f"Required Phy file not found: {p}")

    fs = float(fs) if fs is not None else read_sample_rate(phy_dir)

    samples = np.asarray(np.load(times_path)).ravel()
    clusters = np.asarray(np.load(clusters_path)).ravel().astype(np.int64)
    if samples.size != clusters.size:
        raise PhyIngestError(
            f"spike_times.npy has {samples.size} entries but spike_clusters.npy "
            f"has {clusters.size}. These files must be parallel."
        )
    if samples.size == 0:
        raise PhyIngestError(f"{times_path} is empty — nothing to decode.")

    if np.issubdtype(samples.dtype, np.floating) and not np.isfinite(samples).all():
        raise PhyIngestError(
            f"{int((~np.isfinite(samples)).sum())} non-finite entries in "
            f"{times_path}. Corrupt sorter output."
        )

    times_s = samples.astype(np.float64) / fs + float(t0_s)

    # The plausibility gate. This is the whole point of the module.
    duration = float(times_s.max() - times_s.min())
    if not (MIN_PLAUSIBLE_SESSION_S <= duration <= MAX_PLAUSIBLE_SESSION_S):
        raise PhyIngestError(
            f"Converted session duration is {duration:.6g} s, outside the "
            f"plausible range [{MIN_PLAUSIBLE_SESSION_S}, "
            f"{MAX_PLAUSIBLE_SESSION_S}] s.\n"
            f"  fs used: {fs} Hz\n"
            f"  raw sample index range: {samples.min()} .. {samples.max()}\n"
            "This nearly always means the sampling rate is wrong, or the file "
            "already holds seconds rather than sample indices. Either way the "
            "timestamps would be wrong by a large constant factor and every "
            "causal window would be silently misaligned."
        )

    df = pd.DataFrame({"time": times_s, "unit_id": clusters})

    groups = read_cluster_groups(phy_dir)
    kept_note = "all clusters (no quality file)"
    if accepted_groups is not None and not groups.empty:
        accepted = {g.lower() for g in accepted_groups}
        keep_ids = set(groups.loc[groups["quality"].isin(accepted), "unit_id"].tolist())
        before = int(df["unit_id"].nunique())
        df = df[df["unit_id"].isin(keep_ids)].copy()
        after = int(df["unit_id"].nunique())
        kept_note = f"{after} of {before} clusters with quality in {sorted(accepted)}"
        if df.empty:
            raise PhyIngestError(
                f"No clusters remain after filtering to {sorted(accepted)}. "
                f"Labels present: {sorted(groups['quality'].unique().tolist())}. "
                "Pass accepted_groups to widen this, or curate in Phy first."
            )

    df = df.sort_values("time", kind="mergesort").reset_index(drop=True)

    meta = {
        "phy_dir": str(phy_dir),
        "fs_hz": fs,
        "fs_source": "params.py" if fs is None else "explicit-or-params",
        "n_spikes": int(len(df)),
        "n_units": int(df["unit_id"].nunique()),
        "t_start_s": float(df["time"].iloc[0]),
        "t_end_s": float(df["time"].iloc[-1]),
        "duration_s": float(df["time"].iloc[-1] - df["time"].iloc[0]),
        "quality_filter": kept_note,
    }
    return df, meta


def build_units_table(
    phy_dir: Path | str,
    unit_ids: list[int] | np.ndarray,
    *,
    region_by_unit: dict[int, str] | None = None,
    depth_to_region: "list[tuple[float, float, str]] | None" = None,
    cell_type: str = "CA1_pyr",
) -> pd.DataFrame:
    """Build a units.csv-shaped table for a Phy sort.

    Phy knows nothing about anatomy, so a region must come from somewhere:
    either an explicit ``region_by_unit`` mapping, or ``depth_to_region``
    (a list of ``(depth_start_um, depth_end_um, region)`` from the probe
    trajectory table) applied to each cluster's depth in cluster_info.tsv.

    Without one of those, every unit is `unknown` and load_simulation_data
    will refuse the dataset — which is the correct outcome, not a bug.
    """
    phy_dir = Path(phy_dir)
    unit_ids = [int(u) for u in np.asarray(unit_ids).ravel()]

    depths: dict[int, float] = {}
    info_path = phy_dir / "cluster_info.tsv"
    if info_path.exists():
        info = pd.read_csv(info_path, sep="\t")
        idcol = next((c for c in ("cluster_id", "id") if c in info.columns), None)
        depthcol = next((c for c in ("depth", "Depth") if c in info.columns), None)
        if idcol and depthcol:
            depths = {
                int(r[idcol]): float(r[depthcol])
                for _, r in info.iterrows()
                if pd.notna(r[depthcol])
            }

    def _region(uid: int) -> str:
        if region_by_unit and uid in region_by_unit:
            return str(region_by_unit[uid])
        if depth_to_region and uid in depths:
            d = depths[uid]
            for lo, hi, name in depth_to_region:
                if float(lo) <= d < float(hi):
                    return str(name)
        return "unknown"

    groups = read_cluster_groups(phy_dir).set_index("unit_id")["quality"].to_dict()
    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "region": [_region(u) for u in unit_ids],
            "cell_type": [cell_type] * len(unit_ids),
            "depth_um": [depths.get(u, np.nan) for u in unit_ids],
            "phy_quality": [groups.get(u, "") for u in unit_ids],
        }
    )


class PhySpikeStream(SpikeStream):
    """Replay a Phy sort as if it were arriving online.

    Same contract as ReplaySpikeStream — the caller drives the virtual clock
    via ``get_new_spikes(up_to_time=t)`` — but sourced from Kilosort/Phy output
    with explicit sampling-rate handling.

    ``follow=True`` re-reads spike_times.npy on each poll, so a sorter still
    appending to the directory is picked up. That is the closest thing to live
    acquisition available without a streaming sorter endpoint, and unlike the
    Open Ephys stub it actually runs.
    """

    def __init__(
        self,
        phy_dir: Path | str,
        *,
        fs: float | None = None,
        accepted_groups: tuple[str, ...] | None = DEFAULT_ACCEPTED_GROUPS,
        t0_s: float = 0.0,
        follow: bool = False,
    ):
        self.phy_dir = Path(phy_dir)
        self.fs = fs
        self.accepted_groups = accepted_groups
        self.t0_s = float(t0_s)
        self.follow = bool(follow)
        self._spikes: pd.DataFrame | None = None
        self._meta: dict[str, Any] = {}
        self._cursor = 0
        self._connected = False
        self._unit_ids: list[int] = []

    # -- SpikeStream interface -----------------------------------------

    def connect(self) -> None:
        df, meta = load_phy_spikes(
            self.phy_dir, fs=self.fs, accepted_groups=self.accepted_groups,
            t0_s=self.t0_s,
        )
        self._spikes = df
        self._meta = meta
        self.fs = meta["fs_hz"]
        self._unit_ids = sorted(int(u) for u in df["unit_id"].unique())
        self._cursor = 0
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def list_unit_ids(self) -> list[int]:
        return list(self._unit_ids)

    def get_new_spikes(self, *, up_to_time: float | None = None) -> pd.DataFrame:
        empty = pd.DataFrame({"time": pd.Series(dtype=float),
                              "unit_id": pd.Series(dtype=int)})
        if not self._connected or self._spikes is None or self._spikes.empty:
            return empty
        if self.follow:
            self._refresh()
        if up_to_time is None:
            chunk = self._spikes.iloc[self._cursor:].copy()
            self._cursor = len(self._spikes)
            return chunk[["time", "unit_id"]]
        times = self._spikes["time"].to_numpy(dtype=float)
        end = int(np.searchsorted(times, float(up_to_time), side="left"))
        if end <= self._cursor:
            return empty
        chunk = self._spikes.iloc[self._cursor:end][["time", "unit_id"]].copy()
        self._cursor = end
        return chunk

    # -- extras ---------------------------------------------------------

    def _refresh(self) -> None:
        """Re-read the sort; keep the cursor pointed at the same instant."""
        try:
            df, meta = load_phy_spikes(
                self.phy_dir, fs=self.fs, accepted_groups=self.accepted_groups,
                t0_s=self.t0_s,
            )
        except PhyIngestError:
            return  # mid-write; try again next poll
        if self._spikes is not None and self._cursor:
            t_at_cursor = float(
                self._spikes["time"].iloc[min(self._cursor, len(self._spikes)) - 1]
            )
            self._cursor = int(
                np.searchsorted(df["time"].to_numpy(dtype=float), t_at_cursor,
                                side="right")
            )
        self._spikes = df
        self._meta = meta
        self._unit_ids = sorted(int(u) for u in df["unit_id"].unique())

    def seek(self, t: float) -> None:
        if self._spikes is None or self._spikes.empty:
            self._cursor = 0
            return
        times = self._spikes["time"].to_numpy(dtype=float)
        self._cursor = int(np.searchsorted(times, float(t), side="left"))

    @property
    def meta(self) -> dict[str, Any]:
        return dict(self._meta)

    @property
    def t_start(self) -> float:
        return float(self._meta.get("t_start_s", 0.0))

    @property
    def t_end(self) -> float:
        return float(self._meta.get("t_end_s", 0.0))

    @property
    def source_name(self) -> str:
        return f"Phy({self.phy_dir.name} @ {self.fs or '?'} Hz)"
