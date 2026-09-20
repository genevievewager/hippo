"""Synthetic and adversarial inputs.

Nothing here touches real lab data. Every generator is deterministic given a
seed so a finding reported on the server can be reproduced exactly.

The "adversarial" frames encode conditions that occur in real recordings and
that a hippocampal BCI must survive:

  * a unit that never fires in a window (sparse CA1 place cells off-field)
  * spikes arriving slightly out of order (concurrent sorter threads, ZMQ)
  * duplicate timestamps (two units, one sample; or a double-counted event)
  * a single bogus far-future timestamp (clock glitch / uninitialised sample)
  * float32 sample-clock rounding
  * Kilosort/Phy column naming (`spike_time`/`cluster_id`, `times`/`clusters`)
  * unit ids that are not 0..N-1 (Phy cluster ids are arbitrary and sparse)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SpikeSet:
    """A spike table plus the ground truth needed to check counting."""

    name: str
    frame: pd.DataFrame
    unit_ids: list[int]
    time_col: str
    unit_col: str
    note: str = ""

    def canonical(self) -> pd.DataFrame:
        """Sorted frame with canonical `time`/`unit_id` columns."""
        df = self.frame.rename(columns={self.time_col: "time", self.unit_col: "unit_id"})
        return df.sort_values("time", kind="mergesort").reset_index(drop=True)

    def truth_counts(self, t_start: float, t_end: float) -> np.ndarray:
        """Independent ground-truth count for [t_start, t_end), by brute force.

        Deliberately does NOT use searchsorted or any repo code — this is the
        oracle the pipeline is measured against.
        """
        df = self.canonical()
        out = np.zeros(len(self.unit_ids), dtype=float)
        idx = {int(u): i for i, u in enumerate(self.unit_ids)}
        for t, u in zip(df["time"].to_numpy(dtype=float), df["unit_id"].to_numpy()):
            if t_start <= t < t_end:
                j = idx.get(int(u))
                if j is not None:
                    out[j] += 1.0
        return out


def poisson_spikes(
    *,
    n_units: int = 12,
    duration_s: float = 20.0,
    rate_hz: float = 8.0,
    seed: int = 0,
    unit_id_offset: int = 0,
) -> SpikeSet:
    """Well-behaved sorted Poisson spikes — the happy path."""
    rng = np.random.default_rng(seed)
    unit_ids = [unit_id_offset + i for i in range(n_units)]
    times: list[float] = []
    units: list[int] = []
    for u in unit_ids:
        n = rng.poisson(rate_hz * duration_s)
        t = np.sort(rng.uniform(0.0, duration_s, size=int(n)))
        times.extend(t.tolist())
        units.extend([u] * int(n))
    df = pd.DataFrame({"time": times, "unit_id": units}).sort_values(
        "time", kind="mergesort"
    ).reset_index(drop=True)
    return SpikeSet("poisson", df, unit_ids, "time", "unit_id", "sorted Poisson baseline")


def adversarial_sets(seed: int = 0) -> list[SpikeSet]:
    """The set of hostile-but-realistic spike tables."""
    rng = np.random.default_rng(seed)
    out: list[SpikeSet] = []

    base = poisson_spikes(n_units=6, duration_s=5.0, rate_hz=10.0, seed=seed)
    out.append(base)

    # 1. Unsorted arrival order.
    shuffled = base.frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out.append(
        SpikeSet(
            "unsorted",
            shuffled,
            base.unit_ids,
            "time",
            "unit_id",
            "spike times not monotonically increasing",
        )
    )

    # 2. Locally out of order: a jitter window, as from a threaded sorter.
    jit = base.frame.copy()
    k = max(4, len(jit) // 20)
    idx = rng.choice(len(jit) - 1, size=k, replace=False)
    vals = jit["time"].to_numpy(dtype=float).copy()
    vals[idx], vals[idx + 1] = vals[idx + 1], vals[idx]
    jit["time"] = vals
    out.append(
        SpikeSet(
            "locally_out_of_order",
            jit,
            base.unit_ids,
            "time",
            "unit_id",
            "adjacent pairs swapped, as with concurrent sorter output",
        )
    )

    # 3. Duplicate timestamps.
    dup = pd.concat([base.frame, base.frame.head(25)], ignore_index=True)
    dup = dup.sort_values("time", kind="mergesort").reset_index(drop=True)
    out.append(
        SpikeSet("duplicate_timestamps", dup, base.unit_ids, "time", "unit_id",
                 "repeated (time, unit) events")
    )

    # 4. One bogus far-future timestamp (clock glitch).
    glitch = pd.concat(
        [base.frame, pd.DataFrame({"time": [1.0e9], "unit_id": [base.unit_ids[0]]})],
        ignore_index=True,
    )
    out.append(
        SpikeSet("clock_glitch", glitch, base.unit_ids, "time", "unit_id",
                 "single far-future sample, as from an uninitialised clock")
    )

    # 5. Empty.
    out.append(
        SpikeSet(
            "empty",
            pd.DataFrame({"time": pd.Series(dtype=float), "unit_id": pd.Series(dtype=int)}),
            base.unit_ids,
            "time",
            "unit_id",
            "no spikes at all",
        )
    )

    # 6. Single unit, single spike.
    out.append(
        SpikeSet("single_spike", pd.DataFrame({"time": [1.0], "unit_id": [base.unit_ids[0]]}),
                 base.unit_ids, "time", "unit_id", "degenerate minimum input")
    )

    # 7. Silent unit: a listed unit that never fires.
    silent_ids = base.unit_ids + [max(base.unit_ids) + 1]
    out.append(
        SpikeSet("silent_unit", base.frame, silent_ids, "time", "unit_id",
                 "a unit in units.csv with zero spikes (off-field place cell)")
    )

    # 8. float32 sample-clock rounding.
    f32 = base.frame.copy()
    f32["time"] = f32["time"].to_numpy(dtype=np.float32).astype(np.float64)
    out.append(
        SpikeSet("float32_clock", f32, base.unit_ids, "time", "unit_id",
                 "timestamps quantised through float32")
    )

    # 9. Negative / pre-trigger timestamps.
    neg = base.frame.copy()
    neg["time"] = neg["time"] - 2.5
    out.append(
        SpikeSet("negative_times", neg, base.unit_ids, "time", "unit_id",
                 "pre-trigger timestamps relative to a sync pulse")
    )

    return out


# --- Kilosort / Phy shaped frames ------------------------------------------

PHY_SCHEMAS: list[tuple[str, dict[str, str], str]] = [
    ("phy_spike_time_cluster_id", {"time": "spike_time", "unit_id": "cluster_id"},
     "phy `spike_times.npy` + `spike_clusters.npy` exported to a table"),
    ("phy_times_clusters", {"time": "times", "unit_id": "clusters"},
     "SpikeInterface `sorting.to_dataframe()` naming"),
    ("ks_spike_time_s_unit", {"time": "spike_time_s", "unit_id": "unit"},
     "Kilosort seconds-converted export"),
    ("canonical", {}, "repo-canonical `time` / `unit_id`"),
]


def phy_like(schema: str, *, seed: int = 0, sparse_ids: bool = True) -> SpikeSet:
    """A sorted-spike table as it actually comes out of Kilosort/Phy."""
    name, mapping, note = next(s for s in PHY_SCHEMAS if s[0] == schema)
    base = poisson_spikes(n_units=5, duration_s=4.0, rate_hz=9.0, seed=seed)
    df = base.frame.copy()
    unit_ids = base.unit_ids
    if sparse_ids:
        # Phy cluster ids are arbitrary, sparse and never guaranteed 0..N-1.
        remap = {u: int(v) for u, v in zip(unit_ids, [3, 11, 12, 47, 108])}
        df["unit_id"] = df["unit_id"].map(remap)
        unit_ids = sorted(remap.values())
    if mapping:
        # `mapping` is canonical -> vendor name; the frame starts canonical.
        df = df.rename(columns=mapping)
    time_col = mapping.get("time", "time")
    unit_col = mapping.get("unit_id", "unit_id")
    return SpikeSet(name, df, unit_ids, time_col, unit_col, note)


def units_frame(unit_ids: list[int], *, seed: int = 0) -> pd.DataFrame:
    """A minimal units.csv-shaped table."""
    rng = np.random.default_rng(seed)
    regions = ["CA1", "CA3", "DG", "SUB", "MEC"]
    return pd.DataFrame(
        {
            "unit_id": unit_ids,
            "region": [regions[i % len(regions)] for i in range(len(unit_ids))],
            "depth_um": rng.uniform(0, 3000, size=len(unit_ids)),
            "cell_class": ["place" if i % 3 else "interneuron" for i in range(len(unit_ids))],
        }
    )


def behavior_frame(duration_s: float = 20.0, dt: float = 0.02, *, seed: int = 0) -> pd.DataFrame:
    """Open-field behaviour: smooth 2D trajectory plus derived variables."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, duration_s, dt)
    x = np.cumsum(rng.normal(0, 0.02, t.size))
    y = np.cumsum(rng.normal(0, 0.02, t.size))
    x = (x - x.min()) / max(np.ptp(x), 1e-9)
    y = (y - y.min()) / max(np.ptp(y), 1e-9)
    vx, vy = np.gradient(x, dt), np.gradient(y, dt)
    return pd.DataFrame(
        {
            "time": t,
            "x": x,
            "y": y,
            "speed": np.hypot(vx, vy),
            "head_direction": np.arctan2(vy, vx),
        }
    )
