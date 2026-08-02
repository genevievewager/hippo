"""End-to-end latency profiling for causal realtime decoding stages.

Measures wall-clock cost of every stage that contributes to one decoder update
at ``dt_update`` (default 50 ms / 20 Hz budget).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from realtime.spike_binner import count_spikes_in_window

# Default closed-loop update budget (20 Hz).
DEFAULT_UPDATE_BUDGET_MS = 50.0

STAGE_ORDER = (
    "spike_binning",
    "feature_transform",
    "manifold_transform",
    "decode_position",
    "decode_speed",
    "decode_spatial_context",
    "decode_movement_state",
    "decode_primary",
    "closed_loop_policy",
    "total_update",
)


@dataclass
class LatencySample:
    """One timed causal update."""

    time_s: float
    stages_ms: dict[str, float] = field(default_factory=dict)


def _ms(t0: float, t1: float | None = None) -> float:
    end = time.perf_counter() if t1 is None else t1
    return float((end - t0) * 1000.0)


def summarize_latency_samples(
    samples: list[LatencySample],
    *,
    update_budget_ms: float = DEFAULT_UPDATE_BUDGET_MS,
) -> dict[str, Any]:
    """Aggregate per-stage latency statistics."""
    if not samples:
        return {
            "n_updates": 0,
            "update_budget_ms": update_budget_ms,
            "stages": {},
        }
    stage_names = sorted({k for s in samples for k in s.stages_ms})
    stages: dict[str, Any] = {}
    for name in stage_names:
        vals = np.asarray(
            [s.stages_ms[name] for s in samples if name in s.stages_ms],
            dtype=float,
        )
        if vals.size == 0:
            continue
        stages[name] = {
            "mean_ms": float(np.mean(vals)),
            "median_ms": float(np.median(vals)),
            "p95_ms": float(np.percentile(vals, 95)),
            "p99_ms": float(np.percentile(vals, 99)),
            "max_ms": float(np.max(vals)),
            "min_ms": float(np.min(vals)),
            "within_budget_frac": float(np.mean(vals <= update_budget_ms)),
        }
    total = stages.get("total_update", {})
    return {
        "n_updates": len(samples),
        "update_budget_ms": float(update_budget_ms),
        "mean_total_ms": total.get("mean_ms"),
        "median_total_ms": total.get("median_ms"),
        "p95_total_ms": total.get("p95_ms"),
        "within_budget_frac": total.get("within_budget_frac"),
        "stages": stages,
    }


def samples_to_dataframe(samples: list[LatencySample]) -> pd.DataFrame:
    rows = []
    for s in samples:
        row = {"time_s": s.time_s, **{f"{k}_ms": v for k, v in s.stages_ms.items()}}
        rows.append(row)
    return pd.DataFrame(rows)


def save_latency_artifacts(
    samples: list[LatencySample],
    output_dir: Path,
    *,
    update_budget_ms: float = DEFAULT_UPDATE_BUDGET_MS,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write per-update CSV + summary JSON under ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = samples_to_dataframe(samples)
    df.to_csv(output_dir / "latency_per_update.csv", index=False)
    summary = summarize_latency_samples(samples, update_budget_ms=update_budget_ms)
    if extra_meta:
        summary.update(extra_meta)
    # Long-form stage table for plotting
    stage_rows = []
    for name, stats in summary.get("stages", {}).items():
        stage_rows.append({"stage": name, **stats})
    stage_df = pd.DataFrame(stage_rows)
    if not stage_df.empty:
        # Stable plot order
        order = {s: i for i, s in enumerate(STAGE_ORDER)}
        stage_df["_ord"] = stage_df["stage"].map(lambda s: order.get(s, 100))
        stage_df = stage_df.sort_values("_ord").drop(columns="_ord")
        stage_df.to_csv(output_dir / "latency_by_stage.csv", index=False)
    with open(output_dir / "latency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def time_callable_ms(fn: Callable[[], Any], *, repeats: int = 1) -> tuple[Any, float]:
    """Return (last_result, mean_ms) over ``repeats`` calls."""
    result = None
    t0 = time.perf_counter()
    for _ in range(max(1, repeats)):
        result = fn()
    elapsed_ms = _ms(t0) / max(1, repeats)
    return result, elapsed_ms


def benchmark_feature_transforms(
    X_probe: np.ndarray,
    transformers: dict[str, Any],
    *,
    n_warmup: int = 5,
    n_repeats: int = 50,
) -> pd.DataFrame:
    """Per-row transform latency for fitted feature transformers."""
    X_probe = np.asarray(X_probe, dtype=float)
    rows = []
    for name, transformer in transformers.items():
        if transformer is None:
            continue
        # Warmup
        for i in range(min(n_warmup, len(X_probe))):
            transformer.transform(X_probe[i : i + 1])
        times = []
        for i in range(min(n_repeats, len(X_probe))):
            t0 = time.perf_counter()
            transformer.transform(X_probe[i : i + 1])
            times.append(_ms(t0))
        arr = np.asarray(times, dtype=float)
        meta = {}
        if hasattr(transformer, "get_metadata"):
            try:
                meta = transformer.get_metadata() or {}
            except Exception:
                meta = {}
        rows.append({
            "feature_mode": name,
            "mean_ms": float(np.mean(arr)),
            "median_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(np.max(arr)),
            "n_repeats": int(len(arr)),
            "realtime_compatible": bool(
                meta.get("realtime_compatible", True)
            ),
            "runtime_per_transform_ms": meta.get("runtime_per_transform_ms"),
            "manifold_type": meta.get("manifold_type"),
        })
    return pd.DataFrame(rows)


def profile_spike_binning_ms(
    spikes_df: pd.DataFrame,
    unit_ids: np.ndarray | list[int],
    t: float,
    decode_window: float,
) -> float:
    t0 = time.perf_counter()
    count_spikes_in_window(
        spikes_df,
        unit_ids,
        t - decode_window,
        t,
    )
    return _ms(t0)
