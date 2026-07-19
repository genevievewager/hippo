"""Decoder timing: update interval, behavior alignment, temporal config.

Notation
--------
dt_update : decoder update interval (default = behavioral frame period, 0.050 s)
W         : neural integration window for causal spike counts in [t-W, t)
L         : latent temporal history in video frames
tau       : optional neural-to-behavior prediction lag (tau >= 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_BEHAVIOR_SAMPLING_RATE_HZ = 20.0
DEFAULT_UPDATE_DT_S = 1.0 / DEFAULT_BEHAVIOR_SAMPLING_RATE_HZ  # 0.050
DEFAULT_ALIGNMENT_TOLERANCE_S = 0.005
DEFAULT_INTEGRATION_WINDOWS_S = (0.050, 0.100, 0.250, 0.500, 1.000)
DEFAULT_LATENT_HISTORY_FRAMES = (1, 2, 5, 10, 20)
DEFAULT_PREDICTION_LAGS_S = (0.0, 0.050, 0.100, 0.150, 0.250)


@dataclass(frozen=True)
class TemporalDecodingConfig:
    """Timing hyperparameters with clearly separated roles."""

    update_dt_s: float = DEFAULT_UPDATE_DT_S
    integration_window_s: float = 0.250
    latent_history_frames: int = 1
    prediction_lag_s: float = 0.0

    @property
    def latent_history_s(self) -> float:
        return float(self.latent_history_frames) * float(self.update_dt_s)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "update_dt_s": float(self.update_dt_s),
            "integration_window_s": float(self.integration_window_s),
            "latent_history_frames": int(self.latent_history_frames),
            "latent_history_s": float(self.latent_history_s),
            "prediction_lag_s": float(self.prediction_lag_s),
        }


@dataclass
class TimingValidationResult:
    """Result of behavioral timestamp validation."""

    n_timestamps: int
    median_dt_s: float
    mean_dt_s: float
    expected_dt_s: float
    update_dt_s: float
    n_duplicates: int
    n_non_monotonic: int
    max_gap_s: float
    ok: bool
    messages: list[str]


def extract_behavior_times(behavior_df: pd.DataFrame) -> np.ndarray:
    """Return behavioral timestamps in seconds from a behavior dataframe."""
    for col in ("time_s", "time", "t", "timestamp"):
        if col in behavior_df.columns:
            return behavior_df[col].to_numpy(dtype=float)
    raise ValueError(
        f"No behavior time column found. Columns: {list(behavior_df.columns)}"
    )


def resolve_update_dt_s(
    summary: dict[str, Any] | None = None,
    *,
    behavior_sampling_rate_hz: float | None = None,
    derive_from_behavior: bool = True,
    update_dt_s: float | None = None,
    behavior_times: np.ndarray | None = None,
) -> float:
    """
    Resolve decoder update interval.

    Preference order when derive_from_behavior is True:
    1. Explicit update_dt_s if provided and derive_from_behavior is False
    2. summary['behavior_dt']
    3. 1 / behavior_sampling_rate_hz
    4. median interval of behavior_times
    5. DEFAULT_UPDATE_DT_S
    """
    if update_dt_s is not None and not derive_from_behavior:
        return float(update_dt_s)

    if summary is not None and summary.get("behavior_dt") is not None:
        return float(summary["behavior_dt"])

    if behavior_sampling_rate_hz is not None and behavior_sampling_rate_hz > 0:
        return 1.0 / float(behavior_sampling_rate_hz)

    if behavior_times is not None and len(behavior_times) >= 2:
        dts = np.diff(np.asarray(behavior_times, dtype=float))
        dts = dts[np.isfinite(dts) & (dts > 0)]
        if len(dts) > 0:
            return float(np.median(dts))

    if update_dt_s is not None:
        return float(update_dt_s)

    return float(DEFAULT_UPDATE_DT_S)


def validate_behavior_timestamps(
    behavior_times: np.ndarray,
    *,
    expected_dt_s: float = DEFAULT_UPDATE_DT_S,
    alignment_tolerance_s: float = DEFAULT_ALIGNMENT_TOLERANCE_S,
) -> TimingValidationResult:
    """Validate monotonicity, duplicates, and approximate sampling interval."""
    t = np.asarray(behavior_times, dtype=float)
    messages: list[str] = []
    if t.ndim != 1 or len(t) < 2:
        raise ValueError("behavior_times must be a 1D array with at least 2 samples")

    diffs = np.diff(t)
    n_non_monotonic = int(np.sum(diffs < -alignment_tolerance_s))
    n_duplicates = int(np.sum(np.abs(diffs) <= 1e-12))
    positive = diffs[diffs > 1e-12]
    median_dt = float(np.median(positive)) if len(positive) else float("nan")
    mean_dt = float(np.mean(positive)) if len(positive) else float("nan")
    max_gap = float(np.max(positive)) if len(positive) else float("nan")

    if n_non_monotonic > 0:
        messages.append(f"{n_non_monotonic} non-monotonic timestamp steps")
    if n_duplicates > 0:
        messages.append(f"{n_duplicates} duplicate timestamps")
    if not np.isfinite(median_dt):
        messages.append("could not estimate median behavioral interval")
    elif abs(median_dt - expected_dt_s) > alignment_tolerance_s:
        messages.append(
            f"median behavioral interval {median_dt:.6f}s differs from "
            f"expected {expected_dt_s:.6f}s by more than {alignment_tolerance_s}s"
        )

    ok = n_non_monotonic == 0 and len(messages) == 0
    return TimingValidationResult(
        n_timestamps=len(t),
        median_dt_s=median_dt,
        mean_dt_s=mean_dt,
        expected_dt_s=float(expected_dt_s),
        update_dt_s=float(expected_dt_s),
        n_duplicates=n_duplicates,
        n_non_monotonic=n_non_monotonic,
        max_gap_s=max_gap,
        ok=ok,
        messages=messages,
    )


def make_behavior_aligned_decode_times(
    behavior_times: np.ndarray,
    integration_window_s: float,
    *,
    summary: dict[str, Any] | None = None,
    expected_dt_s: float | None = None,
    alignment_tolerance_s: float = DEFAULT_ALIGNMENT_TOLERANCE_S,
    strict: bool = False,
) -> tuple[np.ndarray, TimingValidationResult]:
    """
    Use original behavioral timestamps as decoder update times.

    Keeps irregular timestamps when present. Drops samples before the
    integration window is fully available (t < W).
    """
    t = np.asarray(behavior_times, dtype=float)
    expected = resolve_update_dt_s(
        summary,
        derive_from_behavior=True,
        update_dt_s=expected_dt_s,
        behavior_times=t,
    )
    validation = validate_behavior_timestamps(
        t, expected_dt_s=expected, alignment_tolerance_s=alignment_tolerance_s,
    )
    if strict and not validation.ok:
        raise ValueError(
            "Behavior timestamp validation failed: " + "; ".join(validation.messages)
        )

    decode_times = t[t >= float(integration_window_s) - 1e-12]
    if len(decode_times) == 0:
        raise ValueError(
            f"No behavioral timestamps >= integration_window_s={integration_window_s}"
        )
    return decode_times, validation


def assert_alignment(
    decode_times: np.ndarray,
    behavior_times: np.ndarray,
    *,
    alignment_tolerance_s: float = DEFAULT_ALIGNMENT_TOLERANCE_S,
) -> float:
    """
    Ensure every decode time matches a behavioral timestamp within tolerance.

    Returns the maximum absolute alignment error.
    """
    decode_times = np.asarray(decode_times, dtype=float)
    behavior_times = np.asarray(behavior_times, dtype=float)
    idx = np.searchsorted(behavior_times, decode_times, side="left")
    idx = np.clip(idx, 0, len(behavior_times) - 1)
    prev = np.clip(idx - 1, 0, len(behavior_times) - 1)
    choose_prev = np.abs(behavior_times[prev] - decode_times) <= np.abs(
        behavior_times[idx] - decode_times
    )
    nearest = np.where(choose_prev, prev, idx)
    errors = np.abs(behavior_times[nearest] - decode_times)
    max_err = float(np.max(errors)) if len(errors) else 0.0
    if max_err > alignment_tolerance_s:
        raise ValueError(
            f"Decoder/behavior alignment error {max_err:.6f}s exceeds "
            f"tolerance {alignment_tolerance_s}s"
        )
    return max_err
