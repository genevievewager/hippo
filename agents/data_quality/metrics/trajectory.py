"""Shared trajectory preprocessing for lab and sim behaviour metrics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


# SPEC: same smoothing for both datasets before speed / accel / turning.
# Gaussian on x,y with σ = 0.10 s (≈ 2 samples at behavior_dt=0.05 s).
POS_SMOOTH_SIGMA_S = 0.10


def smooth_positions(pos: np.ndarray, times: np.ndarray, *, sigma_s: float = POS_SMOOTH_SIGMA_S) -> np.ndarray:
    """Gaussian-smooth x,y along time; NaNs linearly interpolated then restored."""
    pos = np.asarray(pos, float)
    times = np.asarray(times, float)
    out = pos.copy()
    if len(times) < 3:
        return out
    dt = float(np.median(np.diff(times)))
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.05
    sigma = max(sigma_s / dt, 0.5)
    for j in range(pos.shape[1]):
        col = pos[:, j].copy()
        ok = np.isfinite(col)
        if ok.sum() < 3:
            continue
        idx = np.arange(len(col))
        col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
        sm = gaussian_filter1d(col, sigma=sigma, mode="nearest")
        sm[~ok] = np.nan
        out[:, j] = sm
    return out


def speed_accel_from_pos(pos: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, float)
    pos = np.asarray(pos, float)
    speed = np.zeros(len(times))
    for i in range(1, len(times)):
        if np.isfinite(pos[i]).all() and np.isfinite(pos[i - 1]).all():
            speed[i] = np.linalg.norm(pos[i] - pos[i - 1]) / max(times[i] - times[i - 1], 1e-9)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    accel = np.diff(speed, prepend=speed[0]) / max(dt, 1e-9)
    return speed, accel


def tracking_jitter_at_rest(
    pos: np.ndarray,
    times: np.ndarray,
    *,
    rest_speed_max: float = 2.0,
) -> float:
    """Median frame-to-frame displacement (cm) among samples with speed < rest_speed_max."""
    pos = np.asarray(pos, float)
    times = np.asarray(times, float)
    if len(times) < 2:
        return float("nan")
    disp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = disp / np.maximum(np.diff(times), 1e-9)
    at_rest = speed < rest_speed_max
    if not at_rest.any():
        return float("nan")
    return float(np.median(disp[at_rest]))
