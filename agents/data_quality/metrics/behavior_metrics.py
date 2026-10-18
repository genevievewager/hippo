"""Behaviour metrics (speed/accel/turning from shared-smoothed positions)."""
from __future__ import annotations

from typing import Any

import numpy as np

from agents.data_quality.metrics.schema import tidy_row
from agents.data_quality.metrics.trajectory import (
    POS_SMOOTH_SIGMA_S,
    smooth_positions,
    speed_accel_from_pos,
    tracking_jitter_at_rest,
)


def behavior_metric_rows(
    pos,
    times,
    *,
    dataset,
    animal_or_seed,
    session,
    chunk,
    region_set,
    arena_half_width_cm,
    already_smoothed: bool = False,
    dt: float | None = None,
) -> list[dict[str, Any]]:
    """Speed / accel / turning from Gaussian-smoothed positions (σ=0.10 s).

    Filter (lab and sim identical): ``gaussian_filter1d`` on x and y with
    ``sigma_s = 0.10`` (≈ 2 samples at 50 ms). Speed/accel/turning derived
    from the smoothed trajectory. Tracking jitter uses raw positions.
    """
    pos = np.asarray(pos, float)
    times = np.asarray(times, float)
    if pos.shape[0] < 10:
        return []
    pos_s = pos if already_smoothed else smooth_positions(pos, times, sigma_s=POS_SMOOTH_SIGMA_S)
    speed, acc = speed_accel_from_pos(pos_s, times)
    jitter = tracking_jitter_at_rest(pos, times)
    hw = float(arena_half_width_cm)
    wall = hw - np.maximum(np.abs(pos_s[:, 0]), np.abs(pos_s[:, 1]))
    wall = np.clip(wall, 0, None)
    H, _, _ = np.histogram2d(pos_s[:, 0], pos_s[:, 1], bins=20)
    p = H.ravel()
    p = p[p > 0]
    p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    v = np.diff(pos_s, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    dang = np.diff(ang)
    dang = (dang + np.pi) % (2 * np.pi) - np.pi
    # Turning only while moving: both flanking samples have speed > 2 cm/s.
    # dang[k] is the heading change between segments ending at k+1 and k+2
    # (indices into speed: use speed[1:-1] aligned to dang).
    if dang.size and speed.size >= 3:
        moving = (speed[1:-1] > 2.0)
        if moving.shape[0] == dang.shape[0]:
            dang = dang[moving]
        else:
            n = min(moving.shape[0], dang.shape[0])
            dang = dang[:n][moving[:n]]
    if dt is None:
        dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    slow = speed < 2.0
    dwell_lens = []
    run = 0
    for s in slow:
        if s:
            run += 1
        elif run:
            dwell_lens.append(run * dt)
            run = 0
    if run:
        dwell_lens.append(run * dt)
    base = dict(
        dataset=dataset,
        animal_or_seed=animal_or_seed,
        session=session,
        chunk=chunk,
        region_set=region_set,
    )
    return [
        tidy_row(**base, metric="speed_mean", W=None, N=None, value=float(np.nanmean(speed))),
        tidy_row(**base, metric="speed_median", W=None, N=None, value=float(np.nanmedian(speed))),
        tidy_row(**base, metric="accel_std", W=None, N=None, value=float(np.nanstd(acc))),
        tidy_row(**base, metric="wall_dist_mean_norm", W=None, N=None, value=float(np.nanmean(wall / max(hw, 1e-9)))),
        tidy_row(**base, metric="occupancy_entropy", W=None, N=None, value=ent),
        tidy_row(
            **base,
            metric="turning_angle_std",
            W=None,
            N=None,
            value=float(np.nanstd(dang)) if dang.size else float("nan"),
        ),
        tidy_row(
            **base,
            metric="dwell_time_mean_s",
            W=None,
            N=None,
            value=float(np.mean(dwell_lens)) if dwell_lens else 0.0,
        ),
        tidy_row(**base, metric="tracking_jitter_rest_cm", W=None, N=None, value=jitter),
        tidy_row(**base, metric="pos_smooth_sigma_s", W=None, N=None, value=float(POS_SMOOTH_SIGMA_S)),
    ]
