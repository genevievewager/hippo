"""Secondary behavioral targets for Report 2 v2 (Phase B).

Primary remains ``(x, y)``. These helpers build secondary arrays on the
decoder grid from the same room-local positions / HD / tracker ``V`` used
in Phase 0b. Latents are not fitted here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

SPEED_WINDOW_S = 0.250
WALL_NEAR_CM = 10.0
WALL_MID_CM = 30.0
SPEED_STILL = 2.0
SPEED_SLOW = 10.0
MIN_CLASS_FRAC = 0.05

WALL_BIN_LABELS = ("near_wall", "middle", "center")
MOVE_STATE_LABELS = ("still", "slow", "fast")


def point_to_segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> np.ndarray:
    """Euclidean distance from points to a line segment (a→b)."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    abx, aby = bx - ax, by - ay
    apx, apy = px - ax, py - ay
    denom = abx * abx + aby * aby
    if denom <= 0.0:
        return np.hypot(apx, apy)
    t = np.clip((apx * abx + apy * aby) / denom, 0.0, 1.0)
    return np.hypot(apx - t * abx, apy - t * aby)


def distance_to_polygon_edges(
    x: np.ndarray,
    y: np.ndarray,
    polygon_xy: np.ndarray,
    *,
    inside: np.ndarray | None = None,
) -> np.ndarray:
    """Nearest distance to polygon edges; clip at 0 outside (signed→0).

    ``polygon_xy`` is closed (N, 2). If ``inside`` is provided, points with
    ``inside==False`` are set to 0.0; otherwise outside is detected via the
    matplotlib path and clipped the same way.
    """
    from matplotlib.path import Path as MplPath

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    poly = np.asarray(polygon_xy, dtype=float)
    if len(poly) < 2:
        raise ValueError("polygon needs ≥2 vertices")
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    d = np.full(x.shape, np.inf, dtype=float)
    for i in range(len(poly) - 1):
        d = np.minimum(
            d,
            point_to_segment_distance(
                x, y, float(poly[i, 0]), float(poly[i, 1]),
                float(poly[i + 1, 0]), float(poly[i + 1, 1]),
            ),
        )
    if inside is None:
        inside = MplPath(poly).contains_points(np.column_stack([x, y]))
    else:
        inside = np.asarray(inside, dtype=bool).ravel()
    out = d.copy()
    out[~inside] = 0.0
    out[~np.isfinite(out)] = np.nan
    return out


def wall_distance_bin(dist_cm: np.ndarray) -> np.ndarray:
    """Integer codes: 0=near_wall, 1=middle, 2=center (nan preserved)."""
    d = np.asarray(dist_cm, dtype=float)
    out = np.full(d.shape, np.nan)
    ok = np.isfinite(d)
    out[ok & (d < WALL_NEAR_CM)] = 0
    out[ok & (d >= WALL_NEAR_CM) & (d < WALL_MID_CM)] = 1
    out[ok & (d >= WALL_MID_CM)] = 2
    return out


def movement_state_from_speed(speed: np.ndarray) -> np.ndarray:
    """Integer codes: 0=still, 1=slow, 2=fast (nan preserved)."""
    s = np.asarray(speed, dtype=float)
    out = np.full(s.shape, np.nan)
    ok = np.isfinite(s)
    out[ok & (s < SPEED_STILL)] = 0
    out[ok & (s >= SPEED_STILL) & (s < SPEED_SLOW)] = 1
    out[ok & (s >= SPEED_SLOW)] = 2
    return out


def backward_lag_index(
    times: np.ndarray,
    window_s: float,
) -> np.ndarray:
    """For each t, largest j with times[t]-times[j] ≥ window_s (else -1)."""
    times = np.asarray(times, dtype=float)
    n = len(times)
    out = np.full(n, -1, dtype=np.int64)
    j = 0
    for i in range(n):
        while j < i and (times[i] - times[j]) >= float(window_s):
            j += 1
        # j is first index with Δt < window; lag is j-1 if ≥0
        lag = j - 1
        if lag >= 0 and (times[i] - times[lag]) >= float(window_s) * 0.5:
            out[i] = lag
    return out


def position_derived_speed(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    *,
    window_s: float = SPEED_WINDOW_S,
) -> np.ndarray:
    """Backward |Δ(x,y)|/Δt over ``window_s`` on the decoder grid."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    lag = backward_lag_index(times, window_s)
    out = np.full(len(times), np.nan, dtype=float)
    idx = np.where(lag >= 0)[0]
    if len(idx) == 0:
        return out
    lags = lag[idx]
    finite = (
        np.isfinite(x[idx]) & np.isfinite(y[idx])
        & np.isfinite(x[lags]) & np.isfinite(y[lags])
    )
    idx = idx[finite]
    lags = lags[finite]
    dt = times[idx] - times[lags]
    dist = np.hypot(x[idx] - x[lags], y[idx] - y[lags])
    with np.errstate(invalid="ignore", divide="ignore"):
        out[idx] = dist / np.maximum(dt, 1e-12)
    out[~np.isfinite(out)] = np.nan
    return out


def acceleration_from_speed(
    speed: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """``dspeed/dt`` via adjacent-sample backward difference on the grid."""
    speed = np.asarray(speed, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    out = np.full(len(times), np.nan, dtype=float)
    if len(times) < 2:
        return out
    dt = np.diff(times)
    ds = np.diff(speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = ds / dt
    out[~np.isfinite(out)] = np.nan
    return out


def head_direction_sincos(hd_deg: np.ndarray) -> np.ndarray:
    """``(n, 2)`` of (sin θ, cos θ); θ from degrees; nan rows where HD bad."""
    hd = np.asarray(hd_deg, dtype=float).ravel()
    theta = np.deg2rad(hd)
    out = np.column_stack([np.sin(theta), np.cos(theta)])
    bad = ~np.isfinite(hd)
    out[bad] = np.nan
    return out


def merge_rare_classes(
    labels: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    *,
    ordered: bool = True,
    min_frac: float = MIN_CLASS_FRAC,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Merge classes with < min_frac in train or test into a neighbour.

    Ordered bins merge into the nearer neighbour by label value. Returns
    possibly remapped labels and a meta dict (``evaluable``, merges).
    """
    lab = np.asarray(labels, dtype=float).copy()
    train_mask = np.asarray(train_mask, dtype=bool)
    test_mask = np.asarray(test_mask, dtype=bool)
    merges: list[dict[str, Any]] = []

    def _fracs(mask: np.ndarray) -> dict[int, float]:
        vals = lab[mask & np.isfinite(lab)]
        if len(vals) == 0:
            return {}
        uniq, cnt = np.unique(vals.astype(int), return_counts=True)
        n = float(cnt.sum())
        return {int(u): float(c) / n for u, c in zip(uniq, cnt)}

    for _ in range(8):
        ft, fte = _fracs(train_mask), _fracs(test_mask)
        classes = sorted(set(ft) | set(fte))
        if len(classes) < 2:
            return lab, {
                "evaluable": False,
                "reason": "fewer than 2 classes after merge",
                "merges": merges,
                "train_frac": ft,
                "test_frac": fte,
            }
        rare = [
            c for c in classes
            if ft.get(c, 0.0) < min_frac or fte.get(c, 0.0) < min_frac
        ]
        if not rare:
            return lab, {
                "evaluable": True,
                "merges": merges,
                "train_frac": ft,
                "test_frac": fte,
                "n_classes": len(classes),
            }
        if not ordered:
            return lab, {
                "evaluable": False,
                "reason": "rare class and unordered merge disabled",
                "merges": merges,
                "rare": rare,
            }
        c = rare[0]
        neighbours = [x for x in classes if x != c]
        target = min(neighbours, key=lambda x: abs(x - c))
        lab[np.isfinite(lab) & (lab.astype(int) == c)] = float(target)
        merges.append({"from": c, "to": target})

    return lab, {
        "evaluable": False,
        "reason": "merge iterations exhausted",
        "merges": merges,
    }


def build_secondary_target_table(
    *,
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    polygon_xy_local: np.ndarray,
    hd_deg: np.ndarray | None = None,
    tracker_v: np.ndarray | None = None,
    valid: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build all secondary target arrays aligned to ``times``.

    Coordinates are room-local cm (same as primary decode ``y``).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    times = np.asarray(times, dtype=float).ravel()
    n = len(times)
    if valid is None:
        valid = np.isfinite(x) & np.isfinite(y)
    else:
        valid = np.asarray(valid, dtype=bool).ravel() & np.isfinite(x) & np.isfinite(y)

    dist = distance_to_polygon_edges(x, y, polygon_xy_local)
    dist[~valid] = np.nan
    wall_bin = wall_distance_bin(dist)
    speed = position_derived_speed(x, y, times, window_s=SPEED_WINDOW_S)
    speed[~valid] = np.nan
    accel = acceleration_from_speed(speed, times)
    accel[~valid] = np.nan
    move = movement_state_from_speed(speed)

    hd_sc = np.full((n, 2), np.nan)
    hd_valid = np.zeros(n, dtype=bool)
    if hd_deg is not None:
        hd_sc = head_direction_sincos(hd_deg)
        hd_valid = valid & np.isfinite(hd_sc).all(axis=1)
        hd_sc[~hd_valid] = np.nan

    pc_v = None
    if tracker_v is not None:
        pc_v = np.asarray(tracker_v, dtype=float).ravel().copy()
        pc_v[~valid] = np.nan

    return {
        "distance_to_wall": dist,
        "wall_distance_bin": wall_bin,
        "speed": speed,
        "acceleration": accel,
        "movement_state": move,
        "head_direction_sincos": hd_sc,
        "head_direction_valid": hd_valid,
        "pc_v": pc_v,
        "valid": valid,
        "speed_window_s": SPEED_WINDOW_S,
        "definitions": {
            "speed": "250ms_backward_position_delta",
            "pc_v": "tracker_V_sensitivity_only",
            "wall_bins_cm": [0, WALL_NEAR_CM, WALL_MID_CM, None],
            "move_bins_cm_s": [0, SPEED_STILL, SPEED_SLOW, None],
        },
    }


def derived_wall_from_xy(
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    polygon_xy_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Beyond-position control: wall distance/bin from decoded ``(x,y)``."""
    dist = distance_to_polygon_edges(x_hat, y_hat, polygon_xy_local)
    return dist, wall_distance_bin(dist)


def derived_movement_from_speed(speed_hat: np.ndarray) -> np.ndarray:
    """Beyond-speed control: movement_state from decoded speed."""
    return movement_state_from_speed(speed_hat)
