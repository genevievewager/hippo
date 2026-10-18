"""Lab boundary polygon helpers for reachable-bin masking."""
from __future__ import annotations

from typing import Any

import numpy as np


def boundary_polygon_xy(cfg: dict[str, Any], room: str = "A") -> np.ndarray:
    """Return (N,2) closed-ish polygon in room-local cm (origin = centre)."""
    from analysis.real_quadrant.adapter import _room_boundary
    import pandas as pd

    cx, cy, _w, _h, _xmin, _ymin = _room_boundary(cfg, room)
    mr = cfg["preprocessing"]["map_rooms"]
    idx_map = {int(k): str(v) for k, v in (mr.get("index") or {}).items()}
    room_idx = None
    for i, lab in idx_map.items():
        if lab == room or lab == room.upper():
            room_idx = i
            break
    if room_idx is None:
        raise KeyError(room)
    bdf = pd.DataFrame(cfg["preprocessing"]["boundary"])
    sub = bdf[bdf["Room"].astype(int) == int(room_idx)]
    xy = np.column_stack([sub["X"].to_numpy(float) - cx, sub["Y"].to_numpy(float) - cy])
    return xy


def point_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Ray-casting; xy (M,2), poly (N,2)."""
    x, y = xy[:, 0], xy[:, 1]
    n = len(poly)
    inside = np.zeros(len(xy), dtype=bool)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi
        )
        inside[intersect] = ~inside[intersect]
        j = i
    return inside


def square_polygon(half_width_cm: float) -> np.ndarray:
    hw = float(half_width_cm)
    return np.array([[-hw, -hw], [hw, -hw], [hw, hw], [-hw, hw]], dtype=float)


def bin_centers_reachable(edges_x: np.ndarray, edges_y: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Boolean (nx, ny) mask: bin centre inside polygon."""
    cx = 0.5 * (edges_x[:-1] + edges_x[1:])
    cy = 0.5 * (edges_y[:-1] + edges_y[1:])
    xx, yy = np.meshgrid(cx, cy, indexing="ij")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return point_in_polygon(pts, poly).reshape(xx.shape)


def path_visited_bins(
    positions: np.ndarray,
    edges_x: np.ndarray,
    edges_y: np.ndarray,
    *,
    min_frames: int = 1,
) -> np.ndarray:
    """Boolean (nx, ny) mask: bins visited by the path (≥ min_frames samples).

    Used when path–polygon containment fails but the session is retained
    (Report 2 flag semantics): spatial metrics ignore off-path bins.
    """
    p = np.asarray(positions, float)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.zeros((len(edges_x) - 1, len(edges_y) - 1), dtype=bool)
    finite = np.isfinite(p).all(axis=1)
    p = p[finite]
    if p.size == 0:
        return np.zeros((len(edges_x) - 1, len(edges_y) - 1), dtype=bool)
    occ, _, _ = np.histogram2d(p[:, 0], p[:, 1], bins=[edges_x, edges_y])
    return occ >= float(min_frames)
