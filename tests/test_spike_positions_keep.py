"""Exact spike→frame assignment: excluded-frame spikes must not enter rate maps."""
from __future__ import annotations

import numpy as np

from agents.data_quality.metrics.spatial import (
    _build_map,
    _rate_map,
    _spike_frame_indices,
    _spike_positions,
)


def test_excluded_frame_spikes_contribute_zero():
    dt = 0.05
    t = np.arange(0.0, 2.0, dt)  # contiguous parent
    # Alternating keep: even frames kept, odd excluded (speed filter / split)
    keep = (np.arange(len(t)) % 2) == 0
    p = np.column_stack([np.linspace(0, 10, len(t)), np.zeros(len(t))])

    # Spikes only during excluded (odd) frames — mid-interval of each odd frame
    excluded_idx = np.flatnonzero(~keep)
    st = t[excluded_idx] + 0.5 * dt

    idx = _spike_frame_indices(st, t, keep=keep)
    assert idx.size == 0

    pos = _spike_positions(st, t, p, keep=keep)
    assert pos.shape == (0, 2)

    edges_x = np.linspace(-1, 11, 6)
    edges_y = np.linspace(-1, 1, 4)
    rate, occ, ok = _build_map(
        st, t, p, edges_x, edges_y, occ_thresh_frames=1.0, dt=dt, keep=keep,
    )
    # Occupancy from kept frames only; spike counts must be zero everywhere
    spike_pos = _spike_positions(st, t, p, keep=keep)
    spk, _, _ = np.histogram2d(
        spike_pos[:, 0] if len(spike_pos) else np.array([]),
        spike_pos[:, 1] if len(spike_pos) else np.array([]),
        bins=[edges_x, edges_y],
    ) if len(spike_pos) else (np.zeros_like(rate), None, None)
    assert float(spk.sum()) == 0.0
    assert float(rate[ok].sum()) == 0.0 if ok.any() else True
    assert int(keep.sum()) > 0
    assert occ[ok].sum() > 0 if ok.any() else True


def test_kept_frame_spikes_are_assigned():
    dt = 0.05
    t = np.arange(0.0, 1.0, dt)
    keep = np.ones(len(t), dtype=bool)
    keep[5:8] = False  # a gap of excluded frames
    p = np.column_stack([np.arange(len(t), dtype=float), np.zeros(len(t))])
    # spike in a kept frame and one in the excluded gap
    st = np.array([t[2] + 0.01, t[6] + 0.01])
    idx = _spike_frame_indices(st, t, keep=keep)
    assert idx.tolist() == [2]
    pos = _spike_positions(st, t, p, keep=keep)
    assert pos.shape == (1, 2)
    assert abs(pos[0, 0] - 2.0) < 1e-9
