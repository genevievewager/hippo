"""Synthetic tests for Report 6 visit counter."""

from __future__ import annotations

import numpy as np

from agents.report6.visits import count_visits


def test_single_visit():
    # 2 s in at start (40 samples @ 50 ms), then out
    m = np.zeros(200, dtype=bool)
    m[:40] = True
    r = count_visits(m, dt_s=0.05, min_outside_s=1.0)
    assert r["n_visits"] == 1
    assert abs(r["occupancy"] - 40 / 200) < 1e-9
    assert abs(r["dwell_s"][0] - 2.0) < 0.06


def test_two_visits_with_gap():
    m = np.zeros(400, dtype=bool)
    m[0:20] = True          # 1 s
    m[60:100] = True        # gap 2 s (>= 1 s) then 2 s in
    r = count_visits(m, dt_s=0.05, min_outside_s=1.0)
    assert r["n_visits"] == 2


def test_short_gap_merges_not_counted_separately():
    # gap 0.5 s < 1 s → second entry does not add a visit
    m = np.zeros(200, dtype=bool)
    m[0:20] = True
    m[30:50] = True  # 0.5 s gap
    r = count_visits(m, dt_s=0.05, min_outside_s=1.0)
    assert r["n_visits"] == 1


def test_empty():
    r = count_visits(np.zeros(0, dtype=bool))
    assert r["n_visits"] == 0
    assert r["occupancy"] == 0.0
