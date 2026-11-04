"""Synthetic tests for Report 6 detector + hysteresis."""

from __future__ import annotations

import numpy as np

from agents.report6.detector import (
    DetectorModel,
    cohen_kappa,
    evaluate_detector,
    fit_detector,
)


def _blob_session(seed: int = 0):
    rng = np.random.default_rng(seed)
    T = 2000
    # two Gaussian blobs in 3D
    Z = np.zeros((T, 3))
    labels = np.zeros(T, dtype=int)
    Z[:1000] = rng.normal(loc=[0, 0, 0], scale=0.3, size=(1000, 3))
    Z[1000:] = rng.normal(loc=[3, 0, 0], scale=0.3, size=(1000, 3))
    labels[1000:] = 1
    # interleave visits: alternate 100-sample blocks for each
    Z2 = np.zeros_like(Z)
    lab2 = np.zeros(T, dtype=int)
    for i in range(20):
        sl = slice(i * 100, (i + 1) * 100)
        if i % 2 == 0:
            Z2[sl] = rng.normal([0, 0, 0], 0.3, size=(100, 3))
            lab2[sl] = 0
        else:
            Z2[sl] = rng.normal([3, 0, 0], 0.3, size=(100, 3))
            lab2[sl] = 1
    train = np.zeros(T, dtype=bool)
    train[:1400] = True
    test = ~train
    times = np.arange(T) * 0.05
    return Z2, lab2, train, test, times


def test_hysteresis_min_dwell():
    p = np.zeros(100)
    p[10:12] = 0.9  # too short to matter once entered... enter then immediately low
    p[10:30] = 0.9
    p[30:40] = 0.2  # want exit but min dwell from entry: entry at 10, min 5 samples
    model = DetectorModel(
        mu=np.zeros(1), precision=np.eye(1), logit_a=0.0, logit_b=1.0,
        enter_p=0.7, exit_p=0.4, min_dwell_samples=5,
    )
    b = model.hysteresis(p)
    assert b[10]
    # still in during early exit attempt if dwell not met — dwell already > 5 by 30
    assert not b[35]


def test_fit_and_held_out_separable():
    Z, labels, train, test, times = _blob_session()
    in_train = labels == 0
    model = fit_detector(Z, in_train, train)
    ev = evaluate_detector(model, Z, labels, 0, test, times_s=times)
    assert ev["delta_p"] is not None
    assert ev["delta_p"] > 0.2
    assert ev["distinguishable_pass"]


def test_cohen_kappa_perfect_and_chance():
    a = np.array([True, True, False, False])
    assert abs(cohen_kappa(a, a) - 1.0) < 1e-9
    b = np.array([True, False, True, False])
    k = cohen_kappa(a, b)
    assert abs(k) < 1e-9
