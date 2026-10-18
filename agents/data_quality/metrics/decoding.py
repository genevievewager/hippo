"""Within-dataset decoding: Ridge on raw causal counts only (frozen quadrant_n5).

User/SPEC override for Report 4: no sqrt transform — raw [t−W, t) counts,
then train-only z-score. W / train_frac / ridge alpha from configs/quadrant_n5.yaml.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import Ridge

from agents.data_quality.metrics.schema import tidy_row
from realtime.quadrant_n5 import load_quadrant_n5_yaml


def frozen_decode_config() -> dict[str, Any]:
    cfg = load_quadrant_n5_yaml()
    feat = cfg["features"]
    split = cfg["split"]
    dec = cfg["decoders"]
    # mid of log-spaced alpha grid (frozen default without nested CV for speed)
    lo = float(dec["ridge_alpha_lo"])
    hi = float(dec["ridge_alpha_hi"])
    n = int(dec["ridge_alpha_n"])
    alphas = np.logspace(np.log10(lo), np.log10(hi), n)
    return {
        "window_s": float(feat["window_s"]),
        "update_dt": float(feat["update_dt"]),
        "train_frac": float(split["train_frac"]),
        "gap_s": float(split["gap_s"]),
        "ridge_alpha": float(alphas[len(alphas) // 2]),
        "ridge_alpha_lo": lo,
        "ridge_alpha_hi": hi,
        "config_sha256": cfg["config_sha256"],
        "raw_counts_only": True,
        "no_sqrt": True,
    }


def _bin_counts(spike_times, times, W):
    """Causal counts in [t−W, t) for each label time."""
    X = np.zeros((len(times), len(spike_times)), dtype=float)
    for j, st in enumerate(spike_times):
        st = np.sort(np.asarray(st, float))
        left = np.searchsorted(st, times - W, side="left")
        right = np.searchsorted(st, times, side="left")
        X[:, j] = right - left
    return X


def decoding_metric_rows(
    spike_times,
    positions,
    times,
    *,
    dataset,
    animal_or_seed,
    session,
    chunk,
    region_set,
    valid=None,
    Ns=(10, 20, 40),
    W_grid=None,
    train_frac=None,
    ridge_alpha=None,
    n_subsamples: int = 20,
    rng=None,
) -> list[dict[str, Any]]:
    frozen = frozen_decode_config()
    if W_grid is None:
        W_grid = (frozen["window_s"],)
    if train_frac is None:
        train_frac = frozen["train_frac"]
    if ridge_alpha is None:
        ridge_alpha = frozen["ridge_alpha"]
    rng = rng or np.random.default_rng(0)
    times = np.asarray(times, float)
    y = np.asarray(positions, float)
    if valid is not None:
        m = np.asarray(valid, bool) & np.isfinite(y).all(1)
    else:
        m = np.isfinite(y).all(1)
    times, y = times[m], y[m]
    if times.size < 50 or len(spike_times) < 5:
        return []
    n_train = int(train_frac * len(times))
    gap_n = int(round(frozen["gap_s"] / max(float(np.median(np.diff(times))), 1e-3)))
    test_start = min(n_train + gap_n, len(times) - 10)
    if test_start >= len(times) - 5:
        test_start = n_train
    rows = []
    base = dict(
        dataset=dataset,
        animal_or_seed=animal_or_seed,
        session=session,
        chunk=chunk,
        region_set=region_set,
    )
    for W in W_grid:
        X = _bin_counts(spike_times, times, float(W))  # RAW causal counts
        for N in Ns:
            if len(spike_times) < N:
                continue
            errs, floors = [], []
            n_rep = int(n_subsamples) if N < len(spike_times) else 1
            for _ in range(n_rep):
                idx = rng.choice(len(spike_times), size=int(N), replace=False)
                Xn = X[:, idx]
                mu = Xn[:n_train].mean(0)
                sd = Xn[:n_train].std(0) + 1e-6
                Xz = (Xn - mu) / sd
                clf = Ridge(alpha=float(ridge_alpha))
                clf.fit(Xz[:n_train], y[:n_train])
                pred = clf.predict(Xz[test_start:])
                errs.append(float(np.median(np.linalg.norm(pred - y[test_start:], axis=1))))
                floors.append(float(np.median(np.linalg.norm(y[test_start:] - y[:n_train].mean(0), axis=1))))
            err = float(np.median(errs))
            floor = float(np.median(floors))
            rows.append(tidy_row(**base, metric="decode_error_cm", W=float(W), N=int(N), value=err))
            rows.append(tidy_row(**base, metric="decode_floor_cm", W=float(W), N=int(N), value=floor))
            rows.append(
                tidy_row(
                    **base,
                    metric="decode_error_over_floor",
                    W=float(W),
                    N=int(N),
                    value=err / max(floor, 1e-9),
                )
            )
    return rows
