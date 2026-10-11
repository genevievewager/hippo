"""Synthetic tests for v2 latent cache + d-cap (no real data)."""

from __future__ import annotations

import numpy as np

from analysis.real_quadrant.adapter import HF_REGION_ALLOWLIST
from analysis.real_quadrant.latent_cache import (
    cache_root,
    cap_latent_dims,
    load_final_latent,
    ridge_predict_from_cached,
    save_final_latent,
    save_fold_latent,
    write_meta,
)
from realtime.quadrant_n5_run import _fit_predict_ridge


def test_cap_latent_dims_floor_half():
    dims, nested = cap_latent_dims([2, 3, 5, 10, 20, 40, 80], n_units=50)
    assert dims == [2, 3, 5, 10, 20]
    assert nested == 20  # ⌊50/2⌋ = 25 → drop 40/80
    dims2, nested2 = cap_latent_dims([2, 3, 5, 10, 20, 40, 80], n_units=200)
    assert dims2 == [2, 3, 5, 10, 20, 40, 80]
    assert nested2 == 80
    assert HF_REGION_ALLOWLIST == frozenset({"MEC", "SUB", "PARA", "LEC"})


def test_latent_cache_roundtrip_identical_ridge_predictions(tmp_path):
    """Save final Z; reload; Ridge predictions match in-memory fit."""
    rng = np.random.default_rng(0)
    n_t, d = 400, 8
    Z = rng.normal(size=(n_t, d))
    y = rng.normal(size=(n_t, 2))
    train_ok = np.zeros(n_t, dtype=bool)
    train_ok[:320] = True
    eval_mask = np.zeros(n_t, dtype=bool)
    eval_mask[320:] = True
    alpha = 1.0

    root = cache_root(tmp_path / "sess")
    write_meta(root, {"n_units": 40, "latent_dims": [2, 3, 5, 8]})
    save_fold_latent(
        root, "pca", 0,
        Z=Z, fit_mask=train_ok, d_fit=d,
    )
    save_final_latent(
        root, "pca", Z=Z, primary_d=d, ridge_alpha=alpha,
    )

    loaded = load_final_latent(root, "pca")
    assert loaded["primary_d"] == d
    assert loaded["Z"].shape == Z.shape
    np.testing.assert_allclose(loaded["Z"], Z, rtol=1e-5, atol=1e-5)

    pred_mem = _fit_predict_ridge(Z[train_ok], y[train_ok], Z[eval_mask], alpha)
    pred_cache = ridge_predict_from_cached(
        loaded["Z"], y, train_ok, eval_mask, alpha,
    )
    np.testing.assert_allclose(pred_cache, pred_mem, rtol=1e-6, atol=1e-6)
