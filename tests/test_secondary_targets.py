"""Synthetic tests for Phase B secondary targets / metrics / decode."""

from __future__ import annotations

import numpy as np

from analysis.real_quadrant.latent_cache import (
    cache_root,
    save_final_latent,
    save_fold_latent,
    write_meta,
)
from analysis.real_quadrant.secondary_decode import (
    decode_target_from_cache,
    select_hparams_from_fold_cache,
)
from analysis.real_quadrant.secondary_metrics import (
    categorical_chance_normalized,
    normalized_circular_error,
    normalized_continuous_error,
)
from analysis.real_quadrant.secondary_targets import (
    SPEED_WINDOW_S,
    build_secondary_target_table,
    derived_movement_from_speed,
    derived_wall_from_xy,
    distance_to_polygon_edges,
    head_direction_sincos,
    merge_rare_classes,
    position_derived_speed,
    wall_distance_bin,
)


def _square_poly(half: float = 50.0) -> np.ndarray:
    return np.array([
        [-half, -half], [half, -half], [half, half], [-half, half], [-half, -half],
    ], dtype=float)


def test_distance_to_wall_inside_and_clip_outside():
    poly = _square_poly(50)
    # centre → 50 cm to edge
    d = distance_to_polygon_edges(np.array([0.0]), np.array([0.0]), poly)
    assert abs(d[0] - 50.0) < 1e-6
    # outside → clipped to 0
    d2 = distance_to_polygon_edges(np.array([80.0]), np.array([0.0]), poly)
    assert d2[0] == 0.0


def test_wall_bins_and_rare_class_merge():
    dist = np.array([1, 2, 15, 16, 40, 41, 42], dtype=float)
    # make center rare
    dist = np.r_[np.full(100, 5.0), np.full(100, 15.0), np.full(2, 40.0)]
    bins = wall_distance_bin(dist)
    train = np.ones(len(bins), dtype=bool)
    test = train.copy()
    merged, meta = merge_rare_classes(bins, train, test)
    assert meta["evaluable"]
    assert 2 not in np.unique(merged[np.isfinite(merged)].astype(int))


def test_position_speed_250ms_and_movement_derived():
    dt = 0.05
    t = np.arange(0, 2.0, dt)
    # constant 10 cm/s along x
    x = 10.0 * t
    y = np.zeros_like(t)
    speed = position_derived_speed(x, y, t, window_s=SPEED_WINDOW_S)
    med = float(np.nanmedian(speed[t >= SPEED_WINDOW_S]))
    assert abs(med - 10.0) / 10.0 < 1e-2
    move = derived_movement_from_speed(speed)
    # 10 cm/s → boundary of slow/fast: <10 slow, ≥10 fast
    assert int(move[np.isfinite(move)][-1]) in (1, 2)


def test_head_direction_sincos_and_circular_metric():
    hd = np.array([0.0, 90.0, 180.0, 270.0])
    sc = head_direction_sincos(hd)
    np.testing.assert_allclose(sc[0], [0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(sc[1], [1.0, 0.0], atol=1e-7)
    # perfect decode → normalized ~ 0
    sc_tr = sc.copy()
    sc_te = sc.copy()
    sc_hat = sc_te.copy()
    sc_out = normalized_circular_error(sc_hat, sc_te, sc_tr)
    assert sc_out["median_error_rad"] < 1e-8


def test_categorical_chance_normalized_perfect():
    y = np.array([0, 0, 1, 1, 2, 2])
    sc = categorical_chance_normalized(y, y)
    assert sc["balanced_accuracy"] == 1.0
    assert abs(sc["normalized_error"] - 0.0) < 1e-12


def test_continuous_normalized_and_derived_wall():
    ytr = np.array([0.0, 10.0, 20.0])
    yte = np.array([5.0, 15.0])
    yhat = yte.copy()
    sc = normalized_continuous_error(yhat, yte, ytr, use_median_floor=True)
    assert sc["median_error"] == 0.0
    poly = _square_poly(50)
    d, b = derived_wall_from_xy(np.array([0.0, 45.0]), np.array([0.0, 0.0]), poly)
    assert abs(d[0] - 50.0) < 1e-6
    assert abs(d[1] - 5.0) < 1e-6
    assert int(b[1]) == 0  # near wall (<10 cm)


def test_causal_ema_reduces_error_tau_not_from_test(tmp_path):
    """EMA on a smooth target lowers error; τ chosen on train blocks only."""
    from analysis.real_quadrant.posthoc_smooth import EMA_TAUS_S, causal_ema
    from analysis.real_quadrant.secondary_decode import (
        apply_ema_selected,
        select_ema_tau_outputs,
    )

    rng = np.random.default_rng(0)
    n = 400
    t = np.arange(n) * 0.05
    # smooth latent target + noisy readout
    true = np.sin(2 * np.pi * t / 4.0)
    pred = true + rng.normal(0, 0.6, size=n)
    train_ok = np.zeros(n, dtype=bool)
    train_ok[:300] = True
    eval_mask = ~train_ok

    tau, _ = select_ema_tau_outputs(
        pred, true, t, train_ok, kind="continuous", taus_s=EMA_TAUS_S,
    )
    assert tau in EMA_TAUS_S
    # Sanity: selecting with eval as "train" must not be how we call it —
    # verify train-only: zeroing test labels shouldn't change τ.
    true_masked = true.copy()
    true_masked[eval_mask] = np.nan
    tau2, _ = select_ema_tau_outputs(
        pred, true_masked, t, train_ok, kind="continuous", taus_s=EMA_TAUS_S,
    )
    assert tau2 == tau

    raw_err = float(np.median(np.abs(pred[eval_mask] - true[eval_mask])))
    out = apply_ema_selected(
        pred, true, t, train_ok, eval_mask, kind="continuous", taus_s=EMA_TAUS_S,
    )
    ema_err = float(out["scores"]["median_error"])
    assert ema_err < raw_err
    # causal_ema itself is causal (depends only on past)
    sm = causal_ema(pred.reshape(-1, 1), t, tau)[:, 0]
    # perturb future test preds; past train EMA unchanged
    pred2 = pred.copy()
    pred2[eval_mask] = rng.normal(size=int(eval_mask.sum()))
    sm2 = causal_ema(pred2.reshape(-1, 1), t, tau)[:, 0]
    np.testing.assert_allclose(sm[train_ok], sm2[train_ok], atol=1e-12)


def test_build_table_and_cache_decode_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    n, d = 300, 6
    t = np.arange(n) * 0.05
    x = 20 * np.sin(t)
    y = 20 * np.cos(t)
    poly = _square_poly(50)
    hd = (np.rad2deg(np.arctan2(np.gradient(y), np.gradient(x))) + 360) % 360
    table = build_secondary_target_table(
        x=x, y=y, times=t, polygon_xy_local=poly,
        hd_deg=hd, tracker_v=np.abs(np.gradient(x)) / 0.05,
    )
    assert table["speed"].shape == (n,)
    assert table["pc_v"] is not None

    # synthetic latents correlated with speed
    speed = table["speed"].copy()
    speed[~np.isfinite(speed)] = 0.0
    Z = rng.normal(size=(n, d))
    Z[:, 0] += 0.5 * (speed / (np.std(speed) + 1e-6))
    train_ok = np.zeros(n, dtype=bool)
    train_ok[:220] = True
    eval_mask = ~train_ok
    root = cache_root(tmp_path / "sess")
    write_meta(root, {"latent_dims": [2, 3, 6], "nested_fit_d": 6})
    for fi in range(3):
        fit = train_ok.copy()
        # leave a validation block inside train
        fit[60 * fi:60 * fi + 40] = False
        fit &= train_ok
        save_fold_latent(root, "pca", fi, Z=Z, fit_mask=fit, d_fit=6)
    save_final_latent(root, "pca", Z=Z, primary_d=3, ridge_alpha=1.0)

    hp = select_hparams_from_fold_cache(
        root, "pca", table["speed"], train_ok,
        latent_dims=[2, 3, 6], nested_fit_d=6,
        alphas=[0.1, 1.0, 10.0], ks=[5, 10],
        kind="continuous",
    )
    assert hp["primary_d"] in (2, 3, 6)
    dec = decode_target_from_cache(
        root, "pca", table["speed"], train_ok, eval_mask,
        kind="continuous", hparams=hp, times=t,
        shift_fracs=[0.3, 0.5], apply_ema=True,
    )
    assert dec["a5"]["status"] == "PASS"
    assert np.isfinite(dec["ridge"]["normalized_error"])
    assert "ridge_ema" in dec and dec["ridge_ema"]["tau_s"] in (0.1, 0.25, 0.5, 1.0, 2.0)
