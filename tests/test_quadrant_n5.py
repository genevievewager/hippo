"""Frozen n=5 config, purge-gap split, and A14 LDS filter readout."""

from __future__ import annotations

import numpy as np
import pytest

from realtime.dynamic_latents.lds import (
    LinearDynamicalSystem,
    assert_readout_latents_are_filtered,
)
from realtime.pipeline_invariants import PipelineInvariantError
from realtime.quadrant_n5 import (
    DEFAULT_CONFIG_PATH,
    derive_seed_streams,
    inner_cv_block_masks,
    load_quadrant_n5_yaml,
    runtime_versions,
)
from realtime.train_decoder import causal_train_test_split, purge_gap_s


def test_frozen_yaml_reuse_transforms_false_and_gap_matches_rule():
    cfg = load_quadrant_n5_yaml()
    assert cfg["reuse_transforms"] is False
    assert cfg["latency_budget_ms"] == 50
    assert cfg["session"]["arena_shape"] == "square"
    assert cfg["decoders"]["target"] == "position_xy_cm"
    assert cfg["split"]["train_frac"] == 0.80
    assert cfg["split"]["gap_s"] == 1.0
    assert cfg["split"]["inner_cv_refit_representation"] is True
    assert cfg["lds_readout"]["filter_from"] == "session_start"
    assert cfg["representations"]["raw_lag"]["implemented"] is True
    assert cfg["representations"]["raw_lag"]["n_lags"] == 5
    assert cfg["representations"]["gpfa"]["expected_label"] == "offline_only"
    assert "offline_only" not in cfg["representations"]["gpfa"] or (
        cfg["representations"]["gpfa"].get("offline_only") is not True
    )
    assert cfg["representations"]["dm"]["local_scale_k"] == 10
    assert cfg["representations"]["dm"]["alpha"] == 1.0
    assert cfg["representations"]["dm"]["diffusion_time"] == 1.0
    assert cfg["representations"]["isomap"]["n_neighbors"] == 10
    assert cfg["representations"]["lds"]["em_iters"] == 15
    assert cfg["representations"]["lds"]["filter_continuous"] is True
    assert cfg["decoders"]["knn_standardize_fit"] == "train_only"
    assert cfg["phase3"]["learning_curve_fracs"] == [0.25, 0.5, 1.0]
    assert cfg["phase3"]["time_shift_s"] >= cfg["session"]["session_s"] / 2
    assert cfg["phase3"]["floor_baseline"] == "mean_position_train_shifted"
    assert cfg["phase3"]["eval_set"] == "intersection_of_valid"
    assert cfg["probe_track"]["file"] == "configs/trajectories/lab_npx2_default.yaml"
    assert len(cfg["probe_track_sha256"]) == 64
    assert "hippocampal_formation_transition" in cfg["unit_inclusion"]["regions"]
    assert cfg["phase3"]["a13"]["n_shifts"] == 20
    assert cfg["phase3"]["a13"]["pass_min_cm"] == -2.0
    assert cfg["phase3"]["coverage"]["occupancy_n_bins"] == 10
    assert cfg["representations"]["nonlinear_dynamic"]["method"] is None
    gap = purge_gap_s(
        cfg["features"]["window_s"],
        max_history_s=cfg["split"]["max_history_s"],
        update_dt=cfg["features"]["update_dt"],
    )
    assert gap == pytest.approx(1.0)
    assert len(cfg["config_sha256"]) == 64
    assert DEFAULT_CONFIG_PATH.is_file()


def test_purge_gap_contains_no_train_or_test_index():
    dt = 0.05
    t = np.arange(0.0, 20.0, dt)
    gap = 1.0
    train, test = causal_train_test_split(t, 0.80, gap_s=gap)
    split_time = t[0] + 0.80 * (t[-1] - t[0])
    in_gap = (t >= split_time) & (t < split_time + gap)
    assert not np.any(train & in_gap)
    assert not np.any(test & in_gap)
    assert not np.any(train & test)
    assert train.sum() > 0 and test.sum() > 0
    last_train = float(t[train][-1])
    first_test = float(t[test][0])
    assert first_test - last_train >= gap - 1e-12
    # No test window [t-W, t) reaches into train (W=0.250).
    W = 0.250
    assert not np.any((t[test] - W) < last_train)


def test_default_split_still_has_no_gap():
    t = np.arange(0.0, 10.0, 0.05)
    train, test = causal_train_test_split(t, 0.70)
    assert train.sum() + test.sum() == len(t)
    assert not np.any(train & test)


def test_a14_accepts_filtered_rejects_smoothed():
    rng = np.random.default_rng(0)
    T, n, k = 80, 6, 2
    model = LinearDynamicalSystem(n_components=k, n_em_iters=3, random_state=0)
    X = rng.normal(size=(T, n))
    model.fit(X[:50])
    X_tr = X[:50]
    Z_f = model.transform(X_tr, causal=True, reset=True)
    Z_s = model.transform(X_tr, causal=False, reset=True)
    assert_readout_latents_are_filtered(model, Z_f, X_tr)
    if not np.allclose(Z_f, Z_s, atol=1e-5):
        with pytest.raises(PipelineInvariantError, match="A14"):
            assert_readout_latents_are_filtered(model, Z_s, X_tr)


def test_inner_cv_refit_masks_have_purge_and_no_overlap():
    t = np.arange(0.0, 96.0, 0.05)
    folds = inner_cv_block_masks(t, n_blocks=5, gap_s=1.0)
    assert len(folds) == 5
    for train, val in folds:
        assert not np.any(train & val)
        assert train.sum() > 0 and val.sum() > 0
        last_tr = float(t[train][-1]) if t[train][-1] < t[val][0] else None
        if last_tr is not None:
            assert t[val][0] - last_tr >= 1.0 - 1e-12


def test_a14_session_start_filter_not_train_subset():
    """Continuous-from-t=0 filter at train rows ≠ re-filter of a leading-dropped subset."""
    rng = np.random.default_rng(1)
    T, n, k = 80, 6, 2
    model = LinearDynamicalSystem(n_components=k, n_em_iters=3, random_state=1)
    X = rng.normal(size=(T, n))
    train = np.zeros(T, dtype=bool)
    train[:50] = True
    train[:5] = False  # same leading drop as raw_lag ∩ identical train set
    model.fit(X[train])
    Z_session = model.transform(X, causal=True, reset=True)
    Z_subset = model.transform(X[train], causal=True, reset=True)
    assert not np.allclose(Z_session[train], Z_subset, atol=1e-6)
    assert_readout_latents_are_filtered(model, Z_session, X)


def test_fit_transform_refits_on_fit_mask_only():
    from realtime.quadrant_n5_run import fit_transform_representation
    from realtime.manifold_features import make_feature_transformer

    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 5))
    fit = np.zeros(40, dtype=bool)
    fit[:20] = True
    model = make_feature_transformer("global_pca", decode_window=0.250, n_components=2, random_state=0)
    Z = fit_transform_representation("pca", model, X, fit)
    assert Z.shape == (40, 2)
    # OOS rows are produced; refitting on all data would change the basis.
    model_all = make_feature_transformer("global_pca", decode_window=0.250, n_components=2, random_state=0)
    Z_all = fit_transform_representation("pca", model_all, X, np.ones(40, dtype=bool))
    assert not np.allclose(Z, Z_all, atol=1e-8)


def test_seed_streams_and_versions():
    a = derive_seed_streams(20260923, 5, 0, ["trajectory", "neural", "methods"])
    b = derive_seed_streams(20260923, 5, 1, ["trajectory", "neural", "methods"])
    assert a["trajectory"] != b["trajectory"]
    assert a["neural"] != a["methods"]
    vers = runtime_versions()
    assert "numpy" in vers and "sklearn" in vers


def test_a13_null_passes_on_uncorrelated_latents():
    from realtime.quadrant_n5_run import _a13_null

    rng = np.random.default_rng(0)
    n = 400
    Z = rng.normal(size=(n, 4))
    y = rng.uniform(0, 100, size=(n, 2))
    train = np.zeros(n, dtype=bool)
    train[:320] = True
    ev = ~train
    cfg = load_quadrant_n5_yaml()
    out = _a13_null(Z[train], Z[ev], y, train, ev, cfg, ridge_alpha=1.0, knn_k=5)
    assert out["ridge_pass"]
    assert out["knn_pass"]
    assert out["n_shifts"] == 20


def test_a13_fails_when_latents_are_the_labels():
    from realtime.quadrant_n5_run import _a13_null

    n = 400
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    y = np.column_stack([np.cos(t), np.sin(t)]) * 30 + 50
    Z = y.copy()
    train = np.zeros(n, dtype=bool)
    train[:320] = True
    ev = ~train
    cfg = load_quadrant_n5_yaml()
    out = _a13_null(Z[train], Z[ev], y, train, ev, cfg, ridge_alpha=1e-6, knn_k=5)
    assert not out["ridge_pass"]


def test_occupancy_coverage_counts_unseen_test_bins():
    from realtime.quadrant_n5_run import _occupancy_coverage

    ytr = np.array([[10.0, 10.0], [12.0, 12.0]])
    yte = np.array([[10.0, 10.0], [90.0, 90.0]])
    cov = _occupancy_coverage(ytr, yte, arena_cm=100.0, n_bins=10)
    assert cov["fraction_test_in_train_occupied_bins"] == pytest.approx(0.5)


def test_sim_fit_hash_keyed_on_seed_and_config(tmp_path):
    from realtime.quadrant_n5 import (
        promote_unkeyed_sim,
        sim_fit_hash,
        sim_provenance_matches,
    )

    cfg = load_quadrant_n5_yaml()
    components = list(cfg["seeds"]["components"])
    s0 = derive_seed_streams(20260923, 5, 0, components)
    s1 = derive_seed_streams(20260923, 5, 1, components)
    h0 = sim_fit_hash(cfg, s0)
    h1 = sim_fit_hash(cfg, s1)
    assert h0 != h1
    other = dict(cfg)
    other["config_sha256"] = "0" * 64
    assert sim_fit_hash(other, s0) != h0
    # methods stream is not part of the sim identity
    s0_methods = dict(s0)
    s0_methods["methods"] = s0["methods"] + 1
    assert sim_fit_hash(cfg, s0_methods) == h0

    dest = tmp_path / "sim"
    dest.mkdir()
    (dest / "behavior.csv").write_text("t\n0\n")
    (dest / "spikes_sorted.csv").write_text("time\n0\n")
    assert not sim_provenance_matches(dest, h0)
    assert not promote_unkeyed_sim(dest, cfg, s0, h0)

    import json

    (dest / "summary.json").write_text(
        json.dumps({"seed": int(s0["data_seed"])}) + "\n"
    )
    (dest / "quadrant_n5_sim.json").write_text(
        json.dumps({
            "config_sha256": cfg["config_sha256"],
            "seed_streams": {k: int(v) for k, v in s0.items()},
        }) + "\n"
    )
    assert promote_unkeyed_sim(dest, cfg, s0, h0)
    assert sim_provenance_matches(dest, h0)
    assert not sim_provenance_matches(dest, h1)
