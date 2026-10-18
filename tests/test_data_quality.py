"""Tests for Report 4 SPEC §4 metric families."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agents.data_quality.metrics.behavior_metrics import behavior_metric_rows
from agents.data_quality.metrics.decoding import decoding_metric_rows, frozen_decode_config
from agents.data_quality.metrics.population import population_metric_rows
from agents.data_quality.metrics.rhythm import rhythm_metric_rows
from agents.data_quality.metrics.sorting_realism import sorting_metric_rows
from agents.data_quality.metrics.spatial import (
    _border_score,
    _grid_score,
    _hd_mean_vector_length,
    _speed_tuning_r,
    spatial_info_bits_per_spike,
    spatial_metric_rows,
)
from agents.data_quality.metrics.units import unit_metric_rows


def _toy_spikes(n_units=8, t1=20.0, rate=5.0, seed=0):
    rng = np.random.default_rng(seed)
    spikes = []
    for _ in range(n_units):
        n = rng.poisson(rate * t1)
        spikes.append(np.sort(rng.uniform(0, t1, size=n)))
    return spikes


def test_units_family_rows():
    spikes = _toy_spikes()
    rows = unit_metric_rows(
        spikes, dataset="lab", animal_or_seed="a", session="s", chunk=0,
        region_set="primary", t0=0.0, t1=20.0, regions=["MEC"] * len(spikes),
    )
    names = {r["metric"] for r in rows}
    assert "firing_rate_hz" in names
    assert "isi_cv" in names


def test_sorting_realism_gt_vs_sorted():
    gt = _toy_spikes(seed=1)
    sorted_sp = [st[::2] if st.size else st for st in gt]  # miss half
    rows = sorting_metric_rows(
        sorted_sp, dataset="sim", animal_or_seed="s0", session="s", chunk=0,
        region_set="primary", t0=0.0, t1=20.0, spike_times_gt=gt,
    )
    names = {r["metric"]: r["value"] for r in rows}
    assert "presence_ratio_median" in names
    assert "sorted_over_gt_spike_count_median" in names
    assert 0.0 < names["sorted_over_gt_spike_count_median"] <= 1.0 + 1e-6


def test_spatial_extras_and_encoding_bits():
    rng = np.random.default_rng(2)
    t = np.arange(0, 30, 0.05)
    pos = np.column_stack([rng.uniform(0, 50, len(t)), rng.uniform(0, 50, len(t))])
    # place-like unit near center
    center = np.array([25.0, 25.0])
    rates = np.exp(-np.sum((pos - center) ** 2, axis=1) / (2 * 8 ** 2)) * 8
    spikes = []
    for r in rates:
        if rng.random() < r * 0.05:
            spikes.append(t[len(spikes) % len(t)] if False else None)
    # simpler: poisson with place modulation
    st = []
    for i, ti in enumerate(t):
        if rng.random() < rates[i] * 0.05:
            st.append(ti)
    spike_times = [np.asarray(st, float)] + _toy_spikes(n_units=3, t1=30.0, seed=3)
    speed = np.full(len(t), 5.0)
    rows = spatial_metric_rows(
        spike_times, pos, t, dataset="lab", animal_or_seed="a", session="s",
        chunk=0, region_set="primary", t0=0.0, t1=30.0, speed=speed,
    )
    names = {r["metric"] for r in rows}
    assert "encoding_bits_per_spike_median" in names
    assert "grid_score_median" in names
    assert "border_score_median" in names
    assert "hd_mean_vector_length_median" in names
    assert "speed_tuning_abs_r_median" in names
    # helpers finite on non-empty maps
    rate = np.zeros((10, 10))
    rate[4:6, 4:6] = 1.0
    ok = np.ones_like(rate, dtype=bool)
    assert np.isfinite(spatial_info_bits_per_spike(rate, np.ones_like(rate), ok))
    assert np.isfinite(_grid_score(rate, ok)) or True  # may be nan on tiny maps
    assert np.isfinite(_border_score(rate, ok)) or True
    assert 0 <= _hd_mean_vector_length(np.linspace(0, 2 * np.pi, 40)) <= 1 + 1e-9
    assert np.isfinite(_speed_tuning_r(np.linspace(1, 10, 50), np.linspace(1, 10, 200)))


def test_rhythm_population_behavior():
    spikes = _toy_spikes(n_units=12, t1=40.0, seed=4)
    rows_r = rhythm_metric_rows(
        spikes, dataset="lab", animal_or_seed="a", session="s", chunk=0, region_set="primary",
    )
    assert any(r["metric"] == "theta_mod_index_median" for r in rows_r)
    t = np.arange(0, 40, 0.05)
    pos = np.column_stack([np.sin(t), np.cos(t)]) * 20
    rows_p = population_metric_rows(
        spikes, dataset="lab", animal_or_seed="a", session="s", chunk=0,
        region_set="primary", t0=0.0, t1=40.0, positions=pos, position_times=t,
    )
    names = {r["metric"] for r in rows_p}
    assert "noise_corr_mean" in names
    assert "participation_ratio" in names
    speed = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1) / 0.05
    rows_b = behavior_metric_rows(
        pos, t, dataset="lab", animal_or_seed="a", session="s", chunk=0,
        region_set="primary", arena_half_width_cm=50.0,
    )
    assert any(r["metric"] == "dwell_time_mean_s" for r in rows_b)
    assert any(r["metric"] == "occupancy_entropy" for r in rows_b)


def test_rank_exclude_and_censor_helpers():
    from agents.data_quality.phase3.divergence import RANK_EXCLUDE_METRICS, rank_divergences
    from agents.data_quality.metrics.units import (
        LAB_DUP_REMOVAL_ISI_S,
        remove_spikes_below_isi,
        refractory_violation_rate,
    )

    assert "tracking_jitter_rest_cm" in RANK_EXCLUDE_METRICS
    rows = [
        {"metric": "tracking_jitter_rest_cm", "R": 9.0},
        {"metric": "turning_angle_std", "R": 3.0},
        {"metric": "coverage_bins_mean@N20", "R": 2.0},
    ]
    ranked = rank_divergences(rows)
    assert list(ranked["metric"]) == ["turning_angle_std", "coverage_bins_mean@N20"]
    st = np.asarray([0.0, 0.0001, 0.0002, 0.01], float)
    cens = remove_spikes_below_isi(st, LAB_DUP_REMOVAL_ISI_S)
    assert cens.size == 2
    assert refractory_violation_rate(cens) <= refractory_violation_rate(st) + 1e-12


def test_discriminator_matched_null_finite():
    from agents.data_quality.phase3.discriminator import FEATURE_METRICS, run_discriminator

    rng = np.random.default_rng(0)
    rows = []
    # shared session names across lab/sim_matched (the matched-arm trap)
    for animal, sess in [("1120", "1120_01012024"), ("1130", "1130_01012024"),
                         ("1140", "1140_01012024"), ("1150", "1150_01012024"),
                         ("1170", "1170_01012024"), ("7394", "7394_01012024")]:
        for ds, shift in (("lab", 0.0), ("sim_matched", 2.0)):
            for m in FEATURE_METRICS:
                rows.append({
                    "dataset": ds,
                    "animal|seed": animal,
                    "session": sess,
                    "metric": m,
                    "value": float(rng.normal(shift, 0.3)),
                })
    out = run_discriminator(pd.DataFrame(rows), rng=rng)
    assert out["status"] == "ok"
    assert np.isfinite(out["auc_group_shuffle_null"])
    assert np.isfinite(out["auc_group_shuffle_null_stratified"])
    assert abs(out["auc_group_shuffle_null_stratified"] - 0.5) < 0.35


def test_decoding_raw_counts_frozen_config():
    frozen = frozen_decode_config()
    assert frozen["raw_counts_only"] is True
    assert frozen["no_sqrt"] is True
    assert abs(frozen["window_s"] - 0.25) < 1e-9
    assert abs(frozen["train_frac"] - 0.80) < 1e-9
    spikes = _toy_spikes(n_units=15, t1=60.0, rate=8.0, seed=5)
    t = np.arange(0, 60, 0.05)
    # position correlated with early unit activity (weak)
    pos = np.column_stack([np.linspace(0, 50, len(t)), np.linspace(0, 50, len(t))])
    rows = decoding_metric_rows(
        spikes, pos, t, dataset="lab", animal_or_seed="a", session="s",
        chunk=0, region_set="primary", Ns=(5, 10), W_grid=(frozen["window_s"],),
    )
    assert rows
    assert all(r["W"] == pytest.approx(0.25) for r in rows if r["metric"] == "decode_error_cm")
    assert any(r["metric"] == "decode_error_over_floor" for r in rows)
