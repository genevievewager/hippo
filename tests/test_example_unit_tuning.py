"""Tests for population primary-feature tuning figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from visualization.figure_captions import caption_for
from visualization.load_outputs import SimulationOutputs
from visualization.neural_plots import (
    _FEATURED_CELL_CLASSES,
    _FEATURED_TUNING_PANELS,
    _PRIMARY_TUNING,
    _mean_rate_1d,
    _spatial_rate_map,
    plot_population_tuning,
    plot_spikes_on_trajectory_by_class,
)


def test_primary_tuning_covers_canonical_classes():
    for ct in (
        "CA1_pyr", "CA2_pyr", "CA3_pyr", "DG_granule",
        "MEC_grid", "MEC_hd", "MEC_speed", "Sub_bvc",
        "INT_CA1", "INT_CA3", "INT_DG", "INT_CA2", "INT_SUB",
    ):
        assert ct in _PRIMARY_TUNING


def test_featured_panels_match_requested_classes():
    assert _FEATURED_CELL_CLASSES == {
        "MEC_grid", "MEC_hd", "MEC_speed", "Sub_bvc",
    }
    assert _FEATURED_TUNING_PANELS["MEC_grid"] == "place_mosaic"
    assert _FEATURED_TUNING_PANELS["MEC_hd"] == "tuning_heatmap"
    assert _FEATURED_TUNING_PANELS["MEC_speed"] == "tuning_heatmap"
    assert _FEATURED_TUNING_PANELS["Sub_bvc"] == "tuning_heatmap"


def test_head_direction_wraps_negative_angles(tmp_path):
    """atan2-style HD in [-π, π] must fill the full [0, 2π) tuning axis."""
    from visualization.neural_plots import _feature_vector, _unit_tuning_1d

    data = _toy_featured_data(tmp_path)
    data.behavior["head_direction"] = np.linspace(-np.pi, np.pi, len(data.behavior))
    feat, feat_range, n_bins = _feature_vector(data, "head_direction")
    assert feat_range == (0.0, 2 * np.pi)
    assert np.min(feat) >= 0.0
    assert np.max(feat) < 2 * np.pi + 1e-9
    uid = int(data.units.loc[data.units["cell_type"] == "MEC_hd", "unit_id"].iloc[0])
    centers, means = _unit_tuning_1d(data, uid, "head_direction")
    assert centers.shape == (n_bins,)
    assert np.isfinite(means).sum() >= n_bins - 2


def test_mean_rate_1d_recovers_linear_speed_tuning():
    speed = np.linspace(0, 20, 500)
    rate = 0.5 * speed + 1.0
    centers, means = _mean_rate_1d(speed, rate, n_bins=10, feat_range=(0.0, 20.0))
    assert centers.shape == (10,)
    finite = np.isfinite(means)
    assert finite.sum() >= 8
    assert means[finite][-1] > means[finite][0]


def test_spatial_rate_map_peaks_near_place_field():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 100, 2000)
    y = rng.uniform(0, 100, 2000)
    rate = np.exp(-((x - 30) ** 2 + (y - 70) ** 2) / (2 * 8.0 ** 2)) * 10
    rmap, extent = _spatial_rate_map(x, y, rate, (0, 100, 0, 100), n_bins=25)
    assert extent == [0, 100, 0, 100]
    assert np.isfinite(rmap).all()
    ix, iy = np.unravel_index(np.nanargmax(rmap), rmap.shape)
    xc = (ix + 0.5) * 4.0
    yc = (iy + 0.5) * 4.0
    assert abs(xc - 30) < 12
    assert abs(yc - 70) < 12


def _toy_featured_data(tmp_path) -> SimulationOutputs:
    n = 200
    t = np.arange(n) * 0.05
    beh = pd.DataFrame({
        "time": t,
        "x": np.linspace(0, 50, n) + np.sin(np.linspace(0, 6, n)) * 5,
        "y": np.linspace(0, 50, n) + np.cos(np.linspace(0, 6, n)) * 5,
        "speed": np.linspace(0, 20, n),
        "head_direction": (t * 0.3) % (2 * np.pi),
        "acceleration": np.zeros(n),
        "distance_to_wall": np.linspace(2, 30, n),
        "theta_phase": (t * 2 * np.pi * 8) % (2 * np.pi),
        "ripple": np.zeros(n),
    })
    # Enough units for featured classes.
    rows = []
    rates_list = []
    uid = 0
    for ct, n_u, maker in [
        ("MEC_grid", 6, "place"),
        ("MEC_hd", 5, "hd"),
        ("MEC_speed", 4, "speed"),
        ("Sub_bvc", 5, "boundary"),
        ("CA1_pyr", 2, "place"),
    ]:
        for i in range(n_u):
            rows.append({
                "unit_id": uid,
                "cell_type": ct,
                "region": ct.split("_")[0] if ct != "Sub_bvc" else "Subiculum",
                "rate_model": f"ratinabox_{ct}",
            })
            if maker == "place":
                cx, cy = 10 + 6 * i, 15 + 5 * i
                rates_list.append(
                    np.exp(-((beh["x"] - cx) ** 2 + (beh["y"] - cy) ** 2) / 40.0) * (2 + i)
                )
            elif maker == "hd":
                pref = i * 0.8
                rates_list.append(1.0 + 3.0 * np.exp(2 * np.cos(beh["head_direction"] - pref)))
            elif maker == "speed":
                rates_list.append((0.3 + 0.2 * i) * beh["speed"].to_numpy())
            else:
                pref = 5 + 4 * i
                rates_list.append(np.exp(-((beh["distance_to_wall"] - pref) ** 2) / 20.0) * (2 + i))
            uid += 1
    units = pd.DataFrame(rows)
    rates = np.vstack(rates_list)
    spikes = pd.DataFrame({
        "time": np.tile(t[::8], len(units)),
        "unit_id": np.repeat(units["unit_id"].to_numpy(), len(t[::8])),
    })
    return SimulationOutputs(
        input_dir=tmp_path,
        behavior=beh,
        units=units,
        spikes_gt=spikes,
        spikes_sorted=spikes,
        summary={"session_duration_s": float(t[-1])},
        anatomy=None,
        bounds=(0.0, 55.0, 0.0, 55.0),
        behavior_dt=0.05,
        session_duration_s=float(t[-1]),
        unit_mean_rates_gt=pd.Series(rates.mean(axis=1), index=units["unit_id"]),
        unit_mean_rates_sorted=pd.Series(rates.mean(axis=1), index=units["unit_id"]),
        rates_hz=rates,
        rate_times_s=t,
    )


def test_plot_population_tuning_writes_3x3_overview(tmp_path):
    data = _toy_featured_data(tmp_path)
    out = tmp_path / "neural"
    out.mkdir(parents=True)
    (out / "population_tuning_MEC_grid.png").write_bytes(b"")
    written = plot_population_tuning(data, out)
    assert len(written) == 1
    assert written[0].name == "fig_population_tuning.png"
    assert written[0].exists()
    assert not (out / "population_tuning_MEC_grid.png").exists()
    path = plot_spikes_on_trajectory_by_class(data, out)
    assert path is not None and path.exists()


def test_population_tuning_caption():
    text = caption_for(Path("figures/neural/fig_population_tuning.png"), figure_number=8)
    assert text.startswith("Figure 8. ")
    assert "heatmap" in text.lower()
    assert "rate" in text.lower()


def test_spikes_overview_caption():
    text = caption_for(
        Path("figures/neural/fig_spikes_on_trajectory_by_class.png"),
        figure_number=3,
    )
    assert text.startswith("Figure 3. ")
    assert "trajectory" in text.lower()
    assert "not in session" in text.lower()
    assert "3" in text or "cell class" in text.lower()
