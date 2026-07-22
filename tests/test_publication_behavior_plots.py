"""Smoke tests for publication behavioral multi-panel figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from visualization.figure_captions import CAPTIONS, caption_for
from visualization.load_outputs import SimulationOutputs
from visualization.publication_behavior_plots import (
    plot_fig_behavior_dynamics,
    plot_fig_behavior_features,
    plot_fig_behavior_overview,
)


def _mini_data() -> SimulationOutputs:
    n = 200
    t = np.linspace(0, 20, n)
    ang = np.linspace(0, 4 * np.pi, n)
    beh = pd.DataFrame({
        "time": t,
        "x": 50 + 30 * np.cos(ang),
        "y": 50 + 30 * np.sin(ang),
        "speed": 10 + 5 * np.sin(ang),
        "head_direction": np.mod(ang, 2 * np.pi),
        "acceleration": np.cos(ang),
        "distance_to_wall": 20 + 5 * np.sin(ang / 2),
        "theta_phase": np.mod(t * 8, 2 * np.pi),
        "ripple": np.zeros(n),
    })
    units = pd.DataFrame({
        "unit_id": [0, 1],
        "cell_type": ["CA1_pyr", "MEC_hd"],
        "region": ["CA1", "MEC"],
        "rate_model": ["ratinabox_CA1_place_pp", "ratinabox_MEC_hd"],
    })
    spikes = pd.DataFrame({"time": [0.1, 0.2], "unit_id": [0, 1]})
    return SimulationOutputs(
        input_dir=Path("."),
        behavior=beh,
        units=units,
        spikes_gt=spikes,
        spikes_sorted=spikes,
        summary={"bounds": [0, 100, 0, 100]},
        bounds=(0.0, 100.0, 0.0, 100.0),
        session_duration_s=20.0,
        behavior_dt=0.1,
        unit_mean_rates_gt=pd.Series({0: 5.0, 1: 3.0}),
    )



def test_behavior_caption_stems():
    for stem in (
        "fig_behavior_overview",
        "fig_behavior_dynamics",
        "fig_behavior_features",
        "fig_neural_drivers",
    ):
        assert stem in CAPTIONS
        text = caption_for(Path(f"figures/behavior/{stem}.png"), figure_number=1)
        assert text.startswith("Figure 1. ")


def test_behavior_figs_write(tmp_path: Path):
    data = _mini_data()
    beh_dir = tmp_path / "behavior"
    feat_dir = tmp_path / "features"
    assert plot_fig_behavior_overview(data, beh_dir).exists()
    assert plot_fig_behavior_dynamics(data, beh_dir).exists()
    assert plot_fig_behavior_features(data, feat_dir).exists()
    # legacy cleanup leaves only fig_*
    (beh_dir / "behavior_trajectory_xy.png").write_bytes(b"x")
    from visualization.publication_behavior_plots import _cleanup_legacy_pngs
    _cleanup_legacy_pngs(beh_dir)
    assert not (beh_dir / "behavior_trajectory_xy.png").exists()
    assert (beh_dir / "fig_behavior_overview.png").exists()
