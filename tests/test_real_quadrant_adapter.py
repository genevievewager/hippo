"""Synthetic tests for the real-data adapter (no HIPPO_DATA_ROOT required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.real_quadrant.adapter import (
    ExtentError,
    IntegrityError,
    assert_path_extent_matches_boundary,
    build_segment_bundle,
    causal_count_matrix,
    centre_window_match_fraction,
    window_count_matrix,
)
from realtime.quadrant_n5_run import (
    SEGMENT_TRIM_END_S,
    SEGMENT_TRIM_START_S,
    analyze_source,
    prepared_source_bundle,
    segment_retained_mask,
)


def _write_synthetic_session(
    dest: Path,
    *,
    corrupt_spikes: bool = False,
    session_s: float = 200.0,
    dt: float = 0.05,
    n_units: int = 40,
    path_scale: float = 1.0,
) -> Path:
    """Create a minimal session dir with centre-labeled Cell_* and spike lists.

    ``postions_dataset.csv`` holds cm coords (boundary-sized path).
    ``dataset.csv`` X/Y are deliberately normalized (~0–1) so using them as
    targets fails the extent check.
    """
    dest.mkdir(parents=True, exist_ok=True)
    t0, t1 = 10.0, 10.0 + session_s
    grid = np.arange(t0, t1, dt)
    rng = np.random.default_rng(0)
    spike_lists = []
    for u in range(n_units):
        n_spk = int(rng.integers(80, 200))
        st = np.sort(rng.uniform(t0 - 0.2, t1 + 0.2, size=n_spk))
        spike_lists.append(st)
    # Centre Cell_* and causal features
    cell = window_count_matrix(spike_lists, grid, -0.125, 0.125)
    if corrupt_spikes:
        # Drop half the spikes from unit 0's list so integrity fails.
        spike_lists[0] = spike_lists[0][::2]
    # cm path covering ~90% of the 80×120 cm boundary (unless path_scale shrinks it).
    x_cm = 40.0 + 36.0 * path_scale * np.sin(2 * np.pi * (grid - t0) / (session_s / 3))
    y_cm = 60.0 + 54.0 * path_scale * np.cos(2 * np.pi * (grid - t0) / (session_s / 5))
    valid = np.ones(len(grid), dtype=bool)
    valid[::17] = False
    # dataset.csv: normalized X/Y (wrong for targets) + Cell_*
    ds = {
        "timestamp": grid,
        "X": x_cm / 80.0,
        "Y": y_cm / 120.0,
        "V": np.ones(len(grid)),
        "HD": np.zeros(len(grid)),
        "valid": valid,
        "room": ["A"] * len(grid),
    }
    for u in range(n_units):
        ds[f"Cell_{u + 1}"] = cell[:, u]
    # NON-SOMA unit present in dataset but must be dropped by BClabel filter.
    ds[f"Cell_{n_units + 1}"] = np.zeros(len(grid), dtype=int)
    pd.DataFrame(ds).to_csv(dest / "dataset.csv", index=False)
    # postions_dataset.csv: true cm coordinates
    pos = pd.DataFrame({
        "timestamp": grid,
        "X": x_cm,
        "Y": y_cm,
        "V": np.ones(len(grid)),
        "HD": np.zeros(len(grid)),
        "valid": valid,
        "room": ["A"] * len(grid),
    })
    pos.to_csv(dest / "postions_dataset.csv", index=False)
    rows = []
    for u in range(n_units):
        st = spike_lists[u]
        rows.append({
            "cell": u + 1,
            "timestamp": ",".join(f"{v:.9f}" for v in st),
            "spike": int(st.size),
            "BClabel": "GOOD",
            "Region": "MEC" if u % 2 == 0 else "SUB",
        })
    rows.append({
        "cell": n_units + 1,
        "timestamp": "1.0,2.0,3.0",
        "spike": 3,
        "BClabel": "NON-SOMA",
        "Region": "VI",
    })
    pd.DataFrame(rows).to_csv(dest / "clusters_dataset.csv", index=False)
    cfg = {
        "preprocessing": {
            "boundary": [
                {"Room": 0, "X": 0.0, "Y": 0.0},
                {"Room": 0, "X": 80.0, "Y": 0.0},
                {"Room": 0, "X": 80.0, "Y": 120.0},
                {"Room": 0, "X": 0.0, "Y": 120.0},
            ],
            "map_rooms": {
                "index": {"0": "A"},
                "rooms_list": ["A"],
                "rooms": {"A": {"index": 0, "order": 0, "range": [t0, t1]}},
            },
        },
        "project_info": {
            "animal": "0", "date": "0", "door": "center",
            "rooms_orientation": 1, "rooms_split": 2, "symmetry": 0,
        },
        "model": {
            "input_dim": n_units,
            "n_total_cells": n_units,
            "n_active_cells": n_units,
        },
    }
    (dest / "config.yaml").write_text(yaml.safe_dump(cfg))
    return dest


def test_causal_features_ignore_future_spikes():
    grid = np.array([1.0, 1.05, 1.10])
    spikes = [np.array([0.9, 0.99, 1.02])]
    base = causal_count_matrix(spikes, grid, window_s=0.25)
    # Only spikes at times >= every label t (strictly after the last label).
    spikes_future = [np.concatenate([spikes[0], np.array([1.10, 1.11, 2.0])])]
    with_future = causal_count_matrix(spikes_future, grid, window_s=0.25)
    assert np.array_equal(base, with_future)
    # And per-label: adding a spike at time t leaves feature[t] unchanged.
    for i, t in enumerate(grid):
        spiked = [np.concatenate([spikes[0], np.array([t, t + 1e-9])])]
        got = causal_count_matrix(spiked, grid, window_s=0.25)
        assert got[i, 0] == base[i, 0]


def test_integrity_passes_and_causal_differs_from_cell(tmp_path):
    sess = _write_synthetic_session(tmp_path / "synth_ok")
    # Point HIPPO_DATA_ROOT at parent; session_name is folder name.
    bundle = build_segment_bundle(
        sess.name,
        room="A",
        data_root=sess.parent,
        cache_root=tmp_path / "cache",
        min_units=30,
        min_valid_frac=0.5,
        trim_start_s=60.0,
        trim_end_s=10.0,
        require_ratemap_stability=False,
    )
    assert bundle["meta"]["integrity_match_fraction"] == pytest.approx(1.0)
    assert bundle["meta"]["target_source"] == "postions_dataset.csv"
    assert bundle["meta"]["target_units"] == "cm"
    # Room-local y should be tens of cm, not ~1.
    y = bundle["y"]
    m = np.isfinite(y).all(axis=1)
    assert float(np.nanmax(np.abs(y[m]))) > 10.0
    # Reload Cell_* and ensure causal ≠ centre Cell on the segment grid
    ds = pd.read_csv(sess / "dataset.csv")
    unit_ids = bundle["unit_ids"]
    cell = np.column_stack([
        ds.loc[
            (ds.timestamp >= bundle["segment_t0"])
            & (ds.timestamp < bundle["segment_t1"]),
            f"Cell_{u}",
        ].to_numpy(dtype=float)
        for u in unit_ids
    ])
    assert bundle["X_counts"].shape == cell.shape
    assert not np.array_equal(
        np.rint(bundle["X_counts"]), np.rint(cell)
    ), "causal features must differ from centre-labeled Cell_*"


def test_extent_check_rejects_normalized_coords():
    # Path bbox ~1×1 on a 100×100 boundary → ratios ~0.01.
    with pytest.raises(ExtentError, match="postions_dataset"):
        assert_path_extent_matches_boundary(
            np.array([0.1, 0.9]),
            np.array([0.2, 0.8]),
            boundary_width_cm=100.0,
            boundary_height_cm=100.0,
        )


def test_extent_check_rejects_shrunken_path_in_adapter(tmp_path):
    sess = _write_synthetic_session(tmp_path / "synth_tiny", path_scale=0.05)
    with pytest.raises(ExtentError):
        build_segment_bundle(
            sess.name,
            room="A",
            data_root=sess.parent,
            cache_root=tmp_path / "cache",
            min_units=30,
            min_valid_frac=0.5,
            require_ratemap_stability=False,
        )


def test_integrity_fails_on_corrupted_spikes(tmp_path):
    sess = _write_synthetic_session(tmp_path / "synth_bad", corrupt_spikes=True)
    with pytest.raises(IntegrityError):
        build_segment_bundle(
            sess.name,
            room="A",
            data_root=sess.parent,
            cache_root=tmp_path / "cache",
            min_units=30,
            min_valid_frac=0.5,
            require_ratemap_stability=False,
        )


def test_ratemap_stability_refuses_unstable_synthetic(tmp_path):
    from analysis.real_quadrant.adapter import RateMapStabilityError

    sess = _write_synthetic_session(tmp_path / "synth_unstable")
    with pytest.raises(RateMapStabilityError, match="rate-map"):
        build_segment_bundle(
            sess.name,
            room="A",
            data_root=sess.parent,
            cache_root=tmp_path / "cache",
            min_units=30,
            min_valid_frac=0.5,
            require_ratemap_stability=True,
        )


def test_segment_trims_and_warmup_in_analyze_source(tmp_path, monkeypatch):
    import realtime.quadrant_n5_run as run
    from realtime.quadrant_n5 import load_quadrant_n5_yaml

    monkeypatch.setattr(run, "OUTPUT_ROOT", tmp_path / "out")
    monkeypatch.setattr(run, "_git_dirty", lambda: False)
    t0, t1 = 0.0, 200.0
    dt = 0.05
    times = np.arange(t0, t1, dt)
    n = len(times)
    rng = np.random.default_rng(1)
    X = rng.poisson(0.3, size=(n, 8)).astype(float)
    y = np.column_stack([
        10.0 * np.sin(times / 30.0),
        10.0 * np.cos(times / 30.0),
    ])
    units_df = pd.DataFrame({
        "unit_id": np.arange(8),
        "region": ["MEC"] * 8,
        "cell_type": ["GOOD"] * 8,
    })
    bundle = prepared_source_bundle(
        X_counts=X,
        y=y,
        decode_times=times,
        unit_ids=list(range(8)),
        units_df=units_df,
        arena_cm=40.0,
        segment_t0=t0,
        segment_t1=t1,
        apply_segment_trims=True,
        y_is_room_local=True,
        meta={"test": "trims"},
    )
    retained = segment_retained_mask(times, t0, t1)
    assert retained.sum() < n
    assert times[retained][0] >= t0 + SEGMENT_TRIM_START_S - 1e-9
    assert times[retained][-1] < t1 - SEGMENT_TRIM_END_S + 1e-9

    cfg = dict(load_quadrant_n5_yaml())
    cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
    streams = {"methods": 1, "data_seed": 1, "master_seed": 1, "seed_index": 0}
    out = analyze_source(
        cfg, None, "real", streams, 0,
        bundle=bundle,
        output_dir=tmp_path / "out" / "seg",
        method_keys=("raw",),
    )
    assert out["segment"]["trim_start_s"] == SEGMENT_TRIM_START_S
    assert out["segment"]["trim_end_s"] == SEGMENT_TRIM_END_S
    assert out["n_train"] + out["n_eval"] <= int(retained.sum())
    # Warm-up: decode_times still include the pre-trim prefix.
    assert float(bundle["decode_times"][0]) == pytest.approx(t0)
