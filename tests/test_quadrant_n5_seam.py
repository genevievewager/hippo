"""Prepared-bundle seam for analyze_source: sim path stays byte-identical."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from realtime.quadrant_n5_run import (
    _load_arrays_for_analyze_source,
    analyze_source,
    prepared_source_bundle,
    segment_retained_mask,
)
from tests.test_quadrant_n5_predictions import (
    FAST_METHOD_KEYS,
    FAST_REDUCING_SWEEP,
    _patch_method_keys,
    _scientific_fields,
    _test_cfg,
    _tiny_sim,
)


def test_segment_retained_mask_trims_edges():
    t0, t1 = 100.0, 300.0
    times = np.arange(t0, t1, 0.05)
    m = segment_retained_mask(times, t0, t1, trim_start_s=60.0, trim_end_s=10.0)
    assert times[m][0] >= t0 + 60.0 - 1e-12
    assert times[m][-1] < t1 - 10.0 + 1e-12
    assert not m[0] and not m[-1]


def test_sim_dir_path_scientific_fields_stable_across_two_runs(tmp_path, monkeypatch):
    """Before/after seam: two sim-dir runs on a tiny sim are byte-identical."""
    import realtime.quadrant_n5_run as run

    _patch_method_keys(monkeypatch, FAST_METHOD_KEYS, FAST_REDUCING_SWEEP)
    cfg = _test_cfg()
    streams = {"methods": 11, "data_seed": 11, "master_seed": 11, "seed_index": 0}
    fields = []
    for name in ("a", "b"):
        root = tmp_path / name
        monkeypatch.setattr(run, "OUTPUT_ROOT", root)
        sim = _tiny_sim(root / "seed_0" / "sim")
        analyze_source(cfg, sim, "sorted", streams, 0)
        fields.append(_scientific_fields(root / "seed_0" / "sorted", FAST_METHOD_KEYS))
    assert fields[0] == fields[1]


def test_bundle_without_trims_matches_sim_dir_scientific_fields(tmp_path, monkeypatch):
    """Same arrays via bundle (trims off) match the sim-dir path."""
    import realtime.quadrant_n5_run as run

    _patch_method_keys(monkeypatch, FAST_METHOD_KEYS, FAST_REDUCING_SWEEP)
    cfg = _test_cfg()
    streams = {"methods": 12, "data_seed": 12, "master_seed": 12, "seed_index": 0}

    root_sim = tmp_path / "sim_path"
    monkeypatch.setattr(run, "OUTPUT_ROOT", root_sim)
    sim = _tiny_sim(root_sim / "seed_0" / "sim")
    analyze_source(cfg, sim, "sorted", streams, 0)
    sim_fields = _scientific_fields(root_sim / "seed_0" / "sorted", FAST_METHOD_KEYS)

    X, y, times, units_df, unit_ids, _meta = _load_arrays_for_analyze_source(
        cfg, sim, "sorted", bundle=None,
    )
    bundle = prepared_source_bundle(
        X_counts=X,
        y=y,
        decode_times=times,
        unit_ids=unit_ids,
        units_df=units_df,
        arena_cm=float(cfg["session"]["arena_size_cm"]),
        segment_t0=float(times[0]),
        segment_t1=float(times[-1] + cfg["features"]["update_dt"]),
        apply_segment_trims=False,
        y_is_room_local=False,
        meta={"note": "sim_equivalence"},
    )
    root_b = tmp_path / "bundle_path"
    monkeypatch.setattr(run, "OUTPUT_ROOT", root_b)
    analyze_source(
        cfg, None, "sorted", streams, 0,
        bundle=bundle,
        output_dir=root_b / "seed_0" / "sorted",
        method_keys=FAST_METHOD_KEYS,
    )
    bundle_fields = _scientific_fields(root_b / "seed_0" / "sorted", FAST_METHOD_KEYS)
    assert bundle_fields == sim_fields


def test_existing_npz_byte_identical_still_passes(tmp_path, monkeypatch):
    """Re-run the established scientific-field identity check after the seam."""
    from tests.test_quadrant_n5_predictions import (
        test_runner_json_byte_identical_with_and_without_npz,
    )

    test_runner_json_byte_identical_with_and_without_npz(tmp_path, monkeypatch)
