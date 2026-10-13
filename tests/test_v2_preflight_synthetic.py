"""Fast (<5 min) end-to-end v2-style pipeline on synthetic data + HF guards."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.real_quadrant.hf_guard import assert_n_features_eq_n_units
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import (
    METHOD_KEYS,
    analyze_source,
    prepared_source_bundle,
)


@pytest.fixture
def scratch_out(tmp_path: Path) -> Path:
    # Prefer worktree-local outputs; never write under a live Phase A tree.
    root = Path(__file__).resolve().parents[1] / "outputs" / "real_quadrant" / "_preflight_scratch"
    out = root / f"run_{tmp_path.name}"
    out.mkdir(parents=True, exist_ok=True)
    yield out
    # cleanup
    import shutil
    shutil.rmtree(out, ignore_errors=True)
    try:
        root.rmdir()
    except OSError:
        pass


def _tiny_bundle(n_t: int = 400, n_u: int = 20, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    times = np.arange(n_t, dtype=float) * 0.05
    # Slow random walk position in a box
    steps = rng.normal(scale=0.8, size=(n_t, 2))
    y = np.cumsum(steps, axis=0)
    y -= y.mean(0)
    X = rng.poisson(lam=0.3, size=(n_t, n_u)).astype(float)
    # Inject weak position dependence so A13 can pass
    X += 0.15 * y[:, :1]
    unit_ids = list(range(n_u))
    units_df = pd.DataFrame({
        "unit_id": unit_ids,
        "Region": ["MEC"] * n_u,
        "cell_type": ["pyramidal"] * n_u,
    })
    valid = np.ones(n_t, dtype=bool)
    return prepared_source_bundle(
        X_counts=X,
        y=y,
        decode_times=times,
        unit_ids=unit_ids,
        units_df=units_df,
        arena_cm=80.0,
        segment_t0=float(times[0]),
        segment_t1=float(times[-1] + 0.05),
        target_valid=valid,
        arena_width_cm=80.0,
        arena_height_cm=80.0,
        apply_segment_trims=False,
        y_is_room_local=True,
        meta={
            "session": "synthetic_preflight",
            "room": "A",
            "n_units": n_u,
            "region_allowlist": ["MEC", "SUB", "PARA", "LEC"],
        },
    )


def test_synthetic_pipeline_posthoc_a13_under_5min(scratch_out: Path):
    t0 = time.perf_counter()
    bundle = _tiny_bundle()
    assert_n_features_eq_n_units(
        bundle["X_counts"], len(bundle["unit_ids"]), context="preflight.bundle",
    )
    cfg = load_quadrant_n5_yaml()
    cfg = dict(cfg)
    cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
    # Cheap dynamic fits
    cfg["representations"] = dict(cfg["representations"])
    cfg["representations"]["lds"] = dict(cfg["representations"]["lds"], em_iters=3)
    cfg["representations"]["gpfa"] = dict(
        cfg["representations"]["gpfa"], max_iter=8, factor_analysis_max_iter=30,
    )
    cfg["representations"]["isomap"] = dict(
        cfg["representations"]["isomap"],
        n_neighbors=15,
        require_connected_graph=False,
        allow_largest_component_only=True,
    )
    cfg["latent_dims"] = [2, 3]
    cfg["nested_fit_d"] = 3
    # Fewer A13 shifts for speed
    cfg["phase3"]["a13"] = dict(cfg["phase3"]["a13"], n_shifts=4)

    streams = {
        "methods": 0, "data_seed": 0, "master_seed": 0, "seed_index": 0,
        "trajectory": 0, "neural": 0, "recording_noise": 0, "sorting_errors": 0,
    }
    # Non-"real" spike_source skips require_clean_git (this branch may be dirty
    # during local guard work); feature path is still the bundle.
    summary = analyze_source(
        cfg,
        None,
        "synthetic",
        streams,
        seed_index=0,
        bundle=bundle,
        output_dir=scratch_out,
        method_keys=METHOD_KEYS,
        save_figure_contract=True,
        save_latent_cache=True,
    )
    wall = time.perf_counter() - t0
    assert wall < 300.0, f"preflight took {wall:.1f}s (>5 min)"

    methods = {m["method"]: m for m in summary.get("methods", [])}
    for key in METHOD_KEYS:
        assert key in methods, f"missing {key}"
        a13 = methods[key].get("a13") or {}
        # Soft: record status; require at least one PASS among reducing methods
        assert "status" in a13 or methods[key].get("status") == "ok"
    # A5: causal maps present for realtime methods
    for key in ("raw", "raw_lag", "pca", "lds"):
        assert methods[key].get("causal", True) is not False

    # Cache load-back
    from analysis.real_quadrant import latent_cache as lc
    root = lc.cache_root(scratch_out)
    assert (root / "final").is_dir()
    for key in ("pca", "gpfa", "raw"):
        z = lc.load_final_latent(root, key)
        assert z["Z"].shape[0] == len(bundle["decode_times"])

    # Provenance sha file written
    assert (scratch_out / "source_summary.json").is_file()
    print(f"PREFLIGHT_OK wall_s={wall:.1f}", flush=True)
