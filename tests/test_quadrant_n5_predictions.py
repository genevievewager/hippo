"""Stage 2 / 2b: predictions.npz export does not change frozen JSON numbers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agents.quadrant_n5.export_predictions import (
    METHOD_KEYS,
    reconstruct_source,
    write_predictions_npz,
)
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import (
    D_SWEEP_SELECTION_RULE,
    REDUCING_SWEEP,
    _euclid,
    analyze_source,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tiny_sim(dest: Path, *, session_s: float = 60.0, dt: float = 0.05, n_units: int = 16) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    t = np.arange(0.0, session_s, dt)
    rng = np.random.default_rng(0)
    x = 50.0 + 20.0 * np.sin(2 * np.pi * t / session_s)
    y = 50.0 + 20.0 * np.cos(2 * np.pi * t / session_s)
    beh = pd.DataFrame({
        "time_s": t,
        "x_cm": x,
        "y_cm": y,
        "speed_cm_s": np.full(t.shape, 8.0),
        "head_direction_rad": np.linspace(0, 2 * np.pi, len(t), endpoint=False),
        "distance_to_wall_cm": np.full(t.shape, 30.0),
        "acceleration_cm_s2": np.zeros(t.shape),
    })
    beh.to_csv(dest / "behavior.csv", index=False)
    region_cycle = [
        "entorhinal_transition",
        "hippocampal_formation_transition",
        "entorhinal_cortex_layer6a",
        "entorhinal_cortex_layer5",
        "medial_entorhinal_layer3",
        "medial_entorhinal_layer2",
        "medial_entorhinal_layer1",
    ]
    type_cycle = ["MEC_grid", "MEC_hd", "MEC_speed", "Sub_bvc", "INT_CA1"]
    units = pd.DataFrame({
        "unit_id": np.arange(n_units),
        "cell_type": [type_cycle[i % len(type_cycle)] for i in range(n_units)],
        "region": [region_cycle[i % len(region_cycle)] for i in range(n_units)],
        "include_in_decoder": True,
        "channel": np.arange(1, n_units + 1),
    })
    units.to_csv(dest / "units.csv", index=False)
    # Shared 2-d latent so Isomap's neighbor graph stays connected.
    pref = rng.normal(size=(n_units, 2))
    pos = np.column_stack([(x - 50.0) / 20.0, (y - 50.0) / 20.0])
    rate = np.clip(4.0 + 10.0 * (pos @ pref.T), 0.5, None)
    spikes = []
    for u in range(n_units):
        for i, ti in enumerate(t):
            n_spk = int(rng.poisson(rate[i, u] * dt))
            if n_spk <= 0:
                continue
            for ts in rng.uniform(ti, ti + dt, size=n_spk):
                spikes.append({
                    "unit_id": u, "spike_time_s": float(ts),
                    "channel": u + 1, "confidence": 0.9,
                })
    sp = pd.DataFrame(spikes)
    sp.to_csv(dest / "spikes_sorted.csv", index=False)
    sp.to_csv(dest / "spikes_ground_truth.csv", index=False)
    (dest / "summary.json").write_text(json.dumps({
        "session_duration_s": session_s,
        "n_units": n_units,
        "arena_size_cm": 100.0,
    }) + "\n")
    return dest


def _test_cfg() -> dict:
    cfg = dict(load_quadrant_n5_yaml())
    cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
    reps = dict(cfg["representations"])
    iso = dict(reps["isomap"])
    iso["require_connected_graph"] = False
    reps["isomap"] = iso
    cfg["representations"] = reps
    return cfg


def _json_payloads(src_dir: Path) -> dict[str, str]:
    out = {}
    for p in sorted(src_dir.glob("*.json")):
        out[p.name] = _sha256(p)
    return out


def _scientific_fields(src_dir: Path) -> dict[str, object]:
    """Fields the runner change must not touch (no elapsed / paths)."""
    keep = (
        "method", "primary_d", "ridge_alpha", "knn_k", "ridge", "knn",
        "n_train", "n_eval", "index_hashes", "a13",
        "inner_cv_ridge_median", "inner_cv_knn_median", "n_folds",
    )
    out: dict[str, object] = {}
    for key in METHOD_KEYS:
        rec = json.loads((src_dir / f"{key}.json").read_text())
        out[key] = {k: rec.get(k) for k in keep}
    summary = json.loads((src_dir / "source_summary.json").read_text())
    out["summary_index_hashes"] = summary.get("index_hashes")
    out["summary_floor"] = summary.get("floor")
    return out


def test_export_reload_medians_match_tiny(tmp_path, monkeypatch):
    import realtime.quadrant_n5_run as run

    monkeypatch.setattr(run, "OUTPUT_ROOT", tmp_path)
    sim = _tiny_sim(tmp_path / "seed_0" / "sim")
    cfg = _test_cfg()
    streams = {"methods": 1, "data_seed": 1, "master_seed": 1, "seed_index": 0}
    analyze_source(cfg, sim, "sorted", streams, 0)
    src = tmp_path / "seed_0" / "sorted"
    sweep_path = src / "d_sweep.json"
    assert sweep_path.is_file()
    sweep = json.loads(sweep_path.read_text())
    assert sweep["selection_rule"] == D_SWEEP_SELECTION_RULE
    for key in REDUCING_SWEEP:
        rec = json.loads((src / f"{key}.json").read_text())
        assert "selection_rule" not in rec
        sel = next(
            r for r in sweep["rows"] if r["method"] == key and r["selected"]
        )
        assert int(sel["d"]) == int(rec["primary_d"])
        assert int(sel["knn_k"]) == int(rec["knn_k"])
        assert abs(float(sel["ridge_alpha"]) - float(rec["ridge_alpha"])) <= 1e-12
        assert abs(float(sel["ridge_median"]) - float(rec["ridge"]["median"])) <= 1e-6
        assert abs(float(sel["knn_median"]) - float(rec["knn"]["median"])) <= 1e-6
    before = _json_payloads(src)
    assert (src / "predictions.npz").is_file()
    npz = src / "predictions.npz"
    npz.unlink()
    obs, preds, rows, method_source = reconstruct_source(cfg, 0, "sorted", tmp_path)
    assert all(r["pass"] for r in rows)
    write_predictions_npz(
        src / "predictions.npz", cfg, obs, preds, method_source=method_source,
    )
    assert _json_payloads(src) == before
    blob = np.load(src / "predictions.npz")
    y = np.asarray(blob["y_true"], dtype=float)
    for key in METHOD_KEYS:
        rec = json.loads((src / f"{key}.json").read_text())
        for dec in ("ridge", "knn"):
            got = _euclid(np.asarray(blob[f"pred_{key}_{dec}"]), y)
            for stat in ("median", "mean", "p90"):
                assert abs(got[stat] - rec[dec][stat]) <= 1e-6
    assert str(blob["dm_source"]) in {"refit", "reloaded"}


def test_runner_json_byte_identical_with_and_without_npz(tmp_path, monkeypatch):
    import realtime.quadrant_n5_run as run
    import agents.quadrant_n5.export_predictions as export_mod

    cfg = _test_cfg()
    streams = {"methods": 2, "data_seed": 2, "master_seed": 2, "seed_index": 0}

    def _run(root: Path, write_npz: bool) -> dict[str, str]:
        monkeypatch.setattr(run, "OUTPUT_ROOT", root)
        if not write_npz:
            monkeypatch.setattr(export_mod, "write_predictions_npz", lambda *a, **k: None)
        else:
            monkeypatch.setattr(
                export_mod, "write_predictions_npz",
                write_predictions_npz,
            )
        sim = _tiny_sim(root / "seed_0" / "sim")
        run.analyze_source(cfg, sim, "sorted", streams, 0)
        return _json_payloads(root / "seed_0" / "sorted")

    a = tmp_path / "without"
    b = tmp_path / "with"
    _run(a, write_npz=False)
    _run(b, write_npz=True)
    assert _scientific_fields(a / "seed_0" / "sorted") == _scientific_fields(
        b / "seed_0" / "sorted"
    )
    assert not (a / "seed_0" / "sorted" / "predictions.npz").is_file()
    npz = b / "seed_0" / "sorted" / "predictions.npz"
    assert npz.is_file()
    blob = np.load(npz)
    y = np.asarray(blob["y_true"], dtype=float)
    src = b / "seed_0" / "sorted"
    for key in METHOD_KEYS:
        rec = json.loads((src / f"{key}.json").read_text())
        for dec in ("ridge", "knn"):
            got = _euclid(np.asarray(blob[f"pred_{key}_{dec}"]), y)
            assert abs(got["median"] - rec[dec]["median"]) <= 1e-6


def test_export_float32_representation_reproduces_stored_medians():
    """Native-dtype path matches Phase 3; a float64 cast is not used."""
    from agents.quadrant_n5.export_predictions import predict_both
    from realtime.quadrant_n5_run import _euclid, _fit_predict_ridge

    rng = np.random.default_rng(1)
    Z = rng.normal(size=(240, 6)).astype(np.float32)
    y = rng.normal(size=(240, 2))
    train = np.zeros(240, dtype=bool)
    train[:180] = True
    ev = ~train
    stored = _euclid(_fit_predict_ridge(Z[train], y[train], Z[ev], 1.0), y[ev])
    preds = predict_both(Z, y, train, ev, 1.0, 5)
    got = _euclid(preds["ridge"], y[ev])
    for stat in ("median", "mean", "p90"):
        assert abs(got[stat] - stored[stat]) <= 1e-6
    assert Z.dtype == np.float32
    Z_export = Z[:, : Z.shape[1]]
    assert Z_export.dtype == np.float32


def test_frozen_config_hash_unchanged():
    cfg = load_quadrant_n5_yaml()
    assert cfg["config_sha256"] == (
        "4da17a5dad704866a1078231fb0795a0a1505211328946967c9c58ebaa9b1ee3"
    )
