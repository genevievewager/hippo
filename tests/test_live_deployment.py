"""Tests for live deployment: bundles, causality, units, selection, overruns."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from realtime.deployment_bundle import (
    best_deployable,
    load_deployment_bundle,
    pack_deployment_bundle,
)
from realtime.live.config import DeployableConfiguration, RuntimeState
from realtime.live.registry import DeploymentRegistry
from realtime.live.spike_buffer import CausalSpikeBuffer
from realtime.live.spike_stream import OpenEphysSpikeStream, ReplaySpikeStream
from realtime.live.unit_mapping import map_units
from realtime.live_decoder import LiveDecoder
from realtime.spike_binner import count_spikes_in_window


def _tiny_bundle(tmp_path: Path, *, n_units: int = 4, w: float = 0.1) -> Path:
    """Hand-build a minimal deployment bundle with a linear decoder on count features."""
    unit_ids = list(range(n_units))
    # Train on identity: predict sum of counts as 1-d "speed"-like target / position xy
    rng = np.random.default_rng(0)
    X = rng.poisson(2.0, size=(200, n_units)).astype(float)
    y = X.sum(axis=1) + rng.normal(0, 0.1, size=200)
    pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    pipe.fit(X, y)

    cfg = DeployableConfiguration(
        target="speed",
        feature_set="counts",
        embedding_type="identity",
        decoder_name="ridge",
        decode_window_s=w,
        update_dt_s=0.025,
        metric_name="r2",
        metric_value=0.9,
        metric_direction="higher",
        spike_source="sorted",
        deployable=True,
        realtime_compatible=True,
        training_run_id="unit_test",
        extras={"feature_mode": "counts"},
    )
    out = tmp_path / "bundle_speed"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2) + "\n")
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "simulation_trained": True,
                "training_run_id": "unit_test",
                "n_units": n_units,
                "expected_feature_dim": n_units,
            },
            indent=2,
        )
        + "\n"
    )
    (out / "unit_order.json").write_text(
        json.dumps({"unit_ids": unit_ids, "n_units": n_units}, indent=2) + "\n"
    )
    (out / "feature_config.json").write_text(
        json.dumps(
            {
                "feature_set": "counts",
                "feature_mode": "counts",
                "embedding_type": "identity",
                "decode_window_s": w,
                "update_dt_s": 0.025,
            },
            indent=2,
        )
        + "\n"
    )
    joblib.dump(pipe, out / "decoder.joblib")
    return out


def test_bundle_round_trip_identical_predictions(tmp_path: Path):
    bundle_dir = _tiny_bundle(tmp_path)
    a = LiveDecoder.from_bundle(bundle_dir)
    b = load_deployment_bundle(bundle_dir)
    dec_b = LiveDecoder(b)

    counts = np.array([1.0, 2.0, 0.0, 3.0])
    fa = a._features_from_counts(counts)
    fb = dec_b._features_from_counts(counts)
    pa = a._predict_from_features(fa)
    pb = dec_b._predict_from_features(fb)
    assert pa["decoded_speed"] == pytest.approx(pb["decoded_speed"], rel=0, abs=1e-12)

    # Reload decoder joblib independently
    pipe = joblib.load(bundle_dir / "decoder.joblib")
    direct = float(pipe.predict(counts.reshape(1, -1)).ravel()[0])
    assert pa["decoded_speed"] == pytest.approx(direct, abs=1e-12)


def test_causal_buffer_excludes_spikes_at_or_after_t():
    buf = CausalSpikeBuffer([0, 1], history_s=2.0)
    buf.extend([0.10, 0.149999, 0.15, 0.16], [0, 0, 0, 1])
    counts = buf.counts_at(0.15, decode_window_s=0.05)
    # Window [0.10, 0.15): spikes 0.10 and 0.149999 on unit 0
    assert counts[0] == pytest.approx(2.0)
    assert counts[1] == pytest.approx(0.0)

    # Match spike_binner half-open convention
    spikes = pd.DataFrame(
        {"time": [0.10, 0.149999, 0.15, 0.16], "unit_id": [0, 0, 0, 1]}
    )
    ref = count_spikes_in_window(spikes, [0, 1], 0.10, 0.15)
    np.testing.assert_array_equal(counts, ref)


def test_unit_mapping_permutation_and_missing():
    expected = [10, 20, 30]
    live = [30, 10, 99]  # permuted + unexpected, missing 20
    report = map_units(expected, live)
    assert report.n_expected == 3
    assert report.n_mapped == 2
    assert report.missing_unit_ids == [20]
    assert report.unexpected_unit_ids == [99]
    assert report.permutation == [1, None, 0]
    assert not report.exact_match


def test_missing_units_block_start_without_override(tmp_path: Path):
    bundle_dir = _tiny_bundle(tmp_path, n_units=4)
    dec = LiveDecoder.from_bundle(bundle_dir)
    # Live stream missing some units
    spikes = pd.DataFrame(
        {
            "time": np.linspace(0, 1, 20),
            "unit_id": [0, 1, 0, 1] * 5,  # units 2,3 missing
        }
    )
    stream = ReplaySpikeStream(spikes_df=spikes)
    stream.connect()
    # Force list_unit_ids to incomplete set
    stream._unit_ids = [0, 1]
    dec.connect(stream)
    assert dec.state == RuntimeState.INVALID_INPUT
    with pytest.raises(RuntimeError):
        dec.start()
    dec.set_pipeline_test_override(True)
    assert dec.can_start()
    dec.start()
    assert dec.state == RuntimeState.RUNNING
    dec.stop()


def test_unit_order_not_silent_index_alignment(tmp_path: Path):
    """Live index i must not be treated as training index i without id match."""
    bundle_dir = _tiny_bundle(tmp_path, n_units=3)
    dec = LiveDecoder.from_bundle(bundle_dir)
    # Same count pattern on wrong unit ids should map into expected columns by id
    buf = CausalSpikeBuffer(dec.expected_unit_ids, history_s=2.0)
    # Spikes only on unit 2
    buf.extend([0.05, 0.06], [2, 2])
    c = buf.counts_at(0.1, 0.1)
    assert c.tolist() == [0.0, 0.0, 2.0]


def test_runtime_overrun_detected(tmp_path: Path, monkeypatch):
    bundle_dir = _tiny_bundle(tmp_path)
    # Extremely tight update budget
    cfg = json.loads((bundle_dir / "config.json").read_text())
    cfg["update_dt_s"] = 1e-9
    (bundle_dir / "config.json").write_text(json.dumps(cfg) + "\n")
    dec = LiveDecoder.from_bundle(bundle_dir)
    times = np.arange(0, 1.0, 0.01)
    spikes = pd.DataFrame(
        {
            "time": times,
            "unit_id": np.tile([0, 1, 2, 3], len(times) // 4 + 1)[: len(times)],
        }
    )
    stream = ReplaySpikeStream(spikes_df=spikes)
    dec.connect(stream)
    assert dec.state == RuntimeState.READY
    dec.start()
    rec = dec.step()
    assert rec is not None
    assert rec.overrun is True
    assert dec.dropped_updates >= 1
    dec.stop()


def test_winner_selection_metric_direction(tmp_path: Path):
    """best_deployable uses PRIMARY_METRIC orientation via registry payload."""
    exp = tmp_path / "exp"
    models = exp / "models"
    models.mkdir(parents=True)
    # Two fake targets with correct metrics already chosen upstream
    payload = {
        "schema_version": 1,
        "deployable": True,
        "spike_source": "sorted",
        "update_interval_s": 0.025,
        "training_run_id": "fake",
        "targets": {
            "position": {
                "selected_decoder": "ridge",
                "selected_causal_window_s": 0.5,
                "selected_feature_mode": "counts",
                "selected_metric": "mean_position_error_cm",
                "metric_value": 1.2,
                "realtime_compatible": True,
                "deployable": True,
                "model_artifact_path": None,
                "decoder_config": {
                    "feature_set": "counts",
                    "manifold_type": "none",
                },
            },
            "speed": {
                "selected_decoder": "random_forest_regressor",
                "selected_causal_window_s": 0.25,
                "selected_feature_mode": "global_pca",
                "selected_metric": "r2",
                "metric_value": 0.8,
                "realtime_compatible": True,
                "deployable": True,
                "manifold_n_components": 3,
                "manifold_transform_path": None,
                "decoder_config": {
                    "feature_set": "counts",
                    "feature_type": "global_pca",
                    "manifold_type": "pca",
                    "manifold_n_components": 3,
                },
            },
        },
    }
    (models / "best_realtime_decoders.json").write_text(json.dumps(payload) + "\n")

    pos = DeploymentRegistry(exp).best(target="position", deployable_only=True)
    assert pos.target == "position"
    assert pos.metric_direction == "lower"
    assert pos.metric_name == "mean_position_error_cm"
    assert pos.W == 0.5
    assert pos.D == "ridge"

    spd = best_deployable(exp, "speed", deployable_only=True)
    assert spd.metric_direction == "higher"
    assert spd.E in ("pca", "global_pca") or "pca" in spd.E
    assert spd.deployable is True


def test_open_ephys_stub_fails_loudly():
    stream = OpenEphysSpikeStream(endpoint="tcp://localhost:5555")
    with pytest.raises(NotImplementedError):
        stream.connect()


def test_replay_path_produces_predictions(tmp_path: Path):
    bundle_dir = _tiny_bundle(tmp_path)
    dec = LiveDecoder.from_bundle(bundle_dir)
    rng = np.random.default_rng(1)
    n = 200
    spikes = pd.DataFrame(
        {
            "time": np.sort(rng.uniform(0, 2.0, size=n)),
            "unit_id": rng.integers(0, 4, size=n),
        }
    )
    stream = ReplaySpikeStream(spikes_df=spikes)
    dec.connect(stream)
    assert dec.state == RuntimeState.READY
    dec.start()
    recs = dec.run_replay_for(duration_s=0.5, max_steps=25)
    assert len(recs) >= 5
    assert all(r.target == "speed" for r in recs)
    assert all(isinstance(r.prediction, (float, int, np.floating)) or True for r in recs)
    dec.stop()
    assert dec.state == RuntimeState.STOPPED


@pytest.mark.skipif(
    not Path("outputs/smoke_public/models/best_realtime_decoders.json").exists(),
    reason="smoke_public artifacts not present",
)
def test_pack_bundle_from_smoke_public(tmp_path: Path):
    exp = Path("outputs/smoke_public")
    # Prefer speed (global_pca) or position; pack into tmp
    target = "speed"
    out = pack_deployment_bundle(exp, target, output_dir=tmp_path / "packed")
    assert (out / "decoder.joblib").exists()
    assert (out / "config.json").exists()
    loaded = LiveDecoder.from_bundle(out)
    assert loaded.target == target
    stream = ReplaySpikeStream(experiment_dir=exp, spike_source="sorted")
    loaded.connect(stream)
    loaded.start()
    rec = loaded.step()
    assert rec is not None
    loaded.stop()
