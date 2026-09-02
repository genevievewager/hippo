"""Plumbing tests for observation ownership, artifacts, cache, and benchmark plans."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtime.artifact_cache import (
    load_feature_matrix,
    lookup_feature_matrix,
    save_feature_matrix,
)
from realtime.benchmark_plan import (
    count_configurations,
    default_staged_search,
    plan_benchmark,
)
from realtime.observation import ObservationTransformer
from realtime.pipeline_artifacts import (
    ArtifactStatus,
    DecoderResult,
    FeatureDataset,
    ObservationConfig,
    RepresentationResult,
    config_hash,
    migrate_legacy_decoder_entry,
    windows_close,
)
from realtime.pipeline_graph import (
    INVALIDATION,
    STAGE_DECODER,
    STAGE_FEATURES,
    STAGE_REPLAY,
    STAGE_REPRESENTATION,
    PipelineRun,
    commit_observation,
    load_or_infer_pipeline,
)
from realtime.pipeline_invariants import (
    PipelineInvariantError,
    assert_decoder_matches_observation,
    assert_replay_window_matches,
    assert_representation_matches_features,
)
from realtime.train_decoder import causal_train_test_split


def _toy_units() -> pd.DataFrame:
    return pd.DataFrame({
        "unit_id": [0, 1, 2, 3],
        "region": ["CA1", "CA1", "Subiculum", "MEC"],
    })


def _toy_spikes() -> pd.DataFrame:
    rows = []
    for t in (0.01, 0.05, 0.10, 0.20, 0.249, 0.30, 0.40, 0.45):
        rows.append({"time": t, "unit_id": 0})
    for t in (0.02, 0.15, 0.35):
        rows.append({"time": t, "unit_id": 1})
    for t in (0.03, 0.12, 0.22):
        rows.append({"time": t, "unit_id": 2})
    rows.append({"time": 0.08, "unit_id": 3})
    return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def _obs(**kwargs) -> ObservationConfig:
    defaults = dict(
        window_s=0.250,
        update_dt=0.050,
        feature_set="counts",
        feature_type="counts",
        source_spikes="sorted",
        simulation_run_id="toy",
        seed=0,
        train_frac=0.70,
    )
    defaults.update(kwargs)
    return ObservationConfig(**defaults)


def test_changing_w_invalidates_downstream():
    run = PipelineRun(experiment_dir=Path("."), simulation_run_id="toy")
    run.mark_committed(STAGE_FEATURES, config_hash="a", extra={"window_s": 0.25})
    run.mark_committed(STAGE_REPRESENTATION, config_hash="e", source_hash="a")
    run.mark_committed(STAGE_DECODER, config_hash="d", source_hash="e")
    run.mark_committed(STAGE_REPLAY, config_hash="r", source_hash="d")
    for name in (STAGE_REPRESENTATION, STAGE_DECODER, STAGE_REPLAY):
        run.stage(name).status = ArtifactStatus.FRESH

    run.set_observation(_obs(window_s=0.250))
    for name in (STAGE_REPRESENTATION, STAGE_DECODER, STAGE_REPLAY):
        run.stage(name).status = ArtifactStatus.FRESH

    stale = run.set_observation(_obs(window_s=0.500))
    assert STAGE_REPRESENTATION in stale
    assert STAGE_DECODER in stale
    assert STAGE_REPLAY in stale
    assert run.stage(STAGE_REPRESENTATION).status == ArtifactStatus.STALE
    assert run.stage(STAGE_DECODER).status == ArtifactStatus.STALE
    assert run.stage(STAGE_REPLAY).status == ArtifactStatus.STALE
    assert set(INVALIDATION[STAGE_FEATURES]) >= {
        STAGE_REPRESENTATION, STAGE_DECODER, STAGE_REPLAY,
    }


def test_same_observation_does_not_stale():
    run = PipelineRun(experiment_dir=Path("."), simulation_run_id="toy")
    run.set_observation(_obs(window_s=0.250))
    run.mark_committed(STAGE_REPRESENTATION, config_hash="e")
    run.stage(STAGE_REPRESENTATION).status = ArtifactStatus.FRESH
    stale = run.set_observation(_obs(window_s=0.250))
    assert stale == []
    assert run.stage(STAGE_REPRESENTATION).status == ArtifactStatus.FRESH


def test_feature_cache_reused_when_only_decoder_would_change(tmp_path):
    obs = _obs()
    times = np.array([0.25, 0.50, 0.75, 1.00])
    X = np.arange(16, dtype=float).reshape(4, 4)
    names = ["u0", "u1", "u2", "u3"]
    save_feature_matrix(tmp_path, obs, X, times, names)
    lookup = lookup_feature_matrix(tmp_path, obs)
    assert lookup.hit is True
    loaded = load_feature_matrix(tmp_path, obs)
    assert loaded is not None
    X2, t2, n2, _ = loaded
    np.testing.assert_allclose(X2, X)
    np.testing.assert_allclose(t2, times)
    assert n2 == names
    other = lookup_feature_matrix(tmp_path, obs)
    assert other.config_hash == lookup.config_hash


def test_feature_cache_miss_when_window_changes(tmp_path):
    obs = _obs(window_s=0.250)
    times = np.array([0.25, 0.50])
    X = np.ones((2, 4))
    save_feature_matrix(tmp_path, obs, X, times, ["a", "b", "c", "d"])
    miss = lookup_feature_matrix(tmp_path, _obs(window_s=0.500))
    assert miss.hit is False


def test_representation_cache_key_ignores_unsupervised_target():
    from realtime.pipeline_artifacts import representation_cache_key

    k1 = representation_cache_key(
        feature_hash="abc", representation_name="global_pca",
        n_components=3, n_neighbors=None, seed=0, target=None,
    )
    k2 = representation_cache_key(
        feature_hash="abc", representation_name="global_pca",
        n_components=3, n_neighbors=None, seed=0, target="speed",
    )
    k3 = representation_cache_key(
        feature_hash="abc", representation_name="global_pca",
        n_components=3, n_neighbors=None, seed=0, target=None,
    )
    assert k1 == k3
    assert k1 != k2


def test_cache_key_includes_spike_source_and_feature_set(tmp_path):
    times = np.array([0.25])
    X = np.ones((1, 2))
    save_feature_matrix(tmp_path, _obs(source_spikes="sorted"), X, times, ["a", "b"])
    assert lookup_feature_matrix(tmp_path, _obs(source_spikes="ground_truth")).hit is False
    assert lookup_feature_matrix(tmp_path, _obs(feature_set="counts_dynamics")).hit is False
    assert lookup_feature_matrix(tmp_path, _obs(source_spikes="sorted")).hit is True


def test_train_and_realtime_observation_equivalent():
    units = _toy_units()
    spikes = _toy_spikes()
    times = np.array([0.25, 0.50])
    train_mask, _ = causal_train_test_split(times, 0.70)
    obs = _obs(window_s=0.250, update_dt=0.250)
    tr = ObservationTransformer(obs, units_df=units, unit_ids=units["unit_id"].to_numpy())
    ds = tr.fit_transform(spikes, times, train_mask=train_mask)
    assert ds.X is not None
    tr.reset_history()
    online = [tr.transform_one(spikes, float(t)) for t in times]
    online_x = np.vstack(online)
    np.testing.assert_allclose(online_x, ds.X, atol=1e-9)


def test_f_transform_fit_on_train_only():
    units = _toy_units()
    spikes = _toy_spikes()
    times = np.array([0.25, 0.50, 0.75, 1.00])
    train_mask = np.array([True, True, False, False])
    obs = _obs(feature_type="zscore_counts", window_s=0.250, update_dt=0.250)
    tr = ObservationTransformer(obs, units_df=units, unit_ids=units["unit_id"].to_numpy())
    X_raw, _ = tr.extract_raw(spikes, times)
    tr.fit(X_raw, train_mask=train_mask)
    mu = np.mean(X_raw[train_mask], axis=0)
    np.testing.assert_allclose(tr.f_transform.mean_, mu)


def test_replay_rejects_mismatched_window():
    with pytest.raises(PipelineInvariantError, match="250 ms"):
        assert_replay_window_matches(0.250, 0.500, decoder_name="ridge")


def test_decoder_result_rejects_mismatched_observation():
    dec = DecoderResult(
        target="speed",
        decoder_name="ridge",
        observation=_obs(window_s=0.250),
        source_feature_hash="x",
    )
    with pytest.raises(PipelineInvariantError, match="250 ms"):
        assert_decoder_matches_observation(dec, _obs(window_s=1.000))


def test_representation_must_inherit_feature_window():
    feats = FeatureDataset(observation=_obs(window_s=0.250), feature_names=["a"])
    rep = RepresentationResult(
        representation_name="global_pca",
        source_feature_hash=feats.observation.hash(),
        window_s=0.500,
    )
    with pytest.raises(PipelineInvariantError, match="owned by observation"):
        assert_representation_matches_features(rep, feats)


def test_config_hash_deterministic():
    a = _obs(window_s=0.250, seed=1)
    b = _obs(window_s=0.250, seed=1)
    assert a.hash() == b.hash()
    assert a.fit_hash() == b.fit_hash()
    assert config_hash({"x": 1, "y": 2}) == config_hash({"y": 2, "x": 1})
    assert _obs(window_s=0.250).hash() != _obs(window_s=0.500).hash()


def test_consume_action_is_edge_triggered(monkeypatch):
    from ui import state

    class _Fake(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    store = _Fake()
    monkeypatch.setattr(state.st, "session_state", store)
    state.init_session_state()
    assert state.consume_action(state.KEY_FEATURE_ANALYSIS_REQUESTED) is False
    state.request_action(state.KEY_FEATURE_ANALYSIS_REQUESTED)
    assert state.consume_action(state.KEY_FEATURE_ANALYSIS_REQUESTED) is True
    assert state.consume_action(state.KEY_FEATURE_ANALYSIS_REQUESTED) is False


def test_full_benchmark_configuration_count():
    n_pairs, n_cfg = count_configurations(
        targets=("position",),
        windows=(0.05, 0.10, 0.25, 0.50, 1.0),
        feature_sets=("counts",),
        representations=("identity", "global_pca"),
        decoders=("ridge",),
        n_components=(3,),
        max_models="quick",
    )
    assert n_pairs == 2
    assert n_cfg == 10

    plan = plan_benchmark(
        mode="full",
        targets=("position", "speed"),
        windows=(0.25, 0.50),
        feature_sets=("counts",),
        representations=("identity",),
        decoders=("ridge",),
        n_components=(3,),
        max_models="quick",
    )
    assert plan.n_configurations == 4
    assert "window" in plan.swept_axes
    assert "target" in plan.swept_axes
    assert "Benchmark plan" in plan.summary_lines()[0]


def test_quick_plan_inherits_single_window():
    plan = plan_benchmark(
        mode="quick",
        targets=("position", "speed"),
        windows=(0.05, 0.25, 1.0),
        feature_sets=("counts", "counts_dynamics"),
        representations=("identity", "global_pca", "region_pca", "global_lds"),
        decoders=("ridge", "pca_ridge", "logistic_regression"),
        inherited_window_s=0.250,
    )
    assert plan.windows == (0.250,)
    assert len(plan.targets) == 1
    assert len(plan.feature_sets) == 1
    assert len(plan.decoders) <= 2


def test_staged_search_has_four_stages():
    stages = default_staged_search(target="position")
    assert len(stages) == 4
    assert stages[0].name.startswith("stage1")
    assert 0.250 in stages[0].windows


def test_legacy_decoder_entry_migrates_window():
    raw = {
        "selected_decoder": "ridge",
        "selected_causal_window_s": 0.25,
        "selected_feature_mode": "region_pca",
        "spike_source": "sorted",
    }
    migrated = migrate_legacy_decoder_entry(raw)
    assert migrated["window_s"] == 0.25
    assert migrated.get("config_hash")
    assert migrated.get("_migrated") is True
    obs = ObservationConfig.from_dict(migrated)
    assert windows_close(obs.window_s, 0.25)
    assert obs.source_spikes == "sorted"


def test_infer_pipeline_from_existing_experiment(tmp_path):
    exp = tmp_path / "ratinabox_legacy"
    exp.mkdir()
    (exp / "summary.json").write_text(
        '{"seed": 1, "session_duration_s": 10, "n_units": 4}'
    )
    feat = (
        exp / "decoder_comparison" / "sorted" / "models" / "feature_transforms"
        / "counts__counts_w0250ms"
    )
    feat.mkdir(parents=True)
    (feat / "meta.json").write_text(
        '{"feature_set": "counts", "decode_window_s": 0.25, "feature_type_eff": "counts"}'
    )
    run = load_or_infer_pipeline(exp)
    assert run.observation is not None
    assert windows_close(run.observation.window_s, 0.25)
    assert run.stage(STAGE_FEATURES).status in {
        ArtifactStatus.CACHED, ArtifactStatus.FRESH,
    }


def test_commit_observation_persists(tmp_path):
    exp = tmp_path / "run"
    exp.mkdir()
    (exp / "summary.json").write_text('{"seed": 2, "n_units": 1}')
    commit_observation(exp, _obs(simulation_run_id=exp.name))
    again = load_or_infer_pipeline(exp)
    assert again.observation is not None
    assert windows_close(again.observation.window_s, 0.250)
