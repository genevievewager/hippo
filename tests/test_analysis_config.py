"""Architectural invariants: AnalysisConfig, window ownership, cache identity."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from realtime.pipeline_artifacts import (
    AnalysisConfig,
    DecoderResult,
    FeatureDataset,
    ObservationConfig,
    RepresentationResult,
    analysis_config_hash,
)
from realtime.pipeline_graph import PipelineRun, save_pipeline_run
from realtime.pipeline_invariants import (
    PipelineInvariantError,
    assert_cache_matches_analysis,
    assert_decoder_matches_analysis,
    assert_feature_matches_analysis,
    assert_realtime_compatible,
    assert_representation_matches_features,
    assert_timestamps_align,
    validate_analysis_config,
)
from realtime.representation_registry import (
    QUADRANT_LINEAR_DYNAMIC,
    QUADRANT_LINEAR_STATIC,
    QUADRANT_NONLINEAR_DYNAMIC,
    QUADRANT_NONLINEAR_STATIC,
    get_spec,
)


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


def test_active_window_propagates_to_features_and_representation():
    obs = _obs(window_s=0.500)
    times = np.array([0.50, 1.00, 1.50])
    feats = FeatureDataset(
        observation=obs,
        timestamps=times,
        X=np.ones((3, 4)),
        feature_names=["a", "b", "c", "d"],
    )
    cfg = AnalysisConfig.from_observation(obs, representation="global_pca")
    assert cfg.window_s == 0.500
    assert feats.window_s == 0.500
    assert_feature_matches_analysis(feats, cfg)

    rep = RepresentationResult(
        representation_name="global_pca",
        source_feature_hash=obs.hash(),
        window_s=0.500,
        update_dt=0.050,
        timestamps=times.copy(),
        Z=np.zeros((3, 3)),
    )
    assert_representation_matches_features(rep, feats)
    assert_timestamps_align(feats.timestamps, rep.timestamps)


def test_overlay_cannot_replace_committed_window():
    obs = _obs(window_s=0.250)
    resolved = AnalysisConfig.resolve(
        obs,
        {"window_s": 1.0, "W": 0.025, "representation": "global_lds", "decoder": "ridge"},
    )
    assert resolved.window_s == pytest.approx(0.250)
    assert resolved.representation == "global_lds"


def test_changing_decoder_does_not_change_observation_hash():
    obs = _obs()
    a1 = AnalysisConfig.from_observation(obs, decoder="ridge", representation="global_pca")
    a2 = AnalysisConfig.from_observation(obs, decoder="random_forest", representation="global_pca")
    assert a1.observation.hash() == a2.observation.hash()
    assert a1.hash() != a2.hash()
    with pytest.raises(PipelineInvariantError, match="different configuration"):
        assert_cache_matches_analysis(a1.hash(), a2)


def test_changing_window_changes_observation_and_analysis_hash():
    a1 = AnalysisConfig.from_observation(_obs(window_s=0.250))
    a2 = AnalysisConfig.from_observation(_obs(window_s=0.500))
    assert a1.observation.hash() != a2.observation.hash()
    assert a1.hash() != a2.hash()
    feats = FeatureDataset(observation=_obs(window_s=0.250), timestamps=np.array([0.25]))
    with pytest.raises(PipelineInvariantError):
        assert_feature_matches_analysis(feats, a2)


def test_pipeline_run_set_observation_inherits_into_analysis(tmp_path: Path):
    run = PipelineRun(experiment_dir=tmp_path, simulation_run_id="toy")
    run.set_observation(_obs(window_s=0.250))
    run.set_analysis(AnalysisConfig.from_observation(_obs(window_s=0.250), target="speed"))
    assert run.analysis is not None
    assert run.analysis.target == "speed"
    run.set_observation(_obs(window_s=0.500))
    assert run.analysis.window_s == pytest.approx(0.500)
    assert run.analysis.target == "speed"
    save_pipeline_run(run)
    from realtime.pipeline_graph import load_pipeline_run

    loaded = load_pipeline_run(tmp_path)
    assert loaded is not None
    assert loaded.observation is not None
    assert loaded.observation.window_s == pytest.approx(0.500)
    assert loaded.analysis is not None
    assert loaded.analysis.window_s == pytest.approx(0.500)
    assert loaded.analysis.target == "speed"


def test_old_pipeline_run_json_without_analysis_still_loads(tmp_path: Path):
    run = PipelineRun(experiment_dir=tmp_path, simulation_run_id="toy")
    run.set_observation(_obs())
    payload = run.to_dict()
    payload.pop("analysis", None)
    import json

    (tmp_path / "pipeline_run.json").write_text(json.dumps(payload))
    from realtime.pipeline_graph import load_pipeline_run

    loaded = load_pipeline_run(tmp_path)
    assert loaded is not None
    assert loaded.analysis is None
    active = loaded.active_analysis()
    assert active is not None
    assert active.window_s == pytest.approx(0.250)


def test_representation_metadata_quadrants():
    assert get_spec("global_pca").quadrant == QUADRANT_LINEAR_STATIC
    assert get_spec("global_pca").linearity == "linear"
    assert get_spec("global_pca").temporal_type == "static"
    assert get_spec("diffusion_nystrom").quadrant == QUADRANT_NONLINEAR_STATIC
    assert get_spec("global_lds").quadrant == QUADRANT_LINEAR_DYNAMIC
    assert get_spec("global_lds").temporal_type == "dynamic"
    assert get_spec("gpfa").quadrant == QUADRANT_LINEAR_DYNAMIC
    assert get_spec("gpfa").offline_only is True
    spec_nd = get_spec("lfads")
    assert spec_nd.quadrant == QUADRANT_NONLINEAR_DYNAMIC
    assert spec_nd.implemented is False
    meta = RepresentationResult(
        representation_name="global_isomap",
        source_feature_hash="abc",
    ).to_meta()
    assert meta["quadrant"] == QUADRANT_NONLINEAR_STATIC
    assert meta["offline_only"] is True
    assert meta["realtime_capable"] is False


def test_replay_rejects_offline_only_and_allows_realtime():
    with pytest.raises(PipelineInvariantError, match="offline-only"):
        assert_realtime_compatible("global_isomap", replay=True)
    with pytest.raises(PipelineInvariantError, match="offline-only"):
        assert_realtime_compatible("gpfa", replay=True)
    with pytest.raises(PipelineInvariantError, match="offline-only"):
        assert_realtime_compatible("global_isomap", deployment=True)
    assert_realtime_compatible("global_lds", replay=True)
    assert_realtime_compatible("global_pca", replay=True)
    assert_realtime_compatible("diffusion_nystrom", replay=True)
    assert_realtime_compatible("counts", replay=True)
    cfg = AnalysisConfig.from_observation(
        _obs(), representation="gpfa", realtime=True,
    )
    with pytest.raises(PipelineInvariantError):
        validate_analysis_config(cfg, require_realtime=True)


def test_ui_and_cli_share_core_analysis_functions():
    from realtime.decoder_comparison import run_decoder_comparison
    from realtime.quadrant_experiment import run_controlled_quadrant_experiment
    from ui.services.comparison import run_benchmark
    from ui.services.quadrant_comparison import run_quadrant_comparison_job

    bench_src = inspect.getsource(run_benchmark)
    assert "run_decoder_comparison" in bench_src
    quad_src = inspect.getsource(run_quadrant_comparison_job)
    assert "run_controlled_quadrant_experiment" in quad_src
    assert callable(run_decoder_comparison)
    assert callable(run_controlled_quadrant_experiment)


def test_timestamps_and_sample_counts_align_across_stages():
    times = np.array([0.25, 0.30, 0.35, 0.40])
    obs = _obs()
    feats = FeatureDataset(
        observation=obs,
        timestamps=times,
        X=np.arange(16, dtype=float).reshape(4, 4),
    )
    rep = RepresentationResult(
        representation_name="global_pca",
        source_feature_hash=obs.hash(),
        window_s=obs.window_s,
        timestamps=times,
        Z=np.zeros((4, 3)),
    )
    assert_representation_matches_features(rep, feats)
    assert feats.n_samples == 4
    assert rep.Z is not None and rep.Z.shape[0] == feats.n_samples
    with pytest.raises(PipelineInvariantError, match="length mismatch"):
        assert_timestamps_align(times, times[:-1])
    decoder = DecoderResult(
        target="position",
        decoder_name="ridge",
        observation=obs,
        source_feature_hash=obs.hash(),
        representation_name="global_pca",
    )
    cfg = AnalysisConfig.from_observation(obs, representation="global_pca", decoder="ridge")
    assert_decoder_matches_analysis(decoder, cfg)
    assert decoder.to_meta()["analysis_config_hash"] == analysis_config_hash(
        obs, representation="global_pca", decoder="ridge", target="position",
        train_frac=0.70, seed=42,
    )


def test_dynamic_representation_requires_history_window():
    cfg = AnalysisConfig.from_observation(
        _obs(window_s=0.001, update_dt=0.050),
        representation="global_lds",
    )
    with pytest.raises(PipelineInvariantError, match="causal history window"):
        validate_analysis_config(cfg)


def test_get_active_analysis_config_uses_pipeline_window(monkeypatch, tmp_path: Path):
    from ui import state

    class _FakeSession(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    store = _FakeSession()
    monkeypatch.setattr(state.st, "session_state", store)
    run = PipelineRun(experiment_dir=tmp_path, simulation_run_id="toy")
    run.set_observation(_obs(window_s=0.250, simulation_run_id="toy"))
    save_pipeline_run(run)
    store[state.KEY_ACTIVE_DATASET] = str(tmp_path)
    store[state.KEY_SELECTED_ANALYSIS_CONFIG] = {
        "window_s": 1.0,
        "representation": "diffusion_nystrom",
        "target": "speed",
        "decoder": "ridge",
    }
    cfg = state.get_active_analysis_config()
    assert cfg is not None
    assert cfg.window_s == pytest.approx(0.250)
    assert cfg.representation == "diffusion_nystrom"
    assert cfg.target == "speed"
