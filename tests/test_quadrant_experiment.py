"""Tests for quadrant metadata, fair controlled comparison, and compatibility."""

from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtime.benchmark_plan import BENCHMARK_MODES, plan_benchmark
from realtime.decoder_comparison import ComparisonRunConfig, comparison_targets
from realtime.manifold_features import make_feature_transformer
from realtime.pipeline_artifacts import ObservationConfig
from realtime.quadrant_experiment import (
    QuadrantComparisonError,
    QuadrantExperimentConfig,
    annotate_metrics_dataframe,
    assert_metrics_share_observation,
    assert_result_matches_observation,
    default_decoder_for_target,
    load_legacy_metrics_with_quadrants,
    to_comparison_config,
    validate_controlled_quadrant_comparison,
)
from realtime.representation_registry import (
    DEFAULT_QUADRANT_METHODS,
    QUADRANT_LINEAR_DYNAMIC,
    QUADRANT_LINEAR_STATIC,
    QUADRANT_NONLINEAR_DYNAMIC,
    QUADRANT_NONLINEAR_STATIC,
    QUADRANT_UNKNOWN,
    canonicalize_quadrant,
    embedding_metadata_columns,
    get_spec,
    infer_quadrant,
    methods_for_quadrant,
    nonlinear_dynamic_available,
    representation_fit_dependencies,
    resolve_representation_name,
)
from ui.services.representations import (
    REALTIME_QUADRANT_DEFAULTS,
    REPRESENTATION_QUADRANTS,
    selectable_methods_for_quadrant,
)


def _obs(**kwargs) -> ObservationConfig:
    defaults = dict(
        window_s=0.250,
        update_dt=0.050,
        feature_set="counts",
        feature_type="counts",
        source_spikes="sorted",
        simulation_run_id="toy",
        seed=42,
        train_frac=0.70,
    )
    defaults.update(kwargs)
    return ObservationConfig(**defaults)


def _cfg(tmp_path: Path, **kwargs) -> QuadrantExperimentConfig:
    obs = kwargs.pop("observation", _obs())
    methods = kwargs.pop("methods", dict(DEFAULT_QUADRANT_METHODS))
    return QuadrantExperimentConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "quadrant_comparison" / "sorted",
        observation=obs,
        methods=methods,
        **kwargs,
    )


def test_existing_pca_implementation_still_produces_results():
    rng = np.random.default_rng(0)
    X = np.abs(rng.normal(size=(80, 12)))
    tr = make_feature_transformer("global_pca", decode_window=0.250, n_components=3)
    assert tr is not None
    tr.fit(X)
    Z = tr.transform(X)
    assert Z.shape == (80, 3)
    assert np.all(np.isfinite(Z))


def test_registry_does_not_replace_computation():
    tr = make_feature_transformer("global_pca", decode_window=0.25, n_components=2)
    assert tr.__class__.__name__ in {"GlobalPCAManifold", "PCAManifoldEncoder"}
    spec = get_spec("global_pca")
    assert spec.quadrant == QUADRANT_LINEAR_STATIC
    assert spec.representation_name == "global_pca"


def test_existing_decoder_benchmark_config_still_constructs():
    cfg = ComparisonRunConfig(input_dir=".", output_dir=".")
    targets = comparison_targets(cfg)
    assert "position" in targets
    assert "spatial_context" in targets
    plan = plan_benchmark(
        mode="quick",
        targets=("position",),
        windows=(0.250,),
        feature_sets=("counts",),
        representations=("global_pca",),
        decoders=("ridge",),
        n_components=(3,),
        inherited_window_s=0.250,
    )
    assert plan.mode == "quick"
    assert "full" in BENCHMARK_MODES
    assert "targeted" in BENCHMARK_MODES
    assert cfg.compute_latent_stability is True
    assert cfg.write_dynamic_latent_figures is True
    assert cfg.compute_dynamic_latent_extras is True


def test_cli_entry_points_still_importable():
    import run_decoder
    import run_decoder_comparison
    import run_realtime_decoding
    import run_simulation

    assert callable(run_decoder.parse_args)
    assert callable(run_decoder_comparison.parse_args)
    assert callable(run_realtime_decoding.parse_args)
    assert callable(run_simulation.parse_args)


def test_method_to_quadrant_mapping():
    assert infer_quadrant("global_pca") == QUADRANT_LINEAR_STATIC
    assert infer_quadrant("region_pca") == QUADRANT_LINEAR_STATIC
    assert infer_quadrant("layer_pca") == QUADRANT_LINEAR_STATIC
    assert infer_quadrant("counts") == QUADRANT_LINEAR_STATIC
    assert resolve_representation_name("counts") == "identity"
    assert infer_quadrant("diffusion_nystrom") == QUADRANT_NONLINEAR_STATIC
    assert infer_quadrant("global_isomap") == QUADRANT_NONLINEAR_STATIC
    assert infer_quadrant("global_lds") == QUADRANT_LINEAR_DYNAMIC
    assert infer_quadrant("gpfa") == QUADRANT_LINEAR_DYNAMIC
    gpfa = get_spec("gpfa")
    assert gpfa.offline_only is True
    assert gpfa.realtime_capable is False
    assert gpfa.linearity == "linear"
    assert infer_quadrant("definitely_not_a_method") == QUADRANT_UNKNOWN
    assert canonicalize_quadrant("static_linear") == QUADRANT_LINEAR_STATIC


def test_pls_and_bayesian_are_documented_not_defaults():
    pls = get_spec("pls")
    assert pls.scientifically_ambiguous
    assert pls.supervised
    assert DEFAULT_QUADRANT_METHODS[QUADRANT_LINEAR_STATIC] != "pls"
    bayes = get_spec("bayesian_place_tuning")
    assert bayes.quadrant == QUADRANT_UNKNOWN
    assert bayes.scientifically_ambiguous


def test_nonlinear_dynamic_unavailable():
    assert nonlinear_dynamic_available() is False
    assert methods_for_quadrant(QUADRANT_NONLINEAR_DYNAMIC) == ()
    assert DEFAULT_QUADRANT_METHODS[QUADRANT_NONLINEAR_DYNAMIC] is None


def test_public_ui_quadrants_unchanged():
    assert REPRESENTATION_QUADRANTS["static_linear"] == (
        "counts", "global_pca", "region_pca",
    )
    assert "layer_pca" not in REPRESENTATION_QUADRANTS["static_linear"]
    public = [m for methods in REPRESENTATION_QUADRANTS.values() for m in methods]
    assert "layer_pca" not in public
    assert REALTIME_QUADRANT_DEFAULTS["dynamic_nonlinear"] is None
    assert "layer_pca" not in selectable_methods_for_quadrant(
        QUADRANT_LINEAR_STATIC, advanced=False,
    )
    assert "layer_pca" in selectable_methods_for_quadrant(
        QUADRANT_LINEAR_STATIC, advanced=True,
    )


def test_controlled_config_shares_one_feature_dataset(tmp_path: Path):
    cfg = _cfg(tmp_path)
    validate_controlled_quadrant_comparison(cfg)
    cmp = to_comparison_config(cfg)
    assert cmp.decode_windows == (0.250,)
    assert cmp.feature_sets == ("counts",)
    assert cmp.update_dt == 0.050
    assert cmp.targets == ("position",)
    assert cmp.decoder_names == ("ridge",)
    assert cmp.train_frac == 0.70
    assert cmp.seed == 42
    assert cmp.reuse_transforms is True
    assert cmp.compute_latent_stability is False
    assert cmp.write_dynamic_latent_figures is False
    assert cmp.compute_dynamic_latent_extras is False
    assert cmp.n_jobs == 1
    embeddings = set(cmp.embedding_types)
    assert "global_pca" in embeddings
    assert "diffusion_nystrom" in embeddings
    assert "global_lds" in embeddings
    assert "lfads" not in embeddings
    assert cfg.observation.hash() == _obs().hash()


def test_same_train_test_identity_across_branches(tmp_path: Path):
    cfg = _cfg(tmp_path)
    cmp = to_comparison_config(cfg)
    assert len(cmp.decode_windows) == 1
    assert len(cmp.feature_sets) == 1
    assert len(cmp.targets) == 1
    assert len(cmp.decoder_names) == 1
    keys = cfg.representation_keys()
    hashes = {k["observation_hash"] for k in keys}
    assert hashes == {cfg.observation.hash()}
    assert {k["seed"] for k in keys} == {42}
    assert {k["train_frac"] for k in keys} == {0.70}


def test_changing_representation_does_not_change_observation_hash(tmp_path: Path):
    a = _cfg(tmp_path, methods={
        QUADRANT_LINEAR_STATIC: "global_pca",
        QUADRANT_NONLINEAR_STATIC: "diffusion_nystrom",
        QUADRANT_LINEAR_DYNAMIC: "global_lds",
        QUADRANT_NONLINEAR_DYNAMIC: None,
    })
    b = _cfg(tmp_path, methods={
        QUADRANT_LINEAR_STATIC: "region_pca",
        QUADRANT_NONLINEAR_STATIC: "global_isomap",
        QUADRANT_LINEAR_DYNAMIC: "gpfa",
        QUADRANT_NONLINEAR_DYNAMIC: None,
    })
    assert a.observation.hash() == b.observation.hash()


def test_changing_decoder_does_not_change_representation_keys(tmp_path: Path):
    ridge = _cfg(tmp_path, decoder_name="ridge")
    rf = _cfg(tmp_path, decoder_name="random_forest_regressor")
    assert ridge.observation.hash() == rf.observation.hash()
    assert ridge.representation_keys() == rf.representation_keys()
    deps = representation_fit_dependencies(
        observation_hash=ridge.observation.hash(),
        embedding_type="global_pca",
        n_components=3,
        seed=42,
        train_frac=0.70,
    )
    assert "decoder" not in deps
    assert "decoder_name" not in deps


def test_mismatched_window_is_rejected():
    obs = _obs(window_s=0.250)
    with pytest.raises(QuadrantComparisonError, match="500 ms"):
        assert_result_matches_observation(
            {"window_s": 0.500, "decode_window_s": 0.500, "feature_set": "counts"},
            obs,
        )
    rows = pd.DataFrame([
        {
            "decode_window_s": 0.250,
            "window_s": 0.250,
            "update_dt_s": 0.050,
            "feature_set": "counts",
            "spike_source": "sorted",
            "target_name": "position",
            "decoder_name": "ridge",
            "embedding_type": "global_pca",
        },
        {
            "decode_window_s": 0.500,
            "window_s": 0.500,
            "update_dt_s": 0.050,
            "feature_set": "counts",
            "spike_source": "sorted",
            "target_name": "position",
            "decoder_name": "ridge",
            "embedding_type": "diffusion_nystrom",
        },
    ])
    with pytest.raises(QuadrantComparisonError, match="500 ms|mix incompatible windows"):
        assert_metrics_share_observation(
            rows, obs, target="position", decoder_name="ridge",
        )


def test_wrong_quadrant_method_is_rejected(tmp_path: Path):
    cfg = _cfg(tmp_path, methods={
        QUADRANT_LINEAR_STATIC: "diffusion_nystrom",
        QUADRANT_NONLINEAR_STATIC: "global_pca",
        QUADRANT_LINEAR_DYNAMIC: "global_lds",
        QUADRANT_NONLINEAR_DYNAMIC: None,
    })
    with pytest.raises(QuadrantComparisonError, match="belongs to"):
        validate_controlled_quadrant_comparison(cfg)


def test_unimplemented_method_is_rejected(tmp_path: Path):
    cfg = _cfg(tmp_path, methods={
        **DEFAULT_QUADRANT_METHODS,
        QUADRANT_NONLINEAR_DYNAMIC: "lfads",
    })
    with pytest.raises(QuadrantComparisonError, match="not implemented"):
        validate_controlled_quadrant_comparison(cfg)


def test_empty_nonlinear_dynamic_is_allowed(tmp_path: Path):
    cfg = _cfg(tmp_path)
    validate_controlled_quadrant_comparison(cfg)
    assert QUADRANT_NONLINEAR_DYNAMIC not in {
        infer_quadrant(e) for e in cfg.selected_embeddings()
    }


def test_legacy_metrics_without_quadrant_still_load(tmp_path: Path):
    csv = tmp_path / "decoder_comparison_metrics.csv"
    pd.DataFrame([
        {
            "embedding_type": "global_pca",
            "decoder_name": "ridge",
            "target_name": "position",
            "decode_window_s": 0.250,
            "r2": 0.1,
        },
        {
            "embedding_type": "mystery_embedding",
            "decoder_name": "ridge",
            "target_name": "position",
            "decode_window_s": 0.250,
            "r2": 0.0,
        },
    ]).to_csv(csv, index=False)
    loaded = load_legacy_metrics_with_quadrants(csv)
    assert loaded.loc[0, "quadrant"] == QUADRANT_LINEAR_STATIC
    assert loaded.loc[1, "quadrant"] == QUADRANT_UNKNOWN
    original = pd.read_csv(csv)
    assert "quadrant" not in original.columns


def test_annotate_metrics_preserves_method_rows():
    df = pd.DataFrame([
        {"embedding_type": "global_pca", "decoder_name": "ridge"},
        {"embedding_type": "region_pca", "decoder_name": "ridge"},
    ])
    out = annotate_metrics_dataframe(df)
    assert len(out) == 2
    assert set(out["quadrant"]) == {QUADRANT_LINEAR_STATIC}


def test_streamlit_rerun_does_not_auto_start_expensive_work():
    from ui.views import quadrant_comparison
    from ui.views import decoder_benchmark

    quad_src = inspect.getsource(quadrant_comparison)
    assert "st.button" in quad_src
    assert "request_action" in quad_src
    assert "consume_action" in quad_src
    assert "KEY_QUADRANT_EXPERIMENT_REQUESTED" in quad_src
    bench_src = inspect.getsource(decoder_benchmark)
    assert "request_action" in bench_src


def test_advanced_mode_exposes_broader_methods():
    default = selectable_methods_for_quadrant(QUADRANT_LINEAR_STATIC, advanced=False)
    advanced = selectable_methods_for_quadrant(QUADRANT_LINEAR_STATIC, advanced=True)
    assert "global_pca" in default
    assert "counts" in default
    assert "layer_pca" in advanced
    assert set(default).issubset(set(advanced))
    nl = selectable_methods_for_quadrant(QUADRANT_NONLINEAR_STATIC, advanced=True)
    assert "diffusion_nystrom" in nl
    assert "global_isomap" in nl


def test_decoder_comparison_base_row_includes_quadrant():
    meta = embedding_metadata_columns("diffusion_nystrom")
    assert meta["quadrant"] == QUADRANT_NONLINEAR_STATIC
    assert meta["linearity"] == "nonlinear"
    assert meta["temporal_type"] == "static"


def test_default_decoder_by_target():
    assert default_decoder_for_target("position") == "ridge"
    assert default_decoder_for_target("spatial_context") == "logistic_regression"


def test_window_sweep_keeps_single_w_in_controlled_adapter(tmp_path: Path):
    cfg = _cfg(tmp_path, window_sweep_s=(0.050, 0.500))
    validate_controlled_quadrant_comparison(cfg)
    cmp = to_comparison_config(replace(cfg, window_sweep_s=()))
    assert cmp.decode_windows == (0.250,)
