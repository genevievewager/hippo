"""Tests for adaptive W grids and workflow profiles."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from realtime.adaptive_windows import (
    COARSE_DECODE_WINDOWS,
    WINDOW_CANDIDATE_POOL,
    propose_refined_windows,
    windows_from_comparison_dir,
)
from realtime.workflow_profiles import get_profile


def test_propose_refined_windows_adds_neighbors():
    tested = COARSE_DECODE_WINDOWS  # 0.05, 0.25, 0.5, 1.0
    best = (0.25, 0.5)
    extras = propose_refined_windows(tested, best)
    assert 0.1 in extras
    assert 0.05 not in extras
    assert 0.25 not in extras


def test_propose_refined_windows_empty_when_pool_exhausted():
    extras = propose_refined_windows(WINDOW_CANDIDATE_POOL, WINDOW_CANDIDATE_POOL)
    assert extras == ()


def test_windows_from_comparison_dir_inherits_and_flanks(tmp_path: Path):
    src = tmp_path / "sorted"
    src.mkdir()
    pd.DataFrame(
        [
            {
                "target_name": "position",
                "best_decode_window_s": 0.5,
                "recommended_realtime_window_s": 0.25,
                "spike_source": "sorted",
            },
            {
                "target_name": "spatial_context",
                "best_decode_window_s": 0.5,
                "recommended_realtime_window_s": 0.5,
                "spike_source": "sorted",
            },
        ]
    ).to_csv(src / "best_decoder_by_target.csv", index=False)

    windows = windows_from_comparison_dir(tmp_path, ("sorted",))
    assert 0.05 in windows  # short flank
    assert 1.0 in windows  # long flank
    assert 0.25 in windows
    assert 0.5 in windows


def test_standard_profile_keeps_manifold_features():
    prof = get_profile("standard")
    assert "counts" in prof.feature_modes
    assert "global_pca" in prof.feature_modes
    assert "region_pca" in prof.feature_modes
    # Full causal-window grid for deployment selection (not forced to 0.250 s).
    assert prof.decode_windows == WINDOW_CANDIDATE_POOL
    assert prof.adaptive_windows is False
    assert prof.compare_sources is False
    assert prof.enable_temporal_manifold is False
    assert prof.representations == ("pca",)
    assert prof.latent_history_frames == (1, 5, 20)
    assert prof.enable_isomap_distillation is False


def test_manifolds_profile_covers_rt_embeddings():
    from realtime.adaptive_windows import COARSE_DECODE_WINDOWS
    from realtime.manifold_features import (
        MANIFOLDS_FEATURE_MODES,
        MANIFOLDS_ISOMAP_N_NEIGHBORS,
        MANIFOLDS_N_COMPONENTS,
    )

    prof = get_profile("manifolds")
    assert prof.feature_modes == MANIFOLDS_FEATURE_MODES
    assert "region_pca" in prof.feature_modes
    assert "layer_pca" in prof.feature_modes
    assert "global_isomap" in prof.feature_modes
    assert "global_isomap_distilled" in prof.feature_modes
    assert "cell_type_pca" not in prof.feature_modes
    assert prof.decode_windows == COARSE_DECODE_WINDOWS
    assert prof.manifold_n_components == MANIFOLDS_N_COMPONENTS
    assert prof.isomap_n_neighbors == MANIFOLDS_ISOMAP_N_NEIGHBORS
    assert prof.enable_isomap_distillation is True
    assert prof.max_models == "quick"
    assert prof.enable_temporal_manifold is False


def test_run_decoder_default_profile_is_manifolds(monkeypatch):
    import sys

    import run_decoder

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_decoder.py",
            "--input",
            "outputs/ratinabox_005",
            "--output",
            "outputs/ratinabox_005",
        ],
    )
    args = run_decoder.parse_args()
    assert args.profile == "manifolds"


def test_full_profile_dense_temporal():
    prof = get_profile("full")
    assert prof.adaptive_windows is False
    assert "raw" in prof.representations
    assert 2 in prof.latent_history_frames
    assert "shuffled_sequence" in prof.temporal_models


def test_feature_robustness_profile_covers_axes():
    from realtime.decoder_comparison import DEFAULT_DECODE_WINDOWS
    from realtime.neural_features import ALL_FEATURE_SETS
    from realtime.workflow_profiles import FEATURE_ROBUSTNESS_EMBEDDINGS

    prof = get_profile("feature_robustness")
    assert prof.feature_sets == ALL_FEATURE_SETS
    assert prof.embedding_types == FEATURE_ROBUSTNESS_EMBEDDINGS
    assert not hasattr(prof, "degradation_levels")
    assert prof.decode_windows == DEFAULT_DECODE_WINDOWS
    assert prof.run_feature_ablation is True
    assert "identity" in prof.embedding_types
    assert "global_isomap_distilled" in prof.embedding_types


def test_pipeline_timer_summary(tmp_path: Path):
    from realtime.pipeline_timing import PipelineTimer

    timer = PipelineTimer()
    with timer.stage("decoder_comparison", notes="grid"):
        timer.add("comparison_window", 1.5, detail="W=0.25")
        timer.add("comparison_feature_set", 2.0, detail="counts")
    paths = timer.save(tmp_path / "latency_profiling")
    assert paths["summary"].exists()
    summary = __import__("pandas").read_csv(paths["summary"])
    assert "decoder_comparison" in summary["stage"].tolist()
