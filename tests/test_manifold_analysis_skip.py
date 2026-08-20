"""Incremental skip / reuse for Manifold Analysis."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ui.services.manifold_analysis import (
    ManifoldAnalysisRequest,
    job_is_covered,
    list_manifold_analysis_runs,
    planned_manifold_fit_jobs,
    run_manifold_analysis,
    selection_already_covered,
    _fit_key,
)


def _write_fake_run(
    experiment: Path,
    run_id: str,
    *,
    feature_sets: list[str],
    manifolds: list[str],
    decode_windows: list[float] | None = None,
    n_components: int = 3,
) -> None:
    out = experiment / "manifolds" / run_id
    out.mkdir(parents=True)
    wins = decode_windows or [0.25]
    results = [
        {
            "feature_set": fs,
            "manifold": m,
            "decode_window_s": w,
            "n_components": n_components,
        }
        for fs in feature_sets
        for m in manifolds
        for w in wins
    ]
    meta = {
        "run_id": run_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "request": {
            "feature_sets": feature_sets,
            "manifolds": manifolds,
            "decode_windows": wins,
            "preferred_window_s": 0.25,
            "n_components": n_components,
            "spike_source": "sorted",
        },
        "results": results,
        "figures": [],
        "n_jobs_run": len(results),
    }
    (out / "analysis_summary.json").write_text(json.dumps(meta) + "\n")


def _touch_geometry(experiment: Path, feature_set: str = "counts", colors=("position", "speed")) -> None:
    from visualization.constants import FIGURE_SUBDIR_DECODER
    from ui.services.manifold_analysis import _geometry_stem

    fig = experiment / "figures" / FIGURE_SUBDIR_DECODER
    fig.mkdir(parents=True, exist_ok=True)
    for c in colors:
        stem = _geometry_stem(c, feature_set)
        (fig / f"{stem}.png").write_bytes(b"png")


def _write_feature_window_cache(experiment: Path, decode_window: float = 0.25) -> None:
    from realtime.transform_cache import window_ms

    d = (
        experiment / "decoder_comparison" / "sorted" / "models" / "feature_transforms"
        / f"counts__counts_w{window_ms(decode_window):04d}ms"
    )
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(
        json.dumps({"decode_window_s": float(decode_window), "source": "test"})
    )


def test_job_is_covered_requires_exact_window_by_default():
    covered_w = {_fit_key("counts", "global_pca", 0.25)}
    covered_fm = {_fit_key("counts", "global_pca")}
    assert job_is_covered(
        "counts", "global_pca", 0.25,
        covered_with_window=covered_w, covered_fs_m=covered_fm,
    )
    # Different window is NOT covered unless fallback is enabled
    assert not job_is_covered(
        "counts", "global_pca", 0.5,
        covered_with_window=covered_w, covered_fs_m=covered_fm,
    )
    assert job_is_covered(
        "counts", "global_pca", 0.5,
        covered_with_window=set(),
        covered_fs_m=covered_fm,
        require_exact_window=False,
    )
    assert not job_is_covered(
        "counts", "region_pca", 0.25,
        covered_with_window=covered_w, covered_fs_m=covered_fm,
    )


def test_planned_jobs_are_full_cartesian(tmp_path: Path):
    jobs = planned_manifold_fit_jobs(
        tmp_path,
        ["counts"],
        ["counts", "global_pca"],
        [0.1, 0.25, 0.5],
    )
    assert len(jobs) == 6  # 2 manifolds × 3 windows


def test_selection_coverage_requires_transform_on_disk(tmp_path: Path):
    wins = [0.025, 0.05, 0.1, 0.25, 0.5]
    _write_fake_run(
        tmp_path,
        "manifold_old",
        feature_sets=["counts"],
        manifolds=["counts", "global_pca"],
        decode_windows=wins,
    )
    from visualization.publication_isomap_plots import COLOR_FEATURES

    _touch_geometry(tmp_path, "counts", colors=tuple(c for c, _ in COLOR_FEATURES))

    runs = list_manifold_analysis_runs(tmp_path)
    cov = selection_already_covered(
        runs,
        feature_sets=["counts"],
        manifolds=["counts", "global_pca"],
        decode_windows=wins,
        spike_source="sorted",
        input_dir=tmp_path,
        require_geometry_pages=True,
    )
    # Figure-only prior runs do not cover the shared transform grid.
    assert cov["n_wanted"] == 10
    assert cov["n_covered"] == 0
    assert cov["fully_covered"] is False
    assert cov["n_to_compute"] == 10


def test_run_skips_when_transform_on_disk(tmp_path: Path):
    from visualization.publication_isomap_plots import COLOR_FEATURES
    from realtime.manifold_features import GlobalPCAManifold
    from realtime.transform_cache import (
        ensure_comparison_root,
        save_manifold_transform_checkpoint,
    )
    import numpy as np

    _write_fake_run(
        tmp_path,
        "manifold_prior",
        feature_sets=["rates"],
        manifolds=["global_pca"],
        decode_windows=[0.25],
    )
    _write_feature_window_cache(tmp_path, 0.25)
    _touch_geometry(tmp_path, "rates", colors=tuple(c for c, _ in COLOR_FEATURES))
    root = ensure_comparison_root(tmp_path, spike_source="sorted")
    X = np.random.default_rng(0).normal(size=(40, 8))
    tf = GlobalPCAManifold(n_components=3, random_state=0)
    tf.fit(X[:28])
    save_manifold_transform_checkpoint(
        tf,
        root,
        feature_set="rates",
        embedding_type="global_pca",
        decode_window=0.25,
        n_components=3,
        extra_meta={"fit_scope": "train_split", "source": "test"},
    )

    calls: list[tuple] = []

    def fake_diag(input_dir, embedding_type, **kwargs):
        calls.append((kwargs.get("feature_set"), embedding_type, kwargs.get("decode_window")))
        raise AssertionError("compute_manifold_diagnostics should not be called when transform exists")

    req = ManifoldAnalysisRequest(
        input_dir=tmp_path,
        feature_sets=("rates",),
        manifolds=("global_pca",),
        decode_windows=(0.25,),
        n_components=3,
        spike_source="sorted",
        force_recompute=False,
        behavioral_colors=tuple(c for c, _ in COLOR_FEATURES[:1]),
    )
    _touch_geometry(tmp_path, "rates", colors=(COLOR_FEATURES[0][0],))

    with patch(
        "ui.services.manifold_analysis.compute_manifold_diagnostics",
        side_effect=fake_diag,
    ), patch(
        "ui.services.manifold_analysis.has_decoder_comparison",
        return_value=False,
    ):
        meta = run_manifold_analysis(req)

    assert meta["n_skipped"] >= 1
    assert meta["n_manifold_fits"] == 0
    assert calls == []


def test_force_recompute_runs_fit(tmp_path: Path):
    from visualization.publication_isomap_plots import COLOR_FEATURES
    import numpy as np
    import pandas as pd

    _write_fake_run(
        tmp_path,
        "manifold_prior",
        feature_sets=["rates"],
        manifolds=["global_pca"],
        decode_windows=[0.25],
    )
    _write_feature_window_cache(tmp_path, 0.25)
    color = COLOR_FEATURES[0][0]
    _touch_geometry(tmp_path, "rates", colors=(color,))

    class _Diag:
        embedding_type = "pca"
        n_components = 3
        latent = np.zeros((10, 3))
        behavior = pd.DataFrame({"time": np.linspace(0, 1, 10), "x": 0.0, "y": 0.0})
        extras = {"n_neural_features": 4, "metadata": {}, "saved_path": "x"}
        from_cache = False

    with patch(
        "ui.services.manifold_analysis.compute_manifold_diagnostics",
        return_value=_Diag(),
    ), patch(
        "ui.services.manifold_analysis.has_decoder_comparison",
        return_value=False,
    ), patch(
        "ui.services.manifold_analysis.plot_latent_geometry_page_from_embeddings",
        return_value=tmp_path / "figures" / "decoder_comparison" / f"fig_latent_geometry_{color}__rates.png",
    ), patch(
        "ui.services.manifold_analysis.register_artifact",
    ):
        req = ManifoldAnalysisRequest(
            input_dir=tmp_path,
            feature_sets=("rates",),
            manifolds=("global_pca",),
            decode_windows=(0.25,),
            force_recompute=True,
            behavioral_colors=(color,),
        )
        meta = run_manifold_analysis(req)

    assert meta["request"]["force_recompute"] is True
    assert meta["n_skipped"] == 0
    assert meta["n_manifold_fits"] == 1


def test_save_manifold_transform_checkpoint(tmp_path: Path):
    from realtime.manifold_features import GlobalPCAManifold
    from realtime.transform_cache import (
        find_manifold_transform,
        read_transform_meta,
        save_manifold_transform_checkpoint,
    )
    import numpy as np

    root = tmp_path / "decoder_comparison" / "sorted"
    (root / "models" / "manifold_transforms").mkdir(parents=True)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8))
    tf = GlobalPCAManifold(n_components=3, random_state=0)
    tf.fit(X[:28])
    path = save_manifold_transform_checkpoint(
        tf,
        root,
        feature_set="counts",
        embedding_type="global_pca",
        decode_window=0.25,
        n_components=3,
        extra_meta={"fit_scope": "train_split", "source": "manifold_explorer"},
    )
    assert path.is_dir()
    assert (path / "meta.json").exists()
    meta = read_transform_meta(path)
    assert meta["fit_scope"] == "train_split"
    assert meta["source"] == "manifold_explorer"
    assert meta["class_name"] == "GlobalPCAManifold"
    hit = find_manifold_transform(
        root,
        feature_set="counts",
        embedding_type="global_pca",
        decode_window=0.25,
        n_components=3,
    )
    assert hit is not None
    assert hit.resolve() == path.resolve()
