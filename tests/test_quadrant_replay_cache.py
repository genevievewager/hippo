"""Quadrant replay cache: E required, D must match embedding (never counts)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from realtime.decoder_comparison import windowed_model_path
from realtime.best_decoder_selection import SUITE_TARGETS
from ui.services.realtime import (
    _load_cached_embedding,
    _try_load_decoder_suite,
    list_cached_n_components,
    list_replay_ready_windows,
    run_quadrant_comparison,
)
from ui.services.representations import QUADRANT_ORDER, REALTIME_QUADRANT_DEFAULTS


def test_missing_embedding_cache_fails_loud(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="No cached `global_pca`"):
        _load_cached_embedding(
            tmp_path,
            "sorted",
            "global_pca",
            0.250,
            3,
        )


def test_list_cached_n_components_empty(tmp_path: Path):
    assert list_cached_n_components(tmp_path, "sorted", 0.250) == []


class _FakeE:
    def __init__(self, n_out: int):
        self.actual_n_components_ = int(n_out)


def _ridge_xy(n_features: int) -> Pipeline:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, n_features))
    y = rng.normal(size=(40, 2))
    pipe = Pipeline([("ridge", Ridge(alpha=1.0))])
    pipe.fit(X, y)
    return pipe


def _ridge_1d(n_features: int) -> Pipeline:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, n_features))
    y = rng.normal(size=(40,))
    pipe = Pipeline([("ridge", Ridge(alpha=1.0))])
    pipe.fit(X, y)
    return pipe


def _clf(n_features: int) -> Pipeline:
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, n_features))
    y = np.array(["a", "b"] * 20)
    pipe = Pipeline([
        ("rf", RandomForestClassifier(n_estimators=4, random_state=0)),
    ])
    pipe.fit(X, y)
    return pipe


def _write_suite(models_dir: Path, feature_type: str, n_features: int, k: int = 3) -> None:
    import joblib

    heads = {
        "position": _ridge_xy(n_features),
        "speed": _ridge_1d(n_features),
        "spatial_context": _clf(n_features),
        "movement_state": _clf(n_features),
    }
    for target in SUITE_TARGETS:
        path = windowed_model_path(
            models_dir, target, 0.250, feature_type, k,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(heads[target], path)


def test_counts_suite_is_not_used_for_embedding(tmp_path: Path):
    models = tmp_path / "models"
    _write_suite(models, "counts", n_features=50, k=3)
    suite, primary = _try_load_decoder_suite(
        models,
        embedding_type="global_pca",
        decode_window=0.250,
        n_components=3,
        target="position",
        feature_transformer=_FakeE(3),
    )
    assert suite is None
    assert primary is None


def test_matching_embedding_suite_loads(tmp_path: Path):
    models = tmp_path / "models"
    _write_suite(models, "global_pca", n_features=3, k=3)
    suite, primary = _try_load_decoder_suite(
        models,
        embedding_type="global_pca",
        decode_window=0.250,
        n_components=3,
        target="position",
        feature_transformer=_FakeE(3),
    )
    assert suite is not None
    assert primary is not None
    assert suite.position.n_features_in_ == 3


def test_feature_dim_mismatch_discards_suite(tmp_path: Path):
    models = tmp_path / "models"
    _write_suite(models, "global_pca", n_features=50, k=3)
    suite, primary = _try_load_decoder_suite(
        models,
        embedding_type="global_pca",
        decode_window=0.250,
        n_components=3,
        target="position",
        feature_transformer=_FakeE(3),
    )
    assert suite is None
    assert primary is None


def test_does_not_prefer_counts_when_both_exist(tmp_path: Path):
    models = tmp_path / "models"
    _write_suite(models, "counts", n_features=50, k=3)
    _write_suite(models, "global_pca", n_features=3, k=3)
    suite, primary = _try_load_decoder_suite(
        models,
        embedding_type="global_pca",
        decode_window=0.250,
        n_components=3,
        target="position",
        feature_transformer=_FakeE(3),
    )
    assert suite is not None
    assert suite.position.n_features_in_ == 3
    assert primary.n_features_in_ == 3


def _quadrant_embs() -> tuple[str, ...]:
    return tuple(
        str(REALTIME_QUADRANT_DEFAULTS[q])
        for q in QUADRANT_ORDER
        if REALTIME_QUADRANT_DEFAULTS.get(q)
    )


def _write_e_cache(exp: Path, embedding_type: str, decode_window: float = 0.250, k: int = 3) -> None:
    from realtime.transform_cache import ensure_comparison_root, manifold_transform_path

    root = ensure_comparison_root(exp, spike_source="sorted")
    path = manifold_transform_path(
        root,
        feature_set="counts",
        embedding_type=embedding_type,
        decode_window=float(decode_window),
        n_components=int(k),
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text("{}")


def _write_metrics_stub(exp: Path) -> Path:
    sorted_root = exp / "decoder_comparison" / "sorted"
    sorted_root.mkdir(parents=True, exist_ok=True)
    (sorted_root / "decoder_comparison_metrics.csv").write_text("target_name\nposition\n")
    return sorted_root / "models"


def test_list_replay_ready_windows_empty(tmp_path: Path):
    assert list_replay_ready_windows(tmp_path, "sorted") == []


def test_list_replay_ready_windows_needs_all_three_e_and_d(tmp_path: Path):
    exp = tmp_path / "exp"
    models = _write_metrics_stub(exp)
    embs = _quadrant_embs()
    for emb in embs:
        _write_e_cache(exp, emb)
    assert list_replay_ready_windows(exp, "sorted") == []
    for emb in embs[:-1]:
        _write_suite(models, emb, n_features=3, k=3)
    assert list_replay_ready_windows(exp, "sorted") == []
    _write_suite(models, embs[-1], n_features=3, k=3)
    assert list_replay_ready_windows(exp, "sorted") == [0.250]


def test_run_quadrant_comparison_missing_d_does_not_retrain(tmp_path: Path, monkeypatch):
    called: list[object] = []

    def _should_not_run(*_a, **_k):
        called.append(True)
        raise AssertionError("run_realtime_pipeline must not retrain when D is missing")

    monkeypatch.setattr(
        "realtime.evaluate_realtime.run_realtime_pipeline",
        _should_not_run,
    )
    monkeypatch.setattr(
        "ui.services.realtime._load_cached_embedding",
        lambda *_a, **_k: (_FakeE(3), tmp_path / "e"),
    )
    monkeypatch.setattr(
        "realtime.transform_cache.assert_cached_decode_windows",
        lambda *_a, **_k: None,
    )
    with pytest.raises(FileNotFoundError, match="Decoder Benchmark"):
        run_quadrant_comparison(
            input_dir=tmp_path,
            spike_source="sorted",
            target="position",
            decode_window=0.250,
        )
    assert called == []
