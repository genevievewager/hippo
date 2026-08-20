"""Tests for Diffusion Maps + Nyström realtime embedding."""

from __future__ import annotations

import numpy as np

from realtime.latency_profiler import (
    LatencySample,
    qualify_latency_values,
    summarize_latency_samples,
)
from realtime.manifold_features import (
    is_realtime_compatible_feature_mode,
    load_feature_transformer,
    make_feature_transformer,
)
from realtime.manifolds.diffusion_nystrom import DiffusionNystrom
from realtime.manifolds.diffusion_nystrom_features import DiffusionNystromManifold
from realtime.manifolds.registry import (
    available_manifolds,
    is_realtime_compatible_manifold,
    make_manifold_encoder,
)
from realtime.search_space import expand_fe_jobs, is_diffusion_nystrom


def _blob_data(n: int = 240, d: int = 12, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(4, d))
    labels = rng.integers(0, 4, size=n)
    X = centers[labels] + 0.15 * rng.normal(size=(n, d))
    X = np.maximum(X, 0.0)
    return X, labels


def test_registered_and_realtime():
    assert "diffusion_nystrom" in available_manifolds()
    assert is_realtime_compatible_manifold("diffusion_nystrom") is True
    assert is_realtime_compatible_feature_mode("diffusion_nystrom") is True
    assert is_diffusion_nystrom("diffusion_nystrom")


def test_fit_transform_one_dim_and_finite():
    X, _ = _blob_data()
    enc = DiffusionNystrom(
        n_landmarks=48, n_components=4, local_scale_k=5, random_state=0,
    )
    enc.fit(X[:160])
    z = enc.transform_one(X[160])
    assert z.shape == (4,)
    assert np.all(np.isfinite(z))
    Z = enc.transform(X[160:])
    assert Z.shape == (80, 4)
    assert np.all(np.isfinite(Z))
    assert enc.latent_dim == 4


def test_batch_matches_transform_one():
    X, _ = _blob_data(n=120, d=8)
    enc = DiffusionNystrom(n_landmarks=32, n_components=3, random_state=1)
    enc.fit(X[:80])
    probe = X[80:90]
    Z_batch = enc.transform(probe)
    Z_one = np.stack([enc.transform_one(row) for row in probe], axis=0)
    np.testing.assert_allclose(Z_batch, Z_one, rtol=1e-4, atol=1e-4)


def test_landmark_seed_reproducible():
    X, _ = _blob_data(seed=2)
    a = DiffusionNystrom(
        n_landmarks=40, landmark_method="random", n_components=3, random_state=7,
    )
    b = DiffusionNystrom(
        n_landmarks=40, landmark_method="random", n_components=3, random_state=7,
    )
    a.fit(X)
    b.fit(X)
    np.testing.assert_allclose(a.landmarks_, b.landmarks_)
    np.testing.assert_allclose(a.transform(X[:20]), b.transform(X[:20]), rtol=1e-5, atol=1e-5)


def test_serialization_roundtrip(tmp_path):
    X, _ = _blob_data()
    enc = DiffusionNystrom(n_landmarks=36, n_components=3, random_state=3)
    enc.fit(X[:150])
    Z = enc.transform(X[150:])
    out = tmp_path / "diff"
    enc.save(out)
    loaded = DiffusionNystrom.load(out)
    Z2 = loaded.transform(X[150:])
    np.testing.assert_allclose(Z, Z2, rtol=1e-5, atol=1e-5)
    z1 = enc.transform_one(X[-1])
    z2 = loaded.transform_one(X[-1])
    np.testing.assert_allclose(z1, z2, rtol=1e-5, atol=1e-5)
    wrapper = DiffusionNystromManifold.load(out)
    np.testing.assert_allclose(wrapper.transform(X[150:]), Z, rtol=1e-5, atol=1e-5)


def test_test_data_not_used_during_fit():
    rng = np.random.default_rng(4)
    X_train = np.maximum(rng.normal(size=(100, 10)), 0.0)
    X_test = np.maximum(rng.normal(size=(40, 10)) + 8.0, 0.0)
    enc = DiffusionNystrom(n_landmarks=20, n_components=3, random_state=0)
    enc.fit(X_train)
    assert enc.scaler_ is not None
    train_mean = enc.scaler_.mean_
    test_raw_mean = X_test.mean(axis=0)
    assert np.linalg.norm(train_mean - test_raw_mean) > 1.0
    Z = enc.transform(X_test)
    assert Z.shape[1] == 3
    assert enc.fit_n_samples_ == 100


def test_query_scale_dense_vs_sparse():
    rng = np.random.default_rng(5)
    dense = rng.normal(loc=0.0, scale=0.2, size=(100, 6))
    X = np.maximum(dense - dense.min(), 0.05)
    enc = DiffusionNystrom(
        n_landmarks=40, n_components=3, local_scale_k=5, random_state=0,
        transform="none", standardize=True,
    )
    enc.fit(X)
    in_dist = enc.query_diagnostics(X[0])
    far = enc.query_diagnostics(X[0] + 25.0)
    assert far["sigma_x"] > in_dist["sigma_x"]
    assert far["nearest_landmark_distance"] > in_dist["nearest_landmark_distance"]


def test_latency_instrumentation_and_qualification():
    X, _ = _blob_data(n=80, d=6)
    enc = DiffusionNystrom(n_landmarks=24, n_components=3, random_state=0)
    enc.fit(X)
    enc.transform_one(X[0])
    lat = enc.last_stage_latencies_ms_
    assert lat is not None
    assert "feature_scaling_ms" in lat
    assert "diffusion_nystrom_transform_ms" in lat
    samples = [
        LatencySample(time_s=i * 0.025, stages_ms={"total_operation": 1.0 + 0.01 * i})
        for i in range(20)
    ]
    summary = summarize_latency_samples(samples, operation_deadline_ms=25.0)
    assert summary["realtime_qualified"] is True
    assert summary["p99_total_ms"] < 25.0
    assert summary["deadline_miss_count"] == 0
    slow = qualify_latency_values([30.0] * 10, deadline_ms=25.0)
    assert slow["realtime_qualified"] is False


def test_replay_never_calls_fit(tmp_path):
    X, _ = _blob_data(n=100, d=8)
    enc = DiffusionNystrom(n_landmarks=20, n_components=3, random_state=0)
    enc.fit(X[:70])
    enc.save(tmp_path / "m")
    loaded = DiffusionNystrom.load(tmp_path / "m")

    def _boom(*_a, **_k):
        raise AssertionError("fit() called")

    loaded.fit = _boom
    Z = loaded.transform(X[70:])
    z = loaded.transform_one(X[-1])
    assert Z.shape[1] == 3
    assert z.shape == (3,)


def test_make_feature_transformer_and_wrapper(tmp_path):
    X, _ = _blob_data()
    tr = make_feature_transformer(
        "diffusion_nystrom",
        decode_window=0.25,
        n_components=4,
        n_landmarks=32,
        random_state=0,
    )
    assert isinstance(tr, DiffusionNystromManifold)
    tr.fit(X[:100])
    Z = tr.transform(X[100:110])
    assert Z.shape == (10, 4)
    tr.save(tmp_path / "wrap")
    loaded = load_feature_transformer(tmp_path / "wrap")
    np.testing.assert_allclose(loaded.transform(X[100:110]), Z, rtol=1e-5, atol=1e-5)


def test_expand_jobs_uses_n_landmarks_not_isomap_nn():
    jobs = expand_fe_jobs(
        feature_modes=("diffusion_nystrom",),
        manifold_n_components=(3, 8),
        isomap_n_neighbors=(10, 30),
        n_landmarks=(128, 512),
    )
    pairs = {(k, nl) for _f, e, k, nl in jobs if e == "diffusion_nystrom"}
    assert pairs == {(3, 128), (3, 512), (8, 128), (8, 512)}


def test_landmark_benchmark_table(tmp_path):
    from realtime.diffusion_landmark_benchmark import run_diffusion_landmark_benchmark

    X, _ = _blob_data(n=120, d=8)
    y = np.stack([np.linspace(0, 1, 120), np.linspace(1, 0, 120)], axis=1)
    df = run_diffusion_landmark_benchmark(
        X[:80], X[80:],
        y_train=y[:80], y_test=y[80:],
        landmark_counts=(16, 32),
        n_components=3,
        n_latency_repeats=8,
        output_dir=tmp_path,
        random_state=0,
    )
    assert set(df["n_landmarks"]) <= {16, 32}
    assert "realtime_qualified" in df.columns
    assert "p99_embedding_latency_ms" in df.columns
    assert (tmp_path / "diffusion_landmark_benchmark.csv").exists()


def test_factory_registry():
    enc = make_manifold_encoder(
        "diffusion_nystrom", n_landmarks=16, n_components=2, random_state=0,
    )
    assert isinstance(enc, DiffusionNystrom)
    rng = np.random.default_rng(0)
    X = np.maximum(rng.normal(size=(60, 5)), 0.0)
    enc.fit(X)
    assert enc.transform_one(X[0]).shape == (2,)


def test_realtime_decoder_uses_transform_one_not_fit():
    from realtime.realtime_decoder import RealTimeDecoder
    from realtime.train_decoder import TrainedDecoders

    class Dummy:
        classes_ = np.array(["a", "b"])

        def predict(self, X):
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            return np.zeros((len(X), 2))

        def predict_proba(self, X):
            return np.array([[1.0, 0.0]])

    X, _ = _blob_data(n=80, d=6)
    enc = DiffusionNystrom(n_landmarks=16, n_components=2, random_state=0)
    enc.fit(X)

    def _boom(*_a, **_k):
        raise AssertionError("fit() during replay")

    enc.fit = _boom
    models = TrainedDecoders(
        position=Dummy(),
        speed=Dummy(),
        spatial_context=Dummy(),
        movement_state=Dummy(),
        spatial_context_classes=["a", "b"],
        movement_state_classes=["a", "b"],
    )
    rt = RealTimeDecoder(models=models, unit_ids=np.arange(6), feature_transformer=enc)
    z = rt._apply_embedding(X[0:1])
    assert z.shape[1] == 2
