"""Unit and integration tests for dynamic latent-state models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtime.dynamic_latents.adapters import DynamicLatentEmbedding
from realtime.dynamic_latents.behavioral_association import associate_latent_with_behavior
from realtime.dynamic_latents.gpfa import GPFAModel
from realtime.dynamic_latents.lds import LinearDynamicalSystem
from realtime.dynamic_latents.metrics import compute_dynamic_latent_metrics
from realtime.dynamic_latents.registry import (
    DYNAMIC_LATENT_REGISTRY,
    is_dynamic_latent,
    is_realtime_compatible_dynamic,
    make_dynamic_latent,
)
from realtime.manifold_features import (
    OFFLINE_ONLY_FEATURE_MODES,
    is_realtime_compatible_feature_mode,
    load_feature_transformer,
    make_feature_transformer,
)
from realtime.search_space import (
    ALL_EMBEDDING_TYPES,
    expand_fe_jobs,
    is_dynamic_embedding,
    representation_family,
)


def _synthetic_lds_data(T: int = 200, n: int = 12, k: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    A = 0.9 * np.eye(k) + 0.05 * rng.normal(size=(k, k))
    C = rng.normal(size=(n, k))
    d = rng.normal(size=n)
    z = np.zeros((T, k))
    x = np.zeros((T, n))
    z[0] = rng.normal(size=k)
    for t in range(T):
        if t > 0:
            z[t] = A @ z[t - 1] + 0.1 * rng.normal(size=k)
        x[t] = C @ z[t] + d + 0.2 * rng.normal(size=n)
    return x, z, A, C


def test_registry_contains_lds_and_gpfa():
    assert "global_lds" in DYNAMIC_LATENT_REGISTRY
    assert "gpfa" in DYNAMIC_LATENT_REGISTRY
    assert is_dynamic_latent("global_lds")
    assert is_realtime_compatible_dynamic("global_lds")
    assert not is_realtime_compatible_dynamic("gpfa")
    assert "global_lds" in ALL_EMBEDDING_TYPES
    assert "gpfa" in ALL_EMBEDDING_TYPES


def test_lds_fit_transform_shapes():
    X, Z_true, *_ = _synthetic_lds_data()
    model = LinearDynamicalSystem(n_components=3, n_em_iters=5, random_state=0)
    model.fit(X[:140])
    Z = model.transform(X[140:], causal=True, reset=True)
    assert Z.shape == (X[140:].shape[0], 3)


def test_lds_step_and_reset():
    X, *_ = _synthetic_lds_data(T=80)
    model = LinearDynamicalSystem(n_components=2, n_em_iters=3, random_state=1).fit(X[:50])
    model.reset_state()
    z0 = model.step(X[50])
    z1 = model.step(X[51])
    assert z0.shape == (2,)
    assert z1.shape == (2,)
    model.reset_state()
    z0b = model.step(X[50])
    np.testing.assert_allclose(z0, z0b)


def test_lds_causal_no_future_leakage():
    """Filtering x[:t+1] then taking last state equals sequential step up to t."""
    X, *_ = _synthetic_lds_data(T=60, n=8, k=2, seed=2)
    model = LinearDynamicalSystem(n_components=2, n_em_iters=4, random_state=2).fit(X[:40])
    Z_batch = model.transform(X[40:55], causal=True, reset=True)
    model.reset_state()
    Z_step = []
    for t in range(40, 55):
        Z_step.append(model.step(X[t]))
    Z_step = np.asarray(Z_step)
    np.testing.assert_allclose(Z_batch, Z_step, rtol=1e-5, atol=1e-5)

    # Extending the batch by one future frame must not change earlier latents.
    Z_ext = model.transform(X[40:56], causal=True, reset=True)
    np.testing.assert_allclose(Z_ext[:-1], Z_batch, rtol=1e-5, atol=1e-5)


def test_lds_save_load(tmp_path: Path):
    X, *_ = _synthetic_lds_data(T=100, seed=3)
    model = LinearDynamicalSystem(n_components=3, n_em_iters=4, random_state=3).fit(X[:70])
    Z1 = model.transform(X[70:], causal=True, reset=True)
    model.save(tmp_path / "lds")
    loaded = LinearDynamicalSystem.load(tmp_path / "lds")
    Z2 = loaded.transform(X[70:], causal=True, reset=True)
    np.testing.assert_allclose(Z1, Z2, rtol=1e-5, atol=1e-5)


def test_gpfa_offline_only():
    X, *_ = _synthetic_lds_data(T=120, seed=4)
    model = GPFAModel(n_components=3, max_iter=3, random_state=4).fit(X)
    Z = model.transform(X, causal=False)
    assert Z.shape[1] == 3
    assert model.supports_realtime is False
    with pytest.raises(NotImplementedError):
        model.step(X[0])
    assert "gpfa" in OFFLINE_ONLY_FEATURE_MODES
    assert not is_realtime_compatible_feature_mode("gpfa")


def test_adapter_plugs_into_make_feature_transformer(tmp_path: Path):
    X, *_ = _synthetic_lds_data(T=90, seed=5)
    tr = make_feature_transformer("global_lds", decode_window=0.25, n_components=3, update_dt=0.025)
    tr.fit(X[:60])
    Z = tr.transform(X[60:], causal=True, reset=True)
    assert Z.shape == (30, 3)
    tr.save(tmp_path / "emb")
    loaded = load_feature_transformer(tmp_path / "emb")
    assert isinstance(loaded, DynamicLatentEmbedding)
    Z2 = loaded.transform(X[60:], causal=True, reset=True)
    np.testing.assert_allclose(Z, Z2, rtol=1e-5, atol=1e-5)


def test_expand_fe_jobs_includes_dynamic_dims():
    jobs = expand_fe_jobs(
        feature_types=("counts",),
        embedding_types=("global_lds", "gpfa"),
        manifold_n_components=(3, 5),
        use_fe_grid=True,
    )
    assert len(jobs) == 4
    assert all(is_dynamic_embedding(e) for _, e, _, _ in jobs)
    assert representation_family("global_lds") == "dynamic"
    assert representation_family("global_pca") == "static"


def test_dynamic_metrics_and_association():
    X, *_ = _synthetic_lds_data(T=100, seed=6)
    model = LinearDynamicalSystem(n_components=3, n_em_iters=3, random_state=6).fit(X)
    Z = model.transform(X, causal=True, reset=True)
    X_hat = model.reconstruct(Z)
    metrics = compute_dynamic_latent_metrics(
        X=X, Z_causal=Z, X_hat=X_hat, A=model.A_, train_loglik=model.train_loglik_,
    )
    assert "observation_reconstruction_mse" in metrics
    assert "one_step_latent_prediction_mse" in metrics

    beh = pd.DataFrame({
        "x": np.linspace(0, 1, len(Z)),
        "y": np.linspace(1, 0, len(Z)),
        "speed": np.abs(np.gradient(np.linspace(0, 1, len(Z)))),
        "spatial_context": np.where(np.arange(len(Z)) > 50, "wall", "center"),
        "movement_state": np.where(np.arange(len(Z)) % 2 == 0, "fast", "slow"),
    })
    assoc = associate_latent_with_behavior(Z, beh, representation="global_lds")
    assert not assoc.empty
    assert set(assoc.columns) >= {
        "latent_representation", "latent_dimension", "behavioral_variable", "association_score",
    }


def test_make_dynamic_latent_factory():
    m = make_dynamic_latent("global_lds", n_components=2)
    assert isinstance(m, LinearDynamicalSystem)


def test_realtime_decoder_uses_step(tmp_path: Path):
    """Sequential realtime path updates state rather than refitting."""
    from realtime.realtime_decoder import RealTimeDecoder

    X, *_ = _synthetic_lds_data(T=80, n=6, k=2, seed=7)
    emb = DynamicLatentEmbedding(model_name="global_lds", n_components=2, random_state=7)
    emb.fit(X[:50])

    class _Models:
        pass

    rt = RealTimeDecoder(
        models=_Models(),  # type: ignore[arg-type]
        unit_ids=list(range(6)),
        decode_window=0.25,
        update_dt=0.025,
        feature_transformer=emb,
    )
    assert rt._is_dynamic

    # Drive step updates directly through the embedding application path.
    emb.reset_state()
    z_hist = []
    for t in range(50, 60):
        z = rt._apply_embedding(X[t].reshape(1, -1))
        z_hist.append(z.ravel().copy())
    z_hist = np.asarray(z_hist)
    assert z_hist.shape == (10, 2)
    # State must change across steps (not a frozen/recomputed-from-scratch identity).
    assert np.linalg.norm(z_hist[-1] - z_hist[0]) > 1e-8

    # reset_state returns inference to the same initial filtered state for same x0.
    emb.reset_state()
    z_a = rt._apply_embedding(X[50].reshape(1, -1)).ravel()
    emb.reset_state()
    z_b = rt._apply_embedding(X[50].reshape(1, -1)).ravel()
    np.testing.assert_allclose(z_a, z_b)
