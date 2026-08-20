"""Static-decoding wrapper for ``diffusion_nystrom`` (realtime Nyström).

Delegates to :class:`~realtime.manifolds.diffusion_nystrom.DiffusionNystrom`.
Tagged ``realtime_compatible=True`` — online path is query-to-landmark kernel
plus a precomputed projection; ``fit`` must not run during replay/live.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from realtime.manifolds.diffusion_nystrom import (
    DEFAULT_ALPHA,
    DEFAULT_DIFFUSION_TIME,
    DEFAULT_EPS,
    DEFAULT_LANDMARK_METHOD,
    DEFAULT_LOCAL_SCALE_K,
    DEFAULT_N_COMPONENTS,
    DEFAULT_N_LANDMARKS,
    DiffusionNystrom,
)


class DiffusionNystromManifold(BaseEstimator, TransformerMixin):
    """Sklearn-style wrapper used by decoder comparison / feature factory."""

    realtime_compatible = True
    deployment_tag = "realtime_nystrom"

    def __init__(
        self,
        n_components: int = DEFAULT_N_COMPONENTS,
        n_landmarks: int = DEFAULT_N_LANDMARKS,
        *,
        landmark_method: str = DEFAULT_LANDMARK_METHOD,
        local_scale_k: int = DEFAULT_LOCAL_SCALE_K,
        alpha: float = DEFAULT_ALPHA,
        diffusion_time: int | float = DEFAULT_DIFFUSION_TIME,
        eps: float = DEFAULT_EPS,
        transform: str = "sqrt_counts",
        standardize: bool = True,
        random_state: int = 42,
        dtype: str = "float32",
    ):
        self.n_components = int(n_components)
        self.n_landmarks = int(n_landmarks)
        self.landmark_method = landmark_method
        self.local_scale_k = int(local_scale_k)
        self.alpha = float(alpha)
        self.diffusion_time = float(diffusion_time)
        self.eps = float(eps)
        self.transform_name = transform
        self.standardize = bool(standardize)
        self.random_state = int(random_state)
        self.dtype = dtype
        self._encoder: DiffusionNystrom | None = None

    def _make_encoder(self) -> DiffusionNystrom:
        return DiffusionNystrom(
            n_landmarks=self.n_landmarks,
            landmark_method=self.landmark_method,
            n_components=self.n_components,
            local_scale_k=self.local_scale_k,
            alpha=self.alpha,
            diffusion_time=self.diffusion_time,
            eps=self.eps,
            random_state=self.random_state,
            transform=self.transform_name,
            standardize=self.standardize,
            dtype=self.dtype,
        )

    def fit(self, X: np.ndarray, y: Any = None):
        self._encoder = self._make_encoder()
        self._encoder.fit(np.asarray(X))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("DiffusionNystromManifold must be fit before transform")
        return self._encoder.transform(np.asarray(X))

    def transform_one(self, x: np.ndarray) -> np.ndarray:
        if self._encoder is None:
            raise RuntimeError("DiffusionNystromManifold must be fit before transform_one")
        return self._encoder.transform_one(x)

    def fit_transform(self, X: np.ndarray, y: Any = None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    @property
    def last_ood_(self) -> dict[str, Any] | None:
        if self._encoder is None:
            return None
        return self._encoder.last_ood_

    @property
    def last_stage_latencies_ms_(self) -> dict[str, float] | None:
        if self._encoder is None:
            return None
        return self._encoder.last_stage_latencies_ms_

    def query_diagnostics_batch(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self._encoder is None:
            raise RuntimeError("DiffusionNystromManifold must be fit first")
        return self._encoder.query_diagnostics_batch(X)

    def get_metadata(self) -> dict[str, Any]:
        if self._encoder is None:
            return {
                "manifold_type": "diffusion_nystrom",
                "manifold_grouping": None,
                "manifold_n_components": self.n_components,
                "actual_n_features": None,
                "n_landmarks": self.n_landmarks,
                "landmark_method": self.landmark_method,
                "local_scale_k": self.local_scale_k,
                "alpha": self.alpha,
                "diffusion_time": self.diffusion_time,
                "realtime_compatible": self.realtime_compatible,
                "supports_realtime": True,
                "deployment_tag": self.deployment_tag,
                "explained_variance_ratio": None,
                "groups": [],
            }
        return self._encoder.get_metadata()

    def save(self, output_dir: Path) -> None:
        if self._encoder is None:
            raise RuntimeError("DiffusionNystromManifold must be fit before save")
        output_dir = Path(output_dir)
        self._encoder.save(output_dir)
        with open(output_dir / "meta.json") as f:
            meta = json.load(f)
        meta["class_name"] = "DiffusionNystromManifold"
        meta["feature_mode"] = "diffusion_nystrom"
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> DiffusionNystromManifold:
        enc = DiffusionNystrom.load(input_dir)
        obj = cls(
            n_components=enc.n_components,
            n_landmarks=enc.n_landmarks,
            landmark_method=enc.landmark_method,
            local_scale_k=enc.local_scale_k,
            alpha=enc.alpha,
            diffusion_time=enc.diffusion_time,
            eps=enc.eps,
            transform=enc.transform_name,
            standardize=enc.standardize,
            random_state=enc.random_state,
            dtype=str(enc.dtype),
        )
        obj._encoder = enc
        return obj
