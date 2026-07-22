"""Static feature wrapper: offline Isomap → parametric distilled encoder.

``global_isomap_distilled`` fits classic Isomap on training counts, then trains
a causal parametric map ``E_θ(x) ≈ z_Isomap`` used for all transforms.
Realtime-compatible when the distilled model passes latency + distortion gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from realtime.manifolds.isomap import IsomapManifoldEncoder
from realtime.manifolds.isomap_distillation import (
    IsomapDistilledEncoder,
    distill_isomap_encoder,
)


class IsomapDistilledManifold(BaseEstimator, TransformerMixin):
    """Parametric approximation of Isomap coordinates for streaming decode."""

    name = "isomap_distilled"
    deployment_tag = "isomap_distilled"

    def __init__(
        self,
        n_components: int = 8,
        n_neighbors: int = 10,
        *,
        distiller_model: str = "ridge",
        realtime_latency_budget_ms: float = 50.0,
        max_procrustes_mse: float = 0.5,
        min_procrustes_correlation: float = 0.70,
        min_pairwise_distance_correlation: float = 0.50,
        transform: str = "sqrt_counts",
        standardize: bool = True,
        pre_pca_enabled: bool = True,
        pre_pca_n_components: int = 50,
        require_connected_graph: bool = True,
        allow_largest_component_only: bool = False,
        minimum_largest_component_fraction: float = 0.95,
        n_jobs: int | None = -1,
        random_state: int = 42,
        validation_frac: float = 0.15,
    ):
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.distiller_model = str(distiller_model)
        self.realtime_latency_budget_ms = float(realtime_latency_budget_ms)
        self.max_procrustes_mse = float(max_procrustes_mse)
        self.min_procrustes_correlation = float(min_procrustes_correlation)
        self.min_pairwise_distance_correlation = float(
            min_pairwise_distance_correlation
        )
        self.transform_name = transform
        self.standardize = bool(standardize)
        self.pre_pca_enabled = bool(pre_pca_enabled)
        self.pre_pca_n_components = int(pre_pca_n_components)
        self.require_connected_graph = bool(require_connected_graph)
        self.allow_largest_component_only = bool(allow_largest_component_only)
        self.minimum_largest_component_fraction = float(
            minimum_largest_component_fraction
        )
        self.n_jobs = n_jobs
        self.random_state = int(random_state)
        self.validation_frac = float(validation_frac)
        self._isomap: IsomapManifoldEncoder | None = None
        self._distiller: IsomapDistilledEncoder | None = None

    @property
    def realtime_compatible(self) -> bool:
        if self._distiller is None:
            return False
        return bool(self._distiller.realtime_compatible)

    def fit(self, X: np.ndarray, y: Any = None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        if n < 20:
            raise ValueError("Need at least 20 training samples to distill Isomap")

        self._isomap = IsomapManifoldEncoder(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            transform=self.transform_name,
            standardize=self.standardize,
            pre_pca_enabled=self.pre_pca_enabled,
            pre_pca_n_components=self.pre_pca_n_components,
            require_connected_graph=self.require_connected_graph,
            allow_largest_component_only=self.allow_largest_component_only,
            minimum_largest_component_fraction=self.minimum_largest_component_fraction,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._isomap.fit(X)
        Z = self._isomap.transform(X)

        # Hold out a trailing fraction of the *training* split for distillation
        # validation (still never uses decoder test data).
        n_val = max(5, int(round(n * self.validation_frac)))
        n_val = min(n_val, n // 4)
        n_fit = n - n_val
        X_fit, X_val = X[:n_fit], X[n_fit:]
        Z_fit, Z_val = Z[:n_fit], Z[n_fit:]

        self._distiller = distill_isomap_encoder(
            X_fit,
            Z_fit,
            model_name=self.distiller_model,
            X_val=X_val,
            Z_val=Z_val,
            realtime_latency_budget_ms=self.realtime_latency_budget_ms,
            seed=self.random_state,
        )
        # Re-fit distiller on full train for best streaming approximation while
        # keeping gate metrics from the held-out validation slice.
        gate_metrics = dict(self._distiller.metrics_)
        gate_compatible = bool(self._distiller.realtime_compatible_)
        full = IsomapDistilledEncoder(
            model_name=self.distiller_model,
            realtime_latency_budget_ms=self.realtime_latency_budget_ms,
            max_procrustes_mse=self.max_procrustes_mse,
            min_procrustes_correlation=self.min_procrustes_correlation,
            min_pairwise_distance_correlation=self.min_pairwise_distance_correlation,
            seed=self.random_state,
        )
        full.fit(X, Z)
        full.metrics_.update({
            k: v for k, v in gate_metrics.items()
            if k.startswith("validation_") or k in (
                "latency_ok", "distortion_ok", "realtime_compatible",
            )
        })
        # Prefer validation-based gate for realtime_compatible flag.
        full.realtime_compatible_ = gate_compatible
        full.metrics_["realtime_compatible"] = gate_compatible
        full.metrics_["distiller_model"] = self.distiller_model
        full.metrics_["teacher"] = "isomap"
        self._distiller = full
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self._distiller is None:
            raise RuntimeError("IsomapDistilledManifold must be fit before transform")
        return self._distiller.transform(np.asarray(X, dtype=float))

    def transform_teacher(self, X: np.ndarray) -> np.ndarray:
        """Offline Isomap teacher coordinates (not for streaming deploy)."""
        if self._isomap is None:
            raise RuntimeError("IsomapDistilledManifold must be fit before transform_teacher")
        return self._isomap.transform(np.asarray(X, dtype=float))

    def get_metadata(self) -> dict[str, Any]:
        latent = self.n_components
        if self._distiller is not None:
            latent = self._distiller.latent_dim
        meta: dict[str, Any] = {
            "manifold_type": "isomap_distilled",
            "manifold_grouping": None,
            "manifold_n_components": self.n_components,
            "actual_n_features": latent if self._distiller is not None else None,
            "n_neighbors": self.n_neighbors,
            "realtime_compatible": self.realtime_compatible,
            "deployment_tag": self.deployment_tag,
            "distiller_model": self.distiller_model,
            "explained_variance_ratio": None,
            "groups": [],
            "teacher": "isomap",
            "note": (
                "Parametric approximation of Isomap; not mathematically identical."
            ),
        }
        if self._distiller is not None:
            meta["distillation_metrics"] = self._distiller.metrics_
            meta["runtime_per_transform_ms"] = self._distiller.metrics_.get(
                "runtime_per_transform_ms"
            )
        if self._isomap is not None:
            teacher_meta = self._isomap.get_metadata()
            meta["teacher_geometry_metrics"] = teacher_meta.get("geometry_metrics")
            meta["teacher_graph_diagnostics"] = teacher_meta.get("graph_diagnostics")
            meta["pre_pca_enabled"] = teacher_meta.get("pre_pca_enabled")
            meta["pre_pca_dim"] = teacher_meta.get("pre_pca_dim")
        return meta

    def save(self, output_dir: Path) -> None:
        if self._isomap is None or self._distiller is None:
            raise RuntimeError("IsomapDistilledManifold must be fit before save")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._isomap.save(output_dir / "teacher_isomap")
        self._distiller.save(output_dir / "distiller")
        meta = self.get_metadata()
        meta["class_name"] = "IsomapDistilledManifold"
        meta["feature_mode"] = "global_isomap_distilled"
        meta["n_components"] = self.n_components
        meta["n_neighbors"] = self.n_neighbors
        meta["transform"] = self.transform_name
        meta["standardize"] = self.standardize
        meta["pre_pca_enabled"] = self.pre_pca_enabled
        meta["pre_pca_n_components"] = self.pre_pca_n_components
        meta["require_connected_graph"] = self.require_connected_graph
        meta["allow_largest_component_only"] = self.allow_largest_component_only
        meta["minimum_largest_component_fraction"] = (
            self.minimum_largest_component_fraction
        )
        meta["n_jobs"] = self.n_jobs
        meta["random_state"] = self.random_state
        meta["validation_frac"] = self.validation_frac
        meta["realtime_latency_budget_ms"] = self.realtime_latency_budget_ms
        meta["max_procrustes_mse"] = self.max_procrustes_mse
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> IsomapDistilledManifold:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            n_components=int(meta.get("n_components", meta.get("manifold_n_components", 8))),
            n_neighbors=int(meta.get("n_neighbors", 10)),
            distiller_model=str(meta.get("distiller_model", "ridge")),
            realtime_latency_budget_ms=float(
                meta.get("realtime_latency_budget_ms", 50.0)
            ),
            max_procrustes_mse=float(meta.get("max_procrustes_mse", 0.5)),
            min_procrustes_correlation=float(
                meta.get("min_procrustes_correlation", 0.70)
            ),
            min_pairwise_distance_correlation=float(
                meta.get("min_pairwise_distance_correlation", 0.50)
            ),
            transform=str(meta.get("transform", "sqrt_counts")),
            standardize=bool(meta.get("standardize", True)),
            pre_pca_enabled=bool(meta.get("pre_pca_enabled", True)),
            pre_pca_n_components=int(meta.get("pre_pca_n_components", 50)),
            require_connected_graph=bool(meta.get("require_connected_graph", True)),
            allow_largest_component_only=bool(
                meta.get("allow_largest_component_only", False)
            ),
            minimum_largest_component_fraction=float(
                meta.get("minimum_largest_component_fraction", 0.95)
            ),
            n_jobs=meta.get("n_jobs", -1),
            random_state=int(meta.get("random_state", 42)),
            validation_frac=float(meta.get("validation_frac", 0.15)),
        )
        obj._isomap = IsomapManifoldEncoder.load(input_dir / "teacher_isomap")
        obj._distiller = IsomapDistilledEncoder.load(input_dir / "distiller")
        return obj
