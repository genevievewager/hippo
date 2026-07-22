"""Isomap nonlinear manifold encoder: z_t = Isomap(preprocess(x_t)).

Isomap is an **offline** graph-based manifold method. It constructs a
k-nearest-neighbor graph on training neural observations, estimates geodesic
distances via shortest paths, and embeds those distances with classical MDS.

Standard Isomap is tagged ``realtime_compatible=False``. Out-of-sample
``transform`` uses the training graph and is suitable for held-out evaluation,
not automatic streaming deployment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler

from realtime.manifolds.base import ManifoldEncoder
from realtime.manifolds.isomap_diagnostics import (
    DisconnectedGraphError,
    compute_graph_diagnostics,
    validate_graph_connectivity,
)
from realtime.manifolds.isomap_metrics import evaluate_isomap_geometry


class IsomapManifoldEncoder(ManifoldEncoder):
    """Nonlinear Isomap latent coordinates fit on training spike counts only.

    Preprocessing pipeline (training-only fit)::

        X → count transform → standardization → optional PCA → Isomap

    Recommended default transform is ``sqrt_counts`` with standardization.
    """

    name = "isomap"
    realtime_compatible = False
    deployment_tag = "offline_analysis_only"

    def __init__(
        self,
        *,
        n_components: int = 3,
        n_neighbors: int = 10,
        neighbors_algorithm: str = "auto",
        metric: str = "euclidean",
        p: int = 2,
        path_method: str = "auto",
        eigen_solver: str = "auto",
        n_jobs: int | None = -1,
        transform: str = "sqrt_counts",
        standardize: bool = True,
        pre_pca_enabled: bool = True,
        pre_pca_n_components: int = 50,
        pre_pca_variance_threshold: float | None = None,
        require_connected_graph: bool = True,
        allow_largest_component_only: bool = False,
        minimum_largest_component_fraction: float = 0.95,
        trustworthiness_neighbors: tuple[int, ...] = (5, 10, 20),
        compute_residual_variance: bool = True,
        compute_geodesic_preservation: bool = True,
        sampled_distance_pairs: int = 100_000,
        random_state: int = 42,
    ):
        if transform not in ("counts", "sqrt_counts", "rates", "zscore_rates"):
            raise ValueError(
                "transform must be one of: counts, sqrt_counts, rates, zscore_rates"
            )
        self.n_components = int(n_components)
        self.n_neighbors = int(n_neighbors)
        self.neighbors_algorithm = neighbors_algorithm
        self.metric = metric
        self.p = int(p)
        self.path_method = path_method
        self.eigen_solver = eigen_solver
        self.n_jobs = n_jobs
        self.transform_name = transform
        self.standardize = bool(standardize)
        self.pre_pca_enabled = bool(pre_pca_enabled)
        self.pre_pca_n_components = int(pre_pca_n_components)
        self.pre_pca_variance_threshold = pre_pca_variance_threshold
        self.require_connected_graph = bool(require_connected_graph)
        self.allow_largest_component_only = bool(allow_largest_component_only)
        self.minimum_largest_component_fraction = float(
            minimum_largest_component_fraction
        )
        self.trustworthiness_neighbors = tuple(int(k) for k in trustworthiness_neighbors)
        self.compute_residual_variance = bool(compute_residual_variance)
        self.compute_geodesic_preservation = bool(compute_geodesic_preservation)
        self.sampled_distance_pairs = int(sampled_distance_pairs)
        self.random_state = int(random_state)

        self.scaler_: StandardScaler | None = None
        self.pre_pca_: PCA | None = None
        self.isomap_: Isomap | None = None
        self.graph_diagnostics_: dict[str, Any] | None = None
        self.geometry_metrics_: dict[str, Any] | None = None
        self.exclusion_reason_: str | None = None
        self.accepted_: bool = False
        self.n_features_in_: int | None = None
        self.train_component_mask_: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _count_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.transform_name == "sqrt_counts":
            return np.sqrt(np.maximum(X, 0.0))
        if self.transform_name in ("rates", "zscore_rates"):
            # Rates are expected to already be in rate units when supplied;
            # callers using spike counts should prefer sqrt_counts.
            return np.maximum(X, 0.0)
        return X

    def _fit_preprocess(self, X_train: np.ndarray) -> np.ndarray:
        X = self._count_transform(X_train)
        self.n_features_in_ = int(X.shape[1])
        if self.standardize or self.transform_name == "zscore_rates":
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        self.pre_pca_ = None
        if self.pre_pca_enabled:
            n_samples, n_features = X.shape
            if self.pre_pca_variance_threshold is not None:
                n_comp: int | float = float(self.pre_pca_variance_threshold)
            else:
                n_comp = min(
                    self.pre_pca_n_components,
                    n_features,
                    max(1, n_samples - 1),
                )
            self.pre_pca_ = PCA(n_components=n_comp, random_state=self.random_state)
            X = self.pre_pca_.fit_transform(X)
        return X

    def _transform_preprocess(self, X: np.ndarray) -> np.ndarray:
        X = self._count_transform(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        if self.pre_pca_ is not None:
            X = self.pre_pca_.transform(X)
        return X

    # ------------------------------------------------------------------
    # Fit / transform
    # ------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, labels: Any | None = None) -> IsomapManifoldEncoder:
        X_train = np.asarray(X_train, dtype=float)
        if X_train.ndim != 2:
            raise ValueError("X_train must be 2-D [n_times, n_units]")
        X = self._fit_preprocess(X_train)

        n_samples = X.shape[0]
        n_neighbors = min(self.n_neighbors, max(1, n_samples - 1))
        n_components = min(self.n_components, max(1, n_samples - 1), X.shape[1])

        self.isomap_ = Isomap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            neighbors_algorithm=self.neighbors_algorithm,
            metric=self.metric,
            p=self.p,
            path_method=self.path_method,
            eigen_solver=self.eigen_solver,
            n_jobs=self.n_jobs,
        )
        self.isomap_.fit(X)

        self.graph_diagnostics_ = compute_graph_diagnostics(self.isomap_, X)
        accepted, reason = validate_graph_connectivity(
            self.graph_diagnostics_,
            require_connected_graph=self.require_connected_graph,
            allow_largest_component_only=self.allow_largest_component_only,
            minimum_largest_component_fraction=self.minimum_largest_component_fraction,
        )
        self.accepted_ = accepted
        self.exclusion_reason_ = reason
        if not accepted:
            raise DisconnectedGraphError(reason)

        if (
            not self.graph_diagnostics_["graph_connected"]
            and self.allow_largest_component_only
        ):
            labels_cc = self.graph_diagnostics_["component_labels"]
            largest = int(self.graph_diagnostics_["largest_component_label"])
            self.train_component_mask_ = labels_cc == largest
            # Refit on the largest connected component only.
            X_cc = X[self.train_component_mask_]
            n_neighbors_cc = min(n_neighbors, max(1, X_cc.shape[0] - 1))
            n_components_cc = min(n_components, max(1, X_cc.shape[0] - 1), X_cc.shape[1])
            self.isomap_ = Isomap(
                n_components=n_components_cc,
                n_neighbors=n_neighbors_cc,
                neighbors_algorithm=self.neighbors_algorithm,
                metric=self.metric,
                p=self.p,
                path_method=self.path_method,
                eigen_solver=self.eigen_solver,
                n_jobs=self.n_jobs,
            )
            self.isomap_.fit(X_cc)
            self.graph_diagnostics_ = compute_graph_diagnostics(self.isomap_, X_cc)
            X_for_geometry = X_cc
        else:
            self.train_component_mask_ = np.ones(n_samples, dtype=bool)
            X_for_geometry = X

        Z_train = self.isomap_.transform(X_for_geometry)
        geo = getattr(self.isomap_, "dist_matrix_", None)
        self.geometry_metrics_ = evaluate_isomap_geometry(
            X_for_geometry,
            Z_train,
            geo_dist=geo,
            trustworthiness_neighbors=self.trustworthiness_neighbors,
            compute_residual_variance=self.compute_residual_variance,
            compute_geodesic_preservation=self.compute_geodesic_preservation,
            sampled_distance_pairs=self.sampled_distance_pairs,
            random_state=self.random_state,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.isomap_ is None:
            raise RuntimeError("IsomapManifoldEncoder must be fit before transform")
        X = self._transform_preprocess(X)
        return self.isomap_.transform(X)

    def fit_transform(
        self,
        X_train: np.ndarray,
        labels: Any | None = None,
    ) -> np.ndarray:
        self.fit(X_train, labels=labels)
        return self.transform(X_train)

    @property
    def latent_dim(self) -> int:
        if self.isomap_ is None:
            raise RuntimeError("IsomapManifoldEncoder must be fit before latent_dim")
        return int(self.isomap_.embedding_.shape[1])

    def get_metadata(self) -> dict[str, Any]:
        pre_pca_dim = None
        if self.pre_pca_ is not None:
            pre_pca_dim = int(getattr(self.pre_pca_, "n_components_", self.pre_pca_n_components))
        diag = {
            k: v
            for k, v in (self.graph_diagnostics_ or {}).items()
            if k != "component_labels"
        }
        return {
            "manifold_type": "isomap",
            "manifold_method": "isomap",
            "manifold_grouping": None,
            "manifold_n_components": self.n_components,
            "actual_n_features": self.latent_dim if self.isomap_ is not None else None,
            "n_neighbors": self.n_neighbors,
            "pre_pca_enabled": self.pre_pca_enabled,
            "pre_pca_dim": pre_pca_dim,
            "activity_representation": self.transform_name,
            "standardize": self.standardize,
            "realtime_compatible": self.realtime_compatible,
            "deployment_tag": self.deployment_tag,
            "graph_diagnostics": diag,
            "geometry_metrics": self.geometry_metrics_ or {},
            "exclusion_reason": self.exclusion_reason_,
            "accepted": self.accepted_,
            "explained_variance_ratio": None,
            "groups": [{
                "group_name": "all",
                "n_units": self.n_features_in_,
                "n_components": self.latent_dim if self.isomap_ is not None else None,
            }],
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "class_name": "IsomapManifoldEncoder",
            "n_components": self.n_components,
            "n_neighbors": self.n_neighbors,
            "neighbors_algorithm": self.neighbors_algorithm,
            "metric": self.metric,
            "p": self.p,
            "path_method": self.path_method,
            "eigen_solver": self.eigen_solver,
            "n_jobs": self.n_jobs,
            "transform": self.transform_name,
            "standardize": self.standardize,
            "pre_pca_enabled": self.pre_pca_enabled,
            "pre_pca_n_components": self.pre_pca_n_components,
            "pre_pca_variance_threshold": self.pre_pca_variance_threshold,
            "require_connected_graph": self.require_connected_graph,
            "allow_largest_component_only": self.allow_largest_component_only,
            "minimum_largest_component_fraction": self.minimum_largest_component_fraction,
            "trustworthiness_neighbors": list(self.trustworthiness_neighbors),
            "compute_residual_variance": self.compute_residual_variance,
            "compute_geodesic_preservation": self.compute_geodesic_preservation,
            "sampled_distance_pairs": self.sampled_distance_pairs,
            "random_state": self.random_state,
            "latent_dim": self.latent_dim if self.isomap_ is not None else None,
            "n_features_in": self.n_features_in_,
            "realtime_compatible": self.realtime_compatible,
            "deployment_tag": self.deployment_tag,
            "accepted": self.accepted_,
            "exclusion_reason": self.exclusion_reason_,
            "graph_diagnostics": {
                k: v
                for k, v in (self.graph_diagnostics_ or {}).items()
                if k != "component_labels"
            },
            "geometry_metrics": self.geometry_metrics_ or {},
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.scaler_ is not None:
            joblib.dump(self.scaler_, output_dir / "scaler.joblib")
        if self.pre_pca_ is not None:
            joblib.dump(self.pre_pca_, output_dir / "pre_pca.joblib")
        if self.isomap_ is not None:
            joblib.dump(self.isomap_, output_dir / "isomap.joblib")
        if self.train_component_mask_ is not None:
            np.save(output_dir / "train_component_mask.npy", self.train_component_mask_)
        if self.graph_diagnostics_ is not None:
            diag_path = output_dir / "graph_connectivity.json"
            with open(diag_path, "w") as f:
                json.dump(
                    {
                        k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in self.graph_diagnostics_.items()
                        if k != "component_labels"
                    },
                    f,
                    indent=2,
                )
        if self.geometry_metrics_ is not None:
            with open(output_dir / "geometry_metrics.json", "w") as f:
                json.dump(self.geometry_metrics_, f, indent=2)

    @classmethod
    def load(cls, input_dir: Path) -> IsomapManifoldEncoder:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            n_components=meta["n_components"],
            n_neighbors=meta["n_neighbors"],
            neighbors_algorithm=meta.get("neighbors_algorithm", "auto"),
            metric=meta.get("metric", "euclidean"),
            p=meta.get("p", 2),
            path_method=meta.get("path_method", "auto"),
            eigen_solver=meta.get("eigen_solver", "auto"),
            n_jobs=meta.get("n_jobs", -1),
            transform=meta.get("transform", "sqrt_counts"),
            standardize=meta.get("standardize", True),
            pre_pca_enabled=meta.get("pre_pca_enabled", True),
            pre_pca_n_components=meta.get("pre_pca_n_components", 50),
            pre_pca_variance_threshold=meta.get("pre_pca_variance_threshold"),
            require_connected_graph=meta.get("require_connected_graph", True),
            allow_largest_component_only=meta.get("allow_largest_component_only", False),
            minimum_largest_component_fraction=meta.get(
                "minimum_largest_component_fraction", 0.95
            ),
            trustworthiness_neighbors=tuple(
                meta.get("trustworthiness_neighbors", [5, 10, 20])
            ),
            compute_residual_variance=meta.get("compute_residual_variance", True),
            compute_geodesic_preservation=meta.get(
                "compute_geodesic_preservation", True
            ),
            sampled_distance_pairs=meta.get("sampled_distance_pairs", 100_000),
            random_state=meta.get("random_state", 42),
        )
        obj.n_features_in_ = meta.get("n_features_in")
        obj.accepted_ = bool(meta.get("accepted", True))
        obj.exclusion_reason_ = meta.get("exclusion_reason")
        obj.geometry_metrics_ = meta.get("geometry_metrics")
        obj.graph_diagnostics_ = meta.get("graph_diagnostics")
        scaler_path = input_dir / "scaler.joblib"
        if scaler_path.exists():
            obj.scaler_ = joblib.load(scaler_path)
        pre_pca_path = input_dir / "pre_pca.joblib"
        if pre_pca_path.exists():
            obj.pre_pca_ = joblib.load(pre_pca_path)
        isomap_path = input_dir / "isomap.joblib"
        if isomap_path.exists():
            obj.isomap_ = joblib.load(isomap_path)
        mask_path = input_dir / "train_component_mask.npy"
        if mask_path.exists():
            obj.train_component_mask_ = np.load(mask_path)
        return obj
