"""PCA manifold baseline: z_t = PCA(preprocess(x_t))."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from realtime.manifolds.base import ManifoldEncoder


class PCAManifoldEncoder(ManifoldEncoder):
    """Linear PCA latent coordinates fit on training spike counts only."""

    name = "pca"

    def __init__(
        self,
        *,
        n_components: int | float = 16,
        transform: str = "sqrt_counts",
        standardize: bool = True,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.transform_name = transform
        self.standardize = standardize
        self.random_state = random_state
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.transform_name == "sqrt_counts":
            X = np.sqrt(np.maximum(X, 0.0))
        return X

    def fit(self, X_train: np.ndarray, labels: Any | None = None) -> PCAManifoldEncoder:
        X = self._preprocess(X_train)
        if self.standardize:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        n_features = X.shape[1]
        n_comp = self.n_components
        if isinstance(n_comp, int):
            n_comp = min(n_comp, n_features, max(1, X.shape[0] - 1))
        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        self.pca_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("PCAManifoldEncoder must be fit before transform")
        X = self._preprocess(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.pca_.transform(X)

    @property
    def latent_dim(self) -> int:
        if self.pca_ is None:
            raise RuntimeError("PCAManifoldEncoder must be fit before latent_dim")
        return int(self.pca_.n_components_)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "n_components": self.n_components,
            "transform": self.transform_name,
            "standardize": self.standardize,
            "random_state": self.random_state,
            "latent_dim": self.latent_dim,
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.scaler_ is not None:
            joblib.dump(self.scaler_, output_dir / "scaler.joblib")
        joblib.dump(self.pca_, output_dir / "pca.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> PCAManifoldEncoder:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            n_components=meta["n_components"],
            transform=meta["transform"],
            standardize=meta["standardize"],
            random_state=meta.get("random_state", 42),
        )
        scaler_path = input_dir / "scaler.joblib"
        if scaler_path.exists():
            obj.scaler_ = joblib.load(scaler_path)
        obj.pca_ = joblib.load(input_dir / "pca.joblib")
        return obj
