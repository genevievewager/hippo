"""Identity / raw manifold: z_t = x_t (optional sqrt + standardization)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from realtime.manifolds.base import ManifoldEncoder


class RawManifoldEncoder(ManifoldEncoder):
    """Pass-through representation with optional sqrt counts and scaling."""

    name = "raw"

    def __init__(self, *, transform: str = "sqrt_counts", standardize: bool = True):
        if transform not in ("counts", "sqrt_counts"):
            raise ValueError("transform must be 'counts' or 'sqrt_counts'")
        self.transform_name = transform
        self.standardize = standardize
        self.scaler_: StandardScaler | None = None
        self.n_features_: int | None = None

    def _preprocess(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.transform_name == "sqrt_counts":
            X = np.sqrt(np.maximum(X, 0.0))
        return X

    def fit(self, X_train: np.ndarray, labels: Any | None = None) -> RawManifoldEncoder:
        X = self._preprocess(X_train)
        self.n_features_ = int(X.shape[1])
        if self.standardize:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.n_features_ is None:
            raise RuntimeError("RawManifoldEncoder must be fit before transform")
        X = self._preprocess(X)
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return X

    @property
    def latent_dim(self) -> int:
        if self.n_features_ is None:
            raise RuntimeError("RawManifoldEncoder must be fit before latent_dim")
        return self.n_features_

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "transform": self.transform_name,
            "standardize": self.standardize,
            "n_features": self.n_features_,
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        if self.scaler_ is not None:
            joblib.dump(self.scaler_, output_dir / "scaler.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> RawManifoldEncoder:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        obj = cls(transform=meta["transform"], standardize=meta["standardize"])
        obj.n_features_ = meta["n_features"]
        scaler_path = input_dir / "scaler.joblib"
        if scaler_path.exists():
            obj.scaler_ = joblib.load(scaler_path)
        return obj
