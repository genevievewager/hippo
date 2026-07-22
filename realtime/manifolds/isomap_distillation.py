"""Optional parametric distillation of offline Isomap coordinates.

Trains a causal parametric encoder E_theta(x_t) ≈ z_t^Isomap so that a
streaming-friendly approximation can optionally be deployed. The distilled
model is **not** mathematically identical to Isomap.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from realtime.manifolds.isomap_metrics import procrustes_aligned_error


def _make_distiller(name: str, *, seed: int = 42) -> Any:
    key = name.lower()
    if key == "ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=1.0)),
        ])
    if key == "kernel_ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", KernelRidge(alpha=1.0, kernel="rbf", gamma=None)),
        ])
    if key == "mlp":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                max_iter=400,
                random_state=seed,
            )),
        ])
    raise ValueError(f"Unknown distillation model {name!r}; use ridge, kernel_ridge, mlp")


class IsomapDistilledEncoder:
    """Parametric approximation trained to reproduce Isomap training coordinates.

    Tagged ``isomap_distilled``. Realtime-compatible only when latency and
    held-out distortion thresholds pass.
    """

    name = "isomap_distilled"

    def __init__(
        self,
        *,
        model_name: str = "ridge",
        realtime_latency_budget_ms: float = 50.0,
        max_procrustes_mse: float = 0.5,
        min_procrustes_correlation: float = 0.70,
        min_pairwise_distance_correlation: float = 0.50,
        seed: int = 42,
    ):
        self.model_name = model_name
        self.realtime_latency_budget_ms = float(realtime_latency_budget_ms)
        self.max_procrustes_mse = float(max_procrustes_mse)
        self.min_procrustes_correlation = float(min_procrustes_correlation)
        self.min_pairwise_distance_correlation = float(
            min_pairwise_distance_correlation
        )
        self.seed = int(seed)
        self.pipeline_: Any | None = None
        self.n_features_in_: int | None = None
        self.latent_dim_: int | None = None
        self.metrics_: dict[str, Any] = {}
        self.realtime_compatible_: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        Z_train: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        Z_val: np.ndarray | None = None,
        X_test: np.ndarray | None = None,
        Z_test: np.ndarray | None = None,
    ) -> IsomapDistilledEncoder:
        X_train = np.asarray(X_train, dtype=float)
        Z_train = np.asarray(Z_train, dtype=float)
        self.n_features_in_ = int(X_train.shape[1])
        self.latent_dim_ = int(Z_train.shape[1])
        self.pipeline_ = _make_distiller(self.model_name, seed=self.seed)
        self.pipeline_.fit(X_train, Z_train)

        self.metrics_ = {
            "train_mse": float(np.mean((self.pipeline_.predict(X_train) - Z_train) ** 2)),
        }
        for split_name, Xs, Zs in (
            ("validation", X_val, Z_val),
            ("test", X_test, Z_test),
        ):
            if Xs is None or Zs is None:
                continue
            Xs = np.asarray(Xs, dtype=float)
            Zs = np.asarray(Zs, dtype=float)
            Z_hat = self.pipeline_.predict(Xs)
            mse = float(np.mean((Z_hat - Zs) ** 2))
            proc = procrustes_aligned_error(Zs, Z_hat)
            # Pairwise distance preservation on a subset
            n = min(len(Zs), 200)
            d_true = np.linalg.norm(Zs[:n, None] - Zs[None, :n], axis=-1)
            d_hat = np.linalg.norm(Z_hat[:n, None] - Z_hat[None, :n], axis=-1)
            iu = np.triu_indices(n, k=1)
            if np.std(d_true[iu]) > 1e-12 and np.std(d_hat[iu]) > 1e-12:
                dist_corr = float(np.corrcoef(d_true[iu], d_hat[iu])[0, 1])
            else:
                dist_corr = float("nan")
            self.metrics_[f"{split_name}_mse"] = mse
            self.metrics_[f"{split_name}_procrustes_mse"] = proc["procrustes_mse"]
            self.metrics_[f"{split_name}_procrustes_correlation"] = proc[
                "procrustes_correlation"
            ]
            self.metrics_[f"{split_name}_pairwise_distance_correlation"] = dist_corr

        # Latency: mean time per single causal transform
        probe = X_train[: max(1, min(50, len(X_train)))]
        t0 = time.perf_counter()
        for row in probe:
            self.pipeline_.predict(row.reshape(1, -1))
        elapsed = time.perf_counter() - t0
        per_transform_s = elapsed / max(len(probe), 1)
        self.metrics_["runtime_per_transform_s"] = float(per_transform_s)
        self.metrics_["runtime_per_transform_ms"] = float(per_transform_s * 1000.0)

        budget_s = self.realtime_latency_budget_ms / 1000.0
        latency_ok = per_transform_s <= budget_s
        # Absolute Procrustes MSE is scale-dependent (Isomap coords are not unit-
        # variance). Prefer correlation gates; keep MSE as a secondary option.
        proc_corr = self.metrics_.get(
            "validation_procrustes_correlation",
            self.metrics_.get("test_procrustes_correlation"),
        )
        pair_corr = self.metrics_.get(
            "validation_pairwise_distance_correlation",
            self.metrics_.get("test_pairwise_distance_correlation"),
        )
        distortion = self.metrics_.get(
            "validation_procrustes_mse",
            self.metrics_.get("test_procrustes_mse", self.metrics_["train_mse"]),
        )
        corr_ok = (
            proc_corr is not None
            and np.isfinite(float(proc_corr))
            and float(proc_corr) >= self.min_procrustes_correlation
        )
        pair_ok = (
            pair_corr is not None
            and np.isfinite(float(pair_corr))
            and float(pair_corr) >= self.min_pairwise_distance_correlation
        )
        mse_ok = float(distortion) <= self.max_procrustes_mse
        distortion_ok = bool(corr_ok or pair_ok or mse_ok)
        self.realtime_compatible_ = bool(latency_ok and distortion_ok)
        self.metrics_["realtime_compatible"] = self.realtime_compatible_
        self.metrics_["latency_ok"] = latency_ok
        self.metrics_["distortion_ok"] = distortion_ok
        self.metrics_["distortion_corr_ok"] = corr_ok
        self.metrics_["distortion_pair_ok"] = pair_ok
        self.metrics_["distortion_mse_ok"] = mse_ok
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pipeline_ is None:
            raise RuntimeError("IsomapDistilledEncoder must be fit before transform")
        return np.asarray(self.pipeline_.predict(np.asarray(X, dtype=float)))

    @property
    def latent_dim(self) -> int:
        if self.latent_dim_ is None:
            raise RuntimeError("IsomapDistilledEncoder must be fit before latent_dim")
        return self.latent_dim_

    @property
    def realtime_compatible(self) -> bool:
        return bool(self.realtime_compatible_)

    def save(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        meta = {
            "name": self.name,
            "class_name": "IsomapDistilledEncoder",
            "model_name": self.model_name,
            "realtime_latency_budget_ms": self.realtime_latency_budget_ms,
            "max_procrustes_mse": self.max_procrustes_mse,
            "min_procrustes_correlation": self.min_procrustes_correlation,
            "min_pairwise_distance_correlation": self.min_pairwise_distance_correlation,
            "seed": self.seed,
            "n_features_in": self.n_features_in_,
            "latent_dim": self.latent_dim_,
            "realtime_compatible": self.realtime_compatible_,
            "metrics": self.metrics_,
            "note": (
                "Parametric approximation of Isomap coordinates; "
                "not mathematically identical to Isomap."
            ),
        }
        with open(output_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        joblib.dump(self.pipeline_, output_dir / "distiller.joblib")

    @classmethod
    def load(cls, input_dir: Path) -> IsomapDistilledEncoder:
        input_dir = Path(input_dir)
        with open(input_dir / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            model_name=meta["model_name"],
            realtime_latency_budget_ms=meta.get("realtime_latency_budget_ms", 50.0),
            max_procrustes_mse=meta.get("max_procrustes_mse", 0.5),
            min_procrustes_correlation=meta.get("min_procrustes_correlation", 0.70),
            min_pairwise_distance_correlation=meta.get(
                "min_pairwise_distance_correlation", 0.50
            ),
            seed=meta.get("seed", 42),
        )
        obj.n_features_in_ = meta.get("n_features_in")
        obj.latent_dim_ = meta.get("latent_dim")
        obj.metrics_ = meta.get("metrics", {})
        obj.realtime_compatible_ = bool(meta.get("realtime_compatible", False))
        obj.pipeline_ = joblib.load(input_dir / "distiller.joblib")
        return obj


def distill_isomap_encoder(
    X_train: np.ndarray,
    Z_train: np.ndarray,
    *,
    model_name: str = "ridge",
    X_val: np.ndarray | None = None,
    Z_val: np.ndarray | None = None,
    X_test: np.ndarray | None = None,
    Z_test: np.ndarray | None = None,
    output_dir: Path | None = None,
    realtime_latency_budget_ms: float = 50.0,
    seed: int = 42,
) -> IsomapDistilledEncoder:
    """Fit a distilled encoder and optionally save under ``output_dir``."""
    enc = IsomapDistilledEncoder(
        model_name=model_name,
        realtime_latency_budget_ms=realtime_latency_budget_ms,
        seed=seed,
    )
    enc.fit(
        X_train, Z_train,
        X_val=X_val, Z_val=Z_val,
        X_test=X_test, Z_test=Z_test,
    )
    if output_dir is not None:
        enc.save(Path(output_dir))
    return enc
