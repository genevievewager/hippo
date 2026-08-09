"""Linear Dynamical System (linear Gaussian state-space) for neural latents.

Model
-----
    z_t = A z_(t-1) + w_t,   w_t ~ N(0, Q)
    x_t = C z_t + d + v_t,   v_t ~ N(0, R)

Inference uses a numerically stable Kalman filter (causal) and optional
RTS smoother (acausal). Parameter fitting uses a small number of EM
iterations initialized from PCA.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from realtime.dynamic_latents.base import DynamicLatentModel
from realtime.dynamic_latents.kalman import (
    _ensure_psd,
    kalman_filter,
    kalman_filter_step,
    rts_smooth,
)
from realtime.dynamic_latents.metadata import build_model_metadata, try_git_commit


class LinearDynamicalSystem(DynamicLatentModel):
    """Global linear Gaussian latent-state model (``global_lds``)."""

    name = "global_lds"
    supports_realtime = True
    supports_causal_transform = True
    supports_time_varying_observation = False  # reserved for adaptive_lds / C_t

    def __init__(
        self,
        n_components: int = 5,
        *,
        n_em_iters: int = 15,
        random_state: int = 42,
        update_dt: float = 0.025,
        process_noise_scale: float = 0.1,
        observation_noise_scale: float = 1.0,
        feature_set: str | None = None,
        decode_window: float | None = None,
        spike_source: str | None = None,
    ):
        self.n_components = int(n_components)
        self.n_em_iters = int(n_em_iters)
        self.random_state = int(random_state)
        self.update_dt = float(update_dt)
        self.process_noise_scale = float(process_noise_scale)
        self.observation_noise_scale = float(observation_noise_scale)
        self.feature_set = feature_set
        self.decode_window = decode_window
        self.spike_source = spike_source

        self.A_: np.ndarray | None = None
        self.C_: np.ndarray | None = None
        self.d_: np.ndarray | None = None
        self.Q_: np.ndarray | None = None
        self.R_: np.ndarray | None = None
        self.mu0_: np.ndarray | None = None
        self.P0_: np.ndarray | None = None
        self.actual_n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.train_loglik_: float | None = None
        self.train_n_samples_: int | None = None
        self.fit_timestamp_: str | None = None

        # Online filter state
        self._mu: np.ndarray | None = None
        self._P: np.ndarray | None = None
        self._step_count: int = 0

    @property
    def latent_dim(self) -> int:
        if self.actual_n_components_ is None:
            raise RuntimeError("Model is not fitted")
        return int(self.actual_n_components_)

    def fit(self, X: np.ndarray, timestamps: np.ndarray | None = None, **kwargs: Any) -> LinearDynamicalSystem:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D [T, n_features]")
        T, n = X.shape
        if T < 3:
            raise ValueError("LDS fit requires at least 3 timesteps")

        k = min(self.n_components, n, max(1, T - 1))
        self.actual_n_components_ = int(k)
        self.n_features_in_ = int(n)
        self.train_n_samples_ = int(T)

        # PCA initialization of loading / latents.
        pca = PCA(n_components=k, random_state=self.random_state)
        Z = pca.fit_transform(X)
        self.C_ = np.asarray(pca.components_.T, dtype=float)  # [n, k]
        self.d_ = np.asarray(pca.mean_, dtype=float)
        # Soft AR(1)-like init for A.
        Z0, Z1 = Z[:-1], Z[1:]
        AtA = Z0.T @ Z0 + 1e-6 * np.eye(k)
        self.A_ = np.linalg.solve(AtA, Z0.T @ Z1).T  # z1 ≈ A z0
        resid_z = Z1 - (self.A_ @ Z0.T).T
        self.Q_ = _ensure_psd(np.cov(resid_z.T) + self.process_noise_scale * np.eye(k))
        resid_x = X - (Z @ self.C_.T + self.d_)
        self.R_ = _ensure_psd(
            np.diag(np.maximum(np.var(resid_x, axis=0), 1e-4))
            + self.observation_noise_scale * 1e-3 * np.eye(n)
        )
        self.mu0_ = Z[0].copy()
        self.P0_ = _ensure_psd(np.cov(Z.T) + 0.1 * np.eye(k))

        # EM iterations (E: filter/smooth, M: closed-form Gaussian updates).
        loglik = None
        for _ in range(max(1, self.n_em_iters)):
            filt = kalman_filter(X, self.A_, self.C_, self.d_, self.Q_, self.R_, self.mu0_, self.P0_)
            loglik = filt.loglik
            Zs, Ps = rts_smooth(filt, self.A_, self.Q_)

            # M-step: observation
            ones = np.ones((T, 1))
            Z_aug = np.hstack([Zs, ones])
            # [C, d] via least squares: x ≈ Z C^T + d
            W, _, _, _ = np.linalg.lstsq(Z_aug, X, rcond=None)
            self.C_ = W[:k].T
            self.d_ = W[k]
            resid = X - (Zs @ self.C_.T + self.d_)
            self.R_ = _ensure_psd(np.diag(np.maximum(np.var(resid, axis=0), 1e-4)))

            # M-step: dynamics
            Z0, Z1 = Zs[:-1], Zs[1:]
            AtA = Z0.T @ Z0 + 1e-6 * np.eye(k)
            self.A_ = np.linalg.solve(AtA, Z0.T @ Z1).T
            resid_z = Z1 - (self.A_ @ Z0.T).T
            # Include posterior covariance contribution approximately via diag mean.
            P_extra = np.mean(Ps[1:], axis=0) + self.A_ @ np.mean(Ps[:-1], axis=0) @ self.A_.T
            self.Q_ = _ensure_psd(np.cov(resid_z.T) + 0.1 * _ensure_psd(P_extra) + 1e-6 * np.eye(k))
            self.mu0_ = Zs[0].copy()
            self.P0_ = _ensure_psd(Ps[0] + 1e-6 * np.eye(k))

        self.train_loglik_ = float(loglik) if loglik is not None else None
        self.fit_timestamp_ = datetime.now(timezone.utc).isoformat()
        self.reset_state()
        return self

    def transform(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
        *,
        causal: bool = True,
        reset: bool = True,
    ) -> np.ndarray:
        self._require_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if reset:
            self.reset_state()

        if causal:
            Z = np.zeros((X.shape[0], self.latent_dim))
            for t in range(X.shape[0]):
                Z[t] = self.step(X[t])
            return Z

        # Acausal smoothed trajectory (offline analysis only).
        filt = kalman_filter(
            X, self.A_, self.C_, self.d_, self.Q_, self.R_, self.mu0_, self.P0_
        )
        Zs, _ = rts_smooth(filt, self.A_, self.Q_)
        # Keep internal state aligned with last filtered belief for optional step().
        self._mu = filt.mu[-1].copy()
        self._P = filt.P[-1].copy()
        self._step_count = X.shape[0]
        return Zs

    def step(self, x_t: np.ndarray) -> np.ndarray:
        self._require_fit()
        x_t = np.asarray(x_t, dtype=float).ravel()
        if self._mu is None or self._P is None:
            self.reset_state()
        assert self._mu is not None and self._P is not None
        is_first = self._step_count == 0
        self._mu, self._P = kalman_filter_step(
            x_t,
            self._mu,
            self._P,
            self.A_,
            self.C_,
            self.d_,
            self.Q_,
            self.R_,
            is_first=is_first,
        )
        self._step_count += 1
        return self._mu.copy()

    def reset_state(self) -> None:
        if self.mu0_ is None or self.P0_ is None:
            self._mu = None
            self._P = None
            self._step_count = 0
            return
        self._mu = self.mu0_.copy()
        self._P = self.P0_.copy()
        self._step_count = 0

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
        """Map latents back to observation space: ``x_hat = C z + d``."""
        self._require_fit()
        Z = np.asarray(Z, dtype=float)
        return Z @ self.C_.T + self.d_

    def get_metadata(self) -> dict[str, Any]:
        meta = super().get_metadata()
        meta.update(
            build_model_metadata(
                model_type=self.name,
                representation_family="dynamic",
                latent_dimension=self.actual_n_components_,
                feature_set=self.feature_set,
                decode_window=self.decode_window,
                update_dt=self.update_dt,
                spike_source=self.spike_source,
                causal_status="causal_filtered",
                random_seed=self.random_state,
                hyperparameters={
                    "n_em_iters": self.n_em_iters,
                    "process_noise_scale": self.process_noise_scale,
                    "observation_noise_scale": self.observation_noise_scale,
                },
                training_timestamp=self.fit_timestamp_,
                git_commit=try_git_commit(),
                extra={
                    "manifold_type": "lds",
                    "manifold_grouping": None,
                    "manifold_n_components": self.n_components,
                    "actual_n_features": self.actual_n_components_,
                    "train_loglik": self.train_loglik_,
                    "train_n_samples": self.train_n_samples_,
                    "n_features_in": self.n_features_in_,
                    "realtime_compatible": True,
                    "supports_realtime": True,
                    "supports_causal_transform": True,
                    "groups": [{
                        "group_name": "all",
                        "n_units": self.n_features_in_,
                        "n_components": self.actual_n_components_,
                        "explained_variance_sum": None,
                        "explained_variance_by_component": None,
                    }],
                },
            )
        )
        return meta

    def save(self, path: Path) -> None:
        self._require_fit()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        payload = {
            "A": self.A_,
            "C": self.C_,
            "d": self.d_,
            "Q": self.Q_,
            "R": self.R_,
            "mu0": self.mu0_,
            "P0": self.P0_,
            "actual_n_components": self.actual_n_components_,
            "n_features_in": self.n_features_in_,
            "train_loglik": self.train_loglik_,
            "train_n_samples": self.train_n_samples_,
            # Placeholder for future time-varying C_t trajectories.
            "C_t": None,
            "observation_mapping_time_varying": False,
        }
        np.savez_compressed(path / "lds_params.npz", **payload)
        meta = {
            "class_name": "LinearDynamicalSystem",
            "n_components": self.n_components,
            "n_em_iters": self.n_em_iters,
            "random_state": self.random_state,
            "update_dt": self.update_dt,
            "process_noise_scale": self.process_noise_scale,
            "observation_noise_scale": self.observation_noise_scale,
            "feature_set": self.feature_set,
            "decode_window": self.decode_window,
            "spike_source": self.spike_source,
            "fit_timestamp": self.fit_timestamp_,
            "git_commit": try_git_commit(),
            **{k: v for k, v in self.get_metadata().items() if _jsonable(v)},
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> LinearDynamicalSystem:
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            n_components=meta["n_components"],
            n_em_iters=meta.get("n_em_iters", 15),
            random_state=meta.get("random_state", 42),
            update_dt=meta.get("update_dt", 0.025),
            process_noise_scale=meta.get("process_noise_scale", 0.1),
            observation_noise_scale=meta.get("observation_noise_scale", 1.0),
            feature_set=meta.get("feature_set"),
            decode_window=meta.get("decode_window"),
            spike_source=meta.get("spike_source"),
        )
        data = np.load(path / "lds_params.npz", allow_pickle=True)
        obj.A_ = np.asarray(data["A"])
        obj.C_ = np.asarray(data["C"])
        obj.d_ = np.asarray(data["d"])
        obj.Q_ = np.asarray(data["Q"])
        obj.R_ = np.asarray(data["R"])
        obj.mu0_ = np.asarray(data["mu0"])
        obj.P0_ = np.asarray(data["P0"])
        obj.actual_n_components_ = int(data["actual_n_components"])
        obj.n_features_in_ = int(data["n_features_in"])
        obj.train_loglik_ = float(data["train_loglik"]) if data["train_loglik"] is not None else None
        tns = data["train_n_samples"]
        obj.train_n_samples_ = int(tns) if tns is not None else None
        obj.fit_timestamp_ = meta.get("fit_timestamp")
        obj.reset_state()
        return obj

    def _require_fit(self) -> None:
        if self.A_ is None or self.C_ is None:
            raise RuntimeError("LinearDynamicalSystem must be fit before use")


def _jsonable(v: Any) -> bool:
    return isinstance(v, (str, int, float, bool, type(None), list, dict))
