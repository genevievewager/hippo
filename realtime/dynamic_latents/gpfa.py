"""Gaussian Process Factor Analysis (offline dynamic latent representation).

This is a practical GPFA-style model for offline comparisons:

* Factor-analysis / PCA loading matrix ``C``
* Per-dimension temporal AR(1) / exponential-GP autocorrelation
* Acausal RTS smoothing for latent trajectories

Marked ``supports_realtime=False``. Causal filtering is available for
diagnostics but not advertised as deployable realtime inference.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import FactorAnalysis

from realtime.dynamic_latents.base import DynamicLatentModel
from realtime.dynamic_latents.kalman import (
    _ensure_psd,
    kalman_filter,
    rts_smooth,
)
from realtime.dynamic_latents.metadata import build_model_metadata, try_git_commit


class GPFAModel(DynamicLatentModel):
    """Offline GPFA-style dynamic neural population representation."""

    name = "gpfa"
    supports_realtime = False
    supports_causal_transform = False  # primary path is acausal smoothed
    supports_time_varying_observation = False

    def __init__(
        self,
        n_components: int = 5,
        *,
        max_iter: int = 20,
        random_state: int = 42,
        update_dt: float = 0.025,
        default_tau: float = 0.250,
        feature_set: str | None = None,
        decode_window: float | None = None,
        spike_source: str | None = None,
    ):
        self.n_components = int(n_components)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        self.update_dt = float(update_dt)
        self.default_tau = float(default_tau)
        self.feature_set = feature_set
        self.decode_window = decode_window
        self.spike_source = spike_source

        self.C_: np.ndarray | None = None
        self.d_: np.ndarray | None = None
        self.R_: np.ndarray | None = None
        self.A_: np.ndarray | None = None  # diagonal AR coefficients
        self.Q_: np.ndarray | None = None
        self.tau_: np.ndarray | None = None
        self.mu0_: np.ndarray | None = None
        self.P0_: np.ndarray | None = None
        self.actual_n_components_: int | None = None
        self.n_features_in_: int | None = None
        self.train_loglik_: float | None = None
        self.fit_timestamp_: str | None = None

    @property
    def latent_dim(self) -> int:
        if self.actual_n_components_ is None:
            raise RuntimeError("Model is not fitted")
        return int(self.actual_n_components_)

    def fit(self, X: np.ndarray, timestamps: np.ndarray | None = None, **kwargs: Any) -> GPFAModel:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D [T, n_features]")
        T, n = X.shape
        if T < 5:
            raise ValueError("GPFA fit requires at least 5 timesteps")

        k = min(self.n_components, n, max(1, T - 1))
        self.actual_n_components_ = int(k)
        self.n_features_in_ = int(n)

        fa = FactorAnalysis(n_components=k, random_state=self.random_state, max_iter=500)
        Z = fa.fit_transform(X)
        self.C_ = np.asarray(fa.components_.T, dtype=float)  # [n, k]
        self.d_ = np.mean(X, axis=0)
        noise_var = np.asarray(fa.noise_variance_, dtype=float)
        self.R_ = _ensure_psd(np.diag(np.maximum(noise_var, 1e-4)))

        # Estimate per-dimension AR(1) coeffs / lengthscales from FA scores.
        taus = np.zeros(k)
        a_diag = np.zeros(k)
        q_diag = np.zeros(k)
        dt = self.update_dt
        for i in range(k):
            zi = Z[:, i]
            if np.std(zi) < 1e-12:
                a_diag[i] = 0.9
                q_diag[i] = 0.1
                taus[i] = self.default_tau
                continue
            num = float(np.dot(zi[:-1], zi[1:]))
            den = float(np.dot(zi[:-1], zi[:-1])) + 1e-12
            a = float(np.clip(num / den, -0.99, 0.99))
            a_diag[i] = a
            resid = zi[1:] - a * zi[:-1]
            q_diag[i] = max(float(np.var(resid)), 1e-6)
            # Map AR(1) to exponential timescale: a ≈ exp(-dt / tau)
            if a > 1e-6:
                taus[i] = float(max(-dt / np.log(a), dt))
            else:
                taus[i] = self.default_tau

        self.tau_ = taus
        self.A_ = np.diag(a_diag)
        self.Q_ = _ensure_psd(np.diag(q_diag))
        self.mu0_ = Z[0].copy()
        self.P0_ = _ensure_psd(np.cov(Z.T) + 0.1 * np.eye(k))

        # Optional refining iterations: smooth → re-estimate C/R/A/Q
        loglik = None
        for _ in range(max(1, self.max_iter)):
            filt = kalman_filter(X, self.A_, self.C_, self.d_, self.Q_, self.R_, self.mu0_, self.P0_)
            loglik = filt.loglik
            Zs, _ = rts_smooth(filt, self.A_, self.Q_)

            ones = np.ones((T, 1))
            Z_aug = np.hstack([Zs, ones])
            W, _, _, _ = np.linalg.lstsq(Z_aug, X, rcond=None)
            self.C_ = W[:k].T
            self.d_ = W[k]
            resid = X - (Zs @ self.C_.T + self.d_)
            self.R_ = _ensure_psd(np.diag(np.maximum(np.var(resid, axis=0), 1e-4)))

            for i in range(k):
                zi = Zs[:, i]
                num = float(np.dot(zi[:-1], zi[1:]))
                den = float(np.dot(zi[:-1], zi[:-1])) + 1e-12
                a = float(np.clip(num / den, -0.99, 0.99))
                a_diag[i] = a
                resid_z = zi[1:] - a * zi[:-1]
                q_diag[i] = max(float(np.var(resid_z)), 1e-6)
                if a > 1e-6:
                    taus[i] = float(max(-dt / np.log(a), dt))
            self.A_ = np.diag(a_diag)
            self.Q_ = _ensure_psd(np.diag(q_diag))
            self.tau_ = taus
            self.mu0_ = Zs[0].copy()
            self.P0_ = _ensure_psd(np.cov(Zs.T) + 0.1 * np.eye(k))

        self.train_loglik_ = float(loglik) if loglik is not None else None
        self.fit_timestamp_ = datetime.now(timezone.utc).isoformat()
        return self

    def transform(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
        *,
        causal: bool = False,
        reset: bool = True,
    ) -> np.ndarray:
        """Return latent trajectories.

        Default is **acausal smoothed** (offline GPFA). Setting ``causal=True``
        runs a diagnostic filtered pass but does not change the offline badge.
        """
        self._require_fit()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        filt = kalman_filter(
            X, self.A_, self.C_, self.d_, self.Q_, self.R_, self.mu0_, self.P0_
        )
        if causal:
            return filt.mu
        Zs, _ = rts_smooth(filt, self.A_, self.Q_)
        return Zs

    def step(self, x_t: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "GPFA is offline-only in this project. Use global_lds for realtime "
            "causal latent updates."
        )

    def reset_state(self) -> None:
        """No-op: GPFA has no deployable online state."""
        return None

    def reconstruct(self, Z: np.ndarray) -> np.ndarray:
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
                causal_status="acausal_smoothed",
                random_seed=self.random_state,
                hyperparameters={
                    "max_iter": self.max_iter,
                    "default_tau": self.default_tau,
                    "tau": self.tau_.tolist() if self.tau_ is not None else None,
                },
                training_timestamp=self.fit_timestamp_,
                git_commit=try_git_commit(),
                extra={
                    "manifold_type": "gpfa",
                    "manifold_grouping": None,
                    "manifold_n_components": self.n_components,
                    "actual_n_features": self.actual_n_components_,
                    "train_loglik": self.train_loglik_,
                    "n_features_in": self.n_features_in_,
                    "realtime_compatible": False,
                    "supports_realtime": False,
                    "supports_causal_transform": False,
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
        np.savez_compressed(
            path / "gpfa_params.npz",
            C=self.C_,
            d=self.d_,
            R=self.R_,
            A=self.A_,
            Q=self.Q_,
            tau=self.tau_,
            mu0=self.mu0_,
            P0=self.P0_,
            actual_n_components=self.actual_n_components_,
            n_features_in=self.n_features_in_,
            train_loglik=self.train_loglik_,
            C_t=None,
            observation_mapping_time_varying=False,
        )
        meta = {
            "class_name": "GPFAModel",
            "n_components": self.n_components,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
            "update_dt": self.update_dt,
            "default_tau": self.default_tau,
            "feature_set": self.feature_set,
            "decode_window": self.decode_window,
            "spike_source": self.spike_source,
            "fit_timestamp": self.fit_timestamp_,
            "git_commit": try_git_commit(),
            "supports_realtime": False,
            "supports_causal_transform": False,
            "realtime_compatible": False,
            "representation_family": "dynamic",
            "causal_status": "acausal_smoothed",
            "latent_dimension": self.actual_n_components_,
        }
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> GPFAModel:
        path = Path(path)
        with open(path / "meta.json") as f:
            meta = json.load(f)
        obj = cls(
            n_components=meta["n_components"],
            max_iter=meta.get("max_iter", 20),
            random_state=meta.get("random_state", 42),
            update_dt=meta.get("update_dt", 0.025),
            default_tau=meta.get("default_tau", 0.250),
            feature_set=meta.get("feature_set"),
            decode_window=meta.get("decode_window"),
            spike_source=meta.get("spike_source"),
        )
        data = np.load(path / "gpfa_params.npz", allow_pickle=True)
        obj.C_ = np.asarray(data["C"])
        obj.d_ = np.asarray(data["d"])
        obj.R_ = np.asarray(data["R"])
        obj.A_ = np.asarray(data["A"])
        obj.Q_ = np.asarray(data["Q"])
        obj.tau_ = np.asarray(data["tau"])
        obj.mu0_ = np.asarray(data["mu0"])
        obj.P0_ = np.asarray(data["P0"])
        obj.actual_n_components_ = int(data["actual_n_components"])
        obj.n_features_in_ = int(data["n_features_in"])
        tl = data["train_loglik"]
        obj.train_loglik_ = float(tl) if tl is not None else None
        obj.fit_timestamp_ = meta.get("fit_timestamp")
        return obj

    def _require_fit(self) -> None:
        if self.C_ is None or self.A_ is None:
            raise RuntimeError("GPFAModel must be fit before use")
