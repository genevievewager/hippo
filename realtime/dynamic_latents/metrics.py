"""Latent-state evaluation metrics for dynamic representations."""

from __future__ import annotations

from typing import Any

import numpy as np


def latent_trajectory_smoothness(Z: np.ndarray) -> dict[str, float]:
    """Smoothness / velocity / acceleration summaries of ``Z`` [T, k]."""
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2 or Z.shape[0] < 3:
        return {
            "latent_velocity_mean": float("nan"),
            "latent_velocity_std": float("nan"),
            "latent_acceleration_mean": float("nan"),
            "latent_smoothness": float("nan"),
            "state_transition_magnitude_mean": float("nan"),
        }
    vel = np.diff(Z, axis=0)
    acc = np.diff(vel, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    acc_mag = np.linalg.norm(acc, axis=1)
    # High smoothness ⇒ small successive velocity change relative to speed.
    smoothness = float(1.0 / (1.0 + np.mean(acc_mag)))
    return {
        "latent_velocity_mean": float(np.mean(speed)),
        "latent_velocity_std": float(np.std(speed)),
        "latent_acceleration_mean": float(np.mean(acc_mag)),
        "latent_smoothness": smoothness,
        "state_transition_magnitude_mean": float(np.mean(speed)),
    }


def observation_reconstruction_error(X: np.ndarray, X_hat: np.ndarray) -> dict[str, float]:
    X = np.asarray(X, dtype=float)
    X_hat = np.asarray(X_hat, dtype=float)
    resid = X - X_hat
    mse = float(np.mean(resid ** 2))
    mae = float(np.mean(np.abs(resid)))
    var = float(np.var(X))
    r2 = float(1.0 - np.sum(resid ** 2) / max(np.sum((X - np.mean(X)) ** 2), 1e-12))
    return {
        "observation_reconstruction_mse": mse,
        "observation_reconstruction_mae": mae,
        "observation_reconstruction_r2": r2,
        "observation_variance": var,
    }


def one_step_latent_prediction_error(Z: np.ndarray, A: np.ndarray) -> dict[str, float]:
    """``||z_t - A z_(t-1)||`` under the fitted transition."""
    Z = np.asarray(Z, dtype=float)
    A = np.asarray(A, dtype=float)
    if Z.shape[0] < 2:
        return {"one_step_latent_prediction_mse": float("nan")}
    pred = (A @ Z[:-1].T).T
    resid = Z[1:] - pred
    return {"one_step_latent_prediction_mse": float(np.mean(resid ** 2))}


def compute_dynamic_latent_metrics(
    *,
    X: np.ndarray,
    Z_causal: np.ndarray | None = None,
    Z_smoothed: np.ndarray | None = None,
    X_hat: np.ndarray | None = None,
    A: np.ndarray | None = None,
    train_loglik: float | None = None,
    causal_status: str = "causal_filtered",
) -> dict[str, Any]:
    """Aggregate latent metrics; never mix causal/acausal scores silently."""
    out: dict[str, Any] = {"causal_status": causal_status}
    Z = Z_causal if Z_causal is not None else Z_smoothed
    if Z is not None:
        out.update(latent_trajectory_smoothness(Z))
        if A is not None:
            out.update(one_step_latent_prediction_error(Z, A))
    if X_hat is not None:
        out.update(observation_reconstruction_error(X, X_hat))
    if train_loglik is not None:
        out["train_log_likelihood"] = float(train_loglik)
    if Z_causal is not None and Z_smoothed is not None:
        # Optional divergence diagnostic — keeps both series separate.
        out["causal_vs_smoothed_mse"] = float(np.mean((Z_causal - Z_smoothed) ** 2))
    return out
