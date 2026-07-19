"""Causal latent-history sequence construction."""

from __future__ import annotations

import numpy as np


def build_causal_latent_sequences(
    latent_states: np.ndarray,
    history_frames: int,
    pad_mode: str = "drop",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sequences [z_{t-L+1}, ..., z_t] using only current and past states.

    Parameters
    ----------
    latent_states
        Array of shape [n_times, latent_dim], ordered by increasing time.
    history_frames
        L, number of frames in each sequence (including the current frame).
    pad_mode
        ``drop`` (default): omit targets until L frames exist.
        ``edge``: left-pad with the first available latent (no future).

    Returns
    -------
    sequences
        Shape [n_valid, L, latent_dim]
    valid_target_indices
        Indices into the original latent_states / timestamp arrays for each
        sequence's target time t.
    """
    Z = np.asarray(latent_states)
    if Z.ndim != 2:
        raise ValueError(f"latent_states must be 2D, got shape {Z.shape}")
    if history_frames < 1:
        raise ValueError("history_frames must be >= 1")
    if pad_mode not in ("drop", "edge"):
        raise ValueError("pad_mode must be 'drop' or 'edge'")

    n_times, latent_dim = Z.shape
    L = int(history_frames)

    if pad_mode == "drop":
        if n_times < L:
            return (
                np.zeros((0, L, latent_dim), dtype=Z.dtype),
                np.zeros((0,), dtype=np.int64),
            )
        n_valid = n_times - L + 1
        sequences = np.zeros((n_valid, L, latent_dim), dtype=Z.dtype)
        indices = np.arange(L - 1, n_times, dtype=np.int64)
        for i, t in enumerate(indices):
            sequences[i] = Z[t - L + 1 : t + 1]
        return sequences, indices

    # edge pad: every frame is a valid target
    sequences = np.zeros((n_times, L, latent_dim), dtype=Z.dtype)
    indices = np.arange(n_times, dtype=np.int64)
    for t in range(n_times):
        start = t - L + 1
        if start >= 0:
            sequences[t] = Z[start : t + 1]
        else:
            # left-pad with Z[0], never future
            pad = -start
            sequences[t, :pad] = Z[0]
            sequences[t, pad:] = Z[: t + 1]
    return sequences, indices


def apply_prediction_lag(
    valid_indices: np.ndarray,
    lag_frames: int,
    n_times: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map sequence target indices to future behavior indices with lag tau.

    For each sequence ending at t, the target becomes y_{t+lag}.
    Sequences whose lagged target falls outside [0, n_times) are dropped.
    """
    if lag_frames < 0:
        raise ValueError("prediction lag must be >= 0 for causal decoding")
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    target_indices = valid_indices + int(lag_frames)
    keep = (target_indices >= 0) & (target_indices < int(n_times))
    return valid_indices[keep], target_indices[keep]


def lag_seconds_to_frames(lag_s: float, update_dt_s: float) -> int:
    """Convert prediction lag in seconds to whole frames (rounded)."""
    if lag_s < 0:
        raise ValueError("prediction lag must be >= 0")
    if update_dt_s <= 0:
        raise ValueError("update_dt_s must be positive")
    return int(np.round(float(lag_s) / float(update_dt_s)))
