"""Flattened finite-history and control feature builders."""

from __future__ import annotations

import numpy as np


def flatten_latent_sequences(sequences: np.ndarray) -> np.ndarray:
    """Flatten [n, L, D] -> [n, L*D]."""
    sequences = np.asarray(sequences)
    if sequences.ndim != 3:
        raise ValueError(f"Expected [n, L, D], got {sequences.shape}")
    n, L, D = sequences.shape
    return sequences.reshape(n, L * D)


def shuffle_sequence_order(
    sequences: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Shuffle order within each history while preserving the same latent states.

    Does not use future information; only permutes the L frames already in the
    causal sequence.
    """
    sequences = np.asarray(sequences).copy()
    if sequences.ndim != 3:
        raise ValueError(f"Expected [n, L, D], got {sequences.shape}")
    n, L, _ = sequences.shape
    if L <= 1:
        return sequences
    for i in range(n):
        order = rng.permutation(L)
        sequences[i] = sequences[i, order]
    return sequences


def average_latent_history(sequences: np.ndarray) -> np.ndarray:
    """Replace ordered history with the mean latent over L frames -> [n, D]."""
    sequences = np.asarray(sequences)
    if sequences.ndim != 3:
        raise ValueError(f"Expected [n, L, D], got {sequences.shape}")
    return sequences.mean(axis=1)
