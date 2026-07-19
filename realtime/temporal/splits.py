"""Temporally contiguous train / validation / test splits with leakage gaps."""

from __future__ import annotations

import numpy as np


def causal_train_val_test_split(
    decode_times: np.ndarray,
    *,
    train_frac: float = 0.60,
    val_frac: float = 0.15,
    gap_s: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split with optional discard gaps between splits.

    Frames are never shuffled. Samples inside gap windows are excluded from
    all splits to prevent latent-history / integration-window leakage.
    """
    t = np.asarray(decode_times, dtype=float)
    if len(t) < 3:
        raise ValueError("Need at least 3 decode times for train/val/test split")
    if not 0.0 < train_frac < 1.0 or not 0.0 < val_frac < 1.0:
        raise ValueError("train_frac and val_frac must be in (0, 1)")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1")

    t0, t1 = float(t[0]), float(t[-1])
    span = t1 - t0
    train_end = t0 + train_frac * span
    val_end = train_end + val_frac * span

    train_mask = t < (train_end - gap_s)
    val_mask = (t >= (train_end + gap_s)) & (t < (val_end - gap_s))
    test_mask = t >= (val_end + gap_s)

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError(
            "Split produced an empty partition; reduce gap_s or adjust fractions "
            f"(train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()})"
        )
    return train_mask, val_mask, test_mask


def required_split_gap_s(
    max_integration_window_s: float,
    max_latent_history_s: float,
    max_prediction_lag_s: float,
) -> float:
    """Minimum gap separating contiguous splits."""
    return float(
        max_integration_window_s + max_latent_history_s + max_prediction_lag_s
    )


def mask_sequences_within_split(
    sequence_end_indices: np.ndarray,
    history_frames: int,
    split_mask: np.ndarray,
) -> np.ndarray:
    """
    Keep sequences whose entire history lies inside the same split.

    sequence_end_indices indexes the target time t for each sequence.
    """
    sequence_end_indices = np.asarray(sequence_end_indices, dtype=np.int64)
    split_mask = np.asarray(split_mask, dtype=bool)
    keep = np.zeros(len(sequence_end_indices), dtype=bool)
    L = int(history_frames)
    for i, t in enumerate(sequence_end_indices):
        start = int(t) - L + 1
        if start < 0:
            continue
        if split_mask[start : int(t) + 1].all():
            keep[i] = True
    return keep
