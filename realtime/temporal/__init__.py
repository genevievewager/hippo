"""Temporal decoding package (Phase 1: sequences, flattened history, controls)."""

from __future__ import annotations

from realtime.temporal.sequences import (
    apply_prediction_lag,
    build_causal_latent_sequences,
    lag_seconds_to_frames,
)
from realtime.temporal.splits import (
    causal_train_val_test_split,
    mask_sequences_within_split,
    required_split_gap_s,
)

__all__ = [
    "apply_prediction_lag",
    "build_causal_latent_sequences",
    "causal_train_val_test_split",
    "lag_seconds_to_frames",
    "mask_sequences_within_split",
    "required_split_gap_s",
]
