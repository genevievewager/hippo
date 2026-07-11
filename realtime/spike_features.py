"""Causal spike-count feature construction shared by decoder scripts.

At each decoder update time t, features are population spike counts from the
past causal window [t - decode_window, t). The decoder never uses future spikes.
"""

from __future__ import annotations

import numpy as np

from realtime.spike_binner import (
    build_causal_spike_matrix,
    count_spikes_in_window,
)

__all__ = [
    "apply_feature_mode",
    "build_causal_spike_matrix",
    "count_spikes_in_window",
]


def apply_feature_mode(X: np.ndarray, feature_mode: str, decode_window: float) -> np.ndarray:
    """Convert raw spike counts to counts or rate-normalized features."""
    if feature_mode == "counts":
        return X
    if feature_mode == "rates":
        return X / decode_window
    raise ValueError(f"Unknown feature_mode: {feature_mode}")
