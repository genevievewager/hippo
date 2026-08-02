"""Causal spike-count feature construction shared by decoder scripts.

At each decoder update time t, features are population spike counts from the
past causal window [t - decode_window, t). The decoder never uses future spikes.

Feature representation search (F) lives in ``realtime.feature_representations``.
"""

from __future__ import annotations

import numpy as np

from realtime.feature_representations import (
    ALL_FEATURE_TYPES,
    SpikeFeatureTransformer,
    make_spike_feature_transformer,
)
from realtime.spike_binner import (
    build_causal_spike_matrix,
    count_spikes_in_window,
)

__all__ = [
    "ALL_FEATURE_TYPES",
    "SpikeFeatureTransformer",
    "apply_feature_mode",
    "build_causal_spike_matrix",
    "count_spikes_in_window",
    "make_spike_feature_transformer",
]


def apply_feature_mode(X: np.ndarray, feature_mode: str, decode_window: float) -> np.ndarray:
    """Convert raw spike counts to counts or rate-normalized features.

    For train-fit normalizations (z-score, grouped), use SpikeFeatureTransformer.
    """
    if feature_mode == "counts":
        return X
    if feature_mode == "rates":
        return X / decode_window
    if feature_mode == "sqrt_counts":
        return np.sqrt(np.maximum(np.asarray(X, dtype=float), 0.0))
    if feature_mode == "log1p_counts":
        return np.log1p(np.maximum(np.asarray(X, dtype=float), 0.0))
    raise ValueError(
        f"Unknown feature_mode: {feature_mode}. "
        "For train-only normalizations use SpikeFeatureTransformer."
    )
