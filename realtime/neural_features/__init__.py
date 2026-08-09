"""Causal neural feature representations for hippocampal decoding.

This package sits *upstream* of manifold embeddings:

    spikes → NeuralFeatureExtractor(feature_set) → FeatureVector
          → scaling → manifold(embedding) → decoder

``counts`` is the baseline feature set and must reproduce
``build_causal_spike_matrix``.
"""

from realtime.neural_features.ablation import ablation_feature_sets, describe_ablation
from realtime.neural_features.buffer import CausalSpikeBuffer, required_buffer_seconds
from realtime.neural_features.extractor import NeuralFeatureExtractor
from realtime.neural_features.feature_sets import (
    ALL_FEATURE_SETS,
    DEFAULT_FEATURE_SETS,
    FEATURE_SET_DEFINITIONS,
    QUICK_FEATURE_SETS,
    embedding_compatible_with_feature_set,
    families_for_feature_set,
    resolve_feature_sets,
)
from realtime.neural_features.stability import compute_latent_stability
from realtime.neural_features.types import FeatureExtractionResult, FeatureSpec

__all__ = [
    "ALL_FEATURE_SETS",
    "CausalSpikeBuffer",
    "DEFAULT_FEATURE_SETS",
    "FEATURE_SET_DEFINITIONS",
    "FeatureExtractionResult",
    "FeatureSpec",
    "NeuralFeatureExtractor",
    "QUICK_FEATURE_SETS",
    "ablation_feature_sets",
    "compute_latent_stability",
    "describe_ablation",
    "embedding_compatible_with_feature_set",
    "families_for_feature_set",
    "required_buffer_seconds",
    "resolve_feature_sets",
]
