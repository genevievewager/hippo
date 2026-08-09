"""Decoder workflow profiles: quick / standard / full / manifolds / feature_robustness.

Profiles set lean defaults so a casual user gets a strong decoder without a
research-scale sweep. Explicit CLI flags always override profile values.

``manifolds`` is the one-command path for testing all realtime-relevant
embeddings (PCA family + classic / distilled Isomap) with a bounded grid.

``feature_robustness`` is the public modular full-grid path:
W × FeatureSet × Manifold × Decoder (re-runnable on existing sim outputs
without re-simulating).
"""

from __future__ import annotations

from dataclasses import dataclass

from realtime.adaptive_windows import COARSE_DECODE_WINDOWS, WINDOW_CANDIDATE_POOL
from realtime.decoder_comparison import DEFAULT_DECODE_WINDOWS, DEFAULT_MANIFOLD_N_COMPONENTS
from realtime.manifold_features import (
    DEFAULT_ISOMAP_N_NEIGHBORS,
    MANIFOLDS_FEATURE_MODES,
    MANIFOLDS_ISOMAP_N_NEIGHBORS,
    MANIFOLDS_N_COMPONENTS,
    QUICK_FEATURE_MODES,
)
from realtime.neural_features import ALL_FEATURE_SETS
from realtime.timing import DEFAULT_LATENT_HISTORY_FRAMES


LEAN_LATENT_HISTORY_FRAMES: tuple[int, ...] = (1, 5, 20)
LEAN_TEMPORAL_REPRESENTATIONS: tuple[str, ...] = ("pca",)
LEAN_TEMPORAL_MODELS: tuple[str, ...] = (
    "raw_static",
    "static_latent",
    "flattened_history",
)
FULL_TEMPORAL_REPRESENTATIONS: tuple[str, ...] = ("raw", "pca")
FULL_TEMPORAL_MODELS: tuple[str, ...] = (
    "raw_static",
    "static_latent",
    "flattened_history",
    "shuffled_sequence",
    "averaged_history",
)

# Embedding / manifold grid for the feature-robustness public path.
# ``identity`` = no manifold (CLI alias ``counts``).
FEATURE_ROBUSTNESS_EMBEDDINGS: tuple[str, ...] = (
    "identity",
    "global_pca",
    "region_pca",
    "layer_pca",
    "global_isomap",
    "global_isomap_distilled",
)


@dataclass(frozen=True)
class WorkflowProfile:
    """Resolved defaults for one named workflow profile."""

    name: str
    decode_windows: tuple[float, ...]
    adaptive_windows: bool
    feature_modes: tuple[str, ...]
    manifold_n_components: tuple[int, ...]
    max_models: str
    compare_sources: bool
    enable_temporal_manifold: bool
    temporal_inherit_windows: bool
    representations: tuple[str, ...]
    latent_history_frames: tuple[int, ...]
    temporal_models: tuple[str, ...]
    prediction_lags: tuple[float, ...]
    isomap_n_neighbors: tuple[int, ...] = (DEFAULT_ISOMAP_N_NEIGHBORS,)
    enable_isomap_distillation: bool = False
    # Neural feature-set axis (upstream of manifolds). Default preserves counts-only.
    feature_sets: tuple[str, ...] = ("counts",)
    # When set, expand F×E grid with feature_types=("counts",) passthrough.
    embedding_types: tuple[str, ...] | None = None
    run_feature_ablation: bool = False


PROFILES: dict[str, WorkflowProfile] = {
    "quick": WorkflowProfile(
        name="quick",
        decode_windows=COARSE_DECODE_WINDOWS,
        adaptive_windows=False,
        feature_modes=QUICK_FEATURE_MODES,
        manifold_n_components=DEFAULT_MANIFOLD_N_COMPONENTS,
        max_models="quick",
        compare_sources=False,
        enable_temporal_manifold=False,
        temporal_inherit_windows=True,
        representations=LEAN_TEMPORAL_REPRESENTATIONS,
        latent_history_frames=LEAN_LATENT_HISTORY_FRAMES,
        temporal_models=LEAN_TEMPORAL_MODELS,
        prediction_lags=(0.0,),
    ),
    "standard": WorkflowProfile(
        name="standard",
        decode_windows=WINDOW_CANDIDATE_POOL,
        adaptive_windows=False,
        feature_modes=QUICK_FEATURE_MODES,
        manifold_n_components=DEFAULT_MANIFOLD_N_COMPONENTS,
        max_models="quick",
        compare_sources=False,
        enable_temporal_manifold=False,
        temporal_inherit_windows=True,
        representations=LEAN_TEMPORAL_REPRESENTATIONS,
        latent_history_frames=LEAN_LATENT_HISTORY_FRAMES,
        temporal_models=LEAN_TEMPORAL_MODELS,
        prediction_lags=(0.0,),
    ),
    "full": WorkflowProfile(
        name="full",
        decode_windows=DEFAULT_DECODE_WINDOWS,
        adaptive_windows=False,
        feature_modes=QUICK_FEATURE_MODES,
        manifold_n_components=DEFAULT_MANIFOLD_N_COMPONENTS,
        max_models="full",
        compare_sources=False,
        enable_temporal_manifold=False,
        temporal_inherit_windows=False,
        representations=FULL_TEMPORAL_REPRESENTATIONS,
        latent_history_frames=DEFAULT_LATENT_HISTORY_FRAMES,
        temporal_models=FULL_TEMPORAL_MODELS,
        prediction_lags=(0.0,),
    ),
    "manifolds": WorkflowProfile(
        name="manifolds",
        decode_windows=COARSE_DECODE_WINDOWS,
        adaptive_windows=False,
        feature_modes=MANIFOLDS_FEATURE_MODES,
        manifold_n_components=MANIFOLDS_N_COMPONENTS,
        max_models="quick",
        compare_sources=False,
        enable_temporal_manifold=False,
        temporal_inherit_windows=True,
        representations=LEAN_TEMPORAL_REPRESENTATIONS,
        latent_history_frames=LEAN_LATENT_HISTORY_FRAMES,
        temporal_models=LEAN_TEMPORAL_MODELS,
        prediction_lags=(0.0,),
        isomap_n_neighbors=MANIFOLDS_ISOMAP_N_NEIGHBORS,
        enable_isomap_distillation=True,
    ),
    # Public modular full grid (reuses existing simulation outputs).
    "feature_robustness": WorkflowProfile(
        name="feature_robustness",
        decode_windows=DEFAULT_DECODE_WINDOWS,
        adaptive_windows=False,
        feature_modes=MANIFOLDS_FEATURE_MODES,
        manifold_n_components=MANIFOLDS_N_COMPONENTS,
        max_models="quick",
        compare_sources=False,
        enable_temporal_manifold=False,
        temporal_inherit_windows=True,
        representations=LEAN_TEMPORAL_REPRESENTATIONS,
        latent_history_frames=LEAN_LATENT_HISTORY_FRAMES,
        temporal_models=LEAN_TEMPORAL_MODELS,
        prediction_lags=(0.0,),
        isomap_n_neighbors=MANIFOLDS_ISOMAP_N_NEIGHBORS,
        enable_isomap_distillation=True,
        feature_sets=ALL_FEATURE_SETS,
        embedding_types=FEATURE_ROBUSTNESS_EMBEDDINGS,
        run_feature_ablation=True,
    ),
}


def get_profile(name: str) -> WorkflowProfile:
    key = str(name).strip().lower()
    if key not in PROFILES:
        raise ValueError(
            f"Unknown profile {name!r}. Choose one of: {', '.join(sorted(PROFILES))}."
        )
    return PROFILES[key]


__all__ = [
    "COARSE_DECODE_WINDOWS",
    "FEATURE_ROBUSTNESS_EMBEDDINGS",
    "FULL_TEMPORAL_MODELS",
    "FULL_TEMPORAL_REPRESENTATIONS",
    "LEAN_LATENT_HISTORY_FRAMES",
    "LEAN_TEMPORAL_MODELS",
    "LEAN_TEMPORAL_REPRESENTATIONS",
    "PROFILES",
    "WINDOW_CANDIDATE_POOL",
    "WorkflowProfile",
    "get_profile",
]
