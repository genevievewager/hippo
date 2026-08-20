"""F × E × D × W × C search-space helpers for decoder comparison."""

from __future__ import annotations

from typing import Any

from realtime.feature_representations import (
    ALL_FEATURE_TYPES,
    QUICK_FEATURE_TYPES,
    resolve_feature_types,
)
from realtime.manifold_features import (
    ALL_FEATURE_MODES,
    DEFAULT_ISOMAP_N_NEIGHBORS,
    DEFAULT_N_LANDMARKS,
    IDENTITY_FEATURE_MODES,
    MANIFOLD_FEATURE_MODES,
    is_manifold_feature_mode,
)

# Embedding / manifold types (search dimension E).
ALL_EMBEDDING_TYPES = (
    "identity",
    "global_pca",
    "region_pca",
    "layer_pca",
    "cell_type_pca",
    "rate_model_pca",
    "pls",
    "bayesian_place_tuning",
    # Legacy / advanced (kept for backward compatibility with --feature-modes)
    "global_isomap",
    "global_isomap_distilled",
    "diffusion_nystrom",
    # Dynamic latent-state embeddings
    "global_lds",
    "gpfa",
)

QUICK_EMBEDDING_TYPES = ("identity", "global_pca", "region_pca")
FULL_EMBEDDING_TYPES = (
    "identity",
    "global_pca",
    "region_pca",
    "layer_pca",
    "cell_type_pca",
    "rate_model_pca",
    "pls",
    "bayesian_place_tuning",
)

DEFAULT_DECODE_WINDOWS = (0.025, 0.050, 0.100, 0.250, 0.500, 1.000)
DEFAULT_MANIFOLD_N_COMPONENTS = (2, 3, 5, 10)

# Map legacy combined feature_mode → (feature_type F, embedding_type E)
LEGACY_MODE_TO_FE: dict[str, tuple[str, str]] = {
    "counts": ("counts", "identity"),
    "rates": ("rates", "identity"),
    "global_pca": ("counts", "global_pca"),
    "region_pca": ("counts", "region_pca"),
    "layer_pca": ("counts", "layer_pca"),
    "cell_type_pca": ("counts", "cell_type_pca"),
    "rate_model_pca": ("counts", "rate_model_pca"),
    "global_isomap": ("counts", "global_isomap"),
    "global_isomap_distilled": ("counts", "global_isomap_distilled"),
    "diffusion_nystrom": ("counts", "diffusion_nystrom"),
    "global_lds": ("counts", "global_lds"),
    "gpfa": ("counts", "gpfa"),
}

# CLI ``--manifolds`` aliases: ``counts`` means identity embedding (no manifold),
# not the neural feature set named ``counts``.
MANIFOLD_CLI_ALIASES: dict[str, str] = {
    "counts": "identity",
    "none": "identity",
    "identity": "identity",
    "no_manifold": "identity",
}


def resolve_manifold_alias(name: str) -> str:
    """Resolve a manifold / embedding CLI token to an embedding_type."""
    key = str(name).strip()
    return MANIFOLD_CLI_ALIASES.get(key, key)


def resolve_embedding_types(
    embedding_types: tuple[str, ...] | list[str] | None,
    *,
    max_models: str = "quick",
) -> tuple[str, ...]:
    if embedding_types:
        unknown = [e for e in embedding_types if e not in ALL_EMBEDDING_TYPES]
        if unknown:
            raise ValueError(f"Unknown embedding type(s): {unknown}")
        return tuple(dict.fromkeys(embedding_types))
    if max_models == "full":
        return FULL_EMBEDDING_TYPES
    return QUICK_EMBEDDING_TYPES


def compose_feature_mode(feature_type: str, embedding_type: str) -> str:
    """Legacy-compatible composite key for model paths and viz fallbacks."""
    if embedding_type == "identity":
        return feature_type
    if feature_type == "counts":
        return embedding_type
    return f"{feature_type}__{embedding_type}"


def legacy_mode_to_fe(feature_mode: str) -> tuple[str, str]:
    if feature_mode in LEGACY_MODE_TO_FE:
        return LEGACY_MODE_TO_FE[feature_mode]
    if "__" in feature_mode:
        f, e = feature_mode.split("__", 1)
        return f, e
    if feature_mode in ALL_FEATURE_TYPES:
        return feature_mode, "identity"
    if feature_mode in ALL_EMBEDDING_TYPES:
        return "counts", feature_mode
    raise ValueError(f"Cannot map feature_mode {feature_mode!r} to (F, E)")


def is_diffusion_nystrom(embedding_type: str) -> bool:
    return (
        embedding_type == "diffusion_nystrom"
        or embedding_type.endswith("_diffusion_nystrom")
    )


def expand_fe_jobs(
    *,
    feature_types: tuple[str, ...] | None = None,
    embedding_types: tuple[str, ...] | None = None,
    feature_modes: tuple[str, ...] | None = None,
    manifold_n_components: tuple[int, ...] = (3,),
    isomap_n_neighbors: tuple[int, ...] = (DEFAULT_ISOMAP_N_NEIGHBORS,),
    n_landmarks: tuple[int, ...] = (DEFAULT_N_LANDMARKS,),
    max_models: str = "quick",
    use_fe_grid: bool = False,
) -> list[tuple[str, str, int | None, int | None]]:
    """
    Return jobs as (feature_type, embedding_type, n_components, n_neighbors).

    For ``diffusion_nystrom``, the 4th slot stores ``n_landmarks`` (not Isomap k).
    """
    jobs: list[tuple[str, str, int | None, int | None]] = []

    if use_fe_grid or feature_types is not None or embedding_types is not None:
        f_types = resolve_feature_types(feature_types, max_models=max_models)
        e_types = resolve_embedding_types(embedding_types, max_models=max_models)
        for f in f_types:
            for e in e_types:
                jobs.extend(
                    _expand_embedding_jobs(
                        f, e, manifold_n_components, isomap_n_neighbors, n_landmarks,
                    )
                )
        return jobs

    modes = feature_modes or (
        ("counts", "global_pca", "region_pca") if max_models != "full" else ALL_FEATURE_MODES
    )
    for mode in modes:
        f, e = legacy_mode_to_fe(mode)
        jobs.extend(
            _expand_embedding_jobs(
                f, e, manifold_n_components, isomap_n_neighbors, n_landmarks,
            )
        )
    return jobs


def _expand_embedding_jobs(
    feature_type: str,
    embedding_type: str,
    manifold_n_components: tuple[int, ...],
    isomap_n_neighbors: tuple[int, ...],
    n_landmarks: tuple[int, ...] = (DEFAULT_N_LANDMARKS,),
) -> list[tuple[str, str, int | None, int | None]]:
    if embedding_type == "identity" or embedding_type == "bayesian_place_tuning":
        return [(feature_type, embedding_type, None, None)]

    if is_diffusion_nystrom(embedding_type):
        out = []
        for k in manifold_n_components:
            for nl in n_landmarks:
                out.append((feature_type, embedding_type, int(k), int(nl)))
        return out

    if (
        embedding_type in ("global_isomap", "global_isomap_distilled")
        or embedding_type.endswith("_isomap")
        or embedding_type.endswith("_isomap_distilled")
    ):
        out = []
        for k in manifold_n_components:
            for nn in isomap_n_neighbors:
                out.append((feature_type, embedding_type, int(k), int(nn)))
        return out

    # PCA / PLS / LDS / GPFA and other component-based embeddings
    if (
        embedding_type.endswith("_pca")
        or embedding_type == "global_pca"
        or embedding_type == "pls"
        or embedding_type in ("global_lds", "gpfa")
        or is_manifold_feature_mode(embedding_type)
    ):
        return [
            (feature_type, embedding_type, int(k), None)
            for k in manifold_n_components
        ]

    return [(feature_type, embedding_type, None, None)]


def embedding_needs_targets(embedding_type: str) -> bool:
    """True when the embedding fit requires behavioral labels (supervised)."""
    return embedding_type in {"pls", "bayesian_place_tuning"}


def is_identity_embedding(embedding_type: str) -> bool:
    return embedding_type == "identity"


def is_counts_like_mode(feature_type: str, embedding_type: str) -> bool:
    return embedding_type == "identity" and feature_type in IDENTITY_FEATURE_MODES


def is_manifold_embedding(embedding_type: str) -> bool:
    if embedding_type == "identity":
        return False
    if embedding_type in MANIFOLD_FEATURE_MODES:
        return True
    return embedding_type in {
        "pls",
        "bayesian_place_tuning",
        "global_pca",
        "region_pca",
        "layer_pca",
        "cell_type_pca",
        "rate_model_pca",
        "global_lds",
        "gpfa",
        "diffusion_nystrom",
    }


def is_dynamic_embedding(embedding_type: str) -> bool:
    return embedding_type in {"global_lds", "gpfa"}


def representation_family(embedding_type: str) -> str:
    return "dynamic" if is_dynamic_embedding(embedding_type) else "static"


def fe_job_dict(
    feature_type: str,
    embedding_type: str,
    n_components: int | None,
    n_neighbors: int | None,
) -> dict[str, Any]:
    return {
        "feature_type": feature_type,
        "embedding_type": embedding_type,
        "feature_mode": compose_feature_mode(feature_type, embedding_type),
        "manifold_n_components": n_components,
        "n_neighbors": n_neighbors,
    }
