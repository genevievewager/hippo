"""Shared representation capability labels for UI badges."""

from __future__ import annotations

from typing import Any

from realtime.dynamic_latents.registry import (
    ALL_DYNAMIC_LATENT_TYPES,
    dynamic_latent_capabilities,
    is_dynamic_latent,
)
from realtime.manifold_features import (
    OFFLINE_ONLY_FEATURE_MODES,
    is_realtime_compatible_feature_mode,
)
from realtime.search_space import resolve_manifold_alias


UI_STATIC_MANIFOLD_OPTIONS: tuple[str, ...] = (
    "counts",
    "global_pca",
    "region_pca",
    "layer_pca",
    "global_isomap",
    "global_isomap_distilled",
)

UI_DYNAMIC_LATENT_OPTIONS: tuple[str, ...] = tuple(ALL_DYNAMIC_LATENT_TYPES)


def representation_capabilities(name: str) -> dict[str, Any]:
    """Return badge metadata for a static or dynamic representation."""
    emb = resolve_manifold_alias(name)
    if is_dynamic_latent(emb):
        caps = dynamic_latent_capabilities(emb)
        return {
            "name": emb,
            "representation_family": "dynamic",
            "supports_realtime": caps["supports_realtime"],
            "supports_causal_transform": caps["supports_causal_transform"],
            "badge": caps["label"],
            "available": True,
        }
    rt = is_realtime_compatible_feature_mode(emb) and emb not in OFFLINE_ONLY_FEATURE_MODES
    # Distilled isomap is realtime-eligible; classic isomap is not.
    if emb == "global_isomap":
        rt = False
    return {
        "name": emb,
        "representation_family": "static",
        "supports_realtime": rt,
        "supports_causal_transform": True,
        "badge": "REALTIME / CAUSAL" if rt else "OFFLINE / ACAUSAL",
        "available": True,
    }


def format_representation_label(name: str) -> str:
    caps = representation_capabilities(name)
    short = "REALTIME" if caps["supports_realtime"] else "OFFLINE"
    return f"{name}  [{short}]"
