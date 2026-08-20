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
    "global_isomap",
    "global_isomap_distilled",
    "diffusion_nystrom",
)

UI_DYNAMIC_LATENT_OPTIONS: tuple[str, ...] = tuple(ALL_DYNAMIC_LATENT_TYPES)

# Scientific 2×2 on representation class E (linearity × temporal dynamics).
REPRESENTATION_QUADRANTS: dict[str, tuple[str, ...]] = {
    "static_linear": ("counts", "global_pca", "region_pca"),
    "static_nonlinear": (
        "global_isomap",
        "global_isomap_distilled",
        "diffusion_nystrom",
    ),
    "dynamic_linear": ("global_lds", "gpfa"),
    "dynamic_nonlinear": (),
}

REPRESENTATION_QUADRANT_LABELS: dict[str, str] = {
    "static_linear": "Static linear",
    "static_nonlinear": "Static nonlinear",
    "dynamic_linear": "Dynamic linear",
    "dynamic_nonlinear": "Dynamic nonlinear",
}

# Realtime Replay quadrant representatives (realtime-capable only).
REALTIME_QUADRANT_DEFAULTS: dict[str, str | None] = {
    "static_linear": "global_pca",
    "static_nonlinear": "diffusion_nystrom",
    "dynamic_linear": "global_lds",
    "dynamic_nonlinear": None,
}

QUADRANT_ORDER: tuple[str, ...] = (
    "static_linear",
    "static_nonlinear",
    "dynamic_linear",
    "dynamic_nonlinear",
)


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
