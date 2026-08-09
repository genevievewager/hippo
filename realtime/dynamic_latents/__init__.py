"""Dynamic latent-state representations (LDS, GPFA, …).

These models estimate a temporal latent neural state ``z_t`` from causal
neural features ``x_t``. The behavioral decoder may remain a static map
``z_t → y_hat``.
"""

from realtime.dynamic_latents.adapters import DynamicLatentEmbedding
from realtime.dynamic_latents.base import DynamicLatentModel
from realtime.dynamic_latents.gpfa import GPFAModel
from realtime.dynamic_latents.lds import LinearDynamicalSystem
from realtime.dynamic_latents.registry import (
    ALL_DYNAMIC_LATENT_TYPES,
    DYNAMIC_LATENT_REGISTRY,
    FUTURE_DYNAMIC_LATENTS,
    OFFLINE_ONLY_DYNAMIC_LATENTS,
    available_dynamic_latents,
    dynamic_latent_capabilities,
    is_dynamic_latent,
    is_realtime_compatible_dynamic,
    list_representation_catalog,
    make_dynamic_latent,
)

__all__ = [
    "ALL_DYNAMIC_LATENT_TYPES",
    "DYNAMIC_LATENT_REGISTRY",
    "DynamicLatentEmbedding",
    "DynamicLatentModel",
    "FUTURE_DYNAMIC_LATENTS",
    "GPFAModel",
    "LinearDynamicalSystem",
    "OFFLINE_ONLY_DYNAMIC_LATENTS",
    "available_dynamic_latents",
    "dynamic_latent_capabilities",
    "is_dynamic_latent",
    "is_realtime_compatible_dynamic",
    "list_representation_catalog",
    "make_dynamic_latent",
]
