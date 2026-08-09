"""Registry for dynamic latent-state representations.

Future placeholders (not yet implemented):
``plds``, ``switching_lds``, ``recurrent_slds``, ``lfads``, ``adaptive_lds``.
"""

from __future__ import annotations

from typing import Any, Type

from realtime.dynamic_latents.base import DynamicLatentModel
from realtime.dynamic_latents.gpfa import GPFAModel
from realtime.dynamic_latents.lds import LinearDynamicalSystem

DYNAMIC_LATENT_REGISTRY: dict[str, Type[DynamicLatentModel]] = {
    "global_lds": LinearDynamicalSystem,
    "gpfa": GPFAModel,
}

# Names reserved for future dynamic models (registered later).
FUTURE_DYNAMIC_LATENTS: tuple[str, ...] = (
    "region_lds",
    "plds",
    "switching_lds",
    "recurrent_slds",
    "lfads",
    "adaptive_lds",
)

ALL_DYNAMIC_LATENT_TYPES: tuple[str, ...] = tuple(DYNAMIC_LATENT_REGISTRY.keys())

OFFLINE_ONLY_DYNAMIC_LATENTS = frozenset(
    name
    for name, cls in DYNAMIC_LATENT_REGISTRY.items()
    if not getattr(cls, "supports_realtime", False)
)


def available_dynamic_latents() -> list[str]:
    return sorted(DYNAMIC_LATENT_REGISTRY)


def is_dynamic_latent(name: str) -> bool:
    return str(name).lower() in DYNAMIC_LATENT_REGISTRY


def is_realtime_compatible_dynamic(name: str) -> bool:
    key = str(name).lower()
    if key not in DYNAMIC_LATENT_REGISTRY:
        return False
    return bool(getattr(DYNAMIC_LATENT_REGISTRY[key], "supports_realtime", False))


def dynamic_latent_capabilities(name: str) -> dict[str, Any]:
    key = str(name).lower()
    if key not in DYNAMIC_LATENT_REGISTRY:
        raise ValueError(f"Unknown dynamic latent {name!r}")
    cls = DYNAMIC_LATENT_REGISTRY[key]
    return {
        "name": key,
        "representation_family": "dynamic",
        "supports_realtime": bool(getattr(cls, "supports_realtime", False)),
        "supports_causal_transform": bool(getattr(cls, "supports_causal_transform", False)),
        "supports_time_varying_observation": bool(
            getattr(cls, "supports_time_varying_observation", False)
        ),
        "label": "REALTIME / CAUSAL"
        if getattr(cls, "supports_realtime", False)
        else "OFFLINE / ACAUSAL",
    }


def list_representation_catalog() -> list[dict[str, Any]]:
    """UI-friendly catalog of registered dynamic methods."""
    out = []
    for name in available_dynamic_latents():
        caps = dynamic_latent_capabilities(name)
        out.append(caps)
    for name in FUTURE_DYNAMIC_LATENTS:
        if name in DYNAMIC_LATENT_REGISTRY:
            continue
        out.append({
            "name": name,
            "representation_family": "dynamic",
            "supports_realtime": False,
            "supports_causal_transform": False,
            "supports_time_varying_observation": name == "adaptive_lds",
            "label": "FUTURE",
            "available": False,
        })
    return out


def make_dynamic_latent(name: str, **kwargs: Any) -> DynamicLatentModel:
    key = str(name).lower()
    if key not in DYNAMIC_LATENT_REGISTRY:
        raise ValueError(
            f"Unknown dynamic latent {name!r}. Available: {available_dynamic_latents()}. "
            f"Reserved for later: {list(FUTURE_DYNAMIC_LATENTS)}"
        )
    return DYNAMIC_LATENT_REGISTRY[key](**kwargs)
