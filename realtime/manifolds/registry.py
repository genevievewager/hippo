"""Registry for manifold encoders (Phase 1: raw, pca)."""

from __future__ import annotations

from typing import Any

from realtime.manifolds.base import ManifoldEncoder
from realtime.manifolds.pca import PCAManifoldEncoder
from realtime.manifolds.raw import RawManifoldEncoder

_REGISTRY = {
    "raw": RawManifoldEncoder,
    "pca": PCAManifoldEncoder,
}


def available_manifolds() -> list[str]:
    return sorted(_REGISTRY)


def make_manifold_encoder(name: str, **kwargs: Any) -> ManifoldEncoder:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown manifold {name!r}. Available: {available_manifolds()}. "
            "Autoencoder/CEBRA arrive in later phases."
        )
    return _REGISTRY[key](**kwargs)
