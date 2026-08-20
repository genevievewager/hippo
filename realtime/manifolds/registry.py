"""Registry for manifold encoders (raw, pca, isomap)."""

from __future__ import annotations

from typing import Any

from realtime.manifolds.base import ManifoldEncoder
from realtime.manifolds.diffusion_nystrom import DiffusionNystrom
from realtime.manifolds.isomap import IsomapManifoldEncoder
from realtime.manifolds.pca import PCAManifoldEncoder
from realtime.manifolds.raw import RawManifoldEncoder

_REGISTRY = {
    "raw": RawManifoldEncoder,
    "pca": PCAManifoldEncoder,
    "isomap": IsomapManifoldEncoder,
    "diffusion_nystrom": DiffusionNystrom,
}

# Offline-only methods that must not auto-deploy into realtime replay.
OFFLINE_ONLY_MANIFOLDS = frozenset({"isomap"})


def available_manifolds() -> list[str]:
    return sorted(_REGISTRY)


def is_realtime_compatible_manifold(name: str) -> bool:
    return name.lower() not in OFFLINE_ONLY_MANIFOLDS


def make_manifold_encoder(name: str, **kwargs: Any) -> ManifoldEncoder:
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown manifold {name!r}. Available: {available_manifolds()}. "
            "Autoencoder/CEBRA arrive in later phases."
        )
    return _REGISTRY[key](**kwargs)
