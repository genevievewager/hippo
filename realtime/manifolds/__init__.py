"""Neural manifold encoders: current-frame latent states z_t = enc(x_t)."""

from __future__ import annotations

from realtime.manifolds.base import ManifoldEncoder
from realtime.manifolds.isomap import IsomapManifoldEncoder
from realtime.manifolds.pca import PCAManifoldEncoder
from realtime.manifolds.raw import RawManifoldEncoder
from realtime.manifolds.registry import (
    available_manifolds,
    is_realtime_compatible_manifold,
    make_manifold_encoder,
)

__all__ = [
    "ManifoldEncoder",
    "PCAManifoldEncoder",
    "RawManifoldEncoder",
    "IsomapManifoldEncoder",
    "available_manifolds",
    "is_realtime_compatible_manifold",
    "make_manifold_encoder",
]
