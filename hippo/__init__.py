"""Shared hierarchical-manifold package: datasets, partitions, unit schemas.

Phase 1 provides foundations used by ``realtime`` decoding without replacing
existing simulation or decoder APIs. Probe trajectory import / cell capture
live under ``hippo.anatomy``.

``ManifoldDataset`` is loaded lazily so ``from hippo.anatomy...`` does not
pull ``realtime.data_loading`` (avoids an import cycle).
"""

from __future__ import annotations

from typing import Any

from hippo.unit_metadata import (
    OPTIONAL_UNIT_COLUMNS,
    REQUIRED_UNIT_COLUMNS,
    normalize_unit_metadata,
)

__all__ = [
    "ManifoldDataset",
    "load_manifold_dataset",
    "REQUIRED_UNIT_COLUMNS",
    "OPTIONAL_UNIT_COLUMNS",
    "normalize_unit_metadata",
]


def __getattr__(name: str) -> Any:
    if name in ("ManifoldDataset", "load_manifold_dataset"):
        from hippo.dataset import ManifoldDataset, load_manifold_dataset

        return ManifoldDataset if name == "ManifoldDataset" else load_manifold_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
