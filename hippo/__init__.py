"""Shared hierarchical-manifold package: datasets, partitions, unit schemas.

Phase 1 provides foundations used by ``realtime`` decoding without replacing
existing simulation or decoder APIs.
"""

from hippo.dataset import ManifoldDataset, load_manifold_dataset
from hippo.unit_metadata import (
    REQUIRED_UNIT_COLUMNS,
    OPTIONAL_UNIT_COLUMNS,
    normalize_unit_metadata,
)

__all__ = [
    "ManifoldDataset",
    "load_manifold_dataset",
    "REQUIRED_UNIT_COLUMNS",
    "OPTIONAL_UNIT_COLUMNS",
    "normalize_unit_metadata",
]
