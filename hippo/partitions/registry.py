"""Partition registry."""

from __future__ import annotations

from typing import Iterable

from hippo.partitions.anatomical import (
    AllUnitsPartition,
    CA1DeepSuperficialPartition,
    CellClassPartition,
    CellTypePartition,
    DeepSuperficialPartition,
    LayerPartition,
    RateModelPartition,
    SubfieldPartition,
)
from hippo.partitions.base import PartitionResult, PartitionStrategy

_REGISTRY: dict[str, type[PartitionStrategy]] = {
    "all_units": AllUnitsPartition,
    "subfield": SubfieldPartition,
    "layer": LayerPartition,
    "ca1_deep_superficial": CA1DeepSuperficialPartition,
    "deep_superficial": DeepSuperficialPartition,
    "cell_class": CellClassPartition,
    "cell_type": CellTypePartition,
    "rate_model": RateModelPartition,
}

# Planned Phase 2–3 partition names (registered later).
PLANNED_PARTITIONS = (
    "ca1_proximodistal",
    "dorsoventral",
    "projection_target",
    "network_state",
    "physiology_cluster",
    "functional_loading_cluster",
    "coactivity_cluster",
    "communication_loading_cluster",
)


def available_partitions() -> list[str]:
    return sorted(_REGISTRY)


def make_partition(name: str) -> PartitionStrategy:
    key = name.lower()
    if key not in _REGISTRY:
        planned = f" (planned: {key})" if key in PLANNED_PARTITIONS else ""
        raise ValueError(
            f"Unknown partition {name!r}{planned}. Available: {available_partitions()}"
        )
    return _REGISTRY[key]()


def apply_partitions(
    unit_metadata,
    partition_names: Iterable[str],
    *,
    min_units_per_group: int = 5,
) -> dict[str, PartitionResult]:
    results = {}
    for name in partition_names:
        results[name] = make_partition(name).apply(
            unit_metadata, min_units_per_group=min_units_per_group,
        )
    return results
