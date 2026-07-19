"""Population partition strategies."""

from hippo.partitions.base import PartitionResult, PartitionStrategy
from hippo.partitions.registry import (
    PLANNED_PARTITIONS,
    apply_partitions,
    available_partitions,
    make_partition,
)

__all__ = [
    "PartitionResult",
    "PartitionStrategy",
    "PLANNED_PARTITIONS",
    "apply_partitions",
    "available_partitions",
    "make_partition",
]
