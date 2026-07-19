"""Data-driven partition strategies (Phase 3–5 — planned).

Secondary discovery tools based on coactivity, manifold loadings,
task-variable loadings, or communication-subspace loadings.
"""

from __future__ import annotations

from hippo.partitions.base import PartitionResult, PartitionStrategy


class CoactivityClusterPartition(PartitionStrategy):
    name = "coactivity_cluster"

    def apply(self, unit_metadata, *, min_units_per_group: int = 5) -> PartitionResult:
        raise NotImplementedError(
            "coactivity_cluster is planned for Phase 3+. "
            "Not available in the baseline pipeline."
        )


class CommunicationLoadingClusterPartition(PartitionStrategy):
    name = "communication_loading_cluster"

    def apply(self, unit_metadata, *, min_units_per_group: int = 5) -> PartitionResult:
        raise NotImplementedError(
            "communication_loading_cluster is planned for Phase 5."
        )
