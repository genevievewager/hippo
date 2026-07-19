"""Network-state partitions (Phase 2+ — planned)."""

from __future__ import annotations

from hippo.partitions.base import PartitionResult, PartitionStrategy


class NetworkStatePartition(PartitionStrategy):
    name = "network_state"

    def apply(self, unit_metadata, *, min_units_per_group: int = 5) -> PartitionResult:
        raise NotImplementedError(
            "network_state partitions require per-time annotations and are "
            "planned for Phase 2+. Unit metadata alone is insufficient."
        )
