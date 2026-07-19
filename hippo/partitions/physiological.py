"""Physiological partition strategies (Phase 3 — planned).

Candidate continuous features for unsupervised clustering:
mean rate, burst index, waveform width, theta modulation, ripple participation,
spatial information, population coupling.

These must be fit on training data only and must not use target labels unless
explicitly labeled supervised.
"""

from __future__ import annotations

from hippo.partitions.base import PartitionResult, PartitionStrategy


class PhysiologyClusterPartition(PartitionStrategy):
    """Planned: GMM / hierarchical clustering on physiological features."""

    name = "physiology_cluster"

    def apply(self, unit_metadata, *, min_units_per_group: int = 5) -> PartitionResult:
        raise NotImplementedError(
            "physiology_cluster is planned for Phase 3. "
            "Use anatomical partitions (subfield, cell_class, …) until then."
        )
