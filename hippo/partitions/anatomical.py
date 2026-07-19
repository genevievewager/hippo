"""Anatomical population partitions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hippo.partitions.base import PartitionResult, PartitionStrategy, filter_small_groups


class AllUnitsPartition(PartitionStrategy):
    name = "all_units"

    def apply(self, unit_metadata: pd.DataFrame, *, min_units_per_group: int = 5) -> PartitionResult:
        ids = unit_metadata["unit_id"].to_numpy(dtype=int)
        labels = {"all": ids}
        kept, excluded = filter_small_groups(labels, min_units_per_group)
        meta = pd.DataFrame([{"group": "all", "n_units": len(ids), "basis": "all_units"}])
        return PartitionResult(
            partition_name=self.name,
            group_labels=kept,
            included_unit_ids=ids if "all" in kept else np.array([], dtype=int),
            group_metadata=meta,
            exclusion_reason=excluded,
        )


class SubfieldPartition(PartitionStrategy):
    name = "subfield"

    def apply(self, unit_metadata: pd.DataFrame, *, min_units_per_group: int = 5) -> PartitionResult:
        col = "subfield" if "subfield" in unit_metadata.columns else "region"
        labels: dict[str, np.ndarray] = {}
        for group, sub in unit_metadata.groupby(unit_metadata[col].astype(str)):
            labels[str(group)] = sub["unit_id"].to_numpy(dtype=int)
        kept, excluded = filter_small_groups(labels, min_units_per_group)
        included = np.concatenate(list(kept.values())) if kept else np.array([], dtype=int)
        meta = pd.DataFrame([
            {"group": g, "n_units": len(ids), "basis": col} for g, ids in kept.items()
        ])
        return PartitionResult(
            partition_name=self.name,
            group_labels=kept,
            included_unit_ids=included,
            group_metadata=meta,
            exclusion_reason=excluded,
        )


class LayerPartition(PartitionStrategy):
    name = "layer"

    def apply(self, unit_metadata: pd.DataFrame, *, min_units_per_group: int = 5) -> PartitionResult:
        if "layer" not in unit_metadata.columns:
            return PartitionResult(
                partition_name=self.name,
                group_labels={},
                included_unit_ids=np.array([], dtype=int),
                group_metadata=pd.DataFrame(),
                exclusion_reason={"layer": "column missing from unit_metadata"},
            )
        labels = {
            str(g): sub["unit_id"].to_numpy(dtype=int)
            for g, sub in unit_metadata.groupby(unit_metadata["layer"].astype(str))
        }
        kept, excluded = filter_small_groups(labels, min_units_per_group)
        included = np.concatenate(list(kept.values())) if kept else np.array([], dtype=int)
        meta = pd.DataFrame([
            {"group": g, "n_units": len(ids), "basis": "layer"} for g, ids in kept.items()
        ])
        return PartitionResult(
            partition_name=self.name,
            group_labels=kept,
            included_unit_ids=included,
            group_metadata=meta,
            exclusion_reason=excluded,
        )


class CA1DeepSuperficialPartition(PartitionStrategy):
    """Partition CA1 units by deep/superficial group (Phase-1 layer-derived)."""

    name = "ca1_deep_superficial"

    def apply(self, unit_metadata: pd.DataFrame, *, min_units_per_group: int = 5) -> PartitionResult:
        ca1 = unit_metadata[unit_metadata["region"].astype(str) == "CA1"]
        if ca1.empty or "deep_superficial_group" not in ca1.columns:
            return PartitionResult(
                partition_name=self.name,
                group_labels={},
                included_unit_ids=np.array([], dtype=int),
                group_metadata=pd.DataFrame(),
                exclusion_reason={"ca1_deep_superficial": "no CA1 units or group column"},
            )
        labels = {}
        for group, sub in ca1.groupby(ca1["deep_superficial_group"].astype(str)):
            if group in ("unknown",):
                continue
            labels[f"CA1_{group}"] = sub["unit_id"].to_numpy(dtype=int)
        kept, excluded = filter_small_groups(labels, min_units_per_group)
        included = np.concatenate(list(kept.values())) if kept else np.array([], dtype=int)
        meta = pd.DataFrame([
            {"group": g, "n_units": len(ids), "basis": "deep_superficial_group"}
            for g, ids in kept.items()
        ])
        return PartitionResult(
            partition_name=self.name,
            group_labels=kept,
            included_unit_ids=included,
            group_metadata=meta,
            exclusion_reason=excluded,
        )


class CellClassPartition(PartitionStrategy):
    name = "cell_class"

    def apply(self, unit_metadata: pd.DataFrame, *, min_units_per_group: int = 5) -> PartitionResult:
        col = "cell_class" if "cell_class" in unit_metadata.columns else "cell_type"
        labels = {
            str(g): sub["unit_id"].to_numpy(dtype=int)
            for g, sub in unit_metadata.groupby(unit_metadata[col].astype(str))
        }
        kept, excluded = filter_small_groups(labels, min_units_per_group)
        included = np.concatenate(list(kept.values())) if kept else np.array([], dtype=int)
        meta = pd.DataFrame([
            {"group": g, "n_units": len(ids), "basis": col} for g, ids in kept.items()
        ])
        return PartitionResult(
            partition_name=self.name,
            group_labels=kept,
            included_unit_ids=included,
            group_metadata=meta,
            exclusion_reason=excluded,
        )


class RateModelPartition(PartitionStrategy):
    name = "rate_model"

    def apply(self, unit_metadata: pd.DataFrame, *, min_units_per_group: int = 5) -> PartitionResult:
        col = "rate_model" if "rate_model" in unit_metadata.columns else "ratinabox_class"
        if col not in unit_metadata.columns:
            return PartitionResult(
                partition_name=self.name,
                group_labels={},
                included_unit_ids=np.array([], dtype=int),
                group_metadata=pd.DataFrame(),
                exclusion_reason={"rate_model": "column missing"},
            )
        labels = {
            str(g): sub["unit_id"].to_numpy(dtype=int)
            for g, sub in unit_metadata.groupby(unit_metadata[col].astype(str))
        }
        kept, excluded = filter_small_groups(labels, min_units_per_group)
        included = np.concatenate(list(kept.values())) if kept else np.array([], dtype=int)
        meta = pd.DataFrame([
            {"group": g, "n_units": len(ids), "basis": col} for g, ids in kept.items()
        ])
        return PartitionResult(
            partition_name=self.name,
            group_labels=kept,
            included_unit_ids=included,
            group_metadata=meta,
            exclusion_reason=excluded,
        )
