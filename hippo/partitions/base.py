"""Population partition protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PartitionResult:
    """Result of applying a named population partition."""

    partition_name: str
    group_labels: dict[str, np.ndarray]  # group_name -> unit_ids
    included_unit_ids: np.ndarray
    group_metadata: pd.DataFrame
    exclusion_reason: dict[str, str] = field(default_factory=dict)

    @property
    def group_names(self) -> list[str]:
        return sorted(self.group_labels.keys())

    def unit_count(self, group: str) -> int:
        return int(len(self.group_labels.get(group, [])))


class PartitionStrategy:
    """Base class for anatomical / physiological / data-driven partitions."""

    name: str = "base"

    def apply(
        self,
        unit_metadata: pd.DataFrame,
        *,
        min_units_per_group: int = 5,
    ) -> PartitionResult:
        raise NotImplementedError


def filter_small_groups(
    labels: dict[str, np.ndarray],
    min_units: int,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    kept = {}
    excluded = {}
    for name, ids in labels.items():
        if len(ids) < min_units:
            excluded[name] = f"n_units={len(ids)} < min_units_per_group={min_units}"
        else:
            kept[name] = np.asarray(ids, dtype=int)
    return kept, excluded
