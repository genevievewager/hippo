"""Shared types for causal neural feature extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FeatureSpec:
    """Describe one scalar feature column.

    Parameters
    ----------
    name :
        Stable column name aligned with the feature matrix.
    family :
        Feature family id (``counts``, ``count_dynamics``, …).
    region :
        Anatomical region when applicable.
    region_b :
        Second region for cross-region / lagged features.
    unit_id :
        Unit id when the feature is unit-aligned.
    lag_bins :
        Discrete lag in internal coactivity bins (lagged coupling).
    realtime_safe :
        Whether the feature can be computed online with bounded history.
    computational_complexity :
        Qualitative complexity tag (``O(N)``, ``O(R^2 * B)``, …).
    requires_history :
        True when a previous decoder update (or longer buffer) is needed.
    required_history_seconds :
        Minimum causal history length required beyond the decode window.
    """

    name: str
    family: str
    region: str | None = None
    region_b: str | None = None
    unit_id: int | None = None
    lag_bins: int | None = None
    realtime_safe: bool = True
    computational_complexity: str = "O(N)"
    requires_history: bool = False
    required_history_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FeatureExtractionResult:
    """Return payload from NeuralFeatureExtractor."""

    feature_vector: Any  # np.ndarray
    feature_names: list[str]
    feature_metadata: list[FeatureSpec]
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)
