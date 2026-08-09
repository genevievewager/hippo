"""Common interface for dynamic latent-state neural representations.

Static manifolds map ``x_t → z_t``. Dynamic latent models map
``z_(t-1), x_t → z_t`` and maintain internal state across timesteps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class DynamicLatentModel(ABC):
    """Abstract dynamic latent-state estimator.

    Deployable models must support offline fit/transform, causal sequential
    inference via :meth:`step`, reset, and serialization.
    """

    name: str = "base"
    representation_family: str = "dynamic"
    supports_realtime: bool = False
    supports_causal_transform: bool = False
    # Future plasticity: observation mapping may become time-varying C_t.
    supports_time_varying_observation: bool = False

    @abstractmethod
    def fit(self, X: np.ndarray, timestamps: np.ndarray | None = None, **kwargs: Any) -> DynamicLatentModel:
        """Fit model parameters on a temporal sequence of neural features."""

    @abstractmethod
    def transform(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
        *,
        causal: bool = True,
        reset: bool = True,
    ) -> np.ndarray:
        """Map a sequence of observations to latent states ``z_t``.

        Parameters
        ----------
        causal :
            If True, use filtered inference (past observations only).
            If False and supported, use acausal smoothing.
        reset :
            If True, reset latent state before processing the sequence.
            If False, continue from the current internal state (warmup).
        """

    def fit_transform(
        self,
        X: np.ndarray,
        timestamps: np.ndarray | None = None,
        *,
        causal: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        self.fit(X, timestamps=timestamps, **kwargs)
        return self.transform(X, timestamps=timestamps, causal=causal, reset=True)

    @abstractmethod
    def step(self, x_t: np.ndarray) -> np.ndarray:
        """Causal online update: ``z_(t-1), x_t → z_t``.

        Returns shape ``(latent_dim,)``.
        """

    @abstractmethod
    def reset_state(self) -> None:
        """Reset the recurrent latent filter state."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Serialize model parameters and metadata."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> DynamicLatentModel:
        """Load a previously saved model."""

    @property
    def latent_dim(self) -> int:
        raise NotImplementedError

    def get_metadata(self) -> dict[str, Any]:
        """Return representation metadata for metrics tables / UI badges."""
        return {
            "model_type": self.name,
            "representation_family": self.representation_family,
            "supports_realtime": self.supports_realtime,
            "supports_causal_transform": self.supports_causal_transform,
            "supports_time_varying_observation": self.supports_time_varying_observation,
            "realtime_compatible": self.supports_realtime,
            "causal_status": "causal" if self.supports_causal_transform else "acausal",
            "deployment_tag": (
                "realtime_causal" if self.supports_realtime else "offline_analysis_only"
            ),
        }
