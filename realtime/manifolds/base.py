"""Common interface for current-frame manifold encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class ManifoldEncoder(ABC):
    """Map a current spike-count vector x_t to a latent state z_t.

    Encoders must not use prior or future frames. Fit only on training data.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X_train: np.ndarray, labels: Any | None = None) -> ManifoldEncoder:
        """Fit preprocessing / encoder on training observations only."""

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform observations to latent coordinates, shape [n, latent_dim]."""

    @abstractmethod
    def save(self, output_dir: Path) -> None:
        """Serialize encoder artifacts into output_dir."""

    @classmethod
    @abstractmethod
    def load(cls, input_dir: Path) -> ManifoldEncoder:
        """Load a previously saved encoder."""

    @property
    def latent_dim(self) -> int:
        raise NotImplementedError
