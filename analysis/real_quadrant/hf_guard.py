"""Hard guards for Report 2 v2 HF-only feature width.

A shape mismatch between session HF unit count and a feature / model
observation dimension must RAISE. Silent skip is forbidden.
"""

from __future__ import annotations

import numpy as np


class HFFeatureWidthError(ValueError):
    """Features or model obs-dim disagree with the session HF unit count."""


def assert_n_features_eq_n_units(
    X: np.ndarray,
    n_units: int,
    *,
    context: str,
) -> None:
    """Require ``X.shape[1] == n_units`` (HF width at feature entry)."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise HFFeatureWidthError(
            f"{context}: expected 2-D feature matrix, got shape={X.shape}"
        )
    n_feat = int(X.shape[1])
    n_units = int(n_units)
    if n_feat != n_units:
        raise HFFeatureWidthError(
            f"{context}: n_features={n_feat} != n_units={n_units} "
            f"(HF allow-list / model unit set mismatch)"
        )


def assert_model_obs_eq_n_units(
    model: object,
    n_units: int,
    *,
    context: str,
) -> None:
    """Require a fitted GPFA/LDS-like model's observation dim == n_units."""
    n_units = int(n_units)
    C = getattr(model, "C", None)
    if C is not None:
        n_obs = int(np.asarray(C).shape[0])
    else:
        n_obs = int(getattr(model, "n_neurons", -1))
    if n_obs != n_units:
        raise HFFeatureWidthError(
            f"{context}: model_n_obs={n_obs} != n_units={n_units}"
        )
