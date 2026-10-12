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


def model_n_obs(model: object) -> int:
    """Observation / feature width of a fitted latent model (GPFA uses ``C_``)."""
    for attr in ("C_", "C", "emission_matrix"):
        C = getattr(model, attr, None)
        if C is not None:
            return int(np.asarray(C).shape[0])
    for attr in ("n_features_in_", "n_neurons", "n_obs"):
        v = getattr(model, attr, None)
        if v is not None:
            return int(v)
    raise HFFeatureWidthError(
        f"cannot determine observation dim on {type(model).__name__}"
    )


def assert_model_obs_eq_n_units(
    model: object,
    n_units: int,
    *,
    context: str,
) -> None:
    """Require a fitted GPFA/LDS-like model's observation dim == n_units."""
    n_units = int(n_units)
    n_obs = model_n_obs(model)
    if n_obs != n_units:
        raise HFFeatureWidthError(
            f"{context}: model_n_obs={n_obs} != n_units={n_units}"
        )
