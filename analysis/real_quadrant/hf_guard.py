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


def _positive_int(v: object, *, label: str) -> int:
    n = int(v)  # type: ignore[arg-type]
    if n < 0:
        raise HFFeatureWidthError(f"{label}={n} is not a valid obs dim")
    return n


def model_n_obs(model: object) -> int:
    """Observation / feature width of a fitted latent / decoder model.

    Unwraps ``DynamicLatentEmbedding.model_`` and nested sklearn estimators
    (``pca_``, ``_encoder``, …). Prefers emission matrices (``C_`` / ``C``),
    then ``n_features_in_``, then ``get_metadata()``. Never returns a
    sentinel like ``-1`` — unknown attributes raise.
    """
    inner = getattr(model, "model_", None)
    if inner is not None and inner is not model:
        try:
            return model_n_obs(inner)
        except HFFeatureWidthError:
            pass

    for attr in ("C_", "C", "emission_matrix"):
        C = getattr(model, attr, None)
        if C is not None:
            arr = np.asarray(C)
            if arr.ndim >= 2:
                return int(arr.shape[0])

    for attr in ("n_features_in_", "n_neurons", "n_obs"):
        if not hasattr(model, attr):
            continue
        v = getattr(model, attr)
        if v is None:
            continue
        return _positive_int(v, label=f"{type(model).__name__}.{attr}")

    # Nested fitted estimators (PCA manifold → pca_, Isomap → _encoder, …)
    for attr in ("pca_", "_encoder", "encoder_", "isomap_", "pre_pca_"):
        nested = getattr(model, attr, None)
        if nested is None or nested is model:
            continue
        if hasattr(nested, "n_features_in_") and nested.n_features_in_ is not None:
            return _positive_int(
                nested.n_features_in_,
                label=f"{type(model).__name__}.{attr}.n_features_in_",
            )

    meta_fn = getattr(model, "get_metadata", None)
    if callable(meta_fn):
        try:
            meta = meta_fn()
        except Exception:
            meta = None
        if isinstance(meta, dict):
            for key in ("n_features_in", "n_units"):
                if meta.get(key) is not None:
                    return _positive_int(
                        meta[key], label=f"{type(model).__name__}.meta.{key}",
                    )
            groups = meta.get("groups")
            if isinstance(groups, list) and groups:
                g0 = groups[0]
                if isinstance(g0, dict) and g0.get("n_units") is not None:
                    return _positive_int(
                        g0["n_units"],
                        label=f"{type(model).__name__}.meta.groups[0].n_units",
                    )

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
