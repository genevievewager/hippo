"""Thin wrappers around Report 2 PCA/LDS fit paths (never reimplement)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from realtime.quadrant_n5_run import (
    _make_rep,
    _sqrt_zscore_train,
    fit_transform_representation,
)

_CFG: dict[str, Any] | None = None


def load_quadrant_cfg() -> dict[str, Any]:
    global _CFG
    if _CFG is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "quadrant_n5.yaml"
        with path.open() as f:
            _CFG = yaml.safe_load(f)
    return _CFG


def fit_transform_pca_lds(
    method: str,
    X_counts: np.ndarray,
    fit_mask: np.ndarray,
    *,
    d: int,
    seed: int = 0,
) -> np.ndarray:
    """Sqrt+zscore on train, then Report 2 ``_make_rep`` + ``fit_transform_representation``."""
    if method not in ("pca", "lds"):
        raise ValueError(method)
    cfg = load_quadrant_cfg()
    X, _ = _sqrt_zscore_train(np.asarray(X_counts, dtype=float), np.asarray(fit_mask, dtype=bool))
    n_fit = int(np.asarray(fit_mask, dtype=bool).sum())
    model = _make_rep(method, int(d), cfg, int(seed), n_fit)
    return fit_transform_representation(method, model, X, fit_mask)


def circular_shift_units(X: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Independent circular shift per column (unit). ``shifts`` length = n_units."""
    X = np.asarray(X)
    out = np.empty_like(X)
    T = X.shape[0]
    for j, s in enumerate(np.asarray(shifts, dtype=int)):
        out[:, j] = np.roll(X[:, j], int(s) % T)
    return out


def circular_shift_rows(Z: np.ndarray, shift: int) -> np.ndarray:
    return np.roll(np.asarray(Z), int(shift), axis=0)
