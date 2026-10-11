"""Reusable latent cache for Report 2 v2 (secondary targets share fits).

Layout under ``<session_out>/latent_cache/``::

    meta.json
    final/<method>.npz          # Z (n_times, d_sel), optional Z_fit, primary_d
    folds/<method>/fold_{i}.npz  # Z (n_times, d_fit), fit_mask, d_fit

``d`` selection for a new target reuses fold ``Z`` + ``fit_mask`` (target-free).
Final ``Z`` is reused for readout / EMA / derived controls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CACHE_DIRNAME = "latent_cache"


def cache_root(session_out: Path) -> Path:
    return Path(session_out) / CACHE_DIRNAME


def cap_latent_dims(
    latent_dims: list[int] | tuple[int, ...],
    n_units: int,
) -> tuple[list[int], int]:
    """Restrict the d grid to ``d <= ⌊n_units / 2⌋``; nested_fit_d = max(grid)."""
    half = int(n_units) // 2
    if half < 1:
        raise ValueError(f"n_units={n_units} too small for a latent grid")
    capped = sorted({int(d) for d in latent_dims if 1 <= int(d) <= half})
    if not capped:
        capped = [min(half, 2)]
    return capped, int(max(capped))


def write_meta(root: Path, meta: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def read_meta(root: Path) -> dict[str, Any] | None:
    path = root / "meta.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def save_fold_latent(
    root: Path,
    method: str,
    fold_index: int,
    *,
    Z: np.ndarray,
    fit_mask: np.ndarray,
    d_fit: int,
) -> Path:
    """Save one fold fit. Filename encodes fold and d_fit for non-nested grids."""
    dest = root / "folds" / method
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"fold_{int(fold_index)}_d{int(d_fit)}.npz"
    np.savez_compressed(
        path,
        Z=np.asarray(Z, dtype=np.float32),
        fit_mask=np.asarray(fit_mask, dtype=bool),
        d_fit=np.asarray([int(d_fit)], dtype=np.int32),
        fold=np.asarray([int(fold_index)], dtype=np.int32),
    )
    return path


def load_fold_latent(
    root: Path, method: str, fold_index: int, d_fit: int,
) -> dict[str, Any]:
    path = root / "folds" / method / f"fold_{int(fold_index)}_d{int(d_fit)}.npz"
    blob = np.load(path)
    return {
        "Z": np.asarray(blob["Z"], dtype=float),
        "fit_mask": np.asarray(blob["fit_mask"], dtype=bool),
        "d_fit": int(blob["d_fit"][0]),
        "fold": int(blob["fold"][0]) if "fold" in blob.files else int(fold_index),
    }


def list_fold_files(root: Path, method: str) -> list[Path]:
    dest = root / "folds" / method
    if not dest.is_dir():
        return []
    return sorted(dest.glob("fold_*_d*.npz"))


def save_final_latent(
    root: Path,
    method: str,
    *,
    Z: np.ndarray,
    primary_d: int | None,
    ridge_alpha: float | None = None,
    Z_fit: np.ndarray | None = None,
    nested_fit_d: int | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    dest = root / "final"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"{method}.npz"
    # float64 for final Z so readout round-trips match in-memory fits.
    payload: dict[str, Any] = {
        "Z": np.asarray(Z, dtype=np.float64),
        "primary_d": np.asarray(
            [-1 if primary_d is None else int(primary_d)], dtype=np.int32,
        ),
    }
    if ridge_alpha is not None:
        payload["ridge_alpha"] = np.asarray([float(ridge_alpha)], dtype=np.float64)
    if Z_fit is not None:
        payload["Z_fit"] = np.asarray(Z_fit, dtype=np.float64)
    if nested_fit_d is not None:
        payload["nested_fit_d"] = np.asarray([int(nested_fit_d)], dtype=np.int32)
    np.savez_compressed(path, **payload)
    if extra:
        (dest / f"{method}.meta.json").write_text(
            json.dumps(extra, indent=2, default=str) + "\n"
        )
    return path


def load_final_latent(root: Path, method: str) -> dict[str, Any]:
    path = root / "final" / f"{method}.npz"
    blob = np.load(path)
    primary_raw = int(blob["primary_d"][0])
    out: dict[str, Any] = {
        "Z": np.asarray(blob["Z"], dtype=float),
        "primary_d": None if primary_raw < 0 else primary_raw,
    }
    if "ridge_alpha" in blob.files:
        out["ridge_alpha"] = float(blob["ridge_alpha"][0])
    if "Z_fit" in blob.files:
        out["Z_fit"] = np.asarray(blob["Z_fit"], dtype=float)
    if "nested_fit_d" in blob.files:
        out["nested_fit_d"] = int(blob["nested_fit_d"][0])
    return out


def ridge_predict_from_cached(
    Z: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Train Ridge on cached latents; return eval predictions (same as Phase 3)."""
    from realtime.quadrant_n5_run import _fit_predict_ridge

    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    return _fit_predict_ridge(
        Z[train_ok], y[train_ok], Z[eval_mask], float(alpha),
    )
