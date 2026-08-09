"""Model metadata helpers for dynamic latent representations."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any


def try_git_commit(repo_root: Path | None = None) -> str | None:
    """Best-effort short git SHA for reproducibility metadata."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except Exception:
        return None


def build_model_metadata(
    *,
    model_type: str,
    representation_family: str,
    latent_dimension: int | None,
    feature_set: str | None = None,
    decode_window: float | None = None,
    update_dt: float | None = None,
    spike_source: str | None = None,
    training_interval: dict[str, Any] | None = None,
    causal_status: str = "causal",
    random_seed: int | None = None,
    hyperparameters: dict[str, Any] | None = None,
    training_timestamp: str | None = None,
    git_commit: str | None = None,
    software_version: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the common reproducibility metadata block."""
    meta: dict[str, Any] = {
        "model_type": model_type,
        "representation_family": representation_family,
        "latent_dimension": latent_dimension,
        "feature_set": feature_set,
        "decode_window": decode_window,
        "update_dt": update_dt,
        "spike_source": spike_source,
        "training_interval": training_interval,
        "causal_status": causal_status,
        "random_seed": random_seed,
        "hyperparameters": hyperparameters or {},
        "training_timestamp": training_timestamp,
        "software_version": software_version,
        "git_commit": git_commit,
    }
    if extra:
        meta.update(extra)
    return meta
