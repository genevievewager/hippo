"""Disk cache for expensive deterministic pipeline nodes.

Keys include true scientific dependencies. Fitted transforms also include
train/test identity so test information cannot leak into a reused fit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from realtime.pipeline_artifacts import (
    ArtifactOrigin,
    ObservationConfig,
    config_hash,
    read_provenance,
    write_provenance,
)

FEATURE_DATASET_SUBDIR = "feature_datasets"


@dataclass
class CacheLookup:
    hit: bool
    path: Path | None
    origin: ArtifactOrigin
    config_hash: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hit": bool(self.hit),
            "path": str(self.path) if self.path else None,
            "origin": self.origin.value,
            "config_hash": self.config_hash,
            "reason": self.reason,
        }


def feature_dataset_dir(comparison_root: Path) -> Path:
    return Path(comparison_root) / "models" / FEATURE_DATASET_SUBDIR


def feature_dataset_dirname(observation: ObservationConfig) -> str:
    return (
        f"{observation.feature_set}__{observation.feature_type}_"
        f"w{observation.window_ms:04d}ms_{observation.hash()}"
    )


def raw_feature_cache_key(observation: ObservationConfig) -> str:
    """Key for unfitted extractor output (no train/test leakage)."""
    return observation.hash()


def fitted_feature_cache_key(observation: ObservationConfig) -> str:
    """Key for F transforms that must be train-only."""
    return observation.fit_hash()


def save_feature_matrix(
    comparison_root: Path,
    observation: ObservationConfig,
    X: np.ndarray,
    timestamps: np.ndarray,
    feature_names: list[str],
    *,
    origin: ArtifactOrigin = ArtifactOrigin.COMPUTED,
) -> Path:
    root = feature_dataset_dir(comparison_root)
    dest = root / feature_dataset_dirname(observation)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / "features.npz",
        X=np.asarray(X, dtype=np.float32),
        timestamps=np.asarray(timestamps, dtype=float),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    meta = {
        "kind": "FeatureDataset",
        "observation": observation.to_dict(),
        "config_hash": observation.hash(),
        "n_samples": int(np.asarray(X).shape[0]),
        "n_features": int(np.asarray(X).shape[1]),
        "origin": origin.value,
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    write_provenance(dest, {
        "kind": "FeatureDataset",
        "config_hash": observation.hash(),
        "origin": origin.value,
        "observation": observation.to_dict(),
        "includes_fitted_transform": False,
        "train_frac": None,
    })
    return dest


def load_feature_matrix(
    comparison_root: Path,
    observation: ObservationConfig,
) -> tuple[np.ndarray, np.ndarray, list[str], Path] | None:
    dest = feature_dataset_dir(comparison_root) / feature_dataset_dirname(observation)
    npz_path = dest / "features.npz"
    if not npz_path.exists():
        # Backward compatible: match by window/feature_set without hash suffix.
        parent = feature_dataset_dir(comparison_root)
        if not parent.exists():
            return None
        prefix = f"{observation.feature_set}__{observation.feature_type}_w{observation.window_ms:04d}ms"
        for cand in sorted(parent.glob(prefix + "*")):
            npz = cand / "features.npz"
            if not npz.exists():
                continue
            prov = read_provenance(cand)
            stored = (prov.get("observation") or {}).get("window_s")
            if stored is not None and abs(float(stored) - observation.window_s) > 1e-9:
                continue
            dest = cand
            npz_path = npz
            break
        else:
            return None
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            X = np.asarray(data["X"], dtype=float)
            timestamps = np.asarray(data["timestamps"], dtype=float)
            names = [str(n) for n in data["feature_names"].tolist()]
    except (OSError, KeyError, ValueError):
        return None
    return X, timestamps, names, dest


def lookup_feature_matrix(
    comparison_root: Path,
    observation: ObservationConfig,
) -> CacheLookup:
    key = observation.hash()
    loaded = load_feature_matrix(comparison_root, observation)
    if loaded is None:
        return CacheLookup(
            hit=False,
            path=None,
            origin=ArtifactOrigin.COMPUTED,
            config_hash=key,
            reason="no matching FeatureDataset cache",
        )
    _, _, _, path = loaded
    prov = read_provenance(path)
    stored = str(prov.get("config_hash") or "")
    if stored and stored != key:
        return CacheLookup(
            hit=False,
            path=path,
            origin=ArtifactOrigin.UNKNOWN,
            config_hash=key,
            reason="cached hash does not match observation",
        )
    return CacheLookup(
        hit=True,
        path=path,
        origin=ArtifactOrigin.CACHE,
        config_hash=key,
        reason="loaded FeatureDataset from cache",
    )


def inspect_cache_entry(path: Path) -> dict[str, Any]:
    loc = Path(path)
    meta: dict[str, Any] = {}
    meta_path = loc / "meta.json" if loc.is_dir() else loc
    if meta_path.exists() and meta_path.suffix == ".json":
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            meta = {}
    prov = read_provenance(loc if loc.is_dir() else loc.parent)
    origin = prov.get("origin") or meta.get("origin") or "unknown"
    return {
        "path": str(loc),
        "origin": origin,
        "from_cache": origin == ArtifactOrigin.CACHE.value,
        "newly_computed": origin == ArtifactOrigin.COMPUTED.value,
        "config_hash": prov.get("config_hash") or meta.get("config_hash"),
        "provenance": prov,
        "meta": meta,
    }
