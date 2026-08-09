"""Manifold diagnostics: reuse saved transforms when present, else fit on demand."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.manifold_features import make_feature_transformer
from realtime.neural_features.comparison import resolve_manifolds_arg
from realtime.search_space import ALL_EMBEDDING_TYPES, resolve_manifold_alias
from realtime.timing import extract_behavior_times, resolve_update_dt_s


@dataclass
class ManifoldDiagnostics:
    embedding_type: str
    n_components: int
    latent: np.ndarray
    times: np.ndarray
    behavior: pd.DataFrame
    explained_variance_ratio: list[float] | None
    extras: dict[str, Any]
    from_cache: bool


def list_embedding_types() -> tuple[str, ...]:
    return tuple(ALL_EMBEDDING_TYPES)


def resolve_manifold_name(name: str) -> str:
    return resolve_manifold_alias(name)


def find_cached_manifold_transform(
    comparison_root: Path,
    embedding_type: str,
) -> Path | None:
    """Best-effort locate a saved manifold transform joblib."""
    root = Path(comparison_root)
    candidates = [
        root / "models" / "manifold_transforms",
        root / "manifold_transforms",
    ]
    emb = resolve_manifold_name(embedding_type)
    for base in candidates:
        if not base.exists():
            continue
        matches = sorted(base.rglob("*.joblib"))
        for path in matches:
            key = path.as_posix().lower()
            if emb.replace("_", "") in key.replace("_", "") or emb in key:
                return path
        # Fallback: first joblib if only one exists
        if len(matches) == 1:
            return matches[0]
    return None


def load_explained_variance_table(comparison_root: Path) -> pd.DataFrame | None:
    path = Path(comparison_root) / "manifold_explained_variance.csv"
    if path.exists():
        return pd.read_csv(path)
    nested = list(Path(comparison_root).rglob("manifold_explained_variance.csv"))
    if nested:
        return pd.read_csv(nested[0])
    return None


def compute_manifold_diagnostics(
    input_dir,
    embedding_type: str,
    *,
    feature_set: str = "counts",
    spike_source: str = "sorted",
    decode_window: float = 0.250,
    n_components: int = 3,
    max_samples: int = 2500,
    comparison_root: Path | None = None,
) -> ManifoldDiagnostics:
    """Fit or reuse a manifold on a neural feature set; return latents + behavior.

    Does not write comparison artifacts. Prefer cached transforms when a
    ``comparison_root`` is supplied and a matching joblib exists.

    ``feature_set`` selects the neural feature extractor (e.g. ``counts``,
    ``counts_dynamics``) that is fed into the manifold encoder — matching the
    decoder-comparison feature×embedding path.
    """
    from realtime.neural_features import (
        NeuralFeatureExtractor,
        embedding_compatible_with_feature_set,
    )

    emb = resolve_manifold_name(embedding_type)
    if emb not in ALL_EMBEDDING_TYPES and emb not in ("counts",):
        raise ValueError(f"Unknown embedding type: {embedding_type}")
    if not embedding_compatible_with_feature_set(emb, feature_set):
        raise ValueError(
            f"Embedding `{emb}` is incompatible with feature set `{feature_set}`"
        )

    data = load_simulation_data(input_dir, spike_source)
    behavior = data["behavior_df"]
    behavior_times = extract_behavior_times(behavior)
    update_dt = resolve_update_dt_s(
        data["summary"],
        derive_from_behavior=True,
        update_dt_s=0.050,
        behavior_times=behavior_times,
    )
    decode_times = make_decode_times(
        data["session_duration"],
        decode_window,
        update_dt,
        behavior_times=behavior_times,
    )
    if len(decode_times) > max_samples:
        idx = np.linspace(0, len(decode_times) - 1, max_samples).astype(int)
        decode_times = decode_times[idx]

    extractor = NeuralFeatureExtractor.from_feature_set(
        feature_set,
        units_df=data["units_df"],
        unit_ids=data["unit_ids"],
        decode_window=decode_window,
        update_dt=update_dt,
    )
    extracted = extractor.extract_matrix(data["spikes_df"], decode_times)
    X = np.asarray(extracted.feature_vector, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise RuntimeError(
            f"Empty feature matrix for feature_set={feature_set!r} "
            f"(n_unit_ids={len(data['unit_ids'])})."
        )

    # Canonical decoding targets (x/y/speed/HD/…), not raw behavior.csv names
    # like x_cm / speed_cm_s — latent-geometry color pages require these.
    from realtime.decoding_targets import align_extended_behavior_to_decoder_times

    beh_aligned = align_extended_behavior_to_decoder_times(
        behavior, decode_times, data.get("summary"),
    )

    from_cache = False
    extras: dict[str, Any] = {
        "feature_set": feature_set,
        "n_neural_features": int(X.shape[1]),
    }
    explained: list[float] | None = None
    transformer = None

    if comparison_root is not None and feature_set == "counts":
        # Cached transforms from decoder comparison are counts-based.
        cached = find_cached_manifold_transform(comparison_root, emb)
        if cached is not None:
            try:
                transformer = joblib.load(cached)
                from_cache = True
                extras["cached_path"] = str(cached)
            except Exception as exc:  # noqa: BLE001
                extras["cache_load_error"] = str(exc)
                transformer = None

    mode = "counts" if emb == "identity" else emb
    # Signed / mixed neural feature sets must not use sqrt-counts preprocessing.
    isomap_transform = "sqrt_counts" if feature_set == "counts" else "counts"
    if transformer is None:
        transformer = make_feature_transformer(
            mode,
            decode_window=decode_window,
            n_components=n_components,
            units_df=data["units_df"],
            unit_ids=data["unit_ids"],
            isomap_transform=isomap_transform,
        )
        if transformer is None:
            raise RuntimeError(f"Could not build transformer for {embedding_type}")
        if hasattr(transformer, "fit"):
            y = None
            if "x" in beh_aligned.columns and "y" in beh_aligned.columns:
                y = beh_aligned[["x", "y"]].to_numpy()
            try:
                if y is not None and emb in ("pls", "bayesian_place_tuning"):
                    transformer.fit(X, y)
                else:
                    transformer.fit(X)
            except TypeError:
                transformer.fit(X)

    latent = np.asarray(transformer.transform(X), dtype=float)
    if hasattr(transformer, "explained_variance_ratio_"):
        ev = getattr(transformer, "explained_variance_ratio_")
        if ev is not None:
            explained = np.asarray(ev, dtype=float).tolist()
    if hasattr(transformer, "get_diagnostics"):
        try:
            extras["diagnostics"] = transformer.get_diagnostics()
        except Exception as exc:  # noqa: BLE001
            extras["diagnostics_error"] = str(exc)
    meta = getattr(transformer, "get_metadata", None)
    if callable(meta):
        try:
            extras["metadata"] = meta()
        except Exception:
            pass

    return ManifoldDiagnostics(
        embedding_type=emb,
        n_components=int(latent.shape[1]) if latent.ndim == 2 else n_components,
        latent=latent,
        times=decode_times,
        behavior=beh_aligned,
        explained_variance_ratio=explained,
        extras=extras,
        from_cache=from_cache,
    )


def ui_manifold_choices() -> list[str]:
    """CLI-friendly manifold tokens for widgets (static + dynamic)."""
    from ui.services.representations import UI_DYNAMIC_LATENT_OPTIONS, UI_STATIC_MANIFOLD_OPTIONS

    return list(UI_STATIC_MANIFOLD_OPTIONS) + list(UI_DYNAMIC_LATENT_OPTIONS)


def parse_manifold_list(names: list[str]) -> tuple[str, ...]:
    resolved = resolve_manifolds_arg(names)
    return resolved or ()
