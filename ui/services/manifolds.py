"""Manifold diagnostics: reuse / write shared transform checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.decoder_comparison import ALL_TARGETS, PRIMARY_METRIC
from realtime.manifold_features import DEFAULT_ISOMAP_N_NEIGHBORS, make_feature_transformer
from realtime.neural_features.comparison import resolve_manifolds_arg
from realtime.search_space import ALL_EMBEDDING_TYPES, resolve_manifold_alias
from realtime.timing import extract_behavior_times, resolve_update_dt_s
from realtime.train_decoder import causal_train_test_split
from realtime.transform_cache import (
    discover_comparison_roots,
    ensure_comparison_root,
    find_manifold_transform_in_roots,
    preferred_comparison_root,
    read_transform_meta,
    save_manifold_transform_checkpoint,
    try_load_manifold,
)

FALLBACK_WINDOW_S = 0.250
FALLBACK_FEATURE_SET = "counts"
FALLBACK_COUNTS_EMBEDDING = "identity"
FALLBACK_MANIFOLD_EMBEDDING = "global_pca"
FALLBACK_N_COMPONENTS = 3

TARGET_LABELS: dict[str, str] = {
    "position": "Position",
    "speed": "Speed",
    "acceleration": "Acceleration",
    "head_direction": "Head direction",
    "distance_to_wall": "Distance to wall",
    "spatial_context": "Spatial context",
    "movement_state": "Movement state",
    "wall_distance_bin": "Wall-distance bin",
}


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


@dataclass
class EmbeddingWinner:
    """Best-accuracy counts or manifold spec for one behavioral target."""

    target: str
    embedding_type: str
    feature_set: str
    decode_window: float
    n_components: int
    n_neighbors: int | None
    decoder_name: str | None
    metric_name: str
    metric_value: float | None
    metric_direction: str
    from_metrics: bool
    is_counts: bool

    def caption(self) -> str:
        w_ms = int(round(self.decode_window * 1000))
        bits = [
            self.embedding_type,
            self.feature_set,
            f"W={w_ms} ms",
            f"k={self.n_components}",
        ]
        if self.n_neighbors:
            bits.append(f"nn={self.n_neighbors}")
        if self.decoder_name:
            bits.append(self.decoder_name)
        if self.metric_value is not None:
            bits.append(f"{self.metric_name}={self.metric_value:.3g}")
        if not self.from_metrics:
            bits.append("fallback")
        return " · ".join(bits)


def list_embedding_types() -> tuple[str, ...]:
    return tuple(ALL_EMBEDDING_TYPES)


def resolve_manifold_name(name: str) -> str:
    return resolve_manifold_alias(name)


def find_cached_manifold_transform(
    comparison_root: Path,
    embedding_type: str,
    *,
    feature_set: str = "counts",
    decode_window: float = 0.250,
    n_components: int = 3,
) -> Path | None:
    """Locate a saved manifold transform directory under a comparison root."""
    return find_manifold_transform_in_roots(
        [Path(comparison_root), *discover_comparison_roots(Path(comparison_root).parent)],
        feature_set=feature_set,
        embedding_type=embedding_type,
        decode_window=decode_window,
        n_components=n_components,
    )


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
    n_neighbors: int | None = None,
    max_samples: int = 2500,
    comparison_root: Path | None = None,
    persist: bool = True,
    train_frac: float = 0.70,
    force_refit: bool = False,
) -> ManifoldDiagnostics:
    """Fit or reuse a manifold; optionally checkpoint to shared transform cache.

    Prefer cached transforms under ``decoder_comparison/.../models/manifold_transforms``.
    New fits use the same causal train split as Decoder Benchmark (``train_frac``),
    then transform the full sampled session for visualization. When ``persist`` is
    True, newly fitted transforms are written into the shared cache so Benchmark
    can reuse them.
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

    from realtime.decoding_targets import align_extended_behavior_to_decoder_times

    beh_aligned = align_extended_behavior_to_decoder_times(
        behavior, decode_times, data.get("summary"),
    )

    from_cache = False
    persisted = False
    extras: dict[str, Any] = {
        "feature_set": feature_set,
        "n_neural_features": int(X.shape[1]),
    }
    explained: list[float] | None = None
    transformer = None
    nn = int(n_neighbors) if n_neighbors is not None else DEFAULT_ISOMAP_N_NEIGHBORS

    write_root = (
        Path(comparison_root)
        if comparison_root is not None
        else ensure_comparison_root(Path(input_dir), spike_source=spike_source)
    )
    roots: list[Path] = [write_root]
    preferred = preferred_comparison_root(Path(input_dir), spike_source=spike_source)
    if preferred is not None and preferred.resolve() != write_root.resolve():
        roots.append(preferred)
    for r in discover_comparison_roots(Path(input_dir)):
        if r.resolve() not in {p.resolve() for p in roots}:
            roots.append(r)

    cached = None
    if not force_refit:
        cached = find_manifold_transform_in_roots(
            roots,
            feature_set=feature_set,
            embedding_type=emb,
            decode_window=decode_window,
            n_components=n_components,
            n_neighbors=nn,
        )
    if cached is not None:
        loaded = try_load_manifold(cached)
        if loaded is not None:
            transformer = loaded
            from_cache = True
            extras["cached_path"] = str(cached)
            extras["fit_scope"] = read_transform_meta(cached).get("fit_scope")

    mode = "counts" if emb == "identity" else emb
    isomap_transform = "sqrt_counts" if feature_set == "counts" else "counts"
    if transformer is None:
        transformer = make_feature_transformer(
            mode,
            decode_window=decode_window,
            n_components=n_components,
            units_df=data["units_df"],
            unit_ids=data["unit_ids"],
            isomap_transform=isomap_transform,
            n_neighbors=nn,
            update_dt=update_dt,
            feature_set=feature_set,
            spike_source=spike_source,
        )
        if transformer is None:
            raise RuntimeError(f"Could not build transformer for {embedding_type}")
        if hasattr(transformer, "fit"):
            train_mask, _test_mask = causal_train_test_split(decode_times, train_frac)
            if int(train_mask.sum()) < 2:
                train_mask = np.ones(len(decode_times), dtype=bool)
            X_fit = X[train_mask]
            y = None
            if "x" in beh_aligned.columns and "y" in beh_aligned.columns:
                y = beh_aligned.loc[train_mask, ["x", "y"]].to_numpy()
            try:
                if y is not None and emb in ("pls", "bayesian_place_tuning"):
                    transformer.fit(X_fit, y)
                else:
                    transformer.fit(X_fit)
            except TypeError:
                transformer.fit(X_fit)
            extras["fit_scope"] = "train_split"
            extras["train_frac"] = float(train_frac)
            extras["n_train_samples"] = int(train_mask.sum())

        if persist and hasattr(transformer, "save"):
            try:
                saved = save_manifold_transform_checkpoint(
                    transformer,
                    write_root,
                    feature_set=feature_set,
                    embedding_type=emb,
                    decode_window=decode_window,
                    n_components=n_components,
                    n_neighbors=nn,
                    feature_type="counts",
                    extra_meta={
                        "fit_scope": "train_split",
                        "train_frac": float(train_frac),
                        "source": "manifold_explorer",
                        "spike_source": spike_source,
                        "n_train_samples": extras.get("n_train_samples"),
                    },
                )
                persisted = True
                extras["saved_path"] = str(saved)
                extras["cached_path"] = str(saved)
            except Exception as exc:  # noqa: BLE001
                extras["persist_error"] = str(exc)

    if emb in ("global_lds", "gpfa") and hasattr(transformer, "transform"):
        try:
            latent = np.asarray(
                transformer.transform(X, causal=(emb != "gpfa"), reset=True),
                dtype=float,
            )
        except TypeError:
            latent = np.asarray(transformer.transform(X), dtype=float)
    else:
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
    extras["persisted"] = persisted
    extras["comparison_root"] = str(write_root)

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
    """Manifold tokens for widgets (static + dynamic)."""
    from ui.services.representations import UI_DYNAMIC_LATENT_OPTIONS, UI_STATIC_MANIFOLD_OPTIONS

    return list(UI_STATIC_MANIFOLD_OPTIONS) + list(UI_DYNAMIC_LATENT_OPTIONS)


def parse_manifold_list(names: list[str]) -> tuple[str, ...]:
    resolved = resolve_manifolds_arg(names)
    return resolved or ()


def _metrics_for_source(metrics: pd.DataFrame, spike_source: str) -> pd.DataFrame:
    if metrics.empty or "spike_source" not in metrics.columns:
        return metrics
    sub = metrics[metrics["spike_source"].astype(str) == str(spike_source)]
    return sub if not sub.empty else metrics


def _best_metric_row(df: pd.DataFrame, target: str) -> pd.Series | None:
    if df.empty:
        return None
    metric, direction = PRIMARY_METRIC.get(target, (None, "higher"))
    if not metric or metric not in df.columns:
        return None
    valid = df.dropna(subset=[metric])
    if valid.empty:
        return None
    idx = valid[metric].idxmin() if direction == "lower" else valid[metric].idxmax()
    return valid.loc[idx]


def _fallback_winner(target: str, *, is_counts: bool) -> EmbeddingWinner:
    metric, direction = PRIMARY_METRIC.get(target, ("score", "higher"))
    return EmbeddingWinner(
        target=target,
        embedding_type=FALLBACK_COUNTS_EMBEDDING if is_counts else FALLBACK_MANIFOLD_EMBEDDING,
        feature_set=FALLBACK_FEATURE_SET,
        decode_window=FALLBACK_WINDOW_S,
        n_components=FALLBACK_N_COMPONENTS,
        n_neighbors=None,
        decoder_name=None,
        metric_name=metric,
        metric_value=None,
        metric_direction=direction,
        from_metrics=False,
        is_counts=is_counts,
    )


def _winner_from_row(row: pd.Series, target: str, *, is_counts: bool) -> EmbeddingWinner:
    metric, direction = PRIMARY_METRIC.get(target, ("score", "higher"))
    emb = row.get("embedding_type")
    if emb is None or (isinstance(emb, float) and pd.isna(emb)) or str(emb) in {"", "nan"}:
        emb = row.get("feature_mode") or row.get("feature_type") or "identity"
    emb = resolve_manifold_name(str(emb))
    if is_counts:
        emb = "identity"
    fs = "counts"
    if "feature_set" in row.index and pd.notna(row.get("feature_set")):
        fs = str(row.get("feature_set"))
    elif str(row.get("feature_type") or "") in ("counts", "rates"):
        fs = str(row.get("feature_type"))
    k = row.get("manifold_n_components")
    n_comp = int(k) if k is not None and pd.notna(k) else FALLBACK_N_COMPONENTS
    nn = row.get("n_neighbors")
    n_neighbors = int(nn) if nn is not None and pd.notna(nn) else None
    value = row.get(metric)
    decoder = row.get("decoder_name")
    return EmbeddingWinner(
        target=target,
        embedding_type=emb,
        feature_set=fs or FALLBACK_FEATURE_SET,
        decode_window=float(row.get("decode_window_s") or FALLBACK_WINDOW_S),
        n_components=max(n_comp, 2),
        n_neighbors=n_neighbors,
        decoder_name=(
            None if decoder is None or (isinstance(decoder, float) and pd.isna(decoder))
            else str(decoder)
        ),
        metric_name=metric,
        metric_value=(
            None if value is None or (isinstance(value, float) and pd.isna(value))
            else float(value)
        ),
        metric_direction=direction,
        from_metrics=True,
        is_counts=is_counts,
    )


def best_counts_and_manifold_winners(
    metrics: pd.DataFrame | None,
    target: str,
    *,
    spike_source: str = "sorted",
) -> tuple[EmbeddingWinner, EmbeddingWinner]:
    """Best-accuracy counts and manifold specs among available comparison rows."""
    from realtime.manifold_summaries import _is_counts_like_row, _is_manifold_row

    if metrics is None or metrics.empty or target not in ALL_TARGETS:
        return (
            _fallback_winner(target, is_counts=True),
            _fallback_winner(target, is_counts=False),
        )

    src = _metrics_for_source(metrics, spike_source)
    if "target_name" in src.columns:
        src = src[src["target_name"] == target]
    elif "target" in src.columns:
        src = src[src["target"].astype(str) == target]
    counts_row = (
        _best_metric_row(src[_is_counts_like_row(src)], target) if not src.empty else None
    )
    man_row = (
        _best_metric_row(src[_is_manifold_row(src)], target) if not src.empty else None
    )
    counts = (
        _winner_from_row(counts_row, target, is_counts=True)
        if counts_row is not None
        else _fallback_winner(target, is_counts=True)
    )
    manifold = (
        _winner_from_row(man_row, target, is_counts=False)
        if man_row is not None
        else _fallback_winner(target, is_counts=False)
    )
    return counts, manifold


def winner_summary_rows(
    metrics: pd.DataFrame | None,
    *,
    spike_source: str = "sorted",
) -> list[dict[str, Any]]:
    """One summary row per behavioral target (counts vs best manifold)."""
    rows = []
    for target in ALL_TARGETS:
        counts, man = best_counts_and_manifold_winners(
            metrics, target, spike_source=spike_source,
        )
        rows.append({
            "target": TARGET_LABELS.get(target, target),
            "counts_W_ms": int(round(counts.decode_window * 1000)),
            "counts_decoder": counts.decoder_name or "—",
            "counts_metric": (
                f"{counts.metric_value:.3g}" if counts.metric_value is not None else "—"
            ),
            "manifold": man.embedding_type,
            "manifold_W_ms": int(round(man.decode_window * 1000)),
            "manifold_decoder": man.decoder_name or "—",
            "manifold_metric": (
                f"{man.metric_value:.3g}" if man.metric_value is not None else "—"
            ),
            "source": "metrics" if (counts.from_metrics or man.from_metrics) else "fallback",
        })
    return rows
