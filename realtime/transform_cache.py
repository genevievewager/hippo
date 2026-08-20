"""Locate and load previously fitted neural / F / E transforms.

Used by decoder comparison (reuse while still retraining decoder heads) and
by UI explorers so manifold/feature pages avoid redundant fits.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realtime.manifold_features import (
    load_feature_transformer,
    manifold_transform_dirname,
)
from realtime.search_space import compose_feature_mode, resolve_manifold_alias


@dataclass(frozen=True)
class TransformHit:
    """A reusable on-disk transform directory."""

    path: Path
    kind: str  # "manifold" | "feature" | "neural"
    reused: bool = True


def comparison_model_roots(comparison_root: Path | None) -> dict[str, Path]:
    """Return standard model subdirs under a comparison output root."""
    root = Path(comparison_root) if comparison_root is not None else None
    if root is None:
        return {}
    models = root / "models"
    return {
        "root": root,
        "models": models,
        "manifold": models / "manifold_transforms",
        "feature": models / "feature_transforms",
        "neural": models / "neural_feature_extractors",
    }


def discover_comparison_roots(experiment_dir: Path) -> list[Path]:
    """Find decoder_comparison output trees that contain model transforms."""
    exp = Path(experiment_dir)
    candidates = [
        exp / "decoder_comparison" / "sorted",
        exp / "decoder_comparison" / "dynamic",
        exp / "decoder_comparison" / "ground_truth",
        exp / "decoder_comparison",
    ]
    found: list[Path] = []
    seen: set[Path] = set()
    for root in candidates:
        man = root / "models" / "manifold_transforms"
        if man.exists() and root.resolve() not in seen:
            found.append(root)
            seen.add(root.resolve())
    # Any nested comparison tree with manifold_transforms
    for man in sorted(exp.glob("decoder_comparison/**/models/manifold_transforms")):
        root = man.parent.parent
        if root.resolve() not in seen:
            found.append(root)
            seen.add(root.resolve())
    return found


def preferred_comparison_root(
    experiment_dir: Path,
    *,
    family: str | None = None,
    spike_source: str = "sorted",
) -> Path | None:
    """Pick the most relevant comparison root for reuse."""
    roots = discover_comparison_roots(experiment_dir)
    if not roots:
        return None
    preferred_names: list[str] = []
    if family == "dynamic":
        preferred_names.append("dynamic")
    if spike_source:
        preferred_names.append(spike_source)
    preferred_names.extend(["sorted", "dynamic", "ground_truth"])
    for name in preferred_names:
        for root in roots:
            if root.name == name:
                return root
    return roots[0]


def window_ms(decode_window: float) -> int:
    return int(round(float(decode_window) * 1000))


_WINDOW_DIR_RE = re.compile(r"_w(\d+)ms$", re.IGNORECASE)

_FEATURE_CONSTRUCTION_WINDOW_MSG = (
    "Decode window {label} has no Feature Construction cache. "
    "Generate windows on the Feature Construction page first."
)


def _format_window_ms(ms: int) -> str:
    if ms >= 1000 and ms % 1000 == 0:
        return f"{ms // 1000} s"
    return f"{ms} ms"


def _feature_transform_dirs(
    experiment_dir: Path,
    spike_source: str = "sorted",
) -> list[Path]:
    """``feature_transforms`` folders Decoder / Latent Representations search."""
    exp = Path(experiment_dir)
    source = str(spike_source or "sorted").strip() or "sorted"
    feat_dirs: list[Path] = []
    direct = exp / "decoder_comparison" / source / "models" / "feature_transforms"
    if direct.exists():
        feat_dirs.append(direct)
    flat = exp / "decoder_comparison" / "models" / "feature_transforms"
    if flat.exists() and flat not in feat_dirs:
        feat_dirs.append(flat)
    skip_roots = {"ground_truth", "dynamic", "sorted"}
    for feat in sorted(exp.glob("decoder_comparison/**/models/feature_transforms")):
        if feat in feat_dirs:
            continue
        root = feat.parent.parent
        if root.name in skip_roots and root.name != source:
            continue
        feat_dirs.append(feat)
    return feat_dirs


def comparison_roots_for_feature_cache(
    experiment_dir: Path,
    spike_source: str = "sorted",
) -> list[Path]:
    """Comparison roots that contain F caches for ``spike_source``."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for feat_dir in _feature_transform_dirs(experiment_dir, spike_source):
        root = feat_dir.parent.parent
        key = root.resolve() if root.exists() else root
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def inventory_feature_construction_cache(
    experiment_dir: Path,
    *,
    feature_sets: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...],
    spike_source: str = "sorted",
    feature_type: str = "counts",
) -> dict[str, Any]:
    """Coverage of F caches that downstream pages actually load.

    Keys match Decoder Benchmark: ``feature_set`` × effective F type × ``W``.
    Run JSON / Feature Construction summaries are ignored.
    """
    from realtime.neural_features.comparison import effective_spike_feature_type

    wanted: set[tuple[str, float]] = {
        (str(fs), float(w))
        for fs in feature_sets
        for w in decode_windows
    }
    empty = {
        "n_wanted": 0,
        "n_covered": 0,
        "n_to_compute": 0,
        "covered": set(),
        "missing": set(),
        "matching_run_ids": [],
        "fully_covered": False,
        "hits": [],
        "source": "feature_transforms",
    }
    if not wanted:
        return empty

    roots = comparison_roots_for_feature_cache(experiment_dir, spike_source)
    covered: set[tuple[str, float]] = set()
    hits: list[dict[str, Any]] = []
    for fs, w in sorted(wanted, key=lambda p: (p[0], p[1])):
        f_eff = effective_spike_feature_type(str(fs), feature_type)
        path = find_feature_transform_in_roots(
            roots,
            feature_set=str(fs),
            feature_type_eff=str(f_eff),
            decode_window=float(w),
        )
        if path is None:
            continue
        covered.add((str(fs), float(w)))
        hits.append({
            "feature_set": str(fs),
            "feature_type_eff": str(f_eff),
            "decode_window_s": float(w),
            "path": str(path),
        })

    missing = wanted - covered
    return {
        "n_wanted": len(wanted),
        "n_covered": len(covered),
        "n_to_compute": len(missing),
        "covered": covered,
        "missing": missing,
        "matching_run_ids": [],
        "fully_covered": not missing and len(wanted) > 0,
        "hits": hits,
        "source": "feature_transforms",
    }


def list_cached_decode_windows(
    experiment_dir: Path,
    spike_source: str = "sorted",
    feature_sets: list[str] | tuple[str, ...] | None = None,
    feature_type: str = "counts",
) -> list[float]:
    """Return sorted unique W values that have F caches (feature_transforms).

    Does not require manifold transforms. Scans
    ``decoder_comparison/<spike_source>/models/feature_transforms/`` and any
    nested ``**/models/feature_transforms`` whose comparison root name matches
    ``spike_source`` (or is the flat ``decoder_comparison`` root).

    When ``feature_sets`` is given (non-empty), a window is listed only if
    **every** named set has an F cache at that W.
    """
    found_ms: set[int] = set()
    for feat_dir in _feature_transform_dirs(experiment_dir, spike_source):
        for meta_path in feat_dir.glob("*/meta.json"):
            ms: int | None = None
            try:
                meta = json.loads(meta_path.read_text())
                raw = meta.get("decode_window_s")
                if raw is not None:
                    ms = window_ms(float(raw))
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                ms = None
            if ms is None:
                match = _WINDOW_DIR_RE.search(meta_path.parent.name)
                if match:
                    ms = int(match.group(1))
            if ms is not None:
                found_ms.add(ms)
    windows = [ms / 1000.0 for ms in sorted(found_ms)]
    wanted = tuple(str(fs) for fs in (feature_sets or ()) if str(fs).strip())
    if not wanted:
        return windows
    from realtime.neural_features.comparison import effective_spike_feature_type

    roots = comparison_roots_for_feature_cache(experiment_dir, spike_source)
    kept: list[float] = []
    for w in windows:
        if all(
            find_feature_transform_in_roots(
                roots,
                feature_set=fs,
                feature_type_eff=effective_spike_feature_type(fs, feature_type),
                decode_window=float(w),
            ) is not None
            for fs in wanted
        ):
            kept.append(float(w))
    return kept


def assert_cached_decode_windows(
    experiment_dir: Path,
    decode_windows: list[float] | tuple[float, ...],
    *,
    spike_source: str = "sorted",
) -> None:
    """Raise ValueError if any requested W is missing from Feature Construction caches."""
    cached = list_cached_decode_windows(experiment_dir, spike_source=spike_source)
    cached_ms = {window_ms(w) for w in cached}
    missing = [
        float(w) for w in decode_windows
        if window_ms(w) not in cached_ms
    ]
    if not missing:
        return
    labels = ", ".join(_format_window_ms(window_ms(w)) for w in missing)
    raise ValueError(
        _FEATURE_CONSTRUCTION_WINDOW_MSG.format(label=labels)
    )


def neural_extractor_dirname(feature_set: str, decode_window: float) -> str:
    return f"{feature_set}_w{window_ms(decode_window):04d}ms"


def feature_transform_dirname(
    feature_set: str,
    feature_type_eff: str,
    decode_window: float,
) -> str:
    return (
        f"{feature_set}__{feature_type_eff}_w"
        f"{window_ms(decode_window):04d}ms"
    )


def _isomap_like_mode(feature_mode: str) -> bool:
    return (
        feature_mode in ("global_isomap", "global_isomap_distilled")
        or feature_mode.endswith("_isomap")
        or feature_mode.endswith("_isomap_distilled")
    )


def _diffusion_like_mode(feature_mode: str) -> bool:
    return (
        feature_mode == "diffusion_nystrom"
        or feature_mode.endswith("_diffusion_nystrom")
        or "__diffusion_nystrom" in feature_mode
    )


def manifold_candidate_dirnames(
    *,
    feature_set: str,
    feature_mode: str,
    decode_window: float,
    n_components: int | None,
    n_neighbors: int | None = None,
    target: str | None = None,
) -> list[str]:
    """New naming first, then legacy (no feature_set prefix)."""
    from realtime.manifold_features import DEFAULT_ISOMAP_N_NEIGHBORS, DEFAULT_N_LANDMARKS

    mode = feature_mode
    if target:
        mode = f"{feature_mode}_{target}"
    neighbor_candidates: list[int | None] = [n_neighbors]
    # Saves often embed default nn; inventory/UI lookups may omit it.
    if n_neighbors is None and _isomap_like_mode(mode):
        neighbor_candidates.append(DEFAULT_ISOMAP_N_NEIGHBORS)
    if n_neighbors is None and _diffusion_like_mode(mode):
        neighbor_candidates.append(DEFAULT_N_LANDMARKS)

    names: list[str] = []
    for nn in neighbor_candidates:
        modern = manifold_transform_dirname(
            f"{feature_set}__{mode}",
            decode_window,
            n_components,
            nn,
        )
        legacy = manifold_transform_dirname(
            mode,
            decode_window,
            n_components,
            nn,
        )
        for n in (modern, legacy):
            if n not in names:
                names.append(n)
    # Identity sometimes stored as counts_w####ms under legacy layouts.
    if mode in ("counts", "identity") and n_components is not None:
        names.append(f"counts_w{window_ms(decode_window):04d}ms")
    return names


def ensure_comparison_root(
    experiment_dir: Path,
    *,
    spike_source: str = "sorted",
    family: str | None = None,
) -> Path:
    """Return a decoder_comparison root, creating the standard layout if needed."""
    preferred = preferred_comparison_root(
        Path(experiment_dir), family=family, spike_source=spike_source,
    )
    if preferred is not None:
        return preferred
    source = spike_source or "sorted"
    root = Path(experiment_dir) / "decoder_comparison" / source
    (root / "models" / "manifold_transforms").mkdir(parents=True, exist_ok=True)
    (root / "models" / "feature_transforms").mkdir(parents=True, exist_ok=True)
    (root / "models" / "neural_feature_extractors").mkdir(parents=True, exist_ok=True)
    return root


def manifold_transform_path(
    comparison_root: Path,
    *,
    feature_set: str,
    embedding_type: str,
    decode_window: float,
    n_components: int | None = 3,
    n_neighbors: int | None = None,
    feature_type: str = "counts",
    target: str | None = None,
) -> Path:
    """Canonical write path for a manifold transform (modern naming)."""
    from realtime.manifold_features import DEFAULT_ISOMAP_N_NEIGHBORS, DEFAULT_N_LANDMARKS

    emb = resolve_manifold_alias(embedding_type)
    feature_mode = compose_feature_mode(feature_type, emb)
    mode = f"{feature_mode}_{target}" if target else feature_mode
    nn = n_neighbors
    if nn is None and _isomap_like_mode(mode):
        nn = DEFAULT_ISOMAP_N_NEIGHBORS
    if nn is None and _diffusion_like_mode(mode):
        nn = DEFAULT_N_LANDMARKS
    name = manifold_transform_dirname(
        f"{feature_set}__{mode}",
        decode_window,
        n_components,
        nn,
    )
    roots = comparison_model_roots(comparison_root)
    return roots["manifold"] / name


def enrich_transform_meta(path: Path, extra: dict[str, Any]) -> dict[str, Any]:
    """Merge keys into ``meta.json`` without dropping existing fields."""
    path = Path(path)
    meta = read_transform_meta(path)
    meta.update({k: v for k, v in extra.items() if v is not None})
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return meta


def save_manifold_transform_checkpoint(
    transformer: Any,
    comparison_root: Path,
    *,
    feature_set: str,
    embedding_type: str,
    decode_window: float,
    n_components: int | None = 3,
    n_neighbors: int | None = None,
    feature_type: str = "counts",
    target: str | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted manifold into the shared decoder_comparison cache."""
    if transformer is None or not hasattr(transformer, "save"):
        raise TypeError("transformer must implement save(output_dir)")
    path = manifold_transform_path(
        comparison_root,
        feature_set=feature_set,
        embedding_type=embedding_type,
        decode_window=decode_window,
        n_components=n_components,
        n_neighbors=n_neighbors,
        feature_type=feature_type,
        target=target,
    )
    path.mkdir(parents=True, exist_ok=True)
    transformer.save(path)
    emb = resolve_manifold_alias(embedding_type)
    enrich_transform_meta(
        path,
        {
            "feature_set": feature_set,
            "embedding_type": emb,
            "feature_type": feature_type,
            "decode_window_s": float(decode_window),
            "n_components": n_components,
            "n_neighbors": n_neighbors,
            **(extra_meta or {}),
        },
    )
    return path


def _is_usable_transform_dir(path: Path) -> bool:
    return path.is_dir() and (path / "meta.json").exists()


def find_transform_dir(base: Path, candidates: list[str]) -> Path | None:
    base = Path(base)
    if not base.exists():
        return None
    for name in candidates:
        path = base / name
        if _is_usable_transform_dir(path):
            return path
    return None


def find_manifold_transform(
    comparison_root: Path,
    *,
    feature_set: str,
    embedding_type: str,
    decode_window: float,
    n_components: int | None = 3,
    n_neighbors: int | None = None,
    feature_type: str = "counts",
    target: str | None = None,
) -> Path | None:
    """Resolve a saved manifold transform dir (modern or legacy naming)."""
    emb = resolve_manifold_alias(embedding_type)
    feature_mode = compose_feature_mode(feature_type, emb)
    roots = comparison_model_roots(comparison_root)
    man_base = roots.get("manifold")
    if man_base is None:
        return None
    return find_transform_dir(
        man_base,
        manifold_candidate_dirnames(
            feature_set=feature_set,
            feature_mode=feature_mode,
            decode_window=decode_window,
            n_components=n_components,
            n_neighbors=n_neighbors,
            target=target,
        ),
    )


def find_manifold_transform_in_roots(
    comparison_roots: list[Path] | tuple[Path, ...],
    **kwargs: Any,
) -> Path | None:
    """Search multiple comparison roots; first hit wins."""
    for root in comparison_roots:
        if root is None:
            continue
        hit = find_manifold_transform(Path(root), **kwargs)
        if hit is not None:
            return hit
    return None


def find_feature_transform(
    comparison_root: Path,
    *,
    feature_set: str,
    feature_type_eff: str,
    decode_window: float,
) -> Path | None:
    roots = comparison_model_roots(comparison_root)
    base = roots.get("feature")
    if base is None:
        return None
    name = feature_transform_dirname(feature_set, feature_type_eff, decode_window)
    path = base / name
    return path if _is_usable_transform_dir(path) else None


def find_feature_transform_in_roots(
    comparison_roots: list[Path] | tuple[Path, ...],
    **kwargs: Any,
) -> Path | None:
    for root in comparison_roots:
        if root is None:
            continue
        hit = find_feature_transform(Path(root), **kwargs)
        if hit is not None:
            return hit
    return None


def find_neural_extractor(
    comparison_root: Path,
    *,
    feature_set: str,
    decode_window: float,
) -> Path | None:
    roots = comparison_model_roots(comparison_root)
    base = roots.get("neural")
    if base is None:
        return None
    path = base / neural_extractor_dirname(feature_set, decode_window)
    # Neural extractors use neural_feature_extractor.joblib
    if path.is_dir() and (
        (path / "meta.json").exists()
        or (path / "neural_feature_extractor.joblib").exists()
    ):
        return path
    return None


def try_load_manifold(path: Path) -> Any | None:
    try:
        return load_feature_transformer(path)
    except Exception:
        return None


def try_load_feature_transform(path: Path) -> Any | None:
    try:
        from realtime.feature_representations import SpikeFeatureTransformer

        return SpikeFeatureTransformer.load(path)
    except Exception:
        return None


def try_load_neural_extractor(path: Path) -> Any | None:
    try:
        from realtime.neural_features import NeuralFeatureExtractor

        return NeuralFeatureExtractor.load(path)
    except Exception:
        return None


def inventory_reusable_transforms(
    comparison_root: Path,
    *,
    feature_sets: list[str] | tuple[str, ...],
    embeddings: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...],
    n_components: list[int] | tuple[int, ...] = (3,),
    feature_type: str = "counts",
    n_neighbors: int | None = None,
) -> dict[str, Any]:
    """Count saved manifolds for **compatible** (fs × emb × W × k) only.

    Incompatible pairs (e.g. ``region_pca`` on non-unit-aligned feature sets)
    are excluded from the denominator so reuse banners reflect valid checkpoints.
    """
    from realtime.neural_features import embedding_compatible_with_feature_set

    root = Path(comparison_root)
    total = 0
    hits: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    skipped_incompatible: list[dict[str, Any]] = []
    for fs in feature_sets:
        for emb in embeddings:
            resolved = resolve_manifold_alias(emb)
            if not embedding_compatible_with_feature_set(resolved, fs):
                skipped_incompatible.append({
                    "feature_set": fs,
                    "embedding_type": resolved,
                })
                continue
            for w in decode_windows:
                for k in n_components:
                    total += 1
                    path = find_manifold_transform(
                        root,
                        feature_set=fs,
                        embedding_type=resolved,
                        decode_window=float(w),
                        n_components=int(k),
                        n_neighbors=n_neighbors,
                        feature_type=feature_type,
                    )
                    row = {
                        "feature_set": fs,
                        "embedding_type": resolved,
                        "decode_window_s": float(w),
                        "n_components": int(k),
                        "path": str(path) if path else None,
                    }
                    if path is not None:
                        hits.append(row)
                    else:
                        missing.append(row)
    return {
        "comparison_root": str(root),
        "n_requested": total,
        "n_reusable": len(hits),
        "n_missing": len(missing),
        "n_incompatible_skipped": len(skipped_incompatible),
        "hits": hits,
        "missing": missing,
        "incompatible": skipped_incompatible,
    }


def inventory_reusable_feature_transforms(
    comparison_root: Path,
    *,
    feature_sets: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...],
    feature_types: list[str] | tuple[str, ...] = ("counts",),
) -> dict[str, Any]:
    """Count saved SpikeFeatureTransformer F dirs for selected fs × W × F-type.

    Uses ``effective_spike_feature_type`` so richer neural feature sets map to
    the same F type Decoder Benchmark will request (usually ``counts``).
    """
    from realtime.neural_features.comparison import effective_spike_feature_type

    root = Path(comparison_root)
    total = 0
    hits: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for fs in feature_sets:
        for ft in feature_types:
            f_eff = effective_spike_feature_type(str(fs), str(ft))
            for w in decode_windows:
                key = (str(fs), str(f_eff), float(w))
                if key in seen:
                    continue
                seen.add(key)
                total += 1
                path = find_feature_transform(
                    root,
                    feature_set=str(fs),
                    feature_type_eff=str(f_eff),
                    decode_window=float(w),
                )
                row = {
                    "feature_set": str(fs),
                    "feature_type_eff": str(f_eff),
                    "decode_window_s": float(w),
                    "path": str(path) if path else None,
                }
                if path is not None:
                    hits.append(row)
                else:
                    missing.append(row)
    return {
        "comparison_root": str(root),
        "n_requested": total,
        "n_reusable": len(hits),
        "n_missing": len(missing),
        "hits": hits,
        "missing": missing,
    }


def feature_transform_path(
    comparison_root: Path,
    *,
    feature_set: str,
    feature_type_eff: str,
    decode_window: float,
) -> Path:
    roots = comparison_model_roots(comparison_root)
    return roots["feature"] / feature_transform_dirname(
        feature_set, feature_type_eff, decode_window,
    )


def save_feature_transform_checkpoint(
    transformer: Any,
    comparison_root: Path,
    *,
    feature_set: str,
    feature_type_eff: str,
    decode_window: float,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Persist a fitted SpikeFeatureTransformer into the shared F cache."""
    if transformer is None or not hasattr(transformer, "save"):
        raise TypeError("transformer must implement save(output_dir)")
    path = feature_transform_path(
        comparison_root,
        feature_set=feature_set,
        feature_type_eff=feature_type_eff,
        decode_window=decode_window,
    )
    path.mkdir(parents=True, exist_ok=True)
    transformer.save(path)
    enrich_transform_meta(
        path,
        {
            "feature_set": feature_set,
            "feature_type": feature_type_eff,
            "feature_type_eff": feature_type_eff,
            "decode_window_s": float(decode_window),
            **(extra_meta or {}),
        },
    )
    return path


def read_transform_meta(path: Path) -> dict[str, Any]:
    meta_path = Path(path) / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
