"""Discover analysis figures under an experiment without recomputing science."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from ui.artifacts.models import (
    ALL_CATEGORIES,
    IMAGE_SUFFIXES,
    MANIFOLD_STEM_PREFIXES,
    PDF_UI_STEM_ORDER,
    PRIMARY_STEMS,
    STEM_CATEGORY_OVERRIDES,
    SUBDIR_CATEGORY,
    UI_HIDDEN_STEM_PREFIXES,
    VISUAL_SUFFIXES,
    AnalysisArtifact,
    CATEGORY_DECODING,
    CATEGORY_MANIFOLDS,
    CATEGORY_OTHER,
)

logger = logging.getLogger(__name__)

# Soft dependency: discovery must work even if visualization plotting stack
# is incomplete. Prefer visualization.artifact_manifest when present.
try:
    from visualization.artifact_manifest import MANIFEST_FILENAME, load_manifest
except Exception:  # noqa: BLE001 — UI boot must not require full viz package
    MANIFEST_FILENAME = "artifacts.json"

    def load_manifest(manifest_file: Path) -> list[dict]:
        path = Path(manifest_file)
        if not path.exists():
            return []
        try:
            import json

            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Failed to read artifact manifest %s: %s", path, exc)
            return []
        if isinstance(data, dict):
            arts = data.get("artifacts", [])
            return list(arts) if isinstance(arts, list) else []
        if isinstance(data, list):
            return data
        return []

# Directories / name fragments to skip while walking
_IGNORE_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "cache",
    ".cache",
    "tmp",
    "temp",
    "node_modules",
    "models",  # joblib weights, not figures
    "lab_deployable_profiles",
    "decoded_examples",
    "pipeline_timing",
    "feature_transforms",
    "manifold_transforms",
    "neural_feature_extractors",
}

_IGNORE_FILE_PREFIXES = (".", "~", "tmp_", "temp_")

_TARGET_FROM_STEM = re.compile(
    r"(?:geometry|heatmap|decoding|error|accuracy)_(?P<target>"
    r"position|speed|acceleration|head_direction|distance_to_wall|"
    r"spatial_context|movement_state|wall_distance_bin)(?:_|$)",
)

_WINDOW_RE = re.compile(r"(?:window[_\-]?|w)(?P<ms>\d{2,4})(?:ms)?", re.I)
_DECODER_HINTS = (
    "ridge", "random_forest", "logistic", "elastic_net", "pls",
    "hist_gradient", "knn", "svr", "svc", "mlp", "bayesian",
)


def _mtime(path: Path) -> datetime | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    except OSError:
        return None


def _artifact_type(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in IMAGE_SUFFIXES:
        return "image"
    if suf in {".html", ".htm"}:
        return "html"
    if suf == ".pdf":
        return "pdf"
    return "data"


def category_for_path(path: Path, stem: str | None = None) -> str:
    """Map a figure path / stem onto a UI category."""
    stem = stem or path.stem
    parts = {p.lower() for p in path.parts}

    override = STEM_CATEGORY_OVERRIDES.get(stem)
    if override:
        return override

    # Explicit subdir under figures/
    for part in path.parts:
        if part in SUBDIR_CATEGORY:
            cat = SUBDIR_CATEGORY[part]
            if cat == CATEGORY_DECODING and stem.lower().startswith(MANIFOLD_STEM_PREFIXES):
                return CATEGORY_MANIFOLDS
            return cat

    if any(stem.lower().startswith(p) for p in MANIFOLD_STEM_PREFIXES):
        return CATEGORY_MANIFOLDS
    if "realtime" in parts:
        from ui.artifacts.models import CATEGORY_REALTIME
        return CATEGORY_REALTIME
    if "latency" in parts:
        from ui.artifacts.models import CATEGORY_PERFORMANCE
        return CATEGORY_PERFORMANCE
    return CATEGORY_OTHER


def _iter_visual_files(root: Path) -> list[Path]:
    import os

    found: list[Path] = []
    if not root.exists():
        return found
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not _should_skip_dir(d)]
        base = Path(dirpath)
        for name in filenames:
            path = base / name
            if _should_skip_file(path):
                continue
            found.append(path)
    return found


def readable_title(path: Path, *, manifest_title: str | None = None) -> str:
    """Prefer manifest title, then curated map / captions, then humanized stem."""
    if manifest_title:
        return manifest_title
    from ui.artifacts.models import READABLE_TITLES

    stem = path.stem
    if stem in READABLE_TITLES:
        return READABLE_TITLES[stem]

    for prefix, label in (
        ("fig_latent_geometry_", "Latent geometry colored by"),
        ("fig_decoder_geometry_", "Decoder geometry —"),
        ("population_tuning_", "Population tuning —"),
    ):
        if stem.startswith(prefix):
            feature = stem[len(prefix):].replace("_", " ")
            return f"{label} {feature}"

    try:
        from visualization.figure_captions import CAPTIONS, title_for, _caption_body

        if stem in CAPTIONS:
            sentence = _caption_body(stem).split(".")[0].strip()
            if 8 < len(sentence) < 120:
                return sentence
        return title_for(path)
    except Exception:
        cleaned = stem.replace("_", " ").removeprefix("fig ").strip()
        return cleaned.title()


def description_for(path: Path) -> str | None:
    try:
        from visualization.figure_captions import _caption_body
        return _caption_body(path.stem)
    except Exception:
        return None


def _infer_target(stem: str) -> str | None:
    m = _TARGET_FROM_STEM.search(stem)
    if m:
        return m.group("target")
    for t in (
        "position", "speed", "acceleration", "head_direction",
        "distance_to_wall", "spatial_context", "movement_state", "wall_distance_bin",
    ):
        if stem.endswith("_" + t) or f"_{t}_" in stem:
            return t
    return None


def _infer_window(stem: str) -> float | None:
    m = _WINDOW_RE.search(stem)
    if not m:
        return None
    ms = int(m.group("ms"))
    # Heuristic: 0250 style vs 250
    if ms >= 1000:
        return ms / 1000.0
    if ms in (25, 50, 100, 250, 500):
        return ms / 1000.0
    if ms in (25, 50) or (100 <= ms <= 1000):
        return ms / 1000.0
    return None


def _should_skip_dir(name: str) -> bool:
    return name in _IGNORE_DIR_NAMES or name.startswith(".")


def _should_skip_file(path: Path) -> bool:
    name = path.name
    if any(name.startswith(p) for p in _IGNORE_FILE_PREFIXES):
        return True
    if path.suffix.lower() not in VISUAL_SUFFIXES:
        return True
    if any(path.stem.startswith(p) for p in UI_HIDDEN_STEM_PREFIXES):
        return True
    # Compiled multi-figure PDF is offered separately; still discoverable
    return False


def figure_search_roots(experiment_dir: Path) -> list[Path]:
    """Directories worth scanning for analysis visualizations."""
    exp = Path(experiment_dir)
    roots: list[Path] = []
    primary = exp / "figures"
    if primary.exists():
        roots.append(primary)
    # Nested comparison sidecars (e.g. ui_smoke_v1/figures)
    for cand in (
        exp / "decoder_comparison",
        exp / "realtime_decoding",
        exp / "deployment_decoder_selection",
        exp / "manifolds",
        exp / "features",
    ):
        if cand.exists():
            roots.append(cand)
    return roots


def _artifact_from_path(
    path: Path,
    *,
    experiment_dir: Path,
    source: str = "filesystem",
    overrides: dict | None = None,
) -> AnalysisArtifact:
    overrides = overrides or {}
    stem = path.stem
    category = overrides.get("category") or category_for_path(path, stem)
    title = readable_title(path, manifest_title=overrides.get("title"))
    rel = None
    try:
        rel = path.relative_to(experiment_dir).as_posix()
    except ValueError:
        rel = path.name

    created = None
    if overrides.get("created_at"):
        try:
            created = datetime.fromisoformat(str(overrides["created_at"]))
        except ValueError:
            created = _mtime(path)
    else:
        created = _mtime(path)

    return AnalysisArtifact(
        path=path,
        artifact_type=_artifact_type(path),
        category=str(category),
        title=title,
        description=overrides.get("description") or description_for(path),
        run_id=overrides.get("run_id"),
        target=overrides.get("target") or _infer_target(stem),
        feature_set=overrides.get("feature_set"),
        manifold=overrides.get("manifold"),
        decoder=overrides.get("decoder"),
        decode_window=(
            float(overrides["decode_window"])
            if overrides.get("decode_window") is not None
            else _infer_window(stem)
        ),
        degradation_level=(
            float(overrides["degradation_level"])
            if overrides.get("degradation_level") is not None
            else None
        ),
        created_at=created,
        relative_path=overrides.get("relative_path") or rel,
        experiment_dir=experiment_dir,
        source=source,
        extras={k: v for k, v in overrides.items() if k not in {
            "path", "category", "title", "description", "run_id", "target",
            "feature_set", "manifold", "decoder", "decode_window",
            "degradation_level", "created_at", "relative_path",
        }},
    )


def _load_manifest_artifacts(experiment_dir: Path) -> list[AnalysisArtifact]:
    exp = Path(experiment_dir)
    arts: list[AnalysisArtifact] = []
    candidates = [
        exp / "figures" / MANIFEST_FILENAME,
        exp / MANIFEST_FILENAME,
    ]
    # Nested manifests under comparison outputs
    for root in figure_search_roots(exp):
        for manifest in root.rglob(MANIFEST_FILENAME):
            if manifest not in candidates:
                candidates.append(manifest)

    seen: set[Path] = set()
    for man in candidates:
        if not man.exists() or man in seen:
            continue
        seen.add(man)
        base = man.parent
        for entry in load_manifest(man):
            rel = entry.get("relative_path") or entry.get("path")
            if not rel:
                continue
            path = Path(rel)
            if not path.is_absolute():
                path = base / rel
            if not path.exists():
                # Try resolving against experiment root
                alt = exp / rel
                if alt.exists():
                    path = alt
                else:
                    continue
            arts.append(
                _artifact_from_path(
                    path,
                    experiment_dir=exp,
                    source="manifest",
                    overrides=entry,
                )
            )
    return arts


def discover_artifacts(experiment_dir: Path) -> list[AnalysisArtifact]:
    """Discover visualization artifacts for an experiment.

    Preference order per file: manifest metadata → filesystem classification →
    filename inference. Never regenerates scientific analyses.
    """
    exp = Path(experiment_dir)
    if not exp.exists():
        return []

    by_path: dict[Path, AnalysisArtifact] = {}

    for art in _load_manifest_artifacts(exp):
        by_path[art.path.resolve()] = art

    for root in figure_search_roots(exp):
        for path in _iter_visual_files(root):
            # Skip compiled PDF at figures root unless listed (still include)
            key = path.resolve()
            if key in by_path:
                continue
            # Prefer images under figures/ or **/figures/; allow nested comparison figures
            parts = path.parts
            if "figures" not in parts and path.suffix.lower() == ".pdf":
                continue
            if "figures" not in parts and path.suffix.lower() not in IMAGE_SUFFIXES | {".html", ".htm"}:
                continue
            by_path[key] = _artifact_from_path(path, experiment_dir=exp, source="filesystem")

    arts = list(by_path.values())
    # Stable sort: category order, then publication PDF order, then title
    cat_rank = {c: i for i, c in enumerate(ALL_CATEGORIES)}
    pdf_rank = {s: i for i, s in enumerate(PDF_UI_STEM_ORDER)}
    primary_rank = {s: i for i, s in enumerate(PRIMARY_STEMS)}

    def sort_key(a: AnalysisArtifact):
        return (
            cat_rank.get(a.category, 99),
            pdf_rank.get(a.stem, 999),
            primary_rank.get(a.stem, 999),
            a.title.lower(),
            str(a.path),
        )

    arts.sort(key=sort_key)
    return arts


def filter_artifacts(
    artifacts: list[AnalysisArtifact],
    *,
    categories: list[str] | None = None,
    targets: list[str] | None = None,
    feature_sets: list[str] | None = None,
    manifolds: list[str] | None = None,
    decoders: list[str] | None = None,
    stems: list[str] | None = None,
    artifact_types: list[str] | None = None,
) -> list[AnalysisArtifact]:
    out = list(artifacts)
    if categories:
        out = [a for a in out if a.category in categories]
    if targets:
        out = [a for a in out if a.target in targets]
    if feature_sets:
        out = [a for a in out if a.feature_set in feature_sets]
    if manifolds:
        out = [a for a in out if a.manifold in manifolds]
    if decoders:
        out = [a for a in out if a.decoder in decoders]
    if stems:
        stem_set = set(stems)
        out = [a for a in out if a.stem in stem_set]
    if artifact_types:
        out = [a for a in out if a.artifact_type in artifact_types]
    return out


def primary_artifacts(
    artifacts: list[AnalysisArtifact],
    *,
    limit: int = 4,
) -> list[AnalysisArtifact]:
    """Select a compact overview set from available figures."""
    images = [a for a in artifacts if a.is_image]
    chosen: list[AnalysisArtifact] = []
    by_stem = {a.stem: a for a in images}
    for stem in PRIMARY_STEMS:
        if stem in by_stem:
            chosen.append(by_stem[stem])
        if len(chosen) >= limit:
            return chosen
    # Fill with first image from under-represented categories
    seen_cat = {a.category for a in chosen}
    for a in images:
        if a in chosen:
            continue
        if a.category not in seen_cat or len(chosen) < limit:
            chosen.append(a)
            seen_cat.add(a.category)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def artifacts_by_category(
    artifacts: list[AnalysisArtifact],
) -> dict[str, list[AnalysisArtifact]]:
    grouped: dict[str, list[AnalysisArtifact]] = {}
    for a in artifacts:
        grouped.setdefault(a.category, []).append(a)
    return grouped


def experiment_mtime_token(experiment_dir: Path) -> str:
    """Cache-bust token based on figures directory mtimes."""
    exp = Path(experiment_dir)
    figures = exp / "figures"
    try:
        if figures.exists():
            return str(figures.stat().st_mtime_ns)
        return str(exp.stat().st_mtime_ns)
    except OSError:
        return "0"
