"""Benchmark execution profiles: quick / targeted / full, plus staged search.

Quick is the interactive default (one observation, few models). Full factorial
is an explicit opt-in. Targeted lets the caller name which axes are swept.

These helpers only *plan* work. They never start computation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from realtime.decoder_models import (
    QUICK_CATEGORICAL,
    QUICK_CONTINUOUS,
    resolve_model_names_for_target,
)

CONTINUOUS_TARGETS = (
    "position",
    "speed",
    "acceleration",
    "head_direction",
    "distance_to_wall",
)
CATEGORICAL_TARGETS = (
    "spatial_context",
    "movement_state",
    "wall_distance_bin",
)
ALL_TARGETS = CONTINUOUS_TARGETS + CATEGORICAL_TARGETS
from realtime.neural_features import embedding_compatible_with_feature_set
from realtime.search_space import (
    expand_fe_jobs,
    resolve_embedding_types,
    resolve_manifold_alias,
)

QUICK_WINDOWS_S: tuple[float, ...] = (0.250,)
STAGE1_WINDOWS_S: tuple[float, ...] = (0.050, 0.100, 0.250, 0.500, 1.000)
STAGE1_EMBEDDINGS: tuple[str, ...] = ("identity", "global_pca")
STAGE1_DECODERS: tuple[str, ...] = ("ridge", "logistic_regression")
STAGE2_EMBEDDINGS: tuple[str, ...] = (
    "identity",
    "global_pca",
    "region_pca",
    "diffusion_nystrom",
    "global_lds",
)
QUICK_EMBEDDINGS: tuple[str, ...] = ("identity", "global_pca", "region_pca")
QUICK_FEATURE_SETS: tuple[str, ...] = ("counts",)

BENCHMARK_MODES: tuple[str, ...] = ("quick", "targeted", "full")


def _unique(seq: Iterable[Any]) -> tuple[Any, ...]:
    out: list[Any] = []
    seen: set[str] = set()
    for item in seq:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)


def _as_tuple(values: Sequence[Any] | None, fallback: tuple[Any, ...]) -> tuple[Any, ...]:
    if values is None:
        return fallback
    return _unique(values)


@dataclass(frozen=True)
class BenchmarkPlan:
    """Estimated combinatorial size of a decoder search (not runtime)."""

    mode: str
    targets: tuple[str, ...]
    windows: tuple[float, ...]
    feature_sets: tuple[str, ...]
    representations: tuple[str, ...]
    decoders: tuple[str, ...]
    n_components: tuple[int, ...] = (3,)
    n_valid_pairs: int = 0
    n_configurations: int = 0
    swept_axes: tuple[str, ...] = ()
    inherited_window_s: float | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "targets": list(self.targets),
            "n_targets": len(self.targets),
            "windows": [float(w) for w in self.windows],
            "n_windows": len(self.windows),
            "feature_sets": list(self.feature_sets),
            "n_features": len(self.feature_sets),
            "representations": list(self.representations),
            "n_representations": len(self.representations),
            "decoders": list(self.decoders),
            "n_decoders": len(self.decoders),
            "n_components": list(self.n_components),
            "n_valid_feature_representation_pairs": int(self.n_valid_pairs),
            "n_configurations": int(self.n_configurations),
            "swept_axes": list(self.swept_axes),
            "inherited_window_s": self.inherited_window_s,
            "notes": self.notes,
        }

    def summary_lines(self) -> list[str]:
        return [
            f"Benchmark plan ({self.mode})",
            f"Targets: {len(self.targets)}",
            f"Windows: {len(self.windows)}",
            f"Features: {len(self.feature_sets)}",
            f"Representations: {len(self.representations)}",
            f"Decoders: {len(self.decoders)}",
            f"Total model configurations: {self.n_configurations}",
        ]


@dataclass(frozen=True)
class StagedSearchSpec:
    """Optional four-stage search; never auto-applied to saved results."""

    name: str
    windows: tuple[float, ...]
    feature_sets: tuple[str, ...]
    representations: tuple[str, ...]
    decoders: tuple[str, ...]
    targets: tuple[str, ...]
    notes: str = ""


def default_staged_search(
    *,
    target: str = "position",
    feature_set: str = "counts",
    promising_windows: Sequence[float] | None = None,
    promising_representations: Sequence[str] | None = None,
) -> tuple[StagedSearchSpec, ...]:
    wins = tuple(float(w) for w in (promising_windows or STAGE1_WINDOWS_S))
    reps = tuple(promising_representations or STAGE2_EMBEDDINGS)
    t = (str(target),)
    fs = (str(feature_set),)
    return (
        StagedSearchSpec(
            name="stage1_observation_timescale",
            windows=STAGE1_WINDOWS_S,
            feature_sets=fs,
            representations=STAGE1_EMBEDDINGS,
            decoders=STAGE1_DECODERS,
            targets=t,
            notes="Hold basic E/D fixed; compare W.",
        ),
        StagedSearchSpec(
            name="stage2_representation",
            windows=wins[:2] if len(wins) > 2 else wins,
            feature_sets=fs,
            representations=STAGE2_EMBEDDINGS,
            decoders=("ridge",),
            targets=t,
            notes="For sensible W, compare representations.",
        ),
        StagedSearchSpec(
            name="stage3_decoder",
            windows=wins[:1] or QUICK_WINDOWS_S,
            feature_sets=fs,
            representations=reps[:3] or QUICK_EMBEDDINGS,
            decoders=tuple(QUICK_CONTINUOUS) + tuple(QUICK_CATEGORICAL),
            targets=t,
            notes="Compare decoder families on promising O/E cells.",
        ),
        StagedSearchSpec(
            name="stage4_confirmation",
            windows=wins[:1] or QUICK_WINDOWS_S,
            feature_sets=fs,
            representations=reps[:2] or ("global_pca",),
            decoders=("ridge",),
            targets=t,
            notes="Broader validation on finalists only.",
        ),
    )


def valid_pairs(
    feature_sets: Sequence[str],
    representations: Sequence[str],
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for fs in feature_sets:
        for raw in representations:
            emb = resolve_manifold_alias(str(raw))
            try:
                ok = embedding_compatible_with_feature_set(emb, fs)
            except Exception:  # noqa: BLE001
                ok = True
            if ok:
                pairs.append((str(fs), str(emb)))
    return pairs


def count_configurations(
    *,
    targets: Sequence[str],
    windows: Sequence[float],
    feature_sets: Sequence[str],
    representations: Sequence[str],
    decoders: Sequence[str] | None,
    n_components: Sequence[int] = (3,),
    max_models: str = "quick",
) -> tuple[int, int]:
    """Return (n_valid_pairs, n_model_configurations).

    A configuration is one (W, F, E, k, D, target) eval after dropping
    incompatible feature/representation pairs. Decoder names that do not apply
    to a target family are skipped.
    """
    pairs = valid_pairs(feature_sets, representations)
    n_w = max(len(tuple(windows)), 1)
    n_k = max(len(tuple(n_components)), 1)
    n = 0
    selected = tuple(decoders) if decoders else None
    for fs, emb in pairs:
        jobs = expand_fe_jobs(
            feature_types=( "counts", ),
            embedding_types=(emb,),
            manifold_n_components=tuple(int(k) for k in n_components) or (3,),
            max_models=max_models,
            use_fe_grid=True,
        )
        n_jobs = max(len(jobs), 1)
        # expand_fe_jobs already includes k; don't multiply n_k again when jobs exist.
        k_mult = 1 if jobs else n_k
        for target in targets:
            names = resolve_model_names_for_target(
                str(target),
                max_models=max_models,
                selected=selected,
            )
            n += n_w * n_jobs * k_mult * max(len(names), 0)
    return len(pairs), int(n)


def swept_axes(
    *,
    targets: Sequence[str],
    windows: Sequence[float],
    feature_sets: Sequence[str],
    representations: Sequence[str],
    decoders: Sequence[str],
) -> tuple[str, ...]:
    axes: list[str] = []
    if len(tuple(targets)) > 1:
        axes.append("target")
    if len(tuple(windows)) > 1:
        axes.append("window")
    if len(tuple(feature_sets)) > 1:
        axes.append("feature")
    if len(tuple(representations)) > 1:
        axes.append("representation")
    if len(tuple(decoders)) > 1:
        axes.append("decoder")
    return tuple(axes)


def plan_benchmark(
    *,
    mode: str,
    targets: Sequence[str] | None = None,
    windows: Sequence[float] | None = None,
    feature_sets: Sequence[str] | None = None,
    representations: Sequence[str] | None = None,
    decoders: Sequence[str] | None = None,
    n_components: Sequence[int] | None = None,
    inherited_window_s: float | None = None,
    max_models: str | None = None,
) -> BenchmarkPlan:
    """Build a plan. Quick inherits one W; full uses the supplied grid as-is."""
    key = str(mode or "quick").strip().lower()
    if key not in BENCHMARK_MODES:
        raise ValueError(f"Unknown benchmark mode {mode!r}; expected {BENCHMARK_MODES}")

    t = _as_tuple(targets, CONTINUOUS_TARGETS[:1])
    fs = _as_tuple(feature_sets, QUICK_FEATURE_SETS)
    reps = _as_tuple(representations, QUICK_EMBEDDINGS)
    n_comp = tuple(int(k) for k in (n_components or (3,)))
    notes = ""

    if key == "quick":
        w = (float(inherited_window_s),) if inherited_window_s is not None else QUICK_WINDOWS_S
        if windows and len(tuple(windows)) == 1:
            w = (float(tuple(windows)[0]),)
        fs = fs[:1] or QUICK_FEATURE_SETS
        reps = reps[:3] or QUICK_EMBEDDINGS
        t = t[:1] or ("position",)
        d = _as_tuple(decoders, tuple(QUICK_CONTINUOUS[:1]) + tuple(QUICK_CATEGORICAL[:1]))
        d = d[:2]
        mx = max_models or "quick"
        notes = (
            "Interactive: one active observation, few representations, "
            "one or two decoders. Window is inherited from Feature Construction."
        )
    elif key == "targeted":
        if inherited_window_s is not None and not windows:
            w = (float(inherited_window_s),)
        else:
            w = _as_tuple(windows, QUICK_WINDOWS_S)
        d = _as_tuple(decoders, tuple(QUICK_CONTINUOUS) + tuple(QUICK_CATEGORICAL))
        mx = max_models or "quick"
        notes = "Targeted comparison on the selected axes only."
    else:
        w = _as_tuple(windows, STAGE1_WINDOWS_S)
        t = _as_tuple(targets, ALL_TARGETS)
        d = _as_tuple(decoders, tuple(QUICK_CONTINUOUS) + tuple(QUICK_CATEGORICAL))
        mx = max_models or "full"
        notes = (
            "Full factorial opt-in. This never starts from a widget change; "
            "it requires an explicit Run."
        )

    n_pairs, n_cfg = count_configurations(
        targets=t,
        windows=w,
        feature_sets=fs,
        representations=reps,
        decoders=d,
        n_components=n_comp,
        max_models=mx,
    )
    return BenchmarkPlan(
        mode=key,
        targets=tuple(str(x) for x in t),
        windows=tuple(float(x) for x in w),
        feature_sets=tuple(str(x) for x in fs),
        representations=tuple(str(x) for x in reps),
        decoders=tuple(str(x) for x in d),
        n_components=n_comp,
        n_valid_pairs=n_pairs,
        n_configurations=n_cfg,
        swept_axes=swept_axes(
            targets=t, windows=w, feature_sets=fs, representations=reps, decoders=d,
        ),
        inherited_window_s=float(inherited_window_s) if inherited_window_s is not None else None,
        notes=notes,
    )


def plan_from_staged_spec(spec: StagedSearchSpec) -> BenchmarkPlan:
    return plan_benchmark(
        mode="targeted",
        targets=spec.targets,
        windows=spec.windows,
        feature_sets=spec.feature_sets,
        representations=spec.representations,
        decoders=spec.decoders,
        inherited_window_s=spec.windows[0] if spec.windows else None,
    )
