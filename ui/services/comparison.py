"""Build ComparisonRunConfig from UI selections (no Streamlit imports)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from realtime.decoder_comparison import (
    DEFAULT_DECODE_WINDOWS,
    ComparisonRunConfig,
    run_compare_sources,
    run_decoder_comparison,
)
from realtime.neural_features import ALL_FEATURE_SETS, DEFAULT_FEATURE_SETS
from realtime.neural_features.comparison import resolve_manifolds_arg
from realtime.search_space import (
    ALL_EMBEDDING_TYPES,
    DEFAULT_MANIFOLD_N_COMPONENTS,
    MANIFOLD_CLI_ALIASES,
)
from realtime.timing import DEFAULT_UPDATE_DT_S
from ui.services.representations import (
    UI_DYNAMIC_LATENT_OPTIONS,
    UI_STATIC_MANIFOLD_OPTIONS,
    format_representation_label,
    representation_capabilities,
)

# Manifold labels shown in the UI (CLI ``counts`` alias → identity embedding).
UI_MANIFOLD_OPTIONS: tuple[str, ...] = UI_STATIC_MANIFOLD_OPTIONS + UI_DYNAMIC_LATENT_OPTIONS

UI_FEATURE_SET_OPTIONS: tuple[str, ...] = tuple(DEFAULT_FEATURE_SETS)

DECODE_WINDOW_LABELS: dict[float, str] = {
    0.025: "25 ms",
    0.050: "50 ms",
    0.100: "100 ms",
    0.250: "250 ms",
    0.500: "500 ms",
    1.000: "1 s",
}


@dataclass
class UIBenchmarkSelection:
    """User-facing benchmark form values (thin UI DTO)."""

    input_dir: Path
    output_dir: Path
    spike_source: str = "sorted"
    feature_sets: tuple[str, ...] = DEFAULT_FEATURE_SETS
    manifolds: tuple[str, ...] = UI_MANIFOLD_OPTIONS
    decode_windows: tuple[float, ...] = DEFAULT_DECODE_WINDOWS
    update_dt: float = DEFAULT_UPDATE_DT_S
    train_frac: float = 0.70
    max_models: str = "quick"
    n_jobs: int = -1
    seed: int = 42
    run_feature_ablation: bool = False
    compare_sources: bool = False
    manifold_n_components: tuple[int, ...] = (3,)
    include_controls: bool = False
    region_ablation: bool = False
    layer_ablation: bool = False
    population_ablation: bool = False
    adaptive_windows: bool = False
    no_trigger_search: bool = False
    max_compute_ms: float = 25.0
    max_effective_history_s: float = 0.500
    run_id: str | None = None
    # Reserved for documentation / future parity with older design docs.
    # Current scientific pipeline uses GT vs sorted comparison instead of a
    # continuous degradation grid.
    degradation_levels: tuple[float, ...] = field(default_factory=tuple)


def format_decode_window(seconds: float) -> str:
    return DECODE_WINDOW_LABELS.get(float(seconds), f"{float(seconds) * 1000:.0f} ms")


def available_feature_sets() -> tuple[str, ...]:
    return tuple(ALL_FEATURE_SETS)


def available_manifolds() -> tuple[str, ...]:
    """CLI-oriented manifold tokens (includes ``counts`` → identity)."""
    return UI_MANIFOLD_OPTIONS


def available_embedding_types() -> tuple[str, ...]:
    return tuple(ALL_EMBEDDING_TYPES)


def resolve_ui_manifolds(manifolds: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    resolved = resolve_manifolds_arg(list(manifolds))
    if resolved is None:
        return ()
    return resolved


def valid_feature_manifold_pairs(
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
) -> list[tuple[str, str]]:
    """Return valid (feature_set, embedding) pairs; drop unit-aligned incompatibles."""
    from realtime.neural_features import embedding_compatible_with_feature_set

    pairs: list[tuple[str, str]] = []
    embeddings = resolve_ui_manifolds(manifolds)
    for fs in feature_sets:
        for emb in embeddings:
            if embedding_compatible_with_feature_set(emb, fs):
                pairs.append((fs, emb))
    return pairs


def estimate_workload(
    *,
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...],
    max_models: str = "quick",
    compare_sources: bool = False,
    run_feature_ablation: bool = False,
    n_components: list[int] | tuple[int, ...] = (3,),
    n_decoders_hint: int | None = None,
) -> dict[str, Any]:
    """Instant combinatorial estimate (no computation)."""
    pairs = valid_feature_manifold_pairs(feature_sets, manifolds)
    n_fs_emb = max(len(pairs), 1)
    n_w = max(len(decode_windows), 1)
    n_k = max(len(n_components), 1)
    # Quick zoo sizes (approximate; display only)
    if n_decoders_hint is not None:
        n_dec = n_decoders_hint
    else:
        n_dec = 8 if max_models == "quick" else 20
    n_sources = 2 if compare_sources else 1
    # Ablations add full_minus_* feature sets (~6)
    ablation_extra = 6 if run_feature_ablation else 0
    n_fs_emb_eff = n_fs_emb + ablation_extra * max(len(resolve_ui_manifolds(manifolds)), 1)
    planned = n_fs_emb_eff * n_w * n_k * n_dec * n_sources
    invalid = len(feature_sets) * len(resolve_ui_manifolds(manifolds)) - len(pairs)
    return {
        "n_feature_sets": len(feature_sets),
        "n_manifolds": len(resolve_ui_manifolds(manifolds)),
        "n_valid_feature_manifold_pairs": len(pairs),
        "n_invalid_pairs_excluded": max(invalid, 0),
        "n_windows": n_w,
        "n_components": n_k,
        "n_decoders_approx": n_dec,
        "n_spike_sources": n_sources,
        "planned_configurations": int(planned),
        "valid_pairs": pairs,
    }


def validate_benchmark_selection(sel: UIBenchmarkSelection) -> list[str]:
    """Return human-readable validation errors (empty ⇒ ok)."""
    errors: list[str] = []
    if not Path(sel.input_dir).exists():
        errors.append(f"Input dataset not found: {sel.input_dir}")
    if not sel.feature_sets:
        errors.append("Select at least one feature set.")
    if not sel.manifolds:
        errors.append("Select at least one manifold / embedding.")
    if not sel.decode_windows:
        errors.append("Select at least one decode window.")
    unknown_fs = [f for f in sel.feature_sets if f not in ALL_FEATURE_SETS]
    if unknown_fs:
        errors.append(f"Unknown feature set(s): {unknown_fs}")
    for m in sel.manifolds:
        emb = MANIFOLD_CLI_ALIASES.get(m, m)
        if emb not in ALL_EMBEDDING_TYPES:
            errors.append(f"Unknown manifold / embedding: {m}")
    if sel.max_models not in ("quick", "full"):
        errors.append("max_models must be 'quick' or 'full'")
    if sel.degradation_levels:
        errors.append(
            "degradation_levels are not supported by the current decoder "
            "comparison backend; use compare_sources (GT vs sorted) instead."
        )
    return errors


def build_comparison_config(sel: UIBenchmarkSelection) -> ComparisonRunConfig:
    """Translate UI selections into the scientific ComparisonRunConfig."""
    embeddings = resolve_ui_manifolds(sel.manifolds)
    return ComparisonRunConfig(
        input_dir=Path(sel.input_dir),
        output_dir=Path(sel.output_dir),
        spike_source=sel.spike_source,
        decode_windows=tuple(float(w) for w in sel.decode_windows),
        update_dt=float(sel.update_dt),
        train_frac=float(sel.train_frac),
        feature_sets=tuple(sel.feature_sets),
        feature_types=("counts",),
        embedding_types=embeddings,
        use_fe_grid=True,
        run_feature_ablation=bool(sel.run_feature_ablation),
        manifold_n_components=tuple(int(k) for k in sel.manifold_n_components),
        max_models=sel.max_models,
        n_jobs=int(sel.n_jobs),
        seed=int(sel.seed),
        include_controls=bool(sel.include_controls),
        region_ablation=bool(sel.region_ablation),
        layer_ablation=bool(sel.layer_ablation),
        population_ablation=bool(sel.population_ablation),
        adaptive_windows=bool(sel.adaptive_windows),
        enable_trigger_search=not bool(sel.no_trigger_search),
        max_compute_ms=float(sel.max_compute_ms),
        max_effective_history_s=float(sel.max_effective_history_s),
        run_id=sel.run_id,
    )


def selection_to_dict(sel: UIBenchmarkSelection) -> dict[str, Any]:
    d = asdict(sel)
    d["input_dir"] = str(sel.input_dir)
    d["output_dir"] = str(sel.output_dir)
    d["resolved_embeddings"] = list(resolve_ui_manifolds(sel.manifolds))
    return d


def run_benchmark(sel: UIBenchmarkSelection):
    """Execute comparison (or compare-sources) from a UI selection."""
    errors = validate_benchmark_selection(sel)
    if errors:
        raise ValueError("; ".join(errors))
    Path(sel.output_dir).mkdir(parents=True, exist_ok=True)
    if sel.compare_sources:
        embeddings = resolve_ui_manifolds(sel.manifolds)
        return run_compare_sources(
            input_dir=Path(sel.input_dir),
            output_dir=Path(sel.output_dir),
            decode_windows=tuple(float(w) for w in sel.decode_windows),
            update_dt=float(sel.update_dt),
            train_frac=float(sel.train_frac),
            feature_sets=tuple(sel.feature_sets),
            feature_types=("counts",),
            embedding_types=embeddings,
            manifold_n_components=tuple(int(k) for k in sel.manifold_n_components),
            max_models=sel.max_models,
            n_jobs=int(sel.n_jobs),
            seed=int(sel.seed),
            run_feature_ablation=bool(sel.run_feature_ablation),
            include_controls=bool(sel.include_controls),
            region_ablation=bool(sel.region_ablation),
            layer_ablation=bool(sel.layer_ablation),
            population_ablation=bool(sel.population_ablation),
            adaptive_windows=bool(sel.adaptive_windows),
            enable_trigger_search=not bool(sel.no_trigger_search),
            max_compute_ms=float(sel.max_compute_ms),
            max_effective_history_s=float(sel.max_effective_history_s),
        )
    cfg = build_comparison_config(sel)
    return run_decoder_comparison(cfg)
