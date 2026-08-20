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
from realtime.decoder_models import (
    FULL_CATEGORICAL,
    FULL_CONTINUOUS,
    QUICK_CATEGORICAL,
    QUICK_CONTINUOUS,
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

# Named decoder pickers (full zoo; defaults use the quick subset).
UI_CONTINUOUS_DECODER_OPTIONS: tuple[str, ...] = tuple(
    n for n in FULL_CONTINUOUS if n != "state_space_or_kalman_optional"
)
UI_CATEGORICAL_DECODER_OPTIONS: tuple[str, ...] = tuple(FULL_CATEGORICAL)
UI_DEFAULT_CONTINUOUS_DECODERS: tuple[str, ...] = tuple(QUICK_CONTINUOUS)
UI_DEFAULT_CATEGORICAL_DECODERS: tuple[str, ...] = tuple(QUICK_CATEGORICAL)


def available_continuous_decoders() -> tuple[str, ...]:
    return UI_CONTINUOUS_DECODER_OPTIONS


def available_categorical_decoders() -> tuple[str, ...]:
    return UI_CATEGORICAL_DECODER_OPTIONS


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
    # Prefer loading saved manifolds / F transforms when present.
    reuse_transforms: bool = True
    # Explicit decoder names (continuous + categorical). Empty → fall back to max_models.
    decoder_names: tuple[str, ...] = ()
    targets: tuple[str, ...] = ()
    # Reserved for documentation / future parity with older design docs.
    # Current scientific pipeline uses GT vs sorted comparison instead of a
    # continuous degradation grid.
    degradation_levels: tuple[float, ...] = field(default_factory=tuple)


def format_decode_window(seconds: float) -> str:
    return DECODE_WINDOW_LABELS.get(float(seconds), f"{float(seconds) * 1000:.0f} ms")


def format_duration(seconds: float | None) -> str:
    """Human-readable duration for ETA / elapsed labels."""
    if seconds is None:
        return "—"
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if s < 0:
        s = 0.0
    if s < 60:
        return f"{s:.0f} s"
    if s < 3600:
        m = int(s // 60)
        rem = int(round(s - 60 * m))
        return f"{m} m {rem:02d} s"
    h = int(s // 3600)
    m = int((s - 3600 * h) // 60)
    return f"{h} h {m:02d} m"


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
    n_targets_hint: int | None = None,
    family: str = "static",
    session_duration_s: float | None = None,
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
    n_targets = max(int(n_targets_hint), 1) if n_targets_hint else 1
    # Ablations add full_minus_* feature sets (~6)
    ablation_extra = 6 if run_feature_ablation else 0
    n_fs_emb_eff = n_fs_emb + ablation_extra * max(len(resolve_ui_manifolds(manifolds)), 1)
    planned = n_fs_emb_eff * n_w * n_k * n_dec * n_sources * n_targets
    invalid = len(feature_sets) * len(resolve_ui_manifolds(manifolds)) - len(pairs)

    sec_per = 2.5 if (family or "static") == "static" else 4.0
    if max_models == "full":
        sec_per *= 1.6
    duration_scale = 1.0
    if session_duration_s is not None and float(session_duration_s) > 0:
        duration_scale = max(float(session_duration_s) / 600.0, 0.25)
    estimated_s = float(planned) * sec_per * duration_scale
    estimated_low_s = estimated_s * 0.7
    estimated_high_s = estimated_s * 1.6

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
        "family": family or "static",
        "sec_per_config": sec_per,
        "duration_scale": duration_scale,
        "estimated_runtime_s": float(estimated_s),
        "estimated_runtime_low_s": float(estimated_low_s),
        "estimated_runtime_high_s": float(estimated_high_s),
        "estimated_runtime_label": format_duration(estimated_s),
        "estimated_runtime_range_label": (
            f"{format_duration(estimated_low_s)} – {format_duration(estimated_high_s)}"
        ),
    }


_RUNTIME_KEYS = (
    "estimated_runtime_s",
    "estimated_runtime_low_s",
    "estimated_runtime_high_s",
)

# Fraction of Decoder Benchmark heuristic attributed to F/E fitting.
# Decoder heads still train when transforms are reused, so the displayed ETA
# never drops below ``1 - DECODER_TRANSFORM_TIME_SHARE``.
DECODER_TRANSFORM_TIME_SHARE = 0.35


def apply_remaining_work_to_workload(
    workload: dict[str, Any],
    *,
    n_wanted: int,
    n_covered: int = 0,
    n_to_compute: int | None = None,
    fully_covered: bool = False,
    hide_when_complete: bool = True,
    scale_floor: float = 0.0,
    scale_planned: bool = True,
    remaining_detail: str | None = None,
) -> dict[str, Any]:
    """Scale a display-only ETA by unfinished work.

    Does not change job skip/run behavior. Callers that still train a
    downstream stage (e.g. decoder heads) should pass
    ``hide_when_complete=False`` and a ``scale_floor``.
    """
    out = dict(workload)
    wanted = max(int(n_wanted or 0), 0)
    covered = max(int(n_covered or 0), 0)
    remaining = (
        max(wanted - covered, 0)
        if n_to_compute is None
        else max(int(n_to_compute), 0)
    )

    complete = bool(fully_covered) or (wanted > 0 and remaining <= 0)
    if hide_when_complete and complete:
        out["skip_estimate"] = True
        if scale_planned:
            out["planned_configurations"] = 0
        for key in _RUNTIME_KEYS:
            if key in out:
                out[key] = 0.0
        out["estimated_runtime_label"] = format_duration(0.0)
        out["estimated_runtime_range_label"] = (
            f"{format_duration(0.0)} – {format_duration(0.0)}"
        )
        return out

    if wanted <= 0:
        return out

    frac = min(max(remaining / float(wanted), 0.0), 1.0)
    if scale_floor > 0.0:
        frac = min(max(scale_floor + (1.0 - scale_floor) * frac, scale_floor), 1.0)

    for key in _RUNTIME_KEYS:
        if out.get(key) is not None:
            try:
                out[key] = float(out[key]) * frac
            except (TypeError, ValueError):
                continue

    if scale_planned:
        out["planned_configurations"] = int(remaining)

    est = float(out.get("estimated_runtime_s") or 0.0)
    low = out.get("estimated_runtime_low_s")
    high = out.get("estimated_runtime_high_s")
    try:
        low_s = float(low) if low is not None else est * 0.7
    except (TypeError, ValueError):
        low_s = est * 0.7
    try:
        high_s = float(high) if high is not None else est * 1.6
    except (TypeError, ValueError):
        high_s = est * 1.6
    out["estimated_runtime_label"] = format_duration(est)
    out["estimated_runtime_range_label"] = (
        f"{format_duration(low_s)} – {format_duration(high_s)}"
    )

    prefix = remaining_detail
    if prefix is None and covered > 0:
        prefix = f"{remaining} new · {covered} reusable"
    existing = str(out.get("detail_label") or "").strip()
    if prefix:
        out["detail_label"] = f"{prefix} · {existing}" if existing else prefix
    return out


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
    if not sel.decoder_names:
        errors.append("Select at least one decoder.")
    unknown_dec = [
        d for d in sel.decoder_names
        if d not in UI_CONTINUOUS_DECODER_OPTIONS
        and d not in UI_CATEGORICAL_DECODER_OPTIONS
    ]
    if unknown_dec:
        errors.append(f"Unknown decoder(s): {unknown_dec}")
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
    kwargs: dict[str, Any] = dict(
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
    fields = ComparisonRunConfig.__dataclass_fields__
    if "reuse_transforms" in fields:
        kwargs["reuse_transforms"] = bool(getattr(sel, "reuse_transforms", True))
    if "decoder_names" in fields and sel.decoder_names:
        kwargs["decoder_names"] = tuple(sel.decoder_names)
    if "targets" in fields and sel.targets:
        kwargs["targets"] = tuple(sel.targets)
    return ComparisonRunConfig(**kwargs)


def inventory_transform_reuse(sel: UIBenchmarkSelection) -> dict[str, Any]:
    """Summarize reusable manifold E and feature F checkpoints on disk."""
    empty = {
        "n_requested": 0,
        "n_reusable": 0,
        "n_missing": 0,
        "hits": [],
        "n_feature_requested": 0,
        "n_feature_reusable": 0,
        "n_feature_missing": 0,
        "feature_hits": [],
        "search_roots": [],
        "reuse_enabled": bool(getattr(sel, "reuse_transforms", True)),
    }
    try:
        from realtime.transform_cache import (
            discover_comparison_roots,
            inventory_reusable_feature_transforms,
            inventory_reusable_transforms,
        )
    except Exception:
        return empty

    embeddings = resolve_ui_manifolds(sel.manifolds)
    roots = [Path(sel.output_dir), *discover_comparison_roots(Path(sel.input_dir))]
    seen: set[Path] = set()
    unique_roots: list[Path] = []
    for r in roots:
        key = r.resolve() if r.exists() else r
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(r)

    combined_hits: list[dict[str, Any]] = []
    combined_feature_hits: list[dict[str, Any]] = []
    requested = 0
    feature_requested = 0
    for root in unique_roots:
        models = root / "models"
        has_man = (models / "manifold_transforms").exists()
        has_feat = (models / "feature_transforms").exists()
        if not has_man and not has_feat and root != Path(sel.output_dir):
            continue
        if has_man or root == Path(sel.output_dir):
            inv = inventory_reusable_transforms(
                root,
                feature_sets=sel.feature_sets,
                embeddings=embeddings,
                decode_windows=sel.decode_windows,
                n_components=sel.manifold_n_components or (3,),
            )
            requested = inv["n_requested"]
            for hit in inv["hits"]:
                key = (
                    hit["feature_set"],
                    hit["embedding_type"],
                    hit["decode_window_s"],
                    hit["n_components"],
                )
                existing_keys = {
                    (
                        h["feature_set"],
                        h["embedding_type"],
                        h["decode_window_s"],
                        h["n_components"],
                    )
                    for h in combined_hits
                }
                if key not in existing_keys:
                    combined_hits.append(hit)
        if has_feat or root == Path(sel.output_dir):
            finv = inventory_reusable_feature_transforms(
                root,
                feature_sets=sel.feature_sets,
                decode_windows=sel.decode_windows,
                feature_types=("counts",),
            )
            feature_requested = finv["n_requested"]
            for hit in finv["hits"]:
                key = (
                    hit["feature_set"],
                    hit["feature_type_eff"],
                    hit["decode_window_s"],
                )
                existing_keys = {
                    (
                        h["feature_set"],
                        h["feature_type_eff"],
                        h["decode_window_s"],
                    )
                    for h in combined_feature_hits
                }
                if key not in existing_keys:
                    combined_feature_hits.append(hit)

    if requested == 0:
        pairs = valid_feature_manifold_pairs(sel.feature_sets, embeddings)
        requested = (
            max(len(pairs), 0)
            * max(len(sel.decode_windows), 1)
            * max(len(sel.manifold_n_components or (3,)), 1)
        )
    if feature_requested == 0:
        feature_requested = (
            max(len(sel.feature_sets), 0) * max(len(sel.decode_windows), 1)
        )
    return {
        "n_requested": requested,
        "n_reusable": len(combined_hits),
        "n_missing": max(requested - len(combined_hits), 0),
        "hits": combined_hits,
        "n_feature_requested": feature_requested,
        "n_feature_reusable": len(combined_feature_hits),
        "n_feature_missing": max(feature_requested - len(combined_feature_hits), 0),
        "feature_hits": combined_feature_hits,
        "search_roots": [str(r) for r in unique_roots],
        "reuse_enabled": bool(getattr(sel, "reuse_transforms", True)),
    }


def list_benchmark_runs(experiment_dir: Path) -> list[dict[str, Any]]:
    """Discover prior decoder-comparison runs via ``hippo_run_metadata.json``."""
    from ui.services.registry import discover_runs
    from ui.services.results import find_metrics_csv

    exp = Path(experiment_dir)
    runs_meta = discover_runs(search_roots=[exp])
    out: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for meta in runs_meta:
        try:
            inp = Path(meta.input_dataset).resolve()
        except OSError:
            inp = Path(meta.input_dataset)
        try:
            out_p = Path(meta.output_directory).resolve()
        except OSError:
            out_p = Path(meta.output_directory)
        exp_res = exp.resolve() if exp.exists() else exp
        under_exp = (
            inp == exp_res
            or str(out_p).startswith(str(exp_res))
            or "decoder_comparison" in Path(meta.output_directory).parts
        )
        if not under_exp:
            continue
        if meta.run_id in seen_ids:
            continue
        seen_ids.add(meta.run_id)
        metrics = find_metrics_csv(Path(meta.output_directory))
        if metrics is None:
            metrics = find_metrics_csv(exp)
        cfg = meta.configuration or {}
        n_comp = cfg.get("manifold_n_components") or ()
        out.append({
            "run_id": meta.run_id,
            "created_at": meta.timestamp,
            "status": meta.status,
            "spike_source": meta.spike_source,
            "feature_sets": list(meta.feature_sets or []),
            "manifolds": list(meta.manifolds or []),
            "decode_windows": [float(w) for w in (meta.decode_windows or [])],
            "n_components": [int(k) for k in n_comp] if n_comp else [3],
            "output_directory": meta.output_directory,
            "has_metrics": metrics is not None and Path(metrics).exists(),
            "metrics_path": str(metrics) if metrics else None,
            "family": (cfg.get("out_name") or Path(meta.output_directory).name),
        })
    out.sort(key=lambda r: str(r.get("created_at") or r.get("run_id") or ""), reverse=True)
    return out


def selection_already_covered(
    runs: list[dict[str, Any]],
    *,
    feature_sets: list[str] | tuple[str, ...],
    manifolds: list[str] | tuple[str, ...],
    decode_windows: list[float] | tuple[float, ...],
    n_components: list[int] | tuple[int, ...] = (3,),
    spike_source: str | None = None,
) -> dict[str, Any]:
    """Whether the current feature×manifold×window×k grid appears in prior runs."""
    from realtime.search_space import resolve_manifold_alias

    def _norm_m(name: str) -> str:
        emb = resolve_manifold_alias(name)
        return "counts" if emb == "identity" else emb

    ks = [int(k) for k in (n_components or (3,))]
    wanted = {
        (str(fs), _norm_m(str(m)), float(w), int(k))
        for fs in feature_sets
        for m in manifolds
        for w in decode_windows
        for k in ks
    }
    if not wanted:
        return {
            "n_wanted": 0,
            "n_covered": 0,
            "covered": set(),
            "missing": set(),
            "matching_run_ids": [],
            "fully_covered": False,
        }

    covered: set[tuple] = set()
    matching_run_ids: list[str] = []
    for run in runs:
        if spike_source and run.get("spike_source") and run["spike_source"] != spike_source:
            continue
        run_ks = [int(k) for k in (run.get("n_components") or [3])]
        pairs = {
            (str(fs), _norm_m(str(m)), float(w), int(k))
            for fs in (run.get("feature_sets") or [])
            for m in (run.get("manifolds") or [])
            for w in (run.get("decode_windows") or [])
            for k in run_ks
        }
        hit = wanted & pairs
        if hit:
            matching_run_ids.append(str(run["run_id"]))
            covered |= hit

    missing = wanted - covered
    return {
        "n_wanted": len(wanted),
        "n_covered": len(covered),
        "covered": covered,
        "missing": missing,
        "matching_run_ids": matching_run_ids,
        "fully_covered": not missing,
    }


def selection_to_dict(sel: UIBenchmarkSelection) -> dict[str, Any]:
    d = asdict(sel)
    d["input_dir"] = str(sel.input_dir)
    d["output_dir"] = str(sel.output_dir)
    d["resolved_embeddings"] = list(resolve_ui_manifolds(sel.manifolds))
    return d


def run_benchmark(sel: UIBenchmarkSelection, progress_callback=None):
    """Execute comparison (or compare-sources) from a UI selection."""
    # Background Streamlit workers often keep stale science modules in memory.
    # Reload so PLS / decoder_names / skip-guards pick up disk fixes mid-session.
    import importlib
    import realtime.decoder_models as _dm
    import realtime.manifold_features as _mf
    import realtime.decoder_comparison as _dc

    importlib.reload(_dm)
    importlib.reload(_mf)
    _dc = importlib.reload(_dc)

    errors = validate_benchmark_selection(sel)
    if errors:
        raise ValueError("; ".join(errors))
    from realtime.transform_cache import assert_cached_decode_windows

    assert_cached_decode_windows(
        Path(sel.input_dir),
        tuple(float(w) for w in sel.decode_windows),
        spike_source=str(sel.spike_source),
    )
    Path(sel.output_dir).mkdir(parents=True, exist_ok=True)
    search_roots: tuple[str, ...] = ()
    try:
        from realtime.transform_cache import discover_comparison_roots

        search_roots = tuple(
            str(p)
            for p in discover_comparison_roots(Path(sel.input_dir))
            if Path(p).resolve() != Path(sel.output_dir).resolve()
        )
    except Exception:
        search_roots = ()
    if sel.compare_sources:
        embeddings = resolve_ui_manifolds(sel.manifolds)
        extras: dict[str, Any] = {}
        cfg_fields = _dc.ComparisonRunConfig.__dataclass_fields__
        if "decoder_names" in cfg_fields and sel.decoder_names:
            extras["decoder_names"] = tuple(sel.decoder_names)
        if "reuse_transforms" in cfg_fields:
            extras["reuse_transforms"] = bool(getattr(sel, "reuse_transforms", True))
        if "reuse_search_roots" in cfg_fields:
            extras["reuse_search_roots"] = search_roots
        return _dc.run_compare_sources(
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
            progress_callback=progress_callback,
            **extras,
        )

    # Rebuild config against the reloaded ComparisonRunConfig class.
    embeddings = resolve_ui_manifolds(sel.manifolds)
    kwargs: dict[str, Any] = dict(
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
    fields = _dc.ComparisonRunConfig.__dataclass_fields__
    if "reuse_transforms" in fields:
        kwargs["reuse_transforms"] = bool(getattr(sel, "reuse_transforms", True))
    if "reuse_search_roots" in fields:
        kwargs["reuse_search_roots"] = search_roots
    if "decoder_names" in fields and sel.decoder_names:
        kwargs["decoder_names"] = tuple(sel.decoder_names)
    if "targets" in fields and getattr(sel, "targets", ()):
        kwargs["targets"] = tuple(sel.targets)
    cfg = _dc.ComparisonRunConfig(**kwargs)
    if sel.decoder_names and not hasattr(cfg, "decoder_names"):
        cfg.decoder_names = tuple(sel.decoder_names)
    try:
        return _dc.run_decoder_comparison(cfg, progress_callback=progress_callback)
    except TypeError:
        return _dc.run_decoder_comparison(cfg)
