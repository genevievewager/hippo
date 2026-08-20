"""Realtime replay helpers for the Streamlit UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from run_realtime_decoding import RealtimeReplayConfig, run_realtime_decoding


@dataclass
class ReplayArtifacts:
    root: Path
    decoded: pd.DataFrame
    metrics: dict[str, Any]
    selected_config: dict[str, Any] | None = None
    closed_loop: pd.DataFrame | None = None


def find_realtime_runs(experiment_dir: Path) -> list[Path]:
    """Directories that contain ``decoded_realtime.csv``."""
    root = Path(experiment_dir) / "realtime_decoding"
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("decoded_realtime.csv")})


def load_replay_artifacts(run_dir: Path) -> ReplayArtifacts:
    run_dir = Path(run_dir)
    decoded_path = run_dir / "decoded_realtime.csv"
    if not decoded_path.exists():
        matches = list(run_dir.rglob("decoded_realtime.csv"))
        if not matches:
            raise FileNotFoundError(f"No decoded_realtime.csv under {run_dir}")
        decoded_path = matches[0]
        run_dir = decoded_path.parent

    decoded = pd.read_csv(decoded_path)
    metrics: dict[str, Any] = {}
    metrics_path = run_dir / "realtime_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    selected = None
    cfg_path = run_dir / "selected_realtime_decoder_config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            selected = json.load(f)
    closed = None
    cl_path = run_dir / "closed_loop_events.csv"
    if cl_path.exists():
        closed = pd.read_csv(cl_path)

    return ReplayArtifacts(
        root=run_dir,
        decoded=decoded,
        metrics=metrics,
        selected_config=selected,
        closed_loop=closed,
    )


def default_comparison_dir(experiment_dir: Path) -> Path | None:
    cand = Path(experiment_dir) / "decoder_comparison" / "sorted"
    if (cand / "decoder_comparison_metrics.csv").exists():
        return cand
    cand2 = Path(experiment_dir) / "decoder_comparison"
    if (cand2 / "decoder_comparison_metrics.csv").exists():
        return cand2
    return None


def build_replay_config(
    *,
    input_dir: Path,
    output_dir: Path,
    spike_source: str = "sorted",
    update_dt: float = 0.025,
    decode_window: float = 0.250,
    use_best_decoder: bool = True,
    closed_loop_target: str = "position",
    comparison_dir: Path | None = None,
    feature_type: str = "counts",
    manifold_n_components: int = 3,
) -> RealtimeReplayConfig:
    if not use_best_decoder:
        from realtime.transform_cache import assert_cached_decode_windows

        assert_cached_decode_windows(
            Path(input_dir),
            (float(decode_window),),
            spike_source=str(spike_source),
        )
    return RealtimeReplayConfig(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir),
        spike_source=spike_source,
        update_dt=float(update_dt),
        decode_window=float(decode_window),
        use_best_decoder=use_best_decoder,
        comparison_dir=comparison_dir or default_comparison_dir(input_dir),
        closed_loop_target=closed_loop_target,
        feature_type=feature_type,
        manifold_n_components=int(manifold_n_components),
    )


def execute_replay(config: RealtimeReplayConfig):
    """Call the shared realtime decoding backend."""
    return run_realtime_decoding(config)


CANONICAL_N_COMPONENTS: tuple[int, ...] = (2, 3, 5, 8, 10)
QUADRANT_SIDECAR_NAME = "quadrant_comparison.json"
_GALLERY_FEATURE_SET = "counts"


def _comparison_search_roots(experiment_dir: Path, spike_source: str) -> list[Path]:
    from realtime.transform_cache import (
        discover_comparison_roots,
        preferred_comparison_root,
    )

    roots: list[Path] = []
    seen: set[Path] = set()
    preferred = preferred_comparison_root(experiment_dir, spike_source=spike_source)
    for root in (
        *([preferred] if preferred is not None else []),
        *discover_comparison_roots(experiment_dir),
    ):
        if root is None:
            continue
        key = root.resolve() if root.exists() else root
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _find_cached_embedding(
    experiment_dir: Path,
    spike_source: str,
    embedding_type: str,
    decode_window: float,
    n_components: int,
) -> Path | None:
    from realtime.search_space import resolve_manifold_alias
    from realtime.transform_cache import find_manifold_transform_in_roots

    roots = _comparison_search_roots(experiment_dir, spike_source)
    if not roots:
        return None
    return find_manifold_transform_in_roots(
        roots,
        feature_set=_GALLERY_FEATURE_SET,
        embedding_type=resolve_manifold_alias(embedding_type),
        decode_window=float(decode_window),
        n_components=int(n_components),
    )


def _load_cached_embedding(
    experiment_dir: Path,
    spike_source: str,
    embedding_type: str,
    decode_window: float,
    n_components: int,
):
    from realtime.transform_cache import try_load_manifold

    hit = _find_cached_embedding(
        experiment_dir, spike_source, embedding_type, decode_window, n_components,
    )
    if hit is None:
        raise FileNotFoundError(
            f"No cached `{embedding_type}` transform for F=`counts` W={decode_window:.3f}s "
            f"k={int(n_components)}. Run Latent Representations for that class first "
            f"(counts × {embedding_type} × {decode_window:.3f}s)."
        )
    loaded = try_load_manifold(Path(hit))
    if loaded is None:
        raise FileNotFoundError(
            f"Cached `{embedding_type}` transform at `{hit}` could not be loaded. "
            "Re-run Latent Representations for that class."
        )
    return loaded, Path(hit)


def list_cached_n_components(
    experiment_dir: Path,
    spike_source: str,
    decode_window: float,
    embeddings: list[str] | tuple[str, ...] | None = None,
) -> list[int]:
    """k values that have a cached E for every requested embedding at W.

    Defaults to the three realtime quadrant embeddings. A k is listed only
    when all of them are on disk so a full quadrant run can proceed.
    """
    from ui.services.representations import REALTIME_QUADRANT_DEFAULTS, QUADRANT_ORDER

    if embeddings is None:
        embeddings = tuple(
            str(REALTIME_QUADRANT_DEFAULTS[q])
            for q in QUADRANT_ORDER
            if REALTIME_QUADRANT_DEFAULTS.get(q)
        )
    found: list[int] = []
    for k in CANONICAL_N_COMPONENTS:
        if all(
            _find_cached_embedding(
                experiment_dir, spike_source, str(emb), float(decode_window), int(k),
            ) is not None
            for emb in embeddings
        ):
            found.append(int(k))
    return found


_REPLAY_WINDOWS: tuple[float, ...] = (0.025, 0.050, 0.100, 0.250, 0.500, 1.000)


def _quadrant_embeddings(
    embeddings: list[str] | tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    from ui.services.representations import REALTIME_QUADRANT_DEFAULTS, QUADRANT_ORDER

    if embeddings is not None:
        return tuple(str(e) for e in embeddings)
    return tuple(
        str(REALTIME_QUADRANT_DEFAULTS[q])
        for q in QUADRANT_ORDER
        if REALTIME_QUADRANT_DEFAULTS.get(q)
    )


def _decoder_suite_on_disk(
    models_dir: Path | None,
    *,
    embedding_type: str,
    decode_window: float,
    n_components: int,
) -> bool:
    """True when every suite target has a windowed joblib for this E × W × k."""
    if models_dir is None:
        return False
    from realtime.best_decoder_selection import SUITE_TARGETS
    from realtime.decoder_comparison import windowed_model_path
    from realtime.search_space import resolve_manifold_alias

    feat = resolve_manifold_alias(embedding_type)
    return all(
        windowed_model_path(
            models_dir, target, float(decode_window), feat, int(n_components),
        ).exists()
        for target in SUITE_TARGETS
    )


def list_replay_ready_windows(
    experiment_dir: Path,
    spike_source: str,
    *,
    n_components: int = 3,
    embeddings: list[str] | tuple[str, ...] | None = None,
) -> list[float]:
    """W values that have E and a full D suite for every realtime quadrant class."""
    embs = _quadrant_embeddings(embeddings)
    k = int(n_components)
    cmp = default_comparison_dir(Path(experiment_dir))
    models_dir = _comparison_models_dir(cmp, spike_source)
    ready: list[float] = []
    for w in _REPLAY_WINDOWS:
        if not all(
            _find_cached_embedding(
                experiment_dir, spike_source, str(emb), float(w), k,
            ) is not None
            for emb in embs
        ):
            continue
        if not all(
            _decoder_suite_on_disk(
                models_dir,
                embedding_type=str(emb),
                decode_window=float(w),
                n_components=k,
            )
            for emb in embs
        ):
            continue
        ready.append(float(w))
    return ready


def _estimator_n_features_in(est) -> int | None:
    if est is None:
        return None
    n = getattr(est, "n_features_in_", None)
    if n is not None:
        try:
            return int(n)
        except (TypeError, ValueError):
            pass
    steps = getattr(est, "steps", None)
    if steps:
        for _, step in steps:
            n = _estimator_n_features_in(step)
            if n is not None:
                return n
    inner = getattr(est, "estimator_", None) or getattr(est, "named_steps", None)
    if inner is not None and inner is not est:
        if isinstance(inner, dict):
            for step in inner.values():
                n = _estimator_n_features_in(step)
                if n is not None:
                    return n
        else:
            return _estimator_n_features_in(inner)
    return None


def _transformer_n_features_out(transformer, fallback: int | None = None) -> int | None:
    if transformer is None:
        return int(fallback) if fallback is not None else None
    for attr in (
        "n_features_out_",
        "actual_n_features_",
        "actual_n_components_",
        "latent_dim",
    ):
        v = getattr(transformer, attr, None)
        if callable(v):
            try:
                v = v()
            except Exception:
                continue
        if isinstance(v, (int, float)) and v == v and int(v) > 0:
            return int(v)
    for inner_name in ("pca_", "model_", "encoder_", "_model"):
        inner = getattr(transformer, inner_name, None)
        if inner is not None:
            n = getattr(inner, "n_components_", None) or getattr(inner, "n_components", None)
            if isinstance(n, (int, float)) and n == n and int(n) > 0:
                return int(n)
    return int(fallback) if fallback is not None else None


def _features_compatible(est, expected: int | None) -> bool:
    if expected is None:
        return True
    got = _estimator_n_features_in(est)
    if got is None:
        return True
    return int(got) == int(expected)


def _comparison_models_dir(comparison_root: Path | None, spike_source: str) -> Path | None:
    if comparison_root is None:
        return None
    root = Path(comparison_root)
    for cand in (
        root / "models",
        root / spike_source / "models",
        root,
    ):
        if cand.is_dir():
            return cand
    return None


def _try_load_decoder_suite(
    models_dir: Path | None,
    *,
    embedding_type: str,
    decode_window: float,
    n_components: int,
    target: str,
    feature_transformer=None,
):
    """Load D only for this embedding × W × k. Never fall back to counts.

    If a cached head's ``n_features_in_`` does not match the frozen embedding
    output dim, discard it and let the caller train on frozen E.
    """
    if models_dir is None:
        return None, None
    from realtime.best_decoder_selection import load_pretrained_suite, load_windowed_model
    from realtime.search_space import resolve_manifold_alias

    emb = resolve_manifold_alias(embedding_type)
    expected = _transformer_n_features_out(feature_transformer, fallback=int(n_components))
    suite = None
    primary = None
    try:
        suite = load_pretrained_suite(
            models_dir, decode_window, feature_type=emb, n_components=n_components,
        )
    except FileNotFoundError:
        suite = None
    if suite is not None:
        heads = (
            suite.position,
            suite.spatial_context,
            suite.movement_state,
            suite.speed,
        )
        if any(not _features_compatible(h, expected) for h in heads):
            suite = None
    try:
        primary = load_windowed_model(
            models_dir, target, decode_window, emb, n_components,
        )
    except FileNotFoundError:
        primary = None
    if primary is not None and not _features_compatible(primary, expected):
        primary = None
    return suite, primary


def _write_quadrant_figures(
    input_dir: Path,
    *,
    decode_window: float,
    target: str,
    decoded_by_q: dict[str, pd.DataFrame | None],
    metrics_by_q: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    from visualization.publication_quadrant_plots import (
        plot_quadrant_behavior,
        plot_quadrant_stability,
    )

    fig_dir = Path(input_dir) / "figures" / "realtime_decoding"
    fig_dir.mkdir(parents=True, exist_ok=True)
    stab_path = find_stability_csv(input_dir)
    stab_df = pd.read_csv(stab_path) if stab_path is not None else None
    stability_png = plot_quadrant_stability(
        stab_df, fig_dir / "fig_quadrant_stability.png",
        decode_window_s=float(decode_window),
    )
    behavior_png = plot_quadrant_behavior(
        decoded_by_q, metrics_by_q, fig_dir / "fig_quadrant_behavior.png",
        target=target,
    )
    return Path(stability_png), Path(behavior_png)


def find_stability_csv(experiment_dir: Path) -> Path | None:
    matches = sorted(Path(experiment_dir).glob("decoder_comparison/**/latent_stability_metrics.csv"))
    return matches[0] if matches else None


def load_quadrant_sidecar(experiment_dir: Path) -> dict[str, Any] | None:
    path = Path(experiment_dir) / "realtime_decoding" / QUADRANT_SIDECAR_NAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_quadrant_comparison(
    *,
    input_dir: Path,
    spike_source: str,
    target: str,
    decode_window: float,
    update_dt: float = 0.050,
    manifold_n_components: int = 3,
    progress_callback=None,
) -> dict[str, Any]:
    """Replay the three realtime-capable quadrant defaults and write two figures."""
    from realtime.evaluate_realtime import run_realtime_pipeline
    from realtime.transform_cache import assert_cached_decode_windows
    from ui.services.representations import (
        QUADRANT_ORDER,
        REALTIME_QUADRANT_DEFAULTS,
        REPRESENTATION_QUADRANT_LABELS,
    )

    input_dir = Path(input_dir)
    assert_cached_decode_windows(
        input_dir, (float(decode_window),), spike_source=str(spike_source),
    )
    cmp = default_comparison_dir(input_dir)
    models_dir = _comparison_models_dir(cmp, spike_source)
    occupied = [
        q for q in QUADRANT_ORDER if REALTIME_QUADRANT_DEFAULTS.get(q)
    ]
    total = max(len(occupied), 1)
    runs: dict[str, Any] = {}
    decoded_by_q: dict[str, pd.DataFrame | None] = {
        "dynamic_nonlinear": None,
    }
    metrics_by_q: dict[str, dict[str, Any]] = {}
    k = int(manifold_n_components)
    stability_png: Path | None = None
    behavior_png: Path | None = None

    for i, qid in enumerate(occupied, start=1):
        emb = REALTIME_QUADRANT_DEFAULTS[qid]
        label = REPRESENTATION_QUADRANT_LABELS[qid]
        transformer, cached_path = _load_cached_embedding(
            input_dir, spike_source, str(emb), float(decode_window), k,
        )
        suite, primary = _try_load_decoder_suite(
            models_dir,
            embedding_type=str(emb),
            decode_window=float(decode_window),
            n_components=k,
            target=target,
            feature_transformer=transformer,
        )
        if suite is None or primary is None:
            from realtime.decoder_comparison import windowed_model_path
            from realtime.search_space import resolve_manifold_alias

            feat = resolve_manifold_alias(str(emb))
            missing = (
                windowed_model_path(
                    models_dir, target, float(decode_window), feat, k,
                )
                if models_dir is not None
                else None
            )
            raise FileNotFoundError(
                f"No Decoder Benchmark head for `{emb}` at W={decode_window:.3f}s "
                f"k={k} target={target}"
                + (f": {missing}" if missing is not None else ".")
                + " Run Decoder Benchmark with this representation and window."
            )
        if progress_callback:
            progress_callback(
                f"Replay {label} ({emb}) · loaded E from cache · loaded D from cache",
                i,
                total,
            )
        out_dir = input_dir / "realtime_decoding" / "quadrants" / qid
        run_realtime_pipeline(
            input_dir=input_dir,
            output_dir=out_dir / spike_source,
            spike_source=spike_source,
            update_dt=float(update_dt),
            decode_window=float(decode_window),
            closed_loop_target=target,
            feature_type=str(emb),
            manifold_n_components=k,
            feature_transformer=transformer,
            pretrained_decoders=suite,
            primary_model=primary,
        )
        arts = load_replay_artifacts(out_dir)
        decoded_by_q[qid] = arts.decoded
        metrics_by_q[qid] = dict(arts.metrics or {})
        runs[qid] = {
            "representation": emb,
            "label": label,
            "output_dir": str(out_dir),
            "cached_embedding": str(cached_path),
            "decoders_from_cache": True,
            "decoder_source": "cached",
            "n_components": k,
        }
        stability_png, behavior_png = _write_quadrant_figures(
            input_dir,
            decode_window=float(decode_window),
            target=target,
            decoded_by_q=decoded_by_q,
            metrics_by_q=metrics_by_q,
        )

    if stability_png is None or behavior_png is None:
        stability_png, behavior_png = _write_quadrant_figures(
            input_dir,
            decode_window=float(decode_window),
            target=target,
            decoded_by_q=decoded_by_q,
            metrics_by_q=metrics_by_q,
        )
    stab_csv = find_stability_csv(input_dir)
    sidecar = {
        "target": target,
        "decode_window_s": float(decode_window),
        "update_dt_s": float(update_dt),
        "n_components": k,
        "spike_source": spike_source,
        "runs": runs,
        "figures": {
            "stability": str(stability_png),
            "behavior": str(behavior_png),
        },
        "stability_csv": str(stab_csv) if stab_csv else None,
    }
    side_path = input_dir / "realtime_decoding" / QUADRANT_SIDECAR_NAME
    side_path.parent.mkdir(parents=True, exist_ok=True)
    side_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    sidecar["sidecar_path"] = str(side_path)
    return sidecar


RIDGE_DIAG_EMBEDDINGS: tuple[str, ...] = (
    "global_pca",
    "region_pca",
    "diffusion_nystrom",
    "global_lds",
)
QUADRANT_RIDGE_SIDECAR_NAME = "quadrant_ridge_summary.json"


def list_ridge_diag_ready_windows(
    experiment_dir: Path,
    spike_source: str,
    *,
    n_components: int = 3,
) -> list[float]:
    """Decode windows with cached E for all ridge-diagnostic latent representations."""
    from realtime.transform_cache import list_cached_decode_windows

    k = int(n_components)
    ready: list[float] = []
    for w in list_cached_decode_windows(experiment_dir, spike_source=spike_source):
        if all(
            _find_cached_embedding(
                experiment_dir, spike_source, emb, float(w), k,
            ) is not None
            for emb in RIDGE_DIAG_EMBEDDINGS
        ):
            ready.append(float(w))
    return ready


def load_quadrant_ridge_sidecar(experiment_dir: Path) -> dict[str, Any] | None:
    path = (
        Path(experiment_dir)
        / "realtime_decoding"
        / "quadrant_ridge"
        / QUADRANT_RIDGE_SIDECAR_NAME
    )
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_quadrant_ridge_diagnostics(
    *,
    input_dir: Path,
    spike_source: str,
    decode_window: float,
    update_dt: float = 0.050,
    n_components: int = 3,
    progress_callback=None,
) -> dict[str, Any]:
    """Ridge-only center-bias diagnostics across quadrant representations."""
    from realtime.quadrant_ridge_diagnostics import run_quadrant_ridge_diagnostics as _run
    from realtime.transform_cache import assert_cached_decode_windows

    input_dir = Path(input_dir)
    assert_cached_decode_windows(
        input_dir, (float(decode_window),), spike_source=str(spike_source),
    )
    return _run(
        input_dir=input_dir,
        spike_source=spike_source,
        decode_window=float(decode_window),
        update_dt=float(update_dt),
        n_components=int(n_components),
        progress_callback=progress_callback,
    )
