"""Page: Latent Representations — four E-class quadrants and per-tab galleries.

Fits write ``decoder_comparison/<spike_source>/models/manifold_transforms/`` so
Decoder Benchmark can reuse them via ``reuse_transforms=True`` (default).
Galleries are display-only: ``counts`` at 250 ms, colored by behavior.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from realtime.decoder_comparison import (
    ALL_TARGETS,
    CATEGORICAL_TARGETS,
    CONTINUOUS_TARGETS,
)
from ui.components.controls import (
    active_spike_source,
    require_active_dataset,
)
from ui.components.run_status import (
    render_job_autofresh,
    render_run_action_row,
    render_workload_estimate,
)
from ui.jobs import get_slot_job, submit_job
from ui.services.comparison import (
    DECODE_WINDOW_LABELS,
    UI_FEATURE_SET_OPTIONS,
    apply_remaining_work_to_workload,
    format_decode_window,
    valid_feature_manifold_pairs,
)
from ui.services.manifold_analysis import (
    ManifoldAnalysisRequest,
    estimate_manifold_workload,
    list_manifold_analysis_runs,
    run_manifold_analysis,
    selection_already_covered,
)
from ui.services.manifolds import (
    FALLBACK_MANIFOLD_EMBEDDING,
    FALLBACK_N_COMPONENTS,
    FALLBACK_WINDOW_S,
    TARGET_LABELS,
)
from ui.services.representations import (
    QUADRANT_ORDER,
    REPRESENTATION_QUADRANT_LABELS,
    REPRESENTATION_QUADRANTS,
    format_representation_label,
)
from ui.services.results import find_metrics_csv
from ui import state

logger = logging.getLogger(__name__)

_JOB_SLOT = "manifold:analysis"
_N_WINNER_EXPECTED = len(ALL_TARGETS) * 2  # counts + manifold per target
_PENDING_KEY = "man_pending_request"
_QUADRANT_KEY = "man_quadrant_id"
_DURABLE_KEY = "man_quadrant_control_state"
_GALLERY_FEATURE_SET = "counts"
_GALLERY_WINDOW_S = FALLBACK_WINDOW_S
_GALLERY_TARGETS: tuple[str, ...] = CONTINUOUS_TARGETS + CATEGORICAL_TARGETS


def _status_mtime_token(dataset: Path) -> str:
    parts: list[str] = []
    for rel in (
        "manifolds",
        "figures/manifolds",
        "figures/decoder_comparison",
        "decoder_comparison",
    ):
        p = Path(dataset) / rel
        try:
            parts.append(str(p.stat().st_mtime_ns) if p.exists() else "0")
        except OSError:
            parts.append("0")
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _cached_manifold_runs(experiment_dir: str, mtime_token: str) -> list[dict]:
    del mtime_token
    return list_manifold_analysis_runs(Path(experiment_dir))


@st.cache_data(show_spinner=False)
def _cached_session_duration_s(experiment_dir: str, summary_mtime: str) -> float | None:
    del summary_mtime
    summary_path = Path(experiment_dir) / "summary.json"
    if not summary_path.exists():
        return None
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    raw = summary.get("session_duration_s", summary.get("session_duration"))
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _has_comparison_metrics(dataset: Path) -> bool:
    return find_metrics_csv(dataset) is not None


def _get_prior_runs(dataset: Path) -> list[dict]:
    return _cached_manifold_runs(str(dataset.resolve()), _status_mtime_token(dataset))


def _control_keys(quadrant_id: str) -> tuple[str, str, str, str]:
    prefix = f"man_{quadrant_id}"
    return f"{prefix}_fs", f"{prefix}_mans", f"{prefix}_k", f"{prefix}_wins"


def _snapshot_quadrant_controls() -> None:
    durable = st.session_state.setdefault(_DURABLE_KEY, {})
    for qid in QUADRANT_ORDER:
        fs_key, mans_key, k_key, wins_prefix = _control_keys(qid)
        if fs_key not in st.session_state and mans_key not in st.session_state:
            continue
        wins = {
            key: st.session_state[key]
            for key in list(st.session_state.keys())
            if str(key).startswith(f"{wins_prefix}_")
        }
        durable[qid] = {
            "fs": list(st.session_state.get(fs_key, ["counts"])),
            "mans": list(st.session_state.get(mans_key, [])),
            "k": st.session_state.get(k_key, 3),
            "wins": wins,
        }


def _restore_quadrant_controls(quadrant_id: str, methods: list[str]) -> None:
    fs_key, mans_key, k_key, _wins_prefix = _control_keys(quadrant_id)
    saved = (st.session_state.get(_DURABLE_KEY) or {}).get(quadrant_id)
    k_options = (2, 3, 5, 8, 10)
    if saved:
        if fs_key not in st.session_state:
            fs = [x for x in saved.get("fs") or [] if x in UI_FEATURE_SET_OPTIONS]
            st.session_state[fs_key] = fs or ["counts"]
        if mans_key not in st.session_state:
            mans = [m for m in saved.get("mans") or [] if m in methods]
            st.session_state[mans_key] = mans or list(methods)
        if k_key not in st.session_state:
            k = saved.get("k", 3)
            st.session_state[k_key] = k if k in k_options else 3
        for wk, wv in (saved.get("wins") or {}).items():
            if wk not in st.session_state:
                st.session_state[wk] = wv
        return
    if fs_key not in st.session_state:
        st.session_state[fs_key] = ["counts"]
    if mans_key not in st.session_state:
        st.session_state[mans_key] = list(methods)
    if k_key not in st.session_state:
        st.session_state[k_key] = 3


def _render_quadrant_selector() -> str:
    if _QUADRANT_KEY not in st.session_state:
        st.session_state[_QUADRANT_KEY] = QUADRANT_ORDER[0]
    if st.session_state[_QUADRANT_KEY] not in QUADRANT_ORDER:
        st.session_state[_QUADRANT_KEY] = QUADRANT_ORDER[0]
    return st.radio(
        "Representation class",
        options=list(QUADRANT_ORDER),
        format_func=lambda q: REPRESENTATION_QUADRANT_LABELS[q],
        horizontal=True,
        key=_QUADRANT_KEY,
    )


def _gated_windows_for_page(
    cached_windows: list[float] | tuple[float, ...],
    *,
    key: str,
    defaults: list[float],
) -> list[float]:
    from realtime.transform_cache import window_ms

    options = list(DECODE_WINDOW_LABELS.keys())
    cached_ms = {window_ms(w) for w in cached_windows}
    default_ms = {window_ms(w) for w in defaults}
    st.markdown("**Compare windows**")
    selected: list[float] = []
    cols = st.columns(len(options))
    for col, w in zip(cols, options):
        enabled = window_ms(w) in cached_ms
        ck = f"{key}_{window_ms(w):04d}"
        if ck not in st.session_state:
            st.session_state[ck] = bool(enabled and window_ms(w) in default_ms)
        with col:
            checked = st.checkbox(
                format_decode_window(w),
                disabled=not enabled,
                key=ck,
                help=(
                    None
                    if enabled
                    else "Generate this window on Feature Construction first."
                ),
            )
        if checked and enabled:
            selected.append(float(w))
    if not cached_ms:
        st.caption("Generate windows on **Feature Construction** first.")
    else:
        missing = [
            format_decode_window(w)
            for w in options
            if window_ms(w) not in cached_ms
        ]
        if missing:
            st.caption("Not yet generated (grey): " + ", ".join(missing))
    return selected


def _gallery_search_roots(dataset: Path, spike_source: str) -> list[Path]:
    from realtime.transform_cache import (
        discover_comparison_roots,
        preferred_comparison_root,
    )

    roots: list[Path] = []
    seen: set[Path] = set()
    preferred = preferred_comparison_root(dataset, spike_source=spike_source)
    for root in (
        *([preferred] if preferred is not None else []),
        *discover_comparison_roots(dataset),
    ):
        if root is None:
            continue
        key = root.resolve() if root.exists() else root
        if key in seen:
            continue
        seen.add(key)
        roots.append(root)
    return roots


def _find_gallery_transform(
    dataset: Path,
    spike_source: str,
    embedding_type: str,
    n_components: int,
):
    from realtime.manifold_features import DEFAULT_ISOMAP_N_NEIGHBORS
    from realtime.search_space import resolve_manifold_alias
    from realtime.transform_cache import find_manifold_transform_in_roots

    emb = resolve_manifold_alias(embedding_type)
    roots = _gallery_search_roots(dataset, spike_source)
    if not roots:
        return None
    return find_manifold_transform_in_roots(
        roots,
        feature_set=_GALLERY_FEATURE_SET,
        embedding_type=emb,
        decode_window=_GALLERY_WINDOW_S,
        n_components=int(n_components),
        n_neighbors=DEFAULT_ISOMAP_N_NEIGHBORS,
    )


@st.cache_data(show_spinner=False)
def _cached_gallery_embedding(
    experiment_dir: str,
    spike_source: str,
    embedding_type: str,
    n_components: int,
    mtime_token: str,
) -> dict | None:
    del mtime_token
    dataset = Path(experiment_dir)
    hit = _find_gallery_transform(dataset, spike_source, embedding_type, n_components)
    if hit is None:
        return None
    from realtime.search_space import resolve_manifold_alias
    from realtime.transform_cache import try_load_manifold
    from ui.services.manifolds import compute_manifold_diagnostics

    emb = resolve_manifold_alias(embedding_type)
    if emb not in ("identity", "counts") and try_load_manifold(Path(hit)) is None:
        return None
    comparison_root = Path(hit).parent.parent.parent
    try:
        diag = compute_manifold_diagnostics(
            dataset,
            embedding_type,
            feature_set=_GALLERY_FEATURE_SET,
            spike_source=spike_source,
            decode_window=_GALLERY_WINDOW_S,
            n_components=int(n_components),
            max_samples=2500,
            comparison_root=comparison_root,
            persist=False,
            force_refit=False,
        )
    except Exception:
        return None
    if not diag.from_cache and emb not in ("identity", "counts"):
        return None
    return {
        "embedding_type": diag.embedding_type,
        "latent": diag.latent,
        "behavior": diag.behavior,
        "from_cache": bool(diag.from_cache),
    }


def _load_gallery_embedding(
    dataset: Path,
    spike_source: str,
    embedding_type: str,
    n_components: int,
) -> dict | None:
    payload = _cached_gallery_embedding(
        str(dataset.resolve()),
        spike_source,
        embedding_type,
        int(n_components),
        _status_mtime_token(dataset),
    )
    if payload is not None:
        return payload
    if int(n_components) != int(FALLBACK_N_COMPONENTS):
        return _cached_gallery_embedding(
            str(dataset.resolve()),
            spike_source,
            embedding_type,
            int(FALLBACK_N_COMPONENTS),
            _status_mtime_token(dataset),
        )
    return None


def _plot_gallery_panel(payload: dict, target: str, title: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from visualization.publication_isomap_plots import (
        _color_for_feature,
        _scatter_latent,
        _scatter_position_2d,
        to_display_latent_2d,
    )

    Z = to_display_latent_2d(
        np.asarray(payload["latent"], dtype=float),
        str(payload.get("embedding_type") or "counts"),
    )
    beh = payload["behavior"]
    if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[1] < 2:
        st.caption("Embedding is not 2-D.")
        return
    n = min(len(beh), len(Z))
    Z = Z[:n]
    beh = beh.iloc[:n].reset_index(drop=True)
    color = _color_for_feature(beh, target)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    if color is None:
        ax.scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.65, c="0.4")
        ax.set_xlabel("z₁")
        ax.set_ylabel("z₂")
    elif isinstance(color, str) and color == "position_xy":
        _scatter_position_2d(ax, Z, beh)
    else:
        _scatter_latent(ax, Z, color, label=TARGET_LABELS.get(target, target))
    ax.set_title(title, fontsize=10)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def _render_quadrant_gallery(
    dataset: Path,
    spike_source: str,
    *,
    methods: list[str],
    n_components: int,
    label: str,
) -> None:
    st.subheader(f"{label} gallery")
    st.caption(
        "Locked to **F=`counts` · W=250 ms** for every quadrant so classes are "
        "comparable. Color order: position first, then other continuous behaviors, "
        "then discrete. Missing checkpoints use counts vs `global_pca` at 250 ms."
    )
    payloads: dict[str, dict | None] = {}
    with st.spinner("Loading cached counts embeddings at 250 ms…"):
        for method in methods:
            payloads[method] = _load_gallery_embedding(
                dataset, spike_source, method, n_components,
            )
        fallback_counts = _load_gallery_embedding(
            dataset, spike_source, "counts", FALLBACK_N_COMPONENTS,
        )
        fallback_pca = _load_gallery_embedding(
            dataset, spike_source, FALLBACK_MANIFOLD_EMBEDDING, FALLBACK_N_COMPONENTS,
        )

    any_loaded = any(p is not None for p in payloads.values())
    if not any_loaded and fallback_counts is None and fallback_pca is None:
        st.info(
            "No `counts` embeddings at 250 ms in the transform cache yet. "
            "Run this class (include `counts` and 250 ms) to fill the gallery."
        )
        return

    for target in _GALLERY_TARGETS:
        st.markdown(f"**{TARGET_LABELS.get(target, target)}**")
        cols = st.columns(max(len(methods), 1))
        for col, method in zip(cols, methods):
            with col:
                st.caption(format_representation_label(method))
                payload = payloads.get(method)
                used_fallback = False
                if payload is None:
                    if method in ("counts", "identity") and fallback_counts is not None:
                        payload = fallback_counts
                        used_fallback = True
                    elif fallback_pca is not None:
                        payload = fallback_pca
                        used_fallback = True
                    elif fallback_counts is not None:
                        payload = fallback_counts
                        used_fallback = True
                if payload is None:
                    st.caption("Not on disk at counts · 250 ms.")
                    continue
                title = method
                if used_fallback:
                    title = f"{method} (fallback)"
                    st.caption("Fallback: counts vs global_pca @ 250 ms")
                _plot_gallery_panel(payload, target, title)


def render(outputs_root: Path) -> None:
    st.header("Latent Representations")
    st.caption(
        "Fit and inspect population-state representations `E` in four classes: "
        "static vs dynamic × linear vs nonlinear. Checkpoints go to the shared "
        "transform cache for Decoder Benchmark."
    )

    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return
    spike_source = active_spike_source(dataset, readonly=True)

    _render_job_status(dataset, spike_source)

    _snapshot_quadrant_controls()
    qid = _render_quadrant_selector()
    _render_quadrant_tab(dataset, spike_source, qid)

    _submit_pending_job()


def _render_quadrant_tab(dataset: Path, spike_source: str, quadrant_id: str) -> None:
    methods = list(REPRESENTATION_QUADRANTS[quadrant_id])
    label = REPRESENTATION_QUADRANT_LABELS[quadrant_id]
    if not methods:
        st.info(
            f"**{label}** is not implemented. Reserved for a future dynamic "
            "nonlinear method (for example LFADS or switching LDS). "
            "GPFA is a linear-Gaussian latent and lives under Dynamic linear."
        )
        return

    _render_run_analysis(
        dataset,
        spike_source,
        quadrant_id=quadrant_id,
        methods=methods,
        label=label,
    )


def _render_run_analysis(
    dataset: Path,
    spike_source: str,
    *,
    quadrant_id: str,
    methods: list[str],
    label: str,
) -> None:
    st.subheader(f"Run {label}")
    st.markdown(
        f"**Active dataset:** `{dataset.name}` · spike source `{spike_source}`. "
        "Each selected feature × representation × window is fitted once and written to "
        "the shared transform cache for Decoder Benchmark. After fits, winner PNGs "
        "are regenerated under `figures/manifolds/`."
    )

    prefix = f"man_{quadrant_id}"
    fs_key, mans_key, k_key, wins_prefix = _control_keys(quadrant_id)
    _restore_quadrant_controls(quadrant_id, methods)
    if fs_key in st.session_state:
        st.session_state[fs_key] = [
            x for x in list(st.session_state[fs_key]) if x in UI_FEATURE_SET_OPTIONS
        ] or ["counts"]
    if mans_key in st.session_state:
        kept = [m for m in list(st.session_state[mans_key]) if m in methods]
        st.session_state[mans_key] = kept or list(methods)
    feature_sets = st.multiselect(
        "Feature sets",
        options=list(UI_FEATURE_SET_OPTIONS),
        key=fs_key,
        help="Groupwise PCA (region) requires the `counts` feature set.",
    )
    manifolds = st.multiselect(
        "Representations",
        options=methods,
        key=mans_key,
    )
    from realtime.transform_cache import list_cached_decode_windows

    cached_windows = list_cached_decode_windows(
        dataset, spike_source=spike_source, feature_sets=feature_sets,
    )
    windows = _gated_windows_for_page(
        cached_windows,
        key=wins_prefix,
        defaults=[0.100, 0.250, 0.500],
    )
    n_components = st.select_slider(
        "Components", options=[2, 3, 5, 8, 10], key=k_key,
    )

    pairs = valid_feature_manifold_pairs(feature_sets, manifolds)
    from realtime.neural_features import embedding_compatible_with_feature_set
    from realtime.search_space import resolve_manifold_alias

    compatible_manifolds = [
        m for m in manifolds
        if any(
            embedding_compatible_with_feature_set(resolve_manifold_alias(m), fs)
            for fs in feature_sets
        )
    ]
    if feature_sets and manifolds and len(compatible_manifolds) < len(manifolds):
        st.warning(
            "Some representations were excluded as incompatible with the selected "
            "feature sets (e.g. region_pca requires `counts`)."
        )

    coverage: dict = {
        "fully_covered": False, "n_covered": 0, "n_wanted": 0, "matching_run_ids": [],
    }
    if feature_sets and compatible_manifolds and windows:
        coverage = selection_already_covered(
            _get_prior_runs(dataset),
            feature_sets=feature_sets,
            manifolds=compatible_manifolds,
            decode_windows=windows,
            spike_source=spike_source,
            n_components=int(n_components),
            input_dir=dataset,
            require_geometry_pages=False,
        )
        if coverage["fully_covered"]:
            st.success(
                f"All **{coverage['n_wanted']}** selected feature×representation×window "
                "checkpoints already exist in the shared transform cache. "
                "Decoder Benchmark can reuse them without refitting."
            )
        elif coverage["n_covered"]:
            st.info(
                f"**{coverage['n_covered']}/{coverage['n_wanted']}** checkpoints "
                "already on disk; the rest will be fitted on run."
            )

    summary_path = dataset / "summary.json"
    try:
        summary_mtime = str(summary_path.stat().st_mtime_ns) if summary_path.exists() else "0"
    except OSError:
        summary_mtime = "0"
    session_duration_s = _cached_session_duration_s(str(dataset.resolve()), summary_mtime)
    workload = estimate_manifold_workload(
        feature_sets=feature_sets,
        manifolds=compatible_manifolds or manifolds,
        decode_windows=windows,
        session_duration_s=session_duration_s,
        has_decoder_comparison=_has_comparison_metrics(dataset),
    )
    if coverage.get("n_wanted"):
        workload = apply_remaining_work_to_workload(
            workload,
            n_wanted=int(coverage["n_wanted"]),
            n_covered=int(coverage.get("n_covered") or 0),
            n_to_compute=coverage.get("n_to_compute"),
            fully_covered=bool(coverage.get("fully_covered")),
        )
    render_workload_estimate(workload)

    existing = get_slot_job(_JOB_SLOT)
    busy = existing is not None and existing.is_active

    run_clicked, force_regenerate = render_run_action_row(
        label=f"Run {label}",
        key_prefix=prefix,
        disabled=not pairs or not windows or busy,
        help="Background job — you can navigate away while it runs.",
        regenerate_required=bool(coverage.get("fully_covered")),
        regenerate_label="Regenerate existing checkpoints",
        regenerate_help="Overwrite shared manifold_transforms for this selection.",
        blocked_caption=(
            "All planned checkpoints already exist — check **Regenerate** "
            "to overwrite. Decoder Benchmark will reuse the existing cache."
        ),
    )
    if run_clicked:
        state.request_action(state.KEY_MANIFOLD_COMPUTE_REQUESTED)
        st.session_state[_PENDING_KEY] = {
            "input_dir": dataset,
            "feature_sets": tuple(feature_sets),
            "manifolds": tuple(m for m in manifolds if m in set(compatible_manifolds)),
            "decode_windows": tuple(float(w) for w in windows),
            "n_components": int(n_components),
            "spike_source": spike_source,
            "force_recompute": bool(force_regenerate),
            "label": label,
        }

    if not feature_sets:
        st.caption("Select at least one feature set.")
    elif not manifolds:
        st.caption("Select at least one representation.")
    elif not windows:
        st.caption("Select at least one decode window.")
    elif not pairs:
        st.caption("No compatible feature×representation pairs in the current selection.")

    _render_quadrant_gallery(
        dataset,
        spike_source,
        methods=list(methods),
        n_components=int(n_components),
        label=label,
    )


def _submit_pending_job() -> None:
    if not state.consume_action(state.KEY_MANIFOLD_COMPUTE_REQUESTED):
        return
    pending = st.session_state.pop(_PENDING_KEY, None)
    if not pending:
        return
    req = ManifoldAnalysisRequest(
        input_dir=pending["input_dir"],
        feature_sets=tuple(pending["feature_sets"]),
        manifolds=tuple(pending["manifolds"]),
        decode_windows=tuple(pending["decode_windows"]),
        n_components=int(pending["n_components"]),
        spike_source=pending["spike_source"],
        force_recompute=bool(pending["force_recompute"]),
    )
    label = pending.get("label") or "Latent representation"

    def _job_fn(*, progress_callback=None):
        meta = run_manifold_analysis(req, progress_callback=progress_callback)
        try:
            from visualization.publication_winner_plots import (
                generate_publication_winner_figures,
            )

            def _winner_cb(msg: str, step: int, n: int) -> None:
                if progress_callback:
                    progress_callback(f"Winner PNGs: {msg}", step, n)

            paths = generate_publication_winner_figures(
                Path(req.input_dir),
                progress_callback=_winner_cb,
                spike_source=req.spike_source,
            )
            meta["n_winner_pngs"] = len(paths)
            meta["winner_pngs"] = [str(p) for p in paths]
            if len(paths) < _N_WINNER_EXPECTED:
                meta["winner_png_error"] = (
                    f"Incomplete: wrote {len(paths)}/{_N_WINNER_EXPECTED} winner PNGs"
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Winner PNG generation failed: %s", exc)
            meta["winner_png_error"] = str(exc)
            meta["n_winner_pngs"] = 0
        return meta

    submit_job(
        kind="manifold_analysis",
        label=f"{label} analysis",
        fn=_job_fn,
        slot=_JOB_SLOT,
        pass_progress=True,
    )
    st.rerun()


def _render_job_status(dataset: Path, spike_source: str) -> None:
    """Show active/completed job + winner_png_error without duplicating run widgets."""
    del spike_source
    existing = get_slot_job(_JOB_SLOT)
    if existing is None:
        return
    job = render_job_autofresh(slot=_JOB_SLOT)
    if job is None:
        return
    if job.status == "failed":
        st.error(f"Latent representation analysis failed: {job.error or 'unknown error'}")
        return
    if job.status != "completed" or job.result is None:
        return
    meta = job.result
    state.set_active_analysis_run(meta["run_id"])
    st.success(f"LATENT REPRESENTATION ANALYSIS COMPLETE — run `{meta['run_id']}`")
    st.write({
        "n_jobs_run": meta.get("n_jobs_run"),
        "n_skipped": meta.get("n_skipped"),
        "n_geometry_pages": meta.get("n_geometry_pages"),
        "n_winner_pngs": meta.get("n_winner_pngs"),
        "winner_png_error": meta.get("winner_png_error"),
        "output": str(dataset / "manifolds" / meta["run_id"]),
    })
    if meta.get("winner_png_error"):
        st.error(f"Winner PNG generation failed: {meta['winner_png_error']}")
    n_win = meta.get("n_winner_pngs")
    if n_win is not None and int(n_win) < _N_WINNER_EXPECTED:
        st.warning(
            f"Only **{n_win}/{_N_WINNER_EXPECTED}** winner PNGs were written."
        )
    _cached_manifold_runs.clear()
