"""Page: Feature Construction — run controls, then panneled feature-set overview."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui.artifacts.discovery import filter_artifacts
from ui.artifacts.models import CATEGORY_FEATURES
from ui.artifacts.rendering import load_artifacts
from ui.components.controls import active_spike_source, require_active_dataset
from ui.components.run_status import (
    render_job_autofresh,
    render_workload_estimate,
)
from ui.components.viz_actions import render_generate_viz_panel
from ui.jobs import get_slot_job, submit_job
from ui.services.comparison import (
    apply_remaining_work_to_workload,
    format_decode_window,
    format_duration,
)
from ui.services.datasets import inspect_dataset
from ui.services.feature_analysis import (
    FeatureAnalysisRequest,
    run_feature_analysis,
    selection_already_covered,
)
from ui.services.features import list_feature_sets
from ui import state
from visualization.publication_feature_plots import (
    DEFAULT_FEATURE_PANEL_WINDOW_S,
    PANEL_KINDS,
)

logger = logging.getLogger(__name__)

_WINDOW_OPTIONS = [0.025, 0.050, 0.100, 0.250, 0.500, 1.000]

_FEATURE_JOB_SEC = 12.0
_FEATURE_GALLERY_SEC = 40.0
_FEATURE_OVERHEAD_S = 15.0
_REFERENCE_SESSION_S = 600.0

_WRITE_PANELS = True
_WRITE_MATRIX = True
_REGEN_FIGS = False


def _estimate_feature_workload(
    *,
    feature_sets: list[str],
    windows: list[float],
    regenerate_figures: bool,
    session_duration_s: float | None = None,
) -> dict:
    n_sets = len(feature_sets)
    n_windows = max(len(windows), 1) if windows else 0
    planned = n_sets * n_windows
    duration_scale = 1.0
    if session_duration_s is not None and session_duration_s > 0:
        duration_scale = max(float(session_duration_s) / _REFERENCE_SESSION_S, 0.25)
    estimated_s = (
        _FEATURE_OVERHEAD_S
        + float(planned) * _FEATURE_JOB_SEC * duration_scale
        + (_FEATURE_GALLERY_SEC if regenerate_figures and planned else 0.0)
    )
    low_s = max(estimated_s * 0.7, _FEATURE_OVERHEAD_S)
    high_s = estimated_s * 1.5
    return {
        "planned_configurations": int(planned),
        "n_feature_sets": n_sets,
        "n_windows": n_windows,
        "detail_label": f"{n_sets} feature set(s) · {n_windows} window(s)",
        "estimated_runtime_s": float(estimated_s),
        "estimated_runtime_low_s": float(low_s),
        "estimated_runtime_high_s": float(high_s),
        "estimated_runtime_label": format_duration(estimated_s),
        "estimated_runtime_range_label": (
            f"{format_duration(low_s)} – {format_duration(high_s)}"
        ),
    }


def render(outputs_root: Path) -> None:
    st.header("Feature Construction")

    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return
    spike_source = active_spike_source(dataset, readonly=True)

    _render_run_analysis(dataset, spike_source)
    st.divider()
    _render_feature_panels(dataset)


def _render_feature_panels(dataset: Path) -> None:
    arts = load_artifacts(dataset)
    feat_arts = filter_artifacts(arts, categories=[CATEGORY_FEATURES])

    panel_arts = [a for a in feat_arts if a.stem.startswith("fig_feature_panel_")]
    panel_by_kind: dict[str, object] = {}
    for kind in PANEL_KINDS:
        matches = [a for a in panel_arts if a.stem == f"fig_feature_panel_{kind}"]
        if not matches:
            continue
        preferred = [
            a for a in matches
            if "/figures/features/" in str(a.path).replace("\\", "/")
        ]
        panel_by_kind[kind] = (preferred or matches)[0]

    if panel_by_kind:
        st.subheader("Feature panels (feature sets × diagnostic)")
        st.caption(
            "Three overview pages — variance, traces, correlation — with one panel "
            "per feature set at that set's selected window."
        )
        titles = {
            "variance": "Variance",
            "traces": "Traces",
            "correlation": "Correlation",
        }
        for kind in PANEL_KINDS:
            art = panel_by_kind.get(kind)
            if art is None:
                continue
            st.markdown(f"**{titles.get(kind, kind)}**")
            st.image(str(art.path), width="stretch")
        return

    st.info(
        "No panneled feature overview yet. Run Feature Analysis above to "
        "generate variance / traces / correlation pages (250 ms until decoders run)."
    )
    if render_generate_viz_panel(
        dataset,
        key="feat_viz",
        compact=True,
        default_comparison=False,
        default_realtime=False,
    ):
        st.rerun()


def _render_run_analysis(dataset: Path, spike_source: str) -> None:
    st.subheader("Run Feature Analysis")
    st.markdown(
        f"**Active dataset:** `{dataset.name}`. "
        "Pick **feature sets** and **decode windows** for a new run matrix "
        "(one diagnostic package per feature×window). Also writes three panneled "
        "overview pages under `figures/features/` — panels use **250 ms** until "
        "decoder metrics exist, then each feature set's own best window."
    )

    sets = list(list_feature_sets())
    default_sets = (
        ["counts", "counts_dynamics"] if "counts_dynamics" in sets else ["counts"]
    )
    col_fs, col_w = st.columns(2)
    with col_fs:
        feature_sets = st.multiselect(
            "Feature sets",
            options=sets,
            default=default_sets,
            key="feat_run_sets_v2",
            help="Neural feature families to extract. One panel per set on overview pages.",
        )
    with col_w:
        windows = st.multiselect(
            "Decode windows",
            options=_WINDOW_OPTIONS,
            default=[DEFAULT_FEATURE_PANEL_WINDOW_S],
            format_func=format_decode_window,
            key="feat_run_wins_v2",
            help="Windows to run for each selected feature set (feature × window matrix).",
        )

    coverage: dict | None = None
    if feature_sets and windows:
        coverage = selection_already_covered(
            dataset,
            feature_sets=feature_sets,
            decode_windows=windows,
            spike_source=spike_source,
        )
        if coverage["fully_covered"]:
            st.info(
                "Selected feature×window transforms are already in the shared "
                "cache. Downstream pages can use them."
            )
        elif coverage["n_covered"]:
            st.info(
                f"**{coverage['n_covered']}/{coverage['n_wanted']}** selected "
                f"feature×window transforms are on disk. "
                "Missing caches will be written on run."
            )

    session_duration_s = None
    try:
        summary = inspect_dataset(dataset).summary or {}
        raw_dur = summary.get("session_duration_s", summary.get("session_duration"))
        if raw_dur is not None:
            session_duration_s = float(raw_dur)
    except (TypeError, ValueError):
        session_duration_s = None

    workload = _estimate_feature_workload(
        feature_sets=feature_sets,
        windows=list(windows) if windows else [DEFAULT_FEATURE_PANEL_WINDOW_S],
        regenerate_figures=_REGEN_FIGS,
        session_duration_s=session_duration_s,
    )
    if feature_sets and windows:
        workload["planned_configurations"] = len(feature_sets) * len(windows)
        workload["detail_label"] = (
            f"{len(feature_sets)} feature set(s) × {len(windows)} window(s) · + panel pages"
        )
    if coverage is not None and coverage.get("n_wanted"):
        workload = apply_remaining_work_to_workload(
            workload,
            n_wanted=int(coverage["n_wanted"]),
            n_covered=int(coverage.get("n_covered") or 0),
            n_to_compute=coverage.get("n_to_compute"),
            fully_covered=bool(coverage.get("fully_covered")),
        )
    render_workload_estimate(workload)

    slot = "feature:analysis"
    existing = get_slot_job(slot)
    busy = existing is not None and existing.is_active
    can_run = bool(feature_sets and windows)

    if st.button(
        "Run Feature Analysis",
        type="primary",
        disabled=not can_run or busy,
        key="feat_run_btn",
        help="Runs in the background — you can switch pages while it works.",
    ):
        state.request_action(state.KEY_FEATURE_ANALYSIS_REQUESTED)

    if not feature_sets:
        st.caption("Select at least one feature set.")
    elif not windows:
        st.caption("Select at least one decode window.")

    if existing is not None:
        job = render_job_autofresh(
            slot=slot,
            estimated_runtime_s=workload.get("estimated_runtime_s"),
        )
        if job is not None and job.status == "completed" and job.result is not None:
            meta = job.result
            if meta.get("skipped"):
                st.info(
                    "Selected feature×window transforms are already on disk. "
                    "Nothing new was written."
                )
            else:
                if meta.get("run_id"):
                    state.set_active_analysis_run(meta["run_id"])
                st.success(
                    f"Feature analysis ready → `{dataset / 'figures' / 'features'}` "
                    f"(run `{meta.get('run_id')}`)"
                )
            sim = meta.get("simulation_figures") or {}
            if isinstance(sim, dict) and sim.get("error"):
                st.warning(f"Gallery figures: {sim['error']}")
            st.cache_data.clear()

    if not state.consume_action(state.KEY_FEATURE_ANALYSIS_REQUESTED):
        return

    req = FeatureAnalysisRequest(
        input_dir=dataset,
        feature_sets=tuple(feature_sets),
        decode_windows=tuple(float(w) for w in windows),
        spike_source=spike_source,
        regenerate_simulation_figures=_REGEN_FIGS,
        write_panel_pages=_WRITE_PANELS,
        write_per_window_diagnostics=_WRITE_MATRIX,
    )

    def _job_fn(*, progress_callback=None):
        return run_feature_analysis(req, progress_callback=progress_callback)

    submit_job(
        kind="feature_analysis",
        label="Feature analysis",
        fn=_job_fn,
        slot=slot,
        pass_progress=True,
    )
    st.rerun()
