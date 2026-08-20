"""Page: Feature Explorer — panneled feature-set overview (Manifold-style)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from ui.artifacts.discovery import filter_artifacts
from ui.artifacts.models import CATEGORY_FEATURES
from ui.artifacts.rendering import load_artifacts, render_artifact_gallery
from ui.components.controls import dataset_selector, spike_source_selector
from ui.components.plots import correlation_heatmap, feature_traces
from ui.components.run_status import (
    render_job_autofresh,
    render_workload_estimate,
)
from ui.components.viz_actions import render_generate_viz_panel
from ui.jobs import get_slot_job, submit_job
from ui.services.comparison import format_decode_window, format_duration
from ui.services.datasets import inspect_dataset
from ui.services.feature_analysis import (
    FeatureAnalysisRequest,
    list_feature_analysis_runs,
    run_feature_analysis,
    selection_already_covered,
)
from ui.services.features import (
    compute_feature_diagnostics,
    list_feature_sets,
)
from ui import state
from visualization.publication_feature_plots import (
    DEFAULT_FEATURE_PANEL_WINDOW_S,
    PANEL_KINDS,
    resolve_feature_panel_windows,
)

logger = logging.getLogger(__name__)

_WINDOW_OPTIONS = [0.025, 0.050, 0.100, 0.250, 0.500, 1.000]

_FEATURE_JOB_SEC = 12.0
_FEATURE_GALLERY_SEC = 40.0
_FEATURE_OVERHEAD_S = 15.0
_REFERENCE_SESSION_S = 600.0


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


_TAB_SAVED = "Saved figures"
_TAB_RUN = "Run Feature Analysis"
_TAB_LIVE = "Quick diagnostics"
_FEAT_TABS_KEY = "feat_main_tabs"


def render(outputs_root: Path) -> None:
    st.header("Feature Explorer")
    st.caption(
        "Panneled overview like Manifold Explorer: pick feature sets and decode "
        "windows for the run matrix. Overview panels use **250 ms** first; after a "
        "decoder run, each panel updates to that feature set's **own best window**. "
        "Widget changes never start computation."
    )

    dataset = dataset_selector(outputs_root, key="feat_dataset")
    spike_source = spike_source_selector(dataset, key="feat_spike_source")
    if dataset is None:
        return

    # Persist the active sub-tab across job submit / fragment reruns (no teleport).
    tab_figs, tab_run, tab_live = st.tabs(
        [_TAB_SAVED, _TAB_RUN, _TAB_LIVE],
        key=_FEAT_TABS_KEY,
    )

    with tab_figs:
        _render_saved_figures(dataset)

    with tab_run:
        _render_run_analysis(dataset, spike_source)

    with tab_live:
        _render_quick_diagnostics(dataset, spike_source)


def _render_saved_figures(dataset: Path) -> None:
    prior = list_feature_analysis_runs(dataset)
    if prior:
        st.caption(
            f"**{len(prior)}** prior feature analysis run(s) on disk under "
            f"`{dataset.name}/features/` — figures below are the published view."
        )

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

    meta_path = dataset / "figures" / "features" / "feature_panel_windows.json"
    if meta_path.exists():
        try:
            panel_meta = json.loads(meta_path.read_text())
            windows = panel_meta.get("windows") or {}
            if windows:
                st.subheader("Panel windows")
                st.dataframe(
                    [
                        {
                            "feature_set": fs,
                            "window_ms": int(round(float(info["window_s"]) * 1000)),
                            "source": info.get("source"),
                        }
                        for fs, info in windows.items()
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
                st.caption(
                    "Default first pass = 250 ms. After decoding, each feature set "
                    "uses its own best window from decoder metrics."
                )
        except Exception:
            pass

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
            st.image(str(art.path), use_container_width=True)
    else:
        st.info(
            "No panneled feature overview yet. Use **Run Feature Analysis** to "
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

    other = [
        a for a in feat_arts
        if a not in panel_by_kind.values()
        and (
            "features" in a.path.parts
            or a.stem.startswith("fig_neural")
            or a.stem.startswith("fig_feature_")
        )
    ]
    if other:
        with st.expander(f"Other feature figures ({len(other)})", expanded=False):
            render_artifact_gallery(other, key="feat_gallery_other", columns=2)


def _render_prior_runs(dataset: Path, spike_source: str) -> list[dict]:
    """Show prior feature analyses already on disk for this dataset."""
    runs = list_feature_analysis_runs(dataset)
    pub_panels = [
        dataset / "figures" / "features" / f"fig_feature_panel_{k}.png"
        for k in PANEL_KINDS
    ]
    n_pub = sum(1 for p in pub_panels if p.exists())

    if not runs and n_pub == 0:
        st.info(
            "No prior feature analysis runs found under "
            f"`{dataset.name}/features/`. Click **Run Feature Analysis** below "
            "to generate the first set of results."
        )
        return []

    st.subheader("Prior runs (already on disk)")
    if n_pub:
        st.success(
            f"Published panneled overview exists under "
            f"`figures/features/` ({n_pub}/{len(PANEL_KINDS)} pages). "
            "Browse them on **Saved figures** — no need to re-run just to view."
        )
    if runs:
        st.caption(
            f"Found **{len(runs)}** analysis folder(s) under `features/`. "
            "Selecting features/windows below does not overwrite them unless you "
            "click **Run Feature Analysis** again."
        )
        rows = []
        for r in runs[:12]:
            wins = r.get("decode_windows") or []
            rows.append({
                "run_id": r["run_id"],
                "created_at": (r.get("created_at") or "")[:19],
                "spike_source": r.get("spike_source") or "—",
                "feature_sets": ", ".join(r.get("feature_sets") or []) or "—",
                "windows_ms": ", ".join(
                    str(int(round(float(w) * 1000))) for w in wins
                ) or "—",
                "figures": r.get("n_figures", 0),
                "panels": "yes" if r.get("has_panels") else "no",
            })
        st.dataframe(rows, hide_index=True, use_container_width=True)
        if len(runs) > 12:
            st.caption(f"Showing 12 of {len(runs)} runs.")
    return runs


def _render_run_analysis(dataset: Path, spike_source: str) -> None:
    st.markdown(f"**Active dataset:** `{dataset.name}`")
    prior_runs = _render_prior_runs(dataset, spike_source)

    st.markdown("---")
    st.markdown(
        "Pick **feature sets** and **decode windows** for a **new** run matrix "
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

    if feature_sets and windows and prior_runs:
        coverage = selection_already_covered(
            prior_runs,
            feature_sets=feature_sets,
            decode_windows=windows,
            spike_source=spike_source,
        )
        if coverage["fully_covered"]:
            st.warning(
                f"This exact feature×window selection is already covered by prior run(s): "
                f"`{'`, `'.join(coverage['matching_run_ids'][:4])}`. "
                "Browse **Saved figures** instead of re-running unless you want to "
                "regenerate."
            )
        elif coverage["n_covered"]:
            st.info(
                f"**{coverage['n_covered']}/{coverage['n_wanted']}** selected "
                f"feature×window configs already exist in prior runs. "
                "Re-running will recompute the full selection (including ones you "
                "already have)."
            )

    resolved = (
        resolve_feature_panel_windows(
            dataset, feature_sets, spike_source=spike_source,
        )
        if feature_sets
        else {}
    )
    if resolved:
        st.caption(
            "Panel overview windows (independent of the matrix above): "
            + ", ".join(
                f"`{fs}`→{int(round(float(info['window_s']) * 1000))}ms ({info['source']})"
                for fs, info in resolved.items()
            )
        )

    with st.expander("Advanced", expanded=False):
        write_panels = st.checkbox(
            "Write panneled overview pages (variance / traces / correlation)",
            value=True,
            key="feat_run_panels_v2",
            help="Saved under figures/features/fig_feature_panel_*.png",
        )
        write_matrix = st.checkbox(
            "Write per-window diagnostic PNGs for the feature × window matrix",
            value=True,
            key="feat_run_matrix_v2",
            help="Per feature×window variance / traces / correlation files.",
        )
        regen_figs = st.checkbox(
            "Also regenerate simulation / neural / feature gallery figures",
            value=False,
            key="feat_run_regen_v2",
        )

    session_duration_s = None
    try:
        summary = inspect_dataset(dataset).summary or {}
        raw_dur = summary.get("session_duration_s", summary.get("session_duration"))
        if raw_dur is not None:
            session_duration_s = float(raw_dur)
    except (TypeError, ValueError):
        session_duration_s = None

    matrix_windows = list(windows) if write_matrix else []
    estimate_windows = matrix_windows or (
        [DEFAULT_FEATURE_PANEL_WINDOW_S] if write_panels else []
    )
    workload = _estimate_feature_workload(
        feature_sets=feature_sets,
        windows=estimate_windows if estimate_windows else [DEFAULT_FEATURE_PANEL_WINDOW_S],
        regenerate_figures=bool(regen_figs),
        session_duration_s=session_duration_s,
    )
    if write_matrix and feature_sets and windows:
        workload["planned_configurations"] = len(feature_sets) * len(windows)
        workload["detail_label"] = (
            f"{len(feature_sets)} feature set(s) × {len(windows)} window(s)"
            + (" · + panel pages" if write_panels else "")
        )
    elif write_panels and feature_sets:
        workload["planned_configurations"] = len(feature_sets)
        workload["detail_label"] = (
            f"{len(feature_sets)} feature panel(s) @ resolved windows"
        )
    render_workload_estimate(workload)

    slot = "feature:analysis"
    existing = get_slot_job(slot)
    busy = existing is not None and existing.is_active
    can_run = bool(feature_sets and windows and (write_panels or write_matrix))

    if st.button(
        "Run Feature Analysis",
        type="primary",
        disabled=not can_run or busy,
        key="feat_run_btn",
        help="Runs in the background — you can switch pages while it works.",
    ):
        # Keep the Run tab selected across the post-submit rerun.
        st.session_state[_FEAT_TABS_KEY] = _TAB_RUN
        state.request_action(state.KEY_FEATURE_ANALYSIS_REQUESTED)

    if not feature_sets:
        st.caption("Select at least one feature set.")
    elif not windows:
        st.caption("Select at least one decode window.")
    elif not (write_panels or write_matrix):
        st.caption("Enable panel pages and/or per-window diagnostics in Advanced.")

    if existing is not None:
        job = render_job_autofresh(
            slot=slot,
            estimated_runtime_s=workload.get("estimated_runtime_s"),
        )
        if job is not None and job.status == "completed" and job.result is not None:
            meta = job.result
            state.set_active_analysis_run(meta["run_id"])
            st.success(
                f"Feature analysis ready → `{dataset / 'figures' / 'features'}` "
                f"(run `{meta['run_id']}`)"
            )
            resolved_meta = meta.get("resolved_panel_windows") or {}
            if resolved_meta:
                st.dataframe(
                    [
                        {
                            "feature_set": fs,
                            "window_ms": int(round(float(info["window_s"]) * 1000)),
                            "source": info.get("source"),
                        }
                        for fs, info in resolved_meta.items()
                    ],
                    hide_index=True,
                    use_container_width=True,
                )
            sim = meta.get("simulation_figures") or {}
            if isinstance(sim, dict) and sim.get("error"):
                st.warning(f"Gallery figures: {sim['error']}")
            st.info("Open **Saved figures** to browse outputs.")
            st.cache_data.clear()

    if not state.consume_action(state.KEY_FEATURE_ANALYSIS_REQUESTED):
        return

    st.session_state[_FEAT_TABS_KEY] = _TAB_RUN

    req = FeatureAnalysisRequest(
        input_dir=dataset,
        feature_sets=tuple(feature_sets),
        decode_windows=tuple(float(w) for w in windows),
        spike_source=spike_source,
        regenerate_simulation_figures=bool(regen_figs),
        write_panel_pages=bool(write_panels),
        write_per_window_diagnostics=bool(write_matrix),
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
    st.session_state[_FEAT_TABS_KEY] = _TAB_RUN
    st.rerun()


def _render_quick_diagnostics(dataset: Path, spike_source: str) -> None:
    st.caption(
        "Interactive one-off inspection without writing analysis folders. "
        "Prefer **Run Feature Analysis** when you want panneled overview pages on disk."
    )
    sets = list(list_feature_sets())
    selected = st.multiselect(
        "Feature sets",
        options=sets,
        default=["counts"],
        key="feat_quick_sets",
    )
    windows = st.multiselect(
        "Windows",
        options=_WINDOW_OPTIONS,
        default=[0.250],
        format_func=format_decode_window,
        key="feat_quick_wins",
    )

    if st.button(
        "Compute quick diagnostics",
        type="primary",
        disabled=not selected or not windows,
        key="feat_quick_btn",
    ):
        state.request_action(state.KEY_FEATURE_COMPUTE_REQUESTED)

    if not state.consume_action(state.KEY_FEATURE_COMPUTE_REQUESTED):
        return

    for fs in selected:
        for window in windows:
            st.subheader(f"{fs} · {format_decode_window(float(window))}")
            try:
                diag = compute_feature_diagnostics(
                    dataset,
                    fs,
                    spike_source=spike_source,
                    decode_window=float(window),
                )
            except Exception as exc:
                logger.exception("Feature diagnostics failed for %s @ %s", fs, window)
                st.error(f"`{fs}` @ {window}s failed: {exc}")
                continue

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Features", diag.n_features)
            c2.metric("Samples", diag.n_samples)
            c3.metric("Families", len(diag.families))
            c4.metric("Matrix", f"{diag.n_samples} × {diag.n_features}")

            tabs = st.tabs(
                ["Traces", "Correlation", "Meta"],
                key=f"feat_quick_diag_tabs_{fs}_{window}",
            )
            with tabs[0]:
                if diag.example_traces.shape[1] > 1:
                    st.plotly_chart(feature_traces(diag.example_traces), use_container_width=True)
            with tabs[1]:
                if diag.correlation_subset is not None:
                    st.plotly_chart(
                        correlation_heatmap(diag.correlation_subset),
                        use_container_width=True,
                    )
            with tabs[2]:
                st.json(diag.region_blocks)
