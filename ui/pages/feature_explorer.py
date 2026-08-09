"""Page: Feature Explorer — saved figures + multi feature-set × window analysis."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui.artifacts.discovery import filter_artifacts
from ui.artifacts.models import CATEGORY_FEATURES
from ui.artifacts.rendering import load_artifacts, render_artifact_gallery
from ui.components.controls import dataset_selector, spike_source_selector
from ui.components.plots import correlation_heatmap, feature_traces
from ui.components.viz_actions import render_generate_viz_panel
from ui.services.comparison import format_decode_window
from ui.services.feature_analysis import FeatureAnalysisRequest, run_feature_analysis
from ui.services.features import (
    compute_feature_diagnostics,
    feature_set_families,
    list_feature_sets,
)
from ui import state

logger = logging.getLogger(__name__)

_WINDOW_OPTIONS = [0.025, 0.050, 0.100, 0.250, 0.500, 1.000]


def render(outputs_root: Path) -> None:
    st.header("Feature Explorer")
    st.caption(
        "Browse saved feature figures or **Run Feature Analysis** over multiple "
        "feature sets × decode windows (same pattern as Manifold Explorer). "
        "Widget changes never start computation."
    )

    dataset = dataset_selector(outputs_root, key="feat_dataset")
    spike_source = spike_source_selector(dataset, key="feat_spike_source")
    if dataset is None:
        return

    tab_figs, tab_run, tab_live = st.tabs([
        "Saved figures", "Run Feature Analysis", "Quick diagnostics",
    ])

    with tab_figs:
        _render_saved_figures(dataset)

    with tab_run:
        _render_run_analysis(dataset, spike_source)

    with tab_live:
        _render_quick_diagnostics(dataset, spike_source)


def _render_saved_figures(dataset: Path) -> None:
    arts = load_artifacts(dataset)
    feat_arts = filter_artifacts(arts, categories=[CATEGORY_FEATURES])
    prefer = [
        a for a in feat_arts
        if "features" in a.path.parts or a.stem.startswith("fig_neural")
    ]
    show = prefer or feat_arts

    if not show:
        st.info("No feature visualizations for this run yet. Use **Run Feature Analysis**.")
        if render_generate_viz_panel(
            dataset,
            key="feat_viz",
            compact=True,
            default_comparison=False,
            default_realtime=False,
        ):
            st.rerun()
        return

    render_artifact_gallery(show, key="feat_gallery", columns=2)

def _render_run_analysis(dataset: Path, spike_source: str) -> None:
    st.markdown(f"**Active dataset:** `{dataset.name}`")
    sets = list(list_feature_sets())
    feature_sets = st.multiselect(
        "Feature sets",
        options=sets,
        default=["counts", "counts_dynamics"] if "counts_dynamics" in sets else ["counts"],
        key="feat_run_sets",
        help="Each selected set is extracted for every selected decode window.",
    )
    windows = st.multiselect(
        "Compare windows",
        options=_WINDOW_OPTIONS,
        default=[0.100, 0.250, 0.500],
        format_func=format_decode_window,
        key="feat_run_wins",
    )
    regen_figs = st.checkbox(
        "Also regenerate simulation / neural / feature gallery figures",
        value=True,
        key="feat_run_regen",
        help="Calls the same backend as Generate visualizations (simulation suite).",
    )

    with st.expander("Feature-set definitions", expanded=False):
        for name in sets:
            st.markdown(f"**`{name}`** → `{', '.join(feature_set_families(name))}`")

    n_jobs = len(feature_sets) * max(len(windows), 1)
    st.write({"planned jobs": n_jobs, "feature sets": feature_sets, "windows": windows})

    if st.button(
        "Run Feature Analysis",
        type="primary",
        disabled=not feature_sets or not windows,
        key="feat_run_btn",
    ):
        state.request_action(state.KEY_FEATURE_ANALYSIS_REQUESTED)

    if not state.consume_action(state.KEY_FEATURE_ANALYSIS_REQUESTED):
        st.info(
            "Click **Run Feature Analysis** to extract each feature set × window, "
            "write PNGs under `features/<run_id>/`, and refresh the gallery."
        )
        return

    req = FeatureAnalysisRequest(
        input_dir=dataset,
        feature_sets=tuple(feature_sets),
        decode_windows=tuple(float(w) for w in windows),
        spike_source=spike_source,
        regenerate_simulation_figures=bool(regen_figs),
    )
    progress = st.progress(0, text="Starting…")

    def _cb(msg: str, step: int, n: int) -> None:
        progress.progress(min(step / max(n, 1), 1.0), text=f"[{step}/{n}] {msg}")

    try:
        with st.status("Running feature analysis…", expanded=True) as status:
            meta = run_feature_analysis(req, progress_callback=_cb)
            status.update(label="Feature analysis complete", state="complete")
        progress.progress(1.0, text="Done.")
        state.set_active_analysis_run(meta["run_id"])
        st.success(
            f"Saved {meta['n_jobs_run']} job(s) → "
            f"`{meta['output_dir']}` (run `{meta['run_id']}`)"
        )
        sim = meta.get("simulation_figures") or {}
        if isinstance(sim, dict) and sim.get("error"):
            st.warning(f"Gallery figures: {sim['error']}")
        elif isinstance(sim, dict) and sim.get("generated"):
            st.info(f"Gallery figures: {', '.join(sim['generated'])}")
        st.dataframe(
            [
                {
                    "feature_set": r["feature_set"],
                    "window_s": r["decode_window_s"],
                    "n_features": r["n_features"],
                    "n_samples": r["n_samples"],
                    "n_figures": len(r["figures"]),
                }
                for r in meta["results"]
            ],
            hide_index=True,
            use_container_width=True,
        )
        st.info("Open **Saved figures** to browse new gallery entries.")
        st.cache_data.clear()
    except Exception as exc:
        logger.exception("Feature analysis failed")
        st.error(f"Feature analysis failed: {exc}")


def _render_quick_diagnostics(dataset: Path, spike_source: str) -> None:
    st.caption(
        "Interactive one-off inspection without writing analysis folders. "
        "Prefer **Run Feature Analysis** when you want multi-set × window figures on disk."
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

            tabs = st.tabs(["Traces", "Correlation", "Meta"])
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
