"""Page: Quadrant Comparison — primary scientific analysis (progressive disclosure).

Default view: shared observation, one decoder, one representative method per
quadrant, explicit Run. Advanced / exploration stays available below; full
factorial search remains on Decoder Benchmark.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from realtime.decoder_comparison import ALL_TARGETS
from realtime.decoder_models import (
    TARGET_FAMILY,
    categorical_model_names,
    continuous_model_names,
)
from realtime.pipeline_artifacts import AnalysisConfig, ObservationConfig
from realtime.quadrant_experiment import (
    QuadrantComparisonError,
    assert_result_matches_observation,
    default_decoder_for_target,
    load_legacy_metrics_with_quadrants,
    validate_controlled_quadrant_comparison,
)
from realtime.representation_registry import (
    CANONICAL_QUADRANTS,
    DEFAULT_QUADRANT_METHODS,
    QUADRANT_DISPLAY_LABELS,
    QUADRANT_NONLINEAR_DYNAMIC,
    get_spec,
    nonlinear_dynamic_available,
)
from realtime.search_space import resolve_manifold_alias
from ui.components.controls import active_spike_source, require_active_dataset
from ui.components.pipeline_status import render_inherited_observation, render_pipeline_status
from ui.components.plots import error_over_time, true_vs_decoded_position
from ui.components.run_status import render_job_autofresh, render_job_panel
from ui.jobs import get_slot_job, submit_job
from ui.services.comparison import format_decode_window
from ui.services.quadrant_comparison import (
    build_experiment_config,
    existing_summary,
    run_quadrant_comparison_job,
)
from ui.services.representations import (
    format_representation_label,
    selectable_methods_for_quadrant,
)
from ui import state

logger = logging.getLogger(__name__)

_JOB_SLOT = "quadrant:comparison"
_PENDING_KEY = "quadrant_pending_request"
_WINDOW_SWEEP_OPTIONS = (0.050, 0.100, 0.250, 0.500, 1.000)


def render(outputs_root: Path) -> None:
    st.header("Quadrant Comparison")
    st.caption(
        "Primary scientific analysis: compare linear/nonlinear × static/dynamic "
        "representations on **one** shared neural observation. "
        "Broader `F × E × D × W` search stays on **Decoder Benchmark**."
    )
    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return
    spike_source = active_spike_source(dataset, readonly=True, key="quad_spike_source")
    pipe = render_pipeline_status(dataset, current_stage="decoder")
    inherited_w, _ = render_inherited_observation(pipe, allow_benchmark_override=False)
    render_job_autofresh(slot=_JOB_SLOT)

    if pipe is None or pipe.observation is None or inherited_w is None:
        st.info(
            "Quadrant Comparison inherits the active neural observation from "
            "**Feature Construction**. Commit an observation there "
            "(feature set, window W, update interval), then return here."
        )
        _render_saved_results(dataset, observation=None)
        return

    observation = pipe.observation
    _render_experiment_summary(observation, spike_source)

    job = get_slot_job(_JOB_SLOT)
    busy = job is not None and job.is_active

    target = st.selectbox(
        "Behavioral target",
        options=list(ALL_TARGETS),
        index=list(ALL_TARGETS).index("position") if "position" in ALL_TARGETS else 0,
        key="quad_target",
        disabled=busy,
    )
    decoder = _decoder_picker(target, disabled=busy)
    methods, _ = _quadrant_method_pickers(disabled=busy)

    window_sweep: tuple[float, ...] = ()
    n_components = 3
    extra_methods: tuple[str, ...] = ()
    with st.expander("Advanced / exploration", expanded=False):
        st.caption(
            "Default mode keeps one representative method per quadrant and the "
            "inherited observation. These controls add method-level comparison "
            "and a labeled window sweep (separate fair experiments per W). "
            "Full factorial search stays on **Decoder Benchmark**."
        )
        extra_methods = _advanced_extra_methods(methods, disabled=busy)
        n_components = int(
            st.selectbox(
                "Latent components k",
                options=[2, 3, 5, 10],
                index=1,
                key="quad_k",
                disabled=busy,
            )
        )
        st.markdown("**Window Sweep / Observation Timescale Benchmark**")
        st.caption(
            f"Window inherited from neural observation: "
            f"**{format_decode_window(observation.window_s)}**. "
            "A sweep runs a separate four-quadrant comparison at each extra W "
            "rather than mixing windows in one table."
        )
        sweep_on = st.checkbox(
            "Run additional windows as separate experiments",
            value=False,
            key="quad_window_sweep",
            disabled=busy,
        )
        if sweep_on:
            extra_w = st.multiselect(
                "Additional windows",
                options=list(_WINDOW_SWEEP_OPTIONS),
                default=[],
                format_func=format_decode_window,
                key="quad_window_sweep_values",
                disabled=busy,
            )
            window_sweep = tuple(
                float(w) for w in extra_w
                if abs(float(w) - float(observation.window_s)) > 1e-9
            )
        st.markdown("**Full benchmark**")
        st.caption(
            "The existing `F × E × D × W × target` search is on "
            "**Decoder Benchmark** (Quick / Targeted / Full). "
            "Changing widgets here never launches that grid."
        )
        st.caption(
            "Realtime Replay still has a replay view of the three realtime-capable "
            "cells; this page is the controlled offline experiment."
        )

    first_emb = next((m for m in methods.values() if m), "global_pca")
    state.set_active_analysis_config(
        AnalysisConfig.from_observation(
            observation,
            representation=str(first_emb),
            decoder=str(decoder),
            target=str(target),
            n_components=int(n_components),
            causal=True,
        ),
        persist=False,
    )

    try:
        cfg = build_experiment_config(
            experiment_dir=dataset,
            observation=observation,
            target=str(target),
            decoder_name=str(decoder),
            methods=methods,
            extra_methods=tuple(extra_methods),
            n_components=int(n_components),
            window_sweep_s=window_sweep,
        )
        validate_controlled_quadrant_comparison(cfg)
        st.caption(
            f"Shared FeatureDataset hash `{cfg.observation.hash()[:12]}` · "
            f"{len(cfg.selected_embeddings())} representation branch(es) · "
            f"decoder `{decoder}` · target `{target}`"
        )
    except QuadrantComparisonError as exc:
        st.error(str(exc))
        cfg = None

    run_clicked = st.button(
        "Run Quadrant Comparison",
        type="primary",
        disabled=busy or cfg is None,
        key="quad_run_btn",
        help="Runs in the background. Changing widgets does not start a run.",
    )
    if run_clicked and cfg is not None:
        state.request_action(state.KEY_QUADRANT_EXPERIMENT_REQUESTED)
        st.session_state[_PENDING_KEY] = {
            "observation": observation.to_dict(),
            "target": str(target),
            "decoder_name": str(decoder),
            "methods": dict(methods),
            "extra_methods": list(extra_methods),
            "n_components": int(n_components),
            "window_sweep_s": list(window_sweep),
        }

    _submit_pending(dataset)
    if job is not None:
        render_job_panel(job)

    _render_saved_results(dataset, observation=observation)


def _render_experiment_summary(observation: ObservationConfig, spike_source: str) -> None:
    st.subheader("Experiment summary")
    cols = st.columns(4)
    cols[0].markdown(
        f"**Neural observation**  \n"
        f"Source: `{observation.source_spikes or spike_source}`  \n"
        f"Feature: `{observation.feature_set}`"
    )
    cols[1].markdown(
        f"**Window**  \n"
        f"{format_decode_window(observation.window_s)}  \n"
        f"inherited from Feature Construction"
    )
    cols[2].markdown(
        f"**Update interval**  \n"
        f"{observation.update_dt * 1000:.0f} ms"
    )
    cols[3].markdown(
        "**Change observation**  \n"
        "Use **Feature Construction** — "
        "this page does not choose a second W."
    )


def _decoder_picker(target: str, *, disabled: bool) -> str:
    family = TARGET_FAMILY.get(target, "continuous")
    if family == "categorical":
        options = list(categorical_model_names("full", target))
    else:
        options = list(continuous_model_names("full", target))
    default = default_decoder_for_target(target)
    if default not in options and options:
        default = options[0]
    idx = options.index(default) if default in options else 0
    return str(
        st.selectbox(
            "Decoder (same family on every quadrant)",
            options=options,
            index=idx,
            key="quad_decoder",
            disabled=disabled,
            help="Controlled comparison isolates representation E; keep D fixed.",
        )
    )


def _quadrant_method_pickers(*, disabled: bool) -> tuple[dict[str, str | None], tuple[str, ...]]:
    st.subheader("Quadrant configuration")
    methods: dict[str, str | None] = {}
    cols = st.columns(2)
    cells = (
        (CANONICAL_QUADRANTS[0], cols[0]),
        (CANONICAL_QUADRANTS[1], cols[1]),
        (CANONICAL_QUADRANTS[2], cols[0]),
        (CANONICAL_QUADRANTS[3], cols[1]),
    )
    for qid, col in cells:
        with col:
            methods[qid] = _one_quadrant_picker(qid, disabled=disabled)
    return methods, ()


def _one_quadrant_picker(qid: str, *, disabled: bool) -> str | None:
    label = QUADRANT_DISPLAY_LABELS[qid]
    if qid == QUADRANT_NONLINEAR_DYNAMIC and not nonlinear_dynamic_available():
        st.markdown(f"**{label}**")
        st.info("Not yet implemented. No method is substituted.")
        return None
    options = list(selectable_methods_for_quadrant(qid, advanced=False))
    default = DEFAULT_QUADRANT_METHODS.get(qid)
    if default == "identity" and "counts" in options:
        default = "counts"
    if default not in options and options:
        default = options[0]
    if not options:
        st.markdown(f"**{label}**")
        st.info("No implemented methods in this quadrant.")
        return None
    idx = options.index(default) if default in options else 0
    choice = st.selectbox(
        label,
        options=options,
        index=idx,
        format_func=format_representation_label,
        key=f"quad_method_{qid}",
        disabled=disabled,
    )
    spec = get_spec(str(choice))
    st.caption(spec.display_name)
    return str(choice)


def _advanced_extra_methods(
    methods: dict[str, str | None],
    *,
    disabled: bool,
) -> tuple[str, ...]:
    st.markdown("**Additional implementations (same observation)**")
    extra: list[str] = []
    selected = {resolve_manifold_alias(m) for m in methods.values() if m}
    for qid in CANONICAL_QUADRANTS:
        if qid == QUADRANT_NONLINEAR_DYNAMIC and not nonlinear_dynamic_available():
            continue
        options = [
            m for m in selectable_methods_for_quadrant(qid, advanced=True)
            if resolve_manifold_alias(m) not in selected
        ]
        if not options:
            continue
        picked = st.multiselect(
            f"Also compare in {QUADRANT_DISPLAY_LABELS[qid]}",
            options=options,
            default=[],
            format_func=format_representation_label,
            key=f"quad_extra_{qid}",
            disabled=disabled,
        )
        extra.extend(str(x) for x in picked)
    return tuple(extra)


def _submit_pending(dataset: Path) -> None:
    if not state.consume_action(state.KEY_QUADRANT_EXPERIMENT_REQUESTED):
        return
    pending = st.session_state.pop(_PENDING_KEY, None) or {}
    if not pending:
        return

    def _job_fn(*, progress_callback=None):
        return run_quadrant_comparison_job(
            experiment_dir=dataset,
            observation=pending["observation"],
            target=pending["target"],
            decoder_name=pending["decoder_name"],
            methods=pending["methods"],
            extra_methods=tuple(pending.get("extra_methods") or ()),
            n_components=int(pending.get("n_components") or 3),
            window_sweep_s=tuple(pending.get("window_sweep_s") or ()),
            progress_callback=progress_callback,
        )

    submit_job(
        kind="quadrant_comparison",
        label="Quadrant Comparison",
        fn=_job_fn,
        slot=_JOB_SLOT,
        pass_progress=True,
    )


def _render_saved_results(
    dataset: Path,
    *,
    observation: ObservationConfig | None,
) -> None:
    st.subheader("Results")
    summary = existing_summary(dataset)
    if not summary:
        st.caption("No quadrant comparison on this dataset yet.")
        _maybe_show_legacy_benchmark_table(dataset, observation)
        return

    if observation is not None:
        exp = summary.get("experiment") or {}
        obs_meta = exp.get("observation") or {}
        try:
            assert_result_matches_observation(
                {
                    "window_s": obs_meta.get("window_s"),
                    "update_dt_s": obs_meta.get("update_dt"),
                    "feature_set": obs_meta.get("feature_set"),
                    "spike_source": obs_meta.get("source_spikes"),
                    "observation_config_hash": exp.get("observation_hash"),
                },
                observation,
            )
        except QuadrantComparisonError as exc:
            st.warning(str(exc))
            st.caption("Saved results below are from a different observation.")

    method_rows = summary.get("method_results") or []
    if not method_rows and summary.get("runs"):
        st.caption("Window sweep — one fair comparison per W.")
        for run in summary["runs"]:
            w = ((run.get("experiment") or {}).get("observation") or {}).get("window_s")
            st.markdown(f"**W = {format_decode_window(float(w or 0))}**")
            _render_method_table(run.get("method_results") or [])
        return

    _render_quadrant_cards(method_rows)
    _render_method_table(method_rows)
    _render_prediction_traces(dataset, method_rows)


def _render_quadrant_cards(method_rows: list[dict[str, Any]]) -> None:
    by_q: dict[str, dict[str, Any]] = {}
    for row in method_rows:
        qid = str(row.get("quadrant") or "")
        if qid not in by_q:
            by_q[qid] = row
    cols = st.columns(2)
    for i, qid in enumerate(CANONICAL_QUADRANTS):
        with cols[i % 2]:
            st.markdown(f"**{QUADRANT_DISPLAY_LABELS[qid]}**")
            row = by_q.get(qid)
            if row is None:
                if qid == QUADRANT_NONLINEAR_DYNAMIC:
                    st.info("Not implemented")
                else:
                    st.caption("No result")
                continue
            value = row.get("primary_metric_value")
            metric = row.get("primary_metric") or ""
            name = row.get("display_name") or row.get("representation_name")
            if value is None:
                st.metric(label=str(name), value="—")
            else:
                st.metric(
                    label=str(name),
                    value=f"{float(value):.3f}",
                    help=str(metric),
                )
            st.caption(f"`{row.get('representation_name')}` · {metric}")


def _render_method_table(method_rows: list[dict[str, Any]]) -> None:
    if not method_rows:
        return
    df = pd.DataFrame(method_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
    if "primary_metric_value" in df.columns and "quadrant_label" in df.columns:
        plot_df = df.dropna(subset=["primary_metric_value"]).copy()
        if not plot_df.empty:
            metric = str(plot_df["primary_metric"].iloc[0])
            fig = px.bar(
                plot_df,
                x="quadrant_label",
                y="primary_metric_value",
                color="representation_name",
                barmode="group",
                title=f"Method-level {metric} (not quadrant averages)",
            )
            fig.update_layout(height=360, margin=dict(l=40, r=20, t=50, b=40))
            st.plotly_chart(fig, use_container_width=True)


def _render_prediction_traces(dataset: Path, method_rows: list[dict[str, Any]]) -> None:
    traces: list[tuple[str, pd.DataFrame]] = []
    output_dir = Path(dataset) / "quadrant_comparison"
    candidates = list(output_dir.glob("**/predictions"))
    for row in method_rows:
        cid = row.get("config_id")
        if not cid:
            continue
        for pred_root in candidates:
            path = pred_root / f"{cid}.parquet"
            if path.exists():
                try:
                    traces.append((
                        str(row.get("representation_name") or cid),
                        pd.read_parquet(path),
                    ))
                except Exception:
                    continue
                break
    if not traces:
        return
    st.markdown("**Prediction vs actual (held-out)**")
    tabs = st.tabs([name for name, _ in traces])
    for tab, (name, df) in zip(tabs, traces):
        with tab:
            st.caption(name)
            if {"true_x", "true_y", "pred_x", "pred_y"}.issubset(df.columns):
                plot_df = df.rename(columns={"pred_x": "decoded_x", "pred_y": "decoded_y"})
                if "error_cm" in plot_df.columns and "position_error_cm" not in plot_df.columns:
                    plot_df = plot_df.assign(position_error_cm=plot_df["error_cm"])
                st.plotly_chart(
                    true_vs_decoded_position(plot_df),
                    use_container_width=True,
                )
                st.plotly_chart(error_over_time(plot_df), use_container_width=True)
            elif {"true", "pred"}.issubset(df.columns):
                tcol = "time" if "time" in df.columns else None
                fig = go.Figure()
                if tcol is not None:
                    fig.add_trace(go.Scatter(x=df[tcol], y=df["true"], name="True"))
                    fig.add_trace(go.Scatter(x=df[tcol], y=df["pred"], name="Predicted"))
                fig.update_layout(height=320, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)
            residual_col = next(
                (c for c in ("residual", "error_cm", "error") if c in df.columns),
                None,
            )
            if residual_col:
                tcol = "time" if "time" in df.columns else None
                fig = (
                    px.line(df, x=tcol, y=residual_col, title=f"Residual ({residual_col})")
                    if tcol
                    else px.histogram(df, x=residual_col, title="Residual")
                )
                fig.update_layout(height=280, margin=dict(l=40, r=20, t=40, b=40))
                st.plotly_chart(fig, use_container_width=True)


def _maybe_show_legacy_benchmark_table(
    dataset: Path,
    observation: ObservationConfig | None,
) -> None:
    csv = Path(dataset) / "decoder_comparison" / "sorted" / "decoder_comparison_metrics.csv"
    if not csv.exists():
        csv = Path(dataset) / "decoder_comparison" / "decoder_comparison_metrics.csv"
    if not csv.exists():
        return
    st.caption(
        "Existing Decoder Benchmark metrics (quadrant inferred from representation "
        "name; the file is not rewritten)."
    )
    try:
        df = load_legacy_metrics_with_quadrants(csv)
    except Exception:
        return
    cols = [
        c for c in (
            "quadrant", "embedding_type", "decoder_name", "target_name",
            "decode_window_s", "feature_set",
        ) if c in df.columns
    ]
    st.dataframe(df[cols].head(50) if cols else df.head(20), use_container_width=True)
    if observation is not None and "decode_window_s" in df.columns:
        mismatched = df[
            (df["decode_window_s"] - float(observation.window_s)).abs() > 1e-6
        ]
        if not mismatched.empty:
            st.caption(
                "Some benchmark rows used a different window than the active "
                "observation; they are not a controlled quadrant comparison."
            )
