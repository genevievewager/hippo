"""Page: Results — visualization-first experiment browser + leaderboard."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui.artifacts.models import (
    CATEGORY_DECODER_COMPARISON,
    CATEGORY_DECODING,
    CATEGORY_DEPLOYMENT,
    CATEGORY_MANIFOLDS,
    CATEGORY_PERFORMANCE,
    CATEGORY_REALTIME,
)
from ui.artifacts.rendering import (
    load_artifacts,
    render_tabbed_gallery,
)
from ui.components.controls import require_active_dataset
from ui.components.plots import (
    confusion_from_metrics_row,
    degradation_or_source_curve,
    metric_by_category,
)
from ui.components.result_tables import render_filters, render_leaderboard
from ui.components.run_status import render_run_metadata
from ui.services.registry import discover_runs, load_run_metadata
from ui.services.results import (
    filter_metrics,
    find_metrics_csv,
    load_comparison_artifacts,
    list_comparison_roots,
    primary_score_column,
)
from ui import state

logger = logging.getLogger(__name__)


def render(outputs_root: Path) -> None:
    st.header("Results")
    st.caption(
        "Visualizations from disk first; metrics tables second. "
        "Selecting a run never recomputes the decoder grid."
    )

    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return

    runs = discover_runs(outputs_root=outputs_root)
    if runs:
        labels = [f"{r.run_id} · {r.status} · {Path(r.output_directory).name}" for r in runs]
        choice = st.selectbox(
            "Previous UI runs (registry)", options=["(none)"] + labels, key="res_run_pick",
        )
        if choice != "(none)":
            meta = runs[labels.index(choice)]
            render_run_metadata(meta)
            st.session_state[state.KEY_SELECTED_RESULT_RUN] = meta.run_id

    arts = load_artifacts(dataset)
    metrics_path = find_metrics_csv(dataset)
    artifacts_cmp = None
    if metrics_path is not None:
        try:
            artifacts_cmp = load_comparison_artifacts(metrics_path.parent)
        except Exception as exc:
            logger.exception("Failed loading comparison artifacts")
            st.warning(f"Metrics table could not be loaded: {exc}")

    # ── RUN SUMMARY ──
    st.markdown("## Run summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Figures", len([a for a in arts if a.is_image]))
    c2.metric("Metric rows", len(artifacts_cmp.metrics) if artifacts_cmp else 0)
    c3.metric("Has realtime figs", "yes" if any(a.category == CATEGORY_REALTIME for a in arts) else "no")
    c4.metric("Has manifold figs", "yes" if any(a.category == CATEGORY_MANIFOLDS for a in arts) else "no")

    # Best configuration highlight
    if artifacts_cmp is not None and artifacts_cmp.best_by_target is not None:
        st.markdown("### Best configuration")
        st.dataframe(
            artifacts_cmp.best_by_target.head(20),
            width="stretch",
            hide_index=True,
        )

    # Analysis figures only (decoding / manifolds / realtime / …)
    analysis_categories = {
        CATEGORY_DECODING,
        CATEGORY_DECODER_COMPARISON,
        CATEGORY_MANIFOLDS,
        CATEGORY_REALTIME,
        CATEGORY_DEPLOYMENT,
        CATEGORY_PERFORMANCE,
    }
    analysis_arts = [a for a in arts if a.category in analysis_categories]
    if not analysis_arts:
        st.info("No decoding / manifold / realtime figures found yet.")
        from ui.components.viz_actions import render_generate_viz_panel

        if render_generate_viz_panel(
            dataset,
            key="res_viz_empty",
            compact=True,
            default_simulation=False,
        ):
            st.rerun()
    else:
        st.markdown("## Decoding & analysis figures")
        render_tabbed_gallery(
            analysis_arts,
            {
                "Decoding": [CATEGORY_DECODING, CATEGORY_DECODER_COMPARISON],
                "Manifolds": [CATEGORY_MANIFOLDS],
                "Realtime": [CATEGORY_REALTIME],
                "Deployment": [CATEGORY_DEPLOYMENT],
                "Runtime": [CATEGORY_PERFORMANCE],
            },
            key="res_tabs",
        )

    # Interactive plots from metrics CSV (cheap)
    if artifacts_cmp is not None and not artifacts_cmp.metrics.empty:
        st.markdown("## Interactive metric views")
        st.caption("Built from saved `decoder_comparison_metrics.csv` — no recompute.")
        metrics = artifacts_cmp.metrics
        targets = sorted(metrics["target_name"].dropna().unique()) if "target_name" in metrics.columns else []
        target = st.selectbox("Focus target", options=targets or ["position"], key="res_focus_target")
        metric = primary_score_column(target) or "score"
        sub = metrics[metrics["target_name"] == target] if "target_name" in metrics.columns else metrics

        t1, t2, t3, t4 = st.tabs(["By manifold", "By feature set", "By window", "Source / class"])
        with t1:
            man_col = "embedding_type" if "embedding_type" in sub.columns else "manifold"
            if man_col in sub.columns and metric in sub.columns:
                st.plotly_chart(
                    metric_by_category(sub, category=man_col, metric=metric),
                    width="stretch",
                )
        with t2:
            if "feature_set" in sub.columns and metric in sub.columns:
                st.plotly_chart(
                    metric_by_category(sub, category="feature_set", metric=metric),
                    width="stretch",
                )
            else:
                st.info("feature_set column not present in this metrics table.")
        with t3:
            if "decode_window_s" in sub.columns and metric in sub.columns:
                st.plotly_chart(
                    metric_by_category(sub, category="decode_window_s", metric=metric),
                    width="stretch",
                )
        with t4:
            if "spike_source" in metrics.columns and metric in metrics.columns:
                st.plotly_chart(
                    degradation_or_source_curve(metrics, metric=metric),
                    width="stretch",
                )
            if target in ("spatial_context", "movement_state", "wall_distance_bin") and not sub.empty:
                fig = confusion_from_metrics_row(sub.iloc[0])
                if fig is not None:
                    st.plotly_chart(fig, width="stretch")

    # Full results table last
    st.markdown("## Full results table")
    if artifacts_cmp is None:
        roots = list_comparison_roots(dataset)
        if roots:
            st.caption("Comparison directories found but metrics failed to load above.")
        else:
            st.info("No `decoder_comparison_metrics.csv` for this experiment.")
        return

    st.success(
        f"Loaded `{metrics_path}` ({len(artifacts_cmp.metrics)} rows) — no recomputation."
    )
    filters = render_filters(artifacts_cmp.metrics)
    filtered = filter_metrics(
        artifacts_cmp.metrics,
        targets=filters.get("targets") or None,
        feature_sets=filters.get("feature_sets") or None,
        manifolds=filters.get("manifolds") or None,
        decoders=filters.get("decoders") or None,
        decode_windows=filters.get("decode_windows") or None,
        spike_sources=filters.get("spike_sources") or None,
    )
    render_leaderboard(filtered)

    if artifacts_cmp.feature_set_summary is not None:
        with st.expander("Feature-set performance summary"):
            st.dataframe(
                artifacts_cmp.feature_set_summary, width="stretch", hide_index=True
            )
    if artifacts_cmp.loss_summary is not None:
        with st.expander("Sorted information-loss summary"):
            st.dataframe(artifacts_cmp.loss_summary, width="stretch", hide_index=True)

    try:
        from ui.services.registry import metadata_path
        if metadata_path(artifacts_cmp.root).exists():
            render_run_metadata(load_run_metadata(artifacts_cmp.root))
    except Exception:
        pass

    # PDF pack download if present
    pdf = dataset / "figures" / "output.pdf"
    if pdf.exists():
        with st.expander("Compiled PDF report"):
            st.download_button(
                "Download figures/output.pdf",
                data=pdf.read_bytes(),
                file_name=f"{dataset.name}_figures.pdf",
                key="res_pdf_dl",
            )
