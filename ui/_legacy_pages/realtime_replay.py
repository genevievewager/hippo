"""Page: Realtime Replay — recorded/simulated closed-loop inspection."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui.components.controls import dataset_selector, metric_row, spike_source_selector
from ui.components.plots import (
    error_over_time,
    speed_true_vs_pred,
    true_vs_decoded_position,
)
from ui.services.realtime import (
    build_replay_config,
    default_comparison_dir,
    execute_replay,
    find_realtime_runs,
    load_replay_artifacts,
)
from ui import state

logger = logging.getLogger(__name__)


def render(outputs_root: Path) -> None:
    st.header("Realtime Replay")
    st.caption(
        "Saved closed-loop figures load automatically. Interactive CSV scrubbing "
        "and replay execution remain available below. Not coupled to live Open Ephys."
    )

    dataset = dataset_selector(outputs_root, key="rt_dataset")
    spike_source = spike_source_selector(dataset, key="rt_spike_source")
    if dataset is None:
        return

    # Saved analysis figures first
    from ui.artifacts.models import CATEGORY_PERFORMANCE, CATEGORY_REALTIME
    from ui.artifacts.rendering import load_artifacts, render_artifact_gallery
    from ui.artifacts.discovery import filter_artifacts

    figs = filter_artifacts(
        load_artifacts(dataset),
        categories=[CATEGORY_REALTIME, CATEGORY_PERFORMANCE],
    )
    if figs:
        st.markdown("### Saved realtime / latency figures")
        render_artifact_gallery(figs, columns=2, key="rt_figs")
    else:
        st.info("No realtime figures yet — run a replay, then generate visualizations.")
        from ui.components.viz_actions import render_generate_viz_panel

        if render_generate_viz_panel(
            dataset,
            key="rt_viz_empty",
            compact=True,
            default_simulation=False,
            default_comparison=False,
        ):
            st.rerun()

    existing = find_realtime_runs(dataset)
    tab_load, tab_run = st.tabs(["Inspect saved replay CSV", "Run replay"])

    with tab_load:
        if not existing:
            st.info("No `decoded_realtime.csv` found under this experiment yet.")
        else:
            labels = [str(p.relative_to(dataset)) for p in existing]
            idx = st.selectbox(
                "Replay run",
                options=list(range(len(existing))),
                format_func=lambda i: labels[i],
                key="rt_pick",
            )
            try:
                arts = load_replay_artifacts(existing[idx])
            except Exception as exc:
                logger.exception("Failed loading replay")
                st.error(str(exc))
                return
            _render_replay(arts)

    with tab_run:
        st.warning("Replay training/evaluation can take a few minutes. It only runs on button click.")
        from ui.components.controls import render_architecture_diagram, representation_family_selector
        from ui.services.representations import (
            UI_DYNAMIC_LATENT_OPTIONS,
            UI_STATIC_MANIFOLD_OPTIONS,
            format_representation_label,
            representation_capabilities,
        )

        family = representation_family_selector(key="rt_rep_family")
        render_architecture_diagram(family)
        opts = list(UI_DYNAMIC_LATENT_OPTIONS if family == "dynamic" else UI_STATIC_MANIFOLD_OPTIONS)
        rep = st.selectbox(
            "Representation",
            options=opts,
            index=0,
            format_func=format_representation_label,
            key="rt_representation",
        )
        caps = representation_capabilities(rep)
        can_run = True
        if family == "dynamic" and not caps["supports_realtime"]:
            st.error(
                f"{rep} is **OFFLINE / ACAUSAL** and cannot be launched as a realtime model. "
                "Choose Global LDS or switch to Static manifold."
            )
            can_run = False
        else:
            st.caption(f"Capability: `{caps['badge']}`")

        update_dt = st.number_input("Update dt (s)", value=0.025, min_value=0.001, step=0.025, key="rt_dt")
        decode_window = st.select_slider(
            "Decode window",
            options=[0.025, 0.050, 0.100, 0.250, 0.500, 1.000],
            value=0.250,
            format_func=lambda w: f"{w*1000:.0f} ms" if w < 1 else "1 s",
            key="rt_w",
        )
        n_comp = st.select_slider(
            "Latent dimensions",
            options=[2, 3, 5, 8, 10, 20],
            value=5 if family == "dynamic" else 3,
            key="rt_k",
        )
        target = st.selectbox(
            "Closed-loop target",
            options=["position", "spatial_context", "speed", "movement_state"],
            index=0,
            key="rt_target",
        )
        use_best = st.checkbox(
            "Use best deployable decoder from comparison",
            value=(family == "static"),
            key="rt_best",
            disabled=(family == "dynamic"),
            help="For dynamic latents, replay fits the selected representation directly.",
        )
        cmp = default_comparison_dir(dataset)
        if use_best and cmp is None and family == "static":
            st.warning(
                "No decoder_comparison/sorted metrics found for this dataset — "
                "run a benchmark first or disable best-decoder mode."
            )
        out_rel = st.text_input(
            "Output subdirectory",
            value=("realtime_decoding/dynamic" if family == "dynamic" else "realtime_decoding"),
            key="rt_out",
        )

        if st.button("Run Replay", type="primary", disabled=not can_run):
            state.request_action(state.KEY_REPLAY_RUN_REQUESTED)

        if state.consume_action(state.KEY_REPLAY_RUN_REQUESTED):
            if not can_run:
                st.error("Cannot run offline-only representation in realtime.")
                return
            cfg = build_replay_config(
                input_dir=dataset,
                output_dir=dataset / out_rel,
                spike_source=spike_source,
                update_dt=float(update_dt),
                decode_window=float(decode_window),
                use_best_decoder=bool(use_best and family == "static"),
                closed_loop_target=target,
                comparison_dir=cmp,
                feature_type=rep,
                manifold_n_components=int(n_comp),
            )
            with st.spinner("Running realtime replay…"):
                try:
                    result = execute_replay(cfg)
                    st.success("Replay complete.")
                    if hasattr(result, "metrics"):
                        st.write(result.metrics)
                    # Reload newest
                    runs = find_realtime_runs(dataset)
                    if runs:
                        arts = load_replay_artifacts(runs[-1])
                        _render_replay(arts)
                except Exception as exc:
                    logger.exception("Replay failed")
                    st.error(f"Replay failed: {exc}")


def _render_replay(arts) -> None:
    decoded = arts.decoded
    metrics = arts.metrics or {}

    # Metric cards
    cards = {}
    if "position_error_cm" in decoded.columns:
        cards["Mean pos err (cm)"] = f"{float(decoded['position_error_cm'].mean()):.2f}"
    for key in ("mean_position_error_cm", "r2", "balanced_accuracy", "mean_predict_ms", "update_dt_s"):
        if key in metrics:
            val = metrics[key]
            cards[key] = f"{val:.3f}" if isinstance(val, float) else val
    if "decode_window_s" in decoded.columns:
        cards["Decode window"] = f"{float(decoded['decode_window_s'].iloc[0])*1000:.0f} ms"
    if "update_dt_s" in decoded.columns:
        cards["Update dt"] = f"{float(decoded['update_dt_s'].iloc[0])*1000:.0f} ms"
    metric_row(cards)

    # Scrubber
    n = len(decoded)
    idx = st.slider("Replay index", min_value=0, max_value=max(n - 1, 0), value=0, key="rt_idx")
    row = decoded.iloc[idx]
    tcol = "time" if "time" in decoded.columns else "decode_time"
    st.markdown(
        f"**t = {row.get(tcol, '—')} s** · "
        f"spikes in window = `{row.get('n_spikes_in_window', '—')}` · "
        f"active units = `{row.get('n_active_units_in_window', '—')}`"
    )
    if "position_error_cm" in decoded.columns:
        st.markdown(f"Position error: **{row['position_error_cm']:.2f} cm**")
    if {"true_x", "true_y", "decoded_x", "decoded_y"}.issubset(decoded.columns):
        st.markdown(
            f"True (x,y)=`({row['true_x']:.1f}, {row['true_y']:.1f})` · "
            f"Decoded=`({row['decoded_x']:.1f}, {row['decoded_y']:.1f})`"
        )

    t1, t2, t3, t4 = st.tabs(["Trajectory", "Error", "Speed", "Config / triggers"])
    with t1:
        st.plotly_chart(true_vs_decoded_position(decoded.iloc[: idx + 1]), use_container_width=True)
    with t2:
        st.plotly_chart(error_over_time(decoded), use_container_width=True)
    with t3:
        st.plotly_chart(speed_true_vs_pred(decoded), use_container_width=True)
    with t4:
        if arts.selected_config:
            st.json(arts.selected_config)
        if arts.closed_loop is not None and not arts.closed_loop.empty:
            st.dataframe(arts.closed_loop.head(100), use_container_width=True, hide_index=True)
        if metrics:
            with st.expander("realtime_metrics.json"):
                st.json(metrics)

    # Optional mini-raster of nearby spikes is expensive; skip auto-load.
    st.caption("Neural rasters for live scrubbing can be heavy — use Neural Simulation for full rasters.")
