"""Page: Neural Simulation — visual-first inspection of sim outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from realtime.data_loading import load_simulation_data
from ui.artifacts.models import (
    CATEGORY_NEURAL,
    CATEGORY_SPIKE_QUALITY,
)
from ui.artifacts.discovery import filter_artifacts
from ui.artifacts.rendering import load_artifacts, render_tabbed_gallery
from ui.components.controls import dataset_selector, metric_row, spike_source_selector
from ui.components.plots import neural_raster, speed_over_time, trajectory_xy
from ui.services.datasets import (
    cell_type_counts,
    inspect_dataset,
    load_behavior_table,
    load_units_table,
    region_unit_counts,
    sorting_degradation_summary,
)
from ui.services.simulation import available_trajectories
from ui import state

logger = logging.getLogger(__name__)


def render(outputs_root: Path) -> None:
    st.header("Neural Simulation")
    st.caption(
        "Neural / spike-quality figures for the active dataset. "
        "Behavior and probe anatomy live on **Experiment Setup**; "
        "feature matrices on **Feature Explorer**."
    )

    dataset = dataset_selector(outputs_root, key="sim_dataset")
    spike_source = spike_source_selector(dataset, key="sim_spike_source")

    tab_viz, tab_live, tab_run = st.tabs([
        "Analysis figures", "Interactive inspection", "Run new simulation",
    ])

    with tab_viz:
        if dataset is None:
            st.info("Select a dataset to browse visualizations.")
        else:
            _render_figures(dataset)

    with tab_live:
        if dataset is None:
            st.info("Select a dataset.")
        else:
            _render_interactive(dataset, spike_source)

    with tab_run:
        _render_new_sim(outputs_root)


def _render_figures(dataset: Path) -> None:
    info = inspect_dataset(dataset)
    summary = info.summary
    metric_row({
        "Duration (s)": summary.get("session_duration_s", "—"),
        "Units": summary.get("n_units", "—"),
        "GT spikes": summary.get("n_ground_truth_spikes", "—"),
        "Sorted spikes": summary.get("n_sorted_spikes", "—"),
    })

    from ui.components.viz_actions import render_generate_viz_panel

    arts = load_artifacts(dataset)
    neural_arts = filter_artifacts(
        arts, categories=[CATEGORY_NEURAL, CATEGORY_SPIKE_QUALITY],
    )
    if not neural_arts:
        st.info("No neural / sorting visualizations for this run yet.")
        if render_generate_viz_panel(
            dataset,
            key="sim_viz_empty",
            compact=True,
            default_comparison=False,
            default_realtime=False,
        ):
            st.rerun()
        return

    render_tabbed_gallery(
        neural_arts,
        {
            "Spikes / neural": [CATEGORY_NEURAL],
            "Sorting / QC": [CATEGORY_SPIKE_QUALITY],
        },
        key="sim_figs",
    )

def _render_interactive(dataset: Path, spike_source: str) -> None:
    st.caption("On-demand interactive views from raw CSVs (not a re-simulation).")
    info = inspect_dataset(dataset)
    try:
        behavior = load_behavior_table(dataset)
        units = load_units_table(dataset)
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    t1, t2, t3, t4 = st.tabs(["Trajectory", "Speed", "Raster", "Composition"])
    with t1:
        if {"x", "y"}.issubset(behavior.columns):
            st.plotly_chart(trajectory_xy(behavior), use_container_width=True)
        else:
            st.warning("behavior.csv missing x/y columns")
    with t2:
        st.plotly_chart(speed_over_time(behavior), use_container_width=True)
    with t3:
        try:
            data = load_simulation_data(dataset, spike_source)
            st.plotly_chart(
                neural_raster(data["spikes_df"], units_df=data["units_df"]),
                use_container_width=True,
            )
        except Exception as exc:
            logger.exception("Raster failed")
            st.error(f"Could not load raster: {exc}")
    with t4:
        c1, c2 = st.columns(2)
        with c1:
            st.dataframe(region_unit_counts(units), hide_index=True, use_container_width=True)
        with c2:
            ct = cell_type_counts(units)
            if not ct.empty:
                st.dataframe(ct, hide_index=True, use_container_width=True)
        deg = sorting_degradation_summary(info.summary, dataset)
        st.write(deg)


def _render_new_sim(outputs_root: Path) -> None:
    st.info(
        "Prefer **Experiment Setup → Generate New Dataset** for the full "
        "configuration UI (trajectory, sorting, population overrides, figures)."
    )
    st.warning("Short smoke simulation only — refuses to overwrite existing folders.")
    name = st.text_input("Output folder name", value="ratinabox_ui_smoke", key="ns_name")
    duration = st.number_input("Duration (s)", min_value=1.0, value=10.0, step=1.0, key="ns_dur")
    seed = st.number_input("Seed", min_value=0, value=42, step=1, key="ns_seed")
    trajs = available_trajectories()
    traj_names = [t["name"] for t in trajs] if trajs else []
    trajectory = st.selectbox("Trajectory", options=["(default)"] + traj_names, key="ns_traj")

    if st.button("Run Simulation", type="primary", key="ns_run"):
        state.request_action(state.KEY_SIM_RUN_REQUESTED)

    if state.consume_action(state.KEY_SIM_RUN_REQUESTED):
        from ui.services.simulation import UISimulationConfig, generate_ui_dataset, sanitize_dataset_name

        cfg = UISimulationConfig(
            dataset_name=sanitize_dataset_name(name),
            output_root=outputs_root,
            seed=int(seed),
            duration_s=float(duration),
            trajectory=None if trajectory == "(default)" else trajectory,
            generate_diagnostic_figures=True,
        )
        with st.spinner(f"Running simulation → {outputs_root / cfg.dataset_name}"):
            try:
                summary = generate_ui_dataset(cfg)
                path = Path(summary["output_dir"])
                st.success(f"Simulation complete: {path}")
                state.set_active_dataset(path)
            except Exception as exc:
                logger.exception("Simulation failed")
                st.error(f"Simulation failed: {exc}")
