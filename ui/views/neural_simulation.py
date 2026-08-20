"""Page: Neural Simulation — visual-first inspection of sim outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui.artifacts.models import (
    CATEGORY_NEURAL,
    CATEGORY_SPIKE_QUALITY,
)
from ui.artifacts.discovery import filter_artifacts
from ui.artifacts.rendering import load_artifacts, render_tabbed_gallery
from ui.components.controls import metric_row, require_active_dataset, spike_source_selector
from ui.services.datasets import inspect_dataset
from ui.services.simulation import available_trajectories
from ui import state

logger = logging.getLogger(__name__)


def render(outputs_root: Path) -> None:
    st.header("Neural Simulation")
    st.caption(
        "Neural / spike-quality figures for the active dataset. "
        "Set the dataset on **Experiment Setup**; choose spike source here. "
        "Behavior and probe anatomy live on **Experiment Setup**; "
        "feature matrices on **Feature Construction**."
    )

    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return
    spike_source = spike_source_selector(dataset, key="sim_spike_source")

    if "sim_section" not in st.session_state:
        st.session_state["sim_section"] = "viz"
    section = st.radio(
        "Section",
        options=("viz", "run"),
        format_func=lambda k: (
            "Analysis figures" if k == "viz" else "Run new simulation"
        ),
        horizontal=True,
        key="sim_section",
    )

    if section == "viz":
        _render_figures(dataset)
    else:
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
            "Neural": [CATEGORY_NEURAL],
            "Spike Quality": [CATEGORY_SPIKE_QUALITY],
        },
        key="sim_figs",
        columns=1,
        show_pdf_captions=True,
    )


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
        st.session_state["sim_section"] = "run"
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
