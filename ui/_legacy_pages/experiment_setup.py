"""Page: Experiment Setup — generate new datasets or load existing ones."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from hippo_sim.config import BEHAVIOR_DT, RATINABOX_PARAMS
from ui.artifacts.models import (
    CATEGORY_BEHAVIOR,
    CATEGORY_PROBE,
)
from ui.artifacts.discovery import filter_artifacts
from ui.artifacts.rendering import load_artifacts, render_artifact_gallery, render_overview_strip
from ui.components.controls import dataset_selector, dataset_summary_cards
from ui.services.datasets import (
    inspect_dataset,
    list_datasets,
)
from ui.services.simulation import (
    UISimulationConfig,
    available_trajectories,
    behavior_sampling_hz,
    default_recording_sliders,
    default_sorting_sliders,
    generate_ui_dataset,
    population_count_keys,
    sanitize_dataset_name,
    validate_ui_sim_config,
)
from ui import state

logger = logging.getLogger(__name__)


def render(outputs_root: Path) -> None:
    st.header("Experiment Setup")
    st.caption(
        "Generate a new dataset or load an existing one. "
        "The **Active Dataset** is shared by every page."
    )

    tab_gen, tab_load = st.tabs(["Generate New Dataset", "Load Existing Dataset"])

    with tab_gen:
        _render_generate(outputs_root)

    with tab_load:
        _render_load(outputs_root)


def _render_generate(outputs_root: Path) -> None:
    st.subheader("Configure simulation")
    st.caption("Uses the same `generate_dataset(SimConfig)` backend as `run_simulation.py`.")

    # ── Dataset ──
    with st.expander("Dataset", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Dataset name", value="ratinabox_ui_001", key="gen_name")
            seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="gen_seed")
        with c2:
            duration = st.number_input(
                "Duration (seconds)", min_value=1.0, value=10.0, step=1.0, key="gen_dur",
                help=(
                    "Smoke tests: ~10 s. Decoder categorical targets need ~≥30–40 s. "
                    "Full sessions: 600 s."
                ),
            )
            out_root = st.text_input("Output root", value=str(outputs_root), key="gen_root")
        safe = sanitize_dataset_name(name)
        st.info(f"Will write to: `{Path(out_root) / safe}`")

    # ── Environment / behavior ──
    with st.expander("Environment & behavior", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            arena = st.number_input("Arena size (cm)", min_value=10.0, value=100.0, step=5.0, key="gen_arena")
        with c2:
            thigmo = st.slider("Thigmotaxis", 0.0, 1.0, 0.8, 0.05, key="gen_thigmo")
        with c3:
            st.metric("Behavior timestep", f"{BEHAVIOR_DT * 1000:.0f} ms")
            st.metric("Behavior sampling", f"{behavior_sampling_hz():.0f} Hz")
        st.caption(
            f"`behavior_dt` is locked at **{BEHAVIOR_DT} s** (20 Hz) — required by the "
            "decoding pipeline. It is not editable."
        )

    # ── Neural population / trajectory ──
    with st.expander("Neural population / probe trajectory", expanded=True):
        trajs = available_trajectories()
        traj_names = [t["name"] for t in trajs]
        default_ix = 0
        for i, t in enumerate(trajs):
            if t.get("is_default"):
                default_ix = i
                break
        trajectory = st.selectbox(
            "Trajectory config",
            options=["(schematic — no trajectory YAML)"] + traj_names,
            index=default_ix + 1 if traj_names else 0,
            key="gen_traj",
        )
        include_non_hpc = st.checkbox(
            "Include non-hippocampal regions", value=False, key="gen_non_hpc",
        )
        st.caption("Population sizes follow the trajectory cell-capture YAML by default.")
        with st.expander("Advanced — override RatInABox population counts"):
            pop_overrides = {}
            cols = st.columns(3)
            for i, key in enumerate(population_count_keys()):
                default = int(RATINABOX_PARAMS.get(key, 0))
                with cols[i % 3]:
                    val = st.number_input(key, min_value=0, value=default, step=1, key=f"pop_{key}")
                    if val != default:
                        pop_overrides[key] = int(val)

    # ── Recording / sorting ──
    with st.expander("Recording / sorting degradation", expanded=False):
        st.caption(
            "These parameters control **dataset-level** Neuropixels-like recording and "
            "Kilosort-like sorting. This is distinct from GT-vs-sorted robustness "
            "comparison in the decoder benchmark."
        )
        sorting_overrides = {}
        recording_overrides = {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Sorting**")
            for key, (lo, hi, default) in default_sorting_sliders().items():
                val = st.slider(key, lo, hi, float(default), key=f"sort_{key}")
                if abs(val - default) > 1e-12:
                    sorting_overrides[key] = float(val)
        with c2:
            st.markdown("**Recording**")
            for key, (lo, hi, default) in default_recording_sliders().items():
                val = st.slider(key, lo, hi, float(default), key=f"rec_{key}")
                if abs(val - default) > 1e-12:
                    recording_overrides[key] = float(val)

    gen_figs = st.checkbox(
        "Generate diagnostic simulation figures after dataset creation",
        value=True,
        key="gen_figs",
    )

    cfg = UISimulationConfig(
        dataset_name=name,
        output_root=Path(out_root),
        seed=int(seed),
        duration_s=float(duration),
        arena_size_cm=float(arena),
        thigmotaxis=float(thigmo),
        trajectory=None if trajectory.startswith("(") else trajectory,
        no_trajectory=trajectory.startswith("("),
        include_non_hippocampal_regions=include_non_hpc,
        population_overrides=pop_overrides,
        sorting_overrides=sorting_overrides,
        recording_overrides=recording_overrides,
        generate_diagnostic_figures=gen_figs,
    )

    errs = validate_ui_sim_config(cfg)
    if errs:
        for e in errs:
            st.warning(e)

    if st.button("Generate Dataset", type="primary", disabled=bool(errs)):
        state.request_action(state.KEY_SIM_RUN_REQUESTED)

    if state.consume_action(state.KEY_SIM_RUN_REQUESTED):
        progress = st.progress(0, text="Starting simulation…")
        status_box = st.empty()

        def _cb(msg: str, step: int, n: int) -> None:
            frac = min(step / max(n, 1), 1.0)
            progress.progress(frac, text=f"[{step}/{n}] {msg}")
            status_box.info(msg)

        try:
            with st.status("Generating dataset…", expanded=True) as status:
                summary = generate_ui_dataset(cfg, progress_callback=_cb)
                status.update(label="Dataset generated", state="complete")
            out = Path(summary["output_dir"])
            state.set_active_dataset(out)
            st.session_state[state.KEY_LAST_GENERATED_DATASET] = str(out)
            # Bust artifact cache by touching path identity
            progress.progress(1.0, text="Complete.")
            st.success(f"Dataset generated successfully. Active dataset: **{out.name}**")
            st.json({
                "n_units": summary.get("n_units"),
                "session_duration_s": summary.get("session_duration_s"),
                "n_ground_truth_spikes": summary.get("n_ground_truth_spikes"),
                "n_sorted_spikes": summary.get("n_sorted_spikes"),
                "seed": summary.get("seed"),
                "output_dir": str(out),
            })
            if summary.get("figure_generation_warning"):
                st.warning(f"Figures: {summary['figure_generation_warning']}")

            arts = load_artifacts(out)
            setup_arts = filter_artifacts(
                arts, categories=[CATEGORY_BEHAVIOR, CATEGORY_PROBE],
            )
            render_overview_strip(setup_arts, limit=4, key="gen_done_overview")
            st.info(
                "Dataset ready for analysis. Continue to **Neural Simulation**, "
                "**Feature Explorer**, **Manifold Explorer**, or **Decoder Benchmark**."
            )
        except Exception as exc:
            logger.exception("Dataset generation failed")
            st.error(f"Generation failed: {exc}")


def _render_load(outputs_root: Path) -> None:
    st.subheader("Load existing dataset")
    datasets = list_datasets(outputs_root)
    if not datasets:
        st.info("No datasets found. Switch to **Generate New Dataset**.")
        return

    dataset = dataset_selector(outputs_root, key="setup_load_dataset", label="Select dataset")
    if dataset is None:
        return

    if st.button("Set Active Dataset", type="primary"):
        state.set_active_dataset(dataset)
        st.success(f"Active dataset → `{dataset.name}`")

    info = inspect_dataset(dataset)
    dataset_summary_cards(info)

    setup_arts = filter_artifacts(
        load_artifacts(dataset),
        categories=[CATEGORY_BEHAVIOR, CATEGORY_PROBE],
    )
    if setup_arts:
        render_artifact_gallery(
            setup_arts,
            columns=2,
            key="setup_load_figs",
            page_size=8,
        )
    else:
        st.caption("No behavior / probe figures for this dataset yet.")
