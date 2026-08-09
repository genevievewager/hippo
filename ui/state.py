"""UI session-state: active dataset, analysis run, selected config.

Streamlit reruns frequently — expensive work must only start after an explicit
``request_action`` flag is consumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

# ── Explicit global state axes ──
KEY_ACTIVE_DATASET = "hippo_active_dataset"
KEY_ACTIVE_ANALYSIS_RUN = "hippo_active_analysis_run"
KEY_SELECTED_ANALYSIS_CONFIG = "hippo_selected_analysis_config"
KEY_SPIKE_SOURCE = "hippo_spike_source"
KEY_LAST_GENERATED_DATASET = "hippo_last_generated_dataset"

# Action flags (must be consumed, never edge-trigger science alone)
KEY_BENCHMARK_REQUESTED = "hippo_benchmark_requested"
KEY_BENCHMARK_STATUS = "hippo_benchmark_status"
KEY_BENCHMARK_ERROR = "hippo_benchmark_error"
KEY_BENCHMARK_RUN_ID = "hippo_benchmark_run_id"
KEY_LAST_COMPARISON_DIR = "hippo_last_comparison_dir"
KEY_FEATURE_COMPUTE_REQUESTED = "hippo_feature_compute_requested"
KEY_FEATURE_ANALYSIS_REQUESTED = "hippo_feature_analysis_requested"
KEY_MANIFOLD_COMPUTE_REQUESTED = "hippo_manifold_compute_requested"
KEY_SIM_RUN_REQUESTED = "hippo_sim_run_requested"
KEY_REPLAY_RUN_REQUESTED = "hippo_replay_run_requested"
KEY_VIZ_GENERATE_REQUESTED = "hippo_viz_generate_requested"
KEY_REPLAY_INDEX = "hippo_replay_index"
KEY_SELECTED_RESULT_RUN = "hippo_selected_result_run"
KEY_USE_CONFIG_FOR_REPLAY = "hippo_use_config_for_replay"

# Backward-compatible alias
KEY_DATASET = KEY_ACTIVE_DATASET


def init_session_state(outputs_root: Path | None = None) -> None:
    """Initialize defaults once per browser session."""
    defaults: dict[str, Any] = {
        KEY_ACTIVE_DATASET: None,
        KEY_ACTIVE_ANALYSIS_RUN: None,
        KEY_SELECTED_ANALYSIS_CONFIG: None,
        KEY_SPIKE_SOURCE: "sorted",
        KEY_LAST_GENERATED_DATASET: None,
        KEY_BENCHMARK_REQUESTED: False,
        KEY_BENCHMARK_STATUS: "idle",
        KEY_BENCHMARK_ERROR: None,
        KEY_BENCHMARK_RUN_ID: None,
        KEY_LAST_COMPARISON_DIR: None,
        KEY_FEATURE_COMPUTE_REQUESTED: False,
        KEY_FEATURE_ANALYSIS_REQUESTED: False,
        KEY_MANIFOLD_COMPUTE_REQUESTED: False,
        KEY_SIM_RUN_REQUESTED: False,
        KEY_REPLAY_RUN_REQUESTED: False,
        KEY_VIZ_GENERATE_REQUESTED: False,
        KEY_REPLAY_INDEX: 0,
        KEY_SELECTED_RESULT_RUN: None,
        KEY_USE_CONFIG_FOR_REPLAY: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    # Soft default only when nothing active yet
    if outputs_root is not None and st.session_state[KEY_ACTIVE_DATASET] is None:
        from ui.services.datasets import list_datasets

        datasets = list_datasets(outputs_root)
        if datasets:
            st.session_state[KEY_ACTIVE_DATASET] = str(datasets[0].path)


def get_active_dataset() -> Path | None:
    raw = st.session_state.get(KEY_ACTIVE_DATASET)
    if not raw:
        return None
    return Path(raw)


def set_active_dataset(path: Path | str | None) -> None:
    """Set the application-wide active dataset (all pages read this)."""
    st.session_state[KEY_ACTIVE_DATASET] = str(path) if path else None


# Aliases used by older pages
def get_dataset_path() -> Path | None:
    return get_active_dataset()


def set_dataset_path(path: Path | str | None) -> None:
    set_active_dataset(path)


def get_active_analysis_run() -> str | None:
    return st.session_state.get(KEY_ACTIVE_ANALYSIS_RUN)


def set_active_analysis_run(run_id: str | None) -> None:
    st.session_state[KEY_ACTIVE_ANALYSIS_RUN] = run_id


def get_selected_analysis_config() -> dict[str, Any] | None:
    return st.session_state.get(KEY_SELECTED_ANALYSIS_CONFIG)


def set_selected_analysis_config(cfg: dict[str, Any] | None) -> None:
    st.session_state[KEY_SELECTED_ANALYSIS_CONFIG] = cfg


def request_action(flag_key: str) -> None:
    st.session_state[flag_key] = True


def consume_action(flag_key: str) -> bool:
    if st.session_state.get(flag_key):
        st.session_state[flag_key] = False
        return True
    return False
