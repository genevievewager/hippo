#!/usr/bin/env python3
"""HIPPO BCI — Streamlit multipage entry point.

Launch from the repository root::

    streamlit run ui/app.py

The UI is a thin client over the existing scientific Python packages
(``hippo_sim``, ``realtime``, …). It does not reimplement decoding logic.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure repository root is importable when launched via ``streamlit run ui/app.py``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from ui import state
from ui.pages import (
    decoder_benchmark,
    experiment_setup,
    feature_explorer,
    manifold_explorer,
    neural_simulation,
    realtime_replay,
    static_vs_dynamic,
)
from ui.services.datasets import default_outputs_root

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hippo.ui")

st.set_page_config(
    page_title="HIPPO BCI",
    layout="wide",
    initial_sidebar_state="expanded",
)

OUTPUTS_ROOT = default_outputs_root(_REPO_ROOT)
state.init_session_state(OUTPUTS_ROOT)

st.sidebar.title("HIPPO BCI")
st.sidebar.caption("Hippocampal Neural Decoding Platform")
st.sidebar.markdown(
    "End-to-end experiment UI over the same scientific backend as the CLI."
)

from ui.components.controls import render_active_dataset_sidebar

render_active_dataset_sidebar(OUTPUTS_ROOT)
st.sidebar.markdown(f"`outputs/` → `{OUTPUTS_ROOT}`")


def _page(fn):
    """Wrap a render(outputs_root) page for st.Page."""
    def _run() -> None:
        fn.render(OUTPUTS_ROOT)
    _run.__name__ = fn.__name__.split(".")[-1]
    return _run


pages = {
    "Experiment": [
        st.Page(_page(experiment_setup), title="Experiment Setup", icon=":material/science:", default=True),
        st.Page(_page(neural_simulation), title="Neural Simulation", icon=":material/graphic_eq:"),
    ],
    "Representations": [
        st.Page(_page(feature_explorer), title="Feature Explorer", icon=":material/hub:"),
        st.Page(_page(manifold_explorer), title="Manifold Explorer", icon=":material/grain:"),
        st.Page(_page(static_vs_dynamic), title="Static vs Dynamic", icon=":material/compare:"),
    ],
    "Decoding": [
        st.Page(_page(decoder_benchmark), title="Decoder Benchmark", icon=":material/play_arrow:"),
        st.Page(_page(realtime_replay), title="Realtime Replay", icon=":material/timeline:"),
    ],
}

nav = st.navigation(pages, position="sidebar")
st.title("HIPPO BCI")
st.caption("Hippocampal Neural Decoding Platform")
nav.run()
