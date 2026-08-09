"""Reusable Streamlit controls for generating experiment visualizations."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui import state
from ui.services.visualizations import (
    VisualizationRequest,
    default_viz_request,
    detect_viz_inputs,
    generate_visualizations,
)

logger = logging.getLogger(__name__)


def render_generate_viz_panel(
    dataset: Path,
    *,
    key: str,
    compact: bool = False,
    default_simulation: bool | None = None,
    default_comparison: bool | None = None,
    default_realtime: bool | None = None,
) -> bool:
    """Show Generate visualizations controls.

    Returns True if figures were generated on this run (caller may reload gallery).
    """
    avail = detect_viz_inputs(dataset)
    defaults = default_viz_request(dataset)

    if compact:
        st.caption(
            "Populate the figure gallery from saved experiment outputs "
            "(does not retrain decoders)."
        )
    else:
        st.markdown("#### Generate visualizations")
        st.caption(
            "Writes under `figures/` using the same backend as "
            "`run_visualizations.py`. Does not retrain decoders or re-simulate."
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        include_sim = st.checkbox(
            "Simulation / neural / features",
            value=default_simulation if default_simulation is not None else defaults.include_simulation,
            disabled=not avail["simulation"],
            key=f"{key}_sim",
            help="Requires behavior.csv, units.csv, spikes, summary.json",
        )
    with c2:
        include_cmp = st.checkbox(
            "Decoder / manifold figures",
            value=default_comparison if default_comparison is not None else defaults.include_comparison,
            disabled=not avail["comparison"],
            key=f"{key}_cmp",
            help="Requires decoder_comparison metrics on disk",
        )
    with c3:
        include_rt = st.checkbox(
            "Realtime figures",
            value=default_realtime if default_realtime is not None else defaults.include_realtime,
            disabled=not avail["realtime"],
            key=f"{key}_rt",
            help="Requires realtime_decoding CSV outputs",
        )

    if not avail["simulation"] and not avail["comparison"] and not avail["realtime"]:
        st.warning(
            "No figure-generating inputs found yet. Generate a dataset, run a "
            "benchmark, or run a replay first."
        )
        return False

    compile_pdf = st.checkbox("Also compile figures/output.pdf", value=False, key=f"{key}_pdf")
    flag = f"{state.KEY_VIZ_GENERATE_REQUESTED}_{key}"
    if flag not in st.session_state:
        st.session_state[flag] = False

    disabled = not (include_sim or include_cmp or include_rt)
    if st.button(
        "Generate visualizations",
        type="primary",
        disabled=disabled,
        key=f"{key}_btn",
    ):
        st.session_state[flag] = True

    if not st.session_state.get(flag):
        return False
    st.session_state[flag] = False

    req = VisualizationRequest(
        experiment_dir=dataset,
        include_simulation=bool(include_sim),
        include_comparison=bool(include_cmp),
        include_realtime=bool(include_rt),
        compile_pdf=bool(compile_pdf),
    )
    progress = st.progress(0, text="Starting…")

    def _cb(msg: str, step: int, n: int) -> None:
        progress.progress(min(step / max(n, 1), 1.0), text=msg)

    try:
        with st.spinner(f"Generating figures → {dataset.name}/figures"):
            summary = generate_visualizations(req, progress_callback=_cb)
        progress.progress(1.0, text="Done.")
        if summary["generated"]:
            st.success(
                f"Wrote figures ({', '.join(summary['generated'])}) → "
                f"`{summary['figures_dir']}`"
            )
        else:
            st.warning(
                "No matching outputs were found for the selected figure types. "
                "Checkboxes may need decoder/realtime results on disk first."
            )
        if summary.get("pdf_path"):
            st.info(f"PDF: `{summary['pdf_path']}`")
        return True
    except Exception as exc:
        logger.exception("Visualization generation failed")
        st.error(f"Visualization generation failed: {exc}")
        return False
