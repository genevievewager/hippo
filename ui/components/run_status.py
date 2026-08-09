"""Run status / progress panels."""

from __future__ import annotations

import streamlit as st

from ui.services.registry import RunMetadata


def render_status(status: str, *, error: str | None = None) -> None:
    status = (status or "idle").lower()
    if status == "running":
        st.info("Benchmark running… this can take a long time. Do not re-click Run.")
    elif status == "completed":
        st.success("Benchmark completed.")
    elif status == "failed":
        st.error(f"Benchmark failed: {error or 'unknown error'}")
    elif status == "idle":
        st.caption("Status: idle — changing controls does **not** start a run.")
    else:
        st.caption(f"Status: {status}")


def render_run_metadata(meta: RunMetadata | None) -> None:
    if meta is None:
        st.caption("No run metadata selected.")
        return
    cols = st.columns(4)
    cols[0].markdown(f"**Run ID**  \n`{meta.run_id}`")
    cols[1].markdown(f"**Status**  \n`{meta.status}`")
    cols[2].markdown(f"**Spike source**  \n`{meta.spike_source}`")
    cols[3].markdown(f"**Git**  \n`{meta.git_commit or '—'}`")
    with st.expander("Configuration", expanded=False):
        st.json({
            "input_dataset": meta.input_dataset,
            "output_directory": meta.output_directory,
            "feature_sets": meta.feature_sets,
            "manifolds": meta.manifolds,
            "decode_windows": meta.decode_windows,
            "feature_ablation": meta.feature_ablation,
            "compare_sources": meta.compare_sources,
            "timestamp": meta.timestamp,
            "notes": meta.notes,
        })
