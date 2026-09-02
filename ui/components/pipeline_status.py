"""Connected-pipeline status strip (views into one active run)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from realtime.pipeline_artifacts import ArtifactStatus, PIPELINE_STAGES
from realtime.pipeline_graph import PipelineRun, load_or_infer_pipeline
from ui.services.comparison import format_decode_window

STAGE_LABELS: dict[str, str] = {
    "simulation": "Neural Simulation",
    "features": "Feature Construction",
    "representation": "Latent Representations",
    "decoder": "Decoder Benchmark",
    "replay": "Realtime Replay",
    "deployment": "Live Deployment",
}

# Pages that display the strip (named scientific workflow).
PAGE_STAGES: tuple[str, ...] = (
    "simulation",
    "features",
    "representation",
    "decoder",
    "replay",
)


def _glyph(status: ArtifactStatus | str) -> str:
    value = status.value if isinstance(status, ArtifactStatus) else str(status)
    return {
        ArtifactStatus.FRESH.value: "✓",
        ArtifactStatus.CACHED.value: "✓",
        ArtifactStatus.STALE.value: "⚠ stale",
        ArtifactStatus.INCOMPATIBLE.value: "✗ incompatible",
        ArtifactStatus.MISSING.value: "○",
    }.get(value, "○")


def load_active_pipeline(dataset: Path | None) -> PipelineRun | None:
    if dataset is None or not Path(dataset).exists():
        return None
    return load_or_infer_pipeline(Path(dataset))


def render_pipeline_status(
    dataset: Path | None,
    *,
    current_stage: str | None = None,
) -> PipelineRun | None:
    """Compact ✓ / stale strip for the connected scientific workflow."""
    run = load_active_pipeline(dataset)
    if run is None:
        st.caption("No active pipeline run — set a dataset on **Experiment Setup**.")
        return None

    bits: list[str] = []
    for name in PAGE_STAGES:
        rec = run.stage(name)
        label = STAGE_LABELS.get(name, name)
        mark = _glyph(rec.status)
        if rec.status == ArtifactStatus.FRESH and rec.origin.value == "computed":
            mark = "✓ NEW"
        if current_stage == name:
            bits.append(f"**{label}** {mark}")
        else:
            bits.append(f"{label} {mark}")
    st.caption(" · ".join(bits))

    obs = run.observation
    if obs is not None:
        st.caption(
            f"Active observation: source `{obs.source_spikes}` · "
            f"feature `{obs.feature_set}` · "
            f"history window **{format_decode_window(obs.window_s)}** · "
            f"update {obs.update_dt * 1000:.0f} ms"
        )
    return run


def render_inherited_observation(
    run: PipelineRun | None,
    *,
    allow_benchmark_override: bool = False,
) -> tuple[float | None, bool]:
    """Display the inherited pipeline window. Returns (window_s, benchmark_sweep).

    In pipeline mode the window is inherited. Benchmark override is an explicit
    checkbox, never a silent independent selector.
    """
    inherited = run.inherited_window_s() if run is not None else None
    if inherited is None:
        st.info(
            "No active observation window. Generate one on **Feature Construction**."
        )
        return None, False
    st.markdown(
        f"**History window (inherited):** {format_decode_window(inherited)}  \n"
        "Owned by Feature Construction / neural observation. "
        "This page does not choose a different W in pipeline mode."
    )
    sweep = False
    if allow_benchmark_override:
        sweep = st.checkbox(
            "Benchmark: compare additional cached windows",
            value=False,
            key="pipeline_bench_sweep_windows",
            help=(
                "Targeted / full benchmark may sweep W. Pipeline mode keeps "
                "the inherited observation window."
            ),
        )
    return float(inherited), bool(sweep)
