"""Run status / progress panels."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any

import streamlit as st

from realtime.work_progress import eta_seconds
from ui.jobs import JobState, get_job, get_slot_job
from ui.services.comparison import format_duration
from ui.services.registry import RunMetadata


def remaining_s(
    *,
    completed: int,
    total: int,
    ema_unit_s: float | None,
    n_measured: int,
    prior_unit_s: float | None = None,
) -> float | None:
    """Work-based ETA in seconds (EMA of unit duration × remaining units).

    Returns ``None`` until enough units have finished (or a history prior exists).
    Does not use elapsed / pre-run estimate as percent complete.
    """
    return eta_seconds(
        completed=completed,
        total=total,
        ema_unit_s=ema_unit_s,
        n_measured=n_measured,
        prior_unit_s=prior_unit_s,
    )


def render_status(status: str, *, error: str | None = None) -> None:
    status = (status or "idle").lower()
    if status == "running":
        st.info(
            "Job running in the background — you can switch sidebar pages; "
            "progress continues. Avoid starting a second run of the same job."
        )
    elif status == "completed":
        st.success("Benchmark completed.")
    elif status == "failed":
        st.error(f"Benchmark failed: {error or 'unknown error'}")
    elif status == "idle":
        st.caption("Status: idle — changing controls does **not** start a run.")
    else:
        st.caption(f"Status: {status}")


def render_workload_estimate(workload: dict[str, Any]) -> None:
    """Show combinatorial size + heuristic runtime from current selections."""
    if workload.get("skip_estimate"):
        return
    n_cfg = int(workload.get("planned_configurations", 0))
    eta = workload.get("estimated_runtime_label", "—")
    eta_range = workload.get("estimated_runtime_range_label", "—")
    detail = workload.get("detail_label")
    if not detail:
        detail = (
            f"{workload.get('n_valid_feature_manifold_pairs', '—')} "
            f"feature×representation pairs · "
            f"{workload.get('n_windows', '—')} windows"
        )
    st.info(
        f"About **{n_cfg:,}** configurations · {detail}  \n"
        f"Estimated runtime **~{eta}** (likely {eta_range}; "
        "heuristic prior for ETA, not percent complete)"
    )


def render_job_panel(
    job: JobState | None,
    *,
    estimated_runtime_s: float | None = None,
) -> JobState | None:
    """Render progress for a background job (safe across page navigation)."""
    del estimated_runtime_s  # pre-run seconds are a caption prior, never the bar
    if job is None:
        return None

    if job.status == "running" or job.status == "pending":
        st.info(
            f"**{job.label}** is running in the background. "
            "You can switch pages; return here to watch progress."
        )
    elif job.status == "completed":
        st.success(f"**{job.label}** completed in {format_duration(job.elapsed_s)}.")
    elif job.status == "failed":
        st.error(f"**{job.label}** failed: {job.error or 'unknown error'}")
        if job.traceback:
            with st.expander("Traceback"):
                st.code(job.traceback)

    frac = job.progress_fraction()
    pct = int(round(frac * 100))
    label = f"{pct}% [{job.step}/{max(job.total, 1)}] {job.message or job.status}"
    st.progress(frac, text=label)

    stage_bits = [p for p in (job.stage, job.task) if p]
    if stage_bits:
        st.caption(" · ".join(stage_bits))

    elapsed = job.elapsed_s
    eta = job.live_eta_s() if job.is_active else None
    if job.is_active:
        if eta is None:
            remain = "Estimating time remaining…"
        else:
            remain = f"~**{format_duration(eta)}**"
        st.caption(
            f"Elapsed **{format_duration(elapsed)}** · remaining {remain} · `{job.job_id}`"
        )
    else:
        st.caption(f"Elapsed **{format_duration(elapsed)}** · `{job.job_id}`")
    return job


def render_job_autofresh(
    slot: str | None = None,
    job_id: str | None = None,
    *,
    estimated_runtime_s: float | None = None,
    interval_s: float = 2.0,
) -> JobState | None:
    """Poll a background job and auto-refresh while it is active."""

    def _resolve() -> JobState | None:
        if job_id:
            return get_job(job_id)
        if slot:
            return get_slot_job(slot)
        return None

    job = _resolve()
    if job is None:
        return None

    try:
        @st.fragment(run_every=timedelta(seconds=interval_s) if job.is_active else None)
        def _panel() -> JobState | None:
            current = _resolve()
            return render_job_panel(current, estimated_runtime_s=estimated_runtime_s)

        return _panel()
    except Exception:  # noqa: BLE001
        render_job_panel(job, estimated_runtime_s=estimated_runtime_s)
        if job.is_active:
            time.sleep(min(interval_s, 1.0))
            st.rerun()
        return job


def render_run_action_row(
    *,
    label: str,
    key_prefix: str,
    disabled: bool = False,
    help: str | None = None,
    regenerate_required: bool = False,
    regenerate_label: str = "Regenerate existing results",
    regenerate_help: str | None = None,
    blocked_caption: str | None = None,
) -> tuple[bool, bool]:
    """Primary run button with an optional regenerate checkbox in the same row.

    When ``regenerate_required`` is True, outputs for the current selection
    already exist and the run button stays disabled until the user checks
    **Regenerate**.

    Returns ``(run_clicked, force_regenerate)``.
    """
    col_regen, col_run = st.columns([1.15, 1.85], vertical_alignment="bottom")
    force_regenerate = False
    with col_regen:
        if regenerate_required:
            force_regenerate = st.checkbox(
                regenerate_label,
                value=False,
                key=f"{key_prefix}_regen",
                help=regenerate_help or (
                    "Check to overwrite results that already exist for this selection."
                ),
            )
    with col_run:
        blocked = regenerate_required and not force_regenerate
        run_clicked = st.button(
            label,
            type="primary",
            disabled=disabled or blocked,
            key=f"{key_prefix}_btn",
            help=help,
        )
    if regenerate_required and not force_regenerate:
        st.caption(
            blocked_caption
            or "Results for this selection already exist — check **Regenerate** "
            "to run again, or open the saved figures / results tab."
        )
    return run_clicked, force_regenerate


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
