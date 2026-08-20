"""Run status / progress panels."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Any, Callable

import streamlit as st

from ui.jobs import JobState, get_job, get_slot_job, list_jobs
from ui.services.comparison import format_duration
from ui.services.registry import RunMetadata


def remaining_s(
    elapsed: float,
    step: int,
    total: int,
    estimate_s: float | None = None,
    prev_remaining: float | None = None,
    *,
    dt: float = 0.0,
) -> float | None:
    """Non-increasing remaining-time estimate in seconds.

    Uses the pre-run estimate until pace is trustworthy (``step >= 2``), then
    pace ETA. Displayed remaining never increases; optional ``dt`` counts it
    down between polls when the step counter is stuck.
    """
    elapsed = max(float(elapsed or 0.0), 0.0)
    step = max(int(step or 0), 0)
    total = max(int(total or 1), 1)
    dt = max(float(dt or 0.0), 0.0)

    candidate: float | None = None
    if step >= 2 and elapsed > 0.0:
        candidate = (elapsed / float(step)) * float(max(total - step, 0))
    elif estimate_s is not None:
        try:
            candidate = max(float(estimate_s) - elapsed, 0.0)
        except (TypeError, ValueError):
            candidate = None
    elif step >= 1 and elapsed > 0.0:
        candidate = (elapsed / float(step)) * float(max(total - step, 0))

    if candidate is None:
        if prev_remaining is None:
            return None
        return max(float(prev_remaining) - dt, 0.0)

    if prev_remaining is not None:
        ceiling = max(float(prev_remaining) - dt, 0.0)
        candidate = min(float(candidate), ceiling)
    return max(float(candidate), 0.0)


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
        f"Estimated runtime **~{eta}** (likely {eta_range}; heuristic from selected configs)"
    )


class RuntimeProgressTracker:
    """Live progress bar with elapsed time and ETA while a benchmark runs."""

    def __init__(
        self,
        *,
        estimated_runtime_s: float | None = None,
        label: str = "Starting decoder comparison…",
    ) -> None:
        self.estimated_runtime_s = (
            float(estimated_runtime_s) if estimated_runtime_s is not None else None
        )
        self.t0 = time.perf_counter()
        self._prev_remaining: float | None = None
        self.progress = st.progress(0.0, text=label)
        self.status_box = st.empty()
        self.timing_box = st.empty()
        self._update_timing(step=0, total=1, msg=label)

    def _eta_s(self, step: int, total: int, elapsed: float) -> float | None:
        eta = remaining_s(
            elapsed,
            step,
            total,
            self.estimated_runtime_s,
            getattr(self, "_prev_remaining", None),
        )
        self._prev_remaining = eta
        return eta

    def _update_timing(self, *, step: int, total: int, msg: str) -> None:
        elapsed = time.perf_counter() - self.t0
        eta = self._eta_s(step, total, elapsed)
        eta_label = format_duration(eta) if eta is not None else "—"
        self.timing_box.caption(
            f"Elapsed **{format_duration(elapsed)}** · "
            f"remaining ~**{eta_label}** · "
            f"step {step}/{max(total, 1)}"
        )
        self.status_box.caption(msg)

    def callback(self) -> Callable[[str, int, int], None]:
        def _cb(msg: str, step: int, n: int) -> None:
            total = max(int(n), 1)
            frac = min(max(int(step), 0) / total, 1.0)
            self.progress.progress(frac, text=f"[{step}/{n}] {msg}")
            self._update_timing(step=step, total=total, msg=msg)

        return _cb

    def complete(self, text: str = "Complete") -> None:
        elapsed = time.perf_counter() - self.t0
        self.progress.progress(1.0, text=text)
        self.timing_box.caption(f"Finished in **{format_duration(elapsed)}**")
        self.status_box.caption(text)


def render_job_panel(
    job: JobState | None,
    *,
    estimated_runtime_s: float | None = None,
) -> JobState | None:
    """Render progress for a background job (safe across page navigation)."""
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
    label = f"[{job.step}/{max(job.total, 1)}] {job.message or job.status}"
    st.progress(frac, text=label)

    elapsed = job.elapsed_s
    eta = None
    if job.is_active:
        estimate = estimated_runtime_s
        if estimate is None and job.meta:
            raw_est = job.meta.get("estimated_runtime_s")
            try:
                estimate = float(raw_est) if raw_est is not None else None
            except (TypeError, ValueError):
                estimate = None
        now = time.time()
        dt = (now - job.eta_updated_at) if job.eta_updated_at else 0.0
        eta = remaining_s(
            elapsed,
            job.step,
            job.total,
            estimate,
            job.eta_remaining_s,
            dt=dt,
        )
        job.eta_remaining_s = eta
        job.eta_updated_at = now
    eta_label = format_duration(eta) if eta is not None else "—"
    st.caption(
        f"Elapsed **{format_duration(elapsed)}**"
        + (f" · remaining ~**{eta_label}**" if job.is_active else "")
        + f" · `{job.job_id}`"
    )
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


def render_sidebar_jobs() -> None:
    """Show active background jobs in the sidebar (call from app.py)."""
    active = list_jobs(active_only=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Background jobs")
    if not active:
        st.sidebar.caption("No running jobs — safe to navigate freely.")
        return
    for job in active:
        st.sidebar.markdown(f"**{job.label}**")
        st.sidebar.progress(job.progress_fraction())
        st.sidebar.caption(
            f"{job.message or job.status} · {format_duration(job.elapsed_s)}"
        )
    try:
        @st.fragment(run_every=timedelta(seconds=3))
        def _tick() -> None:
            jobs = list_jobs(active_only=True)
            if jobs:
                st.caption(f"{len(jobs)} active · updating…")
            else:
                st.caption("All jobs finished.")

        _tick()
    except Exception:  # noqa: BLE001
        pass


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
