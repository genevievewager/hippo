"""Background job runner so sidebar navigation does not abort long UI work.

Jobs run in daemon threads. Progress is stored in a process-local registry
(not Streamlit widgets), so leaving a page only stops rendering — the worker
keeps going. Pages poll ``get_job`` on each rerun / fragment tick.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]
JobFn = Callable[..., Any]


@dataclass
class JobState:
    job_id: str
    kind: str
    label: str
    status: str = "pending"  # pending | running | completed | failed | cancelled
    message: str = ""
    step: int = 0
    total: int = 1
    error: str | None = None
    traceback: str | None = None
    result: Any = None
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    eta_remaining_s: float | None = None
    eta_updated_at: float | None = None

    @property
    def is_active(self) -> bool:
        return self.status in ("pending", "running")

    @property
    def elapsed_s(self) -> float:
        t0 = self.started_at or self.created_at
        t1 = self.finished_at or time.time()
        return max(t1 - t0, 0.0)

    def progress_fraction(self) -> float:
        total = max(int(self.total), 1)
        return min(max(int(self.step), 0) / total, 1.0)


_LOCK = threading.RLock()
_JOBS: dict[str, JobState] = {}
# Optional slot keys → job_id so a page can resume its job after navigation.
_SLOTS: dict[str, str] = {}


def _new_id(kind: str) -> str:
    return f"{kind}-{uuid.uuid4().hex[:10]}"


def get_job(job_id: str | None) -> JobState | None:
    if not job_id:
        return None
    with _LOCK:
        return _JOBS.get(job_id)


def get_slot_job(slot: str) -> JobState | None:
    with _LOCK:
        job_id = _SLOTS.get(slot)
        if not job_id:
            return None
        return _JOBS.get(job_id)


def bind_slot(slot: str, job_id: str) -> None:
    with _LOCK:
        _SLOTS[slot] = job_id


def clear_slot(slot: str) -> None:
    with _LOCK:
        _SLOTS.pop(slot, None)


def list_jobs(*, active_only: bool = False) -> list[JobState]:
    with _LOCK:
        jobs = list(_JOBS.values())
    if active_only:
        jobs = [j for j in jobs if j.is_active]
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs


def update_progress(
    job_id: str,
    message: str,
    step: int,
    total: int,
) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        job.message = str(message)
        job.step = int(step)
        job.total = max(int(total), 1)
        if job.status == "pending":
            job.status = "running"


def make_progress_callback(job_id: str) -> ProgressCallback:
    def _cb(message: str, step: int, total: int) -> None:
        update_progress(job_id, message, step, total)

    return _cb


def submit_job(
    *,
    kind: str,
    label: str,
    fn: JobFn,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    slot: str | None = None,
    meta: dict[str, Any] | None = None,
    pass_progress: bool = True,
    replace_active_slot: bool = False,
) -> JobState:
    """Start ``fn`` in a background thread.

    If ``pass_progress`` is True, ``fn`` is called with
    ``progress_callback=make_progress_callback(job_id)`` (added to kwargs).
    """
    kwargs = dict(kwargs or {})
    with _LOCK:
        if slot and not replace_active_slot:
            existing_id = _SLOTS.get(slot)
            existing = _JOBS.get(existing_id) if existing_id else None
            if existing is not None and existing.is_active:
                return existing

        job_id = _new_id(kind)
        job = JobState(
            job_id=job_id,
            kind=kind,
            label=label,
            status="pending",
            message="Queued…",
            meta=dict(meta or {}),
        )
        _JOBS[job_id] = job
        if slot:
            _SLOTS[slot] = job_id

    if pass_progress:
        kwargs = {**kwargs, "progress_callback": make_progress_callback(job_id)}

    def _runner() -> None:
        with _LOCK:
            job.status = "running"
            job.started_at = time.time()
            job.message = "Starting…"
        try:
            result = fn(*args, **kwargs)
            with _LOCK:
                job.result = result
                job.status = "completed"
                job.message = "Complete"
                job.step = max(job.step, job.total)
                job.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Background job %s failed", job_id)
            with _LOCK:
                job.status = "failed"
                job.error = str(exc)
                job.traceback = traceback.format_exc()
                job.message = f"Failed: {exc}"
                job.finished_at = time.time()

    thread = threading.Thread(
        target=_runner,
        name=f"hippo-job-{job_id}",
        daemon=True,
    )
    thread.start()
    return job


def active_job_summary() -> list[dict[str, Any]]:
    """Compact summaries for sidebar display."""
    out: list[dict[str, Any]] = []
    for job in list_jobs(active_only=True):
        out.append({
            "job_id": job.job_id,
            "kind": job.kind,
            "label": job.label,
            "message": job.message,
            "step": job.step,
            "total": job.total,
            "elapsed_s": job.elapsed_s,
            "fraction": job.progress_fraction(),
        })
    return out
