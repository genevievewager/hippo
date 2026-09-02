"""Work-based progress tracking (completed / total), independent of wall-clock.

Percent complete is always ``completed / total``. While a task is still running
the displayed fraction is capped at ``RUNNING_FRACTION_CAP`` so the bar never
reads 100% until ``complete()``. ETA is a separate EMA of per-unit duration
(optionally seeded from a JSON history file). History never drives the percent.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Mapping

ProgressCallback = Callable[..., None]

RUNNING_FRACTION_CAP = 0.99
EMA_ALPHA = 0.3
MIN_UNITS_FOR_ETA = 2
HISTORY_CAP = 200
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"

DEFAULT_HISTORY_PATH = Path("outputs") / ".cache" / "progress_history.json"


def running_fraction(
    completed: int,
    total: int,
    *,
    status: str = STATUS_RUNNING,
) -> float:
    """Work-based fraction; 1.0 only after a successful ``complete``."""
    if status == STATUS_COMPLETE:
        return 1.0
    tot = int(total)
    if tot <= 0:
        return 0.0
    raw = min(max(int(completed), 0) / tot, 1.0)
    if status == STATUS_ERROR:
        return min(raw, RUNNING_FRACTION_CAP)
    return min(raw, RUNNING_FRACTION_CAP)


def update_ema(
    prev: float | None,
    sample_s: float,
    *,
    alpha: float = EMA_ALPHA,
) -> float:
    sample = max(float(sample_s), 0.0)
    if prev is None:
        return sample
    a = min(max(float(alpha), 0.0), 1.0)
    return a * sample + (1.0 - a) * float(prev)


def eta_seconds(
    *,
    completed: int,
    total: int,
    ema_unit_s: float | None,
    n_measured: int,
    min_units: int = MIN_UNITS_FOR_ETA,
    prior_unit_s: float | None = None,
) -> float | None:
    """Seconds remaining from EMA (or history prior). Never uses elapsed/estimate as %."""
    remaining = max(int(total) - int(completed), 0)
    if remaining <= 0:
        return 0.0
    if int(n_measured) >= int(min_units) and ema_unit_s is not None:
        return max(float(ema_unit_s), 0.0) * remaining
    if prior_unit_s is not None:
        try:
            return max(float(prior_unit_s), 0.0) * remaining
        except (TypeError, ValueError):
            return None
    return None


def invoke_progress(
    callback: ProgressCallback | None,
    message: str,
    step: int,
    total: int,
    *,
    stage: str = "",
    task: str = "",
) -> None:
    """Call ``(message, step, total)``; extra kwargs are optional for older callbacks."""
    if callback is None:
        return
    try:
        callback(str(message), int(step), int(total), stage=stage, task=task)
    except TypeError:
        callback(str(message), int(step), int(total))


def default_history_path() -> Path:
    return DEFAULT_HISTORY_PATH


def load_progress_history(path: Path | str | None = None) -> list[dict[str, Any]]:
    loc = Path(path) if path is not None else default_history_path()
    try:
        raw = json.loads(loc.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return []
    if isinstance(raw, list):
        return [r for r in raw if isinstance(r, dict)]
    if isinstance(raw, dict):
        records = raw.get("records")
        if isinstance(records, list):
            return [r for r in records if isinstance(r, dict)]
    return []


def prior_unit_seconds(
    operation: str,
    path: Path | str | None = None,
    *,
    extra: Mapping[str, Any] | None = None,
) -> float | None:
    """Median seconds-per-unit from comparable history rows, or None."""
    if not operation:
        return None
    rows = load_progress_history(path)
    wanted = {str(k): extra[k] for k in extra or {} if extra[k] is not None}
    samples: list[float] = []
    for row in rows:
        if str(row.get("operation") or "") != str(operation):
            continue
        if wanted and any(str(row.get(k)) != str(v) for k, v in wanted.items()):
            continue
        n_units = row.get("n_units", row.get("total"))
        duration = row.get("duration_s")
        try:
            n = float(n_units)
            d = float(duration)
        except (TypeError, ValueError):
            continue
        if n > 0 and d >= 0:
            samples.append(d / n)
    if not samples:
        return None
    tail = samples[-10:]
    try:
        return float(statistics.median(tail))
    except statistics.StatisticsError:
        return None


def append_progress_history(
    operation: str,
    duration_s: float,
    n_units: int,
    path: Path | str | None = None,
    **extra: Any,
) -> None:
    """Append one finished-run row; create the cache dir; cap length. Never raises."""
    if not operation:
        return
    loc = Path(path) if path is not None else default_history_path()
    try:
        loc.parent.mkdir(parents=True, exist_ok=True)
        records = load_progress_history(loc)
        row: dict[str, Any] = {
            "operation": str(operation),
            "duration_s": float(duration_s),
            "n_units": int(max(n_units, 0)),
            "recorded_at": time.time(),
        }
        for key, value in extra.items():
            if value is not None:
                row[str(key)] = value
        records.append(row)
        if len(records) > HISTORY_CAP:
            records = records[-HISTORY_CAP:]
        loc.write_text(json.dumps(records, indent=2) + "\n")
    except (OSError, TypeError, ValueError):
        return


class WorkTracker:
    """Count completed work units. ETA is derived, never used as percent."""

    def __init__(
        self,
        total: int,
        *,
        operation: str = "",
        history_path: Path | str | None = None,
        prior_unit_s: float | None = None,
        alpha: float = EMA_ALPHA,
    ) -> None:
        self.total = max(int(total), 0)
        self.completed = 0
        self.stage = ""
        self.task = ""
        self.message = ""
        self.status = STATUS_RUNNING
        self.operation = str(operation or "")
        self._history_path = Path(history_path) if history_path is not None else None
        self._alpha = float(alpha)
        self._t0 = time.perf_counter()
        self._last_unit_t = self._t0
        self.ema_unit_s: float | None = None
        self.n_measured = 0
        if prior_unit_s is not None:
            try:
                self.prior_unit_s: float | None = float(prior_unit_s)
            except (TypeError, ValueError):
                self.prior_unit_s = None
        elif self.operation:
            self.prior_unit_s = prior_unit_seconds(
                self.operation, self._history_path,
            )
        else:
            self.prior_unit_s = None

    @property
    def elapsed_s(self) -> float:
        return max(time.perf_counter() - self._t0, 0.0)

    @property
    def fraction(self) -> float:
        return running_fraction(self.completed, self.total, status=self.status)

    @property
    def eta_s(self) -> float | None:
        if self.status != STATUS_RUNNING:
            return 0.0 if self.status == STATUS_COMPLETE else None
        return eta_seconds(
            completed=self.completed,
            total=self.total,
            ema_unit_s=self.ema_unit_s,
            n_measured=self.n_measured,
            prior_unit_s=self.prior_unit_s,
        )

    def set_total(self, total: int) -> None:
        self.total = max(int(total), 0)

    def bump(
        self,
        n: int = 1,
        *,
        message: str | None = None,
        stage: str | None = None,
        task: str | None = None,
    ) -> None:
        if stage is not None:
            self.stage = str(stage)
        if task is not None:
            self.task = str(task)
        if message is not None:
            self.message = str(message)
        n_add = max(int(n), 0)
        if n_add <= 0:
            return
        now = time.perf_counter()
        sample = (now - self._last_unit_t) / n_add
        self.ema_unit_s = update_ema(self.ema_unit_s, sample, alpha=self._alpha)
        self.n_measured += n_add
        self._last_unit_t = now
        self.completed = min(self.completed + n_add, self.total)

    def complete(self, message: str = "Complete") -> None:
        self.status = STATUS_COMPLETE
        self.completed = self.total
        self.message = message
        if self.operation:
            append_progress_history(
                self.operation,
                self.elapsed_s,
                max(self.total, 1),
                self._history_path,
            )

    def fail(self, message: str = "Failed") -> None:
        self.status = STATUS_ERROR
        self.message = str(message)

    def notify(self, callback: ProgressCallback | None) -> None:
        invoke_progress(
            callback,
            self.message or self.task or self.stage,
            self.completed,
            max(self.total, 1),
            stage=self.stage,
            task=self.task,
        )
