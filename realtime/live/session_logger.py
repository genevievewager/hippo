"""Session logging for live / replay deployment runs."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class SessionLogger:
    """Write predictions, runtime metrics, and events under live_sessions/."""

    def __init__(self, session_dir: Path):
        self.session_dir = Path(session_dir)
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._pred_rows: list[dict[str, Any]] = []
        self._metric_rows: list[dict[str, Any]] = []
        self._events_path = self.session_dir / "events.log"
        self._pred_path = self.session_dir / "predictions.csv"
        self._metrics_path = self.session_dir / "runtime_metrics.csv"

    @classmethod
    def create(
        cls,
        experiment_dir: Path | str,
        *,
        prefix: str = "session",
    ) -> "SessionLogger":
        experiment_dir = Path(experiment_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        root = experiment_dir / "live_sessions" / f"{prefix}_{stamp}"
        return cls(root)

    def write_json(self, name: str, payload: dict[str, Any]) -> Path:
        path = self.session_dir / name
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n")
        return path

    def log_event(self, message: str, **fields: Any) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        extra = (" " + json.dumps(fields, default=str)) if fields else ""
        line = f"{ts} {message}{extra}\n"
        with open(self._events_path, "a") as f:
            f.write(line)

    def log_prediction(self, row: dict[str, Any]) -> None:
        self._pred_rows.append(dict(row))

    def log_runtime(self, row: dict[str, Any]) -> None:
        self._metric_rows.append(dict(row))

    def flush(self) -> None:
        if self._pred_rows:
            pd.DataFrame(self._pred_rows).to_csv(self._pred_path, index=False)
        if self._metric_rows:
            pd.DataFrame(self._metric_rows).to_csv(self._metrics_path, index=False)

    def close(self) -> None:
        self.flush()
        self.log_event("session_closed")
