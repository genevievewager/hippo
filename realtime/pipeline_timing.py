"""Wall-clock timing for full pipeline stages (sim → compare → replay → viz).

Separate from per-update causal latency (``latency_profiler`` / ``latency_benchmark``).
These timings answer: how long does each experimental axis take in a full run?
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import pandas as pd


@dataclass
class StageTiming:
    stage: str
    wall_s: float
    detail: str = ""
    notes: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class PipelineTimer:
    """Accumulate named stage wall times for a full experiment."""

    def __init__(self) -> None:
        self.records: list[StageTiming] = []
        self._t0 = time.perf_counter()

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        detail: str = "",
        notes: str = "",
        **meta: Any,
    ) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.records.append(
                StageTiming(
                    stage=str(name),
                    wall_s=float(time.perf_counter() - t0),
                    detail=str(detail),
                    notes=str(notes),
                    meta={k: _jsonable(v) for k, v in meta.items()},
                )
            )

    def add(
        self,
        name: str,
        wall_s: float,
        *,
        detail: str = "",
        notes: str = "",
        **meta: Any,
    ) -> None:
        self.records.append(
            StageTiming(
                stage=str(name),
                wall_s=float(wall_s),
                detail=str(detail),
                notes=str(notes),
                meta={k: _jsonable(v) for k, v in meta.items()},
            )
        )

    def total_s(self) -> float:
        return float(time.perf_counter() - self._t0)

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for r in self.records:
            row = {
                "stage": r.stage,
                "detail": r.detail,
                "wall_s": r.wall_s,
                "wall_min": r.wall_s / 60.0,
                "notes": r.notes,
            }
            row.update({f"meta_{k}": v for k, v in r.meta.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def summarize_major_stages(self) -> pd.DataFrame:
        """One row per top-level stage (sum of matching detail rows)."""
        if not self.records:
            return pd.DataFrame(columns=["stage", "wall_s", "wall_min", "n_substeps", "notes"])
        df = self.to_frame()
        # Prefer explicit major stage names; otherwise group by stage.
        major_order = (
            "simulation",
            "decoder_comparison",
            "comparison_window",
            "comparison_feature_set",
            "comparison_embedding",
            "deployment_selection",
            "temporal_comparison",
            "closed_loop_replay",
            "latency_benchmark",
            "visualization",
            "pdf_compile",
        )
        rows = []
        for stage in major_order:
            sub = df[df["stage"] == stage]
            if sub.empty:
                continue
            notes = "; ".join(
                sorted({str(n) for n in sub["notes"].tolist() if str(n).strip()})
            )
            rows.append({
                "stage": stage,
                "wall_s": float(sub["wall_s"].sum()),
                "wall_min": float(sub["wall_s"].sum()) / 60.0,
                "n_substeps": int(len(sub)),
                "notes": notes,
            })
        # Any other stages
        known = set(major_order)
        for stage, sub in df.groupby("stage"):
            if stage in known:
                continue
            rows.append({
                "stage": stage,
                "wall_s": float(sub["wall_s"].sum()),
                "wall_min": float(sub["wall_s"].sum()) / 60.0,
                "n_substeps": int(len(sub)),
                "notes": "",
            })
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values("wall_s", ascending=False).reset_index(drop=True)
        return out

    def save(self, output_dir: Path, *, experiment_dir: Path | None = None) -> dict[str, Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        detail = self.to_frame()
        summary = self.summarize_major_stages()
        # Attach simulation wall time from summary.json when available.
        if experiment_dir is not None:
            sim_s = _read_simulation_wall_s(Path(experiment_dir))
            if sim_s is not None and (
                summary.empty or "simulation" not in summary["stage"].astype(str).tolist()
            ):
                self.add(
                    "simulation",
                    sim_s,
                    detail="from summary.json",
                    notes="Wall time recorded during run_simulation.py",
                )
                detail = self.to_frame()
                summary = self.summarize_major_stages()

        detail_path = output_dir / "pipeline_stage_timing.csv"
        summary_path = output_dir / "pipeline_stage_timing_summary.csv"
        json_path = output_dir / "pipeline_stage_timing.json"
        detail.to_csv(detail_path, index=False)
        summary.to_csv(summary_path, index=False)
        payload = {
            "total_tracked_s": float(detail["wall_s"].sum()) if not detail.empty else 0.0,
            "timer_elapsed_s": self.total_s(),
            "records": [asdict(r) for r in self.records],
            "summary": summary.to_dict(orient="records") if not summary.empty else [],
        }
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        return {
            "detail": detail_path,
            "summary": summary_path,
            "json": json_path,
        }


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


def _read_simulation_wall_s(experiment_dir: Path) -> float | None:
    for path in (
        experiment_dir / "summary.json",
        experiment_dir / "simulation_summary.json",
    ):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for key in ("wall_time_s", "simulation_wall_s", "elapsed_s"):
            if key in data and data[key] is not None:
                try:
                    return float(data[key])
                except (TypeError, ValueError):
                    pass
    return None


def load_pipeline_timing_summary(experiment_dir: Path) -> pd.DataFrame:
    path = Path(experiment_dir) / "latency_profiling" / "pipeline_stage_timing_summary.csv"
    if not path.exists():
        # Also accept comparison-local copies
        alt = Path(experiment_dir) / "decoder_comparison" / "pipeline_stage_timing_summary.csv"
        path = alt if alt.exists() else path
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
