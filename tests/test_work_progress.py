"""Work-based progress tracker (percent from completed/total, ETA from EMA)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from realtime.work_progress import (
    RUNNING_FRACTION_CAP,
    WorkTracker,
    append_progress_history,
    eta_seconds,
    invoke_progress,
    load_progress_history,
    prior_unit_seconds,
    running_fraction,
)
from ui.jobs import JobState


def test_starts_at_zero():
    t = WorkTracker(10)
    assert t.completed == 0
    assert t.fraction == 0.0


def test_fraction_capped_while_running():
    t = WorkTracker(4)
    t.bump(4)
    assert t.completed == 4
    assert t.fraction == RUNNING_FRACTION_CAP
    assert t.fraction <= 0.99


def test_complete_reaches_one():
    t = WorkTracker(4)
    t.bump(4)
    t.complete()
    assert t.completed == 4
    assert t.fraction == 1.0


def test_completed_cannot_exceed_total():
    t = WorkTracker(3)
    t.bump(10)
    assert t.completed == 3


def test_eta_none_before_two_units(tmp_path: Path):
    hist = tmp_path / "progress_history.json"
    t = WorkTracker(10, operation="eta-test", history_path=hist)
    assert t.eta_s is None
    t.bump()
    assert t.eta_s is None


def test_eta_available_after_two_units(tmp_path: Path):
    hist = tmp_path / "progress_history.json"
    t = WorkTracker(10, operation="eta-test", history_path=hist)
    t.bump()
    t.bump()
    assert t.eta_s is not None
    assert t.eta_s >= 0.0


def test_fail_does_not_complete_bar():
    t = WorkTracker(5)
    t.bump(5)
    t.fail("boom")
    assert t.fraction <= RUNNING_FRACTION_CAP
    assert t.fraction != 1.0
    assert t.status == "error"


def test_empty_total_safe():
    t = WorkTracker(0)
    assert t.fraction == 0.0
    t.bump(1)
    assert t.completed == 0
    t.complete()
    assert t.fraction == 1.0


def test_missing_history_safe(tmp_path: Path):
    missing = tmp_path / "nope" / "progress_history.json"
    assert load_progress_history(missing) == []
    assert prior_unit_seconds("decoder_comparison", missing) is None


def test_malformed_history_safe(tmp_path: Path):
    path = tmp_path / "progress_history.json"
    path.write_text("{not json")
    assert load_progress_history(path) == []
    append_progress_history("op", 1.0, 2, path)
    rows = load_progress_history(path)
    assert len(rows) == 1
    assert rows[0]["operation"] == "op"


def test_history_cap(tmp_path: Path):
    path = tmp_path / "progress_history.json"
    for i in range(205):
        append_progress_history("op", float(i), 1, path)
    rows = load_progress_history(path)
    assert len(rows) == 200


def test_callback_none_is_noop():
    invoke_progress(None, "x", 1, 2)
    t = WorkTracker(2)
    t.notify(None)


def test_callback_without_kwargs():
    seen: list[tuple] = []

    def cb(message: str, step: int, total: int) -> None:
        seen.append((message, step, total))

    invoke_progress(cb, "hi", 3, 10, stage="S", task="T")
    assert seen == [("hi", 3, 10)]


def test_eta_seconds_helper():
    assert eta_seconds(
        completed=0, total=10, ema_unit_s=None, n_measured=0,
    ) is None
    assert eta_seconds(
        completed=1, total=10, ema_unit_s=2.0, n_measured=1,
    ) is None
    assert eta_seconds(
        completed=2, total=10, ema_unit_s=2.0, n_measured=2,
    ) == pytest.approx(16.0)
    assert eta_seconds(
        completed=10, total=10, ema_unit_s=2.0, n_measured=2,
    ) == 0.0
    assert eta_seconds(
        completed=0, total=10, ema_unit_s=None, n_measured=0, prior_unit_s=3.0,
    ) == pytest.approx(30.0)


def test_jobstate_caps_at_99_until_complete():
    job = JobState("id", "kind", "label", status="running", step=10, total=10)
    assert job.progress_fraction() == RUNNING_FRACTION_CAP
    job.status = "completed"
    assert job.progress_fraction() == 1.0
    job.status = "failed"
    job.step = 10
    assert job.progress_fraction() <= RUNNING_FRACTION_CAP


def test_running_fraction_status():
    assert running_fraction(0, 10) == 0.0
    assert running_fraction(10, 10, status="running") == RUNNING_FRACTION_CAP
    assert running_fraction(10, 10, status="complete") == 1.0


def test_count_decoder_eval_jobs_stable():
    from realtime.decoder_comparison import ComparisonRunConfig, count_decoder_eval_jobs
    from realtime.search_space import expand_fe_jobs

    cfg = ComparisonRunConfig(
        input_dir=Path("."),
        output_dir=Path("."),
        feature_sets=("counts",),
        feature_types=("counts",),
        embedding_types=("identity",),
        use_fe_grid=True,
        decode_windows=(0.25,),
        max_models="quick",
        targets=("speed",),
        decoder_names=("ridge",),
    )
    jobs = expand_fe_jobs(
        feature_types=("counts",),
        embedding_types=("identity",),
        use_fe_grid=True,
        max_models="quick",
    )
    n1 = count_decoder_eval_jobs(cfg, jobs, ["counts"], [0.25])
    n2 = count_decoder_eval_jobs(cfg, jobs, ["counts"], [0.25])
    assert n1 == n2
    assert n1 == 1


def test_fit_and_evaluate_callback_on_off_identity():
    from realtime.decoder_comparison import _fit_and_evaluate

    rng = np.random.default_rng(0)
    n = 40
    x = rng.normal(size=(n, 4))
    speed = x[:, 0] + 0.05 * rng.normal(size=n)
    beh = pd.DataFrame({"speed": speed})
    x_tr, x_te = x[:28], x[28:]
    b_tr, b_te = beh.iloc[:28], beh.iloc[28:]
    calls: list[tuple] = []

    def cb(message: str, step: int, total: int) -> None:
        calls.append((message, step, total))

    def run(progress=None) -> dict:
        if progress is not None:
            progress("start", 0, 1)
        fit = _fit_and_evaluate(
            x_tr, x_te, b_tr, b_te, "speed", "ridge", 0, 1, None,
        )
        if progress is not None:
            progress("done", 1, 1)
        return {k: fit.metrics[k] for k in sorted(fit.metrics)}

    off = run(None)
    on = run(cb)
    assert off == on
    assert calls == [("start", 0, 1), ("done", 1, 1)]
    assert "r2" in off
