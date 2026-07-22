"""Tests for latency profiling helpers."""

from __future__ import annotations

from realtime.latency_profiler import (
    LatencySample,
    save_latency_artifacts,
    summarize_latency_samples,
)


def test_summarize_and_save_latency(tmp_path):
    samples = [
        LatencySample(
            time_s=1.0,
            stages_ms={
                "spike_binning": 0.2,
                "feature_transform": 0.5,
                "decode_position": 1.0,
                "total_update": 2.0,
            },
        ),
        LatencySample(
            time_s=1.05,
            stages_ms={
                "spike_binning": 0.3,
                "feature_transform": 0.4,
                "decode_position": 1.2,
                "total_update": 2.2,
            },
        ),
    ]
    summary = summarize_latency_samples(samples, update_budget_ms=50.0)
    assert summary["n_updates"] == 2
    assert summary["stages"]["total_update"]["mean_ms"] == 2.1
    assert summary["within_budget_frac"] == 1.0

    out = save_latency_artifacts(samples, tmp_path / "lat", update_budget_ms=50.0)
    assert (tmp_path / "lat" / "latency_per_update.csv").exists()
    assert (tmp_path / "lat" / "latency_by_stage.csv").exists()
    assert (tmp_path / "lat" / "latency_summary.json").exists()
    assert out["mean_total_ms"] == 2.1
