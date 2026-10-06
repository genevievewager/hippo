"""Fault tolerance for M3 per-method failures (run_m3 only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis.real_quadrant import run_m3


def test_run_methods_fault_tolerant_records_failure_and_continues(tmp_path, monkeypatch):
    out_dir = tmp_path / "sess"
    out_dir.mkdir()
    monkeypatch.setattr(run_m3, "_session_out_dir", lambda s: out_dir)

    calls: list[tuple[str, ...]] = []

    def fake_analyze(
        cfg, sim_dir, spike_source, streams, seed_index, *,
        bundle=None, output_dir=None, method_keys=None, save_figure_contract=False,
    ):
        keys = tuple(method_keys or ())
        calls.append(keys)
        out = Path(output_dir)
        if keys == ("boom",):
            raise RuntimeError("synthetic boom")
        if len(keys) == 1:
            method = keys[0]
            (out / f"{method}.json").write_text(json.dumps({
                "method": method,
                "primary_d": 2,
                "ridge_alpha": 1.0,
                "knn_k": 5,
                "ridge": {"median": 10.0, "mean": 10.0},
                "knn": {"median": 11.0, "mean": 11.0},
                "config_sha256": cfg.get("config_sha256", "cfg"),
                "git_sha": "deadbeef",
                "dirty_tree": False,
                "elapsed_s": 0.1,
                "n_train": 50,
                "n_eval": 20,
                "a13": {"status": "pass"},
            }) + "\n")
            return {"methods": [{"method": method, "ridge": {"median": 10.0},
                                 "knn": {"median": 11.0}, "primary_d": 2,
                                 "a13": {"status": "pass"}}],
                    "n_units": 3, "n_train": 50, "n_eval": 20,
                    "floor": {"median": 20.0},
                    "config_sha256": cfg.get("config_sha256", "cfg"),
                    "git_sha": "deadbeef"}
        # Final multi-method contract rebuild.
        assert set(keys) == {"raw", "pca"}
        contract = out / "figure_contract"
        contract.mkdir(parents=True, exist_ok=True)
        n = 40
        times = np.arange(n, dtype=float) * 0.05
        y = np.column_stack([times, times])
        train_ok = np.zeros(n, dtype=bool)
        train_ok[:30] = True
        eval_mask = ~train_ok
        np.savez_compressed(
            contract / "arrays.npz",
            decode_times=times, y=y, train_ok=train_ok, eval_mask=eval_mask,
            valid=np.ones(n, dtype=bool), retained=np.ones(n, dtype=bool),
            train_mask=train_ok, test_mask=eval_mask,
            y_floor=np.array([[0.0, 0.0]]),
        )
        Z = np.random.default_rng(0).normal(size=(n, 2))
        np.savez_compressed(contract / "latents.npz", Z_raw=Z, Z_pca=Z)
        pred = y[eval_mask] + 0.1
        np.savez_compressed(
            contract / "predictions.npz",
            pred_raw_ridge=pred, pred_pca_ridge=pred,
            eval_mask=eval_mask, decode_times=times[eval_mask],
            y_true=y[eval_mask],
        )
        methods = []
        for m in keys:
            methods.append({
                "method": m, "ridge": {"median": 10.0}, "knn": {"median": 11.0},
                "primary_d": 2, "a13": {"status": "pass"},
            })
        summary = {
            "methods": methods,
            "n_units": 3, "n_train": 50, "n_eval": 20,
            "floor": {"median": 20.0},
            "config_sha256": cfg.get("config_sha256", "cfg"),
            "git_sha": "deadbeef",
        }
        (out / "source_summary.json").write_text(json.dumps(summary) + "\n")
        return summary

    cfg = {"config_sha256": "cfg", "phase3": {"learning_curve_source": "__skip__"}}
    succeeded, failures, summary = run_m3.run_methods_fault_tolerant(
        cfg=cfg,
        bundle={"meta": {}},
        streams=run_m3._streams(0),
        seed_index=0,
        out_dir=out_dir,
        methods=("raw", "boom", "pca"),
        analyze_fn=fake_analyze,
    )
    assert succeeded == ["raw", "pca"]
    assert "boom" in failures
    assert failures["boom"]["status"] == "failed"
    assert "RuntimeError: synthetic boom" in failures["boom"]["error"]
    boom = json.loads((out_dir / "boom.json").read_text())
    assert boom["status"] == "failed"
    assert summary is not None
    assert {m["method"] for m in summary["methods"]} == {"raw", "pca"}
    # Per-method calls + one rebuild over successes.
    assert ("raw",) in calls and ("boom",) in calls and ("pca",) in calls
    assert ("raw", "pca") in calls


def test_smooth_skipped_when_base_failed(tmp_path):
    out_dir = tmp_path / "sess"
    out_dir.mkdir()
    run_m3.write_method_failure(out_dir, "lds", RuntimeError("nope"))
    skipped = run_m3.write_method_skipped(
        out_dir, "lds_smooth", reason="base method lds failed: RuntimeError: nope",
    )
    assert skipped["status"] == "skipped"
    rec = json.loads((out_dir / "lds_smooth.json").read_text())
    assert rec["status"] == "skipped"
    assert "lds failed" in rec["reason"]


def test_contrast_drops_animal_missing_method():
    means = {
        "A": {"lds": 0.5, "raw_smooth": 0.6},
        "B": {"raw_smooth": 0.7},  # no lds
        "C": {"lds": 0.4, "raw_smooth": 0.55},
    }
    out = run_m3._contrast_pair("lds", "raw_smooth", means)
    assert out["n_animals"] == 2
    assert set(out["per_animal"]) == {"A", "C"}
    assert out["n_a_better"] == 2  # negative diffs
