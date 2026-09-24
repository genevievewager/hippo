"""Realtime replay arm for sorted spikes at primary d (Phase 3)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import (
    METHOD_KEYS,
    OUTPUT_ROOT,
    _euclid,
    _fit_predict_ridge,
    _make_rep,
    _position,
    _result_payload,
    _sqrt_zscore_train,
    _valid_mask,
    _write_json,
    fit_transform_representation,
)
from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.spike_binner import build_causal_spike_matrix
from realtime.timing import extract_behavior_times
from realtime.train_decoder import align_behavior_to_decoder_times, causal_train_test_split


def _step_latents(model, X: np.ndarray) -> tuple[np.ndarray, list[float]]:
    if hasattr(model, "reset_state"):
        model.reset_state()
    rows = []
    times = []
    stepper = None
    if hasattr(model, "transform_one"):
        stepper = model.transform_one
    elif hasattr(model, "step"):
        stepper = model.step
    if stepper is None:
        raise TypeError("no per-step path")
    for i in range(len(X)):
        t0 = time.perf_counter()
        rows.append(np.asarray(stepper(X[i]), dtype=float).ravel())
        times.append((time.perf_counter() - t0) * 1000.0)
    return np.vstack(rows), times


def replay_seed(seed_index: int) -> dict[str, Any]:
    cfg = load_quadrant_n5_yaml()
    src = "sorted"
    sim_dir = OUTPUT_ROOT / f"seed_{seed_index}" / "sim"
    summary = json.loads((OUTPUT_ROOT / f"seed_{seed_index}" / src / "source_summary.json").read_text())
    feat = cfg["features"]
    data = load_simulation_data(
        sim_dir, src, include_regions=list(cfg["unit_inclusion"]["regions"]),
    )
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = float(feat["update_dt"])
    W = float(feat["window_s"])
    decode_times = make_decode_times(
        data["session_duration"], W, update_dt, behavior_times=behavior_times,
    )
    X_counts = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, W,
    )
    train_mask, test_mask = causal_train_test_split(
        decode_times, float(cfg["split"]["train_frac"]), gap_s=float(cfg["split"]["gap_s"]),
    )
    X, _ = _sqrt_zscore_train(X_counts, train_mask)
    n = len(decode_times)
    valid = {m: _valid_mask(m, n, cfg) for m in METHOD_KEYS}
    eval_mask = test_mask.copy()
    for m in METHOD_KEYS:
        eval_mask &= valid[m]
    train_ok = train_mask.copy()
    for m in METHOD_KEYS:
        train_ok &= valid[m]
    beh = align_behavior_to_decoder_times(data["behavior_df"], decode_times)
    y = _position(beh)
    methods_seed = int(summary["seed_streams"]["methods"])
    budget = float(cfg["latency_budget_ms"])
    out_rows = []
    for rec in summary["methods"]:
        key = rec["method"]
        reducing = key not in ("raw", "raw_lag")
        d_final = int(rec["primary_d"]) if reducing else 3
        nested = bool((cfg["representations"].get(key) or {}).get("nested"))
        d_fit = int(cfg["nested_fit_d"]) if (nested and reducing) else d_final
        model = _make_rep(key, d_fit, cfg, methods_seed, n_fit=int(train_ok.sum()))
        Z_batch = fit_transform_representation(key, model, X, train_ok)
        if nested and reducing:
            Z_batch = Z_batch[:, : int(rec["primary_d"])]
        label = "offline_only"
        step_ms: list[float] = []
        max_abs = None
        try:
            Z_step, step_ms = _step_latents(model, X)
            if nested and reducing:
                Z_step = Z_step[:, : int(rec["primary_d"])]
            max_abs = float(np.max(np.abs(Z_batch[eval_mask] - Z_step[eval_mask])))
            label = "realtime_compatible" if max_abs <= 1e-5 else "offline_only"
        except Exception as exc:
            label = f"offline_only ({type(exc).__name__})"
        pred = _fit_predict_ridge(
            Z_batch[train_ok], y[train_ok], Z_batch[eval_mask], float(rec["ridge_alpha"]),
        )
        offline = _euclid(pred, y[eval_mask])
        over = float(np.mean(np.asarray(step_ms) > budget)) if step_ms else None
        row = {
            "method": key,
            "a9_label": label,
            "expected_label": (cfg["representations"].get(key) or {}).get("expected_label"),
            "max_abs_step_vs_batch": max_abs,
            "offline_ridge": offline,
            "step_ms_p50": float(np.median(step_ms)) if step_ms else None,
            "step_ms_p99": float(np.quantile(step_ms, 0.99)) if step_ms else None,
            "step_ms_max": float(np.max(step_ms)) if step_ms else None,
            "frac_over_budget": over,
        }
        out_rows.append(row)
    payload = _result_payload(cfg, {
        "stage": "replay",
        "seed_index": seed_index,
        "spike_source": "sorted",
        "methods": out_rows,
    })
    dest = OUTPUT_ROOT / f"seed_{seed_index}" / "replay"
    _write_json(dest / "sorted_summary.json", payload)
    return payload


if __name__ == "__main__":
    import sys
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(json.dumps(replay_seed(idx), indent=2, default=str))
