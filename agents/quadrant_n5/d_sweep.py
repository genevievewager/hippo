"""Stage 3 latent-d sweep via Phase 3's per-fold representation refit."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.quadrant_n5.export_predictions import (
    build_observation,
    load_quadrant_n5_yaml,
    parse_seeds,
)
from realtime.quadrant_n5_run import (
    D_SWEEP_SELECTION_RULE,
    OUTPUT_ROOT,
    REDUCING_SWEEP,
    _inner_cv_select,
    ridge_alpha_grid,
    score_d_curve,
    write_d_sweep_json,
)

TOL_CM = 1e-6
TOL_ALPHA = 1e-12


def estimate_hours(results_root: Path) -> dict[str, Any]:
    totals = {k: 0.0 for k in REDUCING_SWEEP}
    for seed in range(5):
        for src in ("sorted", "ground_truth"):
            for key in REDUCING_SWEEP:
                rec = json.loads(
                    (results_root / f"seed_{seed}" / src / f"{key}.json").read_text()
                )
                totals[key] += float(rec.get("elapsed_s") or 0.0)
    extra = totals["lds"] + totals["gpfa"]
    return {
        "phase3_seconds": totals,
        "phase3_hours": {k: v / 3600.0 for k, v in totals.items()},
        "phase3_total_hours": sum(totals.values()) / 3600.0,
        "extra_lds_gpfa_hours": 0.15 * extra / 3600.0,
        "estimate_hours_lo": sum(totals.values()) / 3600.0,
        "estimate_hours_hi": (sum(totals.values()) + 0.15 * extra) / 3600.0,
        "note": (
            "Phase 3 elapsed_s already includes per-fold, per-d representation "
            "refits plus one final fit at the selected d. Stage 3 re-runs that "
            "inner CV and adds final fits at the other d values (nested methods "
            "slice the d=20 fit; LDS/GPFA refit each leftover d)."
        ),
    }


def _load_partial(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.is_file():
        return {}
    rec = json.loads(path.read_text())
    return {str(k): list(v) for k, v in (rec.get("methods") or {}).items()}


def _save_partial(path: Path, methods: dict[str, list[dict[str, Any]]]) -> None:
    path.write_text(json.dumps({"methods": methods}, indent=2) + "\n")


def gate_cell(
    seed: int,
    source: str,
    method: str,
    rows: list[dict[str, Any]],
    saved: dict[str, Any],
) -> dict[str, Any]:
    sel = next((r for r in rows if r.get("selected")), None)
    if sel is None:
        raise RuntimeError(f"{method}: no selected d row")
    d_ok = int(sel["d"]) == int(saved["primary_d"])
    a_ok = abs(float(sel["ridge_alpha"]) - float(saved["ridge_alpha"])) <= TOL_ALPHA
    k_ok = int(sel["knn_k"]) == int(saved["knn_k"])
    ridge_diff = abs(float(sel["ridge_median"]) - float(saved["ridge"]["median"]))
    knn_diff = abs(float(sel["knn_median"]) - float(saved["knn"]["median"]))
    ok = d_ok and a_ok and k_ok and ridge_diff <= TOL_CM and knn_diff <= TOL_CM
    return {
        "seed": seed,
        "source": source,
        "method": method,
        "saved_d": int(saved["primary_d"]),
        "sweep_d": int(sel["d"]),
        "saved_alpha": float(saved["ridge_alpha"]),
        "sweep_alpha": float(sel["ridge_alpha"]),
        "saved_k": int(saved["knn_k"]),
        "sweep_k": int(sel["knn_k"]),
        "saved_ridge_med": float(saved["ridge"]["median"]),
        "sweep_ridge_med": float(sel["ridge_median"]),
        "d_ridge": ridge_diff,
        "saved_knn_med": float(saved["knn"]["median"]),
        "sweep_knn_med": float(sel["knn_median"]),
        "d_knn": knn_diff,
        "pass": ok,
    }


def print_gate_header() -> None:
    print(
        "seed\tsource\tmethod\tsaved_d\tsweep_d\t"
        "saved_alpha\tsweep_alpha\tsaved_k\tsweep_k\t"
        "saved_ridge\tsweep_ridge\t|d_ridge|\t"
        "saved_knn\tsweep_knn\t|d_knn|\tpass",
        flush=True,
    )


def print_gate_row(g: dict[str, Any]) -> None:
    print(
        f"{g['seed']}\t{g['source']}\t{g['method']}\t"
        f"{g['saved_d']}\t{g['sweep_d']}\t"
        f"{g['saved_alpha']:.6g}\t{g['sweep_alpha']:.6g}\t"
        f"{g['saved_k']}\t{g['sweep_k']}\t"
        f"{g['saved_ridge_med']:.9f}\t{g['sweep_ridge_med']:.9f}\t"
        f"{g['d_ridge']:.3e}\t"
        f"{g['saved_knn_med']:.9f}\t{g['sweep_knn_med']:.9f}\t"
        f"{g['d_knn']:.3e}\t"
        f"{'PASS' if g['pass'] else 'FAIL'}",
        flush=True,
    )


def sweep_method(
    key: str,
    cfg: dict[str, Any],
    obs: dict[str, Any],
    methods_seed: int,
) -> list[dict[str, Any]]:
    if not bool(cfg["split"]["inner_cv_refit_representation"]):
        raise RuntimeError("inner_cv_refit_representation must be true")
    alphas = ridge_alpha_grid(cfg)
    ks = list(cfg["decoders"]["knn_k"])
    primary_d, _alpha, _knn_k, _cv, per_d_cv = _inner_cv_select(
        key, obs["X"], obs["y"], obs["train_ok"], obs["decode_times"],
        cfg, methods_seed, alphas, ks,
    )
    if primary_d is None:
        raise RuntimeError(f"{key}: expected a selected d")
    return score_d_curve(
        key, obs["X"], obs["y"], obs["train_ok"], obs["eval_mask"],
        cfg, methods_seed, per_d_cv, int(primary_d),
    )


def sweep_source(
    cfg: dict[str, Any],
    seed_index: int,
    spike_source: str,
    results_root: Path,
    gate_rows: list[dict[str, Any]],
) -> int:
    src_dir = results_root / f"seed_{seed_index}" / spike_source
    dest = src_dir / "d_sweep.json"
    if dest.is_file():
        existing = json.loads(dest.read_text())
        if existing.get("selection_rule") == D_SWEEP_SELECTION_RULE:
            print(f"# reuse {dest}", flush=True)
            for key in REDUCING_SWEEP:
                saved = json.loads((src_dir / f"{key}.json").read_text())
                rows = [r for r in existing["rows"] if r["method"] == key]
                g = gate_cell(seed_index, spike_source, key, rows, saved)
                print_gate_row(g)
                gate_rows.append(g)
                if not g["pass"]:
                    print("GATE_FAIL", g, flush=True)
                    return 1
            return 0

    summary = json.loads((src_dir / "source_summary.json").read_text())
    streams = summary["seed_streams"]
    obs = build_observation(
        cfg, results_root / f"seed_{seed_index}" / "sim", spike_source,
    )
    saved_eval = (summary.get("index_hashes") or {}).get("eval")
    if obs["index_hashes"]["eval"] != saved_eval:
        raise RuntimeError(
            f"seed {seed_index} {spike_source} eval hash "
            f"{obs['index_hashes']['eval']} != saved {saved_eval}"
        )
    methods_seed = int(streams["methods"])
    partial_path = src_dir / "d_sweep.partial.json"
    done = _load_partial(partial_path)
    all_rows: list[dict[str, Any]] = []
    for key in REDUCING_SWEEP:
        saved = json.loads((src_dir / f"{key}.json").read_text())
        if key in done:
            rows = done[key]
            print(f"# resume {key} seed {seed_index} {spike_source}", flush=True)
        else:
            print(f"# {key} seed {seed_index} {spike_source}", flush=True)
            t0 = time.perf_counter()
            rows = sweep_method(key, cfg, obs, methods_seed)
            print(
                f"  {key} {time.perf_counter() - t0:.1f}s "
                f"selected_d={next(r['d'] for r in rows if r['selected'])}",
                flush=True,
            )
            done[key] = rows
            _save_partial(partial_path, done)
        g = gate_cell(seed_index, spike_source, key, rows, saved)
        print_gate_row(g)
        gate_rows.append(g)
        if not g["pass"]:
            print("GATE_FAIL", g, flush=True)
            return 1
        all_rows.extend(rows)
    write_d_sweep_json(
        dest, cfg,
        seed_index=seed_index,
        spike_source=spike_source,
        eval_index_hash=obs["index_hashes"]["eval"],
        rows=all_rows,
    )
    if partial_path.is_file():
        partial_path.unlink()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--estimate-only", action="store_true")
    args = ap.parse_args()
    cfg = load_quadrant_n5_yaml()
    est = estimate_hours(args.results)
    print("ESTIMATE", json.dumps(est, indent=2), flush=True)
    if args.estimate_only:
        return 0
    print_gate_header()
    gate_rows: list[dict[str, Any]] = []
    for seed in parse_seeds(args.seeds):
        for src in ("sorted", "ground_truth"):
            print(f"# sweep seed {seed} {src}", flush=True)
            rc = sweep_source(cfg, seed, src, args.results, gate_rows)
            if rc != 0:
                print("STOP after first failing cell; no Stage 4.", flush=True)
                return rc
    n_pass = sum(1 for g in gate_rows if g["pass"])
    print(f"GATE_SUMMARY {n_pass}/{len(gate_rows)}", flush=True)
    if n_pass != 50 or len(gate_rows) != 50:
        print("GATE_FAIL expected 50/50", flush=True)
        return 1
    print("GATE_50 PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
