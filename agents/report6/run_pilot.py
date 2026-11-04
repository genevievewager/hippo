#!/usr/bin/env python3
"""Report 6 pilot orchestrator. Stages: m0m2 | null_a | null_b | split | boot | plant | all_light."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ensure repo root on path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.report6.io_cohort import build_alias_map, iter_sessions, public_cohort_table
from agents.report6.paths import OUT_ROOT
from agents.report6.pipeline import (
    null_a_time_shifts,
    null_b_surrogate,
    run_all_m0_m2,
    run_m0_m2_session,
    save_json,
)
from agents.report6.plant_recover import plant_and_recover
from agents.report6.refit_stability import bootstrap_stability
from agents.report6.split_pop import split_population_kappa


def _lds_cut_aliases(n: int = 3) -> list[str]:
    """Smallest n_units real sessions for LDS refit cuts."""
    rows = [r for r in public_cohort_table() if r["arm"] == "real"]
    rows = sorted(rows, key=lambda r: (r["n_units"], r["alias"]))
    return [r["alias"] for r in rows[:n]]


def _null_a_job(alias: str, method: str) -> dict:
    sess = next(x for x in iter_sessions() if x.alias == alias)
    print(f"[null_a] {alias} {method}", flush=True)
    r = null_a_time_shifts(sess, method=method, n_shifts=20)
    save_json(r, OUT_ROOT / "null_a" / f"{alias}_{method}.json")
    return {k: v for k, v in r.items() if k != "shifts"}


def stage_null_a(methods: tuple[str, ...] = ("lds", "pca"), arms: tuple[str, ...] = ("real", "sim")) -> None:
    """Time-shift null. Parallelise session×method with ≤4 workers."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = []
    for s in iter_sessions():
        if s.arm not in arms:
            continue
        for method in methods:
            jobs.append((s.alias, method))

    out = []
    n_workers = min(4, max(1, len(jobs)))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = {ex.submit(_null_a_job, a, m): (a, m) for a, m in jobs}
        for fut in as_completed(futs):
            out.append(fut.result())
            print(f"  done {futs[fut]}", flush=True)
    save_json(out, OUT_ROOT / "null_a" / "summary.json")


def stage_null_b() -> None:
    build_alias_map(force=True)
    lds_ok = set(_lds_cut_aliases(3))
    out = []
    jobs: list[tuple[str, str]] = []
    for s in iter_sessions(arm="real"):
        jobs.append((s.alias, "pca"))
        if s.alias in lds_ok:
            jobs.append((s.alias, "lds"))
    for s in iter_sessions(arm="sim"):
        jobs.append((s.alias, "pca"))
    for alias, method in jobs:
        sess = next(x for x in iter_sessions() if x.alias == alias)
        print(f"[null_b] {alias} {method} n_units={sess.n_units}", flush=True)
        t0 = time.time()
        try:
            r = null_b_surrogate(sess, method=method, n_surr=5)
        except Exception as e:
            r = {"alias": alias, "method": method, "error": repr(e)}
            print(f"  ERROR {e}", flush=True)
        r["wall_s"] = time.time() - t0
        save_json(r, OUT_ROOT / "null_b" / f"{alias}_{method}.json")
        out.append(r)
        print(f"  done wall_s={r['wall_s']:.1f} n_pass={r.get('n_pass')}", flush=True)
    save_json(out, OUT_ROOT / "null_b" / "summary.json")


def stage_split() -> None:
    lds_ok = set(_lds_cut_aliases(3))
    out = []
    for s in iter_sessions(arm="real"):
        for method in ("pca", "lds"):
            if method == "lds" and s.alias not in lds_ok:
                continue
            print(f"[split] {s.alias} {method}", flush=True)
            t0 = time.time()
            r = split_population_kappa(s, method=method)
            r["wall_s"] = time.time() - t0
            save_json(r, OUT_ROOT / "split" / f"{s.alias}_{method}.json")
            out.append(r)
    save_json(out, OUT_ROOT / "split" / "summary.json")


def stage_boot() -> None:
    lds_ok = set(_lds_cut_aliases(3))
    out = []
    for s in iter_sessions(arm="real"):
        print(f"[boot] {s.alias} pca", flush=True)
        r = bootstrap_stability(s, method="pca", B=10)
        save_json(r, OUT_ROOT / "boot" / f"{s.alias}_pca.json")
        out.append(r)
        if s.alias in lds_ok:
            print(f"[boot] {s.alias} lds", flush=True)
            r = bootstrap_stability(s, method="lds", B=5)
            save_json(r, OUT_ROOT / "boot" / f"{s.alias}_lds.json")
            out.append(r)
    save_json(out, OUT_ROOT / "boot" / "summary.json")


def stage_plant() -> None:
    sims = [s for s in iter_sessions(arm="sim")]
    if not sims:
        print("no sim sessions", flush=True)
        return
    s = sims[0]
    out = []
    for method in ("pca", "lds"):
        print(f"[plant] {s.alias} {method}", flush=True)
        t0 = time.time()
        r = plant_and_recover(s, method=method)
        r["wall_s"] = time.time() - t0
        save_json(r, OUT_ROOT / "plant" / f"{s.alias}_{method}.json")
        out.append(r)
        print(f"  hit={r.get('hit_rate')} fa={r.get('fa_per_min')} pass={r.get('pass')}", flush=True)
    save_json(out, OUT_ROOT / "plant" / "summary.json")


def stage_m4_posthoc() -> None:
    """Post-hoc position/speed description where y exists; never defines targets."""
    import numpy as np

    from agents.report6.detector import fit_detector
    from agents.report6.paths import DT_S
    from agents.report6.target_rule import find_targets
    from agents.report6.visits import binary_runs_from_labels

    out = []
    for s in iter_sessions(arm="real"):
        if s.y is None:
            continue
        Z = s.Z_lds[:, : min(3, s.Z_lds.shape[1])]
        path = OUT_ROOT / "m0_m2" / f"{s.alias}.json"
        if not path.is_file():
            continue
        blob = json.loads(path.read_text())
        te = (blob.get("methods") or {}).get("lds", {}).get("targets_eval") or []
        if not te:
            out.append({"alias": s.alias, "status": "no_target"})
            continue
        y = s.y
        dy = np.linalg.norm(np.diff(y, axis=0, prepend=y[:1]), axis=1) / 0.05
        tgt = find_targets(Z, s.train_ok, times_s=s.decode_times, dt_s=DT_S)
        if not tgt.get("targets"):
            continue
        lab = int(tgt["targets"][0]["label"])
        model = fit_detector(Z, binary_runs_from_labels(tgt["labels"], lab), s.train_ok)
        binary = model.hysteresis(model.predict_proba(Z))
        ev = s.retained & s.valid
        speed_in = float(dy[ev & binary].mean()) if np.any(ev & binary) else None
        speed_out = float(dy[ev & ~binary].mean()) if np.any(ev & ~binary) else None
        out.append({
            "alias": s.alias,
            "mean_speed_in": speed_in,
            "mean_speed_out": speed_out,
            "frac_in": float(binary[ev].mean()) if ev.any() else None,
        })
    save_json(out, OUT_ROOT / "m4_posthoc" / "summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=[
            "alias", "m0m2", "null_a", "null_b", "split", "boot", "plant",
            "m4", "all_light", "cohort",
        ],
    )
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    build_alias_map()
    if args.stage == "alias" or args.stage == "cohort":
        save_json(public_cohort_table(), OUT_ROOT / "cohort_public.json")
        print(json.dumps(public_cohort_table(), indent=2))
        return
    if args.stage == "m0m2":
        run_all_m0_m2()
        return
    if args.stage == "null_a":
        stage_null_a()
        return
    if args.stage == "null_b":
        stage_null_b()
        return
    if args.stage == "split":
        stage_split()
        return
    if args.stage == "boot":
        stage_boot()
        return
    if args.stage == "plant":
        stage_plant()
        return
    if args.stage == "m4":
        stage_m4_posthoc()
        return
    if args.stage == "all_light":
        # no LDS refits
        run_all_m0_m2()
        stage_null_a()
        stage_plant()
        return


if __name__ == "__main__":
    main()
