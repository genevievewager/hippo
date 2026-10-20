#!/usr/bin/env python3
"""Build n_curve_ladder.csv: N ∈ {10, 20, 40, 80, 160, max} with 20 subsamples.

Writes a NEW table only — does not overwrite n_curve.csv.
Frozen pipeline otherwise (information / decode R² curve only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.report.run_behavior_info import (  # noqa: E402
    N_SUBSAMPLES,
    _sqrt_zscore_train,
    causal_train_test_split,
    decode_target,
    iter_jobs,
    load_lab,
    load_sim,
    GAP_S,
    TRAIN_FRAC,
)

FIXED_NS = (10, 20, 40, 80, 160)
VARS = (
    ("position", "continuous_2d", "xy", None),
    ("movement_direction", "direction", "move_dir", "moving"),
    ("speed", "continuous_1d", "speed", None),
)


def _ncurve_ladder(pack) -> list[dict]:
    X_raw, beh, times = pack["X"], pack["beh"], pack["times"]
    n_u = X_raw.shape[1]
    if n_u < 5 or len(times) < 120:
        return []
    train, test = causal_train_test_split(times, TRAIN_FRAC, gap_s=GAP_S)
    moving = beh["speed"] > 2.0
    rng = np.random.default_rng(abs(hash(pack["session"])) % (2**32))
    rows = []
    ladder = list(FIXED_NS) + [n_u]
    for N in ladder:
        if N > n_u:
            continue
        n_rep = N_SUBSAMPLES if N < n_u else 1
        for var, kind, ykey, mask_key in VARS:
            y = beh["xy"] if ykey == "xy" else beh[ykey]
            smask = moving if mask_key == "moving" else None
            vals = []
            for _ in range(n_rep):
                idx = rng.choice(n_u, size=int(N), replace=False)
                Xnz, _ = _sqrt_zscore_train(X_raw[:, idx], train)
                sc, _, _, _ = decode_target(Xnz, train, test, y, kind, score_mask=smask)
                if sc is not None:
                    vals.append(sc["r2"])
            if vals:
                rows.append(dict(
                    dataset=pack["dataset"], session=pack["session"], animal=pack["animal"],
                    variable=var, N=int(N), value=float(np.median(vals)),
                    value_q25=float(np.percentile(vals, 25)),
                    value_q75=float(np.percentile(vals, 75)),
                    N_role="max" if N == n_u else "ladder",
                ))
    return rows


def _worker(payload):
    kind, session, sim_dir, arena = payload
    try:
        if kind == "lab":
            pack = load_lab(session)
        else:
            pack = load_sim(sim_dir, kind, session, arena)
        rows = _ncurve_ladder(pack)
        return {"dataset": kind, "session": session, "status": "ok",
                "n_rows": len(rows), "n_curve": rows}
    except Exception as e:
        return {"dataset": kind, "session": session, "status": "error",
                "error": f"{type(e).__name__}: {e}", "n_curve": [], "n_rows": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pilot", type=int, default=0)
    args = ap.parse_args()

    beh = DATA_QUALITY_ROOT / "reports" / "report4a_v5" / "behavior_info"
    out = assert_dq_path(Path(args.out) if args.out else beh)
    out.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(args.pilot or None))
    print(json.dumps({
        "n_jobs": len(jobs), "fixed_N": list(FIXED_NS), "plus": "max",
        "n_subsamples": N_SUBSAMPLES, "workers": args.workers,
        "dest": str(out / "n_curve_ladder.csv"),
    }), flush=True)

    t0 = time.perf_counter()
    results = []
    if args.workers <= 1:
        for i, job in enumerate(jobs, 1):
            r = _worker(job)
            results.append(r)
            print(json.dumps({"done": i, "total": len(jobs), "session": job[1],
                              "dataset": job[0], "status": r["status"], "n_rows": r.get("n_rows"),
                              "elapsed_s": round(time.perf_counter() - t0, 1)}), flush=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # fork keeps parent's sys.path / imports (spawn re-imports and breaks -m launches)
        ctx = mp.get_context("fork")
        # ensure children see repo even if env was stripped
        os.environ["PYTHONPATH"] = f"{_REPO}:{_REPO / '.pip_local'}" + (
            f":{os.environ['PYTHONPATH']}" if os.environ.get("PYTHONPATH") else ""
        )
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, j): j for j in jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                j = futs[fut]
                r = fut.result()
                results.append(r)
                print(json.dumps({"done": i, "total": len(jobs), "session": j[1],
                                  "dataset": j[0], "status": r["status"], "n_rows": r.get("n_rows"),
                                  "elapsed_s": round(time.perf_counter() - t0, 1)}), flush=True)

    rows = []
    n_err = 0
    for r in results:
        if r.get("status") != "ok":
            n_err += 1
        rows.extend(r.get("n_curve") or [])
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["dataset", "session", "variable", "N"]).reset_index(drop=True)
    dest = out / "n_curve_ladder.csv"
    df.to_csv(dest, index=False)
    note = {
        "wrote": str(dest),
        "n_rows": int(len(df)),
        "n_jobs": len(jobs),
        "n_errors": n_err,
        "fixed_N": list(FIXED_NS),
        "n_subsamples": N_SUBSAMPLES,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "n_curve_csv_untouched": True,
        "backup_pre_amendment": str(beh / "n_curve_pre_ladder80_160.csv"),
    }
    (out / "n_curve_ladder_SUMMARY.json").write_text(json.dumps(note, indent=2) + "\n")
    print(json.dumps(note, indent=2), flush=True)
    if n_err or not len(df):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
