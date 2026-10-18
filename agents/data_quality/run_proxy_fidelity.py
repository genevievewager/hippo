#!/usr/bin/env python3
"""Proxy fidelity gate: default-θ proxy R vs full matched R (within bootstrap CI)."""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.derived_config import (
    build_matched_capture_yaml,
    lab_segment_match_spec,
)
from agents.data_quality.matching.session_discovery import eligibility_by_animal
from agents.data_quality.matching.yield_boost import boost_ratinabox_for_required
from agents.data_quality.phase3.divergence import divergence_for_metric
from agents.data_quality.run_metric_suite import lab_session_rows, sim_session_rows, _animal
from hippo_sim.config import SimConfig
from hippo_sim.pipeline import apply_trajectory_to_config
from run_simulation import generate_dataset


def _gen_proxy(session: str, *, duration_s: float, n_units: int, out: Path, seed: int) -> Path:
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    spec = lab_segment_match_spec(root / session, "A")
    req = dict(spec["required_counts"])
    tot = max(1, int(req.get("MEC", 0)) + int(req.get("SUB", 0)))
    scale = n_units / tot
    req = {k: max(1, int(round(v * scale))) if int(v) > 0 else 0 for k, v in req.items()}
    spec = dict(spec)
    spec["required_counts"] = req
    spec["session_duration_s"] = duration_s
    capture = build_matched_capture_yaml(spec, out / "cell_capture_proxy.yaml")
    (out / "match_spec.json").write_text(json.dumps(spec, indent=2, default=str) + "\n")
    sim_dir = out / "sim"
    sim_dir.mkdir()
    cfg = SimConfig(
        output_dir=sim_dir,
        seed=seed,
        session_duration_s=duration_s,
        behavior_dt=0.05,
        arena_size_cm=float(spec["arena_size_cm"]),
        thigmotaxis=0.8,
    )
    apply_trajectory_to_config(
        cfg,
        trajectory_config="configs/trajectories/lab_npx2_default.yaml",
        cell_capture_config=str(capture),
    )
    boost_ratinabox_for_required(cfg, req, safety=1.3)
    generate_dataset(cfg)
    (out / "pilot_report.json").write_text(
        json.dumps({"arena_size_cm": spec["arena_size_cm"], "required_counts": req}, indent=2) + "\n"
    )
    return sim_dir


def main() -> None:
    duration = float(os.environ.get("DQ_PROXY_DURATION_S", "300"))
    n_units = int(os.environ.get("DQ_PROXY_N_UNITS", "50"))
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    elig = eligibility_by_animal(root)
    sessions = []
    for a in elig["animals"]:
        es = a.get("eligible_sessions") or []
        if es:
            sessions.append(es[0]["session"] if isinstance(es[0], dict) else es[0])
    sessions = sessions[:6]
    out_root = assert_dq_path(DATA_QUALITY_ROOT / "calibration" / "proxy_fidelity")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    # lab: reuse metric_rows_full if present
    full_path = DATA_QUALITY_ROOT / "phase3" / "metric_rows_full.jsonl"
    lab_rows = []
    if full_path.is_file():
        with full_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("dataset") == "lab":
                    lab_rows.append(r)
    else:
        for s in sessions:
            lab_rows.extend(lab_session_rows(s)["rows"])

    t0 = time.perf_counter()
    for i, sess in enumerate(sessions):
        sim_dir = _gen_proxy(
            sess, duration_s=duration, n_units=n_units,
            out=out_root / sess, seed=20261007 + i,
        )
        part = sim_session_rows(
            str(sim_dir), dataset="sim_proxy", session=sess, seed_label=_animal(sess),
        )
        rows.extend(part["rows"])
        print(json.dumps({"proxy_done": sess, "n_rows": part["n_rows"], "wall_s": part["wall_s"]}), flush=True)

    # matched arm rows from full file
    matched_rows = []
    if full_path.is_file():
        with full_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("dataset") == "sim_matched":
                    matched_rows.append(r)

    df_proxy = pd.DataFrame(lab_rows + [{**r, "dataset": "sim_proxy"} for r in rows])
    # only keep metrics present in both
    df_full = pd.DataFrame(lab_rows + matched_rows)

    # compare on a shared core metric set
    core = sorted(
        set(df_proxy["metric"].unique()) & set(df_full["metric"].unique())
    )
    comparisons = []
    n_agree = 0
    for m in core:
        rp = divergence_for_metric(df_proxy, metric=m, sim_dataset="sim_proxy",
                                   log_rates=(m in {"firing_rate", "firing_rate_hz"}))
        rf = divergence_for_metric(df_full, metric=m, sim_dataset="sim_matched",
                                   log_rates=(m in {"firing_rate", "firing_rate_hz"}))
        def _f(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return float("nan")

        R_p, R_f = _f(rp.get("R")), _f(rf.get("R"))
        lo, hi = _f(rf.get("R_lo90")), _f(rf.get("R_hi90"))
        # agree if proxy R within full matched bootstrap interval
        agree = bool(np.isfinite(R_p) and np.isfinite(lo) and np.isfinite(hi) and lo <= R_p <= hi)
        n_agree += int(agree)
        comparisons.append({
            "metric": m,
            "R_proxy": R_p,
            "R_full_matched": R_f,
            "R_full_lo90": lo,
            "R_full_hi90": hi,
            "agree": agree,
        })

    report = {
        "sessions": sessions,
        "duration_s": duration,
        "n_units_target": n_units,
        "n_metrics": len(core),
        "n_agree": n_agree,
        "frac_agree": float(n_agree / max(len(core), 1)),
        "pass": bool(n_agree / max(len(core), 1) >= 0.8),  # ≥80% metrics agree
        "wall_s": time.perf_counter() - t0,
        "comparisons": sorted(comparisons, key=lambda d: abs((d["R_proxy"] or 0) - (d["R_full_matched"] or 0)), reverse=True),
        "rule": "proxy R within full-matched bootstrap 90% CI; pass if ≥80% metrics",
    }
    path = assert_dq_path(out_root / "FIDELITY.json")
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report[k] for k in report if k != "comparisons"}, indent=2))
    print(json.dumps({"top_disagreements": report["comparisons"][:10]}, indent=2, default=str))
    if not report["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
