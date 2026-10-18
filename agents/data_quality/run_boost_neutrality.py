#!/usr/bin/env python3
"""Boost neutrality: ±boost on one MEC-heavy and one SUB-heavy session (same seed).

Writes under outputs/data_quality/neutrality/ only — never overwrites matched/.
Compares per-unit firing rate, ISI CV, spatial info at matched counts.
R = W1(boost, noboost) / MAD(noboost); pass if all R ≲ 1.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.regions import add_mapped_lab_region
from agents.data_quality.matching.yield_boost import boost_ratinabox_for_required
from agents.data_quality.metrics.spatial import spatial_info_bits_per_spike, _rate_map
from agents.data_quality.metrics.units import firing_rate, isi_cv
from hippo_sim.config import SimConfig
from hippo_sim.pipeline import apply_trajectory_to_config
from run_simulation import generate_dataset


def _w1(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, float)); a = a[np.isfinite(a)]
    b = np.sort(np.asarray(b, float)); b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    qs = np.linspace(0.0, 1.0, 64)
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)) * 1.4826)


def _unit_metrics(sim_dir: Path, *, t0: float, t1: float, region: str, n_keep: int, seed: int):
    units = add_mapped_lab_region(pd.read_csv(sim_dir / "units.csv"))
    sp = pd.read_csv(sim_dir / "spikes_sorted.csv")
    beh = pd.read_csv(sim_dir / "behavior.csv")
    t = beh["time_s"].to_numpy(float)
    pos = np.column_stack([beh["x_cm"].to_numpy(float), beh["y_cm"].to_numpy(float)])
    m = (t >= t0) & (t < t1)
    t, pos = t[m], pos[m]
    pool = units[units["lab_region"].astype(str) == region]
    if len(pool) == 0 or n_keep <= 0:
        return {"n": 0, "rate": [], "isi_cv": [], "spatial_info": []}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n_keep, len(pool)), replace=False)
    sel = pool.iloc[idx]
    rates, cvs, infos = [], [], []
    xmin, ymin = pos.min(0); xmax, ymax = pos.max(0)
    edges_x = np.linspace(xmin, xmax, 21)
    edges_y = np.linspace(ymin, ymax, 21)
    for uid in sel["unit_id"].to_numpy():
        st = sp.loc[sp["unit_id"] == uid, "spike_time_s"].to_numpy(float)
        rates.append(firing_rate(st, t0, t1))
        cvs.append(isi_cv(st))
        st_w = st[(st >= t[0]) & (st < t[-1])] if t.size else st[:0]
        if st_w.size == 0 or t.size < 10:
            infos.append(float("nan"))
            continue
        ii = np.clip(np.searchsorted(t, st_w, side="right") - 1, 0, len(t) - 1)
        mask = np.zeros(len(t), dtype=bool)
        mask[ii] = True
        rate, occ, ok = _rate_map(pos, mask, edges_x, edges_y)
        infos.append(spatial_info_bits_per_spike(rate, occ, ok))
    return {
        "n": int(len(sel)),
        "rate": rates,
        "isi_cv": cvs,
        "spatial_info": infos,
        "yield": units["lab_region"].value_counts().to_dict(),
    }


def _generate(session: str, *, with_boost: bool, out: Path, duration_s: float, seed: int) -> dict[str, Any]:
    matched = DATA_QUALITY_ROOT / "matched" / f"{session}_A"
    spec = json.loads((matched / "match_spec.json").read_text())
    capture_path = matched / "cell_capture_matched.yaml"
    prov = json.loads((matched / "provenance.json").read_text())
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sim_dir = out / "sim"
    sim_dir.mkdir()
    cfg = SimConfig(
        output_dir=sim_dir,
        seed=int(seed),
        session_duration_s=float(duration_s),
        behavior_dt=0.05,
        arena_size_cm=float(spec["arena_size_cm"]),
        thigmotaxis=0.8,
    )
    apply_trajectory_to_config(
        cfg,
        trajectory_config="configs/trajectories/lab_npx2_default.yaml",
        cell_capture_config=str(capture_path),
    )
    boost_report = None
    if with_boost:
        n_regen = int(prov.get("n_regenerations") or 0)
        safety = 1.3 * (1.5 ** n_regen)
        boost_report = boost_ratinabox_for_required(
            cfg, spec["required_counts"], safety=safety,
        )
        # also apply FINAL params from provenance if present (exact freeze)
        final = prov.get("final_ratinabox_params") or {}
        if final:
            rp = dict(cfg.ratinabox_params)
            rp.update({k: int(v) for k, v in final.items()})
            cfg.ratinabox_params = rp
    t0 = time.perf_counter()
    summary = generate_dataset(cfg)
    wall = time.perf_counter() - t0
    return {
        "session": session,
        "with_boost": with_boost,
        "duration_s": duration_s,
        "seed": seed,
        "wall_s": wall,
        "summary_n_units": summary.get("n_units"),
        "boost_report": boost_report,
        "ratinabox_params": {k: int(v) for k, v in cfg.ratinabox_params.items()
                            if k.startswith("n_") or k == "rate_scale_hz"},
        "required_counts": spec["required_counts"],
        "sim_dir": str(sim_dir),
    }


def _compare(a: dict, b: dict, metric: str) -> dict[str, Any]:
    xa = np.asarray(a[metric], float)
    xb = np.asarray(b[metric], float)
    mad = _mad(xb)
    if not np.isfinite(mad) or mad < 1e-12:
        mad = max(float(np.std(xb[np.isfinite(xb)])), 1e-6) if np.isfinite(xb).any() else 1.0
    w1 = _w1(xa, xb)
    R = float(w1 / mad) if np.isfinite(w1) else float("nan")
    return {
        "metric": metric,
        "median_boost": float(np.nanmedian(xa)),
        "median_noboost": float(np.nanmedian(xb)),
        "median_ratio": float(np.nanmedian(xa) / max(np.nanmedian(xb), 1e-12)),
        "W1": w1,
        "MAD_noboost": mad,
        "R": R,
        "pass": bool(np.isfinite(R) and R <= 1.05),
    }


def run_one(session: str, heavy_region: str, *, duration_s: float = 300.0, seed: int = 20261005) -> dict[str, Any]:
    root = assert_dq_path(DATA_QUALITY_ROOT / "neutrality" / session)
    root.mkdir(parents=True, exist_ok=True)
    matched = DATA_QUALITY_ROOT / "matched" / f"{session}_A"
    spec = json.loads((matched / "match_spec.json").read_text())
    required = int((spec.get("required_counts") or {}).get(heavy_region, 0))

    gen_b = _generate(session, with_boost=True, out=root / "with_boost", duration_s=duration_s, seed=seed)
    gen_n = _generate(session, with_boost=False, out=root / "no_boost", duration_s=duration_s, seed=seed)

    # Probe available yields, then subsample both arms to the same matched count.
    probe_b = _unit_metrics(
        Path(gen_b["sim_dir"]), t0=0.0, t1=duration_s, region=heavy_region,
        n_keep=10_000, seed=seed + 1,
    )
    probe_n = _unit_metrics(
        Path(gen_n["sim_dir"]), t0=0.0, t1=duration_s, region=heavy_region,
        n_keep=10_000, seed=seed + 1,
    )
    n_cap = required if required > 0 else min(probe_b["n"], probe_n["n"])
    n_match = min(probe_b["n"], probe_n["n"], n_cap)
    if n_match <= 0:
        n_match = min(probe_b["n"], probe_n["n"])
    met_b = _unit_metrics(
        Path(gen_b["sim_dir"]), t0=0.0, t1=duration_s, region=heavy_region,
        n_keep=n_match, seed=seed + 1,
    )
    met_n = _unit_metrics(
        Path(gen_n["sim_dir"]), t0=0.0, t1=duration_s, region=heavy_region,
        n_keep=n_match, seed=seed + 1,
    )

    comps = [_compare(met_b, met_n, m) for m in ("rate", "isi_cv", "spatial_info")]
    out = {
        "session": session,
        "heavy_region": heavy_region,
        "duration_s": duration_s,
        "seed": seed,
        "n_matched": n_match,
        "required": required,
        "gen_with_boost": {k: v for k, v in gen_b.items() if k != "boost_report"} | {
            "boost_report": gen_b.get("boost_report"),
            "yield": met_b.get("yield"),
        },
        "gen_no_boost": {k: v for k, v in gen_n.items() if k != "boost_report"} | {
            "yield": met_n.get("yield"),
        },
        "comparisons": comps,
        "all_pass": all(c["pass"] for c in comps),
        "max_R": float(max((c["R"] for c in comps if np.isfinite(c["R"])), default=float("nan"))),
    }
    (root / "neutrality_report.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def main() -> None:
    pick = DATA_QUALITY_ROOT / "_phase0" / "neutrality_sessions.json"
    if pick.is_file():
        info = json.loads(pick.read_text())
        mec, sub = info["mec_heavy"], info["sub_heavy"]
    else:
        mec, sub = "1120_12052024_2rooms", "1140_20082024_2rooms"
    duration = float(os.environ.get("DQ_NEUTRALITY_DURATION_S", "300"))
    seed = int(os.environ.get("DQ_NEUTRALITY_SEED", "20261005"))
    results = [
        run_one(mec, "MEC", duration_s=duration, seed=seed),
        run_one(sub, "SUB", duration_s=duration, seed=seed),
    ]
    summary = {
        "duration_s": duration,
        "seed": seed,
        "sessions": results,
        "all_pass": all(r["all_pass"] for r in results),
        "max_R": float(max(r["max_R"] for r in results)),
        "rule": "R = W1(boost,noboost)/MAD(noboost) at matched unit counts; pass if R<=1.05",
    }
    out = assert_dq_path(DATA_QUALITY_ROOT / "neutrality" / "SUMMARY.json")
    out.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps(summary, indent=2, default=str))
    if not summary["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
