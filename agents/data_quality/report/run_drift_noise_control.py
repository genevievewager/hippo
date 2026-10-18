#!/usr/bin/env python3
"""Drift vs noise: interleaved splits + 6-block time-separation curve.

Exact spike→frame assignment (``spatial._spike_positions``): spikes on
excluded frames (speed filter, split arm, etc.) are dropped.

Primary drift test: split train into 6 equal-duration blocks; median place-map
r vs block separation (1–5); day-bootstrap 90% CI; OLS slope. Drift = lab
declines with separation and sim is flat; else lab maps are less reliable.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
_R2V2 = Path("/home/gmwager/projects/hippo-r2v2")
for p in (str(_REPO), str(_R2V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agents.data_quality.metrics.boundary import bin_centers_reachable  # noqa: E402
from agents.data_quality.metrics.spatial import (  # noqa: E402
    MIN_OCC_S,
    SPEED_MIN_CM_S,
    _build_map,
    _spike_positions,
    unit_eligible,
)
from agents.data_quality.phase3.divergence import _recording_day  # noqa: E402
from agents.data_quality.report.run_behavior_info import (  # noqa: E402
    W_S,
    iter_jobs,
    load_lab,
    load_sim,
)
from agents.data_quality.report.run_encoding_diagnostics import (  # noqa: E402
    _map_edges,
    end_block_split,
)
from agents.data_quality.report.run_perunit_encoding_ve import (  # noqa: E402
    _load_spikes_aligned,
    place_tuned_flags,
)

warnings.filterwarnings("ignore")

DQ = _REPO / "outputs/data_quality"
OUT = DQ / "reports/report4a_v3/behavior_info"
N_BOOT = 200
N_BLOCKS = 6


def _map_corr(r1, ok1, r2, ok2, min_bins=10):
    m = ok1 & ok2
    if m.sum() < min_bins:
        return float("nan")
    a, b = r1[m], r2[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _occ_coverage(ok):
    return float(ok.mean()) if ok.size else float("nan")


def _count_spikes_keep(st, t_frames, p_frames, keep):
    """Spike count via exact frame-interval assignment (matches map builder)."""
    return int(len(_spike_positions(st, t_frames, p_frames, keep=keep)))


def unit_split_stats(st, t_frames, p_frames, keep_a, keep_b, edges_x, edges_y, reachable, occ_thresh, dt):
    kept = keep_a | keep_b
    kept_t = t_frames[kept]
    if kept_t.size < 2:
        return None
    t_lo, t_hi = float(kept_t[0]), float(kept_t[-1])
    if not unit_eligible(st, t_lo, t_hi):
        return None
    r1, _, ok1 = _build_map(
        st, t_frames, p_frames, edges_x, edges_y, occ_thresh, dt, reachable, None, keep=keep_a,
    )
    r2, _, ok2 = _build_map(
        st, t_frames, p_frames, edges_x, edges_y, occ_thresh, dt, reachable, None, keep=keep_b,
    )
    return {
        "r": _map_corr(r1, ok1, r2, ok2),
        "n_spikes_a": _count_spikes_keep(st, t_frames, p_frames, keep_a),
        "n_spikes_b": _count_spikes_keep(st, t_frames, p_frames, keep_b),
        "occ_cov_a": _occ_coverage(ok1),
        "occ_cov_b": _occ_coverage(ok2),
        "n_bins_joint": int((ok1 & ok2).sum()),
    }


def _block_keeps(t_frames, parent_keep, n_blocks=N_BLOCKS):
    """Equal-duration blocks over kept span; each is a keep mask on parent."""
    kept_idx = np.flatnonzero(parent_keep)
    if kept_idx.size < n_blocks * 10:
        return [np.zeros(len(t_frames), dtype=bool) for _ in range(n_blocks)]
    t_kept = t_frames[kept_idx]
    t0, t1 = float(t_kept[0]), float(t_kept[-1])
    edges = np.linspace(t0, t1, n_blocks + 1)
    out = []
    for i in range(n_blocks):
        lo, hi = edges[i], edges[i + 1]
        if i == n_blocks - 1:
            in_blk = (t_frames >= lo) & (t_frames <= hi + 1e-12)
        else:
            in_blk = (t_frames >= lo) & (t_frames < hi)
        out.append(parent_keep & in_blk)
    return out


def _ols_slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan")
    x, y = x[m], y[m]
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den < 1e-18:
        return float("nan")
    return float(np.sum((x - xm) * (y - ym)) / den)


def run_session(kind, session, sim_dir, arena):
    if kind == "lab":
        pack = load_lab(session)
    else:
        pack = load_sim(sim_dir, kind, session, arena)
    beh = pack["beh"]
    times = np.asarray(pack["times"], float)
    xy = np.asarray(beh["xy"], float)
    speed = np.asarray(beh["speed"], float)
    n_u = pack["X"].shape[1]
    spikes = _load_spikes_aligned(kind, session, sim_dir, pack["regions"], n_u)
    if spikes is None:
        return {"status": "error", "session": session, "dataset": kind, "error": "spike align"}

    place, _ = place_tuned_flags(
        spikes, times, xy, speed, beh.get("poly"), kind, session,
    )
    train, _ = end_block_split(times)
    finite = np.isfinite(xy).all(1) & np.isfinite(speed)
    parent_keep = train & finite & (speed > SPEED_MIN_CM_S)
    if parent_keep.sum() < 80:
        return {"status": "error", "session": session, "dataset": kind, "error": "short train"}

    t_frames, p_frames = times, xy
    p_tr = p_frames[parent_keep]
    t_tr = t_frames[parent_keep]
    edges_x, edges_y = _map_edges(p_tr)
    dt = float(np.median(np.diff(t_tr))) if t_tr.size > 1 else W_S
    occ_thresh = max(1.0, MIN_OCC_S / max(dt, 1e-9))
    reachable = None
    if beh.get("poly") is not None:
        try:
            reachable = bin_centers_reachable(edges_x, edges_y, beh["poly"])
        except Exception:
            reachable = None

    kept_idx = np.flatnonzero(parent_keep)
    mid = kept_idx.size // 2
    keep_first = np.zeros(len(times), dtype=bool)
    keep_last = np.zeros(len(times), dtype=bool)
    keep_first[kept_idx[:mid]] = True
    keep_last[kept_idx[mid:]] = True

    t0_keep = float(t_tr[0])
    minute = np.floor((t_frames - t0_keep) / 60.0).astype(int)
    keep_odd_min = parent_keep & ((minute % 2) == 1)
    keep_even_min = parent_keep & ((minute % 2) == 0)
    blk2 = np.floor((t_frames - t0_keep) / 2.0).astype(int)
    keep_odd_2s = parent_keep & ((blk2 % 2) == 1)
    keep_even_2s = parent_keep & ((blk2 % 2) == 0)
    block_keeps = _block_keeps(t_frames, parent_keep, N_BLOCKS)

    unit_rows = []
    sep_rows = []
    for j in np.flatnonzero(place):
        st = spikes[int(j)]
        fl = unit_split_stats(
            st, t_frames, p_frames, keep_first, keep_last,
            edges_x, edges_y, reachable, occ_thresh, dt,
        )
        oe_min = unit_split_stats(
            st, t_frames, p_frames, keep_odd_min, keep_even_min,
            edges_x, edges_y, reachable, occ_thresh, dt,
        )
        oe_2s = unit_split_stats(
            st, t_frames, p_frames, keep_odd_2s, keep_even_2s,
            edges_x, edges_y, reachable, occ_thresh, dt,
        )
        if fl is None and oe_2s is None and oe_min is None:
            continue
        unit_rows.append({
            "dataset": kind, "session": session, "animal": pack["animal"],
            "recording_day": _recording_day(session),
            "unit": int(j),
            "r_first_last": fl["r"] if fl else float("nan"),
            "r_odd_even_min": oe_min["r"] if oe_min else float("nan"),
            "r_odd_even_2s": oe_2s["r"] if oe_2s else float("nan"),
            "r_odd_even": oe_2s["r"] if oe_2s else float("nan"),
            "n_spikes_first": fl["n_spikes_a"] if fl else 0,
            "n_spikes_last": fl["n_spikes_b"] if fl else 0,
            "n_spikes_odd": oe_2s["n_spikes_a"] if oe_2s else 0,
            "n_spikes_even": oe_2s["n_spikes_b"] if oe_2s else 0,
            "n_spikes_odd_min": oe_min["n_spikes_a"] if oe_min else 0,
            "n_spikes_even_min": oe_min["n_spikes_b"] if oe_min else 0,
            "occ_cov_first": fl["occ_cov_a"] if fl else float("nan"),
            "occ_cov_last": fl["occ_cov_b"] if fl else float("nan"),
            "occ_cov_odd": oe_2s["occ_cov_a"] if oe_2s else float("nan"),
            "occ_cov_even": oe_2s["occ_cov_b"] if oe_2s else float("nan"),
            "occ_cov_odd_min": oe_min["occ_cov_a"] if oe_min else float("nan"),
            "occ_cov_even_min": oe_min["occ_cov_b"] if oe_min else float("nan"),
            "n_bins_joint_fl": fl["n_bins_joint"] if fl else float("nan"),
            "n_bins_joint_oe": oe_2s["n_bins_joint"] if oe_2s else float("nan"),
            "n_bins_joint_oe_min": oe_min["n_bins_joint"] if oe_min else float("nan"),
        })

        # 6-block maps for this unit
        maps = []
        for bi, bk in enumerate(block_keeps):
            if bk.sum() < 20:
                maps.append(None)
                continue
            t_b = t_frames[bk]
            if not unit_eligible(st, float(t_b[0]), float(t_b[-1])):
                maps.append(None)
                continue
            rate, _, ok = _build_map(
                st, t_frames, p_frames, edges_x, edges_y, occ_thresh, dt,
                reachable, None, keep=bk,
            )
            maps.append((
                rate, ok,
                _count_spikes_keep(st, t_frames, p_frames, bk),
                _occ_coverage(ok),
            ))
        for i in range(N_BLOCKS):
            for k in range(i + 1, N_BLOCKS):
                if maps[i] is None or maps[k] is None:
                    continue
                r1, ok1, nsp_i, occ_i = maps[i]
                r2, ok2, nsp_k, occ_k = maps[k]
                r = _map_corr(r1, ok1, r2, ok2)
                if not np.isfinite(r):
                    continue
                sep_rows.append({
                    "dataset": kind, "session": session, "animal": pack["animal"],
                    "recording_day": _recording_day(session),
                    "unit": int(j),
                    "block_i": i, "block_k": k,
                    "sep": k - i,
                    "r": r,
                    "n_bins_joint": int((ok1 & ok2).sum()),
                    "n_spikes_i": nsp_i, "n_spikes_k": nsp_k,
                    "occ_cov_i": occ_i, "occ_cov_k": occ_k,
                })

    day = _recording_day(session)
    if not unit_rows:
        sess = {
            "dataset": kind, "session": session, "animal": pack["animal"],
            "recording_day": day, "n_place": int(place.sum()),
            "r_first_last_med": float("nan"), "r_odd_even_med": float("nan"),
            "r_odd_even_min_med": float("nan"),
        }
        return {
            "status": "ok", "session_row": sess, "unit_rows": [],
            "sep_rows": [], "sep_sess_rows": [],
        }

    udf = pd.DataFrame(unit_rows)
    sess = {
        "dataset": kind, "session": session, "animal": pack["animal"],
        "recording_day": day,
        "n_place": int(place.sum()),
        "r_first_last_med": float(udf["r_first_last"].median()),
        "r_odd_even_med": float(udf["r_odd_even_2s"].median()),
        "r_odd_even_min_med": float(udf["r_odd_even_min"].median()),
        "delta_oe_minus_fl": float((udf["r_odd_even_2s"] - udf["r_first_last"]).median()),
        "delta_oe_min_minus_fl": float((udf["r_odd_even_min"] - udf["r_first_last"]).median()),
        "n_spikes_odd_min_med": float(np.nanmedian(np.r_[
            udf["n_spikes_odd_min"].to_numpy(float), udf["n_spikes_even_min"].to_numpy(float)
        ])),
        "n_spikes_oe2s_med": float(np.nanmedian(np.r_[
            udf["n_spikes_odd"].to_numpy(float), udf["n_spikes_even"].to_numpy(float)
        ])),
        "occ_cov_odd_min_med": float(np.nanmedian(np.r_[
            udf["occ_cov_odd_min"].to_numpy(float), udf["occ_cov_even_min"].to_numpy(float)
        ])),
        "occ_cov_oe2s_med": float(np.nanmedian(np.r_[
            udf["occ_cov_odd"].to_numpy(float), udf["occ_cov_even"].to_numpy(float)
        ])),
        "n_bins_joint_oe_min_med": float(udf["n_bins_joint_oe_min"].median()),
        "n_bins_joint_oe2s_med": float(udf["n_bins_joint_oe"].median()),
        "n_bins_joint_fl_med": float(udf["n_bins_joint_fl"].median()),
        "n_spikes_per_half_med": float(np.nanmedian(np.r_[
            udf["n_spikes_first"].to_numpy(float), udf["n_spikes_last"].to_numpy(float)
        ])),
        "n_spikes_per_oe_med": float(np.nanmedian(np.r_[
            udf["n_spikes_odd"].to_numpy(float), udf["n_spikes_even"].to_numpy(float)
        ])),
        "occ_cov_per_half_med": float(np.nanmedian(np.r_[
            udf["occ_cov_first"].to_numpy(float), udf["occ_cov_last"].to_numpy(float)
        ])),
        "occ_cov_per_oe_med": float(np.nanmedian(np.r_[
            udf["occ_cov_odd"].to_numpy(float), udf["occ_cov_even"].to_numpy(float)
        ])),
    }

    # per-session median r at each separation
    sep_sess_rows = []
    if sep_rows:
        sdf = pd.DataFrame(sep_rows)
        for sep, g in sdf.groupby("sep"):
            sep_sess_rows.append({
                "dataset": kind, "session": session, "animal": pack["animal"],
                "recording_day": day,
                "sep": int(sep),
                "r_med": float(g["r"].median()),
                "n_pairs": int(len(g)),
                "n_bins_joint_med": float(g["n_bins_joint"].median()),
            })

    return {
        "status": "ok",
        "session_row": sess,
        "unit_rows": unit_rows,
        "sep_rows": sep_rows,
        "sep_sess_rows": sep_sess_rows,
    }


def _worker(payload):
    kind, session, sim_dir, arena = payload
    try:
        return run_session(kind, session, sim_dir, arena)
    except Exception as e:
        return {
            "status": "error", "dataset": kind, "session": session,
            "error": f"{type(e).__name__}: {e}",
            "session_row": None, "unit_rows": [], "sep_rows": [], "sep_sess_rows": [],
        }


def day_bootstrap_ci(sess_df, col, n_boot=N_BOOT, rng=None):
    rng = rng or np.random.default_rng(0)
    days = sorted(sess_df["recording_day"].unique())
    if len(days) < 2:
        med = float(sess_df[col].median())
        return med, med, med
    boots = []
    for _ in range(n_boot):
        draw = rng.choice(days, size=len(days), replace=True)
        sub = sess_df[sess_df["recording_day"].isin(draw)]
        boots.append(float(sub[col].median()))
    a = np.asarray(boots, float)
    return float(sess_df[col].median()), float(np.nanquantile(a, 0.05)), float(np.nanquantile(a, 0.95))


def _sep_curve_summary(sep_sess_df, rng):
    """Median r vs sep with day-bootstrap CI; OLS slope on session medians."""
    out = {"by_sep": {}, "slope": float("nan"), "slope_lo90": float("nan"), "slope_hi90": float("nan")}
    if sep_sess_df is None or sep_sess_df.empty:
        return out
    for sep in sorted(sep_sess_df["sep"].unique()):
        sub = sep_sess_df[sep_sess_df["sep"] == sep]
        med, lo, hi = day_bootstrap_ci(sub, "r_med", rng=rng)
        out["by_sep"][str(int(sep))] = {
            "median": med, "lo90": lo, "hi90": hi, "n_sessions": int(len(sub)),
        }
    # session-level slopes then day-bootstrap the median slope
    slopes = []
    for sess, g in sep_sess_df.groupby("session"):
        slopes.append({
            "session": sess,
            "recording_day": g["recording_day"].iloc[0],
            "slope": _ols_slope(g["sep"], g["r_med"]),
        })
    sdf = pd.DataFrame(slopes)
    if sdf.empty:
        return out
    med, lo, hi = day_bootstrap_ci(sdf, "slope", rng=rng)
    out["slope"] = med
    out["slope_lo90"] = lo
    out["slope_hi90"] = hi
    out["n_sessions_slope"] = int(len(sdf))
    return out


def summarize(sess_df, unit_df, sep_sess_df):
    rng = np.random.default_rng(0)
    out = {"by_dataset": {}, "sep_curve": {}, "verdict": None, "bugfix": {
        "issue": (
            "_spike_positions used [t0,t1) on subsetted frames, assigning excluded-frame "
            "spikes (speed filter / split) to the last kept pre-gap frame"
        ),
        "fix": (
            "assign each spike to its parent frame interval; drop if keep[i] is False"
        ),
        "note": "lab still ~29% moving vs sim ~7% — speed-filter gaps hit lab harder",
    }}
    for ds in ("lab", "sim_matched"):
        s = sess_df[sess_df.dataset == ds]
        u = unit_df[unit_df.dataset == ds]
        if s.empty:
            continue
        fl_med, fl_lo, fl_hi = day_bootstrap_ci(s, "r_first_last_med", rng=rng)
        oe_med, oe_lo, oe_hi = day_bootstrap_ci(s, "r_odd_even_med", rng=rng)
        oem_med, oem_lo, oem_hi = day_bootstrap_ci(s, "r_odd_even_min_med", rng=rng)
        d_med, d_lo, d_hi = day_bootstrap_ci(s, "delta_oe_minus_fl", rng=rng)
        out["by_dataset"][ds] = {
            "n_sessions": int(len(s)),
            "n_place_units": int(len(u)),
            "r_first_last": {"median": fl_med, "lo90": fl_lo, "hi90": fl_hi},
            "r_odd_even_2s": {"median": oe_med, "lo90": oe_lo, "hi90": oe_hi},
            "r_odd_even_min": {"median": oem_med, "lo90": oem_lo, "hi90": oem_hi},
            "delta_oe2s_minus_fl": {"median": d_med, "lo90": d_lo, "hi90": d_hi},
            "minute_split_diagnostics": {
                "n_spikes_per_arm_med": float(s["n_spikes_odd_min_med"].median()),
                "occ_cov_per_arm_med": float(s["occ_cov_odd_min_med"].median()),
                "n_bins_joint_med": float(s["n_bins_joint_oe_min_med"].median()),
            },
            "oe2s_diagnostics": {
                "n_spikes_per_arm_med": float(s["n_spikes_oe2s_med"].median()),
                "occ_cov_per_arm_med": float(s["occ_cov_oe2s_med"].median()),
                "n_bins_joint_med": float(s["n_bins_joint_oe2s_med"].median()),
            },
            "fl_diagnostics": {
                "n_spikes_per_half_med": float(s["n_spikes_per_half_med"].median()),
                "occ_cov_per_half_med": float(s["occ_cov_per_half_med"].median()),
                "n_bins_joint_med": float(s["n_bins_joint_fl_med"].median()),
            },
        }
        sep_sub = sep_sess_df[sep_sess_df.dataset == ds] if sep_sess_df is not None else pd.DataFrame()
        out["sep_curve"][ds] = _sep_curve_summary(sep_sub, rng)

    lab_curve = out["sep_curve"].get("lab") or {}
    mat_curve = out["sep_curve"].get("sim_matched") or {}
    lab_slope = lab_curve.get("slope", float("nan"))
    mat_slope = mat_curve.get("slope", float("nan"))
    lab_slo = lab_curve.get("slope_lo90", float("nan"))
    # Drift: lab declines (negative slope, lo90 < 0) and matched is flat (|slope| small)
    lab_declines = np.isfinite(lab_slope) and lab_slope < -0.02 and (
        np.isfinite(lab_slo) and lab_slo < 0
    )
    mat_flat = np.isfinite(mat_slope) and abs(mat_slope) < 0.03
    out["methods"] = {
        "first_last": "first vs last half of speed-filtered train frames",
        "odd_even_2s": "odd vs even 2 s blocks (interleaved noise control)",
        "odd_even_min": "odd vs even minutes (gap-aware spikes; was buggy before fix)",
        "sep_curve": (
            f"train → {N_BLOCKS} equal-duration blocks; place-tuned map r vs "
            "block separation 1–5; day-bootstrap 90% CI; OLS slope"
        ),
    }
    if lab_declines and mat_flat:
        out["verdict"] = "drift"
        out["verdict_note"] = (
            f"lab sep-curve slope={lab_slope:.3f} [{lab_curve.get('slope_lo90'):.3f},"
            f"{lab_curve.get('slope_hi90'):.3f}] declines; matched slope={mat_slope:.3f} "
            f"[{mat_curve.get('slope_lo90'):.3f},{mat_curve.get('slope_hi90'):.3f}] flat → drift"
        )
    else:
        out["verdict"] = "noise"
        out["verdict_note"] = (
            f"lab sep-curve slope={lab_slope:.3f} "
            f"[{lab_curve.get('slope_lo90', float('nan')):.3f},"
            f"{lab_curve.get('slope_hi90', float('nan')):.3f}]; "
            f"matched={mat_slope:.3f} → lab maps are less reliable (no drift)"
        )
    out["lab_slope"] = lab_slope
    out["matched_slope"] = mat_slope
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(args.pilot or None))
    payloads = [
        (kind, session, sim_dir, arena)
        for kind, session, sim_dir, arena in jobs
        if kind in ("lab", "sim_matched")
    ]
    print(json.dumps({"n_jobs": len(payloads), "pilot": args.pilot, "workers": args.workers}), flush=True)
    t0 = time.perf_counter()
    results = []
    if args.workers <= 1:
        for i, p in enumerate(payloads, 1):
            r = _worker(p)
            results.append(r)
            print(json.dumps({
                "done": i, "total": len(payloads), "session": p[1], "dataset": p[0],
                "status": r.get("status"), "elapsed_s": round(time.perf_counter() - t0, 1),
            }), flush=True)
    else:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futs = {ex.submit(_worker, p): p for p in payloads}
            for i, fut in enumerate(as_completed(futs), 1):
                p = futs[fut]
                r = fut.result()
                results.append(r)
                print(json.dumps({
                    "done": i, "total": len(payloads), "session": p[1], "dataset": p[0],
                    "status": r.get("status"), "elapsed_s": round(time.perf_counter() - t0, 1),
                }), flush=True)

    sess_rows, unit_rows, sep_rows, sep_sess_rows = [], [], [], []
    for r in results:
        if r.get("session_row"):
            sess_rows.append(r["session_row"])
        unit_rows.extend(r.get("unit_rows") or [])
        sep_rows.extend(r.get("sep_rows") or [])
        sep_sess_rows.extend(r.get("sep_sess_rows") or [])
    sess_df = pd.DataFrame(sess_rows)
    unit_df = pd.DataFrame(unit_rows)
    sep_df = pd.DataFrame(sep_rows)
    sep_sess_df = pd.DataFrame(sep_sess_rows)
    sess_df.to_csv(out / "drift_noise_sessions.csv", index=False)
    unit_df.to_csv(out / "drift_noise_units.csv", index=False)
    sep_df.to_csv(out / "drift_sep_pairs.csv", index=False)
    sep_sess_df.to_csv(out / "drift_sep_sessions.csv", index=False)
    summary = summarize(sess_df, unit_df, sep_sess_df)
    summary["n_ok"] = sum(1 for r in results if r.get("status") == "ok")
    summary["n_err"] = sum(1 for r in results if r.get("status") == "error")
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    summary["errors"] = [
        {"session": r.get("session"), "error": r.get("error")}
        for r in results if r.get("status") == "error"
    ][:10]
    (out / "drift_noise_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "wrote": str(out), "verdict": summary.get("verdict"),
        "note": summary.get("verdict_note"),
        "lab_slope": summary.get("lab_slope"),
        "matched_slope": summary.get("matched_slope"),
        "n_ok": summary["n_ok"], "elapsed_s": summary["elapsed_s"],
    }, indent=2), flush=True)
    print(json.dumps(summary["by_dataset"], indent=2), flush=True)
    print(json.dumps(summary["sep_curve"], indent=2), flush=True)


if __name__ == "__main__":
    main()
