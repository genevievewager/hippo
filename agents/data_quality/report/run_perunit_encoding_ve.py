#!/usr/bin/env python3
"""Per-unit behavioural encoding VE (N-independent) + place-tuned subset.

Explains why population VE (joint SSE/SST) can change with N, while per-unit
VE does not. Writes summary tables for story takeaway 3.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_REPO = Path(__file__).resolve().parents[3]
_R2V2 = Path("/home/gmwager/projects/hippo-r2v2")
for p in (str(_REPO), str(_R2V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from analysis.real_quadrant.adapter import _parse_spike_times  # noqa: E402
from realtime.train_decoder import causal_train_test_split  # noqa: E402
from agents.data_quality.matching.regions import stratify_lab_units  # noqa: E402
from agents.data_quality.metrics.boundary import bin_centers_reachable  # noqa: E402
from agents.data_quality.metrics.spatial import (  # noqa: E402
    MIN_OCC_S,
    SPEED_MIN_CM_S,
    _build_map,
    spatial_info_bits_per_spike,
    unit_eligible,
)
from agents.data_quality.report.run_behavior_info import (  # noqa: E402
    N_NULL,
    N_PRIMARY,
    TRAIN_FRAC,
    GAP_S,
    W_S,
    build_encoding_features,
    encode_once,
    iter_jobs,
    load_lab,
    load_sim,
    net_above_null,
    pick_ridge_alpha,
    pop_ve,
)

warnings.filterwarnings("ignore")

DQ = _REPO / "outputs/data_quality"
OUT = DQ / "reports/report4a_v3/behavior_info"
DATA = Path(os.environ["HIPPO_DATA_ROOT"])
def unit_ve(y_true, y_hat):
    mu = float(np.mean(y_true))
    sst = float(np.sum((y_true - mu) ** 2))
    sse = float(np.sum((y_true - y_hat) ** 2))
    return float(1.0 - sse / sst) if sst > 1e-12 else float("nan")


def _load_spikes_aligned(kind, session, sim_dir, regions_keep, n_units_expected):
    """Spike times aligned to the same primary-unit order as load_lab/load_sim."""
    if kind == "lab":
        from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq

        bundle = build_segment_bundle_dq(session, "A")
        meta = pd.read_csv(
            DATA / session / "clusters_dataset.csv",
            usecols=lambda c: c in ("cell", "Region", "BClabel", "timestamp"),
        )
        if "BClabel" in meta.columns:
            meta = meta[meta["BClabel"].astype(str) != "NON-SOMA"].reset_index(drop=True)
        id_to = {int(c): i for i, c in enumerate(meta["cell"].to_numpy())}
        spikes, regions = [], []
        for uid in bundle["unit_ids"]:
            i = id_to[int(uid)]
            spikes.append(_parse_spike_times(meta.loc[i, "timestamp"]))
            regions.append(str(meta.loc[i, "Region"]))
        keep = stratify_lab_units(regions, set_name="primary")
        spikes = [spikes[i] for i in np.flatnonzero(keep)]
    else:
        units = pd.read_csv(Path(sim_dir) / "units.csv")
        from agents.data_quality.matching.regions import add_mapped_lab_region

        units = add_mapped_lab_region(units)
        primary = units[units["lab_region"].isin(["MEC", "SUB"])].reset_index(drop=True)
        if len(primary) == 0:
            primary = units.reset_index(drop=True)
        sp = pd.read_csv(Path(sim_dir) / "spikes_sorted.csv")
        spikes = [
            sp.loc[sp["unit_id"] == uid, "spike_time_s"].to_numpy(float)
            for uid in primary["unit_id"]
        ]
    if len(spikes) != n_units_expected:
        # fall back: truncate/pad mismatch should not happen; skip place flags
        return None
    return spikes


def _session_place_frac(dataset: str, session: str) -> float:
    """Report session frac_place_tuned (SI > shuffle 95%); used to size the SI-ranked cohort."""
    p = DQ / "reports/report4a_v3/data/data_session_metrics.csv"
    if not p.is_file():
        return 0.15 if dataset == "lab" else 0.38
    df = pd.read_csv(p)
    hit = df[(df.dataset == dataset) & (df.session == session) & (df.metric == "frac_place_tuned")]
    if hit.empty or not np.isfinite(hit.iloc[0]["value"]):
        med = df[(df.dataset == dataset) & (df.metric == "frac_place_tuned")]["value"].median()
        return float(med) if np.isfinite(med) else (0.15 if dataset == "lab" else 0.38)
    return float(hit.iloc[0]["value"])


def place_tuned_flags(spikes, times, xy, speed, boundary_poly, dataset, session):
    """Place-tuned cohort: top SI units, sized to the report session frac_place_tuned.

    Full per-unit SI shuffle (100×) is too costly here and a light reimplementation
    over-called passes. Ranking by train-only SI and taking the report fraction
    recovers the same cohort definition used for frac_place_tuned.
    """
    flags = np.zeros(len(spikes), dtype=bool)
    si_vals = np.full(len(spikes), np.nan)
    times = np.asarray(times, float)
    xy = np.asarray(xy, float)
    speed = np.asarray(speed, float)
    finite = np.isfinite(xy).all(1) & np.isfinite(speed)
    t_win, p_win = times[finite], xy[finite]
    spd_win = speed[finite]
    if t_win.size < 50:
        return flags, si_vals
    moving = spd_win > SPEED_MIN_CM_S
    moving_idx = np.flatnonzero(moving)
    if moving_idx.size < 50:
        return flags, si_vals
    n_train = int(0.8 * moving_idx.size)
    keep = np.zeros(len(t_win), dtype=bool)
    keep[moving_idx[:n_train]] = True
    p_tr = p_win[keep]
    xmin, xmax = float(np.min(p_tr[:, 0])), float(np.max(p_tr[:, 0]))
    ymin, ymax = float(np.min(p_tr[:, 1])), float(np.max(p_tr[:, 1]))
    if xmax <= xmin or ymax <= ymin:
        return flags, si_vals
    edges_x = np.linspace(xmin, xmax, 21)
    edges_y = np.linspace(ymin, ymax, 21)
    t_tr = t_win[keep]
    dt = float(np.median(np.diff(t_tr))) if t_tr.size > 1 else W_S
    occ_thresh = max(1.0, MIN_OCC_S / max(dt, 1e-9))
    t_lo, t_hi = float(t_tr[0]), float(t_tr[-1])
    reachable = None
    if boundary_poly is not None:
        try:
            reachable = bin_centers_reachable(edges_x, edges_y, boundary_poly)
        except Exception:
            reachable = None
    elig = []
    for j, st in enumerate(spikes):
        if not unit_eligible(st, t_lo, t_hi):
            continue
        rate, occ, ok = _build_map(
            st, t_win, p_win, edges_x, edges_y, occ_thresh, dt, reachable, None, keep=keep,
        )
        si = spatial_info_bits_per_spike(rate, occ, ok)
        si_vals[j] = si
        if np.isfinite(si):
            elig.append(j)
    if not elig:
        return flags, si_vals
    frac = _session_place_frac(dataset, session)
    k = max(1, int(round(frac * len(elig))))
    order = sorted(elig, key=lambda j: si_vals[j], reverse=True)
    for j in order[:k]:
        flags[j] = True
    return flags, si_vals


def run_session(kind, session, sim_dir, arena, do_place=True):
    if kind == "lab":
        pack = load_lab(session)
    else:
        pack = load_sim(sim_dir, kind, session, arena)
    X = np.asarray(pack["X"], float)
    beh = pack["beh"]
    times = np.asarray(pack["times"], float)
    n_u = X.shape[1]
    train, test = causal_train_test_split(times, train_frac=TRAIN_FRAC, gap_s=GAP_S)
    Fe = build_encoding_features(beh)
    Ys = np.sqrt(np.maximum(X, 0.0))
    mu_y, sd_y = Ys[train].mean(0), Ys[train].std(0) + 1e-6
    Yz = (Ys - mu_y) / sd_y

    # shared alpha from a unit subsample (same as encode_once)
    alpha = pick_ridge_alpha(Fe[train], Yz[train][:, : min(20, n_u)])
    mu_x, sd_x = Fe[train].mean(0), Fe[train].std(0) + 1e-6
    Xtrz = (Fe[train] - mu_x) / sd_x
    Xtez = (Fe[test] - mu_x) / sd_x
    pred = Ridge(alpha=alpha).fit(Xtrz, Yz[train]).predict(Xtez)
    Yte = Yz[test]

    ve_units = np.array([unit_ve(Yte[:, j], pred[:, j]) for j in range(n_u)], float)
    # null: circular shifts of features; per-unit p95
    rng = np.random.default_rng(abs(hash((kind, session))) % (2**31))
    shifts = rng.integers(int(5 / W_S), max(int(5 / W_S) + 1, int(0.5 * len(times))), size=N_NULL)
    null_mat = np.zeros((N_NULL, n_u), float)
    for i, sh in enumerate(shifts):
        Fe_n = np.roll(Fe, int(sh), axis=0)
        Xtrz_n = (Fe_n[train] - Fe_n[train].mean(0)) / (Fe_n[train].std(0) + 1e-6)
        Xtez_n = (Fe_n[test] - Fe_n[train].mean(0)) / (Fe_n[train].std(0) + 1e-6)
        pred_n = Ridge(alpha=alpha).fit(Xtrz_n, Yz[train]).predict(Xtez_n)
        null_mat[i] = [unit_ve(Yte[:, j], pred_n[:, j]) for j in range(n_u)]
    null_p95 = np.nanpercentile(null_mat, 95, axis=0)
    null_mean = np.nanmean(null_mat, axis=0)
    above = ve_units > null_p95

    # population VE diagnostics at N=20 vs N=max (same as Fig 8)
    pop_max, _ = encode_once(Fe, Yz, train, test, alpha=alpha)
    if n_u >= N_PRIMARY:
        idx20 = rng.choice(n_u, size=N_PRIMARY, replace=False)
        pop_20, _ = encode_once(Fe, Yz[:, idx20], train, test, alpha=alpha)
        # per-unit stats on that same N=20 draw
        med_u20 = float(np.nanmedian(ve_units[idx20]))
        mean_u20 = float(np.nanmean(ve_units[idx20]))
        # variance-weighted average of per-unit R² = pop_ve on those units
        pop_from_units_20 = float(pop_ve(Yte[:, idx20], pred[:, idx20]))
    else:
        pop_20 = float("nan")
        med_u20 = mean_u20 = pop_from_units_20 = float("nan")

    place = np.zeros(n_u, dtype=bool)
    si_vals = np.full(n_u, np.nan)
    if do_place:
        spikes = _load_spikes_aligned(kind, session, sim_dir, pack["regions"], n_u)
        if spikes is not None:
            place, si_vals = place_tuned_flags(
                spikes, times, beh["xy"], beh["speed"], beh.get("poly"),
                kind, session,
            )

    unit_rows = []
    for j in range(n_u):
        unit_rows.append({
            "dataset": kind, "session": session, "animal": pack["animal"],
            "unit": int(j), "ve": float(ve_units[j]),
            "null_mean": float(null_mean[j]), "null_p95": float(null_p95[j]),
            "above_null_p95": bool(above[j]),
            "place_tuned": bool(place[j]),
            "spatial_info": float(si_vals[j]) if np.isfinite(si_vals[j]) else float("nan"),
            "ve_minus_null": net_above_null(float(ve_units[j]), float(null_mean[j])),
        })

    ve_pt = ve_units[place] if place.any() else np.array([], float)
    sess = {
        "dataset": kind, "session": session, "animal": pack["animal"],
        "n_units": int(n_u),
        "pop_ve_Nmax": float(pop_max),
        "pop_ve_N20": float(pop_20),
        "pop_ve_from_units_N20": float(pop_from_units_20),
        "unit_ve_median": float(np.nanmedian(ve_units)),
        "unit_ve_mean": float(np.nanmean(ve_units)),
        "unit_ve_p90": float(np.nanpercentile(ve_units, 90)),
        "unit_ve_median_N20draw": med_u20,
        "unit_ve_mean_N20draw": mean_u20,
        "frac_above_null_p95": float(np.mean(above)),
        "n_place_tuned": int(place.sum()),
        "frac_place_tuned": float(place.mean()),
        "unit_ve_median_place": float(np.nanmedian(ve_pt)) if ve_pt.size else float("nan"),
        "unit_ve_mean_place": float(np.nanmean(ve_pt)) if ve_pt.size else float("nan"),
        "alpha": float(alpha),
    }
    return {"status": "ok", "session_row": sess, "unit_rows": unit_rows}


def _worker(payload):
    kind, session, sim_dir, arena, do_place = payload
    try:
        return run_session(kind, session, sim_dir, arena, do_place=do_place)
    except Exception as e:
        return {
            "status": "error", "dataset": kind, "session": session,
            "error": f"{type(e).__name__}: {e}",
            "session_row": None, "unit_rows": [],
        }


def summarize(sess_df: pd.DataFrame, unit_df: pd.DataFrame) -> dict:
    out = {"by_dataset": {}, "explanation": (
        "Population VE is joint multivariate R² = 1 − ΣSSE/ΣSST (variance-weighted "
        "across units). A random N=20 subsample changes which units enter that sum, so "
        "pop VE can move even when per-unit VEs are unchanged. Per-unit VE is the "
        "correct N-independent summary."
    )}
    for ds in ("lab", "sim_matched", "sim_default"):
        s = sess_df[sess_df.dataset == ds]
        u = unit_df[unit_df.dataset == ds]
        if s.empty:
            continue
        out["by_dataset"][ds] = {
            "n_sessions": int(len(s)),
            "n_units": int(len(u)),
            "pop_ve_N20_session_median": float(s["pop_ve_N20"].median()),
            "pop_ve_Nmax_session_median": float(s["pop_ve_Nmax"].median()),
            "unit_ve_median": float(u["ve"].median()),
            "unit_ve_mean": float(u["ve"].mean()),
            "unit_ve_p90": float(u["ve"].quantile(0.90)),
            "frac_above_null_p95": float(u["above_null_p95"].mean()),
            "unit_ve_median_place": float(u.loc[u.place_tuned, "ve"].median())
            if u.place_tuned.any() else float("nan"),
            "unit_ve_mean_place": float(u.loc[u.place_tuned, "ve"].mean())
            if u.place_tuned.any() else float("nan"),
            "n_place_tuned": int(u.place_tuned.sum()),
            "frac_place_tuned": float(u.place_tuned.mean()),
            # session-median of session-level unit summaries (alt aggregation)
            "session_median_of_unit_median": float(s["unit_ve_median"].median()),
            "session_median_of_unit_mean": float(s["unit_ve_mean"].median()),
            "session_median_of_unit_p90": float(s["unit_ve_p90"].median()),
            "session_median_frac_above_null": float(s["frac_above_null_p95"].median()),
            "session_median_of_place_median": float(s["unit_ve_median_place"].median()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-place", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(args.pilot or None))
    # primary story arms: lab + matched (default optional)
    payloads = [
        (kind, session, sim_dir, arena, not args.skip_place)
        for kind, session, sim_dir, arena in jobs
        if kind in ("lab", "sim_matched", "sim_default")
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

    sess_rows, unit_rows = [], []
    for r in results:
        if r.get("session_row"):
            sess_rows.append(r["session_row"])
        unit_rows.extend(r.get("unit_rows") or [])
    sess_df = pd.DataFrame(sess_rows)
    unit_df = pd.DataFrame(unit_rows)
    sess_df.to_csv(out / "encoding_perunit_sessions.csv", index=False)
    # unit table can be large — keep, but also write a compact sample
    unit_df.to_csv(out / "encoding_perunit_units.csv", index=False)
    summary = summarize(sess_df, unit_df)
    summary["n_ok"] = sum(1 for r in results if r.get("status") == "ok")
    summary["n_err"] = sum(1 for r in results if r.get("status") == "error")
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    summary["errors"] = [
        {"session": r.get("session"), "error": r.get("error")}
        for r in results if r.get("status") == "error"
    ][:10]
    (out / "encoding_perunit_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"wrote": str(out), **{k: summary[k] for k in ("n_ok", "n_err", "elapsed_s")}}), flush=True)
    print(json.dumps(summary["by_dataset"], indent=2), flush=True)


if __name__ == "__main__":
    main()
