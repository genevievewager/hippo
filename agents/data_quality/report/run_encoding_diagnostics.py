#!/usr/bin/env python3
"""Diagnose lab per-unit encoding VE failure (overfit / drift / measure).

Reports for lab vs matched (all units + place-tuned):
  1a. Position-only train rate-map → held-out VE
  1b. Full ridge with per-unit inner-CV alpha
  2.  Interleaved CV (10 blocks, 1 s purge) vs end-block; half-map r
  3.  Held-out Poisson bits/spike (rate-map vs homogeneous) — primary

VE is secondary; bits/spike is primary for takeaway 3.
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
from sklearn.linear_model import Ridge, RidgeCV

_REPO = Path(__file__).resolve().parents[3]
_R2V2 = Path("/home/gmwager/projects/hippo-r2v2")
for p in (str(_REPO), str(_R2V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from realtime.train_decoder import causal_train_test_split  # noqa: E402
from agents.data_quality.metrics.boundary import bin_centers_reachable  # noqa: E402
from agents.data_quality.metrics.spatial import (  # noqa: E402
    MIN_OCC_S,
    SMOOTH_SIGMA_BINS,
    SPEED_MIN_CM_S,
    _build_map,
    _spike_positions,
    smooth_rate_map,
    unit_eligible,
)
from agents.data_quality.metrics.spatial import _rate_map as _raw_rate_map  # noqa: E402
from agents.data_quality.report.run_behavior_info import (  # noqa: E402
    ALPHAS,
    GAP_S,
    TRAIN_FRAC,
    W_S,
    build_encoding_features,
    iter_jobs,
    load_lab,
    load_sim,
)
from agents.data_quality.report.run_perunit_encoding_ve import (  # noqa: E402
    _load_spikes_aligned,
    place_tuned_flags,
    unit_ve,
)

warnings.filterwarnings("ignore")

DQ = _REPO / "outputs/data_quality"
OUT = DQ / "reports/report4a_v3/behavior_info"
# Stronger + wider alpha grid for per-unit CV
ALPHAS_CV = np.logspace(-2, 5, 14)
N_BLOCKS = 10
PURGE_S = 1.0


def unit_ve_rates(y, yhat):
    return unit_ve(np.asarray(y, float), np.asarray(yhat, float))


def poisson_bits_per_spike(counts, lam_hz, dt, lam0_hz=None):
    """Held-out Poisson bits/spike vs homogeneous rate.

    ΔLL = Σ [ n log(λ/λ0) − (λ−λ0) Δt ]; bits/spike = ΔLL / (N ln 2).
    """
    n = np.asarray(counts, float)
    lam = np.clip(np.asarray(lam_hz, float), 1e-6, None)
    if lam0_hz is None:
        # MLE homogeneous rate on the same frames
        dur = float(len(n) * dt)
        lam0 = float(n.sum() / max(dur, 1e-12))
    else:
        lam0 = float(max(lam0_hz, 1e-6))
    nspk = float(n.sum())
    if nspk < 5:
        return float("nan")
    dll = float(np.sum(n * np.log(lam / lam0) - (lam - lam0) * dt))
    return float(dll / (nspk * np.log(2.0)))


def end_block_split(times):
    return causal_train_test_split(times, train_frac=TRAIN_FRAC, gap_s=GAP_S)


def interleaved_split(times, n_blocks=N_BLOCKS, gap_s=PURGE_S):
    times = np.asarray(times, float)
    t0, t1 = float(times.min()), float(times.max())
    edges = np.linspace(t0, t1, n_blocks + 1)
    train = np.zeros(len(times), bool)
    test = np.zeros(len(times), bool)
    for b in range(n_blocks):
        m = (times >= edges[b]) & (times < edges[b + 1])
        # purge near block edges
        if gap_s > 0:
            m = m & (times >= edges[b] + gap_s) & (times < edges[b + 1] - gap_s)
        if b % 2 == 0:
            train |= m
        else:
            test |= m
    # ensure both non-empty
    if train.sum() < 40 or test.sum() < 25:
        return end_block_split(times)
    return train, test


def _map_edges(xy_train, grid_n=20):
    xmin, xmax = float(xy_train[:, 0].min()), float(xy_train[:, 0].max())
    ymin, ymax = float(xy_train[:, 1].min()), float(xy_train[:, 1].max())
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    return np.linspace(xmin, xmax, grid_n + 1), np.linspace(ymin, ymax, grid_n + 1)


def build_train_rate_maps(spikes, times, xy, train, speed, boundary_poly, dt):
    """20×20 train rate maps (Hz), same smoothing as spatial metrics."""
    times = np.asarray(times, float)
    xy = np.asarray(xy, float)
    train = np.asarray(train, bool)
    spd = np.asarray(speed, float)
    moving = np.isfinite(spd) & (spd > SPEED_MIN_CM_S)
    keep = train & moving
    if keep.sum() < 50:
        keep = train.copy()
    p_tr = xy[keep]
    edges_x, edges_y = _map_edges(p_tr)
    occ_thresh = max(1.0, MIN_OCC_S / max(dt, 1e-9))
    reachable = None
    if boundary_poly is not None:
        try:
            reachable = bin_centers_reachable(edges_x, edges_y, boundary_poly)
        except Exception:
            reachable = None
    maps = []
    kept_t = times[keep]
    t_lo, t_hi = float(kept_t[0]), float(kept_t[-1])
    for st in spikes:
        if not unit_eligible(st, t_lo, t_hi):
            maps.append(None)
            continue
        rate, occ, ok = _build_map(
            st, times, xy, edges_x, edges_y, occ_thresh, dt, reachable, None, keep=keep,
        )
        maps.append((rate, ok, edges_x, edges_y))
    return maps, edges_x, edges_y, reachable, times, xy, keep


def lookup_rate(rate, ok, edges_x, edges_y, xy):
    """Nearest-bin rate (Hz); unoccupied → global occupied mean."""
    xy = np.asarray(xy, float)
    ix = np.clip(np.digitize(xy[:, 0], edges_x[1:-1]), 0, rate.shape[0] - 1)
    iy = np.clip(np.digitize(xy[:, 1], edges_y[1:-1]), 0, rate.shape[1] - 1)
    out = np.full(len(xy), np.nan)
    fill = float(rate[ok].mean()) if ok.any() else 1e-3
    for i in range(len(xy)):
        if ok[ix[i], iy[i]]:
            out[i] = rate[ix[i], iy[i]]
        else:
            out[i] = fill
    return np.clip(out, 1e-6, None)


def half_map_correlation(spikes, t_frames, p_frames, keep, edges_x, edges_y, reachable, dt, place_mask):
    """Pearson r of first-half vs last-half kept train rate maps (median over units)."""
    keep = np.asarray(keep, bool)
    kept_idx = np.flatnonzero(keep)
    if kept_idx.size < 40:
        return float("nan"), float("nan")
    mid = kept_idx.size // 2
    keep_a = np.zeros(len(keep), dtype=bool)
    keep_b = np.zeros(len(keep), dtype=bool)
    keep_a[kept_idx[:mid]] = True
    keep_b[kept_idx[mid:]] = True
    occ_thresh = max(1.0, MIN_OCC_S / max(dt, 1e-9))
    rs_all, rs_pt = [], []
    for j, st in enumerate(spikes):
        r1, _, ok1 = _build_map(
            st, t_frames, p_frames, edges_x, edges_y, occ_thresh, dt, reachable, None, keep=keep_a,
        )
        r2, _, ok2 = _build_map(
            st, t_frames, p_frames, edges_x, edges_y, occ_thresh, dt, reachable, None, keep=keep_b,
        )
        m = ok1 & ok2
        if m.sum() < 10:
            continue
        a, b = r1[m], r2[m]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        r = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(r):
            rs_all.append(r)
            if place_mask is not None and place_mask[j]:
                rs_pt.append(r)
    return (
        float(np.nanmedian(rs_all)) if rs_all else float("nan"),
        float(np.nanmedian(rs_pt)) if rs_pt else float("nan"),
    )


def ridge_per_unit_cv(Fe, rates, train, test, alphas=ALPHAS_CV):
    """Per-unit RidgeCV on train; held-out VE on test rates. Returns ve, alphas."""
    Xtr, Xte = Fe[train], Fe[test]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtrz, Xtez = (Xtr - mu) / sd, (Xte - mu) / sd
    n_u = rates.shape[1]
    ve = np.full(n_u, np.nan)
    alphas_out = np.full(n_u, np.nan)
    Ytr, Yte = rates[train], rates[test]
    # subsample frames for CV speed if huge
    n_tr = Xtrz.shape[0]
    if n_tr > 4000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n_tr, size=4000, replace=False)
    else:
        idx = np.arange(n_tr)
    for j in range(n_u):
        ytr = Ytr[:, j]
        if not np.isfinite(ytr).all() or np.std(ytr) < 1e-12:
            continue
        try:
            m = RidgeCV(alphas=alphas).fit(Xtrz[idx], ytr[idx])
            a = float(m.alpha_)
        except Exception:
            a = float(np.median(alphas))
            m = Ridge(alpha=a).fit(Xtrz[idx], ytr[idx])
        pred = Ridge(alpha=a).fit(Xtrz, ytr).predict(Xtez)
        ve[j] = unit_ve_rates(Yte[:, j], pred)
        alphas_out[j] = a
    return ve, alphas_out


def ridge_shared_alpha(Fe, rates, train, test, alpha):
    Xtr, Xte = Fe[train], Fe[test]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtrz, Xtez = (Xtr - mu) / sd, (Xte - mu) / sd
    pred = Ridge(alpha=alpha).fit(Xtrz, rates[train]).predict(Xtez)
    return np.array([unit_ve_rates(rates[test, j], pred[:, j]) for j in range(rates.shape[1])])


def rate_map_heldout(spikes, times, xy, train, test, speed, boundary_poly, counts, dt):
    """Position-only: train rate map → predicted test rate; VE + bits/spike.

    Gain-calibrate on train: causal-window rates are not on the same scale as
    spike/occupancy maps (speed filter, smoothing, windowing), so
    g = mean(y_train) / mean(λ_map(x_train)).
    """
    maps, edges_x, edges_y, reachable, t_frames, p_frames, keep = build_train_rate_maps(
        spikes, times, xy, train, speed, boundary_poly, dt,
    )
    n_u = len(spikes)
    ve = np.full(n_u, np.nan)
    bits = np.full(n_u, np.nan)
    bits_spk = np.full(n_u, np.nan)  # spike-triggered Σ log2(λ/λ0) / Nspk
    rate_ratio = np.full(n_u, np.nan)  # test/train mean rate (gain drift)
    xy_tr, xy_te = xy[train], xy[test]
    counts_tr, counts_te = counts[train], counts[test]
    y_tr_all = counts_tr / dt
    y_te_all = counts_te / dt
    times_te = times[test]
    for j, mp in enumerate(maps):
        if mp is None:
            continue
        rate, ok, ex, ey = mp
        lam_tr = lookup_rate(rate, ok, ex, ey, xy_tr)
        lam_te = lookup_rate(rate, ok, ex, ey, xy_te)
        y_tr, y_te = y_tr_all[:, j], y_te_all[:, j]
        denom = float(np.mean(lam_tr))
        g = float(np.mean(y_tr) / denom) if denom > 1e-12 else 1.0
        lam_hat = lam_te * g
        ve[j] = unit_ve_rates(y_te, lam_hat)
        lam0 = float(np.mean(y_tr))
        bits[j] = poisson_bits_per_spike(counts_te[:, j], lam_hat, dt, lam0_hz=lam0)
        rate_ratio[j] = float(np.mean(y_te) / max(lam0, 1e-12))
        # spike-triggered bits on test spikes
        st = np.asarray(spikes[j], float)
        if times_te.size and st.size:
            st_te = st[(st >= times_te[0]) & (st < times_te[-1])]
            if st_te.size >= 5:
                idx = np.clip(np.searchsorted(times_te, st_te, side="right") - 1, 0, len(times_te) - 1)
                bits_spk[j] = float(np.mean(np.log2(np.clip(lam_hat[idx], 1e-6, None) / max(lam0, 1e-6))))
    return ve, bits, bits_spk, rate_ratio, edges_x, edges_y, reachable, t_frames, p_frames, keep, maps


def _agg(vals, place):
    vals = np.asarray(vals, float)
    out = {
        "median_all": float(np.nanmedian(vals)),
        "mean_all": float(np.nanmean(vals)),
        "median_place": float(np.nanmedian(vals[place])) if place.any() and np.isfinite(vals[place]).any() else float("nan"),
        "mean_place": float(np.nanmean(vals[place])) if place.any() and np.isfinite(vals[place]).any() else float("nan"),
        "frac_finite": float(np.isfinite(vals).mean()),
    }
    return out


def run_session(kind, session, sim_dir, arena):
    if kind == "lab":
        pack = load_lab(session)
    else:
        pack = load_sim(sim_dir, kind, session, arena)
    X = np.asarray(pack["X"], float)  # counts in W_S
    beh = pack["beh"]
    times = np.asarray(pack["times"], float)
    dt = float(np.median(np.diff(times))) if len(times) > 1 else W_S
    # use actual bin width for rate conversion
    dt_bin = float(W_S)
    rates = X / dt_bin
    n_u = X.shape[1]
    spikes = _load_spikes_aligned(kind, session, sim_dir, pack["regions"], n_u)
    if spikes is None:
        return {"status": "error", "session": session, "dataset": kind, "error": "spike align",
                "session_row": None, "unit_rows": []}

    place, si_vals = place_tuned_flags(
        spikes, times, beh["xy"], beh["speed"], beh.get("poly"), kind, session,
    )
    Fe = build_encoding_features(beh)

    # --- end-block split ---
    tr_e, te_e = end_block_split(times)
    ve_map_e, bits_e, bits_spk_e, rate_ratio_e, edges_x, edges_y, reachable, t_fr, p_fr, keep_tr, maps = (
        rate_map_heldout(
            spikes, times, beh["xy"], tr_e, te_e, beh["speed"], beh.get("poly"), X, dt_bin,
        )
    )
    # per-unit ridge CV (full model)
    ve_cv_e, alphas_e = ridge_per_unit_cv(Fe, rates, tr_e, te_e)
    # shared mid alpha (legacy-like) for reference
    alpha_shared = float(np.nanmedian(alphas_e)) if np.isfinite(alphas_e).any() else float(np.median(ALPHAS))
    ve_shared_e = ridge_shared_alpha(Fe, rates, tr_e, te_e, alpha_shared)

    # --- interleaved split ---
    tr_i, te_i = interleaved_split(times)
    ve_map_i, bits_i, bits_spk_i, rate_ratio_i, _, _, _, _, _, _, _ = rate_map_heldout(
        spikes, times, beh["xy"], tr_i, te_i, beh["speed"], beh.get("poly"), X, dt_bin,
    )
    ve_cv_i, alphas_i = ridge_per_unit_cv(Fe, rates, tr_i, te_i)

    # half-map stability on end-block train
    half_r_all, half_r_pt = half_map_correlation(
        spikes, t_fr, p_fr, keep_tr, edges_x, edges_y, reachable, dt_bin, place,
    )

    unit_rows = []
    for j in range(n_u):
        unit_rows.append({
            "dataset": kind, "session": session, "animal": pack["animal"],
            "unit": int(j), "place_tuned": bool(place[j]),
            "spatial_info": float(si_vals[j]) if np.isfinite(si_vals[j]) else float("nan"),
            "ve_ratemap_end": float(ve_map_e[j]),
            "ve_ridge_cv_end": float(ve_cv_e[j]),
            "ve_ridge_shared_end": float(ve_shared_e[j]),
            "alpha_cv_end": float(alphas_e[j]),
            "ve_ratemap_inter": float(ve_map_i[j]),
            "ve_ridge_cv_inter": float(ve_cv_i[j]),
            "alpha_cv_inter": float(alphas_i[j]),
            "bits_ratemap_end": float(bits_e[j]),
            "bits_spk_end": float(bits_spk_e[j]),
            "bits_ratemap_inter": float(bits_i[j]),
            "bits_spk_inter": float(bits_spk_i[j]),
            "rate_ratio_test_train_end": float(rate_ratio_e[j]),
            "delta_ve_ridge_inter_minus_end": float(ve_cv_i[j] - ve_cv_e[j])
            if np.isfinite(ve_cv_i[j]) and np.isfinite(ve_cv_e[j]) else float("nan"),
            "delta_bits_inter_minus_end": float(bits_i[j] - bits_e[j])
            if np.isfinite(bits_i[j]) and np.isfinite(bits_e[j]) else float("nan"),
        })

    def smed(a):
        a = np.asarray(a, float)
        return float(np.nanmedian(a))

    sess = {
        "dataset": kind, "session": session, "animal": pack["animal"],
        "n_units": int(n_u), "n_place": int(place.sum()),
        "ve_ratemap_end_med": smed(ve_map_e),
        "ve_ratemap_end_med_place": smed(ve_map_e[place]) if place.any() else float("nan"),
        "ve_ridge_cv_end_med": smed(ve_cv_e),
        "ve_ridge_cv_end_med_place": smed(ve_cv_e[place]) if place.any() else float("nan"),
        "ve_ridge_shared_end_med": smed(ve_shared_e),
        "alpha_cv_end_med": smed(alphas_e),
        "alpha_cv_end_p10": float(np.nanpercentile(alphas_e, 10)),
        "alpha_cv_end_p90": float(np.nanpercentile(alphas_e, 90)),
        "ve_ratemap_inter_med": smed(ve_map_i),
        "ve_ridge_cv_inter_med": smed(ve_cv_i),
        "ve_ridge_cv_inter_med_place": smed(ve_cv_i[place]) if place.any() else float("nan"),
        "delta_ve_ridge_med": smed(ve_cv_i - ve_cv_e),
        "bits_end_med": smed(bits_e),
        "bits_end_med_place": smed(bits_e[place]) if place.any() else float("nan"),
        "bits_spk_end_med": smed(bits_spk_e),
        "bits_spk_end_med_place": smed(bits_spk_e[place]) if place.any() else float("nan"),
        "bits_inter_med": smed(bits_i),
        "bits_inter_med_place": smed(bits_i[place]) if place.any() else float("nan"),
        "delta_bits_med": smed(bits_i - bits_e),
        "rate_ratio_med": smed(rate_ratio_e),
        "half_map_r_all": half_r_all,
        "half_map_r_place": half_r_pt,
        "n_train_end": int(tr_e.sum()), "n_test_end": int(te_e.sum()),
        "n_train_inter": int(tr_i.sum()), "n_test_inter": int(te_i.sum()),
    }
    return {"status": "ok", "session_row": sess, "unit_rows": unit_rows}


def _worker(payload):
    kind, session, sim_dir, arena = payload
    try:
        return run_session(kind, session, sim_dir, arena)
    except Exception as e:
        return {
            "status": "error", "dataset": kind, "session": session,
            "error": f"{type(e).__name__}: {e}",
            "session_row": None, "unit_rows": [],
        }


def summarize(sess_df, unit_df):
    # prior frac above null from previous per-unit run
    prior = {}
    prior_p = OUT / "encoding_perunit_summary.json"
    if prior_p.is_file():
        prior = json.loads(prior_p.read_text()).get("by_dataset") or {}

    table = {}
    for ds in ("lab", "sim_matched"):
        s = sess_df[sess_df.dataset == ds]
        u = unit_df[unit_df.dataset == ds]
        if s.empty:
            continue
        pt = u["place_tuned"].to_numpy(bool)
        row = {
            "n_sessions": int(len(s)),
            "n_units": int(len(u)),
            "frac_place": float(pt.mean()),
            # 1a rate-map VE
            "ve_ratemap_end_med_all": float(u["ve_ratemap_end"].median()),
            "ve_ratemap_end_med_place": float(u.loc[pt, "ve_ratemap_end"].median()) if pt.any() else float("nan"),
            # 1b ridge CV
            "ve_ridge_cv_end_med_all": float(u["ve_ridge_cv_end"].median()),
            "ve_ridge_cv_end_med_place": float(u.loc[pt, "ve_ridge_cv_end"].median()) if pt.any() else float("nan"),
            "alpha_cv_med": float(u["alpha_cv_end"].median()),
            "alpha_cv_p10": float(u["alpha_cv_end"].quantile(0.10)),
            "alpha_cv_p90": float(u["alpha_cv_end"].quantile(0.90)),
            # session-median view
            "sess_ve_ratemap_end": float(s["ve_ratemap_end_med"].median()),
            "sess_ve_ridge_cv_end": float(s["ve_ridge_cv_end_med"].median()),
            "sess_ve_ridge_cv_end_place": float(s["ve_ridge_cv_end_med_place"].median()),
            # 2 interleaved
            "ve_ridge_cv_inter_med_all": float(u["ve_ridge_cv_inter"].median()),
            "ve_ridge_cv_inter_med_place": float(u.loc[pt, "ve_ridge_cv_inter"].median()) if pt.any() else float("nan"),
            "delta_ve_ridge_inter_minus_end": float(u["delta_ve_ridge_inter_minus_end"].median()),
            "sess_delta_ve_ridge": float(s["delta_ve_ridge_med"].median()),
            "half_map_r_all": float(s["half_map_r_all"].median()),
            "half_map_r_place": float(s["half_map_r_place"].median()),
            # 3 bits/spike primary (gain-calibrated Poisson + spike-triggered)
            "bits_end_med_all": float(u["bits_ratemap_end"].median()),
            "bits_end_med_place": float(u.loc[pt, "bits_ratemap_end"].median()) if pt.any() else float("nan"),
            "bits_spk_end_med_all": float(u["bits_spk_end"].median()),
            "bits_spk_end_med_place": float(u.loc[pt, "bits_spk_end"].median()) if pt.any() else float("nan"),
            "bits_inter_med_all": float(u["bits_ratemap_inter"].median()),
            "bits_inter_med_place": float(u.loc[pt, "bits_ratemap_inter"].median()) if pt.any() else float("nan"),
            "sess_bits_end": float(s["bits_end_med"].median()),
            "sess_bits_end_place": float(s["bits_end_med_place"].median()),
            "sess_bits_spk_end": float(s["bits_spk_end_med"].median()),
            "sess_bits_spk_end_place": float(s["bits_spk_end_med_place"].median()),
            "sess_bits_inter": float(s["bits_inter_med"].median()),
            "sess_delta_bits": float(s["delta_bits_med"].median()),
            "sess_rate_ratio": float(s["rate_ratio_med"].median()),
            "sess_half_map_r_all": float(s["half_map_r_all"].median()),
            "sess_half_map_r_place": float(s["half_map_r_place"].median()),
        }
        if ds in prior:
            row["frac_above_null_p95_prior"] = prior[ds].get("frac_above_null_p95")
            row["prior_ve_median"] = prior[ds].get("unit_ve_median")
            row["prior_ve_place"] = prior[ds].get("unit_ve_median_place")
        table[ds] = row
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lab-matched-only", action="store_true", default=True)
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

    sess_rows, unit_rows = [], []
    for r in results:
        if r.get("session_row"):
            sess_rows.append(r["session_row"])
        unit_rows.extend(r.get("unit_rows") or [])
    sess_df = pd.DataFrame(sess_rows)
    unit_df = pd.DataFrame(unit_rows)
    sess_df.to_csv(out / "encoding_diag_sessions.csv", index=False)
    unit_df.to_csv(out / "encoding_diag_units.csv", index=False)
    table = summarize(sess_df, unit_df)
    summary = {
        "table": table,
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_err": sum(1 for r in results if r.get("status") == "error"),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "methods": {
            "rate_map": "20×20 train map, σ=1.5 bins, speed>2, held-out position lookup",
            "ridge_cv": f"per-unit RidgeCV alphas={ALPHAS_CV.tolist()[:3]}…{ALPHAS_CV.tolist()[-1]}",
            "end_block": f"train_frac={TRAIN_FRAC}, gap_s={GAP_S}",
            "interleaved": f"{N_BLOCKS} blocks, purge={PURGE_S}s, alternating train/test",
            "bits": "Poisson LL ratio rate-map vs homogeneous train rate, bits/spike",
            "half_map_r": "Pearson r of first-half vs last-half train rate maps",
        },
        "errors": [
            {"session": r.get("session"), "error": r.get("error")}
            for r in results if r.get("status") == "error"
        ][:10],
    }
    (out / "encoding_diag_summary.json").write_text(json.dumps(summary, indent=2))
    # human-readable diagnostic table
    lines = ["# Encoding diagnostics (lab vs matched)\n"]
    for ds, row in table.items():
        lines.append(f"## {ds}\n")
        for k, v in row.items():
            if isinstance(v, float):
                lines.append(f"- {k}: {v:.4f}")
            else:
                lines.append(f"- {k}: {v}")
        lines.append("")
    (out / "encoding_diag_TABLE.md").write_text("\n".join(lines))
    print(json.dumps({"wrote": str(out), "n_ok": summary["n_ok"], "n_err": summary["n_err"],
                      "elapsed_s": summary["elapsed_s"]}), flush=True)
    print(json.dumps(table, indent=2), flush=True)


if __name__ == "__main__":
    main()
