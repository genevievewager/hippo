#!/usr/bin/env python3
"""Report 4a behavioural information content (Step 2/3) — serial, fixed-α nulls."""
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
import yaml
from scipy.interpolate import BSpline
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import balanced_accuracy_score, f1_score, r2_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

_REPO = Path(__file__).resolve().parents[3]
_R2V2 = Path("/home/gmwager/projects/hippo-r2v2")
for p in (str(_REPO), str(_R2V2)):
    if p not in sys.path:
        sys.path.insert(0, p)

from realtime.decoding_targets import classify_wall_distance_bin, circular_error_deg  # noqa: E402
from realtime.train_decoder import (  # noqa: E402
    classify_movement_state,
    classify_spatial_context,
    causal_train_test_split,
)
from realtime.quadrant_n5_run import _sqrt_zscore_train, segment_retained_mask  # noqa: E402
from analysis.real_quadrant.adapter import causal_count_matrix, _room_boundary  # noqa: E402
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq  # noqa: E402
from agents.data_quality.matching.regions import add_mapped_lab_region, stratify_lab_units  # noqa: E402
from agents.data_quality.metrics.boundary import boundary_polygon_xy, square_polygon  # noqa: E402
from agents.data_quality.phase3.divergence import divergence_for_metric  # noqa: E402

warnings.filterwarnings("ignore")

DQ = _REPO / "outputs/data_quality"
OUT = DQ / "reports/report4a_v3/behavior_info"
DATA = Path(os.environ["HIPPO_DATA_ROOT"])

TRAIN_FRAC, GAP_S, W_S = 0.80, 1.0, 0.25
SPEED_LOOKBACK_S = 0.25  # Report 2 v2 primary speed
N_NULL, N_SUBSAMPLES = 20, 20
N_PRIMARY = 20
NS = (10, 20, 40)
ALPHAS = np.logspace(-2, 3, 8)
MIN_CLASS_FRAC, MIN_REGION_UNITS = 0.05, 10
R2V2_SHA = "e139ed0955f75ea03209a4973df7ba759ab65302"
SPEED_SOURCE = "pos_backward_250ms"
HD_SHARED = "movement_direction"
SIM_TODO = (
    "sim has no head/movement dissociation (sim HD = movement heading); "
    "lab head direction differs from movement direction by a median ~45°"
)


def _animal(session: str) -> str:
    return str(session).split("_")[0]


def dist_to_polygon_edges(xy, poly):
    from matplotlib.path import Path as MplPath

    xy = np.asarray(xy, float)
    poly = np.asarray(poly, float)
    if len(poly) and not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    dmin = np.full(len(xy), np.inf)
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = b - a
        L2 = float(np.dot(ab, ab))
        if L2 < 1e-18:
            d = np.linalg.norm(xy - a, axis=1)
        else:
            t = np.clip(((xy - a) @ ab) / L2, 0.0, 1.0)
            d = np.linalg.norm(xy - (a + t[:, None] * ab), axis=1)
        dmin = np.minimum(dmin, d)
    inside = MplPath(poly).contains_points(xy)
    return np.where(inside, dmin, 0.0)


def speed_250ms_backward(xy, times, lookback_s: float = SPEED_LOOKBACK_S):
    """Report 2 v2 primary speed: hypot(ΔX,ΔY)/Δt over backward lookback_s on decode X/Y."""
    xy = np.asarray(xy, float)
    times = np.asarray(times, float)
    j = np.searchsorted(times, times - lookback_s, side="right") - 1
    j = np.clip(j, 0, len(times) - 1)
    dt = times - times[j]
    ok = (j < np.arange(len(times))) & (dt >= 0.8 * lookback_s) & np.isfinite(xy).all(1)
    dxy = xy - xy[j]
    speed = np.full(len(times), np.nan)
    move_dir = np.full(len(times), np.nan)
    speed[ok] = np.linalg.norm(dxy[ok], axis=1) / np.maximum(dt[ok], 1e-9)
    move_dir[ok] = np.arctan2(dxy[ok, 1], dxy[ok, 0])
    # fill leading NaNs with first valid (decode masks handle finite)
    if np.isfinite(speed).any():
        first = int(np.flatnonzero(np.isfinite(speed))[0])
        speed[:first] = speed[first]
        move_dir[:first] = move_dir[first]
        # interpolate remaining gaps
        idx = np.arange(len(speed))
        good = np.isfinite(speed)
        speed = np.interp(idx, idx[good], speed[good])
        # circular-ish fill for direction via sin/cos
        s = np.sin(move_dir)
        c = np.cos(move_dir)
        s = np.interp(idx, idx[good], s[good])
        c = np.interp(idx, idx[good], c[good])
        move_dir = np.arctan2(s, c)
    else:
        speed[:] = 0.0
        move_dir[:] = 0.0
    frame_dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.05
    accel = np.diff(speed, prepend=speed[0]) / max(frame_dt, 1e-9)
    return speed, accel, move_dir


def net_above_null(raw: float, null_mean: float) -> float:
    """Net information: raw − max(null, 0), capped at 1."""
    if not np.isfinite(raw):
        return float("nan")
    baseline = max(float(null_mean) if np.isfinite(null_mean) else 0.0, 0.0)
    return float(min(1.0, raw - baseline))


def wall_bin_2(dist):
    lab = np.full(len(dist), "far", dtype=object)
    lab[dist < 10.0] = "near"
    return lab


def class_ok(y):
    y = np.asarray(y)
    classes = sorted(set(y.tolist()))
    if len(classes) < 2:
        return False
    return all(float(np.mean(y == c)) >= MIN_CLASS_FRAC for c in classes)


def score_continuous(y_true, y_pred, y_train):
    y_true, y_pred, y_train = map(lambda a: np.asarray(a, float), (y_true, y_pred, y_train))
    out = {}
    if y_true.ndim == 1:
        out["r2"] = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan")
        err = np.abs(y_true - y_pred)
        floor = np.abs(y_true - np.median(y_train))
    else:
        r2s = [float(r2_score(y_true[:, j], y_pred[:, j])) for j in range(y_true.shape[1])]
        out["r2_per_dim"] = r2s
        out["r2"] = float(np.mean(r2s))
        if y_true.shape[1] == 2:
            out["r2_x"], out["r2_y"] = r2s[0], r2s[1]
        err = np.linalg.norm(y_true - y_pred, axis=1)
        floor = np.linalg.norm(y_true - y_train.mean(0), axis=1)
    out["median_abs_err"] = float(np.median(err))
    out["err_over_floor"] = float(np.median(err) / max(float(np.median(floor)), 1e-9))
    return out


def score_direction(true_ang, pred_sin, pred_cos, train_ang):
    pred_ang = np.arctan2(pred_sin, pred_cos)
    med_err = float(np.median(circular_error_deg(true_ang, pred_ang)))
    cm = float(np.arctan2(np.mean(np.sin(train_ang)), np.mean(np.cos(train_ang))))
    floor = float(np.median(circular_error_deg(true_ang, np.full_like(true_ang, cm))))
    r2_s = float(r2_score(np.sin(true_ang), pred_sin))
    r2_c = float(r2_score(np.cos(true_ang), pred_cos))
    return {
        "r2": 0.5 * (r2_s + r2_c),
        "r2_sin": r2_s,
        "r2_cos": r2_c,
        "median_ang_err_deg": med_err,
        "ang_err_over_floor": med_err / max(floor, 1e-9),
        "chance_ang_err_deg": floor,
    }


def score_categorical(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = max(len(set(y_true.tolist())), 2)
    chance = 1.0 / k
    ba = float(balanced_accuracy_score(y_true, y_pred))
    return {
        "balanced_accuracy": ba,
        "info": (ba - chance) / max(1.0 - chance, 1e-9),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "chance": chance,
        "n_classes": k,
    }


def pick_ridge_alpha(Xtr, ytr):
    if ytr.ndim == 1:
        m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr)
        return float(m.alpha_)
    alphas = []
    for j in range(ytr.shape[1]):
        m = RidgeCV(alphas=ALPHAS).fit(Xtr, ytr[:, j])
        alphas.append(float(m.alpha_))
    return float(np.median(alphas))


def ridge_predict(Xtr, ytr, Xte, alpha):
    if ytr.ndim == 1:
        return Ridge(alpha=alpha).fit(Xtr, ytr).predict(Xte)
    preds = [Ridge(alpha=alpha).fit(Xtr, ytr[:, j]).predict(Xte) for j in range(ytr.shape[1])]
    return np.column_stack(preds)


def pick_logistic_C(Xtr, ytr):
    le = LabelEncoder()
    y = le.fit_transform(ytr)
    if len(le.classes_) < 2:
        return 1.0, le
    n_splits = min(3, int(np.min(np.bincount(y))))
    if n_splits < 2:
        return 1.0, le
    best_C, best = 1.0, -np.inf
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    for a in ALPHAS:
        C = 1.0 / float(a)
        sc = []
        for tr, va in kf.split(Xtr, y):
            clf = LogisticRegression(penalty="l2", C=C, max_iter=300, solver="lbfgs")
            try:
                clf.fit(Xtr[tr], y[tr])
                sc.append(balanced_accuracy_score(y[va], clf.predict(Xtr[va])))
            except Exception:
                sc.append(0.0)
        m = float(np.mean(sc)) if sc else -np.inf
        if m > best:
            best, best_C = m, C
    return best_C, le


def logistic_predict(Xtr, ytr, Xte, C, le):
    y = le.transform(ytr)
    clf = LogisticRegression(penalty="l2", C=C, max_iter=400, solver="lbfgs")
    clf.fit(Xtr, y)
    return le.inverse_transform(clf.predict(Xte))


def _bspline_basis(x, n_knots=5, degree=3):
    x = np.asarray(x, float)
    finite = x[np.isfinite(x)]
    lo, hi = (np.nanpercentile(finite, [2, 98]) if finite.size else (0.0, 1.0))
    if hi <= lo:
        hi = lo + 1.0
    knots_int = np.linspace(lo, hi, n_knots)
    knots = np.r_[[lo] * degree, knots_int, [hi] * degree]
    n_basis = len(knots) - degree - 1
    B = np.zeros((len(x), n_basis))
    for i in range(n_basis):
        c = np.zeros(n_basis)
        c[i] = 1.0
        B[:, i] = BSpline(knots, c, degree, extrapolate=True)(np.clip(x, lo, hi))
    return B


def _rbf_position(xy, xmin, xmax, ymin, ymax, n=6):
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    centers = np.column_stack([xx.ravel(), yy.ravel()])
    sx = (xmax - xmin) / max(n - 1, 1)
    sy = (ymax - ymin) / max(n - 1, 1)
    sigma2 = (0.5 * (sx + sy)) ** 2 + 1e-9
    d2 = ((xy[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / sigma2)


def build_encoding_features(beh, drop=None):
    drop = drop or set()
    parts = []
    xmin, xmax, ymin, ymax = beh["bbox"]
    if "position" not in drop:
        parts.append(_rbf_position(beh["xy"], xmin, xmax, ymin, ymax))
    if "wall" not in drop:
        parts.append(_bspline_basis(beh["wall"]))
    if "direction" not in drop:
        th = beh["move_dir"]
        parts.append(np.column_stack([np.sin(th), np.cos(th), np.sin(2 * th), np.cos(2 * th)]))
    if "speed" not in drop:
        parts.append(_bspline_basis(beh["speed"]))
    if "accel" not in drop:
        parts.append(_bspline_basis(beh["accel"]))
    if "movement_state" not in drop:
        ms = beh["movement_state"]
        for lab in ("still", "slow", "fast"):
            parts.append((ms == lab).astype(float)[:, None])
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(beh["xy"]), 1))


def pop_ve(Y, Yhat):
    mu = Y.mean(0, keepdims=True)
    sst = float(np.sum((Y - mu) ** 2))
    sse = float(np.sum((Y - Yhat) ** 2))
    return float(1.0 - sse / sst) if sst > 1e-12 else float("nan")


def encode_once(X, Y, train, test, alpha=None):
    Xtr, Ytr = X[train], Y[train]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtrz, Xtez = (Xtr - mu) / sd, (X[test] - mu) / sd
    if alpha is None:
        # cheap alpha: mid of grid on subsample of units
        u = min(Ytr.shape[1], 20)
        alpha = pick_ridge_alpha(Xtrz, Ytr[:, :u])
    pred = Ridge(alpha=alpha).fit(Xtrz, Ytr).predict(Xtez)
    return pop_ve(Y[test], pred), float(alpha)


def reweight_indices(values, ref, n_bins=20, rng=None):
    rng = rng or np.random.default_rng(0)
    values, ref = np.asarray(values, float), np.asarray(ref, float)
    ref = ref[np.isfinite(ref)]
    ok = np.isfinite(values)
    idx_ok = np.flatnonzero(ok)
    if idx_ok.size < 10 or ref.size < 10:
        return idx_ok
    lo = min(np.nanpercentile(values[ok], 1), np.nanpercentile(ref, 1))
    hi = max(np.nanpercentile(values[ok], 99), np.nanpercentile(ref, 99))
    if hi <= lo:
        return idx_ok
    bins = np.linspace(lo, hi, n_bins + 1)
    ref_h, _ = np.histogram(np.clip(ref, lo, hi), bins=bins)
    ref_p = ref_h / max(ref_h.sum(), 1)
    dig = np.digitize(values[idx_ok], bins[1:-1])
    w = np.zeros(len(idx_ok))
    for b in range(n_bins):
        m = dig == b
        if m.any():
            w[m] = ref_p[b] / m.sum()
    if w.sum() <= 0:
        return idx_ok
    w /= w.sum()
    return idx_ok[rng.choice(len(idx_ok), size=len(idx_ok), replace=True, p=w)]


def build_beh(xy, times, poly, bbox, hd_table=None, speed_table=None):
    """xy = decode positions (cm), unsmoothed — same frame as position target."""
    xy = np.asarray(xy, float)
    speed, accel, move_dir = speed_250ms_backward(xy, times)
    wall = dist_to_polygon_edges(xy, poly)
    xmin, xmax, ymin, ymax = bbox
    return dict(
        xy=xy, times=times, speed=speed, accel=accel, wall=wall,
        spatial_context=classify_spatial_context(xy[:, 0], xy[:, 1], xmin, xmax, ymin, ymax),
        movement_state=classify_movement_state(speed),
        move_dir=move_dir,
        wall_bin3=classify_wall_distance_bin(wall),
        wall_bin2=wall_bin_2(wall),
        poly=poly, bbox=bbox, hd_table=hd_table,
        speed_table=speed_table,  # lab V / sim table speed — sensitivity only
    )


def load_lab(session):
    bundle = build_segment_bundle_dq(session, "A")
    times = np.asarray(bundle["decode_times"], float)
    y = np.asarray(bundle["y"], float)
    X = np.asarray(bundle["X_counts"], float)
    valid = np.asarray(bundle["target_valid"], bool)
    units_df = bundle["units_df"]
    regions = units_df["region"].astype(str).to_numpy()
    keep = stratify_lab_units(regions, set_name="primary")
    X, regions = X[:, keep], regions[keep]
    retained = segment_retained_mask(times, float(bundle["segment_t0"]), float(bundle["segment_t1"]))
    pos = pd.read_csv(DATA / session / "postions_dataset.csv")
    ts_pos = pos["timestamp"].to_numpy(float)
    idx = np.clip(np.searchsorted(ts_pos, times, side="left"), 0, len(ts_pos) - 1)
    prev = np.clip(idx - 1, 0, len(ts_pos) - 1)
    idx = np.where(np.abs(ts_pos[prev] - times) < np.abs(ts_pos[idx] - times), prev, idx)
    hd = np.deg2rad(pos["HD"].to_numpy(float)[idx])
    hd_ok = np.isfinite(hd) & pos["valid"].fillna(False).to_numpy()[idx]
    v_tab = pos["V"].to_numpy(float)[idx] if "V" in pos.columns else None
    cfg = yaml.safe_load((DATA / session / "config.yaml").read_text())
    poly = boundary_polygon_xy(cfg, "A")
    cx, cy, bw, bh, xmin_g, ymin_g = _room_boundary(cfg, "A")
    bbox = (xmin_g - cx, xmin_g + bw - cx, ymin_g - cy, ymin_g + bh - cy)
    mask = retained & valid & np.isfinite(y).all(1)
    times, y, X, hd, hd_ok = times[mask], y[mask], X[mask], hd[mask], hd_ok[mask]
    if v_tab is not None:
        v_tab = v_tab[mask]
    beh = build_beh(
        y, times, poly, bbox,
        hd_table=np.where(hd_ok, hd, np.nan),
        speed_table=v_tab,
    )
    return dict(dataset="lab", session=session, animal=_animal(session), X=X, regions=regions, beh=beh, times=times)


def load_sim(sim_dir, dataset, session, arena_size_cm):
    sim_dir = Path(sim_dir)
    units = add_mapped_lab_region(pd.read_csv(sim_dir / "units.csv"))
    primary = units[units["lab_region"].isin(["MEC", "SUB"])].reset_index(drop=True)
    if len(primary) == 0:
        primary = units.reset_index(drop=True)
    sp = pd.read_csv(sim_dir / "spikes_sorted.csv")
    spikes = [sp.loc[sp["unit_id"] == uid, "spike_time_s"].to_numpy(float) for uid in primary["unit_id"]]
    regions = primary["lab_region"].astype(str).to_numpy()
    beh_df = pd.read_csv(sim_dir / "behavior.csv")
    times = beh_df["time_s"].to_numpy(float)
    xy = np.column_stack([beh_df["x_cm"], beh_df["y_cm"]]).astype(float)
    S = float(arena_size_cm)
    xy = xy - S / 2.0
    hw = S / 2.0
    poly, bbox = square_polygon(hw), (-hw, hw, -hw, hw)
    X = causal_count_matrix(spikes, times, window_s=W_S).astype(float)
    valid = np.isfinite(xy).all(1)
    if times.max() - times.min() > 200:
        retained = (times >= times.min() + 60.0) & (times < times.max() - 10.0)
    else:
        retained = np.ones(len(times), bool)
    mask = retained & valid
    times, xy, X = times[mask], xy[mask], X[mask]
    hd = beh_df["head_direction_rad"].to_numpy(float)[mask] if "head_direction_rad" in beh_df.columns else None
    v_tab = beh_df["speed_cm_s"].to_numpy(float)[mask] if "speed_cm_s" in beh_df.columns else None
    beh = build_beh(xy, times, poly, bbox, hd_table=hd, speed_table=v_tab)
    return dict(dataset=dataset, session=session, animal=session if dataset == "sim_default" else _animal(session),
                X=X, regions=regions, beh=beh, times=times)


def decode_target(Xz, train, test, y, kind, alpha=None, C=None, le=None, score_mask=None):
    tr = train if score_mask is None else (train & score_mask)
    te = test if score_mask is None else (test & score_mask)
    if kind == "direction":
        tr = train & score_mask if score_mask is not None else train
        te = test & score_mask if score_mask is not None else test
    if tr.sum() < 40 or te.sum() < 25:
        return None, alpha, C, le
    if kind in ("continuous_1d", "continuous_2d"):
        if alpha is None:
            alpha = pick_ridge_alpha(Xz[tr], y[tr])
        pred = ridge_predict(Xz[tr], y[tr], Xz[te], alpha)
        return score_continuous(y[te], pred, y[tr]), alpha, C, le
    if kind == "direction":
        Y = np.column_stack([np.sin(y), np.cos(y)])
        if alpha is None:
            alpha = pick_ridge_alpha(Xz[tr], Y[tr])
        pred = ridge_predict(Xz[tr], Y[tr], Xz[te], alpha)
        return score_direction(y[te], pred[:, 0], pred[:, 1], y[tr]), alpha, C, le
    if kind == "categorical":
        if not class_ok(y[tr]) or not class_ok(y[te]):
            return None, alpha, C, le
        if C is None or le is None:
            C, le = pick_logistic_C(Xz[tr], y[tr])
        pred = logistic_predict(Xz[tr], y[tr], Xz[te], C, le)
        return score_categorical(y[te], pred), alpha, C, le
    return None, alpha, C, le


def _null_stats(null_vals):
    a = np.asarray(null_vals, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0.0, 0.0, 0.0
    return float(np.mean(a)), float(np.median(a)), float(np.percentile(a, 95))


def _decode_at_N(X_raw, train, test, specs, N, n_rep, shifts, rng):
    """Decode all specs at matched N; return list of rows with null stats."""
    n_u = X_raw.shape[1]
    if N > n_u:
        return []
    n_rep = 1 if N >= n_u else int(n_rep)
    # fixed unit draw for null distribution + alpha
    idx0 = rng.choice(n_u, size=N, replace=False)
    Xz0, _ = _sqrt_zscore_train(X_raw[:, idx0], train)
    rows = []
    for var, kind, y, smask in specs:
        key = "info" if kind == "categorical" else "r2"
        # point estimate: median over unit subsamples
        raws, extras = [], []
        alpha = C = le = None
        for r_i in range(n_rep):
            idx = idx0 if r_i == 0 else rng.choice(n_u, size=N, replace=False)
            Xz, _ = _sqrt_zscore_train(X_raw[:, idx], train)
            sc, alpha, C, le = decode_target(
                Xz, train, test, y, kind, alpha=alpha if r_i else None,
                C=C if r_i else None, le=le if r_i else None, score_mask=smask,
            )
            if sc is not None and np.isfinite(sc.get(key, np.nan)):
                raws.append(float(sc[key]))
                extras.append(sc)
        if not raws:
            continue
        raw = float(np.median(raws))
        # null on fixed unit set
        null_vals = []
        for sh in shifts:
            sn, _, _, _ = decode_target(
                np.roll(Xz0, sh, axis=0), train, test, y, kind,
                alpha=alpha, C=C, le=le, score_mask=smask,
            )
            if sn is not None and np.isfinite(sn.get(key, np.nan)):
                null_vals.append(float(sn[key]))
        n_mean, n_med, n_p95 = _null_stats(null_vals)
        net = net_above_null(raw, n_mean)
        # carry secondary scores from median-raw draw
        sc0 = extras[int(np.argmin(np.abs(np.asarray(raws) - raw)))]
        row = dict(sc0)
        row.update(
            variable=var, kind=kind, N=int(N),
            primary_metric=key, primary_value=raw,
            null_mean=n_mean, null_median=n_med, null_p95=n_p95,
            primary_minus_null=net,
            alpha=alpha if alpha is not None else C,
        )
        rows.append(row)
    return rows


def run_one(pack, pools=None, do_ncurve=True, do_regions=True, do_sens=True):
    pools = pools or {}
    X_raw, beh, regions, times = pack["X"], pack["beh"], pack["regions"], pack["times"]
    n_u = X_raw.shape[1]
    if n_u < 5 or len(times) < 120:
        return {"dataset": pack["dataset"], "session": pack["session"], "status": "too_small"}
    train, test = causal_train_test_split(times, TRAIN_FRAC, gap_s=GAP_S)
    moving = beh["speed"] > 2.0
    use_wbin3 = class_ok(beh["wall_bin3"][train]) and class_ok(beh["wall_bin3"][test])

    specs = [
        ("position", "continuous_2d", beh["xy"], None),
        ("distance_to_wall", "continuous_1d", beh["wall"], None),
        ("speed", "continuous_1d", beh["speed"], None),
        ("acceleration", "continuous_1d", beh["accel"], None),
        ("movement_direction", "direction", beh["move_dir"], moving),
        ("wall_distance_bin2", "categorical", beh["wall_bin2"], None),
        ("spatial_context", "categorical", beh["spatial_context"], None),
        ("movement_state", "categorical", beh["movement_state"], None),
    ]
    if use_wbin3:
        specs.append(("wall_distance_bin3", "categorical", beh["wall_bin3"], None))
    if pack["dataset"] == "lab" and beh.get("hd_table") is not None:
        hd = beh["hd_table"]
        hd_ok = np.isfinite(hd) & moving
        if hd_ok[train].sum() > 50 and hd_ok[test].sum() > 20:
            specs.append(("head_direction", "direction", np.nan_to_num(hd, nan=0.0), hd_ok))

    n = len(times)
    shifts = [int((s + 1) * n / (N_NULL + 1)) for s in range(N_NULL)]
    rng = np.random.default_rng(abs(hash(pack["session"])) % (2**32))

    decode_rows = []
    # Primary N=20
    if n_u >= N_PRIMARY:
        for row in _decode_at_N(X_raw, train, test, specs, N_PRIMARY, N_SUBSAMPLES, shifts, rng):
            row.update(
                dataset=pack["dataset"], session=pack["session"], animal=pack["animal"],
                region="pooled", N_role="primary",
            )
            decode_rows.append(row)
    # Secondary N=max
    for row in _decode_at_N(X_raw, train, test, specs, n_u, 1, shifts, rng):
        row.update(
            dataset=pack["dataset"], session=pack["session"], animal=pack["animal"],
            region="pooled", N_role="secondary_max",
        )
        decode_rows.append(row)

    # N curves (raw R², no null) for position / movement_direction / speed
    n_curve = []
    if do_ncurve:
        for N in list(NS) + [n_u]:
            if N > n_u:
                continue
            n_rep = N_SUBSAMPLES if N < n_u else 1
            for var, kind, y, smask in (
                ("position", "continuous_2d", beh["xy"], None),
                ("movement_direction", "direction", beh["move_dir"], moving),
                ("speed", "continuous_1d", beh["speed"], None),
            ):
                vals = []
                for _ in range(n_rep):
                    idx = rng.choice(n_u, size=N, replace=False)
                    Xnz, _ = _sqrt_zscore_train(X_raw[:, idx], train)
                    sc, _, _, _ = decode_target(Xnz, train, test, y, kind, score_mask=smask)
                    if sc is not None:
                        vals.append(sc["r2"])
                if vals:
                    n_curve.append(dict(
                        dataset=pack["dataset"], session=pack["session"], animal=pack["animal"],
                        variable=var, N=int(N), value=float(np.median(vals)),
                        value_q25=float(np.percentile(vals, 25)), value_q75=float(np.percentile(vals, 75)),
                    ))

    region_rows = []
    if do_regions:
        for reg in ("MEC", "SUB"):
            m = regions == reg
            n_reg = int(m.sum())
            if n_reg < MIN_REGION_UNITS:
                continue
            N_use = min(N_PRIMARY, n_reg)
            for row in _decode_at_N(
                X_raw[:, m], train, test,
                [s for s in specs if s[0] in ("position", "movement_direction", "speed", "wall_distance_bin2")],
                N_use, N_SUBSAMPLES if N_use < n_reg else 1, shifts, rng,
            ):
                row.update(
                    dataset=pack["dataset"], session=pack["session"], animal=pack["animal"],
                    region=reg, N_role="primary" if N_use == N_PRIMARY else "region_max",
                )
                region_rows.append(row)

    # encoding at N=20 (primary) and N=max
    Fe = build_encoding_features(beh)
    groups = ["position", "wall", "direction", "speed", "accel", "movement_state"]

    def _encode_N(N):
        if N > n_u:
            return None
        idx = rng.choice(n_u, size=N, replace=False) if N < n_u else np.arange(n_u)
        Ys = np.sqrt(np.maximum(X_raw[:, idx], 0.0))
        mu_y, sd_y = Ys[train].mean(0), Ys[train].std(0) + 1e-6
        Yz = (Ys - mu_y) / sd_y
        ve_full, a_enc = encode_once(Fe, Yz, train, test)
        unique = {}
        for g in groups:
            ve_lo, _ = encode_once(build_encoding_features(beh, drop={g}), Yz, train, test, alpha=a_enc)
            unique[g] = float(ve_full - ve_lo)
        shared = float(ve_full - sum(unique.values()))
        null_ve = []
        for sh in shifts:
            null_ve.append(encode_once(np.roll(Fe, sh, axis=0), Yz, train, test, alpha=a_enc)[0])
        n_mean, n_med, n_p95 = _null_stats(null_ve)
        return dict(
            dataset=pack["dataset"], session=pack["session"], animal=pack["animal"],
            N=int(N), N_role="primary" if N == N_PRIMARY else "secondary_max",
            ve_full=ve_full,
            null_mean=n_mean, null_median=n_med, null_p95=n_p95,
            ve_full_minus_null=net_above_null(ve_full, n_mean),
            alpha=a_enc, shared=shared, n_units=int(n_u),
            **{f"unique_{g}": unique[g] for g in groups},
        )

    enc_rows = []
    if n_u >= N_PRIMARY:
        e = _encode_N(N_PRIMARY)
        if e:
            enc_rows.append(e)
    e = _encode_N(n_u)
    if e:
        enc_rows.append(e)

    # sensitivity at N=20
    sens = []
    N_sens = min(N_PRIMARY, n_u)
    idx_sens = rng.choice(n_u, size=N_sens, replace=False)
    Xz_sens, _ = _sqrt_zscore_train(X_raw[:, idx_sens], train)
    if do_sens and pools:
        rng_s = np.random.default_rng(3)
        for var, kind, y, ref_key in (
            ("distance_to_wall", "continuous_1d", beh["wall"], "wall"),
            ("speed", "continuous_1d", beh["speed"], "speed"),
            ("acceleration", "continuous_1d", beh["accel"], "accel"),
            ("position", "continuous_2d", beh["xy"], "radial"),
        ):
            ref = pools.get(ref_key)
            if ref is None:
                continue
            if var == "position":
                xmin, xmax, ymin, ymax = beh["bbox"]
                cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
                local = np.hypot(beh["xy"][:, 0] - cx, beh["xy"][:, 1] - cy)[test]
            else:
                local = y[test]
            idx_local = reweight_indices(local, ref, rng=rng_s)
            te_idx = np.flatnonzero(test)[idx_local]
            alpha = pick_ridge_alpha(Xz_sens[train], y[train])
            pred = ridge_predict(Xz_sens[train], y[train], Xz_sens[te_idx], alpha)
            sc = score_continuous(y[te_idx], pred, y[train])
            sc.update(variable=var, dataset=pack["dataset"], session=pack["session"],
                      animal=pack["animal"], reweighted=True, N=int(N_sens))
            sens.append(sc)
    # lab V / sim table speed sensitivity (vs primary 250 ms speed)
    if beh.get("speed_table") is not None and np.isfinite(beh["speed_table"]).sum() > 50:
        yv = beh["speed_table"].astype(float)
        ok = np.isfinite(yv)
        tr_v, te_v = train & ok, test & ok
        if tr_v.sum() > 40 and te_v.sum() > 20:
            sc, _, _, _ = decode_target(Xz_sens, tr_v, te_v, yv, "continuous_1d")
            if sc is not None:
                sc.update(
                    variable="speed_table_V", dataset=pack["dataset"], session=pack["session"],
                    animal=pack["animal"], reweighted=False, N=int(N_sens),
                    note="sensitivity: tracker V / behavior.csv speed_cm_s",
                )
                sens.append(sc)

    xmin, xmax, ymin, ymax = beh["bbox"]
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    out = dict(
        dataset=pack["dataset"], session=pack["session"], animal=pack["animal"],
        status="ok", n_units=int(n_u), use_wall_bin3=bool(use_wbin3),
        decode=decode_rows, region_decode=region_rows, n_curve=n_curve,
        encoding_rows=enc_rows, sensitivity=sens,
        _speed=beh["speed"], _wall=beh["wall"], _accel=beh["accel"],
        _radial=np.hypot(beh["xy"][:, 0] - cx, beh["xy"][:, 1] - cy),
    )
    hd_rows = [r for r in decode_rows if r["variable"] == "head_direction" and r.get("N_role") == "primary"]
    md_rows = [r for r in decode_rows if r["variable"] == "movement_direction" and r.get("N_role") == "primary"]
    if hd_rows and md_rows:
        out["hd_vs_move"] = {
            "head_r2_minus_null": hd_rows[0]["primary_minus_null"],
            "move_r2_minus_null": md_rows[0]["primary_minus_null"],
            "delta_r2": float(hd_rows[0]["primary_minus_null"] - md_rows[0]["primary_minus_null"]),
            "head_raw": hd_rows[0]["primary_value"], "move_raw": md_rows[0]["primary_value"],
            "head_null_mean": hd_rows[0]["null_mean"], "move_null_mean": md_rows[0]["null_mean"],
            "head_ang_err": hd_rows[0].get("median_ang_err_deg"),
            "move_ang_err": md_rows[0].get("median_ang_err_deg"),
            "N": int(hd_rows[0]["N"]),
        }
    return out


def iter_jobs(pilot=None):
    sess_df = pd.read_csv(DQ / "reports/report4a_v3/data/data_session_metrics.csv")
    labs = sorted(sess_df.loc[sess_df.dataset == "lab", "session"].unique())
    if pilot:
        labs = labs[: int(pilot)]
    for s in labs:
        yield ("lab", s, None, None)
        ms = json.loads((DQ / "matched" / f"{s}_A" / "match_spec.json").read_text())
        yield ("sim_matched", s, str(DQ / "matched" / f"{s}_A" / "sim"), float(ms["arena_size_cm"]))
    seeds = range(1) if pilot else range(5)
    for seed in seeds:
        yield ("sim_default", f"seed_{seed}", str(DQ / "sim_inputs" / f"seed_{seed}" / "sim"), 100.0)


def _worker(payload):
    """Picklable worker for ProcessPoolExecutor (pools loaded from .npz path)."""
    kind, session, sim_dir, arena, pools_path, do_ncurve = payload
    try:
        blob = np.load(pools_path)
        pools = {k: blob[k] for k in blob.files}
        if kind == "lab":
            pack = load_lab(session)
        else:
            pack = load_sim(sim_dir, kind, session, arena)
        r = run_one(pack, pools=pools, do_ncurve=do_ncurve, do_sens=True)
        for k in list(r.keys()):
            if k.startswith("_"):
                del r[k]
        return r
    except Exception as e:
        return {"dataset": kind, "session": session, "status": "error",
                "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--skip-ncurve", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = list(iter_jobs(args.pilot or None))
    print(json.dumps({"n_jobs": len(jobs), "pilot": args.pilot, "workers": args.workers}), flush=True)

    # Build lab pools first
    t0 = time.perf_counter()
    pools = {}
    for kind, session, sim_dir, arena in jobs:
        if kind != "lab":
            continue
        pack = load_lab(session)
        beh = pack["beh"]
        xmin, xmax, ymin, ymax = beh["bbox"]
        cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        for key, arr in (
            ("speed", beh["speed"]), ("wall", beh["wall"]), ("accel", beh["accel"]),
            ("radial", np.hypot(beh["xy"][:, 0] - cx, beh["xy"][:, 1] - cy)),
        ):
            pools.setdefault(key, []).append(arr)
    pools = {k: np.concatenate(v) for k, v in pools.items()}
    # subsample pools for reweighting (histogram stable; keeps memory small)
    rng_pool = np.random.default_rng(0)
    pools_sub = {}
    for k, v in pools.items():
        v = np.asarray(v, float)
        if len(v) > 80000:
            v = v[rng_pool.choice(len(v), size=80000, replace=False)]
        pools_sub[k] = v
    pools = pools_sub
    pools_path = out / "lab_pools.npz"
    np.savez_compressed(pools_path, **pools)
    print(json.dumps({"phase": "pools_ready", "n_lab_pool_frames": int(len(pools.get("speed", []))),
                      "pools_path": str(pools_path),
                      "elapsed_s": round(time.perf_counter() - t0, 1)}), flush=True)

    payloads = [
        (kind, session, sim_dir, arena, str(pools_path), not args.skip_ncurve)
        for kind, session, sim_dir, arena in jobs
    ]
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

    sens_all = []
    for r in results:
        sens_all.extend(r.get("sensitivity") or [])

    decode_rows, ncurve, enc, hd_vs = [], [], [], []
    for r in results:
        decode_rows.extend(r.get("decode") or [])
        decode_rows.extend(r.get("region_decode") or [])
        ncurve.extend(r.get("n_curve") or [])
        enc.extend(r.get("encoding_rows") or [])
        if r.get("hd_vs_move"):
            hd_vs.append({"session": r["session"], **r["hd_vs_move"]})

    pd.DataFrame(decode_rows).to_csv(out / "decode_rows.csv", index=False)
    pd.DataFrame(ncurve).to_csv(out / "n_curve.csv", index=False)
    pd.DataFrame(enc).to_csv(out / "encoding_rows.csv", index=False)
    pd.DataFrame(sens_all).to_csv(out / "sensitivity_rows.csv", index=False)
    pd.DataFrame(hd_vs).to_csv(out / "lab_hd_vs_movement.csv", index=False)

    # tidy: primary N=20 only, net = raw − max(null,0) capped at 1
    tidy = []
    for r in decode_rows:
        if r.get("region") != "pooled" or r.get("N_role") != "primary":
            continue
        if not np.isfinite(r.get("primary_minus_null", np.nan)):
            continue
        tidy.append(dict(
            dataset=r["dataset"], session=r["session"],
            animal=r.get("animal", _animal(r["session"])),
            metric=f"info_{r['variable']}",
            value=r["primary_minus_null"],
            raw=r.get("primary_value"),
            null_mean=r.get("null_mean"),
            null_p95=r.get("null_p95"),
            N=r.get("N"),
            **{"animal|seed": r.get("animal", _animal(r["session"]))},
        ))
    tidy_df = pd.DataFrame(tidy)
    tidy_df.to_csv(out / "info_tidy.csv", index=False)

    def _lab_mad(vals):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 2:
            return float("nan")
        med = float(np.median(vals))
        return float(np.median(np.abs(vals - med)) * 1.4826)

    fig3_rows = []
    if not tidy_df.empty:
        for m in sorted(tidy_df["metric"].unique()):
            for arm in ("sim_matched", "sim_default"):
                d = divergence_for_metric(tidy_df, metric=m, sim_dataset=arm)
                if d.get("status") != "ok":
                    continue
                lab_vals = tidy_df[(tidy_df.dataset == "lab") & (tidy_df.metric == m)]["value"]
                sim_vals = tidy_df[(tidy_df.dataset == arm) & (tidy_df.metric == m)]["value"]
                lab_med = float(lab_vals.median())
                sim_med = float(sim_vals.median())
                lab_mad = _lab_mad(lab_vals)
                fig3_rows.append({
                    "metric": m, "family": "information", "arm": arm,
                    "R": d.get("R"), "R_lo90": d.get("R_lo90"), "R_hi90": d.get("R_hi90"),
                    "lab_median": lab_med, "sim_median": sim_med,
                    "lab_mad": lab_mad,
                    "lead_with_abs": bool(np.isfinite(lab_mad) and lab_mad < 0.02),
                    "N": int(N_PRIMARY),
                    "label": m.replace("info_", "").replace("_", " "),
                })
    pd.DataFrame(fig3_rows).to_csv(out / "fig3_information.csv", index=False)

    (out / "decisions.json").write_text(json.dumps({
        "hd_convention": {
            "raw_session_median_abs_deg": 46.5,
            "residual_median_abs_deg": 45.0,
            "conclusion": "not convention mismatch → shared=movement_direction; lab-only=head_direction",
        },
        "speed": {
            "primary": "250 ms backward |Δ(X,Y)|/Δt on decode X/Y (Report 2 v2)",
            "sensitivity": "lab V / sim speed_cm_s",
            "feeds": ["acceleration", "movement_state", "movement_direction"],
        },
        "null_correction": "net = min(1, raw − max(null_mean, 0)); also report raw, null_mean, null_p95",
        "N_primary": N_PRIMARY,
        "hippo_r2v2_commit": R2V2_SHA,
        "hippo_r2v2_note": "Recheck target definitions when Report 2 v2 is final.",
        "imports_from_hippo_r2v2": [
            "classify_spatial_context", "classify_movement_state",
            "classify_wall_distance_bin", "circular_error_deg",
        ],
        "sim_todo": SIM_TODO,
        "split": {"train_frac": TRAIN_FRAC, "gap_s": GAP_S, "W_s": W_S, "sqrt_zscore": True},
    }, indent=2))

    summary = {
        "wrote": str(out),
        "n_results": len(results),
        "n_ok": sum(1 for r in results if r.get("status") == "ok"),
        "n_err": sum(1 for r in results if r.get("status") == "error"),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "errors": [{"session": r.get("session"), "error": r.get("error")} for r in results if r.get("status") == "error"][:10],
    }
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
