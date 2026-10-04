"""Phase 8 — kNN pressure test (post hoc; saved models + predictions).

No config change. No representation refits for analyses 1–3.
Refits ONLY Ridge / kNN with the saved α / k.

Writes under outputs/quadrant_n5/knn_pressure/:
  exclusion.csv, neighbour_diag.csv, neighbour_hist.npz,
  strata.csv, criteria.json, summary.json

Reduced analysis 4 (cell-type covariate): see
run_cell_type_covariate_reduced(). Blocked outer CV (analysis 6) remains
plan-only; see plan_blocked_outer_cv().
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from agents.quadrant_n5.export_predictions import (  # noqa: E402
    build_observation,
    latents_for_method,
)
from realtime.quadrant_n5 import load_quadrant_n5_yaml  # noqa: E402
from realtime.quadrant_n5_run import (  # noqa: E402
    OUTPUT_ROOT,
    _choose_alpha,
    _choose_k,
    _euclid,
    _fit_predict_knn,
    _fit_predict_ridge,
    _make_rep,
    _sqrt_zscore_train,
    fit_transform_representation,
    ridge_alpha_grid,
)
from realtime.quadrant_n5 import inner_cv_block_masks  # noqa: E402
from realtime.train_decoder import (  # noqa: E402
    align_behavior_to_decoder_times,
)

METHODS = ("raw_lag", "pca", "dm", "lds", "gpfa")
SOURCES = ("sorted", "ground_truth")
DELTAS_S = (0.0, 1.0, 5.0, 10.0, 30.0, 60.0)
ARENA_CM = 100.0
N_BINS = 10
LAST_TRAIN_S = 30.0  # neighbour diagnostic: last 30 s of training
HEADING_ATYPICAL_RAD = np.pi / 2.0  # > 90°

# Reduced analysis 4 — pre-registered criterion uses grid_bvc only.
CELL_TYPE_METHODS = ("pca", "lds")
CELL_TYPE_SOURCE = "sorted"
CELL_TYPE_SUBSETS = {
    "grid_bvc": ("MEC_grid", "Sub_bvc"),
    "hd_speed": ("MEC_hd", "MEC_speed"),
}

MECHANISM_SUBSETS: dict[str, tuple[str, ...] | None] = {
    **CELL_TYPE_SUBSETS,
    "all": None,
    "grid_bvc_hd": ("MEC_grid", "Sub_bvc", "MEC_hd"),
    "grid_bvc_speed": ("MEC_grid", "Sub_bvc", "MEC_speed"),
}

SHUFFLE_N_OFFSETS = 5
SHUFFLE_MIN_OFFSET_S = 60.0
CONTROL_DIMS = (5, 10, 20)
NOISE_N_UNITS = 45  # match HD+speed population size from N=5 sorted decoder pool


def preregister_control_criteria(results: Path | None = None) -> dict[str, Any]:
    """Write criteria (c) and (d) as PENDING before C1/C2 runs. Does not touch (a)/(b)."""
    results = Path(results or OUTPUT_ROOT)
    crit_path = _pressure_root(results) / "criteria.json"
    crit_path.parent.mkdir(parents=True, exist_ok=True)
    criteria = json.loads(crit_path.read_text()) if crit_path.exists() else {}
    criteria["control_c_channel_count"] = {
        "status": "PENDING",
        "rule": (
            "Channel-count effect supported if the LDS kNN gain from adding noise "
            "units (error(grid+BVC+noise)−error(grid+BVC)) is ≥ 50% of the intact "
            "HD+speed gain (error(all)−error(grid+BVC)) in ≥ 4/5 seeds"
        ),
        "noise_definition": (
            f"{NOISE_N_UNITS} synthetic units, Poisson with rates matched "
            "unit-for-unit to HD+speed mean train rates; fixed seed from methods "
            "stream; no temporal structure; joint sqrt-zscore with grid+BVC"
        ),
        "gain_ratio_definition": (
            "noise_gain / intact_gain ≥ 0.5 when intact_gain ≠ 0 "
            "(both typically negative when adding units helps)"
        ),
        "required": "≥ 4/5",
    }
    criteria["control_d_fixed_d"] = {
        "status": "PENDING",
        "rule": (
            "Fixed-d artifact supported if LDS kNN on grid+BVC at its best of "
            "d ∈ {5, 10} beats d = 20 by ≥ 2 cm in ≥ 4/5 seeds"
        ),
        "dims": list(CONTROL_DIMS),
        "required": "≥ 4/5",
        "threshold_cm": 2.0,
    }
    crit_path.write_text(json.dumps(criteria, indent=2) + "\n")
    print(json.dumps({"wrote": str(crit_path), "control_c_d": "PENDING"}, indent=2), flush=True)
    return criteria


def _observation_with_counts(
    cfg: dict[str, Any],
    sim_dir: Path,
    spike_source: str,
) -> dict[str, Any]:
    """Like build_observation, but also returns X_counts (pre sqrt-zscore)."""
    from realtime.data_loading import load_simulation_data, make_decode_times
    from realtime.spike_binner import build_causal_spike_matrix
    from realtime.timing import extract_behavior_times
    from realtime.train_decoder import causal_train_test_split
    from realtime.quadrant_n5_run import METHOD_KEYS, _position, _valid_mask
    from realtime.pipeline_artifacts import hash_train_indices

    feat = cfg["features"]
    split = cfg["split"]
    data = load_simulation_data(
        sim_dir, spike_source,
        include_regions=list(cfg["unit_inclusion"]["regions"]),
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
    beh = align_behavior_to_decoder_times(data["behavior_df"], decode_times)
    y = _position(beh)
    train_mask, test_mask = causal_train_test_split(
        decode_times, float(split["train_frac"]), gap_s=float(split["gap_s"]),
    )
    X, _ = _sqrt_zscore_train(X_counts, train_mask)
    n = len(decode_times)
    valid = {m: _valid_mask(m, n, cfg) for m in METHOD_KEYS}
    eval_mask = test_mask.copy()
    train_ok = train_mask.copy()
    for m in METHOD_KEYS:
        eval_mask &= valid[m]
        train_ok &= valid[m]
    return {
        "X": X,
        "X_counts": np.asarray(X_counts, dtype=float),
        "y": y,
        "decode_times": decode_times,
        "train_ok": train_ok,
        "eval_mask": eval_mask,
        "train_mask": train_mask,
        "unit_ids": list(data["unit_ids"]),
    }


def _build_grid_bvc_plus_noise(
    obs: dict[str, Any],
    results: Path,
    seed_index: int,
    cfg: dict[str, Any],
    methods_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """grid+BVC counts + Poisson noise matched to HD+speed train mean rates."""
    gb_cols, _ = _unit_column_indices(
        results, seed_index, CELL_TYPE_SOURCE, cfg, ("MEC_grid", "Sub_bvc"),
    )
    hd_cols, _ = _unit_column_indices(
        results, seed_index, CELL_TYPE_SOURCE, cfg, ("MEC_hd", "MEC_speed"),
    )
    Xc = obs["X_counts"]
    train = obs["train_ok"]
    rates = Xc[train][:, hd_cols].mean(axis=0)
    if len(rates) != NOISE_N_UNITS:
        # Pad/truncate to fixed 45 if population size differs across seeds (should not).
        if len(rates) >= NOISE_N_UNITS:
            rates = rates[:NOISE_N_UNITS]
        else:
            rates = np.concatenate([rates, np.full(NOISE_N_UNITS - len(rates), rates.mean())])
    rng = np.random.default_rng(int(methods_seed))
    noise = rng.poisson(lam=rates, size=(Xc.shape[0], NOISE_N_UNITS)).astype(float)
    X_joint = np.concatenate([Xc[:, gb_cols], noise], axis=1)
    X_z, _ = _sqrt_zscore_train(X_joint, obs["train_mask"])
    meta = {
        "n_grid_bvc": len(gb_cols),
        "n_noise": NOISE_N_UNITS,
        "hd_speed_rates_mean": float(np.mean(rates)),
        "hd_speed_rates_n": int(len(hd_cols)),
    }
    return X_z, meta


def evaluate_control_c(
    noise_df: pd.DataFrame,
    mechanism_df: pd.DataFrame,
) -> dict[str, Any]:
    """(c) channel-count: noise_gain / intact_gain ≥ 0.5 in ≥ 4/5 (LDS kNN)."""
    ratios: list[float] = []
    noise_gains: list[float] = []
    intact_gains: list[float] = []
    for s in range(5):
        gb = mechanism_df[
            (mechanism_df.seed == s) & (mechanism_df.subset == "grid_bvc")
            & (mechanism_df.method == "lds") & (mechanism_df.decoder == "knn")
        ]
        all_u = mechanism_df[
            (mechanism_df.seed == s) & (mechanism_df.subset == "all")
            & (mechanism_df.method == "lds") & (mechanism_df.decoder == "knn")
        ]
        nz = noise_df[
            (noise_df.seed == s) & (noise_df.method == "lds") & (noise_df.decoder == "knn")
        ]
        e_gb = float(gb.median_err.iloc[0])
        e_all = float(all_u.median_err.iloc[0])
        e_nz = float(nz.median_err.iloc[0])
        g_intact = e_all - e_gb
        g_noise = e_nz - e_gb
        intact_gains.append(g_intact)
        noise_gains.append(g_noise)
        if abs(g_intact) > 1e-9:
            ratios.append(g_noise / g_intact)
        else:
            ratios.append(float("nan"))
    n_pass = int(sum(r >= 0.5 for r in ratios if np.isfinite(r)))
    return {
        "status": "PASS" if n_pass >= 4 else "FAIL",
        "per_seed_intact_lds_knn_gain_cm": intact_gains,
        "per_seed_noise_lds_knn_gain_cm": noise_gains,
        "per_seed_noise_over_intact_ratio": ratios,
        "n_pass": n_pass,
        "required": "≥ 4/5",
        "threshold_ratio": 0.5,
    }


def evaluate_control_d(d_sweep_df: pd.DataFrame) -> dict[str, Any]:
    """(d) fixed-d: best(d=5,10) beats d=20 by ≥ 2 cm on LDS kNN grid_bvc."""
    deltas: list[float] = []
    best_ds: list[int] = []
    for s in range(5):
        sub = d_sweep_df[
            (d_sweep_df.seed == s) & (d_sweep_df.subset == "grid_bvc")
            & (d_sweep_df.method == "lds") & (d_sweep_df.decoder == "knn")
        ]
        e20 = float(sub[sub.d == 20].median_err.iloc[0])
        e5 = float(sub[sub.d == 5].median_err.iloc[0])
        e10 = float(sub[sub.d == 10].median_err.iloc[0])
        best_d = 5 if e5 <= e10 else 10
        best_e = min(e5, e10)
        # positive delta = best of {5,10} better (lower error) than d=20
        deltas.append(e20 - best_e)
        best_ds.append(best_d)
    n_pass = int(sum(d >= 2.0 for d in deltas if np.isfinite(d)))
    return {
        "status": "PASS" if n_pass >= 4 else "FAIL",
        "per_seed_err20_minus_best_low_d_cm": deltas,
        "per_seed_best_low_d": best_ds,
        "n_pass": n_pass,
        "required": "≥ 4/5",
        "threshold_cm": 2.0,
    }


def ridge_shuffle_remaining_report(
    mechanism_df: pd.DataFrame,
    shuffle_df: pd.DataFrame,
) -> dict[str, Any]:
    """Fraction of LDS Ridge gain remaining after HD+speed shuffle (report only)."""
    gains = _gain_table(mechanism_df)
    shuf = _shuffle_gain_table(shuffle_df, mechanism_df)
    intact: list[float] = []
    shuffled: list[float] = []
    remaining: list[float] = []
    for s in range(5):
        gi = float(gains[
            (gains.seed == s) & (gains.decoder == "ridge") & (gains.method == "lds")
        ].gain_all_minus_grid_bvc.iloc[0])
        gs = float(shuf[
            (shuf.seed == s) & (shuf.decoder == "ridge") & (shuf.method == "lds")
        ].gain_all_minus_grid_bvc.iloc[0])
        intact.append(gi)
        shuffled.append(gs)
        remaining.append(float(gs / gi) if abs(gi) > 1e-9 else float("nan"))
    return {
        "decoder": "ridge",
        "method": "lds",
        "per_seed_intact_gain_cm": intact,
        "per_seed_mean_shuffled_gain_cm": shuffled,
        "per_seed_remaining_frac": remaining,
        "note": "Report-only; not a pre-registered PASS/FAIL criterion",
    }


def _hd_speed_column_indices(
    results: Path,
    seed_index: int,
    spike_source: str,
    cfg: dict[str, Any],
) -> list[int]:
    cols, _ = _unit_column_indices(
        results, seed_index, spike_source, cfg, ("MEC_hd", "MEC_speed"),
    )
    return cols


def _shuffle_offsets_s(decode_times: np.ndarray) -> list[float]:
    t = np.asarray(decode_times, dtype=float)
    t0, t1 = float(t.min()), float(t.max())
    span = t1 - t0
    if span <= SHUFFLE_MIN_OFFSET_S:
        return [SHUFFLE_MIN_OFFSET_S] * SHUFFLE_N_OFFSETS
    lo = SHUFFLE_MIN_OFFSET_S
    hi = max(lo, span - SHUFFLE_MIN_OFFSET_S)
    return [float(x) for x in np.linspace(lo, hi, SHUFFLE_N_OFFSETS)]


def _apply_hd_speed_circular_shift(
    X: np.ndarray,
    hd_speed_cols: list[int],
    decode_times: np.ndarray,
    offset_s: float,
) -> np.ndarray:
    dt = float(np.median(np.diff(np.asarray(decode_times, dtype=float))))
    if dt <= 0:
        raise ValueError("decode_times must be strictly increasing")
    steps = int(round(float(offset_s) / dt))
    out = np.array(X, copy=True)
    for c in hd_speed_cols:
        out[:, c] = np.roll(out[:, c], steps)
    return out


def _subset_columns(
    results: Path,
    seed_index: int,
    spike_source: str,
    cfg: dict[str, Any],
    subset_name: str,
    n_all: int,
) -> tuple[list[int], str]:
    spec = MECHANISM_SUBSETS[subset_name]
    if spec is None:
        return list(range(n_all)), "all_decoder_units"
    cols, _ = _unit_column_indices(results, seed_index, spike_source, cfg, spec)
    return cols, ",".join(spec)


def _cv_score_alpha(Xtr, ytr, times_tr, alphas, n_blocks, gap_s) -> tuple[float, float]:
    """Return (best_alpha, best_inner_cv_median)."""
    folds = inner_cv_block_masks(times_tr, n_blocks=n_blocks, gap_s=gap_s)
    best_a, best = float(alphas[0]), np.inf
    for a in alphas:
        meds = []
        for tr, va in folds:
            pred = _fit_predict_ridge(Xtr[tr], ytr[tr], Xtr[va], a)
            meds.append(np.median(np.linalg.norm(pred - ytr[va], axis=1)))
        med = float(np.median(meds))
        if med < best or (med == best and a < best_a):
            best, best_a = med, float(a)
    return best_a, float(best)


def _cv_score_k(Xtr, ytr, times_tr, ks, n_blocks, gap_s) -> tuple[int, float]:
    """Return (best_k, best_inner_cv_median)."""
    folds = inner_cv_block_masks(times_tr, n_blocks=n_blocks, gap_s=gap_s)
    best_k, best = int(ks[0]), np.inf
    for k in ks:
        meds = []
        for tr, va in folds:
            pred = _fit_predict_knn(Xtr[tr], ytr[tr], Xtr[va], k)
            meds.append(np.median(np.linalg.norm(pred - ytr[va], axis=1)))
        med = float(np.median(meds))
        if med < best or (med == best and k < best_k):
            best, best_k = med, int(k)
    return best_k, float(best)


def _reduced_fit_decode(
    *,
    method: str,
    primary_d: int,
    cfg: dict[str, Any],
    methods_seed: int,
    X_sub: np.ndarray,
    obs: dict[str, Any],
    alphas: np.ndarray,
    ks: list[int],
    n_blocks: int,
    gap_s: float,
) -> dict[str, Any]:
    if X_sub.shape[1] < primary_d:
        raise ValueError(
            f"n_units={X_sub.shape[1]} < primary_d={primary_d} for {method}"
        )
    t0 = time.time()
    model = _make_rep(
        method, int(primary_d), cfg, methods_seed,
        n_fit=int(obs["train_ok"].sum()),
    )
    Z = fit_transform_representation(method, model, X_sub, obs["train_ok"])
    t_fit = time.time() - t0
    Ztr = Z[obs["train_ok"]]
    ytr = obs["y"][obs["train_ok"]]
    times_tr = np.asarray(obs["decode_times"])[obs["train_ok"]]
    t1 = time.time()
    alpha = _choose_alpha(Ztr, ytr, times_tr, alphas, n_blocks, gap_s)
    knn_k = _choose_k(Ztr, ytr, times_tr, ks, n_blocks, gap_s)
    t_cv = time.time() - t1
    Zte = Z[obs["eval_mask"]]
    yte = obs["y"][obs["eval_mask"]]
    pred_r = _fit_predict_ridge(Ztr, ytr, Zte, alpha)
    pred_k = _fit_predict_knn(Ztr, ytr, Zte, knn_k)
    ridge_m = _euclid(pred_r, yte)
    knn_m = _euclid(pred_k, yte)
    return {
        "primary_d": int(primary_d),
        "ridge_alpha": float(alpha),
        "knn_k": int(knn_k),
        "fit_s": float(t_fit),
        "decoder_cv_s": float(t_cv),
        "wall_s": float(time.time() - t0),
        "ridge_median": float(ridge_m["median"]),
        "knn_median": float(knn_m["median"]),
    }


def _reduced_fit_cv_scores(
    *,
    method: str,
    d: int,
    cfg: dict[str, Any],
    methods_seed: int,
    X_sub: np.ndarray,
    obs: dict[str, Any],
    alphas: np.ndarray,
    ks: list[int],
    n_blocks: int,
    gap_s: float,
) -> dict[str, Any]:
    """Fit representation at d; return decoder-only inner-CV scores (not test)."""
    if X_sub.shape[1] < d:
        raise ValueError(f"n_units={X_sub.shape[1]} < d={d} for {method}")
    model = _make_rep(method, int(d), cfg, methods_seed, n_fit=int(obs["train_ok"].sum()))
    Z = fit_transform_representation(method, model, X_sub, obs["train_ok"])
    Ztr = Z[obs["train_ok"]]
    ytr = obs["y"][obs["train_ok"]]
    times_tr = np.asarray(obs["decode_times"])[obs["train_ok"]]
    alpha, ridge_cv = _cv_score_alpha(Ztr, ytr, times_tr, alphas, n_blocks, gap_s)
    knn_k, knn_cv = _cv_score_k(Ztr, ytr, times_tr, ks, n_blocks, gap_s)
    return {
        "d": int(d),
        "ridge_alpha": float(alpha),
        "knn_k": int(knn_k),
        "ridge_cv_median": float(ridge_cv),
        "knn_cv_median": float(knn_cv),
    }


def update_phase8_status(criteria: dict[str, Any]) -> dict[str, Any]:
    """Set phase8_status summary string from criteria 1–3, 4, and 4′ (if present)."""
    o13 = criteria.get("overall_1_to_3", "PENDING")
    c4 = (criteria.get("cell_type_grid_bvc") or {}).get("status", "PENDING")
    c4p = (criteria.get("cell_type_grid_bvc_cv_d") or {}).get("status")
    parts = [f"1-3 {o13}", f"4 {c4} (as registered, fixed d=20)"]
    if c4p:
        parts.append(f"4′ {c4p} (post hoc, CV-selected d)")
    criteria["phase8_status"] = "; ".join(parts)
    return criteria


def preregister_criterion_4prime(results: Path | None = None) -> dict[str, Any]:
    """Write criterion 4′ as PENDING. Does not change criterion 4 (cell_type_grid_bvc)."""
    results = Path(results or OUTPUT_ROOT)
    crit_path = _pressure_root(results) / "criteria.json"
    criteria = json.loads(crit_path.read_text()) if crit_path.exists() else {}
    criteria["cell_type_grid_bvc_cv_d"] = {
        "status": "PENDING",
        "label": "4′",
        "rule": (
            "Secondary (post hoc, because (d) showed the fixed d=20 confounded "
            "criterion 4): on grid+BVC, with d ∈ {5,10,20} chosen per method and "
            "seed by the decoder-only inner-CV rule (never by test error), LDS kNN "
            "< PCA kNN in ≥ 4/5 seeds."
        ),
        "d_grid": list(CONTROL_DIMS),
        "d_selection": (
            "For each method×seed×subset, choose d ∈ {5,10,20} minimizing the "
            "kNN decoder-only inner-CV median (ties → smaller d). Test medians "
            "taken from saved C2 control_d_sweep.csv at that d."
        ),
        "required": "≥ 4/5",
        "note": "criterion 4 (cell_type_grid_bvc at saved primary_d) remains FAIL unchanged",
    }
    crit_path.write_text(json.dumps(criteria, indent=2) + "\n")
    print(json.dumps({"wrote": str(crit_path), "criterion_4prime": "PENDING"}, indent=2), flush=True)
    return criteria


def evaluate_criterion_4prime(report_df: pd.DataFrame) -> dict[str, Any]:
    """4′: LDS kNN < PCA kNN on grid+BVC at CV-selected d, ≥ 4/5."""
    diffs: list[float] = []
    for s in range(5):
        row = report_df[(report_df.seed == s) & (report_df.subset == "grid_bvc")]
        # one row per method; pivot
        pca = float(row[row.method == "pca"].knn_median.iloc[0])
        lds = float(row[row.method == "lds"].knn_median.iloc[0])
        diffs.append(lds - pca)
    n_lds = int(sum(d < 0 for d in diffs if np.isfinite(d)))
    return {
        "status": "PASS" if n_lds >= 4 else "FAIL",
        "label": "4′",
        "per_seed_lds_minus_pca_knn_cm": diffs,
        "n_lds_better": n_lds,
        "sign_count": {
            "LDS_better": n_lds,
            "PCA_better": int(sum(d > 0 for d in diffs if np.isfinite(d))),
            "tie": int(sum(d == 0 for d in diffs if np.isfinite(d))),
        },
        "required": "≥ 4/5",
    }


def finish_criterion_4prime(
    results: Path | None = None,
    *,
    seeds: range | list[int] | None = None,
) -> dict[str, Any]:
    """Select d by decoder-only CV; report test medians from saved C2 CSV (no new test fits).

    Representation fits are re-run only to recover inner-CV scores for d selection
    (those scores were not saved in control_d_sweep.csv). All reported test errors
    come from the existing C2 file.
    """
    results = Path(results or OUTPUT_ROOT)
    out_dir = _pressure_root(results)
    log_dir = results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    preregister_criterion_4prime(results)

    d_path = out_dir / "control_d_sweep.csv"
    if not d_path.is_file():
        raise FileNotFoundError(f"missing C2 fits: {d_path}")
    d_sweep = pd.read_csv(d_path)

    cfg = load_quadrant_n5_yaml()
    seed_list = list(seeds if seeds is not None else range(5))
    alphas = ridge_alpha_grid(cfg)
    ks = [int(k) for k in cfg["decoders"]["knn_k"]]
    n_blocks = int(cfg["split"]["inner_cv_blocks"])
    gap_s = float(cfg["split"]["gap_s"])

    cv_rows: list[dict[str, Any]] = []
    t0 = time.time()
    for seed_index in seed_list:
        sim_dir = results / f"seed_{seed_index}" / "sim"
        obs = build_observation(cfg, sim_dir, CELL_TYPE_SOURCE)
        summary = json.loads(
            (results / f"seed_{seed_index}" / CELL_TYPE_SOURCE / "source_summary.json").read_text()
        )
        methods_seed = int(summary["seed_streams"]["methods"])
        n_all = obs["X"].shape[1]
        print(f"[criterion_4prime] seed={seed_index} CV scores for d selection", flush=True)
        for subset_name in ("grid_bvc", "all"):
            cols, _ = _subset_columns(
                results, seed_index, CELL_TYPE_SOURCE, cfg, subset_name, n_all,
            )
            X_sub = np.asarray(obs["X"][:, cols], dtype=float)
            for method in CELL_TYPE_METHODS:
                for d in CONTROL_DIMS:
                    sc = _reduced_fit_cv_scores(
                        method=method, d=int(d), cfg=cfg, methods_seed=methods_seed,
                        X_sub=X_sub, obs=obs, alphas=alphas, ks=ks,
                        n_blocks=n_blocks, gap_s=gap_s,
                    )
                    print(
                        f"  {subset_name}/{method}/d={d}: "
                        f"knn_cv={sc['knn_cv_median']:.2f} ridge_cv={sc['ridge_cv_median']:.2f}",
                        flush=True,
                    )
                    cv_rows.append(dict(
                        seed=seed_index, subset=subset_name, method=method, **sc,
                    ))

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "control_d_cv_scores.csv", index=False)

    # Select d per seed×subset×method by min knn_cv (tie → smaller d).
    report_rows: list[dict[str, Any]] = []
    for seed_index in seed_list:
        for subset_name in ("grid_bvc", "all"):
            for method in CELL_TYPE_METHODS:
                sub = cv_df[
                    (cv_df.seed == seed_index) & (cv_df.subset == subset_name)
                    & (cv_df.method == method)
                ].sort_values(["knn_cv_median", "d"])
                chosen = sub.iloc[0]
                d_sel = int(chosen["d"])
                # Test medians from saved C2 CSV (never from this CV recompute).
                te = d_sweep[
                    (d_sweep.seed == seed_index) & (d_sweep.subset == subset_name)
                    & (d_sweep.method == method) & (d_sweep.d == d_sel)
                ]
                knn_med = float(te[te.decoder == "knn"].median_err.iloc[0])
                ridge_med = float(te[te.decoder == "ridge"].median_err.iloc[0])
                report_rows.append(dict(
                    seed=seed_index,
                    subset=subset_name,
                    method=method,
                    cv_selected_d=d_sel,
                    knn_cv_median=float(chosen["knn_cv_median"]),
                    ridge_cv_median=float(chosen["ridge_cv_median"]),
                    knn_median=knn_med,
                    ridge_median=ridge_med,
                    ridge_alpha=float(chosen["ridge_alpha"]),
                    knn_k=int(chosen["knn_k"]),
                ))
    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(out_dir / "criterion_4prime_report.csv", index=False)

    # Gains at CV-selected d: error(all)−error(grid_bvc)
    gain_rows: list[dict[str, Any]] = []
    for seed_index in seed_list:
        for method in CELL_TYPE_METHODS:
            for dec, col in (("knn", "knn_median"), ("ridge", "ridge_median")):
                gb = report_df[
                    (report_df.seed == seed_index) & (report_df.subset == "grid_bvc")
                    & (report_df.method == method)
                ]
                al = report_df[
                    (report_df.seed == seed_index) & (report_df.subset == "all")
                    & (report_df.method == method)
                ]
                e_gb = float(gb[col].iloc[0])
                e_all = float(al[col].iloc[0])
                gain_rows.append(dict(
                    seed=seed_index, method=method, decoder=dec,
                    d_grid_bvc=int(gb.cv_selected_d.iloc[0]),
                    d_all=int(al.cv_selected_d.iloc[0]),
                    err_grid_bvc=e_gb, err_all=e_all,
                    gain_all_minus_grid_bvc=e_all - e_gb,
                ))
    gain_df = pd.DataFrame(gain_rows)
    inter_rows: list[dict[str, Any]] = []
    for seed_index in seed_list:
        for dec in ("knn", "ridge"):
            g_pca = float(gain_df[
                (gain_df.seed == seed_index) & (gain_df.method == "pca") & (gain_df.decoder == dec)
            ].gain_all_minus_grid_bvc.iloc[0])
            g_lds = float(gain_df[
                (gain_df.seed == seed_index) & (gain_df.method == "lds") & (gain_df.decoder == dec)
            ].gain_all_minus_grid_bvc.iloc[0])
            inter_rows.append(dict(
                seed=seed_index, method="interaction", decoder=dec,
                d_grid_bvc=float("nan"), d_all=float("nan"),
                err_grid_bvc=float("nan"), err_all=float("nan"),
                gain_all_minus_grid_bvc=g_lds - g_pca,
            ))
    gain_df = pd.concat([gain_df, pd.DataFrame(inter_rows)], ignore_index=True)
    gain_df.to_csv(out_dir / "criterion_4prime_gains.csv", index=False)

    crit_4p = evaluate_criterion_4prime(report_df)
    # Preserve pre-registered rule text
    crit_path = out_dir / "criteria.json"
    criteria = json.loads(crit_path.read_text())
    prev = dict(criteria.get("cell_type_grid_bvc_cv_d") or {})
    keep = {k: prev[k] for k in prev if k in (
        "label", "rule", "d_grid", "d_selection", "required", "note",
    )}
    criteria["cell_type_grid_bvc_cv_d"] = {**keep, **crit_4p}
    if "rule" in keep:
        criteria["cell_type_grid_bvc_cv_d"]["rule"] = keep["rule"]
    # Explicitly leave criterion 4 unchanged (do not rewrite cell_type_grid_bvc).
    update_phase8_status(criteria)
    crit_path.write_text(json.dumps(criteria, indent=2) + "\n")

    summary = {
        "wall_time_s": time.time() - t0,
        "criterion_4": (criteria.get("cell_type_grid_bvc") or {}).get("status"),
        "criterion_4prime": crit_4p["status"],
        "n_lds_better_4prime": crit_4p["n_lds_better"],
        "outputs": {
            "control_d_cv_scores": str(out_dir / "control_d_cv_scores.csv"),
            "criterion_4prime_report": str(out_dir / "criterion_4prime_report.csv"),
            "criterion_4prime_gains": str(out_dir / "criterion_4prime_gains.csv"),
            "criteria": str(crit_path),
        },
    }
    (out_dir / "criterion_4prime_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def preregister_path_integration_criteria(results: Path | None = None) -> dict[str, Any]:
    """Write path-integration (a)/(b) rules into criteria.json before mechanism runs."""
    results = Path(results or OUTPUT_ROOT)
    crit_path = _pressure_root(results) / "criteria.json"
    crit_path.parent.mkdir(parents=True, exist_ok=True)
    criteria = json.loads(crit_path.read_text()) if crit_path.exists() else {}
    criteria["path_integration_a"] = {
        "status": "PENDING",
        "rule": (
            "Path-integration interpretation supported only if (a1) LDS kNN gain "
            "from adding HD+speed (error(all)−error(grid+BVC)) is negative in ≥ 4/5 "
            "seeds AND (a2) LDS gain is more negative than PCA's gain in ≥ 4/5 seeds"
        ),
        "gain_definition": "error(all) − error(grid+BVC); negative = adding HD+speed helps",
        "required_each": "≥ 4/5",
        "reduction": {
            "methods": list(CELL_TYPE_METHODS),
            "source": CELL_TYPE_SOURCE,
            "protocol": "single fit at saved primary_d; decoder-only inner CV",
        },
    }
    criteria["path_integration_b"] = {
        "status": "PENDING",
        "rule": (
            "Shuffle control: circular time shift (≥ 60 s) on HD+speed units only; "
            "≥ 50% of the intact LDS kNN gain removed in ≥ 4/5 seeds"
        ),
        "removal_definition": (
            "|gain_intact − gain_shuffled| / |gain_intact| ≥ 0.5 when gain_intact ≠ 0; "
            "gain_shuffled = mean over 5 session-spread offsets (same offsets all seeds)"
        ),
        "required": "≥ 4/5",
    }
    crit_path.write_text(json.dumps(criteria, indent=2) + "\n")
    print(json.dumps({"wrote": str(crit_path), "path_integration": "PENDING"}, indent=2), flush=True)
    return criteria


def _gain_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-seed gains error(all)−error(grid+BVC) and LDS−PCA interaction on gain."""
    if "shuffle_offset_s" not in df.columns:
        intact = df
    else:
        intact = df[df["shuffle_offset_s"].isna() | (df["shuffle_offset_s"] == 0)]
    rows: list[dict[str, Any]] = []
    for seed in sorted(int(s) for s in intact["seed"].unique()):
        base = intact[(intact.seed == seed) & (intact.subset == "grid_bvc")]
        all_u = intact[(intact.seed == seed) & (intact.subset == "all")]
        for dec in ("ridge", "knn"):
            for method in CELL_TYPE_METHODS:
                e_gb = float(base[(base.method == method) & (base.decoder == dec)].median_err.iloc[0])
                e_all = float(all_u[(all_u.method == method) & (all_u.decoder == dec)].median_err.iloc[0])
                rows.append(dict(
                    seed=int(seed), decoder=dec, method=method,
                    err_grid_bvc=e_gb, err_all=e_all,
                    gain_all_minus_grid_bvc=e_all - e_gb,
                ))
            g_pca = float(rows[-2]["gain_all_minus_grid_bvc"])
            g_lds = float(rows[-1]["gain_all_minus_grid_bvc"])
            rows.append(dict(
                seed=int(seed), decoder=dec, method="interaction",
                err_grid_bvc=float("nan"), err_all=float("nan"),
                gain_all_minus_grid_bvc=g_lds - g_pca,
            ))
    return pd.DataFrame(rows)


def _shuffle_gain_table(
    shuffle_rows: pd.DataFrame,
    grid_bvc_df: pd.DataFrame,
) -> pd.DataFrame:
    """Gain for shuffled-all vs intact grid+BVC (grid_bvc errors unchanged)."""
    rows: list[dict[str, Any]] = []
    for seed in sorted(int(s) for s in shuffle_rows["seed"].unique()):
        base = grid_bvc_df[(grid_bvc_df.seed == seed) & (grid_bvc_df.subset == "grid_bvc")]
        for offset in sorted(shuffle_rows[shuffle_rows.seed == seed].shuffle_offset_s.unique()):
            all_u = shuffle_rows[
                (shuffle_rows.seed == seed) & (shuffle_rows.shuffle_offset_s == offset)
            ]
            for dec in ("ridge", "knn"):
                for method in CELL_TYPE_METHODS:
                    e_gb = float(base[(base.method == method) & (base.decoder == dec)].median_err.iloc[0])
                    e_all = float(all_u[(all_u.method == method) & (all_u.decoder == dec)].median_err.iloc[0])
                    rows.append(dict(
                        seed=int(seed), decoder=dec, method=method,
                        shuffle_offset_s=float(offset),
                        gain_all_minus_grid_bvc=e_all - e_gb,
                    ))
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.groupby(["seed", "decoder", "method"], as_index=False).agg(
        gain_all_minus_grid_bvc=("gain_all_minus_grid_bvc", "mean"),
    )


def evaluate_path_integration(
    mechanism_df: pd.DataFrame,
    shuffle_df: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gains = _gain_table(mechanism_df)
    g_knn = gains[(gains.decoder == "knn") & (gains.method.isin(("pca", "lds", "interaction")))]
    lds_g: list[float] = []
    inter: list[float] = []
    for s in range(5):
        lds = g_knn[(g_knn.seed == s) & (g_knn.method == "lds")].gain_all_minus_grid_bvc
        intr = g_knn[(g_knn.seed == s) & (g_knn.method == "interaction")].gain_all_minus_grid_bvc
        lds_g.append(float(lds.iloc[0]) if len(lds) else float("nan"))
        inter.append(float(intr.iloc[0]) if len(intr) else float("nan"))
    n_a1 = int(sum(g < 0 for g in lds_g if np.isfinite(g)))
    n_a2 = int(sum(g < 0 for g in inter if np.isfinite(g)))
    crit_a = {
        "status": "PASS" if (n_a1 >= 4 and n_a2 >= 4) else "FAIL",
        "per_seed_lds_knn_gain_cm": lds_g,
        "per_seed_lds_minus_pca_gain_cm": inter,
        "n_lds_gain_negative": n_a1,
        "n_lds_gain_more_negative_than_pca": n_a2,
        "required_each": "≥ 4/5",
        "rule": (
            "Path-integration (a): LDS kNN gain negative in ≥ 4/5 AND "
            "LDS gain more negative than PCA in ≥ 4/5"
        ),
    }

    intact_lds = {
        int(r.seed): float(r.gain_all_minus_grid_bvc)
        for r in gains[(gains.decoder == "knn") & (gains.method == "lds")].itertuples()
    }
    shuffled: list[float] = []
    frac_removed: list[float] = []
    for s in range(5):
        sub = shuffle_df[
            (shuffle_df.seed == s) & (shuffle_df.decoder == "knn") & (shuffle_df.method == "lds")
        ]
        gi = intact_lds.get(s, float("nan"))
        gs = float(sub.gain_all_minus_grid_bvc.iloc[0]) if len(sub) else float("nan")
        shuffled.append(gs)
        if np.isfinite(gi) and abs(gi) > 1e-9 and np.isfinite(gs):
            frac_removed.append(abs(gi - gs) / abs(gi))
        else:
            frac_removed.append(float("nan"))
    n_b = int(sum(f >= 0.5 for f in frac_removed if np.isfinite(f)))
    crit_b = {
        "status": "PASS" if n_b >= 4 else "FAIL",
        "per_seed_intact_lds_knn_gain_cm": [intact_lds.get(s, float("nan")) for s in range(5)],
        "per_seed_mean_shuffled_lds_knn_gain_cm": shuffled,
        "per_seed_fraction_gain_removed": frac_removed,
        "n_pass": n_b,
        "required": "≥ 4/5",
        "rule": "Shuffle removes ≥ 50% of intact LDS kNN gain in ≥ 4/5 seeds",
    }
    return crit_a, crit_b


def _pressure_root(results: Path) -> Path:
    return results / "knn_pressure"


def _train_end_time(decode_times: np.ndarray, train_ok: np.ndarray) -> float:
    return float(np.max(decode_times[train_ok]))


def train_mask_excluding_delta(
    decode_times: np.ndarray,
    train_ok: np.ndarray,
    delta_s: float,
) -> np.ndarray:
    """Drop the last ``delta_s`` seconds of the contiguous training block.

    On this split, training samples within Δ of any test sample are exactly
    those in the last Δ s of training (pre-registered interpretation).
    Δ = 0 keeps the full train_ok mask.
    """
    keep = train_ok.copy()
    if float(delta_s) <= 0.0:
        return keep
    t_end = _train_end_time(decode_times, train_ok)
    keep &= decode_times <= (t_end - float(delta_s))
    return keep


def _fit_predict_knn_with_neighbours(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (predictions, neighbour_indices into the training rows)."""
    sc = StandardScaler()
    Ztr = sc.fit_transform(Xtr)
    Zte = sc.transform(Xte)
    model = KNeighborsRegressor(n_neighbors=int(k), weights="uniform")
    model.fit(Ztr, ytr)
    pred = model.predict(Zte)
    idx = model.kneighbors(Zte, return_distance=False)
    return pred, np.asarray(idx, dtype=int)


def _circular_mean(angles: np.ndarray) -> float:
    if len(angles) == 0:
        return float("nan")
    s = np.sin(angles).sum()
    c = np.cos(angles).sum()
    return float(np.arctan2(s, c))


def _circular_distance(a: np.ndarray, b: float) -> np.ndarray:
    """Smallest absolute circular distance in radians, in [0, π]."""
    d = np.arctan2(np.sin(a - b), np.cos(a - b))
    return np.abs(d)


def _bin_ix(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bx = np.clip((xy[:, 0] / ARENA_CM * N_BINS).astype(int), 0, N_BINS - 1)
    by = np.clip((xy[:, 1] / ARENA_CM * N_BINS).astype(int), 0, N_BINS - 1)
    return bx, by


def build_strata_masks(
    y: np.ndarray,
    heading: np.ndarray,
    speed: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """Typical vs atypical on the eval set (length = n_eval)."""
    bx_all, by_all = _bin_ix(y)
    heading_dist = np.full(int(eval_mask.sum()), np.nan)
    speed_lo = np.full(int(eval_mask.sum()), np.nan)
    speed_hi = np.full(int(eval_mask.sum()), np.nan)
    speed_pct = np.full(int(eval_mask.sum()), np.nan)

    eval_idx = np.where(eval_mask)[0]
    for j, i in enumerate(eval_idx):
        bx, by = int(bx_all[i]), int(by_all[i])
        in_bin = train_ok & (bx_all == bx) & (by_all == by)
        if not in_bin.any():
            continue
        h_tr = heading[in_bin]
        s_tr = speed[in_bin]
        mu = _circular_mean(h_tr)
        heading_dist[j] = float(_circular_distance(np.asarray([heading[i]]), mu)[0])
        q10, q90 = np.percentile(s_tr, [10, 90])
        speed_lo[j] = float(q10)
        speed_hi[j] = float(q90)
        # percentile rank of this speed within the bin's training speeds
        speed_pct[j] = float(100.0 * np.mean(s_tr <= speed[i]))

    heading_atypical = heading_dist > HEADING_ATYPICAL_RAD
    speed_atypical = (speed[eval_mask] < speed_lo) | (speed[eval_mask] > speed_hi)
    # NaN bins (no train occupancy): treat as atypical (cannot be typical)
    unknown = ~np.isfinite(heading_dist) | ~np.isfinite(speed_lo)
    atypical = heading_atypical | speed_atypical | unknown
    typical = ~atypical
    return {
        "atypical": atypical,
        "typical": typical,
        "heading_dist_rad": heading_dist,
        "speed_percentile": speed_pct,
        "heading_atypical": heading_atypical & ~unknown,
        "speed_atypical": speed_atypical & ~unknown,
        "unknown_bin": unknown,
    }


def run_cell(
    cfg: dict[str, Any],
    results: Path,
    seed_index: int,
    spike_source: str,
    method: str,
    obs: dict[str, Any],
    Z: np.ndarray,
    heading: np.ndarray,
    speed: np.ndarray,
    strata: dict[str, np.ndarray],
) -> dict[str, Any]:
    src_dir = results / f"seed_{seed_index}" / spike_source
    rec = json.loads((src_dir / f"{method}.json").read_text())
    alpha = float(rec["ridge_alpha"])
    knn_k = int(rec["knn_k"])
    offline_only = method == "gpfa"

    y = obs["y"]
    times = np.asarray(obs["decode_times"], dtype=float)
    train_ok = obs["train_ok"]
    eval_mask = obs["eval_mask"]
    yte = y[eval_mask]
    t_te = times[eval_mask]
    t_end = _train_end_time(times, train_ok)
    t_last30 = t_end - LAST_TRAIN_S

    exclusion_rows: list[dict[str, Any]] = []
    for delta in DELTAS_S:
        keep = train_mask_excluding_delta(times, train_ok, delta)
        n_tr = int(keep.sum())
        if n_tr < max(2, knn_k):
            for dec in ("ridge", "knn"):
                exclusion_rows.append(dict(
                    seed=seed_index, source=spike_source, method=method,
                    decoder=dec, delta_s=float(delta), n_train=n_tr,
                    median_err=np.nan, mean_err=np.nan, p90_err=np.nan,
                    offline_only=offline_only, note="too_few_train",
                ))
            continue
        Ztr, ytr = Z[keep], y[keep]
        Zte = Z[eval_mask]
        pred_r = _fit_predict_ridge(Ztr, ytr, Zte, alpha)
        pred_k = _fit_predict_knn(Ztr, ytr, Zte, knn_k)
        for dec, pred in (("ridge", pred_r), ("knn", pred_k)):
            stats = _euclid(pred, yte)
            exclusion_rows.append(dict(
                seed=seed_index, source=spike_source, method=method,
                decoder=dec, delta_s=float(delta), n_train=n_tr,
                median_err=stats["median"], mean_err=stats["mean"],
                p90_err=stats["p90"], offline_only=offline_only, note="",
            ))

    # Neighbour-time diagnostic at Δ = 0 (full train), kNN only
    keep0 = train_ok
    Ztr0, ytr0 = Z[keep0], y[keep0]
    t_tr0 = times[keep0]
    pred_k0, nbr_idx = _fit_predict_knn_with_neighbours(
        Ztr0, ytr0, Z[eval_mask], knn_k,
    )
    # nbr_idx: (n_eval, k) into training rows
    t_nbr = t_tr0[nbr_idx]  # (n_eval, k)
    dt = np.abs(t_te[:, None] - t_nbr)
    frac_last30 = float(np.mean(t_nbr >= t_last30))
    hist_counts, hist_edges = np.histogram(dt.ravel(), bins=40, range=(0.0, float(dt.max() + 1e-9)))
    neighbour = dict(
        seed=seed_index, source=spike_source, method=method,
        knn_k=knn_k, n_eval=int(eval_mask.sum()), n_train=int(keep0.sum()),
        median_abs_dt_s=float(np.median(dt)),
        p10_abs_dt_s=float(np.percentile(dt, 10)),
        mean_abs_dt_s=float(np.mean(dt)),
        frac_neighbours_last_30s_train=frac_last30,
        offline_only=offline_only,
        hist_counts=hist_counts.tolist(),
        hist_edges=hist_edges.tolist(),
    )

    # Strata errors at Δ = 0
    pred_r0 = _fit_predict_ridge(Ztr0, ytr0, Z[eval_mask], alpha)
    err_r = np.linalg.norm(pred_r0 - yte, axis=1)
    err_k = np.linalg.norm(pred_k0 - yte, axis=1)
    strata_rows: list[dict[str, Any]] = []
    for dec, err in (("ridge", err_r), ("knn", err_k)):
        for label, mask in (("typical", strata["typical"]), ("atypical", strata["atypical"])):
            if not mask.any():
                med = mean = p90 = np.nan
                n = 0
            else:
                e = err[mask]
                n = int(len(e))
                med = float(np.median(e))
                mean = float(np.mean(e))
                p90 = float(np.quantile(e, 0.90))
            strata_rows.append(dict(
                seed=seed_index, source=spike_source, method=method,
                decoder=dec, stratum=label, n=n,
                median_err=med, mean_err=mean, p90_err=p90,
                offline_only=offline_only,
            ))

    return {
        "exclusion": exclusion_rows,
        "neighbour": neighbour,
        "strata": strata_rows,
        "err_knn_atypical": (
            float(np.median(err_k[strata["atypical"]]))
            if strata["atypical"].any() else float("nan")
        ),
        "ridge_alpha": alpha,
        "knn_k": knn_k,
        "latent_route": None,  # filled by caller
    }


def evaluate_cell_type_criterion(cell_type_df: pd.DataFrame) -> dict[str, Any]:
    """Criterion 4 on grid+BVC: LDS kNN < PCA kNN in ≥ 4/5 seeds."""
    ct = cell_type_df[
        (cell_type_df.source == CELL_TYPE_SOURCE)
        & (cell_type_df.subset == "grid_bvc")
        & (cell_type_df.decoder == "knn")
    ]
    diffs: list[float] = []
    for s in range(5):
        lds = ct[(ct.seed == s) & (ct.method == "lds")].median_err
        pca = ct[(ct.seed == s) & (ct.method == "pca")].median_err
        if len(lds) and len(pca) and np.isfinite(lds.iloc[0]) and np.isfinite(pca.iloc[0]):
            diffs.append(float(lds.iloc[0]) - float(pca.iloc[0]))
        else:
            diffs.append(float("nan"))
    n_lds_better = int(sum(d < 0 for d in diffs if np.isfinite(d)))
    n_neg = n_lds_better
    n_pos = int(sum(d > 0 for d in diffs if np.isfinite(d)))
    n_zero = int(sum(d == 0 for d in diffs if np.isfinite(d)))
    return {
        "status": "PASS" if n_lds_better >= 4 else "FAIL",
        "rule": "LDS kNN < PCA kNN in the grid+BVC-only decode in ≥ 4/5 seeds",
        "per_seed_lds_minus_pca_knn_cm": diffs,
        "n_lds_better": n_lds_better,
        "sign_count": {"LDS_better": n_neg, "PCA_better": n_pos, "tie": n_zero},
        "required": "≥ 4/5",
        "reduction": {
            "methods": list(CELL_TYPE_METHODS),
            "source": CELL_TYPE_SOURCE,
            "subsets": list(CELL_TYPE_SUBSETS.keys()),
            "fit": "single representation fit at saved primary_d; no per-fold refit; no d re-selection",
            "decoder_hparams": "ridge alpha / knn k re-selected by decoder-only inner CV (same folds and purge gaps)",
        },
    }


def evaluate_criteria(
    exclusion: pd.DataFrame,
    neighbour: pd.DataFrame,
    strata: pd.DataFrame,
    cell_type: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Pre-registered PASS/FAIL for LDS+kNN on sorted spikes."""
    out: dict[str, Any] = {"source": "sorted", "method": "lds", "decoder": "knn"}

    # 1) |median(Δ=30) − median(Δ=0)| < 2 cm in ≥ 4/5 seeds
    ex = exclusion[
        (exclusion.source == "sorted") & (exclusion.method == "lds")
        & (exclusion.decoder == "knn")
    ]
    deltas = []
    for s in range(5):
        m0 = ex[(ex.seed == s) & (ex.delta_s == 0.0)].median_err
        m30 = ex[(ex.seed == s) & (ex.delta_s == 30.0)].median_err
        if len(m0) and len(m30) and np.isfinite(m0.iloc[0]) and np.isfinite(m30.iloc[0]):
            deltas.append(abs(float(m30.iloc[0]) - float(m0.iloc[0])))
        else:
            deltas.append(float("nan"))
    n_pass_excl = int(sum(d < 2.0 for d in deltas if np.isfinite(d)))
    out["exclusion_delta30"] = {
        "per_seed_abs_diff_cm": deltas,
        "n_pass": n_pass_excl,
        "threshold_cm": 2.0,
        "required": "≥ 4/5",
        "status": "PASS" if n_pass_excl >= 4 else "FAIL",
        "rule": "|median(Δ=30 s) − median(Δ=0)| < 2 cm in ≥ 4/5 seeds",
    }

    # 2) median |t_test − t_neighbour| > 60 s in ≥ 4/5 seeds
    nb = neighbour[(neighbour.source == "sorted") & (neighbour.method == "lds")]
    meds = []
    for s in range(5):
        row = nb[nb.seed == s]
        meds.append(float(row.median_abs_dt_s.iloc[0]) if len(row) else float("nan"))
    n_pass_nb = int(sum(m > 60.0 for m in meds if np.isfinite(m)))
    out["neighbour_median_dt"] = {
        "per_seed_median_abs_dt_s": meds,
        "n_pass": n_pass_nb,
        "threshold_s": 60.0,
        "required": "≥ 4/5",
        "status": "PASS" if n_pass_nb >= 4 else "FAIL",
        "rule": "median |t_test − t_neighbour| > 60 s in ≥ 4/5 seeds",
    }

    # 3) LDS kNN < PCA kNN on atypical samples in ≥ 4/5 seeds
    st = strata[
        (strata.source == "sorted") & (strata.decoder == "knn")
        & (strata.stratum == "atypical")
    ]
    diffs = []
    for s in range(5):
        lds = st[(st.seed == s) & (st.method == "lds")].median_err
        pca = st[(st.seed == s) & (st.method == "pca")].median_err
        if len(lds) and len(pca) and np.isfinite(lds.iloc[0]) and np.isfinite(pca.iloc[0]):
            diffs.append(float(lds.iloc[0]) - float(pca.iloc[0]))
        else:
            diffs.append(float("nan"))
    n_pass_st = int(sum(d < 0 for d in diffs if np.isfinite(d)))
    out["strata_lds_vs_pca_atypical"] = {
        "per_seed_lds_minus_pca_cm": diffs,
        "n_lds_better": n_pass_st,
        "required": "≥ 4/5",
        "status": "PASS" if n_pass_st >= 4 else "FAIL",
        "rule": "LDS kNN < PCA kNN on atypical samples in ≥ 4/5 seeds",
    }

    # 4) cell-type grid+BVC
    if cell_type is not None and len(cell_type):
        out["cell_type_grid_bvc"] = evaluate_cell_type_criterion(cell_type)
    else:
        out["cell_type_grid_bvc"] = {
            "status": "PENDING",
            "rule": "LDS kNN < PCA kNN in the grid+BVC-only decode in ≥ 4/5 seeds",
            "note": "Requires representation refits on unit subsets; see plan_cell_type_covariate / run_cell_type_covariate_reduced.",
        }
    out["overall_1_to_3"] = (
        "PASS"
        if all(
            out[k]["status"] == "PASS"
            for k in ("exclusion_delta30", "neighbour_median_dt", "strata_lds_vs_pca_atypical")
        )
        else "FAIL"
    )
    # Story flag: full Phase 8 (incl. cell-type) must PASS before dropping the pending note.
    out["phase8_passed"] = bool(
        out["overall_1_to_3"] == "PASS"
        and out["cell_type_grid_bvc"]["status"] == "PASS"
    )
    return out


def _unit_column_indices(
    results: Path,
    seed_index: int,
    spike_source: str,
    cfg: dict[str, Any],
    cell_types: tuple[str, ...],
) -> tuple[list[int], list[str]]:
    from realtime.data_loading import load_simulation_data

    sim_dir = results / f"seed_{seed_index}" / "sim"
    data = load_simulation_data(
        sim_dir, spike_source,
        include_regions=list(cfg["unit_inclusion"]["regions"]),
    )
    units = data["units_df"]
    uid_order = list(data["unit_ids"])
    units = units[units.unit_id.isin(uid_order)].copy()
    keep_ids = set(units[units.cell_type.isin(cell_types)].unit_id.tolist())
    cols = [i for i, u in enumerate(uid_order) if u in keep_ids]
    kept = [uid_order[i] for i in cols]
    return cols, kept


def _append_mechanism_rows(
    rows: list[dict[str, Any]],
    *,
    seed_index: int,
    subset_name: str,
    cell_types_label: str,
    n_units: int,
    method: str,
    fit: dict[str, Any],
    shuffle_offset_s: float | None = None,
) -> None:
    base: dict[str, Any] = dict(
        seed=seed_index,
        source=CELL_TYPE_SOURCE,
        subset=subset_name,
        cell_types=cell_types_label,
        n_units=n_units,
        method=method,
        primary_d=fit["primary_d"],
        ridge_alpha=fit["ridge_alpha"],
        knn_k=fit["knn_k"],
        fit_s=fit["fit_s"],
        decoder_cv_s=fit["decoder_cv_s"],
        wall_s=fit["wall_s"],
        shuffle_offset_s=shuffle_offset_s,
    )
    rows.append({**base, "decoder": "ridge", "median_err": fit["ridge_median"], "mean_err": float("nan")})
    rows.append({**base, "decoder": "knn", "median_err": fit["knn_median"], "mean_err": float("nan")})


def _write_cell_type_compat(mech_df: pd.DataFrame, out_dir: Path, seed_list: list[int]) -> None:
    ct_df = mech_df[mech_df.subset.isin(CELL_TYPE_SUBSETS.keys())].copy()
    ct_path = out_dir / "cell_type_reduced.csv"
    ct_df.to_csv(ct_path, index=False)
    report_rows: list[dict[str, Any]] = []
    for seed_index in seed_list:
        for subset_name in CELL_TYPE_SUBSETS:
            sub = ct_df[(ct_df.seed == seed_index) & (ct_df.subset == subset_name)]
            report_rows.append({
                "seed": seed_index,
                "subset": subset_name,
                "pca_knn_median": float(sub[(sub.method == "pca") & (sub.decoder == "knn")].median_err.iloc[0]),
                "lds_knn_median": float(sub[(sub.method == "lds") & (sub.decoder == "knn")].median_err.iloc[0]),
                "pca_ridge_median": float(sub[(sub.method == "pca") & (sub.decoder == "ridge")].median_err.iloc[0]),
                "lds_ridge_median": float(sub[(sub.method == "lds") & (sub.decoder == "ridge")].median_err.iloc[0]),
                "lds_minus_pca_knn": float(
                    sub[(sub.method == "lds") & (sub.decoder == "knn")].median_err.iloc[0]
                    - sub[(sub.method == "pca") & (sub.decoder == "knn")].median_err.iloc[0]
                ),
            })
    pd.DataFrame(report_rows).to_csv(out_dir / "cell_type_reduced_report.csv", index=False)


def run_mechanism_reduced(
    results: Path | None = None,
    *,
    seeds: range | list[int] | None = None,
    skip_preregister: bool = False,
) -> dict[str, Any]:
    """Reduced protocol: all units, mechanism subsets, HD+speed shuffle control."""
    results = Path(results or OUTPUT_ROOT)
    out_dir = _pressure_root(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not skip_preregister:
        preregister_path_integration_criteria(results)
    cfg = load_quadrant_n5_yaml()
    seed_list = list(seeds if seeds is not None else range(5))
    alphas = ridge_alpha_grid(cfg)
    ks = [int(k) for k in cfg["decoders"]["knn_k"]]
    n_blocks = int(cfg["split"]["inner_cv_blocks"])
    gap_s = float(cfg["split"]["gap_s"])
    offsets: list[float] | None = None

    rows: list[dict[str, Any]] = []
    shuffle_only: list[dict[str, Any]] = []
    t_wall0 = time.time()

    for seed_index in seed_list:
        sim_dir = results / f"seed_{seed_index}" / "sim"
        obs = build_observation(cfg, sim_dir, CELL_TYPE_SOURCE)
        summary = json.loads(
            (results / f"seed_{seed_index}" / CELL_TYPE_SOURCE / "source_summary.json").read_text()
        )
        methods_seed = int(summary["seed_streams"]["methods"])
        n_all = obs["X"].shape[1]
        hd_cols = _hd_speed_column_indices(results, seed_index, CELL_TYPE_SOURCE, cfg)
        if offsets is None:
            offsets = _shuffle_offsets_s(obs["decode_times"])
        print(f"[mechanism_reduced] seed={seed_index}", flush=True)

        for subset_name in MECHANISM_SUBSETS:
            cols, ct_label = _subset_columns(
                results, seed_index, CELL_TYPE_SOURCE, cfg, subset_name, n_all,
            )
            X_sub = np.asarray(obs["X"][:, cols], dtype=float)
            for method in CELL_TYPE_METHODS:
                rec = json.loads(
                    (results / f"seed_{seed_index}" / CELL_TYPE_SOURCE / f"{method}.json").read_text()
                )
                fit = _reduced_fit_decode(
                    method=method,
                    primary_d=int(rec["primary_d"]),
                    cfg=cfg,
                    methods_seed=methods_seed,
                    X_sub=X_sub,
                    obs=obs,
                    alphas=alphas,
                    ks=ks,
                    n_blocks=n_blocks,
                    gap_s=gap_s,
                )
                print(
                    f"  {subset_name}/{method}: ridge={fit['ridge_median']:.2f} "
                    f"knn={fit['knn_median']:.2f}",
                    flush=True,
                )
                _append_mechanism_rows(
                    rows, seed_index=seed_index, subset_name=subset_name,
                    cell_types_label=ct_label, n_units=len(cols), method=method, fit=fit,
                )

        X_full = np.asarray(obs["X"], dtype=float)
        for offset_s in offsets:
            X_sh = _apply_hd_speed_circular_shift(X_full, hd_cols, obs["decode_times"], offset_s)
            for method in CELL_TYPE_METHODS:
                rec = json.loads(
                    (results / f"seed_{seed_index}" / CELL_TYPE_SOURCE / f"{method}.json").read_text()
                )
                fit = _reduced_fit_decode(
                    method=method,
                    primary_d=int(rec["primary_d"]),
                    cfg=cfg,
                    methods_seed=methods_seed,
                    X_sub=X_sh,
                    obs=obs,
                    alphas=alphas,
                    ks=ks,
                    n_blocks=n_blocks,
                    gap_s=gap_s,
                )
                for dec, med in (("ridge", fit["ridge_median"]), ("knn", fit["knn_median"])):
                    shuffle_only.append(dict(
                        seed=seed_index, source=CELL_TYPE_SOURCE, subset="all_shuffled",
                        cell_types="all_decoder_units; HD+speed time-shifted",
                        n_units=n_all, method=method, shuffle_offset_s=float(offset_s),
                        primary_d=fit["primary_d"], decoder=dec, median_err=med,
                    ))

    mech_df = pd.DataFrame(rows)
    mech_path = out_dir / "mechanism_reduced.csv"
    mech_df.to_csv(mech_path, index=False)
    shuf_df = pd.DataFrame(shuffle_only)
    shuf_df.to_csv(out_dir / "mechanism_shuffle.csv", index=False)
    gains = _gain_table(mech_df)
    gains.to_csv(out_dir / "mechanism_gains.csv", index=False)
    shuf_gains = _shuffle_gain_table(shuf_df, mech_df)
    shuf_gains.to_csv(out_dir / "mechanism_shuffle_gains.csv", index=False)
    _write_cell_type_compat(mech_df, out_dir, seed_list)

    crit_a, crit_b = evaluate_path_integration(mech_df, shuf_gains)
    crit_path = out_dir / "criteria.json"
    criteria = json.loads(crit_path.read_text()) if crit_path.exists() else {}
    criteria["path_integration_a"] = crit_a
    criteria["path_integration_b"] = crit_b
    criteria["cell_type_grid_bvc"] = evaluate_cell_type_criterion(mech_df)
    criteria["phase8_passed"] = bool(
        criteria.get("overall_1_to_3") == "PASS"
        and criteria["cell_type_grid_bvc"]["status"] == "PASS"
    )
    crit_path.write_text(json.dumps(criteria, indent=2) + "\n")

    summary = {
        "wall_time_s": time.time() - t_wall0,
        "shuffle_offsets_s": offsets,
        "path_integration_a": crit_a["status"],
        "path_integration_b": crit_b["status"],
        "cell_type_grid_bvc": criteria["cell_type_grid_bvc"]["status"],
        "outputs": {
            "mechanism_reduced": str(mech_path),
            "mechanism_gains": str(out_dir / "mechanism_gains.csv"),
            "criteria": str(crit_path),
        },
    }
    (out_dir / "mechanism_reduced_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_controls_cd(
    results: Path | None = None,
    *,
    seeds: range | list[int] | None = None,
) -> dict[str, Any]:
    """C1 noise-unit + C2 dimension controls (reduced protocol). Preregisters (c)/(d) first."""
    results = Path(results or OUTPUT_ROOT)
    out_dir = _pressure_root(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    preregister_control_criteria(results)

    cfg = load_quadrant_n5_yaml()
    seed_list = list(seeds if seeds is not None else range(5))
    alphas = ridge_alpha_grid(cfg)
    ks = [int(k) for k in cfg["decoders"]["knn_k"]]
    n_blocks = int(cfg["split"]["inner_cv_blocks"])
    gap_s = float(cfg["split"]["gap_s"])

    noise_rows: list[dict[str, Any]] = []
    d_rows: list[dict[str, Any]] = []
    t0 = time.time()

    for seed_index in seed_list:
        sim_dir = results / f"seed_{seed_index}" / "sim"
        obs = _observation_with_counts(cfg, sim_dir, CELL_TYPE_SOURCE)
        summary = json.loads(
            (results / f"seed_{seed_index}" / CELL_TYPE_SOURCE / "source_summary.json").read_text()
        )
        methods_seed = int(summary["seed_streams"]["methods"])
        print(f"[controls_cd] seed={seed_index}", flush=True)

        # C1: grid+BVC + noise
        X_noise, meta = _build_grid_bvc_plus_noise(
            obs, results, seed_index, cfg, methods_seed,
        )
        for method in CELL_TYPE_METHODS:
            rec = json.loads(
                (results / f"seed_{seed_index}" / CELL_TYPE_SOURCE / f"{method}.json").read_text()
            )
            fit = _reduced_fit_decode(
                method=method,
                primary_d=int(rec["primary_d"]),
                cfg=cfg,
                methods_seed=methods_seed,
                X_sub=X_noise,
                obs=obs,
                alphas=alphas,
                ks=ks,
                n_blocks=n_blocks,
                gap_s=gap_s,
            )
            print(
                f"  noise/{method} d={fit['primary_d']}: "
                f"ridge={fit['ridge_median']:.2f} knn={fit['knn_median']:.2f}",
                flush=True,
            )
            for dec, med in (("ridge", fit["ridge_median"]), ("knn", fit["knn_median"])):
                noise_rows.append(dict(
                    seed=seed_index, source=CELL_TYPE_SOURCE, subset="grid_bvc_noise",
                    method=method, decoder=dec, primary_d=fit["primary_d"],
                    ridge_alpha=fit["ridge_alpha"], knn_k=fit["knn_k"],
                    median_err=med, n_units=meta["n_grid_bvc"] + meta["n_noise"],
                    **{k: meta[k] for k in ("n_grid_bvc", "n_noise", "hd_speed_rates_mean")},
                ))

        # C2: d ∈ {5,10,20} on grid_bvc and all
        n_all = obs["X"].shape[1]
        for subset_name in ("grid_bvc", "all"):
            cols, _ = _subset_columns(
                results, seed_index, CELL_TYPE_SOURCE, cfg, subset_name, n_all,
            )
            X_sub = np.asarray(obs["X"][:, cols], dtype=float)
            for method in CELL_TYPE_METHODS:
                for d in CONTROL_DIMS:
                    if X_sub.shape[1] < d:
                        raise ValueError(f"{subset_name} n={X_sub.shape[1]} < d={d}")
                    fit = _reduced_fit_decode(
                        method=method,
                        primary_d=int(d),
                        cfg=cfg,
                        methods_seed=methods_seed,
                        X_sub=X_sub,
                        obs=obs,
                        alphas=alphas,
                        ks=ks,
                        n_blocks=n_blocks,
                        gap_s=gap_s,
                    )
                    print(
                        f"  d_sweep/{subset_name}/{method}/d={d}: "
                        f"ridge={fit['ridge_median']:.2f} knn={fit['knn_median']:.2f}",
                        flush=True,
                    )
                    for dec, med in (("ridge", fit["ridge_median"]), ("knn", fit["knn_median"])):
                        d_rows.append(dict(
                            seed=seed_index, source=CELL_TYPE_SOURCE,
                            subset=subset_name, method=method, decoder=dec,
                            d=int(d), ridge_alpha=fit["ridge_alpha"], knn_k=fit["knn_k"],
                            median_err=med, n_units=len(cols),
                        ))

    noise_df = pd.DataFrame(noise_rows)
    d_df = pd.DataFrame(d_rows)
    noise_path = out_dir / "control_noise.csv"
    d_path = out_dir / "control_d_sweep.csv"
    noise_df.to_csv(noise_path, index=False)
    d_df.to_csv(d_path, index=False)

    mech_df = pd.read_csv(out_dir / "mechanism_reduced.csv")
    shuf_df = pd.read_csv(out_dir / "mechanism_shuffle.csv")
    crit_c = evaluate_control_c(noise_df, mech_df)
    crit_d = evaluate_control_d(d_df)
    ridge_shuf = ridge_shuffle_remaining_report(mech_df, shuf_df)

    crit_path = out_dir / "criteria.json"
    criteria = json.loads(crit_path.read_text())
    for key, measured in (("control_c_channel_count", crit_c), ("control_d_fixed_d", crit_d)):
        prev = dict(criteria.get(key) or {})
        keep = {k: prev[k] for k in prev if k in (
            "rule", "noise_definition", "gain_ratio_definition", "required",
            "dims", "threshold_cm", "threshold_ratio",
        )}
        criteria[key] = {**keep, **measured}
        if "rule" in keep:
            criteria[key]["rule"] = keep["rule"]
    criteria["ridge_shuffle_remaining"] = ridge_shuf
    crit_path.write_text(json.dumps(criteria, indent=2) + "\n")

    summary = {
        "wall_time_s": time.time() - t0,
        "control_c_channel_count": crit_c["status"],
        "control_d_fixed_d": crit_d["status"],
        "ridge_shuffle_remaining": ridge_shuf,
        "outputs": {
            "control_noise": str(noise_path),
            "control_d_sweep": str(d_path),
            "criteria": str(crit_path),
        },
    }
    (out_dir / "controls_cd_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "wall_time_s": summary["wall_time_s"],
        "control_c": crit_c["status"],
        "control_d": crit_d["status"],
        "n_pass_c": crit_c["n_pass"],
        "n_pass_d": crit_d["n_pass"],
        "ridge_remaining": ridge_shuf["per_seed_remaining_frac"],
    }, indent=2), flush=True)
    return summary


def run_cell_type_covariate_reduced(
    results: Path | None = None,
    *,
    seeds: range | list[int] | None = None,
) -> dict[str, Any]:
    """Reduced analysis 4 (grid_bvc + hd_speed); runs full mechanism study."""
    return run_mechanism_reduced(results, seeds=seeds, skip_preregister=True)


def finish_mechanism_from_csv(results: Path | None = None) -> dict[str, Any]:
    """Post-process saved intact + shuffle CSVs (no representation refits).

    Preserves pre-registered path_integration rule text in criteria.json;
    only fills status / per-seed numbers.
    """
    results = Path(results or OUTPUT_ROOT)
    out_dir = _pressure_root(results)
    log_dir = results / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    mech_path = out_dir / "mechanism_reduced.csv"
    shuf_path = out_dir / "mechanism_shuffle.csv"
    if not mech_path.is_file():
        raise FileNotFoundError(f"missing intact fits: {mech_path}")
    if not shuf_path.is_file():
        raise FileNotFoundError(f"missing shuffle fits: {shuf_path}")
    mech_df = pd.read_csv(mech_path)
    shuf_df = pd.read_csv(shuf_path)
    seed_list = sorted(int(s) for s in mech_df["seed"].unique())

    gains = _gain_table(mech_df)
    gains.to_csv(out_dir / "mechanism_gains.csv", index=False)
    shuf_gains = _shuffle_gain_table(shuf_df, mech_df)
    shuf_gains.to_csv(out_dir / "mechanism_shuffle_gains.csv", index=False)
    _write_cell_type_compat(mech_df, out_dir, seed_list)

    crit_a, crit_b = evaluate_path_integration(mech_df, shuf_gains)
    crit_path = out_dir / "criteria.json"
    criteria = json.loads(crit_path.read_text()) if crit_path.exists() else {}
    # Keep pre-registered rule text; only update status + measured fields.
    for key, measured in (("path_integration_a", crit_a), ("path_integration_b", crit_b)):
        prev = dict(criteria.get(key) or {})
        keep = {k: prev[k] for k in prev if k in (
            "rule", "gain_definition", "removal_definition", "required", "required_each", "reduction",
        )}
        criteria[key] = {**keep, **measured}
        # Prefer pre-registered wording over evaluate_*'s shorter rule.
        if "rule" in keep:
            criteria[key]["rule"] = keep["rule"]
    criteria["cell_type_grid_bvc"] = evaluate_cell_type_criterion(mech_df)
    criteria["phase8_passed"] = bool(
        criteria.get("overall_1_to_3") == "PASS"
        and criteria["cell_type_grid_bvc"]["status"] == "PASS"
    )
    crit_path.write_text(json.dumps(criteria, indent=2) + "\n")

    # Per-seed report with remaining-gain fraction for LDS kNN.
    report_rows: list[dict[str, Any]] = []
    for seed in seed_list:
        row: dict[str, Any] = {"seed": seed}
        for dec in ("ridge", "knn"):
            for method in CELL_TYPE_METHODS:
                g = gains[(gains.seed == seed) & (gains.decoder == dec) & (gains.method == method)]
                row[f"{method}_{dec}_gain"] = float(g.gain_all_minus_grid_bvc.iloc[0])
            inter = gains[
                (gains.seed == seed) & (gains.decoder == dec) & (gains.method == "interaction")
            ]
            row[f"interaction_{dec}"] = float(inter.gain_all_minus_grid_bvc.iloc[0])
        gi = float(gains[
            (gains.seed == seed) & (gains.decoder == "knn") & (gains.method == "lds")
        ].gain_all_minus_grid_bvc.iloc[0])
        gs = float(shuf_gains[
            (shuf_gains.seed == seed) & (shuf_gains.decoder == "knn") & (shuf_gains.method == "lds")
        ].gain_all_minus_grid_bvc.iloc[0])
        row["lds_knn_gain_intact"] = gi
        row["lds_knn_gain_shuffled"] = gs
        row["lds_knn_gain_remaining_frac"] = (
            float(gs / gi) if abs(gi) > 1e-9 else float("nan")
        )
        row["lds_knn_gain_removed_frac"] = (
            abs(gi - gs) / abs(gi) if abs(gi) > 1e-9 else float("nan")
        )
        for subset in ("grid_bvc", "grid_bvc_hd", "grid_bvc_speed", "all"):
            for method in CELL_TYPE_METHODS:
                for dec in ("ridge", "knn"):
                    e = float(mech_df[
                        (mech_df.seed == seed) & (mech_df.subset == subset)
                        & (mech_df.method == method) & (mech_df.decoder == dec)
                    ].median_err.iloc[0])
                    row[f"{subset}_{method}_{dec}"] = e
        report_rows.append(row)
    report_df = pd.DataFrame(report_rows)
    report_path = out_dir / "mechanism_report.csv"
    report_df.to_csv(report_path, index=False)

    summary = {
        "status": "COMPLETE_FROM_CSV",
        "path_integration_a": criteria["path_integration_a"]["status"],
        "path_integration_b": criteria["path_integration_b"]["status"],
        "cell_type_grid_bvc": criteria["cell_type_grid_bvc"]["status"],
        "shuffle_offsets_s": sorted(float(x) for x in shuf_df.shuffle_offset_s.unique()),
        "outputs": {
            "mechanism_gains": str(out_dir / "mechanism_gains.csv"),
            "mechanism_shuffle_gains": str(out_dir / "mechanism_shuffle_gains.csv"),
            "mechanism_report": str(report_path),
            "criteria": str(crit_path),
        },
    }
    (out_dir / "mechanism_reduced_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def plan_cell_type_covariate(results: Path) -> dict[str, Any]:
    """Plan analysis 4 — do not run."""
    from realtime.data_loading import load_simulation_data
    from realtime.quadrant_n5 import load_quadrant_n5_yaml

    cfg = load_quadrant_n5_yaml()
    timings = []
    for s in range(5):
        ss = json.loads((results / f"seed_{s}" / "sorted" / "source_summary.json").read_text())
        timings.append(ss.get("timings_s") or {})
    methods_needed = ["raw_lag", "pca", "dm", "lds", "gpfa"]
    med = {}
    for m in methods_needed:
        vals = [t[m] for t in timings if m in t]
        med[m] = float(np.median(vals)) if vals else float("nan")

    data = load_simulation_data(
        results / "seed_0" / "sim", "sorted",
        include_regions=list(cfg["unit_inclusion"]["regions"]),
    )
    units = data["units_df"] if "units_df" in data else pd.read_csv(results / "seed_0" / "sim" / "units.csv")
    # Restrict to the same unit_ids the primary run used.
    uid = set(data["unit_ids"])
    units = units[units.unit_id.isin(uid)].copy()
    pops = {
        "grid_bvc": sorted(units[units.cell_type.isin(["MEC_grid", "Sub_bvc"])].unit_id.tolist()),
        "hd_speed": sorted(units[units.cell_type.isin(["MEC_hd", "MEC_speed"])].unit_id.tolist()),
        "all": sorted(units.unit_id.tolist()),
    }
    n_by = {k: len(v) for k, v in pops.items()}
    per_seed_sorted_s = sum(med[m] for m in methods_needed)
    est_s = 2 * 2 * 5 * per_seed_sorted_s
    return {
        "status": "PLAN_ONLY_ASK_BEFORE_RUN",
        "populations": {
            "grid_bvc": {"cell_types": ["MEC_grid", "Sub_bvc"], "n_units": n_by["grid_bvc"]},
            "hd_speed": {"cell_types": ["MEC_hd", "MEC_speed"], "n_units": n_by["hd_speed"]},
            "all": {"cell_types": "all_decoder_units", "n_units": n_by["all"]},
        },
        "methods": list(methods_needed),
        "requires": (
            "representation refit on unit-subset spike matrix; propose frozen "
            "α/k from the primary run for fairness (no new nested selection)"
        ),
        "median_method_timings_s_sorted": med,
        "est_wall_time_s": est_s,
        "est_wall_time_h": round(est_s / 3600.0, 1),
        "est_note": (
            "Conservative: 2 new populations × 2 sources × 5 seeds × sum of "
            "median per-method timings from the primary sorted run. "
            "'all' can reuse primary results. Ask before launching."
        ),
    }


def plan_blocked_outer_cv(results: Path) -> dict[str, Any]:
    """Plan analysis 6 — blocked outer CV; do not run."""
    timings = []
    for s in range(5):
        ss = json.loads((results / f"seed_{s}" / "sorted" / "source_summary.json").read_text())
        timings.append(ss.get("timings_s") or {})
    methods = ["raw_lag", "pca", "dm", "lds", "gpfa"]  # pressure-test set
    # Include isomap? User said methods raw_lag, pca, dm, lds, gpfa for pressure;
    # blocked CV plan is full nested — use same method set + note isomap/raw optional.
    med = {m: float(np.median([t[m] for t in timings if m in t])) for m in methods}
    # Also report full primary method set cost for context
    all_methods = ["raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"]
    med_all = {
        m: float(np.median([t[m] for t in timings if m in t]))
        for m in all_methods
    }
    per_seed_one_source = sum(med.values())
    # 5 outer folds, each with full nested selection + representation refit.
    # Train fraction per fold ≈ 0.8 of session still (4/5 blocks train), so
    # cost ≈ primary. Folds × sources × seeds.
    n_folds, n_sources, n_seeds = 5, 2, 5
    est_s = n_folds * n_sources * n_seeds * per_seed_one_source
    return {
        "status": "PLAN_ONLY",
        "design": (
            "5 contiguous test blocks per seed; purge + exclusion windows on "
            "BOTH sides of each block; full nested hyperparameter selection "
            "and representation refit per fold (same rule as Phase 3)."
        ),
        "methods": methods,
        "median_method_timings_s_sorted": med,
        "median_all_methods_s_sorted": med_all,
        "est_wall_time_s": est_s,
        "est_wall_time_h": round(est_s / 3600.0, 1),
        "est_note": (
            f"{n_folds} folds × {n_sources} sources × {n_seeds} seeds × "
            f"~{per_seed_one_source / 3600:.2f} h (sum of median method "
            "timings). Nested inner CV inside each fold is already included "
            "in those method timings. Parallelism across seeds could cut "
            "wall time ~5× if cores allow."
        ),
    }


def run_all(
    results: Path | None = None,
    *,
    seeds: range | list[int] | None = None,
) -> dict[str, Any]:
    results = Path(results or OUTPUT_ROOT)
    out_dir = _pressure_root(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_quadrant_n5_yaml()
    seed_list = list(seeds if seeds is not None else range(5))

    exclusion_all: list[dict[str, Any]] = []
    neighbour_all: list[dict[str, Any]] = []
    strata_all: list[dict[str, Any]] = []
    hist_payload: dict[str, Any] = {}
    t_wall0 = time.time()

    for seed_index in seed_list:
        for spike_source in SOURCES:
            print(f"[knn_pressure] seed={seed_index} source={spike_source}", flush=True)
            sim_dir = results / f"seed_{seed_index}" / "sim"
            obs = build_observation(cfg, sim_dir, spike_source)
            from realtime.data_loading import load_simulation_data

            data = load_simulation_data(
                sim_dir, spike_source,
                include_regions=list(cfg["unit_inclusion"]["regions"]),
            )
            beh = align_behavior_to_decoder_times(data["behavior_df"], obs["decode_times"])
            heading = np.asarray(beh["head_direction"], dtype=float)
            speed = np.asarray(beh["speed"], dtype=float)
            strata = build_strata_masks(
                obs["y"], heading, speed, obs["train_ok"], obs["eval_mask"],
            )
            # Persist per-seed strata masks summary
            strata_meta = {
                "n_eval": int(obs["eval_mask"].sum()),
                "n_typical": int(strata["typical"].sum()),
                "n_atypical": int(strata["atypical"].sum()),
                "n_heading_atypical": int(strata["heading_atypical"].sum()),
                "n_speed_atypical": int(strata["speed_atypical"].sum()),
                "n_unknown_bin": int(strata["unknown_bin"].sum()),
            }
            (out_dir / f"strata_meta_seed{seed_index}_{spike_source}.json").write_text(
                json.dumps(strata_meta, indent=2) + "\n"
            )

            summary = json.loads(
                (results / f"seed_{seed_index}" / spike_source / "source_summary.json").read_text()
            )
            methods_seed = int(summary["seed_streams"]["methods"])

            for method in METHODS:
                rec = json.loads(
                    (results / f"seed_{seed_index}" / spike_source / f"{method}.json").read_text()
                )
                model_dir = results / "models" / f"seed_{seed_index}" / spike_source / method
                t0 = time.time()
                Z, route = latents_for_method(
                    method, cfg, obs, rec, model_dir, methods_seed, force_refit=False,
                )
                print(
                    f"  {method}: latents {time.time() - t0:.2f}s route={route} Z={Z.shape}",
                    flush=True,
                )
                cell = run_cell(
                    cfg, results, seed_index, spike_source, method,
                    obs, Z, heading, speed, strata,
                )
                cell["latent_route"] = route
                exclusion_all.extend(cell["exclusion"])
                neighbour_all.append({
                    k: v for k, v in cell["neighbour"].items()
                    if k not in ("hist_counts", "hist_edges")
                })
                key = f"seed{seed_index}_{spike_source}_{method}"
                hist_payload[f"{key}_counts"] = np.asarray(cell["neighbour"]["hist_counts"])
                hist_payload[f"{key}_edges"] = np.asarray(cell["neighbour"]["hist_edges"])
                strata_all.extend(cell["strata"])

    excl_df = pd.DataFrame(exclusion_all)
    nbr_df = pd.DataFrame(neighbour_all)
    st_df = pd.DataFrame(strata_all)
    excl_df.to_csv(out_dir / "exclusion.csv", index=False)
    nbr_df.to_csv(out_dir / "neighbour_diag.csv", index=False)
    st_df.to_csv(out_dir / "strata.csv", index=False)
    np.savez_compressed(out_dir / "neighbour_hist.npz", **hist_payload)

    criteria = evaluate_criteria(excl_df, nbr_df, st_df)
    (out_dir / "criteria.json").write_text(json.dumps(criteria, indent=2) + "\n")

    plan4 = plan_cell_type_covariate(results)
    plan6 = plan_blocked_outer_cv(results)
    summary = {
        "config_sha256": cfg["config_sha256"],
        "seeds": seed_list,
        "methods": list(METHODS),
        "sources": list(SOURCES),
        "deltas_s": list(DELTAS_S),
        "wall_time_s": time.time() - t_wall0,
        "criteria": criteria,
        "plan_cell_type_covariate": plan4,
        "plan_blocked_outer_cv": plan6,
        "outputs": {
            "exclusion": str(out_dir / "exclusion.csv"),
            "neighbour_diag": str(out_dir / "neighbour_diag.csv"),
            "neighbour_hist": str(out_dir / "neighbour_hist.npz"),
            "strata": str(out_dir / "strata.csv"),
            "criteria": str(out_dir / "criteria.json"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({
        "wall_time_s": summary["wall_time_s"],
        "criteria_overall_1_to_3": criteria["overall_1_to_3"],
        "exclusion": criteria["exclusion_delta30"]["status"],
        "neighbour": criteria["neighbour_median_dt"]["status"],
        "strata": criteria["strata_lds_vs_pca_atypical"]["status"],
        "plan4_h": plan4["est_wall_time_h"],
        "plan6_h": plan6["est_wall_time_h"],
    }, indent=2), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--seeds", type=str, default="0-4",
                    help="e.g. 0-4 or 0,1,2")
    ap.add_argument(
        "--cell-type-reduced",
        action="store_true",
        help="Run reduced mechanism study (same as --mechanism-reduced).",
    )
    ap.add_argument(
        "--mechanism-reduced",
        action="store_true",
        help=(
            "Pre-register path-integration criteria, then run reduced pca/lds on "
            "all + cell-type subsets + HD+speed shuffle control."
        ),
    )
    ap.add_argument(
        "--preregister-path-integration",
        action="store_true",
        help="Write path-integration (a)/(b) rules to criteria.json only.",
    )
    ap.add_argument(
        "--finish-mechanism-from-csv",
        action="store_true",
        help=(
            "Compute gains and path-integration PASS/FAIL from saved "
            "mechanism_reduced.csv + mechanism_shuffle.csv (no refits)."
        ),
    )
    ap.add_argument(
        "--preregister-controls-cd",
        action="store_true",
        help="Write criteria (c)/(d) as PENDING without running C1/C2.",
    )
    ap.add_argument(
        "--controls-cd",
        action="store_true",
        help="Preregister (c)/(d), then run noise-unit (C1) and dimension (C2) controls.",
    )
    ap.add_argument(
        "--preregister-criterion-4prime",
        action="store_true",
        help="Write criterion 4′ as PENDING (does not change criterion 4).",
    )
    ap.add_argument(
        "--finish-criterion-4prime",
        action="store_true",
        help=(
            "Select d by decoder-only CV; report test medians from saved C2 CSV; "
            "evaluate criterion 4′."
        ),
    )
    args = ap.parse_args(argv)
    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    if args.preregister_path_integration:
        preregister_path_integration_criteria(args.results)
    elif args.preregister_controls_cd:
        preregister_control_criteria(args.results)
    elif args.preregister_criterion_4prime:
        preregister_criterion_4prime(args.results)
    elif args.finish_mechanism_from_csv:
        finish_mechanism_from_csv(args.results)
    elif args.finish_criterion_4prime:
        finish_criterion_4prime(args.results, seeds=seeds)
    elif args.controls_cd:
        run_controls_cd(args.results, seeds=seeds)
    elif args.mechanism_reduced or args.cell_type_reduced:
        run_mechanism_reduced(args.results, seeds=seeds)
    else:
        run_all(args.results, seeds=seeds)


if __name__ == "__main__":
    main()
