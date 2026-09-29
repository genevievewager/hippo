"""Phase 8 — kNN pressure test (post hoc; saved models + predictions).

No config change. No representation refits for analyses 1–3.
Refits ONLY Ridge / kNN with the saved α / k.

Writes under outputs/quadrant_n5/knn_pressure/:
  exclusion.csv, neighbour_diag.csv, neighbour_hist.npz,
  strata.csv, criteria.json, summary.json

Cell-type covariate (analysis 4) and blocked outer CV (analysis 6) are
planned only; see plan_cell_type_covariate() / plan_blocked_outer_cv().
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
    _euclid,
    _fit_predict_knn,
    _fit_predict_ridge,
)
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


def evaluate_criteria(exclusion: pd.DataFrame, neighbour: pd.DataFrame, strata: pd.DataFrame) -> dict[str, Any]:
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

    # 4) cell-type — not run; mark PENDING
    out["cell_type_grid_bvc"] = {
        "status": "PENDING",
        "rule": "LDS kNN < PCA kNN in the grid+BVC-only decode in ≥ 4/5 seeds",
        "note": "Requires representation refits on unit subsets; see plan_cell_type_covariate.",
    }
    out["overall_1_to_3"] = (
        "PASS"
        if all(
            out[k]["status"] == "PASS"
            for k in ("exclusion_delta30", "neighbour_median_dt", "strata_lds_vs_pca_atypical")
        )
        else "FAIL"
    )
    return out


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
    args = ap.parse_args(argv)
    if "-" in args.seeds and "," not in args.seeds:
        a, b = args.seeds.split("-", 1)
        seeds = list(range(int(a), int(b) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    run_all(args.results, seeds=seeds)


if __name__ == "__main__":
    main()
