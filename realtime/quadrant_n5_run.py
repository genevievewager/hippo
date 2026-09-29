"""Phase 3 runner for the frozen n=5 quadrant experiment."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from realtime.data_loading import load_simulation_data, make_decode_times
from realtime.dynamic_latents.lds import assert_readout_latents_are_filtered
from realtime.manifold_features import make_feature_transformer
from realtime.pipeline_artifacts import hash_train_indices
from realtime.quadrant_n5 import (
    REPO_ROOT,
    derive_seed_streams,
    inner_cv_block_masks,
    load_quadrant_n5_yaml,
    promote_unkeyed_sim,
    read_sim_provenance,
    runtime_versions,
    sim_data_identity,
    sim_dir_is_complete,
    sim_fit_hash,
    sim_provenance_matches,
)
from realtime.spike_binner import build_causal_spike_matrix
from realtime.timing import extract_behavior_times
from realtime.train_decoder import causal_train_test_split

OUTPUT_ROOT = REPO_ROOT / "outputs" / "quadrant_n5"
METHOD_KEYS = ("raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa")
REDUCING_SWEEP = ("pca", "dm", "isomap", "lds", "gpfa")
D_SWEEP_SELECTION_RULE = "phase3_per_fold_refit"


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return None


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
        )
        return bool(out.strip())
    except Exception:
        return True


def _result_payload(cfg: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = {
        "config_sha256": cfg["config_sha256"],
        "git_sha": _git_sha(),
        "dirty_tree": _git_dirty(),
        "versions": runtime_versions(),
        "target": cfg["decoders"]["target"],
        "latency_budget_ms": cfg["latency_budget_ms"],
        "probe_track_sha256": cfg.get("probe_track_sha256"),
        "probe_track_file": (cfg.get("probe_track") or {}).get("file"),
    }
    out.update(extra)
    return out


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


_METHOD_ROW_KEYS = (
    "method", "primary_d", "ridge_alpha", "knn_k", "ridge", "knn",
    "n_train", "n_eval", "index_hashes", "elapsed_s", "a13",
    "n_units", "n_units_by_cell_type", "n_units_by_region", "coverage",
    "inner_cv_ridge_median", "inner_cv_knn_median", "n_folds",
    "refit_representation",
)


def reusable_result_json(path: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Return payload if it exists and its config hash matches the frozen YAML."""
    if not path.is_file():
        return None
    try:
        rec = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(rec, dict):
        return None
    if rec.get("config_sha256") != cfg.get("config_sha256"):
        return None
    return rec


def method_row_from_payload(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in _METHOD_ROW_KEYS if k in rec}


def source_summary_reusable(out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Skip a whole source when every method JSON and the summary match the config hash."""
    summary = reusable_result_json(out_dir / "source_summary.json", cfg)
    if summary is None:
        return None
    for key in METHOD_KEYS:
        if reusable_result_json(out_dir / f"{key}.json", cfg) is None:
            return None
    return summary


def generate_seed_dataset(
    cfg: dict[str, Any],
    seed_index: int,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, dict[str, Any], float]:
    from run_simulation import generate_dataset
    from hippo_sim.config import SimConfig
    from hippo.anatomy.trajectory_config import resolve_trajectory_config
    from hippo_sim.pipeline import apply_trajectory_to_config

    seeds = cfg["seeds"]
    streams = derive_seed_streams(
        int(seeds["master_seed"]),
        int(seeds["n_seeds"]),
        seed_index,
        tuple(seeds["components"]),
    )
    dest = Path(output_dir) if output_dir else OUTPUT_ROOT / f"seed_{seed_index}" / "sim"
    dest.mkdir(parents=True, exist_ok=True)
    expected = sim_fit_hash(cfg, streams)
    data_id = sim_data_identity(cfg, streams)
    if sim_dir_is_complete(dest):
        prior_meta = read_sim_provenance(dest) or {}
        data_ok = prior_meta.get("sim_data_identity") == data_id
        if not data_ok:
            summary_path = dest / "summary.json"
            try:
                summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            except Exception:
                summary = {}
            data_ok = (
                int(summary.get("seed", -1)) == int(streams["data_seed"])
                and abs(float(summary.get("session_duration_s", -1)) - float(cfg["session"]["session_s"])) < 1e-9
            )
        if sim_provenance_matches(dest, expected) or promote_unkeyed_sim(
            dest, cfg, streams, expected,
        ) or data_ok:
            prior = dest / "quadrant_n5_sim.json"
            elapsed = 0.0
            meta = {}
            if prior.exists():
                try:
                    meta = json.loads(prior.read_text())
                    elapsed = float(meta.get("elapsed_s", 0.0))
                except Exception:
                    meta = {}
            meta.update({
                "sim_fit_hash": expected,
                "sim_data_identity": data_id,
                "config_sha256": cfg["config_sha256"],
                "probe_track_sha256": cfg.get("probe_track_sha256"),
                "seed_streams": dict(streams),
            })
            _write_json(prior, meta)
            print(
                f"reusing keyed sim at {dest} fit={expected} "
                f"(prior elapsed {elapsed:.1f}s)",
                flush=True,
            )
            return dest, streams, elapsed
        print(
            f"refusing unkeyed/mismatched sim at {dest}; regenerating "
            f"(expected fit={expected})",
            flush=True,
        )
    session = cfg["session"]
    sim = SimConfig(
        output_dir=dest,
        seed=int(streams["data_seed"]),
        session_duration_s=float(session["session_s"]),
        behavior_dt=float(session["behavior_dt"]),
        arena_size_cm=float(session["arena_size_cm"]),
        thigmotaxis=float(session["thigmotaxis"]),
    )
    probe_path = Path(cfg["probe_track_path"])
    apply_trajectory_to_config(
        sim,
        trajectory_config=resolve_trajectory_config(str(probe_path)),
    )
    t0 = time.perf_counter()
    summary = generate_dataset(sim)
    elapsed = time.perf_counter() - t0
    meta = _result_payload(cfg, {
        "stage": "simulation",
        "seed_index": seed_index,
        "seed_streams": streams,
        "sim_fit_hash": expected,
        "sim_data_identity": data_id,
        "elapsed_s": elapsed,
        "summary_keys": sorted(summary.keys()) if isinstance(summary, dict) else [],
    })
    _write_json(dest / "quadrant_n5_sim.json", meta)
    return dest, streams, elapsed


def _sqrt_zscore_train(X: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    Xs = np.sqrt(np.maximum(np.asarray(X, dtype=float), 0.0))
    scaler = StandardScaler()
    scaler.fit(Xs[train_mask])
    return scaler.transform(Xs), scaler


def _position(behavior) -> np.ndarray:
    return np.column_stack([
        behavior["x"].to_numpy(dtype=float),
        behavior["y"].to_numpy(dtype=float),
    ])


def _euclid(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.linalg.norm(pred - y, axis=1)
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "p90": float(np.quantile(err, 0.90)),
    }


def _dm_n_landmarks(cfg: dict[str, Any], n_fit: int) -> int:
    dm = cfg["representations"]["dm"]
    if int(n_fit) <= int(dm["landmark_cap"]):
        return int(n_fit)
    return int(dm["n_landmarks_if_over_cap"])


def _make_rep(name: str, d: int, cfg: dict[str, Any], seed: int, n_fit: int):
    reps = cfg["representations"]
    feat = cfg["features"]
    if name == "raw":
        return make_feature_transformer("identity", decode_window=feat["window_s"])
    if name == "raw_lag":
        return make_feature_transformer(
            "raw_lag", decode_window=feat["window_s"],
            n_lags=int(reps["raw_lag"]["n_lags"]),
        )
    if name == "pca":
        pca = reps["pca"]
        return make_feature_transformer(
            "global_pca", decode_window=feat["window_s"],
            n_components=int(d), random_state=seed,
            svd_solver=str(pca["svd_solver"]),
            whiten=bool(pca["whiten"]),
        )
    if name == "dm":
        dm = reps["dm"]
        return make_feature_transformer(
            "diffusion_nystrom", decode_window=feat["window_s"],
            n_components=int(d), random_state=seed,
            n_landmarks=_dm_n_landmarks(cfg, n_fit),
            landmark_method=str(dm["landmark_method"]),
            diffusion_local_scale_k=int(dm["local_scale_k"]),
            diffusion_alpha=float(dm["alpha"]),
            diffusion_time=float(dm["diffusion_time"]),
        )
    if name == "lds":
        ld = reps["lds"]
        from realtime.dynamic_latents.adapters import DynamicLatentEmbedding
        return DynamicLatentEmbedding(
            model_name="global_lds",
            n_components=int(d),
            update_dt=float(feat["update_dt"]),
            decode_window=float(feat["window_s"]),
            random_state=seed,
            causal_default=True,
            n_em_iters=int(ld["em_iters"]),
            process_noise_scale=float(ld["process_noise_scale"]),
            observation_noise_scale=float(ld["observation_noise_scale"]),
        )
    if name == "isomap":
        iso = reps["isomap"]
        return make_feature_transformer(
            "global_isomap", decode_window=feat["window_s"],
            n_components=int(d), random_state=seed,
            n_neighbors=int(iso["n_neighbors"]),
            isomap_pre_pca_enabled=bool(iso["pre_pca_enabled"]),
            isomap_pre_pca_n_components=int(iso["pre_pca_n_components"]),
            isomap_require_connected_graph=bool(iso["require_connected_graph"]),
        )
    if name == "gpfa":
        gp = reps["gpfa"]
        from realtime.dynamic_latents.adapters import DynamicLatentEmbedding
        return DynamicLatentEmbedding(
            model_name="gpfa",
            n_components=int(d),
            update_dt=float(feat["update_dt"]),
            decode_window=float(feat["window_s"]),
            random_state=seed,
            causal_default=False,
            max_iter=int(gp["max_iter"]),
            factor_analysis_max_iter=int(gp["factor_analysis_max_iter"]),
            default_tau=float(gp["default_tau_s"]),
        )
    raise KeyError(name)


def fit_transform_representation(name: str, model, X, fit_mask):
    """Fit on ``fit_mask``; transform the full session (same OOS path as test).

    LDS Kalman-filters continuously from session start (``X[0]``). A14 is
    checked on that full-session filter, not a train-subset re-filter.
    """
    X = np.asarray(X, dtype=float)
    model.fit(X[np.asarray(fit_mask, dtype=bool)])
    if name == "lds":
        Z = model.transform(X, causal=True, reset=True)
        assert_readout_latents_are_filtered(model.model_, Z, X)
        return Z
    if name == "gpfa":
        return model.transform(X)
    return model.transform(X)


def _valid_mask(name: str, n: int, cfg: dict[str, Any]) -> np.ndarray:
    if name == "raw_lag":
        k = int(cfg["representations"]["raw_lag"]["n_history_lost"])
        m = np.ones(n, dtype=bool)
        m[:k] = False
        return m
    return np.ones(n, dtype=bool)


def _fit_predict_ridge(Xtr, ytr, Xte, alpha: float) -> np.ndarray:
    sc = StandardScaler()
    Ztr = sc.fit_transform(Xtr)
    Zte = sc.transform(Xte)
    model = Ridge(alpha=float(alpha))
    model.fit(Ztr, ytr)
    return model.predict(Zte)


def _fit_predict_knn(Xtr, ytr, Xte, k: int) -> np.ndarray:
    sc = StandardScaler()
    Ztr = sc.fit_transform(Xtr)
    Zte = sc.transform(Xte)
    model = KNeighborsRegressor(n_neighbors=int(k), weights="uniform")
    model.fit(Ztr, ytr)
    return model.predict(Zte)


def _choose_alpha(Xtr, ytr, times_tr, alphas, n_blocks, gap_s) -> float:
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
    return best_a


def _choose_k(Xtr, ytr, times_tr, ks, n_blocks, gap_s) -> int:
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
    return best_k


def ridge_alpha_grid(cfg: dict[str, Any]) -> np.ndarray:
    dec = cfg["decoders"]
    return np.logspace(
        np.log10(float(dec["ridge_alpha_lo"])),
        np.log10(float(dec["ridge_alpha_hi"])),
        int(dec["ridge_alpha_n"]),
    )


def _inner_cv_select(
    key: str,
    X: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    decode_times: np.ndarray,
    cfg: dict[str, Any],
    methods_seed: int,
    alphas: np.ndarray,
    ks: list[int],
) -> tuple[int | None, float, int, dict[str, Any], list[dict[str, Any]]]:
    """Refit the representation in every fold; choose d / ridge alpha / kNN k.

    Validation rows are transformed with the same operator as test (Nyström,
    session-start Kalman filter, lag stack on the contiguous session).

    The fifth return is the per-d decoder-choice curve. It does not change
    the selected (d, alpha, k), which stay the Phase 3 joint-Ridge then k
    at the winning d.
    """
    split = cfg["split"]
    reducing = key not in ("raw", "raw_lag")
    nested = bool((cfg["representations"].get(key) or {}).get("nested"))
    dims = [int(d) for d in cfg["latent_dims"]]
    d_fit_grid = [int(cfg["nested_fit_d"])] if (nested and reducing) else (
        dims if reducing else [3]
    )
    idx_ok = np.where(train_ok)[0]
    n_fit = int(train_ok.sum())
    ytr = y[train_ok]
    folds = inner_cv_block_masks(
        decode_times[train_ok],
        n_blocks=int(split["inner_cv_blocks"]),
        gap_s=float(split["gap_s"]),
    )
    # d -> list of per-fold (Z_va, y_va, Z_tr, y_tr) after OOS transform
    fold_latents: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {
        (dd if reducing else 0): [] for dd in (dims if reducing else [0])
    }
    for tr, va in folds:
        fit_mask = np.zeros(len(train_ok), dtype=bool)
        fit_mask[idx_ok[tr]] = True
        for d_fit in d_fit_grid:
            model = _make_rep(key, int(d_fit), cfg, methods_seed, n_fit=int(fit_mask.sum()))
            Z = fit_transform_representation(key, model, X, fit_mask)
            if nested and reducing:
                for dd in dims:
                    fold_latents[dd].append(
                        (Z[idx_ok][va][:, :dd], ytr[va], Z[idx_ok][tr][:, :dd], ytr[tr])
                    )
            else:
                d_key = int(d_fit) if reducing else 0
                fold_latents[d_key].append(
                    (Z[idx_ok][va], ytr[va], Z[idx_ok][tr], ytr[tr])
                )

    def _ridge_med(d_key: int, alpha: float) -> float:
        meds = [
            np.median(np.linalg.norm(
                _fit_predict_ridge(Ztr, y_tr, Zva, alpha) - y_va, axis=1,
            ))
            for Zva, y_va, Ztr, y_tr in fold_latents[d_key]
        ]
        return float(np.median(meds))

    def _knn_med(d_key: int, k: int) -> float:
        meds = [
            np.median(np.linalg.norm(
                _fit_predict_knn(Ztr, y_tr, Zva, k) - y_va, axis=1,
            ))
            for Zva, y_va, Ztr, y_tr in fold_latents[d_key]
        ]
        return float(np.median(meds))

    if reducing:
        best_d, best_a, best_med = int(dims[0]), float(alphas[0]), np.inf
        for dd in dims:
            for a in alphas:
                med = _ridge_med(int(dd), float(a))
                if med < best_med or (med == best_med and (
                    dd < best_d or (dd == best_d and float(a) < best_a)
                )):
                    best_med, best_d, best_a = med, int(dd), float(a)
        primary_d: int | None = int(best_d)
        alpha = float(best_a)
        d_key = int(best_d)
    else:
        primary_d = None
        alpha = float(alphas[0])
        best_med = np.inf
        for a in alphas:
            med = _ridge_med(0, float(a))
            if med < best_med or (med == best_med and float(a) < alpha):
                best_med, alpha = med, float(a)
        d_key = 0

    best_k, best_k_med = int(ks[0]), np.inf
    for k in ks:
        med = _knn_med(d_key, int(k))
        if med < best_k_med or (med == best_k_med and int(k) < best_k):
            best_k, best_k_med = int(k), med
    per_d: list[dict[str, Any]] = []
    if reducing:
        for dd in dims:
            best_a_d, best_med_d = float(alphas[0]), np.inf
            for a in alphas:
                med = _ridge_med(int(dd), float(a))
                if med < best_med_d or (med == best_med_d and float(a) < best_a_d):
                    best_med_d, best_a_d = med, float(a)
            best_k_d, best_k_med_d = int(ks[0]), np.inf
            for k in ks:
                med = _knn_med(int(dd), int(k))
                if med < best_k_med_d or (med == best_k_med_d and int(k) < best_k_d):
                    best_k_d, best_k_med_d = int(k), med
            per_d.append({
                "d": int(dd),
                "ridge_alpha": float(best_a_d),
                "knn_k": int(best_k_d),
                "inner_cv_ridge_median": float(best_med_d),
                "inner_cv_knn_median": float(best_k_med_d),
            })
    return primary_d, alpha, best_k, {
        "inner_cv_ridge_median": best_med,
        "inner_cv_knn_median": best_k_med,
        "n_folds": len(folds),
        "refit_representation": True,
    }, per_d


def score_d_curve(
    key: str,
    X: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    cfg: dict[str, Any],
    methods_seed: int,
    per_d_cv: list[dict[str, Any]],
    primary_d: int,
    pred_r_sel: np.ndarray | None = None,
    pred_k_sel: np.ndarray | None = None,
    Z_fit: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Final full-train fit at every d. Reuse selected-d predictions when given."""
    nested = bool((cfg["representations"].get(key) or {}).get("nested"))
    yte = y[eval_mask]
    rows: list[dict[str, Any]] = []
    Z20 = Z_fit if (nested and Z_fit is not None) else None
    if nested and Z20 is None:
        model = _make_rep(
            key, int(cfg["nested_fit_d"]), cfg, methods_seed,
            n_fit=int(train_ok.sum()),
        )
        Z20 = fit_transform_representation(key, model, X, train_ok)
    for spec in per_d_cv:
        d = int(spec["d"])
        reuse = (
            pred_r_sel is not None
            and pred_k_sel is not None
            and d == int(primary_d)
        )
        if reuse:
            pred_r, pred_k = pred_r_sel, pred_k_sel
        elif nested:
            assert Z20 is not None
            Zd = Z20[:, :d]
            pred_r = _fit_predict_ridge(
                Zd[train_ok], y[train_ok], Zd[eval_mask], spec["ridge_alpha"],
            )
            pred_k = _fit_predict_knn(
                Zd[train_ok], y[train_ok], Zd[eval_mask], spec["knn_k"],
            )
        else:
            model = _make_rep(
                key, d, cfg, methods_seed, n_fit=int(train_ok.sum()),
            )
            Zd = fit_transform_representation(key, model, X, train_ok)
            pred_r = _fit_predict_ridge(
                Zd[train_ok], y[train_ok], Zd[eval_mask], spec["ridge_alpha"],
            )
            pred_k = _fit_predict_knn(
                Zd[train_ok], y[train_ok], Zd[eval_mask], spec["knn_k"],
            )
        ridge = _euclid(pred_r, yte)
        knn = _euclid(pred_k, yte)
        rows.append({
            "method": key,
            "d": d,
            "ridge_median": ridge["median"],
            "knn_median": knn["median"],
            "ridge_alpha": float(spec["ridge_alpha"]),
            "knn_k": int(spec["knn_k"]),
            "inner_cv_ridge_median": float(spec["inner_cv_ridge_median"]),
            "inner_cv_knn_median": float(spec["inner_cv_knn_median"]),
            "refit_representation": True,
            "selected": d == int(primary_d),
        })
    return rows


def write_d_sweep_json(
    path: Path,
    cfg: dict[str, Any],
    *,
    seed_index: int,
    spike_source: str,
    eval_index_hash: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = {
        r["method"]: {
            "d": int(r["d"]),
            "ridge_alpha": float(r["ridge_alpha"]),
            "knn_k": int(r["knn_k"]),
        }
        for r in rows if r.get("selected")
    }
    payload = {
        "seed_index": seed_index,
        "spike_source": spike_source,
        "config_sha256": cfg["config_sha256"],
        "eval_index_hash": eval_index_hash,
        "selection_rule": D_SWEEP_SELECTION_RULE,
        "rows": rows,
        "selected": selected,
    }
    _write_json(path, payload)
    return payload


def _unit_counts(units_df, unit_ids) -> dict[str, Any]:
    ids = {int(u) for u in unit_ids}
    sub = units_df.loc[units_df["unit_id"].astype(int).isin(ids)]
    by_ct: dict[str, int] = {}
    if len(sub):
        by_ct = {str(k): int(v) for k, v in sub["cell_type"].astype(str).value_counts().items()}
    by_reg: dict[str, int] = {}
    if len(sub) and "region" in sub.columns:
        by_reg = {str(k): int(v) for k, v in sub["region"].astype(str).value_counts().items()}
    return {
        "n_units": int(len(ids)),
        "n_units_by_cell_type": by_ct,
        "n_units_by_region": by_reg,
    }


def _occupancy_coverage(
    y_train: np.ndarray,
    y_test: np.ndarray,
    *,
    arena_cm: float,
    n_bins: int,
) -> dict[str, Any]:
    def _bins(y):
        ix = np.clip((y[:, 0] / arena_cm * n_bins).astype(int), 0, n_bins - 1)
        iy = np.clip((y[:, 1] / arena_cm * n_bins).astype(int), 0, n_bins - 1)
        return ix, iy

    trx, try_ = _bins(y_train)
    tex, tey = _bins(y_test)
    train_map = np.zeros((n_bins, n_bins), dtype=int)
    test_map = np.zeros((n_bins, n_bins), dtype=int)
    np.add.at(train_map, (trx, try_), 1)
    np.add.at(test_map, (tex, tey), 1)
    occupied = train_map > 0
    frac = float(np.mean(occupied[tex, tey])) if len(y_test) else float("nan")
    return {
        "occupancy_n_bins": int(n_bins),
        "fraction_test_in_train_occupied_bins": frac,
        "n_train_occupied_bins": int(occupied.sum()),
        "n_test_occupied_bins": int((test_map > 0).sum()),
        "train_occupancy": train_map.tolist(),
        "test_occupancy": test_map.tolist(),
    }


def _write_occupancy_maps(path: Path, cov: dict[str, Any], arena_cm: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    for ax, key, title in (
        (axes[0], "train_occupancy", "Train occupancy"),
        (axes[1], "test_occupancy", "Test occupancy"),
    ):
        arr = np.asarray(cov[key], dtype=float)
        im = ax.imshow(
            arr.T, origin="lower", cmap="mako" if False else "viridis",
            extent=(0, arena_cm, 0, arena_cm), aspect="equal",
        )
        ax.set_title(title)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _a13_null(
    Ztr: np.ndarray,
    Zte: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    cfg: dict[str, Any],
    *,
    ridge_alpha: float,
    knn_k: int,
) -> dict[str, Any]:
    """Circular-shift null; representation latents are fixed (decoder-only refit)."""
    a13 = cfg["phase3"]["a13"]
    session_s = float(cfg["session"]["session_s"])
    update_dt = float(cfg["features"]["update_dt"])
    n_shifts = int(a13["n_shifts"])
    lo = float(a13["shift_frac_lo"])
    hi = float(a13["shift_frac_hi"])
    fracs = np.linspace(lo, hi, n_shifts)
    ytr0, yte0 = y[train_ok], y[eval_mask]
    rows = []
    for f in fracs:
        k = int(round(f * session_s / update_dt))
        ys = np.roll(y, k, axis=0)
        ytr, yte = ys[train_ok], ys[eval_mask]
        floor = _euclid(
            np.repeat(np.mean(ytr, axis=0, keepdims=True), len(yte), axis=0), yte,
        )
        pred_r = _fit_predict_ridge(Ztr, ytr, Zte, ridge_alpha)
        pred_k = _fit_predict_knn(Ztr, ytr, Zte, knn_k)
        ridge = _euclid(pred_r, yte)
        knn = _euclid(pred_k, yte)
        rows.append({
            "frac": float(f),
            "shift_s": float(k * update_dt),
            "floor_median": floor["median"],
            "ridge_median": ridge["median"],
            "knn_median": knn["median"],
            "ridge_minus_floor": ridge["median"] - floor["median"],
            "knn_minus_floor": knn["median"] - floor["median"],
        })
    pass_min = float(a13["pass_min_cm"])
    ridge_delta = float(np.median([r["ridge_minus_floor"] for r in rows]))
    knn_delta = float(np.median([r["knn_minus_floor"] for r in rows]))
    return record_a13({
        "shifts": rows,
        "ridge_median_minus_floor": ridge_delta,
        "knn_median_minus_floor": knn_delta,
        "ridge_pass": ridge_delta >= pass_min,
        "knn_pass": knn_delta >= pass_min,
        "pass_min_cm": pass_min,
        "n_shifts": n_shifts,
        "unshifted_n": {"n_train": int(len(ytr0)), "n_eval": int(len(yte0))},
    })


def record_a13(a13: dict[str, Any]) -> dict[str, Any]:
    """Mark A13 PASS/FAIL. Never aborts the run (SPEC: unreliable, not hidden)."""
    out = dict(a13)
    failed = (not bool(out.get("ridge_pass"))) or (not bool(out.get("knn_pass")))
    out["status"] = "FAIL" if failed else "PASS"
    out["unreliable"] = bool(failed)
    return out


def analyze_source(
    cfg: dict[str, Any],
    sim_dir: Path,
    spike_source: str,
    streams: dict[str, int],
    seed_index: int,
) -> dict[str, Any]:
    out_dir = OUTPUT_ROOT / f"seed_{seed_index}" / spike_source
    cached_source = source_summary_reusable(out_dir, cfg)
    if cached_source is not None:
        print(
            f"  [{spike_source}] skip source "
            f"(JSON exists, config_sha256={cfg['config_sha256'][:12]}…)",
            flush=True,
        )
        return cached_source

    feat = cfg["features"]
    split = cfg["split"]
    dec = cfg["decoders"]
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
    from realtime.train_decoder import align_behavior_to_decoder_times
    beh = align_behavior_to_decoder_times(data["behavior_df"], decode_times)
    y = _position(beh)
    train_mask, test_mask = causal_train_test_split(
        decode_times, float(split["train_frac"]), gap_s=float(split["gap_s"]),
    )
    X, _ = _sqrt_zscore_train(X_counts, train_mask)
    n = len(decode_times)
    valid = {m: _valid_mask(m, n, cfg) for m in METHOD_KEYS}
    eval_mask = test_mask.copy()
    for m in METHOD_KEYS:
        eval_mask &= valid[m]
    # Identical training index set for every method (SPEC §4 / A1, A3).
    train_ok = train_mask.copy()
    for m in METHOD_KEYS:
        train_ok &= valid[m]

    alphas = np.logspace(
        np.log10(float(dec["ridge_alpha_lo"])),
        np.log10(float(dec["ridge_alpha_hi"])),
        int(dec["ridge_alpha_n"]),
    )
    methods_seed = int(streams["methods"])
    timings: dict[str, float] = {}
    results: list[dict[str, Any]] = []
    index_hashes = {
        "train": hash_train_indices(train_ok),
        "test": hash_train_indices(test_mask),
        "eval": hash_train_indices(eval_mask),
    }

    unit_counts = _unit_counts(data["units_df"], data["unit_ids"])
    cov = _occupancy_coverage(
        y[train_ok], y[eval_mask],
        arena_cm=float(cfg["session"]["arena_size_cm"]),
        n_bins=int(cfg["phase3"]["coverage"]["occupancy_n_bins"]),
    )
    _write_occupancy_maps(
        out_dir / "occupancy_maps.png", cov, float(cfg["session"]["arena_size_cm"]),
    )
    y_floor = np.mean(y[train_ok], axis=0, keepdims=True)
    floor_err = _euclid(np.repeat(y_floor, int(eval_mask.sum()), axis=0), y[eval_mask])
    latents: dict[str, np.ndarray] = {}
    a13_by_method: dict[str, Any] = {}
    pred_store: dict[str, np.ndarray] = {}
    pred_source: dict[str, str] = {}
    sweep_rows: list[dict[str, Any]] = []
    computed_reducing: set[str] = set()

    for key in METHOD_KEYS:
        cached = reusable_result_json(out_dir / f"{key}.json", cfg)
        if cached is not None:
            row = method_row_from_payload(cached)
            results.append(row)
            a13_by_method[key] = cached.get("a13") or {}
            timings[key] = float(cached.get("elapsed_s") or 0.0)
            print(
                f"  [{spike_source}] {key} skip "
                f"(JSON exists, config_sha256={cfg['config_sha256'][:12]}…)",
                flush=True,
            )
            continue
        print(f"  [{spike_source}] {key} …", flush=True)
        t_method = time.perf_counter()
        reducing = key not in ("raw", "raw_lag")
        primary_d, alpha, knn_k, cv_info, per_d_cv = _inner_cv_select(
            key, X, y, train_ok, decode_times, cfg, methods_seed, alphas,
            list(dec["knn_k"]),
        )
        d_final = int(primary_d) if reducing else 3
        nested = bool((cfg["representations"].get(key) or {}).get("nested"))
        d_fit = int(cfg["nested_fit_d"]) if (nested and reducing) else d_final
        model = _make_rep(key, d_fit, cfg, methods_seed, n_fit=int(train_ok.sum()))
        Z_fit = fit_transform_representation(key, model, X, train_ok)
        Z_all = Z_fit[:, : int(primary_d)] if (nested and reducing) else Z_fit
        Ztr, Zte = Z_all[train_ok], Z_all[eval_mask]
        ytr, yte = y[train_ok], y[eval_mask]
        pred_r = _fit_predict_ridge(Ztr, ytr, Zte, alpha)
        pred_k = _fit_predict_knn(Ztr, ytr, Zte, knn_k)
        # Stage 2b: these are the in-memory Phase 3 scores, not export reload.
        pred_store[f"pred_{key}_ridge"] = pred_r
        pred_store[f"pred_{key}_knn"] = pred_k
        pred_source[key] = "phase3_in_memory"
        latents[key] = Z_all
        model_dir = OUTPUT_ROOT / "models" / f"seed_{seed_index}" / spike_source / key
        if hasattr(model, "save"):
            try:
                model.save(model_dir)
            except Exception:
                pass
        a13 = _a13_null(
            Ztr, Zte, y, train_ok, eval_mask, cfg,
            ridge_alpha=float(alpha), knn_k=int(knn_k),
        )
        a13_by_method[key] = a13
        if a13.get("status") == "FAIL":
            print(
                f"  A13 FAIL {key} on {spike_source}: "
                f"ridge Δ={a13['ridge_median_minus_floor']:.3f} "
                f"knn Δ={a13['knn_median_minus_floor']:.3f} "
                f"(pass ≥ {a13['pass_min_cm']}; marked unreliable, continuing)",
                flush=True,
            )
        elapsed = time.perf_counter() - t_method
        timings[key] = elapsed
        row = {
            "method": key,
            "primary_d": primary_d,
            "ridge_alpha": alpha,
            "knn_k": knn_k,
            "ridge": _euclid(pred_r, yte),
            "knn": _euclid(pred_k, yte),
            "n_train": int(train_ok.sum()),
            "n_eval": int(eval_mask.sum()),
            "index_hashes": index_hashes,
            "elapsed_s": elapsed,
            "a13": {
                "ridge_median_minus_floor": a13["ridge_median_minus_floor"],
                "knn_median_minus_floor": a13["knn_median_minus_floor"],
                "ridge_pass": a13["ridge_pass"],
                "knn_pass": a13["knn_pass"],
                "status": a13.get("status"),
                "unreliable": a13.get("unreliable"),
            },
            **unit_counts,
            "coverage": {
                "fraction_test_in_train_occupied_bins": cov[
                    "fraction_test_in_train_occupied_bins"
                ],
            },
            **cv_info,
        }
        results.append(row)
        print(f"  [{spike_source}] {key} {elapsed:.1f}s  ridge_med={row['ridge']['median']:.3f}", flush=True)
        _write_json(out_dir / f"{key}.json", _result_payload(cfg, {
            "stage": "method",
            "seed_index": seed_index,
            "spike_source": spike_source,
            "seed_streams": streams,
            **row,
        }))
        if reducing:
            computed_reducing.add(key)
            sweep_rows.extend(score_d_curve(
                key, X, y, train_ok, eval_mask, cfg, methods_seed,
                per_d_cv, int(primary_d), pred_r, pred_k,
                Z_fit if nested else None,
            ))

    learning_curves: list[dict[str, Any]] = []
    if spike_source == cfg["phase3"]["learning_curve_source"]:
        idx = np.where(train_ok)[0]
        for frac in cfg["phase3"]["learning_curve_fracs"]:
            n_use = max(1, int(round(len(idx) * float(frac))))
            use = np.zeros(n, dtype=bool)
            use[idx[-n_use:]] = True
            for key in METHOD_KEYS:
                t_lc = time.perf_counter()
                reducing = key not in ("raw", "raw_lag")
                rec = next(r for r in results if r["method"] == key)
                d_final = int(rec["primary_d"]) if reducing else 3
                nested = bool((cfg["representations"].get(key) or {}).get("nested"))
                d_fit = int(cfg["nested_fit_d"]) if (nested and reducing) else d_final
                model = _make_rep(key, d_fit, cfg, methods_seed, n_fit=int(use.sum()))
                Z_all = fit_transform_representation(key, model, X, use)
                if nested and reducing:
                    Z_all = Z_all[:, : int(rec["primary_d"])]
                pred = _fit_predict_ridge(
                    Z_all[use], y[use], Z_all[eval_mask], float(rec["ridge_alpha"]),
                )
                learning_curves.append({
                    "method": key,
                    "frac": float(frac),
                    "n_train": int(use.sum()),
                    "portion": cfg["phase3"]["learning_curve_portion"],
                    "ridge": _euclid(pred, y[eval_mask]),
                    "elapsed_s": time.perf_counter() - t_lc,
                })

    payload = _result_payload(cfg, {
        "stage": "source",
        "seed_index": seed_index,
        "spike_source": spike_source,
        "seed_streams": streams,
        "timings_s": timings,
        "floor": floor_err,
        "a13": a13_by_method,
        "a13_failures": [
            k for k, rec in a13_by_method.items() if rec.get("status") == "FAIL"
        ],
        "n_train": int(train_ok.sum()),
        "n_eval": int(eval_mask.sum()),
        "index_hashes": index_hashes,
        "learning_curves": learning_curves,
        "methods": results,
        **unit_counts,
        "coverage": {
            "fraction_test_in_train_occupied_bins": cov[
                "fraction_test_in_train_occupied_bins"
            ],
            "n_train_occupied_bins": cov["n_train_occupied_bins"],
            "n_test_occupied_bins": cov["n_test_occupied_bins"],
            "occupancy_n_bins": cov["occupancy_n_bins"],
            "occupancy_map_png": str(out_dir / "occupancy_maps.png"),
        },
    })
    _write_json(out_dir / "source_summary.json", payload)
    if set(REDUCING_SWEEP) <= computed_reducing:
        write_d_sweep_json(
            out_dir / "d_sweep.json",
            cfg,
            seed_index=seed_index,
            spike_source=spike_source,
            eval_index_hash=index_hashes["eval"],
            rows=sweep_rows,
        )
    if len(pred_store) == 2 * len(METHOD_KEYS):
        from agents.quadrant_n5.export_predictions import write_predictions_npz

        write_predictions_npz(
            out_dir / "predictions.npz",
            cfg,
            {
                "decode_times": decode_times,
                "y": y,
                "eval_mask": eval_mask,
                "index_hashes": index_hashes,
            },
            pred_store,
            method_source=pred_source,
        )
    return payload


def run_seed_pilot(seed_index: int = 0) -> dict[str, Any]:
    cfg = load_quadrant_n5_yaml()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    sim_dir, streams, sim_s = generate_seed_dataset(cfg, seed_index)
    source_rows = []
    for src in ("sorted", "ground_truth"):
        print(f"analyzing {src}", flush=True)
        t_src = time.perf_counter()
        row = analyze_source(cfg, sim_dir, src, streams, seed_index)
        row["source_elapsed_s"] = time.perf_counter() - t_src
        source_rows.append(row)
    total = time.perf_counter() - t_all
    n_seeds = int(cfg["seeds"]["n_seeds"])
    projection = {
        "seed0_total_s": total,
        "seed0_sim_s": sim_s,
        "seed0_analysis_s": total - sim_s,
        "projected_5_seeds_s": sim_s * n_seeds + (total - sim_s) * n_seeds * 2 / 2,
        "note": (
            "Projection: n_seeds * sim + n_seeds * 2 sources * (seed0_analysis/2). "
            "seed0 analysis already includes both sources."
        ),
        "projected_5_seeds_s_corrected": sim_s * n_seeds + (total - sim_s) * n_seeds,
    }
    a13_failures = [
        f"{r['spike_source']}:{k}"
        for r in source_rows
        for k in (r.get("a13_failures") or [])
    ]
    summary = _result_payload(cfg, {
        "stage": "pilot",
        "seed_index": seed_index,
        "seed_streams": streams,
        "sim_dir": str(sim_dir),
        "elapsed_s": total,
        "sim_elapsed_s": sim_s,
        "projection": projection,
        "a13_failures": a13_failures,
        "exit_code": 0,
        "sources": [
            {
                "spike_source": r["spike_source"],
                "timings_s": r.get("timings_s"),
                "a13_failures": r.get("a13_failures") or [],
            }
            for r in source_rows
        ],
    })
    _write_json(OUTPUT_ROOT / f"seed_{seed_index}" / "pilot_summary.json", summary)
    return summary


if __name__ == "__main__":
    import sys
    import traceback

    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    try:
        out = run_seed_pilot(idx)
    except Exception:
        traceback.print_exc()
        print(f"seed{idx} crashed", flush=True)
        print("EXIT:1", flush=True)
        raise SystemExit(1)
    print(json.dumps(out["projection"], indent=2))
    print(f"seed{idx} total {out['elapsed_s']:.1f}s  sim {out['sim_elapsed_s']:.1f}s")
    code = int(out.get("exit_code", 0))
    print(f"EXIT:{code}", flush=True)
    raise SystemExit(code)
