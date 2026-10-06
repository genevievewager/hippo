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

# Real-data segment edge trims (fit/eval). Warm-up neural history uses segment start.
SEGMENT_TRIM_START_S = 60.0
SEGMENT_TRIM_END_S = 10.0


def prepared_source_bundle(
    *,
    X_counts: np.ndarray,
    y: np.ndarray,
    decode_times: np.ndarray,
    unit_ids: list[int] | np.ndarray,
    units_df,
    arena_cm: float,
    segment_t0: float,
    segment_t1: float,
    target_valid: np.ndarray | None = None,
    arena_width_cm: float | None = None,
    arena_height_cm: float | None = None,
    apply_segment_trims: bool = True,
    y_is_room_local: bool = True,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the prepared-bundle dict accepted by ``analyze_source``."""
    X_counts = np.asarray(X_counts, dtype=float)
    y = np.asarray(y, dtype=float)
    decode_times = np.asarray(decode_times, dtype=float)
    if X_counts.ndim != 2 or y.ndim != 2 or y.shape[1] != 2:
        raise ValueError("X_counts must be (T, U) and y (T, 2)")
    if len(decode_times) != X_counts.shape[0] or len(decode_times) != y.shape[0]:
        raise ValueError("decode_times, X_counts, and y length mismatch")
    if float(segment_t1) <= float(segment_t0):
        raise ValueError("segment_t1 must be > segment_t0")
    out: dict[str, Any] = {
        "X_counts": X_counts,
        "y": y,
        "decode_times": decode_times,
        "unit_ids": [int(u) for u in np.asarray(unit_ids).tolist()],
        "units_df": units_df,
        "arena_cm": float(arena_cm),
        "segment_t0": float(segment_t0),
        "segment_t1": float(segment_t1),
        "apply_segment_trims": bool(apply_segment_trims),
        "y_is_room_local": bool(y_is_room_local),
        "target_valid": (
            np.ones(len(decode_times), dtype=bool)
            if target_valid is None
            else np.asarray(target_valid, dtype=bool)
        ),
        "arena_width_cm": float(
            arena_width_cm if arena_width_cm is not None else arena_cm
        ),
        "arena_height_cm": float(
            arena_height_cm if arena_height_cm is not None else arena_cm
        ),
        "meta": dict(meta or {}),
    }
    if out["target_valid"].shape != decode_times.shape:
        raise ValueError("target_valid must match decode_times shape")
    return out


def segment_retained_mask(
    decode_times: np.ndarray,
    segment_t0: float,
    segment_t1: float,
    *,
    trim_start_s: float = SEGMENT_TRIM_START_S,
    trim_end_s: float = SEGMENT_TRIM_END_S,
) -> np.ndarray:
    """True on times kept for fit/eval after segment edge trims."""
    t = np.asarray(decode_times, dtype=float)
    return (t >= float(segment_t0) + float(trim_start_s)) & (
        t < float(segment_t1) - float(trim_end_s)
    )


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


def require_clean_git_for_real_data() -> None:
    """Real-data runs must start from a clean tree so resume keys stay stable."""
    if _git_dirty():
        raise RuntimeError(
            "Refusing real-data run: working tree is dirty. "
            "Commit or stash first so resume keys "
            "(config_sha256, git_sha, dirty_tree) cannot mix code versions."
        )


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
    "n_train", "n_eval", "index_hashes", "elapsed_s", "fit_s", "transform_s",
    "a13",
    "n_units", "n_units_by_cell_type", "n_units_by_region", "coverage",
    "inner_cv_ridge_median", "inner_cv_knn_median", "n_folds",
    "refit_representation",
)


def reusable_result_json(path: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Return payload if resume key matches (config hash + git SHA + dirty flag)."""
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
    if rec.get("git_sha") != _git_sha():
        return None
    if bool(rec.get("dirty_tree")) != bool(_git_dirty()):
        return None
    return rec


def method_row_from_payload(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: rec[k] for k in _METHOD_ROW_KEYS if k in rec}


def source_summary_reusable(out_dir: Path, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Skip a whole source when every method JSON and the summary match the resume key."""
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


def fit_transform_representation_timed(name: str, model, X, fit_mask):
    """Same as ``fit_transform_representation`` with fit/transform wall times."""
    X = np.asarray(X, dtype=float)
    t0 = time.perf_counter()
    model.fit(X[np.asarray(fit_mask, dtype=bool)])
    fit_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    if name == "lds":
        Z = model.transform(X, causal=True, reset=True)
        assert_readout_latents_are_filtered(model.model_, Z, X)
    else:
        Z = model.transform(X)
    transform_s = time.perf_counter() - t1
    return Z, float(fit_s), float(transform_s)


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
    observed_ridge_median: float | None = None,
    observed_knn_median: float | None = None,
    floor_median: float | None = None,
) -> dict[str, Any]:
    """Circular-shift null; representation latents are fixed (decoder-only refit).

    Pass requires both:
    1. median(null_error − floor) ≥ pass_min_cm (null does not beat chance), and
    2. observed (unshifted) error strictly below the chance floor when provided.
    """
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
        # Real-data invalid targets are NaN outside target_valid; a roll can
        # move them onto train/eval indices. Drop non-finite pairs (no-op on sim).
        tr_ok = np.isfinite(ytr).all(axis=1)
        te_ok = np.isfinite(yte).all(axis=1)
        if not tr_ok.any() or not te_ok.any():
            continue
        ytr_f, yte_f = ytr[tr_ok], yte[te_ok]
        Ztr_f, Zte_f = Ztr[tr_ok], Zte[te_ok]
        floor = _euclid(
            np.repeat(np.mean(ytr_f, axis=0, keepdims=True), len(yte_f), axis=0),
            yte_f,
        )
        pred_r = _fit_predict_ridge(Ztr_f, ytr_f, Zte_f, ridge_alpha)
        pred_k = _fit_predict_knn(Ztr_f, ytr_f, Zte_f, knn_k)
        ridge = _euclid(pred_r, yte_f)
        knn = _euclid(pred_k, yte_f)
        rows.append({
            "frac": float(f),
            "shift_s": float(k * update_dt),
            "floor_median": floor["median"],
            "ridge_median": ridge["median"],
            "knn_median": knn["median"],
            "ridge_minus_floor": ridge["median"] - floor["median"],
            "knn_minus_floor": knn["median"] - floor["median"],
        })
    if not rows:
        return record_a13({
            "shifts": [],
            "ridge_median_minus_floor": float("nan"),
            "knn_median_minus_floor": float("nan"),
            "ridge_pass": False,
            "knn_pass": False,
            "pass_min_cm": float(a13["pass_min_cm"]),
            "n_shifts": 0,
            "unshifted_n": {"n_train": int(len(ytr0)), "n_eval": int(len(yte0))},
            "observed_below_floor_ridge": False,
            "observed_below_floor_knn": False,
        })
    pass_min = float(a13["pass_min_cm"])
    ridge_delta = float(np.median([r["ridge_minus_floor"] for r in rows]))
    knn_delta = float(np.median([r["knn_minus_floor"] for r in rows]))
    ridge_null_ok = ridge_delta >= pass_min
    knn_null_ok = knn_delta >= pass_min
    # Observed-below-floor gate (required when callers provide the medians).
    if observed_ridge_median is None or floor_median is None:
        ridge_obs_ok = True
    else:
        ridge_obs_ok = float(observed_ridge_median) < float(floor_median)
    if observed_knn_median is None or floor_median is None:
        knn_obs_ok = True
    else:
        knn_obs_ok = float(observed_knn_median) < float(floor_median)
    return record_a13({
        "shifts": rows,
        "ridge_median_minus_floor": ridge_delta,
        "knn_median_minus_floor": knn_delta,
        "ridge_pass": bool(ridge_null_ok and ridge_obs_ok),
        "knn_pass": bool(knn_null_ok and knn_obs_ok),
        "pass_min_cm": pass_min,
        "n_shifts": n_shifts,
        "unshifted_n": {"n_train": int(len(ytr0)), "n_eval": int(len(yte0))},
        "observed_ridge_median": (
            None if observed_ridge_median is None else float(observed_ridge_median)
        ),
        "observed_knn_median": (
            None if observed_knn_median is None else float(observed_knn_median)
        ),
        "floor_median": None if floor_median is None else float(floor_median),
        "observed_below_floor_ridge": bool(ridge_obs_ok),
        "observed_below_floor_knn": bool(knn_obs_ok),
        "null_pass_ridge": bool(ridge_null_ok),
        "null_pass_knn": bool(knn_null_ok),
    })


def record_a13(a13: dict[str, Any]) -> dict[str, Any]:
    """Mark A13 PASS/FAIL. Never aborts the run (SPEC: unreliable, not hidden)."""
    out = dict(a13)
    failed = (not bool(out.get("ridge_pass"))) or (not bool(out.get("knn_pass")))
    out["status"] = "FAIL" if failed else "PASS"
    out["unreliable"] = bool(failed)
    return out


def _load_arrays_for_analyze_source(
    cfg: dict[str, Any],
    sim_dir: Path | None,
    spike_source: str,
    *,
    bundle: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any, list[int], dict[str, Any]]:
    """Return X_counts, y, decode_times, units_df, unit_ids, load_meta.

    Sim path (bundle is None) matches the historical loader. Bundle path
    supplies arrays directly and records segment warm-up bounds in load_meta.
    """
    if bundle is not None:
        X_counts = np.asarray(bundle["X_counts"], dtype=float)
        y = np.asarray(bundle["y"], dtype=float)
        decode_times = np.asarray(bundle["decode_times"], dtype=float)
        unit_ids = [int(u) for u in bundle["unit_ids"]]
        units_df = bundle["units_df"]
        meta = {
            "source": "bundle",
            "segment_t0": float(bundle["segment_t0"]),
            "segment_t1": float(bundle["segment_t1"]),
            "apply_segment_trims": bool(bundle.get("apply_segment_trims", True)),
            "y_is_room_local": bool(bundle.get("y_is_room_local", True)),
            "target_valid": np.asarray(bundle["target_valid"], dtype=bool),
            "arena_cm": float(bundle["arena_cm"]),
            "arena_width_cm": float(bundle["arena_width_cm"]),
            "arena_height_cm": float(bundle["arena_height_cm"]),
            "bundle_meta": dict(bundle.get("meta") or {}),
        }
        return X_counts, y, decode_times, units_df, unit_ids, meta

    if sim_dir is None:
        raise ValueError("analyze_source requires sim_dir or bundle")
    data = load_simulation_data(
        sim_dir, spike_source,
        include_regions=list(cfg["unit_inclusion"]["regions"]),
    )
    behavior_times = extract_behavior_times(data["behavior_df"])
    update_dt = float(cfg["features"]["update_dt"])
    W = float(cfg["features"]["window_s"])
    decode_times = make_decode_times(
        data["session_duration"], W, update_dt, behavior_times=behavior_times,
    )
    X_counts = build_causal_spike_matrix(
        data["spikes_df"], data["unit_ids"], decode_times, W,
    )
    from realtime.train_decoder import align_behavior_to_decoder_times
    beh = align_behavior_to_decoder_times(data["behavior_df"], decode_times)
    y = _position(beh)
    meta = {
        "source": "sim_dir",
        "segment_t0": None,
        "segment_t1": None,
        "target_valid": np.ones(len(decode_times), dtype=bool),
        "arena_cm": float(cfg["session"]["arena_size_cm"]),
        "arena_width_cm": float(cfg["session"]["arena_size_cm"]),
        "arena_height_cm": float(cfg["session"]["arena_size_cm"]),
        "bundle_meta": {},
    }
    return (
        X_counts, y, decode_times, data["units_df"],
        list(int(u) for u in data["unit_ids"]), meta,
    )


def _write_figure_contract(
    out_dir: Path,
    *,
    cfg: dict[str, Any],
    decode_times: np.ndarray,
    y: np.ndarray,
    target_valid: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    y_floor: np.ndarray,
    pred_store: dict[str, np.ndarray],
    latents: dict[str, np.ndarray],
    load_meta: dict[str, Any],
    keys: tuple[str, ...],
    a13_by_method: dict[str, Any],
) -> Path:
    """Persist figure-data-contract arrays under out_dir/figure_contract/."""
    dest = Path(out_dir) / "figure_contract"
    dest.mkdir(parents=True, exist_ok=True)
    retained = np.ones(len(decode_times), dtype=bool)
    if load_meta.get("source") == "bundle" and load_meta.get("apply_segment_trims", True):
        retained = segment_retained_mask(
            decode_times,
            float(load_meta["segment_t0"]),
            float(load_meta["segment_t1"]),
        )
    # Floor prediction on eval indices only (broadcast train-mean).
    floor_pred = np.repeat(np.asarray(y_floor, dtype=float), int(eval_mask.sum()), axis=0)
    np.savez_compressed(
        dest / "arrays.npz",
        decode_times=np.asarray(decode_times, dtype=float),
        y=np.asarray(y, dtype=float),
        valid=np.asarray(target_valid, dtype=bool),
        retained=np.asarray(retained, dtype=bool),
        train_mask=np.asarray(train_mask, dtype=bool),
        test_mask=np.asarray(test_mask, dtype=bool),
        train_ok=np.asarray(train_ok, dtype=bool),
        eval_mask=np.asarray(eval_mask, dtype=bool),
        y_floor=np.asarray(y_floor, dtype=float),
        floor_pred_eval=floor_pred,
        segment_t0=np.asarray(
            [load_meta.get("segment_t0")], dtype=float,
        ) if load_meta.get("segment_t0") is not None else np.asarray([np.nan]),
        segment_t1=np.asarray(
            [load_meta.get("segment_t1")], dtype=float,
        ) if load_meta.get("segment_t1") is not None else np.asarray([np.nan]),
        trim_start_s=np.asarray([
            SEGMENT_TRIM_START_S
            if load_meta.get("source") == "bundle"
            and load_meta.get("apply_segment_trims", True)
            else 0.0
        ]),
        trim_end_s=np.asarray([
            SEGMENT_TRIM_END_S
            if load_meta.get("source") == "bundle"
            and load_meta.get("apply_segment_trims", True)
            else 0.0
        ]),
    )
    pred_payload = {
        "eval_mask": np.asarray(eval_mask, dtype=bool),
        "decode_times": np.asarray(decode_times, dtype=float),
        "y_true": np.asarray(y[eval_mask], dtype=float),
    }
    pred_payload.update({k: np.asarray(v, dtype=float) for k, v in pred_store.items()})
    np.savez_compressed(dest / "predictions.npz", **pred_payload)
    if latents:
        # Subsample rule for large latents: keep every step (document in meta).
        np.savez_compressed(
            dest / "latents.npz",
            **{f"Z_{k}": np.asarray(v, dtype=float) for k, v in latents.items()},
            subsample_rule=np.asarray(["full_session_no_subsample"]),
        )
    _write_json(dest / "a13.json", {k: v for k, v in a13_by_method.items()})
    _write_json(dest / "provenance.json", {
        "config_sha256": cfg.get("config_sha256"),
        "methods": list(keys),
        "bundle_meta": load_meta.get("bundle_meta") or {},
        "source": load_meta.get("source"),
        "git_sha": _git_sha(),
        "dirty_tree": _git_dirty(),
    })
    return dest


def analyze_source(
    cfg: dict[str, Any],
    sim_dir: Path | None,
    spike_source: str,
    streams: dict[str, int],
    seed_index: int,
    *,
    bundle: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    method_keys: tuple[str, ...] | list[str] | None = None,
    save_figure_contract: bool = False,
) -> dict[str, Any]:
    """Run the frozen method grid on a sim directory or a prepared bundle.

    When ``bundle`` is provided, ``sim_dir`` may be ``None``. Segment edge
    trims (60 s / 10 s) apply only on the bundle path; the sim path is
    unchanged.
    """
    if spike_source == "real":
        require_clean_git_for_real_data()
    keys = tuple(method_keys) if method_keys is not None else METHOD_KEYS
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = OUTPUT_ROOT / f"seed_{seed_index}" / spike_source
    # Partial method grids must not be skipped via the full-METHOD_KEYS cache.
    if method_keys is None and bundle is None:
        cached_source = source_summary_reusable(out_dir, cfg)
        if cached_source is not None:
            print(
                f"  [{spike_source}] skip source "
                f"(JSON exists, config_sha256={cfg['config_sha256'][:12]}…, "
                f"git={(_git_sha() or '?')[:12]}…, dirty={_git_dirty()})",
                flush=True,
            )
            return cached_source

    feat = cfg["features"]
    split = cfg["split"]
    dec = cfg["decoders"]
    X_counts, y, decode_times, units_df, unit_ids, load_meta = (
        _load_arrays_for_analyze_source(
            cfg, sim_dir, spike_source, bundle=bundle,
        )
    )
    train_mask, test_mask = causal_train_test_split(
        decode_times, float(split["train_frac"]), gap_s=float(split["gap_s"]),
    )
    # Bundle-only (when enabled): drop settling / removal edges from fit/eval.
    # Neural arrays still begin at segment start so LDS / raw_lag can warm up.
    if load_meta["source"] == "bundle" and load_meta.get("apply_segment_trims", True):
        retained = segment_retained_mask(
            decode_times,
            float(load_meta["segment_t0"]),
            float(load_meta["segment_t1"]),
        )
        train_mask = train_mask & retained
        test_mask = test_mask & retained
        # Overlay session length for A13 shift scaling on this segment.
        cfg = dict(cfg)
        cfg["session"] = dict(cfg["session"])
        cfg["session"]["session_s"] = float(
            load_meta["segment_t1"] - load_meta["segment_t0"]
        )
    target_valid = np.asarray(load_meta["target_valid"], dtype=bool)
    train_mask = train_mask & target_valid
    test_mask = test_mask & target_valid
    # Split masks after segment trims + target validity (before method history loss).
    train_mask_split = train_mask.copy()
    test_mask_split = test_mask.copy()

    X, _ = _sqrt_zscore_train(X_counts, train_mask)
    n = len(decode_times)
    valid = {m: _valid_mask(m, n, cfg) for m in keys}
    eval_mask = test_mask.copy()
    for m in keys:
        eval_mask &= valid[m]
    # Identical training index set for every method (SPEC §4 / A1, A3).
    train_ok = train_mask.copy()
    for m in keys:
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

    unit_counts = _unit_counts(units_df, unit_ids)
    arena_cm = float(load_meta["arena_cm"])
    y_occ = np.asarray(y, dtype=float).copy()
    if load_meta["source"] == "bundle" and load_meta.get("y_is_room_local", True):
        # Room-local (centre origin) → occupancy bins in [0, arena_*].
        y_occ[:, 0] = y_occ[:, 0] + 0.5 * float(load_meta["arena_width_cm"])
        y_occ[:, 1] = y_occ[:, 1] + 0.5 * float(load_meta["arena_height_cm"])
    cov = _occupancy_coverage(
        y_occ[train_ok], y_occ[eval_mask],
        arena_cm=arena_cm,
        n_bins=int(cfg["phase3"]["coverage"]["occupancy_n_bins"]),
    )
    _write_occupancy_maps(
        out_dir / "occupancy_maps.png", cov, arena_cm,
    )
    y_floor = np.mean(y[train_ok], axis=0, keepdims=True)
    floor_err = _euclid(np.repeat(y_floor, int(eval_mask.sum()), axis=0), y[eval_mask])
    latents: dict[str, np.ndarray] = {}
    a13_by_method: dict[str, Any] = {}
    pred_store: dict[str, np.ndarray] = {}
    pred_source: dict[str, str] = {}
    sweep_rows: list[dict[str, Any]] = []
    computed_reducing: set[str] = set()

    need_contract_arrays = bool(
        save_figure_contract or load_meta.get("source") == "bundle"
    )

    def _final_fit_predict(
        key: str,
        primary_d: int | None,
        alpha: float,
        knn_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
        """Return (Z_all, Z_fit, pred_r, pred_k, fit_s, transform_s).

        ``Z_fit`` is the un-sliced nested fit (or same as ``Z_all``).
        """
        reducing = key not in ("raw", "raw_lag")
        d_final = int(primary_d) if reducing else 3
        nested = bool((cfg["representations"].get(key) or {}).get("nested"))
        d_fit = int(cfg["nested_fit_d"]) if (nested and reducing) else d_final
        model = _make_rep(key, d_fit, cfg, methods_seed, n_fit=int(train_ok.sum()))
        Z_fit, fit_s, transform_s = fit_transform_representation_timed(
            key, model, X, train_ok,
        )
        Z_all = Z_fit[:, : int(primary_d)] if (nested and reducing) else Z_fit
        Ztr, Zte = Z_all[train_ok], Z_all[eval_mask]
        ytr = y[train_ok]
        pred_r = _fit_predict_ridge(Ztr, ytr, Zte, alpha)
        pred_k = _fit_predict_knn(Ztr, ytr, Zte, knn_k)
        model_dir = OUTPUT_ROOT / "models" / f"seed_{seed_index}" / spike_source / key
        if hasattr(model, "save"):
            try:
                model.save(model_dir)
            except Exception:
                pass
        return Z_all, Z_fit, pred_r, pred_k, fit_s, transform_s

    for key in keys:
        cached = reusable_result_json(out_dir / f"{key}.json", cfg)
        if cached is not None:
            row = method_row_from_payload(cached)
            results.append(row)
            a13_by_method[key] = cached.get("a13") or {}
            timings[key] = float(cached.get("elapsed_s") or 0.0)
            print(
                f"  [{spike_source}] {key} skip "
                f"(JSON exists, config_sha256={cfg['config_sha256'][:12]}…, "
                f"git={(_git_sha() or '?')[:12]}…, dirty={_git_dirty()})",
                flush=True,
            )
            # Refit final latents/preds for the figure contract (no CV).
            if need_contract_arrays:
                reducing = key not in ("raw", "raw_lag")
                Z_all, _Z_fit, pred_r, pred_k, fit_s, transform_s = (
                    _final_fit_predict(
                        key,
                        cached.get("primary_d"),
                        float(cached["ridge_alpha"]),
                        int(cached["knn_k"]),
                    )
                )
                pred_store[f"pred_{key}_ridge"] = pred_r
                pred_store[f"pred_{key}_knn"] = pred_k
                pred_source[key] = "phase3_contract_refit"
                latents[key] = Z_all
                row["fit_s"] = fit_s
                row["transform_s"] = transform_s
                if reducing:
                    computed_reducing.add(key)
            continue
        print(f"  [{spike_source}] {key} …", flush=True)
        t_method = time.perf_counter()
        reducing = key not in ("raw", "raw_lag")
        primary_d, alpha, knn_k, cv_info, per_d_cv = _inner_cv_select(
            key, X, y, train_ok, decode_times, cfg, methods_seed, alphas,
            list(dec["knn_k"]),
        )
        Z_all, Z_fit, pred_r, pred_k, fit_s, transform_s = _final_fit_predict(
            key, primary_d, float(alpha), int(knn_k),
        )
        Ztr, Zte = Z_all[train_ok], Z_all[eval_mask]
        ytr, yte = y[train_ok], y[eval_mask]
        # Stage 2b: these are the in-memory Phase 3 scores, not export reload.
        pred_store[f"pred_{key}_ridge"] = pred_r
        pred_store[f"pred_{key}_knn"] = pred_k
        pred_source[key] = "phase3_in_memory"
        latents[key] = Z_all
        ridge_stats = _euclid(pred_r, yte)
        knn_stats = _euclid(pred_k, yte)
        a13 = _a13_null(
            Ztr, Zte, y, train_ok, eval_mask, cfg,
            ridge_alpha=float(alpha), knn_k=int(knn_k),
            observed_ridge_median=float(ridge_stats["median"]),
            observed_knn_median=float(knn_stats["median"]),
            floor_median=float(floor_err["median"]),
        )
        a13_by_method[key] = a13
        if a13.get("status") == "FAIL":
            print(
                f"  A13 FAIL {key} on {spike_source}: "
                f"ridge Δ={a13['ridge_median_minus_floor']:.3f} "
                f"knn Δ={a13['knn_median_minus_floor']:.3f} "
                f"(null ≥ {a13['pass_min_cm']}; obs<floor "
                f"r={a13.get('observed_below_floor_ridge')} "
                f"k={a13.get('observed_below_floor_knn')}; "
                f"marked unreliable, continuing)",
                flush=True,
            )
        elapsed = time.perf_counter() - t_method
        timings[key] = elapsed
        nested = bool((cfg["representations"].get(key) or {}).get("nested"))
        row = {
            "method": key,
            "primary_d": primary_d,
            "ridge_alpha": alpha,
            "knn_k": knn_k,
            "ridge": ridge_stats,
            "knn": knn_stats,
            "n_train": int(train_ok.sum()),
            "n_eval": int(eval_mask.sum()),
            "index_hashes": index_hashes,
            "elapsed_s": elapsed,
            "fit_s": fit_s,
            "transform_s": transform_s,
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
            for key in keys:
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

    extra: dict[str, Any] = {
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
    }
    if load_meta["source"] == "bundle":
        if load_meta.get("apply_segment_trims", True):
            retained = segment_retained_mask(
                decode_times,
                float(load_meta["segment_t0"]),
                float(load_meta["segment_t1"]),
            )
        else:
            retained = np.ones(len(decode_times), dtype=bool)
        extra["segment"] = {
            "t0": load_meta["segment_t0"],
            "t1": load_meta["segment_t1"],
            "trim_start_s": (
                SEGMENT_TRIM_START_S
                if load_meta.get("apply_segment_trims", True)
                else 0.0
            ),
            "trim_end_s": (
                SEGMENT_TRIM_END_S
                if load_meta.get("apply_segment_trims", True)
                else 0.0
            ),
            "n_retained": int(retained.sum()),
            "dropped_valid_fraction": float(
                1.0 - np.mean(target_valid[retained])
            ) if retained.any() else float("nan"),
            "bundle_meta": load_meta.get("bundle_meta") or {},
        }
    payload = _result_payload(cfg, extra)
    _write_json(out_dir / "source_summary.json", payload)
    if computed_reducing and sweep_rows:
        d_sweep_path = out_dir / "d_sweep.json"
        if set(REDUCING_SWEEP) <= computed_reducing:
            write_d_sweep_json(
                d_sweep_path,
                cfg,
                seed_index=seed_index,
                spike_source=spike_source,
                eval_index_hash=index_hashes["eval"],
                rows=sweep_rows,
            )
        elif method_keys is not None:
            # Partial method grid: replace only the methods just computed.
            old_rows: list[dict[str, Any]] = []
            if d_sweep_path.is_file():
                try:
                    old_rows = list(json.loads(d_sweep_path.read_text()).get("rows") or [])
                except Exception:
                    old_rows = []
            kept = [r for r in old_rows if r.get("method") not in computed_reducing]
            write_d_sweep_json(
                d_sweep_path,
                cfg,
                seed_index=seed_index,
                spike_source=spike_source,
                eval_index_hash=index_hashes["eval"],
                rows=kept + list(sweep_rows),
            )
    if len(pred_store) == 2 * len(keys):
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
    if save_figure_contract or load_meta.get("source") == "bundle":
        # Only write contract when we actually computed predictions this run
        # (skip-only resumes leave pred_store empty).
        if pred_store:
            contract_dir = _write_figure_contract(
                out_dir,
                cfg=cfg,
                decode_times=decode_times,
                y=y,
                target_valid=target_valid,
                train_mask=train_mask_split,
                test_mask=test_mask_split,
                train_ok=train_ok,
                eval_mask=eval_mask,
                y_floor=y_floor,
                pred_store=pred_store,
                latents=latents,
                load_meta=load_meta,
                keys=keys,
                a13_by_method=a13_by_method,
            )
            payload["figure_contract_dir"] = str(contract_dir)
            _write_json(out_dir / "source_summary.json", payload)
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
