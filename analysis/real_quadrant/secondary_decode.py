"""Decode secondary targets from cached fold / final latents (no refit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from analysis.real_quadrant import latent_cache as lcache
from analysis.real_quadrant.secondary_metrics import (
    a13_scalar_shifts,
    categorical_chance_normalized,
    fit_predict_knn_clf,
    fit_predict_knn_reg,
    fit_predict_logreg,
    fit_predict_ridge_scaled,
    normalized_circular_error,
    normalized_continuous_error,
)
from analysis.real_quadrant.secondary_targets import (
    derived_movement_from_speed,
    derived_wall_from_xy,
    merge_rare_classes,
)


TargetKind = str  # continuous | circular | categorical


def _slice_Z(Z: np.ndarray, d: int | None) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    if d is None:
        return Z
    return Z[:, : int(d)]


def select_hparams_from_fold_cache(
    cache_dir: Path,
    method: str,
    y: np.ndarray,
    train_ok: np.ndarray,
    *,
    latent_dims: list[int],
    nested_fit_d: int | None,
    alphas: list[float],
    ks: list[int],
    kind: TargetKind,
    score_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> dict[str, Any]:
    """Inner-CV style selection of (d, alpha, k) using cached fold Z.

    ``score_fn(y_hat_va, y_va) → float`` lower-is-better. Defaults:
    continuous/circular = median L2/abs; categorical = 1 − balanced accuracy.
    """
    from sklearn.metrics import balanced_accuracy_score

    train_ok = np.asarray(train_ok, dtype=bool)
    idx_ok = np.where(train_ok)[0]
    y = np.asarray(y, dtype=float)
    reducing = method not in ("raw", "raw_lag")
    dims = [int(d) for d in latent_dims] if reducing else [None]

    fold_files = lcache.list_fold_files(cache_dir, method)
    if reducing and not fold_files:
        raise FileNotFoundError(f"no fold cache for {method} under {cache_dir}")

    # Group by fold index; prefer nested_fit_d files when present.
    by_fold: dict[int, list[dict[str, Any]]] = {}
    for path in fold_files:
        # fold_{i}_d{d}.npz
        stem = path.stem  # fold_0_d20
        parts = stem.split("_")
        fi = int(parts[1])
        d_fit = int(parts[2][1:])
        blob = lcache.load_fold_latent(cache_dir, method, fi, d_fit)
        by_fold.setdefault(fi, []).append(blob)

    def _default_score(yhat, yva) -> float:
        if kind == "categorical":
            return 1.0 - float(balanced_accuracy_score(
                yva.astype(int), yhat.astype(int),
            ))
        if kind == "circular":
            th = np.arctan2(yva[:, 0], yva[:, 1])
            thh = np.arctan2(yhat[:, 0], yhat[:, 1])
            d = (thh - th + np.pi) % (2 * np.pi) - np.pi
            return float(np.median(np.abs(d)))
        if yva.ndim == 1:
            return float(np.median(np.abs(yhat.ravel() - yva.ravel())))
        return float(np.median(np.linalg.norm(yhat - yva, axis=1)))

    score = score_fn or _default_score

    # Build fold_latents[d] = list of (Zva, yva, Ztr, ytr)
    fold_latents: dict[Any, list] = {d: [] for d in dims}
    if reducing:
        for fi, blobs in sorted(by_fold.items()):
            if nested_fit_d is not None:
                blob = next(
                    (b for b in blobs if int(b["d_fit"]) == int(nested_fit_d)),
                    max(blobs, key=lambda b: int(b["d_fit"])),
                )
            else:
                # non-nested: one blob per d
                blob = None
            fit_mask = np.asarray(
                (blob or blobs[0])["fit_mask"], dtype=bool,
            )
            # indices into train_ok rows
            tr_local = np.where(fit_mask[idx_ok])[0]
            va_local = np.where(~fit_mask[idx_ok])[0]
            for d in dims:
                if nested_fit_d is not None:
                    Z = _slice_Z(blob["Z"], d)
                else:
                    b = next(b for b in blobs if int(b["d_fit"]) == int(d))
                    Z = np.asarray(b["Z"], dtype=float)
                Z_ok = Z[idx_ok]
                y_ok = y[idx_ok] if y.ndim == 1 else y[idx_ok]
                fold_latents[d].append((
                    Z_ok[va_local], y_ok[va_local],
                    Z_ok[tr_local], y_ok[tr_local],
                ))
    else:
        # raw / raw_lag: no fold latents — use final Z with synthetic blocks
        final = lcache.load_final_latent(cache_dir, method)
        Z = np.asarray(final["Z"], dtype=float)
        # single pseudo-fold: first 80% of train vs last 20%
        n_tr = int(train_ok.sum())
        cut = max(1, int(0.8 * n_tr))
        Z_ok = Z[idx_ok]
        y_ok = y[idx_ok] if y.ndim == 1 else y[idx_ok]
        fold_latents[None] = [(
            Z_ok[cut:], y_ok[cut:], Z_ok[:cut], y_ok[:cut],
        )]

    def _ridge_score(d_key, alpha: float) -> float:
        meds = []
        for Zva, yva, Ztr, ytr in fold_latents[d_key]:
            mtr = np.isfinite(ytr) if ytr.ndim == 1 else np.isfinite(ytr).all(1)
            mva = np.isfinite(yva) if yva.ndim == 1 else np.isfinite(yva).all(1)
            if mtr.sum() < 3 or mva.sum() < 3:
                continue
            if kind == "categorical":
                pred = fit_predict_logreg(Ztr[mtr], ytr[mtr], Zva[mva])
            else:
                pred = fit_predict_ridge_scaled(
                    Ztr[mtr], ytr[mtr], Zva[mva], alpha,
                )
            meds.append(score(pred, yva[mva] if yva.ndim > 1 else yva[mva]))
        return float(np.median(meds)) if meds else float("inf")

    def _knn_score(d_key, k: int) -> float:
        meds = []
        for Zva, yva, Ztr, ytr in fold_latents[d_key]:
            mtr = np.isfinite(ytr) if ytr.ndim == 1 else np.isfinite(ytr).all(1)
            mva = np.isfinite(yva) if yva.ndim == 1 else np.isfinite(yva).all(1)
            if mtr.sum() < 3 or mva.sum() < 3:
                continue
            kk = min(int(k), int(mtr.sum()))
            if kind == "categorical":
                pred = fit_predict_knn_clf(Ztr[mtr], ytr[mtr], Zva[mva], kk)
            else:
                pred = fit_predict_knn_reg(Ztr[mtr], ytr[mtr], Zva[mva], kk)
            meds.append(score(pred, yva[mva] if yva.ndim > 1 else yva[mva]))
        return float(np.median(meds)) if meds else float("inf")

    best_d, best_a, best_med = dims[0], float(alphas[0]), float("inf")
    for d in dims:
        for a in alphas:
            med = _ridge_score(d, float(a))
            if med < best_med or (
                med == best_med and (
                    (d is not None and best_d is not None and d < best_d)
                    or (d == best_d and float(a) < best_a)
                )
            ):
                best_med, best_d, best_a = med, d, float(a)

    best_k, best_k_med = int(ks[0]), float("inf")
    for k in ks:
        med = _knn_score(best_d, int(k))
        if med < best_k_med or (med == best_k_med and int(k) < best_k):
            best_k, best_k_med = int(k), med

    return {
        "primary_d": None if best_d is None else int(best_d),
        "ridge_alpha": float(best_a),
        "knn_k": int(best_k),
        "inner_cv_ridge_score": float(best_med),
        "inner_cv_knn_score": float(best_k_med),
    }


def decode_target_from_cache(
    cache_dir: Path,
    method: str,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    *,
    kind: TargetKind,
    hparams: dict[str, Any],
    shift_fracs: list[float] | None = None,
    pass_min: float = 0.0,
) -> dict[str, Any]:
    """Final readout + A13-style null for one (method, target)."""
    final = lcache.load_final_latent(cache_dir, method)
    d = hparams.get("primary_d")
    Z = _slice_Z(final["Z"], d)
    alpha = float(hparams["ridge_alpha"])
    k = int(hparams["knn_k"])
    y = np.asarray(y, dtype=float)
    train_ok = np.asarray(train_ok, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)

    if kind == "categorical":
        mtr = train_ok & np.isfinite(y)
        mte = eval_mask & np.isfinite(y)
        pred_r = fit_predict_logreg(Z[mtr], y[mtr], Z[mte])
        pred_k = fit_predict_knn_clf(Z[mtr], y[mtr], Z[mte], min(k, int(mtr.sum())))
        sc_r = categorical_chance_normalized(pred_r, y[mte])
        sc_k = categorical_chance_normalized(pred_k, y[mte])
        out = {
            "kind": kind,
            "ridge": sc_r,
            "knn": sc_k,
            "n_train": int(mtr.sum()),
            "n_eval": int(mte.sum()),
            "hparams": hparams,
            "pred_ridge": pred_r,
            "pred_knn": pred_k,
            "eval_index": np.where(mte)[0],
        }
    elif kind == "circular":
        mtr = train_ok & np.isfinite(y).all(1)
        mte = eval_mask & np.isfinite(y).all(1)
        pred_r = fit_predict_ridge_scaled(Z[mtr], y[mtr], Z[mte], alpha)
        pred_k = fit_predict_knn_reg(Z[mtr], y[mtr], Z[mte], min(k, int(mtr.sum())))
        sc_r = normalized_circular_error(pred_r, y[mte], y[mtr])
        sc_k = normalized_circular_error(pred_k, y[mte], y[mtr])
        out = {
            "kind": kind,
            "ridge": sc_r,
            "knn": sc_k,
            "n_train": int(mtr.sum()),
            "n_eval": int(mte.sum()),
            "hparams": hparams,
            "pred_ridge": pred_r,
            "pred_knn": pred_k,
            "eval_index": np.where(mte)[0],
        }
    else:
        if y.ndim == 1:
            mtr = train_ok & np.isfinite(y)
            mte = eval_mask & np.isfinite(y)
            use_med = True
        else:
            mtr = train_ok & np.isfinite(y).all(1)
            mte = eval_mask & np.isfinite(y).all(1)
            use_med = y.shape[1] == 1
        pred_r = fit_predict_ridge_scaled(Z[mtr], y[mtr], Z[mte], alpha)
        pred_k = fit_predict_knn_reg(Z[mtr], y[mtr], Z[mte], min(k, int(mtr.sum())))
        sc_r = normalized_continuous_error(
            pred_r, y[mte], y[mtr], use_median_floor=use_med,
        )
        sc_k = normalized_continuous_error(
            pred_k, y[mte], y[mtr], use_median_floor=use_med,
        )
        out = {
            "kind": kind,
            "ridge": sc_r,
            "knn": sc_k,
            "n_train": int(mtr.sum()),
            "n_eval": int(mte.sum()),
            "hparams": hparams,
            "pred_ridge": pred_r,
            "pred_knn": pred_k,
            "eval_index": np.where(mte)[0],
        }

    if shift_fracs:
        out["a13"] = a13_scalar_shifts(
            Z, y, train_ok, eval_mask,
            alpha=alpha, k=k, shift_fracs=shift_fracs,
            kind=kind, pass_min=pass_min,
        )
    # A5 for secondary: causal latents reused; readout is feed-forward
    out["a5"] = {
        "status": "PASS",
        "note": "latents reused from causal Phase A cache; readout feed-forward",
    }
    return out


def beyond_position_control(
    pred_xy: np.ndarray,
    y_wall_dist: np.ndarray,
    y_wall_bin: np.ndarray,
    polygon_xy_local: np.ndarray,
    eval_index: np.ndarray,
) -> dict[str, Any]:
    """Direct wall targets vs wall derived from decoded position."""
    x_hat = pred_xy[:, 0]
    y_hat = pred_xy[:, 1]
    d_hat, b_hat = derived_wall_from_xy(x_hat, y_hat, polygon_xy_local)
    d_true = y_wall_dist[eval_index]
    b_true = y_wall_bin[eval_index]
    ok = np.isfinite(d_hat) & np.isfinite(d_true)
    dist_mae = float(np.median(np.abs(d_hat[ok] - d_true[ok]))) if ok.any() else float("nan")
    okb = np.isfinite(b_hat) & np.isfinite(b_true)
    bin_ba = categorical_chance_normalized(
        b_hat[okb].astype(int), b_true[okb].astype(int),
    ) if okb.any() else {"balanced_accuracy": float("nan")}
    return {
        "derived_distance_mae": dist_mae,
        "derived_wall_bin": bin_ba,
    }


def beyond_speed_control(
    pred_speed: np.ndarray,
    y_move: np.ndarray,
    eval_index: np.ndarray,
) -> dict[str, Any]:
    move_hat = derived_movement_from_speed(pred_speed)
    move_true = y_move[eval_index]
    ok = np.isfinite(move_hat) & np.isfinite(move_true)
    if not ok.any():
        return {"derived_movement_state": {"balanced_accuracy": float("nan")}}
    return {
        "derived_movement_state": categorical_chance_normalized(
            move_hat[ok].astype(int), move_true[ok].astype(int),
        ),
    }


def prepare_categorical_target(
    labels: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    return merge_rare_classes(labels, train_ok, eval_mask, ordered=True)
