"""Decode secondary targets from cached fold / final latents (no refit)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from analysis.real_quadrant import latent_cache as lcache
from analysis.real_quadrant.posthoc_smooth import EMA_TAUS_S, causal_ema
from analysis.real_quadrant.secondary_metrics import (
    a13_scalar_shifts,
    categorical_chance_normalized,
    fit_predict_knn_clf,
    fit_predict_knn_clf_proba,
    fit_predict_knn_reg,
    fit_predict_logreg,
    fit_predict_logreg_proba,
    fit_predict_ridge_scaled,
    normalized_circular_error,
    normalized_continuous_error,
)
from analysis.real_quadrant.secondary_targets import (
    derived_movement_from_speed,
    derived_wall_from_xy,
    merge_rare_classes,
)
from realtime.quadrant_n5 import inner_cv_block_masks


TargetKind = str  # continuous | circular | categorical


def _slice_Z(Z: np.ndarray, d: int | None) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    if d is None:
        return Z
    return Z[:, : int(d)]


def _predict_all_ridge(
    Z: np.ndarray, y: np.ndarray, train_mask: np.ndarray, alpha: float,
) -> np.ndarray:
    """Fit on ``train_mask``; predict every row (continuous / circular)."""
    m = np.asarray(train_mask, dtype=bool)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        m = m & np.isfinite(y)
        pred = fit_predict_ridge_scaled(Z[m], y[m], Z, float(alpha))
        return np.asarray(pred, dtype=float).ravel()
    m = m & np.isfinite(y).all(1)
    return np.asarray(
        fit_predict_ridge_scaled(Z[m], y[m], Z, float(alpha)), dtype=float,
    )


def _predict_all_knn_reg(
    Z: np.ndarray, y: np.ndarray, train_mask: np.ndarray, k: int,
) -> np.ndarray:
    m = np.asarray(train_mask, dtype=bool)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        m = m & np.isfinite(y)
        kk = min(int(k), int(m.sum()))
        return np.asarray(
            fit_predict_knn_reg(Z[m], y[m], Z, kk), dtype=float,
        ).ravel()
    m = m & np.isfinite(y).all(1)
    kk = min(int(k), int(m.sum()))
    return np.asarray(fit_predict_knn_reg(Z[m], y[m], Z, kk), dtype=float)


def select_ema_tau_outputs(
    pred_full: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    train_ok: np.ndarray,
    *,
    kind: TargetKind,
    taus_s: Sequence[float] = EMA_TAUS_S,
    n_blocks: int = 5,
    gap_s: float = 1.0,
    classes: np.ndarray | None = None,
) -> tuple[float, float]:
    """Choose causal EMA τ on training blocks only (never on test).

    ``pred_full`` is the unsmoothed readout on the full timeline.
    For categorical, ``pred_full`` is class probabilities ``(n, k)``.
    """
    times = np.asarray(times, dtype=float)
    train_ok = np.asarray(train_ok, dtype=bool)
    y = np.asarray(y, dtype=float)
    if kind == "categorical":
        train_ok = train_ok & np.isfinite(y)
    elif y.ndim == 1:
        train_ok = train_ok & np.isfinite(y)
    else:
        train_ok = train_ok & np.isfinite(y).all(1)
    train_idx = np.where(train_ok)[0]
    if train_idx.size < 10:
        return float(taus_s[0]), float("inf")
    times_tr = times[train_ok]
    folds = inner_cv_block_masks(times_tr, n_blocks=int(n_blocks), gap_s=float(gap_s))
    best_tau = float(taus_s[0])
    best_med = float("inf")
    for tau in taus_s:
        fold_meds: list[float] = []
        for tr_rel, va_rel in folds:
            va_idx = train_idx[np.asarray(va_rel, dtype=bool)]
            if va_idx.size == 0:
                continue
            # EMA over full timeline (causal); score only validation indices.
            if kind == "categorical":
                sm = causal_ema(pred_full, times, float(tau))
                row = sm[va_idx]
                row = np.clip(row, 0, None)
                row = row / np.maximum(row.sum(axis=1, keepdims=True), 1e-12)
                yhat = np.asarray(classes)[np.argmax(row, axis=1)]
                yva = y[va_idx].astype(int)
                ba = categorical_chance_normalized(yhat, yva)["balanced_accuracy"]
                fold_meds.append(
                    1.0 - float(ba) if np.isfinite(ba) else 1.0
                )
            else:
                pred2 = pred_full if pred_full.ndim == 2 else pred_full.reshape(-1, 1)
                sm = causal_ema(pred2, times, float(tau))
                if kind == "circular":
                    th = np.arctan2(y[va_idx, 0], y[va_idx, 1])
                    thh = np.arctan2(sm[va_idx, 0], sm[va_idx, 1])
                    d = (thh - th + np.pi) % (2 * np.pi) - np.pi
                    fold_meds.append(float(np.median(np.abs(d))))
                elif y.ndim == 1:
                    fold_meds.append(
                        float(np.median(np.abs(sm[va_idx, 0] - y[va_idx])))
                    )
                else:
                    fold_meds.append(
                        float(np.median(np.linalg.norm(sm[va_idx] - y[va_idx], axis=1)))
                    )
        if not fold_meds:
            continue
        med = float(np.median(fold_meds))
        if med < best_med or (med == best_med and float(tau) < best_tau):
            best_med = med
            best_tau = float(tau)
    return best_tau, best_med


def apply_ema_selected(
    pred_full: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    *,
    kind: TargetKind,
    taus_s: Sequence[float] = EMA_TAUS_S,
    classes: np.ndarray | None = None,
) -> dict[str, Any]:
    """Select τ on train CV; apply causal EMA; score eval (τ never from test)."""
    tau, cv = select_ema_tau_outputs(
        pred_full, y, times, train_ok, kind=kind, taus_s=taus_s, classes=classes,
    )
    if kind == "categorical":
        sm = causal_ema(pred_full, times, tau)
        row = np.clip(sm, 0, None)
        row = row / np.maximum(row.sum(axis=1, keepdims=True), 1e-12)
        yhat_full = np.asarray(classes)[np.argmax(row, axis=1)].astype(int)
        mte = np.asarray(eval_mask, dtype=bool) & np.isfinite(y)
        sc = categorical_chance_normalized(yhat_full[mte], y[mte].astype(int))
        return {
            "tau_s": float(tau),
            "inner_cv_score": float(cv),
            "scores": sc,
            "pred_eval": yhat_full[mte],
            "pred_full": yhat_full,
        }
    pred2 = pred_full if pred_full.ndim == 2 else pred_full.reshape(-1, 1)
    sm = causal_ema(pred2, times, tau)
    if kind == "circular":
        mte = np.asarray(eval_mask, dtype=bool) & np.isfinite(y).all(1)
        mtr = np.asarray(train_ok, dtype=bool) & np.isfinite(y).all(1)
        sc = normalized_circular_error(sm[mte], y[mte], y[mtr])
        return {
            "tau_s": float(tau),
            "inner_cv_score": float(cv),
            "scores": sc,
            "pred_eval": sm[mte],
            "pred_full": sm,
        }
    if y.ndim == 1:
        mte = np.asarray(eval_mask, dtype=bool) & np.isfinite(y)
        mtr = np.asarray(train_ok, dtype=bool) & np.isfinite(y)
        sc = normalized_continuous_error(
            sm[mte, 0], y[mte], y[mtr], use_median_floor=True,
        )
        return {
            "tau_s": float(tau),
            "inner_cv_score": float(cv),
            "scores": sc,
            "pred_eval": sm[mte, 0],
            "pred_full": sm[:, 0],
        }
    mte = np.asarray(eval_mask, dtype=bool) & np.isfinite(y).all(1)
    mtr = np.asarray(train_ok, dtype=bool) & np.isfinite(y).all(1)
    sc = normalized_continuous_error(
        sm[mte], y[mte], y[mtr], use_median_floor=False,
    )
    return {
        "tau_s": float(tau),
        "inner_cv_score": float(cv),
        "scores": sc,
        "pred_eval": sm[mte],
        "pred_full": sm,
    }

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
    times: np.ndarray | None = None,
    shift_fracs: list[float] | None = None,
    pass_min: float = 0.0,
    apply_ema: bool = True,
    ema_taus_s: Sequence[float] = EMA_TAUS_S,
) -> dict[str, Any]:
    """Final readout + optional causal EMA + A13-style null for one target.

    EMA τ is chosen by blocked inner CV on training indices only (never test).
    Applied to continuous outputs, HD sin/cos, or class probabilities.
    """
    final = lcache.load_final_latent(cache_dir, method)
    d = hparams.get("primary_d")
    Z = _slice_Z(final["Z"], d)
    alpha = float(hparams["ridge_alpha"])
    k = int(hparams["knn_k"])
    y = np.asarray(y, dtype=float)
    train_ok = np.asarray(train_ok, dtype=bool)
    eval_mask = np.asarray(eval_mask, dtype=bool)
    if times is None:
        times = np.arange(len(Z), dtype=float) * 0.05
    else:
        times = np.asarray(times, dtype=float)

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
        if apply_ema:
            proba_r, classes_r = fit_predict_logreg_proba(Z[mtr], y[mtr], Z)
            proba_k, classes_k = fit_predict_knn_clf_proba(
                Z[mtr], y[mtr], Z, min(k, int(mtr.sum())),
            )
            out["ridge_ema"] = apply_ema_selected(
                proba_r, y, times, train_ok, eval_mask,
                kind="categorical", taus_s=ema_taus_s, classes=classes_r,
            )
            out["knn_ema"] = apply_ema_selected(
                proba_k, y, times, train_ok, eval_mask,
                kind="categorical", taus_s=ema_taus_s, classes=classes_k,
            )
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
        if apply_ema:
            full_r = _predict_all_ridge(Z, y, train_ok, alpha)
            full_k = _predict_all_knn_reg(Z, y, train_ok, k)
            out["ridge_ema"] = apply_ema_selected(
                full_r, y, times, train_ok, eval_mask,
                kind="circular", taus_s=ema_taus_s,
            )
            out["knn_ema"] = apply_ema_selected(
                full_k, y, times, train_ok, eval_mask,
                kind="circular", taus_s=ema_taus_s,
            )
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
        if apply_ema:
            full_r = _predict_all_ridge(Z, y, train_ok, alpha)
            full_k = _predict_all_knn_reg(Z, y, train_ok, k)
            out["ridge_ema"] = apply_ema_selected(
                full_r, y, times, train_ok, eval_mask,
                kind="continuous", taus_s=ema_taus_s,
            )
            out["knn_ema"] = apply_ema_selected(
                full_k, y, times, train_ok, eval_mask,
                kind="continuous", taus_s=ema_taus_s,
            )

    if shift_fracs:
        out["a13"] = a13_scalar_shifts(
            Z, y, train_ok, eval_mask,
            alpha=alpha, k=k, shift_fracs=shift_fracs,
            kind=kind, pass_min=pass_min,
        )
    out["a5"] = {
        "status": "PASS",
        "note": "latents reused from causal Phase A cache; readout feed-forward; EMA causal",
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
