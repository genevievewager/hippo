"""Scoring for Report 2 v2 secondary targets (continuous / circular / class)."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


def median_abs_error(y_hat: np.ndarray, y: np.ndarray) -> float:
    y_hat = np.asarray(y_hat, dtype=float)
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        err = np.abs(y_hat.ravel() - y.ravel())
    else:
        err = np.linalg.norm(y_hat - y, axis=1)
    err = err[np.isfinite(err)]
    if len(err) == 0:
        return float("nan")
    return float(np.median(err))


def continuous_floor_predictor(
    y_train: np.ndarray, *, use_median: bool = True,
) -> np.ndarray:
    """Scalar targets: train median; multi-d (xy): train mean (v1)."""
    y_train = np.asarray(y_train, dtype=float)
    if y_train.ndim == 1 or y_train.shape[1] == 1:
        yt = y_train.ravel()
        yt = yt[np.isfinite(yt)]
        val = float(np.median(yt) if use_median else np.mean(yt))
        return np.asarray([[val]], dtype=float)
    return np.mean(y_train, axis=0, keepdims=True)


def normalized_continuous_error(
    y_hat: np.ndarray,
    y: np.ndarray,
    y_train: np.ndarray,
    *,
    use_median_floor: bool = True,
) -> dict[str, float]:
    floor_vec = continuous_floor_predictor(y_train, use_median=use_median_floor)
    floor_hat = np.repeat(floor_vec, len(np.asarray(y)), axis=0)
    if y.ndim == 1:
        floor_hat = floor_hat.ravel()
    obs = median_abs_error(y_hat, y)
    floor = median_abs_error(floor_hat, y)
    return {
        "median_error": obs,
        "floor_median_error": floor,
        "normalized_error": (
            float(obs / floor) if floor > 0 and np.isfinite(floor) else float("nan")
        ),
    }


def circular_abs_error_rad(theta_hat: np.ndarray, theta: np.ndarray) -> np.ndarray:
    d = (np.asarray(theta_hat, dtype=float) - np.asarray(theta, dtype=float) + np.pi) % (
        2 * np.pi
    ) - np.pi
    return np.abs(d)


def circular_mean_angle(theta: np.ndarray) -> float:
    theta = np.asarray(theta, dtype=float)
    theta = theta[np.isfinite(theta)]
    if len(theta) == 0:
        return float("nan")
    return float(np.arctan2(np.mean(np.sin(theta)), np.mean(np.cos(theta))))


def normalized_circular_error(
    sincos_hat: np.ndarray,
    sincos: np.ndarray,
    sincos_train: np.ndarray,
) -> dict[str, float]:
    """Median |circular error| / floor from train circular mean."""
    th = np.arctan2(sincos[:, 0], sincos[:, 1])
    th_hat = np.arctan2(sincos_hat[:, 0], sincos_hat[:, 1])
    th_tr = np.arctan2(sincos_train[:, 0], sincos_train[:, 1])
    floor_ang = circular_mean_angle(th_tr)
    err = circular_abs_error_rad(th_hat, th)
    floor_err = circular_abs_error_rad(
        np.full_like(th, floor_ang), th,
    )
    obs = float(np.median(err[np.isfinite(err)]))
    floor = float(np.median(floor_err[np.isfinite(floor_err)]))
    return {
        "median_error_rad": obs,
        "floor_median_error_rad": floor,
        "normalized_error": (
            float(obs / floor) if floor > 0 and np.isfinite(floor) else float("nan")
        ),
    }


def categorical_chance_normalized(
    y_hat: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Score = (1 − balanced_accuracy) / (1 − 1/k); 1 ≈ chance."""
    y = np.asarray(y).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    k = len(np.unique(y))
    if k < 2:
        return {
            "balanced_accuracy": float("nan"),
            "normalized_error": float("nan"),
            "n_classes": int(k),
        }
    ba = float(balanced_accuracy_score(y, y_hat))
    denom = 1.0 - 1.0 / float(k)
    score = (1.0 - ba) / denom if denom > 0 else float("nan")
    return {
        "balanced_accuracy": ba,
        "normalized_error": float(score),
        "n_classes": int(k),
    }


def fit_predict_ridge_scaled(
    Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, alpha: float,
) -> np.ndarray:
    """StandardScaler + Ridge (matches Phase 3 continuous readout)."""
    from realtime.quadrant_n5_run import _fit_predict_ridge

    ytr = np.asarray(ytr, dtype=float)
    if ytr.ndim == 1:
        ytr = ytr.reshape(-1, 1)
        pred = _fit_predict_ridge(Ztr, ytr, Zte, float(alpha))
        return pred.ravel()
    return _fit_predict_ridge(Ztr, ytr, Zte, float(alpha))


def fit_predict_knn_reg(
    Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, k: int,
) -> np.ndarray:
    ytr = np.asarray(ytr, dtype=float)
    multi = ytr.ndim > 1 and ytr.shape[1] > 1
    if not multi:
        ytr = ytr.ravel()
    model = KNeighborsRegressor(n_neighbors=int(k), weights="uniform")
    model.fit(Ztr, ytr)
    pred = np.asarray(model.predict(Zte), dtype=float)
    return pred if multi else pred.ravel()


def fit_predict_logreg(
    Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray,
) -> np.ndarray:
    scaler = StandardScaler()
    Ztr_s = scaler.fit_transform(Ztr)
    Zte_s = scaler.transform(Zte)
    clf = LogisticRegression(
        max_iter=500,
        multi_class="multinomial",
        solver="lbfgs",
        random_state=0,
    )
    clf.fit(Ztr_s, np.asarray(ytr).astype(int))
    return np.asarray(clf.predict(Zte_s), dtype=int)


def fit_predict_knn_clf(
    Ztr: np.ndarray, ytr: np.ndarray, Zte: np.ndarray, k: int,
) -> np.ndarray:
    clf = KNeighborsClassifier(n_neighbors=int(k), weights="uniform")
    clf.fit(Ztr, np.asarray(ytr).astype(int))
    return np.asarray(clf.predict(Zte), dtype=int)


def a13_scalar_shifts(
    Z: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    *,
    alpha: float,
    k: int,
    shift_fracs: list[float],
    kind: str = "continuous",
    pass_min: float = 0.0,
) -> dict[str, Any]:
    """Circular time-shift null on eval labels (secondary A13 analogue).

    For continuous/circular, null must not beat the train-constant floor
    (median null_error − floor ≥ pass_min). For categorical, null
    normalized_error must be ≥ ~1 (chance) on median over shifts.
    """
    y = np.asarray(y, dtype=float)
    idx_tr = np.where(train_ok)[0]
    idx_te = np.where(eval_mask)[0]
    n = len(y) if y.ndim == 1 else len(y)
    rows = []
    for frac in shift_fracs:
        shift = int(round(float(frac) * n))
        y_shift = np.roll(y, shift, axis=0)
        Ztr, ytr = Z[idx_tr], y_shift[idx_tr]
        Zte, yte = Z[idx_te], y_shift[idx_te]
        if kind == "categorical":
            # drop nan labels
            mtr = np.isfinite(ytr) if ytr.ndim == 1 else np.isfinite(ytr).all(1)
            mte = np.isfinite(yte) if yte.ndim == 1 else np.isfinite(yte).all(1)
            if mtr.sum() < 5 or mte.sum() < 5:
                continue
            pred_r = fit_predict_logreg(Ztr[mtr], ytr[mtr], Zte[mte])
            pred_k = fit_predict_knn_clf(Ztr[mtr], ytr[mtr], Zte[mte], k)
            sc_r = categorical_chance_normalized(pred_r, yte[mte])
            sc_k = categorical_chance_normalized(pred_k, yte[mte])
            rows.append({
                "frac": float(frac),
                "ridge_norm": sc_r["normalized_error"],
                "knn_norm": sc_k["normalized_error"],
            })
        else:
            mtr = np.isfinite(ytr) if ytr.ndim == 1 else np.isfinite(ytr).all(1)
            mte = np.isfinite(yte) if yte.ndim == 1 else np.isfinite(yte).all(1)
            if mtr.sum() < 5 or mte.sum() < 5:
                continue
            pred_r = fit_predict_ridge_scaled(Ztr[mtr], ytr[mtr], Zte[mte], alpha)
            pred_k = fit_predict_knn_reg(Ztr[mtr], ytr[mtr], Zte[mte], k)
            if kind == "circular":
                sc_r = normalized_circular_error(pred_r, yte[mte], ytr[mtr])
                sc_k = normalized_circular_error(pred_k, yte[mte], ytr[mtr])
                floor = sc_r["floor_median_error_rad"]
                rows.append({
                    "frac": float(frac),
                    "ridge_minus_floor": sc_r["median_error_rad"] - floor,
                    "knn_minus_floor": sc_k["median_error_rad"] - floor,
                })
            else:
                use_med = ytr.ndim == 1 or ytr.shape[1] == 1
                sc_r = normalized_continuous_error(
                    pred_r, yte[mte], ytr[mtr], use_median_floor=use_med,
                )
                sc_k = normalized_continuous_error(
                    pred_k, yte[mte], ytr[mtr], use_median_floor=use_med,
                )
                rows.append({
                    "frac": float(frac),
                    "ridge_minus_floor": (
                        sc_r["median_error"] - sc_r["floor_median_error"]
                    ),
                    "knn_minus_floor": (
                        sc_k["median_error"] - sc_k["floor_median_error"]
                    ),
                })

    if not rows:
        return {"status": "FAIL", "reason": "no valid shifts", "shifts": []}

    if kind == "categorical":
        r_med = float(np.median([r["ridge_norm"] for r in rows]))
        k_med = float(np.median([r["knn_norm"] for r in rows]))
        # null at/above chance → normalized_error ≥ 1 − tol
        ridge_pass = r_med >= 1.0 - 1e-6
        knn_pass = k_med >= 1.0 - 1e-6
        return {
            "status": "PASS" if (ridge_pass and knn_pass) else "FAIL",
            "ridge_null_norm": r_med,
            "knn_null_norm": k_med,
            "ridge_pass": ridge_pass,
            "knn_pass": knn_pass,
            "shifts": rows,
        }

    r_med = float(np.median([r["ridge_minus_floor"] for r in rows]))
    k_med = float(np.median([r["knn_minus_floor"] for r in rows]))
    ridge_pass = r_med >= float(pass_min)
    knn_pass = k_med >= float(pass_min)
    return {
        "status": "PASS" if (ridge_pass and knn_pass) else "FAIL",
        "ridge_median_minus_floor": r_med,
        "knn_median_minus_floor": k_med,
        "ridge_pass": ridge_pass,
        "knn_pass": knn_pass,
        "pass_min": float(pass_min),
        "shifts": rows,
    }
