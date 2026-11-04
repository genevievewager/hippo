"""M2: Mahalanobis → logistic P(in) with hysteresis (PILOT_PLAN.md)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from agents.report6.visits import count_visits

ENTER_P = 0.70
EXIT_P = 0.40
MIN_DWELL_S = 0.250
DT_S = 0.050
DELTA_P_PASS = 0.10
BRIER_PASS = 0.25


@dataclass
class DetectorModel:
    mu: np.ndarray
    precision: np.ndarray  # Σ^{-1}
    logit_a: float
    logit_b: float
    enter_p: float = ENTER_P
    exit_p: float = EXIT_P
    min_dwell_samples: int = int(round(MIN_DWELL_S / DT_S))

    def mahalanobis2(self, Z: np.ndarray) -> np.ndarray:
        X = np.asarray(Z, dtype=float)
        d = X - self.mu
        # (x-μ) Σ^{-1} (x-μ)^T
        return np.einsum("ij,jk,ik->i", d, self.precision, d)

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        d2 = self.mahalanobis2(Z)
        logits = self.logit_a - self.logit_b * d2
        # stable sigmoid
        p = np.empty_like(logits)
        pos = logits >= 0
        p[pos] = 1.0 / (1.0 + np.exp(-logits[pos]))
        exp_x = np.exp(logits[~pos])
        p[~pos] = exp_x / (1.0 + exp_x)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

    def hysteresis(self, p: np.ndarray) -> np.ndarray:
        """Enter/exit thresholds with minimum dwell."""
        p = np.asarray(p, dtype=float).ravel()
        out = np.zeros(p.size, dtype=bool)
        state = False
        dwell = 0
        pending_exit = 0
        for i, pi in enumerate(p):
            if not state:
                if pi >= self.enter_p:
                    state = True
                    dwell = 1
                    out[i] = True
                else:
                    out[i] = False
            else:
                dwell += 1
                if pi <= self.exit_p and dwell >= self.min_dwell_samples:
                    # exit
                    state = False
                    dwell = 0
                    out[i] = False
                else:
                    out[i] = True
                    if pi <= self.exit_p:
                        pending_exit += 1
                    else:
                        pending_exit = 0
        return out


def _fit_cov_precision(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    mu = X.mean(axis=0)
    if X.shape[0] >= 3 * d + 5:
        try:
            lw = LedoitWolf().fit(X)
            cov = np.asarray(lw.covariance_, dtype=float)
            prec = np.asarray(lw.precision_, dtype=float)
            return mu, prec
        except Exception:
            pass
    var = np.maximum(X.var(axis=0), 1e-3)
    cov = np.diag(var)
    prec = np.diag(1.0 / var)
    return mu, prec


def fit_detector(
    Z: np.ndarray,
    in_train: np.ndarray,
    train_mask: np.ndarray,
) -> DetectorModel:
    """Fit Mahalanobis + logistic on training membership."""
    Z = np.asarray(Z, dtype=float)
    train = np.asarray(train_mask, dtype=bool)
    y = np.asarray(in_train, dtype=bool)
    X_pos = Z[train & y]
    if X_pos.shape[0] < 5:
        X_pos = Z[train]
    mu, prec = _fit_cov_precision(X_pos)
    # Floor eigenvalues so a near-point mass target still has usable distances
    # (plant-and-recover / tiny clusters).
    try:
        cov = np.linalg.inv(prec)
    except np.linalg.LinAlgError:
        cov = np.diag(np.maximum(np.var(X_pos, axis=0), 1e-3))
    cov = cov + 1e-2 * np.eye(cov.shape[0])
    prec = np.linalg.pinv(cov)
    model = DetectorModel(mu=mu, precision=prec, logit_a=0.0, logit_b=1.0)
    d2 = model.mahalanobis2(Z[train])
    y_tr = y[train].astype(int)
    # Manual threshold calibration when classes are linearly separable / degenerate
    d_in = d2[y_tr == 1]
    d_out = d2[y_tr == 0]
    if d_in.size and d_out.size:
        thr = 0.5 * (float(np.quantile(d_in, 0.9)) + float(np.quantile(d_out, 0.1)))
        thr = max(thr, float(np.quantile(d_in, 0.99)) + 1e-3)
        # P = σ(a - b D²) with steep b so P≈1 for D²<<thr, P≈0 for D²>>thr
        b = 10.0 / max(thr, 1e-3)
        a = b * thr
        model.logit_a = float(a)
        model.logit_b = float(b)
    else:
        clf = LogisticRegression(max_iter=500, solver="lbfgs")
        clf.fit((-d2).reshape(-1, 1), y_tr)
        model.logit_a = float(clf.intercept_[0])
        model.logit_b = float(clf.coef_[0, 0])
    return model


def evaluate_detector(
    model: DetectorModel,
    Z: np.ndarray,
    labels: np.ndarray,
    target_label: int,
    eval_mask: np.ndarray,
    *,
    times_s: np.ndarray,
    dt_s: float = DT_S,
) -> dict[str, Any]:
    """Held-out detection vs neighbours + calibration + hysteresis binary."""
    Z = np.asarray(Z, dtype=float)
    ev = np.asarray(eval_mask, dtype=bool)
    labs = np.asarray(labels)
    p = model.predict_proba(Z)
    y_true = labs == int(target_label)
    # neighbour means on eval
    neigh = [int(l) for l in np.unique(labs) if int(l) != int(target_label)]
    p_tgt = float(p[ev & y_true].mean()) if np.any(ev & y_true) else float("nan")
    neigh_ps = {}
    best_neigh_p = float("nan")
    for nlab in neigh:
        m = ev & (labs == nlab)
        if m.any():
            neigh_ps[str(nlab)] = float(p[m].mean())
    if neigh_ps:
        best_neigh_p = max(neigh_ps.values())
    delta_p = p_tgt - best_neigh_p if np.isfinite(p_tgt) and np.isfinite(best_neigh_p) else float("nan")

    if ev.sum() >= 10 and y_true[ev].any() and (~y_true[ev]).any():
        brier = float(brier_score_loss(y_true[ev].astype(int), p[ev]))
    else:
        brier = float("nan")

    binary = model.hysteresis(p)
    visit_stats = count_visits(binary[ev], times_s=times_s[ev], dt_s=dt_s)

    # reliability 5 bins on eval
    bins = np.linspace(0, 1, 6)
    reliability = []
    for i in range(5):
        sel = ev & (p >= bins[i]) & (p < bins[i + 1] if i < 4 else p <= bins[i + 1])
        if sel.sum() == 0:
            reliability.append({"bin": i, "n": 0, "p_mean": None, "y_mean": None})
        else:
            reliability.append({
                "bin": i,
                "n": int(sel.sum()),
                "p_mean": float(p[sel].mean()),
                "y_mean": float(y_true[sel].mean()),
            })

    distinguishable = (
        np.isfinite(delta_p)
        and delta_p >= DELTA_P_PASS
        and np.isfinite(brier)
        and brier <= BRIER_PASS
    )
    return {
        "p_target_mean": p_tgt,
        "p_neighbour_means": neigh_ps,
        "best_neighbour_p": best_neigh_p,
        "delta_p": float(delta_p) if np.isfinite(delta_p) else None,
        "brier": brier if np.isfinite(brier) else None,
        "distinguishable_pass": bool(distinguishable),
        "reliability": reliability,
        "eval_visits": visit_stats,
        "p": p,
        "binary": binary,
    }


def cohen_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool).ravel()
    b = np.asarray(b, dtype=bool).ravel()
    if a.size != b.size or a.size == 0:
        return float("nan")
    po = float((a == b).mean())
    pa = float(a.mean())
    pb = float(b.mean())
    pe = pa * pb + (1 - pa) * (1 - pb)
    if abs(1 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return float((po - pe) / (1 - pe))
