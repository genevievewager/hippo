"""Post-hoc causal EMA on Ridge predictions (no representation refit).

Used for M2 ``raw_smooth`` / ``pca_smooth`` / ``dm_smooth`` and for parallel
``raw_smooth`` rows on the frozen sim N=5 outputs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from realtime.quadrant_n5 import inner_cv_block_masks, load_quadrant_n5_yaml
from realtime.quadrant_n5_run import _euclid

REPO_ROOT = Path(__file__).resolve().parents[2]
M2_DIR = REPO_ROOT / "outputs" / "real_quadrant" / "m2" / "room_A" / "all_methods"
SIM_ROOT = REPO_ROOT / "outputs" / "quadrant_n5"

EMA_TAUS_S = (0.1, 0.25, 0.5, 1.0, 2.0)
SMOOTH_BASES = ("raw", "pca", "dm", "lds")
SIM_SMOOTH_BASES = ("raw", "pca", "dm", "lds")
GPFA_LABEL = "offline reference (non-causal smoother)"
CAUSAL_METHODS = (
    "raw", "raw_lag", "pca", "dm", "lds", "isomap",
    "raw_smooth", "pca_smooth", "dm_smooth", "lds_smooth", "gpfa_causal",
)
OFFLINE_METHODS = ("gpfa",)
LAG_MAX_S = 3.0
UPDATE_DT_S = 0.050
GPFA_CAUSAL_VERIFIED = True  # set by leakage suite; see tests/test_gpfa_causal_leakage.py



def causal_ema(
    pred: np.ndarray,
    times: np.ndarray,
    tau_s: float,
    *,
    gap_reset_s: float = 0.5,
) -> np.ndarray:
    """Causal exponential moving average: ``s_t = a s_{t-1} + (1-a) p_t``.

    ``a = exp(-dt / tau)``. Large gaps (``dt > gap_reset_s``) reset the state.
    """
    pred = np.asarray(pred, dtype=float)
    times = np.asarray(times, dtype=float)
    if pred.ndim != 2 or times.ndim != 1 or len(pred) != len(times):
        raise ValueError("pred (n,d) and times (n,) required")
    if float(tau_s) <= 0:
        raise ValueError("tau_s must be positive")
    out = np.empty_like(pred)
    if len(pred) == 0:
        return out
    out[0] = pred[0]
    for i in range(1, len(pred)):
        dt = float(times[i] - times[i - 1])
        if not np.isfinite(dt) or dt <= 0 or dt > float(gap_reset_s):
            out[i] = pred[i]
            continue
        a = float(np.exp(-dt / float(tau_s)))
        out[i] = a * out[i - 1] + (1.0 - a) * pred[i]
    return out


def _ridge_predict_all(
    Z: np.ndarray,
    y: np.ndarray,
    fit_mask: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit Ridge on ``fit_mask``; predict every row of ``Z``."""
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    sc = StandardScaler()
    Ztr = sc.fit_transform(Z[fit_mask])
    model = Ridge(alpha=float(alpha))
    model.fit(Ztr, y[fit_mask])
    return model.predict(sc.transform(Z))


def select_ema_tau(
    Z: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    train_ok: np.ndarray,
    ridge_alpha: float,
    *,
    taus_s: Sequence[float] = EMA_TAUS_S,
    n_blocks: int = 5,
    gap_s: float = 1.0,
) -> tuple[float, float]:
    """Choose EMA time constant by blocked inner CV on training indices only.

    Returns ``(tau_s, inner_cv_median_error_cm)``.
    """
    times = np.asarray(times, dtype=float)
    train_ok = np.asarray(train_ok, dtype=bool)
    train_idx = np.where(train_ok)[0]
    times_tr = times[train_ok]
    folds = inner_cv_block_masks(times_tr, n_blocks=int(n_blocks), gap_s=float(gap_s))
    best_tau = float(taus_s[0])
    best_med = float("inf")
    for tau in taus_s:
        fold_meds: list[float] = []
        for tr_rel, va_rel in folds:
            tr_idx = train_idx[np.asarray(tr_rel, dtype=bool)]
            va_idx = train_idx[np.asarray(va_rel, dtype=bool)]
            if tr_idx.size == 0 or va_idx.size == 0:
                continue
            pred = _ridge_predict_all(Z, y, np.isin(np.arange(len(Z)), tr_idx), ridge_alpha)
            sm = causal_ema(pred, times, float(tau))
            finite = np.isfinite(y[va_idx]).all(axis=1)
            if not finite.any():
                continue
            err = np.linalg.norm(sm[va_idx][finite] - y[va_idx][finite], axis=1)
            fold_meds.append(float(np.median(err)))
        if not fold_meds:
            continue
        med = float(np.median(fold_meds))
        if med < best_med or (med == best_med and float(tau) < best_tau):
            best_med = med
            best_tau = float(tau)
    return best_tau, best_med


def apply_smoothed_method(
    Z: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    ridge_alpha: float,
    *,
    taus_s: Sequence[float] = EMA_TAUS_S,
    n_blocks: int = 5,
    gap_s: float = 1.0,
) -> dict[str, Any]:
    """Select tau on train CV, refit Ridge on all train_ok, EMA, score eval."""
    tau, cv_med = select_ema_tau(
        Z, y, times, train_ok, ridge_alpha,
        taus_s=taus_s, n_blocks=n_blocks, gap_s=gap_s,
    )
    pred = _ridge_predict_all(Z, y, train_ok, ridge_alpha)
    sm = causal_ema(pred, times, tau)
    yte = y[eval_mask]
    pred_te = sm[eval_mask]
    finite = np.isfinite(yte).all(axis=1)
    ridge = _euclid(pred_te[finite], yte[finite])
    return {
        "tau_s": float(tau),
        "inner_cv_median_cm": float(cv_med),
        "ridge": ridge,
        "pred_full": sm,
        "pred_eval": pred_te,
    }


def fraction_outside_boundary(
    xy: np.ndarray,
    outline: np.ndarray,
) -> float:
    """Fraction of finite points strictly outside the closed outline polygon."""
    from matplotlib.path import Path as MplPath

    xy = np.asarray(xy, dtype=float)
    outline = np.asarray(outline, dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    if not finite.any():
        return float("nan")
    path = MplPath(outline)
    inside = path.contains_points(xy[finite], radius=1e-9)
    return float(1.0 - np.mean(inside))


def effective_lag_s(
    pred: np.ndarray,
    y_true: np.ndarray,
    *,
    dt_s: float = UPDATE_DT_S,
    max_lag_s: float = LAG_MAX_S,
) -> dict[str, float]:
    """Lag (seconds) maximizing mean axis-wise Pearson corr of pred vs truth.

    Positive lag means the prediction leads the truth (pred[t] best matches
    y[t+lag]); negative means the prediction lags behind.
    Search is in integer steps of ``dt_s`` within ``±max_lag_s``.
    """
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    finite = np.isfinite(pred).all(axis=1) & np.isfinite(y_true).all(axis=1)
    pred, y_true = pred[finite], y_true[finite]
    if len(pred) < 20:
        return {"lag_s": float("nan"), "corr": float("nan"), "lag_steps": float("nan")}
    max_steps = int(round(float(max_lag_s) / float(dt_s)))
    best = (-np.inf, 0)
    for step in range(-max_steps, max_steps + 1):
        if step < 0:
            p, yt = pred[-step:], y_true[: len(y_true) + step]
        elif step > 0:
            p, yt = pred[: len(pred) - step], y_true[step:]
        else:
            p, yt = pred, y_true
        if len(p) < 20:
            continue
        corrs = []
        for d in range(p.shape[1]):
            if np.std(p[:, d]) < 1e-12 or np.std(yt[:, d]) < 1e-12:
                continue
            corrs.append(float(np.corrcoef(p[:, d], yt[:, d])[0, 1]))
        if not corrs:
            continue
        c = float(np.mean(corrs))
        if c > best[0] or (c == best[0] and abs(step) < abs(best[1])):
            best = (c, step)
    return {
        "lag_s": float(best[1] * dt_s),
        "corr": float(best[0]) if np.isfinite(best[0]) else float("nan"),
        "lag_steps": float(best[1]),
    }


def run_m2_posthoc() -> dict[str, Any]:
    """Add smoothed methods + lag costs to M2 report using saved latents/preds."""
    report = json.loads((M2_DIR / "m2_report.json").read_text())
    contract = Path(report["figure_contract_dir"])
    arrays = np.load(contract / "arrays.npz")
    latents = np.load(contract / "latents.npz")
    preds = dict(np.load(contract / "predictions.npz"))
    cfg = load_quadrant_n5_yaml()
    n_blocks = int(cfg["split"]["inner_cv_blocks"])
    gap_s = float(cfg["split"]["gap_s"])
    floor_med = float(report["floor_median_cm"])
    times = np.asarray(arrays["decode_times"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    train_ok = np.asarray(arrays["train_ok"], dtype=bool)
    eval_mask = np.asarray(arrays["eval_mask"], dtype=bool)
    y_true = y[eval_mask]

    method_rows = {m["method"]: dict(m) for m in report["methods"]}
    if "gpfa" in method_rows:
        method_rows["gpfa"]["display_name"] = GPFA_LABEL
        method_rows["gpfa"]["causal"] = False
        method_rows["gpfa"]["role"] = "offline_reference"

    for base in SMOOTH_BASES:
        rec = json.loads((M2_DIR / f"{base}.json").read_text())
        Z = np.asarray(latents[f"Z_{base}"], dtype=float)
        primary_d = rec.get("primary_d")
        if primary_d is not None and Z.shape[1] > int(primary_d):
            Z = Z[:, : int(primary_d)]
        out = apply_smoothed_method(
            Z, y, times, train_ok, eval_mask, float(rec["ridge_alpha"]),
            n_blocks=n_blocks, gap_s=gap_s,
        )
        name = f"{base}_smooth"
        ridge_med = float(out["ridge"]["median"])
        method_rows[name] = {
            "method": name,
            "base_method": base,
            "selected_d": rec.get("primary_d"),
            "ridge_median_cm": ridge_med,
            "normalized_error": (
                ridge_med / floor_med if floor_med > 0 else None
            ),
            "knn_median_cm": None,
            "a13_status": None,
            "causal": True,
            "ema_tau_s": out["tau_s"],
            "inner_cv_median_cm": out["inner_cv_median_cm"],
            "fit_s": None,
            "transform_s": None,
            "elapsed_s": None,
        }
        preds[f"pred_{name}_ridge"] = np.asarray(out["pred_eval"], dtype=float)

    if GPFA_CAUSAL_VERIFIED:
        gpfa_causal_row = _try_gpfa_causal(
            report, arrays, times, y, train_ok, eval_mask, floor_med, n_blocks, gap_s,
        )
        if gpfa_causal_row is not None:
            method_rows["gpfa_causal"] = gpfa_causal_row["row"]
            preds["pred_gpfa_causal_ridge"] = gpfa_causal_row["pred_eval"]

    # Effective lag for every method with eval predictions.
    for name, row in list(method_rows.items()):
        key = f"pred_{name}_ridge"
        if key not in preds:
            continue
        lag = effective_lag_s(np.asarray(preds[key]), y_true)
        row["effective_lag_s"] = lag["lag_s"]
        row["effective_lag_corr"] = lag["corr"]

    np.savez_compressed(contract / "predictions.npz", **{
        k: np.asarray(v) for k, v in preds.items()
    })

    for m, row in method_rows.items():
        if m == "gpfa":
            row["causal"] = False
            row.setdefault("display_name", GPFA_LABEL)
        elif m == "gpfa_causal":
            row["causal"] = True
            row["display_name"] = "GPFA causal filter"
            row["role"] = "linear_dynamic_causal"
            row["quadrant_cell"] = "linear_dynamic"
        else:
            row.setdefault("causal", True)

    ordered = []
    for key in (
        "raw", "raw_smooth", "raw_lag", "pca", "pca_smooth", "dm", "dm_smooth",
        "lds", "lds_smooth", "isomap", "gpfa_causal", "gpfa",
    ):
        if key in method_rows:
            ordered.append(method_rows[key])
    for key, row in method_rows.items():
        if key not in {r["method"] for r in ordered}:
            ordered.append(row)

    report["methods"] = ordered
    report["gpfa_label"] = GPFA_LABEL
    report["gpfa_causal_verified"] = bool(GPFA_CAUSAL_VERIFIED)
    report["posthoc_smooth_taus_s"] = list(EMA_TAUS_S)
    report["effective_lag_search"] = {
        "dt_s": UPDATE_DT_S,
        "max_lag_s": LAG_MAX_S,
        "rule": "argmax mean axis-wise Pearson corr of pred vs truth",
    }
    (M2_DIR / "m2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _try_gpfa_causal(
    report: dict[str, Any],
    arrays: Any,
    times: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    floor_med: float,
    n_blocks: int,
    gap_s: float,
) -> dict[str, Any] | None:
    """If a fitted GPFA model is on disk, score its causal filter latents."""
    try:
        from analysis.real_quadrant.adapter import build_segment_bundle
        from realtime.dynamic_latents.gpfa import GPFAModel
        from realtime.quadrant_n5_run import (
            OUTPUT_ROOT,
            _sqrt_zscore_train,
        )
    except Exception:
        return None

    model_dir = OUTPUT_ROOT / "models" / "seed_0" / "real" / "gpfa"
    if not model_dir.is_dir() or not (model_dir / "gpfa_params.npz").is_file():
        return None
    try:
        model = GPFAModel.load(model_dir)
    except Exception:
        return None

    sel = json.loads((M2_DIR / "session_selection.json").read_text())
    session = sel["selected_session"]
    try:
        bundle = build_segment_bundle(
            session, room="A", require_ratemap_stability=False,
        )
    except Exception:
        return None

    X_counts = np.asarray(bundle["X_counts"], dtype=float)
    if len(X_counts) != len(times):
        return None
    X, _ = _sqrt_zscore_train(X_counts, train_ok)
    Z = model.transform(X, causal=True, reset=True)
    gpfa_rec = json.loads((M2_DIR / "gpfa.json").read_text())
    primary_d = gpfa_rec.get("primary_d")
    if primary_d is not None and Z.shape[1] > int(primary_d):
        Z = Z[:, : int(primary_d)]
    alpha = float(gpfa_rec["ridge_alpha"])
    pred = _ridge_predict_all(Z, y, train_ok, alpha)
    yte = y[eval_mask]
    pred_te = pred[eval_mask]
    finite = np.isfinite(yte).all(axis=1)
    ridge = _euclid(pred_te[finite], yte[finite])
    ridge_med = float(ridge["median"])
    return {
        "row": {
            "method": "gpfa_causal",
            "display_name": "GPFA causal filter",
            "selected_d": primary_d,
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "knn_median_cm": None,
            "a13_status": None,
            "causal": True,
            "role": "linear_dynamic_causal",
            "quadrant_cell": "linear_dynamic",
            "ema_tau_s": None,
            "fit_s": None,
            "transform_s": None,
            "elapsed_s": None,
        },
        "pred_eval": pred_te,
    }


def run_sim_smooth_rows(
    bases: Sequence[str] = SIM_SMOOTH_BASES,
) -> list[dict[str, Any]]:
    """Write ``{base}_smooth_posthoc.json`` under each sim seed/source (new only)."""
    from agents.quadrant_n5.export_predictions import build_observation, latents_for_method

    cfg = load_quadrant_n5_yaml()
    n_blocks = int(cfg["split"]["inner_cv_blocks"])
    gap_s = float(cfg["split"]["gap_s"])
    rows_out: list[dict[str, Any]] = []
    for seed in range(5):
        for src in ("sorted", "ground_truth"):
            src_dir = SIM_ROOT / f"seed_{seed}" / src
            if not (src_dir / "raw.json").is_file():
                continue
            summary = json.loads((src_dir / "source_summary.json").read_text())
            floor_med = float(summary["floor"]["median"])
            streams = summary["seed_streams"]
            obs = build_observation(cfg, SIM_ROOT / f"seed_{seed}" / "sim", src)
            for base in bases:
                if not (src_dir / f"{base}.json").is_file():
                    continue
                rec = json.loads((src_dir / f"{base}.json").read_text())
                model_dir = SIM_ROOT / "models" / f"seed_{seed}" / src / base
                Z, route = latents_for_method(
                    base, cfg, obs, rec, model_dir, int(streams["methods"]),
                )
                out = apply_smoothed_method(
                    Z, obs["y"], obs["decode_times"], obs["train_ok"], obs["eval_mask"],
                    float(rec["ridge_alpha"]),
                    n_blocks=n_blocks, gap_s=gap_s,
                )
                ridge_med = float(out["ridge"]["median"])
                pred_te = out["pred_eval"]
                lag = effective_lag_s(pred_te, obs["y"][obs["eval_mask"]])
                name = f"{base}_smooth"
                payload = {
                    "stage": "posthoc_smooth",
                    "method": name,
                    "base_method": base,
                    "seed_index": seed,
                    "spike_source": src,
                    "config_sha256": cfg["config_sha256"],
                    "causal": True,
                    "ema_tau_s": out["tau_s"],
                    "inner_cv_median_cm": out["inner_cv_median_cm"],
                    "ridge": out["ridge"],
                    "ridge_median_cm": ridge_med,
                    "floor_median_cm": floor_med,
                    "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
                    "effective_lag_s": lag["lag_s"],
                    "effective_lag_corr": lag["corr"],
                    "latent_route": route,
                    "note": "New row only; existing method JSONs untouched.",
                }
                (src_dir / f"{name}_posthoc.json").write_text(
                    json.dumps(payload, indent=2, default=str) + "\n"
                )
                rows_out.append(payload)
    summary_path = SIM_ROOT / "smooth_posthoc_summary.json"
    summary_path.write_text(json.dumps({"rows": rows_out}, indent=2, default=str) + "\n")
    return rows_out


def sim_lds_vs_raw_smooth() -> dict[str, Any]:
    """Paired sim contrast: lds − raw_smooth (normalized error), sign counts."""
    diffs = []
    for seed in range(5):
        for src in ("sorted", "ground_truth"):
            src_dir = SIM_ROOT / f"seed_{seed}" / src
            lds_path = src_dir / "lds.json"
            sm_path = src_dir / "raw_smooth_posthoc.json"
            if not lds_path.is_file() or not sm_path.is_file():
                continue
            summary = json.loads((src_dir / "source_summary.json").read_text())
            floor = float(summary["floor"]["median"])
            lds = json.loads(lds_path.read_text())
            sm = json.loads(sm_path.read_text())
            lds_ne = float(lds["ridge"]["median"]) / floor
            sm_ne = float(sm["normalized_error"])
            diffs.append({
                "seed": seed,
                "source": src,
                "lds_normalized": lds_ne,
                "raw_smooth_normalized": sm_ne,
                "lds_minus_raw_smooth": lds_ne - sm_ne,
            })
    arr = np.asarray([d["lds_minus_raw_smooth"] for d in diffs], dtype=float)
    out = {
        "contrast": "lds - raw_smooth",
        "metric": "normalized_error",
        "n": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else None,
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
        "n_lds_better": int(np.sum(arr < 0)),
        "n_raw_smooth_better": int(np.sum(arr > 0)),
        "n_tie": int(np.sum(arr == 0)),
        "rows": diffs,
    }
    (SIM_ROOT / "lds_vs_raw_smooth.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    return out


def main() -> int:
    print("M2 posthoc smooth + lag …", flush=True)
    report = run_m2_posthoc()
    for m in report["methods"]:
        if m["method"] in (
            "raw_smooth", "pca_smooth", "dm_smooth", "lds_smooth",
            "lds", "gpfa", "gpfa_causal",
        ):
            print(
                f"  {m['method']}: norm={m.get('normalized_error')} "
                f"tau={m.get('ema_tau_s')} lag_s={m.get('effective_lag_s')} "
                f"causal={m.get('causal')}",
                flush=True,
            )
    print("sim smooth rows …", flush=True)
    sim_rows = run_sim_smooth_rows()
    print(f"  wrote {len(sim_rows)} sim posthoc rows", flush=True)
    contrast = sim_lds_vs_raw_smooth()
    print(
        f"  sim lds-raw_smooth: mean={contrast['mean']} "
        f"n_lds_better={contrast['n_lds_better']}/"
        f"{contrast['n']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
