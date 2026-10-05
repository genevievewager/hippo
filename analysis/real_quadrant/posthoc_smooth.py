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
SMOOTH_BASES = ("raw", "pca", "dm")
GPFA_LABEL = "offline reference (non-causal smoother)"
CAUSAL_METHODS = (
    "raw", "raw_lag", "pca", "dm", "lds", "isomap",
    "raw_smooth", "pca_smooth", "dm_smooth", "gpfa_causal",
)
OFFLINE_METHODS = ("gpfa",)


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


def run_m2_posthoc() -> dict[str, Any]:
    """Add smoothed methods to M2 report using saved figure-contract latents."""
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

    method_rows = {m["method"]: dict(m) for m in report["methods"]}
    # Relabel GPFA as offline reference.
    if "gpfa" in method_rows:
        method_rows["gpfa"]["display_name"] = GPFA_LABEL
        method_rows["gpfa"]["causal"] = False
        method_rows["gpfa"]["role"] = "offline_reference"

    for base in SMOOTH_BASES:
        rec = json.loads((M2_DIR / f"{base}.json").read_text())
        Z = np.asarray(latents[f"Z_{base}"], dtype=float)
        # Nested methods: latents may be nested_fit_d; slice to primary_d.
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

    # Optional GPFA causal filter from saved model + rebuilt features.
    gpfa_causal_row = _try_gpfa_causal(
        report, arrays, times, y, train_ok, eval_mask, floor_med, n_blocks, gap_s,
    )
    if gpfa_causal_row is not None:
        method_rows["gpfa_causal"] = gpfa_causal_row["row"]
        preds["pred_gpfa_causal_ridge"] = gpfa_causal_row["pred_eval"]

    # Persist updated predictions (eval-aligned) without touching base keys' science.
    np.savez_compressed(contract / "predictions.npz", **{
        k: np.asarray(v) for k, v in preds.items()
    })

    # Mark causal flags on base methods.
    for m, row in method_rows.items():
        if m == "gpfa":
            row["causal"] = False
            row.setdefault("display_name", GPFA_LABEL)
        elif m == "gpfa_causal":
            row["causal"] = True
        else:
            row.setdefault("causal", True)

    ordered = []
    for key in (
        "raw", "raw_smooth", "raw_lag", "pca", "pca_smooth", "dm", "dm_smooth",
        "lds", "isomap", "gpfa_causal", "gpfa",
    ):
        if key in method_rows:
            ordered.append(method_rows[key])
    for key, row in method_rows.items():
        if key not in {r["method"] for r in ordered}:
            ordered.append(row)

    report["methods"] = ordered
    report["gpfa_label"] = GPFA_LABEL
    report["posthoc_smooth_taus_s"] = list(EMA_TAUS_S)
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
        from analysis.real_quadrant.session_select import select_median_units_2room_session
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
    # Align length with saved arrays if needed.
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
            "display_name": "GPFA causal filter (diagnostic)",
            "selected_d": primary_d,
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "knn_median_cm": None,
            "a13_status": None,
            "causal": True,
            "role": "gpfa_filter_diagnostic",
            "ema_tau_s": None,
            "fit_s": None,
            "transform_s": None,
            "elapsed_s": None,
        },
        "pred_eval": pred_te,
    }


def run_sim_raw_smooth() -> list[dict[str, Any]]:
    """Write ``raw_smooth_posthoc.json`` under each sim seed/source (new rows only)."""
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
            rec = json.loads((src_dir / "raw.json").read_text())
            model_dir = SIM_ROOT / "models" / f"seed_{seed}" / src / "raw"
            Z, route = latents_for_method(
                "raw", cfg, obs, rec, model_dir, int(streams["methods"]),
            )
            out = apply_smoothed_method(
                Z, obs["y"], obs["decode_times"], obs["train_ok"], obs["eval_mask"],
                float(rec["ridge_alpha"]),
                n_blocks=n_blocks, gap_s=gap_s,
            )
            ridge_med = float(out["ridge"]["median"])
            payload = {
                "stage": "posthoc_smooth",
                "method": "raw_smooth",
                "base_method": "raw",
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
                "latent_route": route,
                "note": "New row only; existing method JSONs untouched.",
            }
            (src_dir / "raw_smooth_posthoc.json").write_text(
                json.dumps(payload, indent=2, default=str) + "\n"
            )
            rows_out.append(payload)
    summary_path = SIM_ROOT / "raw_smooth_posthoc_summary.json"
    summary_path.write_text(json.dumps({"rows": rows_out}, indent=2, default=str) + "\n")
    return rows_out


def main() -> int:
    print("M2 posthoc smooth …", flush=True)
    report = run_m2_posthoc()
    for m in report["methods"]:
        if m["method"].endswith("_smooth") or m["method"] in ("gpfa", "gpfa_causal", "lds"):
            print(
                f"  {m['method']}: norm={m.get('normalized_error')} "
                f"tau={m.get('ema_tau_s')} causal={m.get('causal')}",
                flush=True,
            )
    print("sim raw_smooth …", flush=True)
    sim_rows = run_sim_raw_smooth()
    print(f"  wrote {len(sim_rows)} sim posthoc rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
