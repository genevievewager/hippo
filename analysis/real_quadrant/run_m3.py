"""M3: 2 sessions/animal cohort, room A, all methods + smoothed / gpfa_causal.

Parallel: one process per session, max 6 workers, BLAS threads = 8 each.
Resume-safe via per-method JSON keyed on config_sha256 + git_sha + dirty_tree.
Refuse a dirty working tree. Dry-run: one session, raw only, full parallel path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from analysis.real_quadrant.adapter import build_segment_bundle
from analysis.real_quadrant.posthoc_smooth import (
    EMA_TAUS_S,
    GPFA_CAUSAL_VERIFIED,
    LAG_MAX_S,
    SMOOTH_BASES,
    UPDATE_DT_S,
    apply_smoothed_method,
    effective_lag_s,
    _ridge_predict_all,
)
from analysis.real_quadrant.session_select import (
    M3_RULE,
    append_selection_manifest,
    select_m3_cohort,
)
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import (
    OUTPUT_ROOT,
    _euclid,
    _git_sha,
    _sqrt_zscore_train,
    analyze_source,
    require_clean_git_for_real_data,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "real_quadrant" / "m3"
LOG_DIR = OUT_ROOT / "logs"
MAX_WORKERS = 6
BLAS_THREADS = 8

M3_BASE_METHODS = ("raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa")

# Primary contrasts (recorded before any M3 results). Lower normalized error = better.
PRIMARY_CONTRASTS = (
    ("lds", "raw_smooth"),
    ("gpfa_causal", "raw_smooth"),  # omitted in aggregate if not verified / missing
    ("lds_smooth", "raw_smooth"),
    ("pca_smooth", "raw_smooth"),
    ("dm_smooth", "pca_smooth"),
    ("raw_smooth", "raw"),
)

# Commits whose realtime/analysis path for method fits is identical to HEAD for
# M3 resume purposes (only run_m3.py / tests changed after the cohort run).
M3_ANALYSIS_EQUIVALENT_GIT_SHAS = (
    "b68606eb9e154f0aa8b319ac151130fc6b8c0697",  # cohort run before fault-tolerance
    "c2dfb7c4ccce7e578ad0db00fec8da881df673e8",  # fault-tolerance before resume retarget
)


def retarget_method_resume_keys(
    out_dir: Path,
    methods: tuple[str, ...] | list[str],
    *,
    equivalent_shas: tuple[str, ...] = M3_ANALYSIS_EQUIVALENT_GIT_SHAS,
    cfg_sha: str | None = None,
) -> list[str]:
    """Rewrite git_sha on success JSONs from analysis-equivalent commits to HEAD.

    The resume key includes git_sha; when only run_m3/tests changed, retargeting
    lets completed method scores be reused without touching realtime/.
    Returns list of methods retargeted.
    """
    current = _git_sha()
    if not current:
        return []
    allowed = set(equivalent_shas) | {current}
    touched: list[str] = []
    for method in methods:
        path = _method_status_path(out_dir, method)
        if not path.is_file():
            continue
        try:
            rec = json.loads(path.read_text())
        except Exception:
            continue
        if _is_failed_or_skipped(rec):
            continue
        old = rec.get("git_sha")
        if old not in allowed:
            continue
        if cfg_sha is not None and rec.get("config_sha256") not in (None, cfg_sha):
            continue
        if bool(rec.get("dirty_tree", False)):
            continue
        if old == current:
            continue
        rec["git_sha_original"] = old
        rec["git_sha"] = current
        rec["dirty_tree"] = False
        rec["resume_retarget_note"] = (
            "git_sha retargeted: analysis path unchanged "
            "(diff limited to run_m3.py + tests)"
        )
        path.write_text(json.dumps(rec, indent=2, default=str) + "\n")
        touched.append(method)
    return touched
# Secondary: original unsmoothed sim-report contrasts.
SECONDARY_CONTRASTS = (
    ("dm", "pca"),
    ("lds", "pca"),
    ("raw_lag", "raw"),
    ("lds", "raw_lag"),
    ("pca", "raw"),
    ("dm", "raw"),
    ("lds", "raw"),
    ("isomap", "raw"),
    ("gpfa_causal", "raw"),
)


def _set_blas_threads(n: int = BLAS_THREADS) -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(int(n))


def _session_out_dir(session: str) -> Path:
    return OUT_ROOT / "room_A" / session


def _streams(seed_index: int) -> dict[str, int]:
    return {
        "methods": 0,
        "data_seed": int(seed_index),
        "master_seed": 0,
        "seed_index": int(seed_index),
        "trajectory": 0,
        "neural": 0,
        "recording_noise": 0,
        "sorting_errors": 0,
    }


def _method_status_path(out_dir: Path, method: str) -> Path:
    return out_dir / f"{method}.json"


def write_method_failure(out_dir: Path, method: str, exc: BaseException) -> dict[str, Any]:
    """Record a per-method failure without resume keys (so it is never reused)."""
    payload = {
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "method": method,
    }
    _method_status_path(out_dir, method).write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload


def write_method_skipped(
    out_dir: Path, method: str, *, reason: str,
) -> dict[str, Any]:
    payload = {
        "status": "skipped",
        "reason": reason,
        "method": method,
    }
    _method_status_path(out_dir, method).write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )
    return payload


def _is_failed_or_skipped(rec: dict[str, Any] | None) -> bool:
    if not isinstance(rec, dict):
        return False
    return rec.get("status") in ("failed", "skipped")


def _load_method_rec(out_dir: Path, method: str) -> dict[str, Any] | None:
    path = _method_status_path(out_dir, method)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_methods_fault_tolerant(
    *,
    cfg: dict[str, Any],
    bundle: dict[str, Any],
    streams: dict[str, int],
    seed_index: int,
    out_dir: Path,
    methods: tuple[str, ...],
    analyze_fn=analyze_source,
    log=None,
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Run each method independently; failures are recorded and skipped.

    Returns ``(succeeded, failures, summary)``. ``summary`` is from a final
    resume pass over succeeded methods that rebuilds the figure contract.

    When ``raw_lag`` is among planned methods it is included in every
    single-method call so train/eval masks match the full-grid intersection
    (raw_lag is the only method that drops leading samples).
    """
    failures: dict[str, dict[str, Any]] = {}
    include_raw_lag_anchor = "raw_lag" in methods

    for method in methods:
        keys: list[str] = []
        if include_raw_lag_anchor and method != "raw_lag":
            keys.append("raw_lag")
        keys.append(method)
        try:
            if log:
                log(f"method={method} keys={tuple(keys)} …")
            analyze_fn(
                cfg,
                None,
                "real",
                streams,
                seed_index,
                bundle=bundle,
                output_dir=out_dir,
                method_keys=tuple(keys),
                save_figure_contract=False,
            )
            rec = _load_method_rec(out_dir, method)
            if _is_failed_or_skipped(rec):
                raise RuntimeError(
                    f"method {method} left a non-success status on disk"
                )
            if log:
                log(f"method={method} ok")
        except Exception as exc:
            payload = write_method_failure(out_dir, method, exc)
            failures[method] = payload
            if log:
                log(f"method={method} FAILED {payload['error']}")

    succeeded = [
        m for m in methods
        if m not in failures and not _is_failed_or_skipped(_load_method_rec(out_dir, m))
    ]

    summary: dict[str, Any] | None = None
    if succeeded:
        summary = analyze_fn(
            cfg,
            None,
            "real",
            streams,
            seed_index,
            bundle=bundle,
            output_dir=out_dir,
            method_keys=tuple(succeeded),
            save_figure_contract=True,
        )
    return succeeded, failures, summary


def _lag_behind_truth(
    pred: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Effective lag with positive = prediction behind truth.

    Wraps ``effective_lag_s`` and flips the sign: the underlying search defines
    positive step as pred[t] ↔ y[t+step] (prediction leads). Causal EMA delay
    should appear as positive under this convention.
    """
    lag = effective_lag_s(pred, y_true)
    lag_s = lag["lag_s"]
    if lag_s is not None and np.isfinite(lag_s):
        lag = dict(lag)
        lag["lag_s"] = float(-lag_s)
        lag["lag_steps"] = float(-lag["lag_steps"]) if np.isfinite(lag.get("lag_steps", np.nan)) else float("nan")
    lag["sign_convention"] = "positive_means_prediction_behind_truth"
    return lag


def _apply_session_posthoc(
    out_dir: Path,
    *,
    seed_index: int,
    session: str,
    include_gpfa_causal: bool,
    failures: dict[str, dict[str, Any]] | None = None,
    git_sha_run: str | None = None,
) -> dict[str, Any]:
    """Add smoothed rows, optional gpfa_causal, and effective lag to session report."""
    failures = dict(failures or {})
    summary_path = out_dir / "source_summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"missing source_summary.json under {out_dir}")
    summary = json.loads(summary_path.read_text())
    floor_med = float(summary["floor"]["median"])
    cfg = load_quadrant_n5_yaml()
    n_blocks = int(cfg["split"]["inner_cv_blocks"])
    gap_s = float(cfg["split"]["gap_s"])

    contract = out_dir / "figure_contract"
    arrays = np.load(contract / "arrays.npz")
    latents = np.load(contract / "latents.npz") if (contract / "latents.npz").is_file() else {}
    if hasattr(latents, "files"):
        latents = {k: latents[k] for k in latents.files}
    preds = dict(np.load(contract / "predictions.npz"))
    times = np.asarray(arrays["decode_times"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    train_ok = np.asarray(arrays["train_ok"], dtype=bool)
    eval_mask = np.asarray(arrays["eval_mask"], dtype=bool)
    y_true = y[eval_mask]

    method_rows: dict[str, dict[str, Any]] = {}
    for m in summary.get("methods", []):
        name = m["method"]
        ridge_med = float(m["ridge"]["median"])
        method_rows[name] = {
            "method": name,
            "status": "ok",
            "selected_d": m.get("primary_d"),
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "knn_median_cm": float(m["knn"]["median"]),
            "a13_status": (m.get("a13") or {}).get("status"),
            "causal": name != "gpfa",
            "ema_tau_s": None,
        }
        if name == "gpfa":
            method_rows[name]["display_name"] = (
                "offline reference (non-causal smoother)"
            )
            method_rows[name]["role"] = "offline_reference"

    # Disk failures (including methods not in summary).
    for method, payload in failures.items():
        method_rows[method] = {
            "method": method,
            "status": "failed",
            "error": payload.get("error"),
            "normalized_error": None,
            "ridge_median_cm": None,
            "causal": method != "gpfa",
            "ema_tau_s": None,
        }
    for method in M3_BASE_METHODS:
        rec = _load_method_rec(out_dir, method)
        if _is_failed_or_skipped(rec) and method not in method_rows:
            method_rows[method] = {
                "method": method,
                "status": rec.get("status"),
                "error": rec.get("error"),
                "reason": rec.get("reason"),
                "normalized_error": None,
                "ridge_median_cm": None,
                "causal": method != "gpfa",
                "ema_tau_s": None,
            }
            if rec.get("status") == "failed":
                failures[method] = rec

    for base in SMOOTH_BASES:
        name = f"{base}_smooth"
        if base in failures or (
            _load_method_rec(out_dir, base) or {}
        ).get("status") == "failed":
            reason = (
                f"base method {base} failed: "
                f"{(failures.get(base) or _load_method_rec(out_dir, base) or {}).get('error')}"
            )
            write_method_skipped(out_dir, name, reason=reason)
            method_rows[name] = {
                "method": name,
                "base_method": base,
                "status": "skipped",
                "reason": reason,
                "normalized_error": None,
                "ridge_median_cm": None,
                "causal": True,
                "ema_tau_s": None,
            }
            continue
        if not (out_dir / f"{base}.json").is_file():
            continue
        if f"Z_{base}" not in latents:
            continue
        rec = json.loads((out_dir / f"{base}.json").read_text())
        if _is_failed_or_skipped(rec):
            continue
        Z = np.asarray(latents[f"Z_{base}"], dtype=float)
        primary_d = rec.get("primary_d")
        if primary_d is not None and Z.shape[1] > int(primary_d):
            Z = Z[:, : int(primary_d)]
        out = apply_smoothed_method(
            Z, y, times, train_ok, eval_mask, float(rec["ridge_alpha"]),
            n_blocks=n_blocks, gap_s=gap_s,
        )
        ridge_med = float(out["ridge"]["median"])
        method_rows[name] = {
            "method": name,
            "base_method": base,
            "status": "ok",
            "selected_d": primary_d,
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "knn_median_cm": None,
            "a13_status": None,
            "causal": True,
            "ema_tau_s": out["tau_s"],
            "inner_cv_median_cm": out["inner_cv_median_cm"],
        }
        preds[f"pred_{name}_ridge"] = np.asarray(out["pred_eval"], dtype=float)

    gpfa_failed = (
        "gpfa" in failures
        or (_load_method_rec(out_dir, "gpfa") or {}).get("status") == "failed"
    )
    if (
        include_gpfa_causal
        and GPFA_CAUSAL_VERIFIED
        and (out_dir / "gpfa.json").is_file()
        and not gpfa_failed
        and not _is_failed_or_skipped(_load_method_rec(out_dir, "gpfa"))
    ):
        row = _score_gpfa_causal(
            session, seed_index, arrays, times, y, train_ok, eval_mask,
            floor_med, out_dir,
        )
        if row is not None:
            method_rows["gpfa_causal"] = {**row["row"], "status": "ok"}
            preds["pred_gpfa_causal_ridge"] = row["pred_eval"]
    elif include_gpfa_causal and gpfa_failed:
        reason = f"base method gpfa failed: {(failures.get('gpfa') or {}).get('error')}"
        write_method_skipped(out_dir, "gpfa_causal", reason=reason)
        method_rows["gpfa_causal"] = {
            "method": "gpfa_causal",
            "status": "skipped",
            "reason": reason,
            "normalized_error": None,
            "causal": True,
        }

    for name, row in list(method_rows.items()):
        if row.get("status") not in (None, "ok"):
            continue
        key = f"pred_{name}_ridge"
        if key not in preds:
            continue
        lag = _lag_behind_truth(np.asarray(preds[key]), y_true)
        row["effective_lag_s"] = lag["lag_s"]
        row["effective_lag_corr"] = lag["corr"]
        row["lag_sign_convention"] = lag["sign_convention"]

    np.savez_compressed(contract / "predictions.npz", **{
        k: np.asarray(v) for k, v in preds.items()
    })

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

    report = {
        "session": session,
        "seed_index": int(seed_index),
        "n_units": int(summary["n_units"]),
        "n_train": int(summary["n_train"]),
        "n_eval": int(summary["n_eval"]),
        "floor_median_cm": floor_med,
        "methods": ordered,
        "failed_methods": failures,
        "gpfa_causal_verified": bool(GPFA_CAUSAL_VERIFIED),
        "posthoc_smooth_taus_s": list(EMA_TAUS_S),
        "effective_lag_search": {
            "dt_s": UPDATE_DT_S,
            "max_lag_s": LAG_MAX_S,
            "rule": "argmax mean axis-wise Pearson corr of pred vs truth",
            "sign_convention": "positive_means_prediction_behind_truth",
        },
        "figure_contract_dir": str(contract),
        "config_sha256": summary.get("config_sha256"),
        "git_sha": summary.get("git_sha") or _git_sha(),
        "git_sha_run": git_sha_run or _git_sha(),
    }
    (out_dir / "session_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report


def _score_gpfa_causal(
    session: str,
    seed_index: int,
    arrays: Any,
    times: np.ndarray,
    y: np.ndarray,
    train_ok: np.ndarray,
    eval_mask: np.ndarray,
    floor_med: float,
    out_dir: Path,
) -> dict[str, Any] | None:
    try:
        from realtime.dynamic_latents.gpfa import GPFAModel
    except Exception:
        return None
    model_dir = OUTPUT_ROOT / "models" / f"seed_{seed_index}" / "real" / "gpfa"
    if not (model_dir / "gpfa_params.npz").is_file():
        return None
    try:
        model = GPFAModel.load(model_dir)
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
    gpfa_rec = json.loads((out_dir / "gpfa.json").read_text())
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
        },
        "pred_eval": pred_te,
    }


def _run_one_session(job: dict[str, Any]) -> dict[str, Any]:
    """Worker entry: one session, methods from job, log to file."""
    _set_blas_threads(int(job.get("blas_threads", BLAS_THREADS)))
    session = job["session"]
    seed_index = int(job["seed_index"])
    methods = tuple(job["methods"])
    animal = job.get("animal")
    out_dir = _session_out_dir(session)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{session}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        with log_path.open("a") as f:
            f.write(line)
        print(f"[{session}] {msg}", flush=True)

    t0 = time.perf_counter()
    try:
        require_clean_git_for_real_data()
        data_root = Path(os.environ["HIPPO_DATA_ROOT"])
        _log(f"start methods={methods} seed_index={seed_index}")
        bundle = build_segment_bundle(session, room="A", data_root=data_root)
        cfg = dict(load_quadrant_n5_yaml())
        cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
        streams = _streams(seed_index)
        retargeted = retarget_method_resume_keys(
            out_dir, methods, cfg_sha=cfg.get("config_sha256"),
        )
        if retargeted:
            _log(
                f"retargeted resume git_sha→{_git_sha()[:12]} for {retargeted} "
                f"(analysis-equivalent prior commit)"
            )
        analyze_fn = analyze_source
        succeeded, failures, summary = run_methods_fault_tolerant(
            cfg=cfg,
            bundle=bundle,
            streams=streams,
            seed_index=seed_index,
            out_dir=out_dir,
            methods=methods,
            analyze_fn=analyze_fn,
            log=_log,
        )
        if not succeeded:
            raise RuntimeError(
                f"all methods failed: {[f.get('error') for f in failures.values()]}"
            )
        report = None
        if job.get("posthoc", True):
            report = _apply_session_posthoc(
                out_dir,
                seed_index=seed_index,
                session=session,
                include_gpfa_causal=bool(job.get("include_gpfa_causal", True)),
                failures=failures,
                git_sha_run=_git_sha(),
            )
        wall = time.perf_counter() - t0
        _log(
            f"done wall_s={wall:.1f} succeeded={succeeded} "
            f"failed={list(failures)}"
        )
        return {
            "ok": True,
            "session": session,
            "animal": animal,
            "seed_index": seed_index,
            "wall_s": wall,
            "n_units": int((summary or {}).get("n_units") or 0),
            "methods": list(succeeded),
            "failed_methods": failures,
            "report_path": str(out_dir / "session_report.json") if report else None,
            "git_sha": _git_sha(),
        }
    except Exception as exc:
        wall = time.perf_counter() - t0
        err = f"{type(exc).__name__}: {exc}"
        _log(f"FAILED {err}\n{traceback.format_exc()}")
        return {
            "ok": False,
            "session": session,
            "animal": animal,
            "seed_index": seed_index,
            "wall_s": wall,
            "error": err,
            "git_sha": _git_sha(),
        }


def _contrast_pair(
    a: str, b: str, animal_means: dict[str, dict[str, float]],
) -> dict[str, Any]:
    diffs = []
    for animal, means in sorted(animal_means.items()):
        if a not in means or b not in means:
            continue
        diffs.append(means[a] - means[b])
    arr = np.asarray(diffs, dtype=float)
    return {
        "contrast": f"{a} - {b}",
        "metric": "normalized_error",
        "n_animals": int(arr.size),
        "mean": float(np.mean(arr)) if arr.size else None,
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else None,
        "n_a_better": int(np.sum(arr < 0)),
        "n_b_better": int(np.sum(arr > 0)),
        "n_tie": int(np.sum(arr == 0)),
        "per_animal": {
            animal: float(means[a] - means[b])
            for animal, means in sorted(animal_means.items())
            if a in means and b in means
        },
    }


def aggregate_m3(
    cohort: dict[str, Any],
    *,
    include_gpfa_causal: bool = True,
) -> dict[str, Any]:
    """Per-animal means (sessions nested), then planned contrasts with sign counts.

    A contrast that requires a failed/skipped method drops that animal for that
    contrast only. ``n_animals`` on each contrast is the count retained.
    """
    session_rows = []
    method_failures: list[dict[str, Any]] = []
    git_shas: set[str] = set()
    for sel in cohort["selected_sessions"]:
        session = sel["session"]
        path = _session_out_dir(session) / "session_report.json"
        if not path.is_file():
            continue
        report = json.loads(path.read_text())
        sha = report.get("git_sha") or report.get("git_sha_run")
        if sha:
            git_shas.add(str(sha))
        run_sha = report.get("git_sha_run")
        if run_sha:
            git_shas.add(str(run_sha))
        by_m = {m["method"]: m for m in report["methods"]}
        for m in report["methods"]:
            if m.get("status") == "failed":
                method_failures.append({
                    "session": session,
                    "animal": sel["animal"],
                    "method": m["method"],
                    "error": m.get("error"),
                })
        session_rows.append({
            "animal": sel["animal"],
            "session": session,
            "git_sha": report.get("git_sha"),
            "git_sha_run": report.get("git_sha_run"),
            "methods": {
                name: {
                    "normalized_error": m.get("normalized_error"),
                    "effective_lag_s": m.get("effective_lag_s"),
                    "ridge_median_cm": m.get("ridge_median_cm"),
                    "causal": m.get("causal"),
                    "status": m.get("status", "ok"),
                }
                for name, m in by_m.items()
            },
        })

    by_animal: dict[str, list[dict[str, Any]]] = {}
    for row in session_rows:
        by_animal.setdefault(row["animal"], []).append(row)

    animal_means: dict[str, dict[str, float]] = {}
    animal_lag_means: dict[str, dict[str, float]] = {}
    animal_n_sessions: dict[str, int] = {}
    for animal, rows in by_animal.items():
        animal_n_sessions[animal] = len(rows)
        method_vals: dict[str, list[float]] = {}
        lag_vals: dict[str, list[float]] = {}
        for row in rows:
            for name, rec in row["methods"].items():
                if rec.get("status") not in (None, "ok"):
                    continue
                ne = rec.get("normalized_error")
                if ne is not None and np.isfinite(ne):
                    method_vals.setdefault(name, []).append(float(ne))
                lag = rec.get("effective_lag_s")
                if lag is not None and np.isfinite(lag):
                    lag_vals.setdefault(name, []).append(float(lag))
        animal_means[animal] = {
            k: float(np.mean(v)) for k, v in method_vals.items()
        }
        animal_lag_means[animal] = {
            k: float(np.mean(v)) for k, v in lag_vals.items()
        }

    primary = list(PRIMARY_CONTRASTS)
    if not include_gpfa_causal or not GPFA_CAUSAL_VERIFIED:
        primary = [c for c in primary if "gpfa_causal" not in c]
    secondary = list(SECONDARY_CONTRASTS)
    if not include_gpfa_causal or not GPFA_CAUSAL_VERIFIED:
        secondary = [c for c in secondary if "gpfa_causal" not in c]

    notes = []
    isomap_fails = [
        f for f in method_failures if f["method"] == "isomap"
    ]
    if isomap_fails:
        notes.append(ISOMAP_FAILURE_NOTE)

    out = {
        "n_sessions_completed": len(session_rows),
        "n_animals": len(animal_means),
        "animal_n_sessions": animal_n_sessions,
        "animal_means_normalized_error": animal_means,
        "animal_means_effective_lag_s": animal_lag_means,
        "primary_contrasts": [_contrast_pair(a, b, animal_means) for a, b in primary],
        "secondary_contrasts": [
            _contrast_pair(a, b, animal_means) for a, b in secondary
        ],
        "planned_primary": [f"{a} - {b}" for a, b in primary],
        "planned_secondary": [f"{a} - {b}" for a, b in secondary],
        "gpfa_causal_verified": bool(GPFA_CAUSAL_VERIFIED),
        "method_failures": method_failures,
        "notes": notes,
        "git_shas": sorted(git_shas),
        "sessions": session_rows,
    }
    (OUT_ROOT / "m3_aggregate.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    return out


def _launch_parallel(jobs: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    """One OS process per session (fresh BLAS env); at most ``max_workers`` live."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    live: dict[subprocess.Popen, tuple[dict[str, Any], Path]] = {}
    results: list[dict[str, Any]] = []
    n_workers = max(1, min(int(max_workers), len(jobs) or 1))
    py = sys.executable

    def _start(job: dict[str, Any]) -> None:
        job_path = LOG_DIR / f"job_{job['session']}.json"
        result_path = LOG_DIR / f"result_{job['session']}.json"
        if result_path.is_file():
            result_path.unlink()
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        env = os.environ.copy()
        n_thr = str(int(job.get("blas_threads", BLAS_THREADS)))
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[var] = n_thr
        log_path = LOG_DIR / f"{job['session']}.log"
        log_f = log_path.open("a")
        proc = subprocess.Popen(
            [py, "-m", "analysis.real_quadrant.run_m3", "--worker-job", str(job_path)],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        live[proc] = (job, result_path)
        # Keep log handle alive until process ends via proc reference; close later.
        proc._hippo_log_f = log_f  # type: ignore[attr-defined]

    while pending or live:
        while pending and len(live) < n_workers:
            _start(pending.pop(0))
        done = [p for p in live if p.poll() is not None]
        if not done:
            time.sleep(0.5)
            continue
        for proc in done:
            job, result_path = live.pop(proc)
            log_f = getattr(proc, "_hippo_log_f", None)
            if log_f is not None:
                log_f.close()
            if result_path.is_file():
                results.append(json.loads(result_path.read_text()))
            else:
                results.append({
                    "ok": False,
                    "session": job["session"],
                    "animal": job.get("animal"),
                    "seed_index": job.get("seed_index"),
                    "error": f"worker exit={proc.returncode}; no result JSON",
                })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="1 session, raw only, through the full parallel path (no real M3).",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--worker-job",
        type=str,
        default=None,
        help="Internal: path to job JSON for a single-session worker process.",
    )
    args = parser.parse_args(argv)

    if args.worker_job:
        job = json.loads(Path(args.worker_job).read_text())
        _set_blas_threads(int(job.get("blas_threads", BLAS_THREADS)))
        result = _run_one_session(job)
        result_path = LOG_DIR / f"result_{job['session']}.json"
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
        return 0 if result.get("ok") else 1

    require_clean_git_for_real_data()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data_root = Path(os.environ["HIPPO_DATA_ROOT"])
    cohort = select_m3_cohort(data_root)
    cohort_path = OUT_ROOT / "cohort_manifest.json"
    if not args.aggregate_only:
        # Record cohort before any analysis.
        cohort_path.write_text(json.dumps(cohort, indent=2, default=str) + "\n")
        append_selection_manifest({**cohort, "milestone": "M3", "dry_run": bool(args.dry_run)})
        print(
            f"M3 cohort: {cohort['n_selected_sessions']} sessions across "
            f"{cohort['n_animals']} animals (rule recorded in {cohort_path})",
            flush=True,
        )
        for a in cohort["animals"]:
            print(
                f"  animal={a['animal']}: selected={a['n_selected']}/"
                f"eligible={a['n_eligible']} excluded={a['n_excluded']} "
                f"reasons={a['exclude_reason_counts']}"
                + (f" note={a['note']}" if a.get("note") else ""),
                flush=True,
            )

    if args.aggregate_only:
        if not cohort_path.is_file():
            raise SystemExit("no cohort_manifest.json; run without --aggregate-only first")
        cohort = json.loads(cohort_path.read_text())
        agg = aggregate_m3(cohort)
        print(json.dumps({
            "n_animals": agg["n_animals"],
            "primary": agg["primary_contrasts"],
        }, indent=2), flush=True)
        return 0

    selected = list(cohort["selected_sessions"])
    if args.dry_run:
        if not selected:
            raise SystemExit("dry-run: no selected sessions")
        selected = selected[:1]
        methods = ("raw",)
        posthoc = True
        include_gpfa = False
        print(
            f"DRY-RUN: 1 session={selected[0]['session']}, methods={methods}",
            flush=True,
        )
    else:
        methods = M3_BASE_METHODS
        posthoc = True
        include_gpfa = True

    jobs = []
    for i, sel in enumerate(selected):
        jobs.append({
            "session": sel["session"],
            "animal": sel["animal"],
            "seed_index": i,
            "methods": list(methods),
            "posthoc": posthoc,
            "include_gpfa_causal": include_gpfa,
            "blas_threads": BLAS_THREADS,
        })

    t0 = time.perf_counter()
    results = _launch_parallel(jobs, args.max_workers)
    wall = time.perf_counter() - t0
    (OUT_ROOT / ("dry_run_results.json" if args.dry_run else "run_results.json")).write_text(
        json.dumps({"wall_s": wall, "results": results}, indent=2, default=str) + "\n"
    )
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"finished {n_ok}/{len(results)} sessions in {wall:.1f}s", flush=True)

    if not args.dry_run and n_ok == len(results):
        agg = aggregate_m3(cohort, include_gpfa_causal=include_gpfa)
        print(
            f"aggregate: n_animals={agg['n_animals']} "
            f"primary_contrasts={len(agg['primary_contrasts'])}",
            flush=True,
        )
    elif args.dry_run and n_ok:
        print("dry-run ok; skipping full cohort aggregate", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
