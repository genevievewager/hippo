"""Extend real-data pca/lds latent-d grid; archive grid20 rows alongside.

Rule (PLAN): if selected d hits the grid max on most sessions, double the grid
until the mode selection is below the max or d reaches n_units/2.
Sim grid unchanged.

LDS timing probe (1120_10062024): d=80 one-fold ~162 s; marginal d=80 per
session ~0.28 h (< 3 h) → use full grid {2,3,5,10,20,40,80}.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from analysis.real_quadrant.adapter import build_segment_bundle
from analysis.real_quadrant.posthoc_smooth import (
    apply_smoothed_method,
    _ridge_predict_all,
)
from analysis.real_quadrant.run_m3 import (
    BLAS_THREADS,
    LOG_DIR,
    MAX_WORKERS,
    OUT_ROOT,
    _lag_behind_truth,
    _session_out_dir,
    _set_blas_threads,
    _streams,
)
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import _euclid, _git_sha, analyze_source, require_clean_git_for_real_data

REPO_ROOT = Path(__file__).resolve().parents[2]

# Full extended grid (probe: LDS d=80 under 3 h/session).
GRID_EXT = (2, 3, 5, 10, 20, 40, 80)
GRID20 = (2, 3, 5, 10, 20)
METHODS = ("pca", "lds")


def _overlay_cfg() -> dict[str, Any]:
    import hashlib

    cfg = dict(load_quadrant_n5_yaml())
    cfg["latent_dims"] = list(GRID_EXT)
    cfg["nested_fit_d"] = int(max(GRID_EXT))  # pca nested fit must cover max d
    cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
    cfg["grid_extension"] = {
        "rule": (
            "if selected d equals grid max on most sessions, double until "
            "mode selected d < max or d reaches n_units/2"
        ),
        "grid20": list(GRID20),
        "grid_ext": list(GRID_EXT),
        "nested_fit_d": int(max(GRID_EXT)),
        "sim_grid_unchanged": True,
    }
    # YAML file sha alone would not change; stamp overlay so resume keys invalidate
    # grid20 pca/lds and distinguish extended-grid artifacts.
    stamp = hashlib.sha256(
        json.dumps(
            {
                "base_config_sha256": cfg["config_sha256"],
                "latent_dims": cfg["latent_dims"],
                "nested_fit_d": cfg["nested_fit_d"],
                "grid_extension": True,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    cfg["config_sha256"] = stamp
    return cfg


def _archive_grid20(out_dir: Path) -> list[str]:
    """Copy pca/lds JSONs to *_grid20.json once; remove live JSONs so they refit.

    Resume-safe: if ``*_grid20.json`` already exists, leave current ``*.json``
    in place so analyze_source can skip completed extended-grid fits.
    """
    touched = []
    for m in METHODS:
        src = out_dir / f"{m}.json"
        dst = out_dir / f"{m}_grid20.json"
        if not src.is_file():
            continue
        rec = json.loads(src.read_text())
        if rec.get("status") in ("failed", "skipped"):
            continue
        if not dst.is_file():
            shutil.copy2(src, dst)
            src.unlink()
            touched.append(f"{m}->grid20 (removed live for refit)")
        else:
            touched.append(f"{m}_grid20 exists; keep live for resume")
    fc = out_dir / "figure_contract"
    fc_bak = out_dir / "figure_contract_grid20"
    if fc.is_dir() and not fc_bak.is_dir():
        shutil.copytree(fc, fc_bak)
        touched.append("figure_contract_grid20")
    ds = out_dir / "d_sweep.json"
    ds_bak = out_dir / "d_sweep_grid20.json"
    if ds.is_file() and not ds_bak.is_file():
        shutil.copy2(ds, ds_bak)
        touched.append("d_sweep_grid20")
    return touched


def _merge_figure_contract(out_dir: Path) -> None:
    """Restore non-pca/lds arrays from grid20 backup; keep new pca/lds."""
    fc = out_dir / "figure_contract"
    bak = out_dir / "figure_contract_grid20"
    if not bak.is_dir() or not fc.is_dir():
        return
    # arrays.npz: prefer new (same masks) but keep backup if new missing
    for name in ("arrays.npz",):
        if not (fc / name).is_file() and (bak / name).is_file():
            shutil.copy2(bak / name, fc / name)

    def _merge_npz(primary: Path, backup: Path, keep_prefixes: tuple[str, ...]) -> None:
        if not backup.is_file():
            return
        old = dict(np.load(backup))
        new = dict(np.load(primary)) if primary.is_file() else {}
        merged = dict(old)
        for k, v in new.items():
            if any(k == p or k.startswith(p) for p in keep_prefixes):
                merged[k] = v
        # Drop subsample_rule object arrays carefully
        np.savez_compressed(primary, **{
            k: np.asarray(v) for k, v in merged.items()
        })

    _merge_npz(
        fc / "latents.npz", bak / "latents.npz",
        ("Z_pca", "Z_lds", "subsample_rule"),
    )
    _merge_npz(
        fc / "predictions.npz", bak / "predictions.npz",
        ("pred_pca_", "pred_lds_", "eval_mask", "decode_times", "y_true"),
    )
    # a13: merge method keys
    a13_path = fc / "a13.json"
    a13_bak = bak / "a13.json"
    if a13_bak.is_file():
        old = json.loads(a13_bak.read_text())
        new = json.loads(a13_path.read_text()) if a13_path.is_file() else {}
        old.update({k: new[k] for k in ("pca", "lds") if k in new})
        a13_path.write_text(json.dumps(old, indent=2, default=str) + "\n")


def _merge_source_summary(out_dir: Path, new_summary: dict[str, Any]) -> dict[str, Any]:
    path = out_dir / "source_summary.json"
    # Prefer grid20-era summary if we stashed it
    stash = out_dir / "source_summary_grid20.json"
    if path.is_file() and not stash.is_file():
        # Before overwrite analyze_source already wrote new — use backup from
        # figure_contract_grid20 era: we stash on first archive.
        pass
    base = {}
    if stash.is_file():
        base = json.loads(stash.read_text())
    elif (out_dir / "figure_contract_grid20").exists():
        # Fall back: rebuild method list from per-method JSONs
        base = {"methods": []}
        for m in ("raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"):
            p = out_dir / f"{m}_grid20.json" if m in METHODS else out_dir / f"{m}.json"
            if m in METHODS and not p.is_file():
                p = out_dir / f"{m}.json"
            if p.is_file():
                rec = json.loads(p.read_text())
                if rec.get("status") not in ("failed", "skipped") and "ridge" in rec:
                    base["methods"].append(rec)
    else:
        base = json.loads(path.read_text()) if path.is_file() else {"methods": []}

    by_m = {
        m["method"]: m for m in base.get("methods", [])
        if isinstance(m, dict) and "method" in m
    }
    for m in new_summary.get("methods", []):
        by_m[m["method"]] = m
    out = dict(base)
    out.update({
        k: new_summary[k] for k in (
            "n_units", "n_train", "n_eval", "floor", "config_sha256", "git_sha",
            "dirty_tree",
        ) if k in new_summary
    })
    order = ["raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"]
    methods = [by_m[k] for k in order if k in by_m]
    for k, v in by_m.items():
        if k not in {m["method"] for m in methods}:
            methods.append(v)
    out["methods"] = methods
    path.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def _stash_source_summary(out_dir: Path) -> None:
    src = out_dir / "source_summary.json"
    dst = out_dir / "source_summary_grid20.json"
    if src.is_file() and not dst.is_file():
        shutil.copy2(src, dst)


def _row_from_method_json(
    rec: dict[str, Any], floor_med: float, *, label: str | None = None,
) -> dict[str, Any]:
    ridge_med = float(rec["ridge"]["median"])
    name = rec["method"]
    return {
        "method": f"{name}_{label}" if label else name,
        "base_method": name if label else None,
        "grid_label": label or "grid_ext",
        "status": "ok",
        "selected_d": rec.get("primary_d"),
        "ridge_median_cm": ridge_med,
        "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
        "knn_median_cm": float(rec["knn"]["median"]) if rec.get("knn") else None,
        "a13_status": (rec.get("a13") or {}).get("status"),
        "causal": True,
        "ema_tau_s": None,
    }


def _row_from_pred(
    name: str,
    pred: np.ndarray,
    y_true: np.ndarray,
    floor_med: float,
    **extra: Any,
) -> dict[str, Any]:
    err = np.asarray(pred, float) - np.asarray(y_true, float)
    ridge_med = float(np.median(np.linalg.norm(err, axis=1)))
    row = {
        "method": name,
        "status": "ok",
        "ridge_median_cm": ridge_med,
        "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
        "causal": name != "gpfa",
        "ema_tau_s": None,
    }
    row.update(extra)
    return row


def _eval_lag_plateau(pred: np.ndarray, y_true: np.ndarray) -> dict[str, Any]:
    """Eval lag plateau under behind-truth convention; no point estimate."""
    from analysis.real_quadrant.posthoc_smooth import UPDATE_DT_S, LAG_MAX_S

    pred = np.asarray(pred, float)
    y_true = np.asarray(y_true, float)
    finite = np.isfinite(pred).all(1) & np.isfinite(y_true).all(1)
    p, yt = pred[finite], y_true[finite]
    max_steps = int(round(LAG_MAX_S / UPDATE_DT_S))
    lags_c: list[tuple[float, float]] = []
    for step in range(-max_steps, max_steps + 1):
        if step < 0:
            aa, bb = p[-step:], yt[: len(yt) + step]
        elif step > 0:
            aa, bb = p[: len(p) - step], yt[step:]
        else:
            aa, bb = p, yt
        if len(aa) < 20:
            continue
        cs = []
        for d in range(aa.shape[1]):
            if np.std(aa[:, d]) < 1e-12 or np.std(bb[:, d]) < 1e-12:
                continue
            cs.append(float(np.corrcoef(aa[:, d], bb[:, d])[0, 1]))
        if cs:
            lags_c.append((-step * UPDATE_DT_S, float(np.mean(cs))))
    out: dict[str, Any] = {
        "effective_lag_s": None,
        "lag_sign_convention": "positive_means_prediction_behind_truth",
    }
    if lags_c:
        arr = np.asarray(lags_c)
        i = int(np.argmax(arr[:, 1]))
        peak_c = float(arr[i, 1])
        near = arr[:, 1] >= (peak_c - 0.01)
        out["eval_lag_plateau_s"] = [
            float(arr[near, 0].min()), float(arr[near, 0].max()),
        ]
        out["eval_lag_peak_corr"] = peak_c
    return out


def _update_session_report(out_dir: Path, session: str, seed_index: int) -> dict[str, Any]:
    """Rebuild session_report keeping grid20 pca/lds and new extended rows + smooth."""
    from analysis.real_quadrant.posthoc_smooth import GPFA_CAUSAL_VERIFIED, SMOOTH_BASES
    from analysis.real_quadrant.run_m3 import _score_gpfa_causal

    cfg = _overlay_cfg()
    n_blocks = int(cfg["split"]["inner_cv_blocks"])
    gap_s = float(cfg["split"]["gap_s"])
    summary = json.loads((out_dir / "source_summary.json").read_text())
    floor_med = float(summary["floor"]["median"])
    contract = out_dir / "figure_contract"
    arrays = np.load(contract / "arrays.npz")
    latents = np.load(contract / "latents.npz")
    preds = dict(np.load(contract / "predictions.npz"))
    times = np.asarray(arrays["decode_times"], float)
    y = np.asarray(arrays["y"], float)
    train_ok = np.asarray(arrays["train_ok"], bool)
    eval_mask = np.asarray(arrays["eval_mask"], bool)
    y_true = y[eval_mask]

    method_rows: dict[str, dict[str, Any]] = {}
    # Existing non-pca/lds from summary
    for m in summary.get("methods", []):
        name = m["method"]
        if name in METHODS:
            continue
        if "ridge" not in m:
            continue
        ridge_med = float(m["ridge"]["median"])
        method_rows[name] = {
            "method": name,
            "status": "ok",
            "selected_d": m.get("primary_d"),
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "knn_median_cm": float(m["knn"]["median"]) if m.get("knn") else None,
            "a13_status": (m.get("a13") or {}).get("status"),
            "causal": name != "gpfa",
            "ema_tau_s": None,
            "grid_label": "unchanged",
        }

    # Non-pca/lds smooth controls (raw, dm) from current latents
    for base in SMOOTH_BASES:
        if base in METHODS:
            continue
        if f"Z_{base}" not in latents:
            continue
        base_path = out_dir / f"{base}.json"
        if not base_path.is_file():
            continue
        rec = json.loads(base_path.read_text())
        if rec.get("status") in ("failed", "skipped") or "ridge_alpha" not in rec:
            continue
        Z = np.asarray(latents[f"Z_{base}"], float)
        primary_d = rec.get("primary_d")
        if primary_d is not None and Z.shape[1] > int(primary_d):
            Z = Z[:, : int(primary_d)]
        sm = apply_smoothed_method(
            Z, y, times, train_ok, eval_mask, float(rec["ridge_alpha"]),
            n_blocks=n_blocks, gap_s=gap_s,
        )
        name = f"{base}_smooth"
        ridge_med = float(sm["ridge"]["median"])
        method_rows[name] = {
            "method": name,
            "base_method": base,
            "grid_label": "unchanged",
            "status": "ok",
            "selected_d": primary_d,
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "causal": True,
            "ema_tau_s": sm["tau_s"],
            "inner_cv_median_cm": sm["inner_cv_median_cm"],
        }
        preds[f"pred_{name}_ridge"] = np.asarray(sm["pred_eval"], float)

    # grid20 archived pca/lds
    for m in METHODS:
        p = out_dir / f"{m}_grid20.json"
        if p.is_file():
            rec = json.loads(p.read_text())
            if "ridge" in rec:
                row = _row_from_method_json(rec, floor_med, label="grid20")
                method_rows[row["method"]] = row

    # new extended pca/lds + smooth
    for m in METHODS:
        p = out_dir / f"{m}.json"
        if not p.is_file():
            continue
        rec = json.loads(p.read_text())
        if rec.get("status") in ("failed", "skipped") or "ridge" not in rec:
            continue
        row = _row_from_method_json(rec, floor_med, label=None)
        row["method"] = m
        row["grid_label"] = "grid_ext"
        method_rows[m] = row

        Z = np.asarray(latents[f"Z_{m}"], float)
        primary_d = rec.get("primary_d")
        if primary_d is not None and Z.shape[1] > int(primary_d):
            Z = Z[:, : int(primary_d)]
        sm = apply_smoothed_method(
            Z, y, times, train_ok, eval_mask, float(rec["ridge_alpha"]),
            n_blocks=n_blocks, gap_s=gap_s,
        )
        name = f"{m}_smooth"
        ridge_med = float(sm["ridge"]["median"])
        method_rows[name] = {
            "method": name,
            "base_method": m,
            "grid_label": "grid_ext",
            "status": "ok",
            "selected_d": primary_d,
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "causal": True,
            "ema_tau_s": sm["tau_s"],
            "inner_cv_median_cm": sm["inner_cv_median_cm"],
        }
        preds[f"pred_{name}_ridge"] = np.asarray(sm["pred_eval"], float)

    # grid20 smooth from backup latents if present
    bak = out_dir / "figure_contract_grid20" / "latents.npz"
    if bak.is_file():
        lat_old = np.load(bak)
        for m in METHODS:
            gp = out_dir / f"{m}_grid20.json"
            if not gp.is_file() or f"Z_{m}" not in lat_old.files:
                continue
            rec = json.loads(gp.read_text())
            if "ridge" not in rec:
                continue
            Z = np.asarray(lat_old[f"Z_{m}"], float)
            primary_d = rec.get("primary_d")
            if primary_d is not None and Z.shape[1] > int(primary_d):
                Z = Z[:, : int(primary_d)]
            sm = apply_smoothed_method(
                Z, y, times, train_ok, eval_mask, float(rec["ridge_alpha"]),
                n_blocks=n_blocks, gap_s=gap_s,
            )
            name = f"{m}_smooth_grid20"
            ridge_med = float(sm["ridge"]["median"])
            method_rows[name] = {
                "method": name,
                "base_method": m,
                "grid_label": "grid20",
                "status": "ok",
                "selected_d": primary_d,
                "ridge_median_cm": ridge_med,
                "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
                "causal": True,
                "ema_tau_s": sm["tau_s"],
            }
            preds[f"pred_{name}_ridge"] = np.asarray(sm["pred_eval"], float)

    # gpfa_causal: prefer existing pred; else rescore
    if "pred_gpfa_causal_ridge" in preds:
        method_rows["gpfa_causal"] = _row_from_pred(
            "gpfa_causal",
            preds["pred_gpfa_causal_ridge"],
            y_true,
            floor_med,
            grid_label="unchanged",
            causal=True,
            role="linear_dynamic_causal",
        )
    elif GPFA_CAUSAL_VERIFIED and (out_dir / "gpfa.json").is_file():
        scored = _score_gpfa_causal(
            session, seed_index, arrays, times, y, train_ok, eval_mask,
            floor_med, out_dir,
        )
        if scored is not None:
            method_rows["gpfa_causal"] = {**scored["row"], "status": "ok", "grid_label": "unchanged"}
            preds["pred_gpfa_causal_ridge"] = scored["pred_eval"]

    # Failed isomap etc. from disk
    for m in ("isomap",):
        rec_path = out_dir / f"{m}.json"
        if rec_path.is_file():
            rec = json.loads(rec_path.read_text())
            if rec.get("status") == "failed":
                method_rows[m] = {
                    "method": m, "status": "failed", "error": rec.get("error"),
                    "normalized_error": None, "causal": True,
                }

    # Per-session train sync lag (behind-truth = −lead from sync_check vel)
    sync_path = OUT_ROOT / "sync_check.json"
    train_lag: dict[str, Any] | None = None
    if sync_path.is_file():
        sync = json.loads(sync_path.read_text())
        for row in sync.get("rows") or []:
            if row.get("session") == session:
                # sync stores lead convention; flip to behind-truth
                vel = row.get("vel_lag_s")
                rate = row.get("rate_lag_s")
                train_lag = {
                    "vel_behind_truth_s": float(-vel) if vel is not None else None,
                    "rate_behind_truth_s": float(-rate) if rate is not None else None,
                    "vel_corr": row.get("vel_corr"),
                    "rate_corr": row.get("rate_corr"),
                    "sign_convention": "positive_means_prediction_behind_truth",
                    "source": "sync_check.json train XC",
                }
                break

    for name, row in list(method_rows.items()):
        if row.get("status") not in (None, "ok"):
            continue
        key = f"pred_{name}_ridge"
        if key not in preds:
            continue
        row.update(_eval_lag_plateau(np.asarray(preds[key]), y_true))
        if train_lag is not None:
            row["train_lag"] = train_lag

    np.savez_compressed(contract / "predictions.npz", **{
        k: np.asarray(v) for k, v in preds.items()
    })

    ordered_keys = [
        "raw", "raw_smooth", "raw_lag",
        "pca_grid20", "pca", "pca_smooth_grid20", "pca_smooth",
        "dm", "dm_smooth",
        "lds_grid20", "lds", "lds_smooth_grid20", "lds_smooth",
        "isomap", "gpfa_causal", "gpfa",
    ]
    ordered = [method_rows[k] for k in ordered_keys if k in method_rows]
    for k, row in method_rows.items():
        if k not in {r["method"] for r in ordered}:
            ordered.append(row)

    report = {
        "session": session,
        "seed_index": int(seed_index),
        "n_units": int(summary["n_units"]),
        "n_train": int(summary["n_train"]),
        "n_eval": int(summary["n_eval"]),
        "floor_median_cm": floor_med,
        "methods": ordered,
        "grid_extension": cfg["grid_extension"],
        "train_lag": train_lag,
        "effective_lag_search": {
            "sign_convention": "positive_means_prediction_behind_truth",
            "eval_primary": "plateau_range",
            "eval_point_estimate": "dropped",
            "train_primary": "sync_check train XC (behind-truth)",
        },
        "config_sha256": summary.get("config_sha256"),
        "git_sha": _git_sha(),
        "figure_contract_dir": str(contract),
    }
    (out_dir / "session_report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    return report


def _run_one_session(job: dict[str, Any]) -> dict[str, Any]:
    _set_blas_threads(int(job.get("blas_threads", BLAS_THREADS)))
    session = job["session"]
    seed_index = int(job["seed_index"])
    out_dir = _session_out_dir(session)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{session}_extend_d.log"
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
        _stash_source_summary(out_dir)
        archived = _archive_grid20(out_dir)
        _log(f"archive: {archived}")
        cfg = _overlay_cfg()
        _log(f"grid={cfg['latent_dims']} nested_fit_d={cfg['nested_fit_d']}")
        bundle = build_segment_bundle(session, room="A", data_root=data_root)
        streams = _streams(seed_index)
        # raw_lag anchor keeps train/eval intersection identical to full grid.
        summary = analyze_source(
            cfg, None, "real", streams, seed_index,
            bundle=bundle, output_dir=out_dir,
            method_keys=("raw_lag", "pca", "lds"),
            save_figure_contract=True,
        )
        _merge_figure_contract(out_dir)
        _merge_source_summary(out_dir, summary)
        report = _update_session_report(out_dir, session, seed_index)
        wall = time.perf_counter() - t0
        sel = {
            m["method"]: m.get("selected_d")
            for m in report["methods"]
            if m["method"] in ("pca", "lds", "pca_grid20", "lds_grid20")
        }
        _log(f"done wall_s={wall:.1f} selected_d={sel}")
        return {
            "ok": True, "session": session, "seed_index": seed_index,
            "wall_s": wall, "selected_d": sel, "git_sha": _git_sha(),
        }
    except Exception as exc:
        wall = time.perf_counter() - t0
        err = f"{type(exc).__name__}: {exc}"
        _log(f"FAILED {err}\n{traceback.format_exc()}")
        return {
            "ok": False, "session": session, "seed_index": seed_index,
            "wall_s": wall, "error": err, "git_sha": _git_sha(),
        }


def write_extend_d_report(cohort: dict[str, Any]) -> dict[str, Any]:
    """Selected d, error-vs-d medians, primary contrasts for grid_ext and grid20."""
    from collections import Counter

    from analysis.real_quadrant.run_m3 import (
        PRIMARY_CONTRASTS,
        SECONDARY_CONTRASTS,
        _contrast_pair,
        aggregate_m3,
    )

    selected_d_rows = []
    err_vs_d: dict[str, dict[int, list[float]]] = {
        "pca": {}, "lds": {}, "pca_grid20": {}, "lds_grid20": {},
    }
    for sel in cohort["selected_sessions"]:
        session = sel["session"]
        out_dir = _session_out_dir(session)
        report = json.loads((out_dir / "session_report.json").read_text())
        by_m = {m["method"]: m for m in report["methods"]}
        selected_d_rows.append({
            "animal": sel["animal"],
            "session": session,
            "n_units": report.get("n_units"),
            "pca": (by_m.get("pca") or {}).get("selected_d"),
            "lds": (by_m.get("lds") or {}).get("selected_d"),
            "pca_grid20": (by_m.get("pca_grid20") or {}).get("selected_d"),
            "lds_grid20": (by_m.get("lds_grid20") or {}).get("selected_d"),
            "pca_norm": (by_m.get("pca") or {}).get("normalized_error"),
            "lds_norm": (by_m.get("lds") or {}).get("normalized_error"),
            "pca_grid20_norm": (by_m.get("pca_grid20") or {}).get("normalized_error"),
            "lds_grid20_norm": (by_m.get("lds_grid20") or {}).get("normalized_error"),
        })
        # Extended d_sweep
        ds_path = out_dir / "d_sweep.json"
        if ds_path.is_file():
            floor = float(report["floor_median_cm"])
            for row in json.loads(ds_path.read_text()).get("rows") or []:
                m = row.get("method")
                if m not in ("pca", "lds"):
                    continue
                d = int(row["d"])
                # Prefer eval ridge_median; fall back to inner_cv
                err = row.get("ridge_median")
                if err is None:
                    err = row.get("inner_cv_ridge_median")
                if err is None or floor <= 0:
                    continue
                err_vs_d[m].setdefault(d, []).append(float(err) / floor)
        # grid20 d_sweep archive
        ds20 = out_dir / "d_sweep_grid20.json"
        if ds20.is_file():
            floor = float(report["floor_median_cm"])
            for row in json.loads(ds20.read_text()).get("rows") or []:
                m = row.get("method")
                if m not in ("pca", "lds"):
                    continue
                d = int(row["d"])
                err = row.get("ridge_median")
                if err is None:
                    err = row.get("inner_cv_ridge_median")
                if err is None or floor <= 0:
                    continue
                err_vs_d[f"{m}_grid20"].setdefault(d, []).append(float(err) / floor)

    err_vs_d_median = {
        m: {
            str(d): float(np.median(vals))
            for d, vals in sorted(by_d.items())
        }
        for m, by_d in err_vs_d.items()
    }

    # Aggregate with live method names (= grid_ext for pca/lds)
    agg_ext = aggregate_m3(cohort, include_gpfa_causal=True)

    # Remap session reports temporarily for grid20 contrasts: pca←pca_grid20 etc.
    # Build animal means from session reports with mapped names.
    animal_means_20: dict[str, dict[str, list[float]]] = {}
    animal_means_ext: dict[str, dict[str, list[float]]] = {}
    for sel in cohort["selected_sessions"]:
        report = json.loads(
            (_session_out_dir(sel["session"]) / "session_report.json").read_text()
        )
        by_m = {m["method"]: m for m in report["methods"]}
        animal = sel["animal"]
        animal_means_20.setdefault(animal, {})
        animal_means_ext.setdefault(animal, {})

        def _add(store, name, key):
            rec = by_m.get(key)
            if not rec or rec.get("status") not in (None, "ok"):
                return
            ne = rec.get("normalized_error")
            if ne is None or not np.isfinite(ne):
                return
            store[animal].setdefault(name, []).append(float(ne))

        for name in (
            "raw", "raw_smooth", "raw_lag", "dm", "dm_smooth", "isomap",
            "gpfa", "gpfa_causal",
        ):
            _add(animal_means_20, name, name)
            _add(animal_means_ext, name, name)
        _add(animal_means_ext, "pca", "pca")
        _add(animal_means_ext, "lds", "lds")
        _add(animal_means_ext, "pca_smooth", "pca_smooth")
        _add(animal_means_ext, "lds_smooth", "lds_smooth")
        _add(animal_means_20, "pca", "pca_grid20")
        _add(animal_means_20, "lds", "lds_grid20")
        _add(animal_means_20, "pca_smooth", "pca_smooth_grid20")
        _add(animal_means_20, "lds_smooth", "lds_smooth_grid20")

    def _means(store):
        return {
            animal: {m: float(np.mean(v)) for m, v in methods.items()}
            for animal, methods in store.items()
        }

    means_20 = _means(animal_means_20)
    means_ext = _means(animal_means_ext)
    primary = list(PRIMARY_CONTRASTS)
    secondary = list(SECONDARY_CONTRASTS)

    out = {
        "n_sessions": len(selected_d_rows),
        "grid_ext": list(GRID_EXT),
        "grid20": list(GRID20),
        "selected_d_per_session": selected_d_rows,
        "selected_d_mode": {
            "pca": Counter(r["pca"] for r in selected_d_rows).most_common(1),
            "lds": Counter(r["lds"] for r in selected_d_rows).most_common(1),
            "pca_grid20": Counter(r["pca_grid20"] for r in selected_d_rows).most_common(1),
            "lds_grid20": Counter(r["lds_grid20"] for r in selected_d_rows).most_common(1),
        },
        "n_at_grid_max": {
            "pca": sum(1 for r in selected_d_rows if r["pca"] == max(GRID_EXT)),
            "lds": sum(1 for r in selected_d_rows if r["lds"] == max(GRID_EXT)),
        },
        "error_vs_d_median_normalized": err_vs_d_median,
        "primary_contrasts_grid_ext": [
            _contrast_pair(a, b, means_ext) for a, b in primary
        ],
        "primary_contrasts_grid20": [
            _contrast_pair(a, b, means_20) for a, b in primary
        ],
        "secondary_contrasts_grid_ext": [
            _contrast_pair(a, b, means_ext) for a, b in secondary
        ],
        "secondary_contrasts_grid20": [
            _contrast_pair(a, b, means_20) for a, b in secondary
        ],
        "m3_aggregate_grid_ext_path": str(OUT_ROOT / "m3_aggregate.json"),
        "note": (
            "pca/lds/pca_smooth/lds_smooth in primary_contrasts_grid_ext use "
            "extended-grid rows; *_grid20 contrasts remap those four names to "
            "archived grid20 rows. Other methods unchanged."
        ),
    }
    # Refresh m3_aggregate.json to grid_ext naming
    _ = agg_ext
    (OUT_ROOT / "extend_d_report.json").write_text(
        json.dumps(out, indent=2, default=str) + "\n"
    )
    return out


def _launch_parallel(jobs: list[dict[str, Any]], max_workers: int) -> list[dict[str, Any]]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pending = list(jobs)
    live: dict[subprocess.Popen, tuple[dict[str, Any], Path]] = {}
    results: list[dict[str, Any]] = []
    n_workers = max(1, min(int(max_workers), len(jobs) or 1))
    py = sys.executable

    def _start(job: dict[str, Any]) -> None:
        job_path = LOG_DIR / f"job_extend_{job['session']}.json"
        result_path = LOG_DIR / f"result_extend_{job['session']}.json"
        if result_path.is_file():
            result_path.unlink()
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        env = os.environ.copy()
        n_thr = str(int(job.get("blas_threads", BLAS_THREADS)))
        for var in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        ):
            env[var] = n_thr
        log_path = LOG_DIR / f"{job['session']}_extend_d.log"
        log_f = log_path.open("a")
        proc = subprocess.Popen(
            [py, "-m", "analysis.real_quadrant.run_m3_extend_d",
             "--worker-job", str(job_path)],
            cwd=str(REPO_ROOT), env=env,
            stdout=log_f, stderr=subprocess.STDOUT,
        )
        live[proc] = (job, result_path)
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
                    "ok": False, "session": job["session"],
                    "error": f"worker exit={proc.returncode}; no result JSON",
                })
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--worker-job", type=str, default=None)
    parser.add_argument(
        "--rebuild-reports-only",
        action="store_true",
        help="Rebuild session_report.json for all cohort sessions; no refits.",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Write extend_d_report.json (selected d, error-vs-d, contrasts).",
    )
    args = parser.parse_args(argv)

    if args.worker_job:
        job = json.loads(Path(args.worker_job).read_text())
        _set_blas_threads(int(job.get("blas_threads", BLAS_THREADS)))
        result = _run_one_session(job)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"result_extend_{job['session']}.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        return 0 if result.get("ok") else 1

    if args.rebuild_reports_only or args.report_only:
        cohort = json.loads((OUT_ROOT / "cohort_manifest.json").read_text())
        if args.rebuild_reports_only:
            for i, sel in enumerate(cohort["selected_sessions"]):
                out_dir = _session_out_dir(sel["session"])
                print(f"rebuild {sel['session']} …", flush=True)
                _update_session_report(out_dir, sel["session"], i)
        if args.report_only or args.rebuild_reports_only:
            report = write_extend_d_report(cohort)
            print(
                f"wrote {OUT_ROOT / 'extend_d_report.json'} "
                f"n_sessions={report['n_sessions']}",
                flush=True,
            )
        return 0

    require_clean_git_for_real_data()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cohort = json.loads((OUT_ROOT / "cohort_manifest.json").read_text())
    selected = list(cohort["selected_sessions"])
    # Preserve original seed_index order from cohort
    jobs = []
    for i, sel in enumerate(selected):
        jobs.append({
            "session": sel["session"],
            "animal": sel["animal"],
            "seed_index": i,
            "blas_threads": BLAS_THREADS,
        })
    print(
        f"extend-d: {len(jobs)} sessions, methods={METHODS}, grid={list(GRID_EXT)}",
        flush=True,
    )
    t0 = time.perf_counter()
    results = _launch_parallel(jobs, args.max_workers)
    wall = time.perf_counter() - t0
    (OUT_ROOT / "extend_d_results.json").write_text(
        json.dumps({"wall_s": wall, "grid": list(GRID_EXT), "results": results},
                   indent=2, default=str) + "\n"
    )
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"finished {n_ok}/{len(results)} in {wall:.1f}s", flush=True)
    write_extend_d_report(cohort)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
