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
    row = {
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
    return row


def _update_session_report(out_dir: Path, session: str, seed_index: int) -> dict[str, Any]:
    """Rebuild session_report keeping grid20 pca/lds and new extended rows + smooth."""
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
            "grid_label": "grid20",
        }

    # grid20 archived pca/lds
    for m in METHODS:
        p = out_dir / f"{m}_grid20.json"
        if p.is_file():
            rec = json.loads(p.read_text())
            if "ridge" in rec:
                row = _row_from_method_json(rec, floor_med, label="grid20")
                method_rows[row["method"]] = row

    # new extended pca/lds
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

        # smooth from new latents
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

    # Lag: train-based peak + eval plateau (no eval point estimate as primary)
    for name, row in list(method_rows.items()):
        if row.get("status") not in (None, "ok"):
            continue
        key = f"pred_{name}_ridge"
        if key not in preds:
            # try without _grid20 suffix mapping
            continue
        # Eval plateau under behind-truth convention
        lag = _lag_behind_truth(np.asarray(preds[key]), y_true)
        # Plateau: lags within 0.01 of peak corr
        from analysis.real_quadrant.posthoc_smooth import UPDATE_DT_S, LAG_MAX_S
        pred = np.asarray(preds[key], float)
        finite = np.isfinite(pred).all(1) & np.isfinite(y_true).all(1)
        p, yt = pred[finite], y_true[finite]
        max_steps = int(round(LAG_MAX_S / UPDATE_DT_S))
        lags_c = []
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
                # behind convention = -step
                lags_c.append((-step * UPDATE_DT_S, float(np.mean(cs))))
        if lags_c:
            arr = np.asarray(lags_c)
            i = int(np.argmax(arr[:, 1]))
            peak_c = float(arr[i, 1])
            near = arr[:, 1] >= (peak_c - 0.01)
            row["eval_lag_plateau_s"] = [
                float(arr[near, 0].min()), float(arr[near, 0].max()),
            ]
            row["eval_lag_peak_corr"] = peak_c
        row["effective_lag_s"] = None  # drop eval point estimate as primary
        row["lag_sign_convention"] = "positive_means_prediction_behind_truth"

    # Train-based lag from raw (shared annotation)
    train_lag_note = {
        "rule": "train position XC peak; positive = prediction behind truth",
        "see": "outputs/real_quadrant/m3/sync_check.json and pre-M4 sync report",
    }

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
        "train_lag_note": train_lag_note,
        "effective_lag_search": {
            "sign_convention": "positive_means_prediction_behind_truth",
            "eval_primary": "plateau_range",
            "eval_point_estimate": "dropped",
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
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
