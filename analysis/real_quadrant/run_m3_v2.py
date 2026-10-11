"""Report 2 v2 M3 runner: HF-only units, latent cache, d ≤ ⌊n_HF/2⌋.

Outputs under ``outputs/real_quadrant/m3_v2/`` (gitignored). Never writes
v1 ``m3/`` or ``hippo-dq/``. Cohort list is loaded from the Phase 0 local
JSON (not re-selected with the all-region loader).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from analysis.real_quadrant.adapter import (
    HF_REGION_ALLOWLIST,
    build_segment_bundle,
)
from analysis.real_quadrant import latent_cache as lcache
from analysis.real_quadrant.run_m3 import (
    BLAS_THREADS,
    M3_BASE_METHODS,
    MAX_WORKERS,
    _apply_session_posthoc,
    _set_blas_threads,
    _streams,
    aggregate_m3,
    retarget_method_resume_keys,
    run_methods_fault_tolerant,
)
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import (
    _git_sha,
    analyze_source,
    require_clean_git_for_real_data,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "real_quadrant" / "m3_v2"
LOG_DIR = OUT_ROOT / "logs"
PHASE0_COHORT = (
    REPO_ROOT / "outputs" / "real_quadrant" / "phase0" / "phase0_result.json"
)


def _session_out_dir(session: str) -> Path:
    return OUT_ROOT / "room_A" / session


def load_v2_cohort(*, include_bridge: bool = False) -> dict[str, Any]:
    """HF cohort (+ optional v1-bridge-only sessions) from Phase 0 JSON."""
    if not PHASE0_COHORT.is_file():
        raise FileNotFoundError(
            f"missing {PHASE0_COHORT}; run Phase 0 eligibility first"
        )
    raw = json.loads(PHASE0_COHORT.read_text())
    selected = list(raw["hf_cohort"]["selected_sessions"])
    if include_bridge:
        have = {s["session"] for s in selected}
        for b in raw.get("bridge_v1_sessions") or []:
            if b["session"] not in have:
                selected.append({
                    "animal": b["animal"],
                    "session": b["session"],
                    "n_units": b["n_hf"],
                    "bridge": True,
                })
    return {
        "rule": (
            "HF-only M3 median rule (Phase 0); units Region∈"
            f"{sorted(HF_REGION_ALLOWLIST)}; RegionConfidence ignored"
        ),
        "unit_definition": raw["hf_cohort"]["unit_definition"],
        "n_selected_sessions": len(selected),
        "selected_sessions": selected,
        "animals": raw["hf_cohort"]["animals"],
        "phase0_diff": raw.get("diff_v1_v2"),
    }


def _cap_cfg_for_units(cfg: dict[str, Any], n_units: int) -> dict[str, Any]:
    dims, nested = lcache.cap_latent_dims(list(cfg["latent_dims"]), n_units)
    out = dict(cfg)
    out["latent_dims"] = dims
    out["nested_fit_d"] = nested
    return out


def _save_gpfa_causal_cache(
    out_dir: Path,
    session: str,
    seed_index: int,
    report: dict[str, Any],
) -> None:
    """Persist gpfa_causal final Z into latent_cache after posthoc."""
    from realtime.quadrant_n5_run import _sqrt_zscore_train
    from realtime.dynamic_latents.gpfa import GPFAModel
    from realtime.quadrant_n5_run import OUTPUT_ROOT

    contract = out_dir / "figure_contract"
    if not (contract / "arrays.npz").is_file():
        return
    if not (out_dir / "gpfa.json").is_file():
        return
    model_dir = OUTPUT_ROOT / "models" / f"seed_{seed_index}" / "real" / "gpfa"
    if not (model_dir / "gpfa_params.npz").is_file():
        return
    try:
        model = GPFAModel.load(model_dir)
        arrays = np.load(contract / "arrays.npz")
        train_ok = np.asarray(arrays["train_ok"], dtype=bool)
        X_counts = np.asarray(arrays["X_counts"], dtype=float)
        X, _ = _sqrt_zscore_train(X_counts, train_ok)
        Z = model.transform(X, causal=True, reset=True)
        gpfa_rec = json.loads((out_dir / "gpfa.json").read_text())
        primary_d = gpfa_rec.get("primary_d")
        if primary_d is not None and Z.shape[1] > int(primary_d):
            Z = Z[:, : int(primary_d)]
        root = lcache.cache_root(out_dir)
        lcache.save_final_latent(
            root, "gpfa_causal",
            Z=Z,
            primary_d=primary_d,
            ridge_alpha=float(gpfa_rec.get("ridge_alpha") or 0.0),
            extra={"session": session, "source": "gpfa_causal_filter"},
        )
        # Also merge into figure_contract latents.npz for convenience.
        lat_path = contract / "latents.npz"
        payload = dict(np.load(lat_path)) if lat_path.is_file() else {}
        payload["Z_gpfa_causal"] = np.asarray(Z, dtype=float)
        np.savez_compressed(lat_path, **payload)
    except Exception:
        return


def _run_one_session(job: dict[str, Any]) -> dict[str, Any]:
    _set_blas_threads(int(job.get("blas_threads", BLAS_THREADS)))
    session = job["session"]
    seed_index = int(job["seed_index"])
    methods = tuple(job["methods"])
    animal = job.get("animal")
    out_dir = _session_out_dir(session)
    out_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{session}.log"

    def _log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
        with log_path.open("a") as f:
            f.write(line)
        print(f"[{session}] {msg}", flush=True)

    t0 = time.perf_counter()
    try:
        require_clean_git_for_real_data()
        data_root = Path(os.environ["HIPPO_DATA_ROOT"])
        _log(
            f"start v2 HF methods={methods} seed_index={seed_index} "
            f"regions={sorted(HF_REGION_ALLOWLIST)}"
        )
        bundle = build_segment_bundle(
            session,
            room="A",
            data_root=data_root,
            regions=HF_REGION_ALLOWLIST,
        )
        n_units = int(len(bundle["unit_ids"]))
        cfg = dict(load_quadrant_n5_yaml())
        cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
        cfg = _cap_cfg_for_units(cfg, n_units)
        _log(
            f"n_HF={n_units} latent_dims={cfg['latent_dims']} "
            f"nested_fit_d={cfg['nested_fit_d']} "
            f"(cap ⌊n/2⌋={n_units // 2})"
        )
        streams = _streams(seed_index)
        # Patch module OUT_ROOT paths used by resume helpers via job out_dir.
        retargeted = retarget_method_resume_keys(
            out_dir, methods, cfg_sha=cfg.get("config_sha256"),
        )
        if retargeted:
            _log(f"retargeted resume for {retargeted}")
        succeeded, failures, summary = run_methods_fault_tolerant(
            cfg=cfg,
            bundle=bundle,
            streams=streams,
            seed_index=seed_index,
            out_dir=out_dir,
            methods=methods,
            analyze_fn=analyze_source,
            log=_log,
            save_latent_cache=True,
        )
        if not succeeded:
            raise RuntimeError(
                f"all methods failed: {[f.get('error') for f in failures.values()]}"
            )
        report = None
        if job.get("posthoc", True):
            # _apply_session_posthoc reads OUT_ROOT from run_m3 — temporarily
            # monkeypatch via writing into the session dir only (it uses out_dir).
            import analysis.real_quadrant.run_m3 as m3mod
            old_out = m3mod.OUT_ROOT
            m3mod.OUT_ROOT = OUT_ROOT
            try:
                report = _apply_session_posthoc(
                    out_dir,
                    seed_index=seed_index,
                    session=session,
                    include_gpfa_causal=bool(job.get("include_gpfa_causal", True)),
                    failures=failures,
                    git_sha_run=_git_sha(),
                )
            finally:
                m3mod.OUT_ROOT = old_out
            _save_gpfa_causal_cache(out_dir, session, seed_index, report or {})
        wall = time.perf_counter() - t0
        _log(
            f"done wall_s={wall:.1f} succeeded={succeeded} "
            f"failed={list(failures)} latent_cache={lcache.cache_root(out_dir)}"
        )
        print(f"PROGRESS ok session={session} wall_s={wall:.1f}", flush=True)
        return {
            "ok": True,
            "session": session,
            "animal": animal,
            "seed_index": seed_index,
            "wall_s": wall,
            "n_units": n_units,
            "latent_dims": list(cfg["latent_dims"]),
            "methods": list(succeeded),
            "failed_methods": failures,
            "report_path": str(out_dir / "session_report.json") if report else None,
            "git_sha": _git_sha(),
            "hf_only": True,
        }
    except Exception as exc:
        wall = time.perf_counter() - t0
        err = f"{type(exc).__name__}: {exc}"
        _log(f"FAILED {err}\n{traceback.format_exc()}")
        print(f"PROGRESS fail session={session} wall_s={wall:.1f}", flush=True)
        return {
            "ok": False,
            "session": session,
            "animal": animal,
            "seed_index": seed_index,
            "wall_s": wall,
            "error": err,
            "git_sha": _git_sha(),
            "hf_only": True,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="One cohort session, all methods, position only.")
    parser.add_argument("--session", type=str, default=None,
                        help="Run a single named session (overrides --smoke pick).")
    parser.add_argument("--bridge", action="store_true",
                        help="Include v1-bridge sessions not in the HF cohort.")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--worker-job", type=str, default=None)
    args = parser.parse_args(argv)

    # Ensure workers use this module's OUT_ROOT / _run_one_session.
    import analysis.real_quadrant.run_m3 as m3mod
    m3mod.OUT_ROOT = OUT_ROOT
    m3mod.LOG_DIR = LOG_DIR
    m3mod._run_one_session = _run_one_session  # type: ignore[assignment]
    m3mod._session_out_dir = _session_out_dir  # type: ignore[assignment]

    if args.worker_job:
        job = json.loads(Path(args.worker_job).read_text())
        _set_blas_threads(int(job.get("blas_threads", BLAS_THREADS)))
        result = _run_one_session(job)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / f"result_{job['session']}.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n"
        )
        return 0 if result.get("ok") else 1

    require_clean_git_for_real_data()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"OUT_ROOT={OUT_ROOT.resolve()}", flush=True)

    cohort = load_v2_cohort(include_bridge=bool(args.bridge))
    cohort_path = OUT_ROOT / "cohort_manifest.json"
    if not args.aggregate_only:
        cohort_path.write_text(json.dumps(cohort, indent=2, default=str) + "\n")

    if args.aggregate_only:
        cohort = json.loads(cohort_path.read_text())
        agg = aggregate_m3(cohort)
        print(json.dumps({"n_animals": agg["n_animals"]}, indent=2), flush=True)
        return 0

    selected = list(cohort["selected_sessions"])
    if args.session:
        selected = [s for s in selected if s["session"] == args.session]
        if not selected:
            # allow explicit session not in cohort list (e.g. bridge)
            selected = [{
                "session": args.session,
                "animal": args.session.split("_", 1)[0],
                "n_units": None,
            }]
    elif args.smoke:
        selected = selected[:1]

    methods = M3_BASE_METHODS
    jobs = []
    for i, sel in enumerate(selected):
        jobs.append({
            "session": sel["session"],
            "animal": sel.get("animal"),
            "seed_index": i,
            "methods": list(methods),
            "posthoc": True,
            "include_gpfa_causal": True,
            "blas_threads": BLAS_THREADS,
        })

    # Patch _launch_parallel worker module path to this file.
    def _launch_v2(jobs_in, max_workers):
        import subprocess
        pending = list(jobs_in)
        live: dict[Any, tuple[dict[str, Any], Path]] = {}
        results: list[dict[str, Any]] = []
        n_workers = max(1, min(int(max_workers), len(jobs_in) or 1))
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
                "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            ):
                env[var] = n_thr
            log_path = LOG_DIR / f"{job['session']}.log"
            log_f = log_path.open("a")
            proc = subprocess.Popen(
                [py, "-m", "analysis.real_quadrant.run_m3_v2",
                 "--worker-job", str(job_path)],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
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
                        "ok": False,
                        "session": job["session"],
                        "error": f"worker exit={proc.returncode}; no result JSON",
                    })
                print(
                    f"PROGRESS finished {len(results)}/{len(jobs_in)} "
                    f"last={job['session']} ok={results[-1].get('ok')}",
                    flush=True,
                )
        return results

    t0 = time.perf_counter()
    tag = "smoke" if args.smoke or args.session else "phaseA"
    results = _launch_v2(jobs, args.max_workers)
    wall = time.perf_counter() - t0
    (OUT_ROOT / f"{tag}_results.json").write_text(
        json.dumps({"wall_s": wall, "results": results}, indent=2, default=str) + "\n"
    )
    n_ok = sum(1 for r in results if r.get("ok"))
    print(f"finished {n_ok}/{len(results)} sessions in {wall:.1f}s", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
