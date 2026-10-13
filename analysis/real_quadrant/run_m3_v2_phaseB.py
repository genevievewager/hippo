"""Phase B: secondary-target decode on cached Phase A latents (Report 2 v2).

Reads Phase A latents **read-only** from the hippo-r2v2 m3_v2 tree and writes
results under ``m3_v2/phaseB/<session>/`` only. Never writes into
``room_A/``, ``hippo-dq/``, or v1 ``m3/``.

Do not launch until extend-d finishes and this module has been reviewed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.real_quadrant.adapter import (
    HF_REGION_ALLOWLIST,
    _room_boundary,
    _room_polygon_global,
)
from analysis.real_quadrant import latent_cache as lcache
from analysis.real_quadrant.posthoc_smooth import EMA_TAUS_S
from analysis.real_quadrant.secondary_decode import (
    beyond_position_control,
    beyond_speed_control,
    decode_target_from_cache,
    prepare_categorical_target,
    select_hparams_from_fold_cache,
)
from analysis.real_quadrant.secondary_targets import build_secondary_target_table
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import _git_sha, require_clean_git_for_real_data

# Code may live in a worktree; Phase A artifacts stay under hippo-r2v2.
PHASE_A_ROOT = Path(
    os.environ.get(
        "HIPPO_R2V2_M3_ROOT",
        "/home/gmwager/projects/hippo-r2v2/outputs/real_quadrant/m3_v2",
    )
)
PHASE0_COHORT = Path(
    os.environ.get(
        "HIPPO_R2V2_PHASE0",
        "/home/gmwager/projects/hippo-r2v2/outputs/real_quadrant/phase0/phase0_result.json",
    )
)
# Phase B writes here only (sibling of room_A, not inside it).
PHASE_B_ROOT = PHASE_A_ROOT / "phaseB"


def _phase_a_session_dir(session: str, *, bridge: bool = False) -> Path:
    if bridge:
        return PHASE_A_ROOT / "bridge" / "room_A" / session
    return PHASE_A_ROOT / "room_A" / session


def _phase_b_out(session: str) -> Path:
    return PHASE_B_ROOT / session


def _load_v2_cohort(*, include_bridge: bool = False) -> dict[str, Any]:
    """Load cohort from Phase 0 JSON (absolute path; worktree-safe)."""
    from analysis.real_quadrant.adapter import HF_REGION_ALLOWLIST as _HF

    if not PHASE0_COHORT.is_file():
        raise FileNotFoundError(f"missing {PHASE0_COHORT}")
    raw = json.loads(PHASE0_COHORT.read_text())
    selected = list(raw["hf_cohort"]["selected_sessions"])
    if include_bridge:
        have = {s["session"] for s in selected}
        v1_only = set((raw.get("diff_v1_v2") or {}).get("v1_only") or [])
        by_sess = {
            b["session"]: b for b in (raw.get("bridge_v1_sessions") or [])
        }
        for sess in sorted(v1_only):
            if sess in have:
                continue
            row = by_sess.get(sess) or {}
            selected.append({
                "animal": row.get("animal", sess.split("_", 1)[0]),
                "session": sess,
                "n_units": row.get("n_hf"),
                "bridge": True,
            })
            have.add(sess)
    return {
        "rule": f"HF-only; regions={sorted(_HF)}",
        "selected_sessions": selected,
        "n_selected_sessions": len(selected),
    }


def _load_align_extras(session: str, decode_times: np.ndarray) -> dict[str, Any]:
    """Load HD / V from postions_dataset aligned to decode timestamps."""
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    pos = pd.read_csv(
        root / session / "postions_dataset.csv",
        usecols=["timestamp", "X", "Y", "V", "HD", "valid", "room"],
    )
    pos_t = pos["timestamp"].to_numpy(dtype=float)
    idx = np.searchsorted(pos_t, decode_times, side="left")
    idx = np.clip(idx, 0, len(pos_t) - 1)
    prev = np.clip(idx - 1, 0, len(pos_t) - 1)
    use_prev = np.abs(pos_t[prev] - decode_times) < np.abs(pos_t[idx] - decode_times)
    idx = np.where(use_prev, prev, idx)
    return {
        "hd_deg": pos["HD"].to_numpy(dtype=float)[idx],
        "tracker_v": pos["V"].to_numpy(dtype=float)[idx],
        "x_global": pos["X"].to_numpy(dtype=float)[idx],
        "y_global": pos["Y"].to_numpy(dtype=float)[idx],
    }


def _polygon_local(session: str) -> np.ndarray:
    import yaml

    root = Path(os.environ["HIPPO_DATA_ROOT"])
    with open(root / session / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    poly = _room_polygon_global(cfg, "A")
    cx, cy, *_ = _room_boundary(cfg, "A")
    out = poly.astype(float).copy()
    out[:, 0] -= cx
    out[:, 1] -= cy
    return out


def _slim_ema(block: dict[str, Any] | None) -> dict[str, Any] | None:
    if not block:
        return None
    return {
        k: v for k, v in block.items()
        if k not in ("pred_full", "pred_eval")
    }


def run_session_phase_b(
    session: str,
    *,
    methods: list[str] | None = None,
    bridge: bool = False,
) -> dict[str, Any]:
    phase_a_dir = _phase_a_session_dir(session, bridge=bridge)
    cache_dir = lcache.cache_root(phase_a_dir)
    contract = phase_a_dir / "figure_contract"
    arrays_path = contract / "arrays.npz"
    if not arrays_path.is_file():
        raise FileNotFoundError(f"missing {arrays_path}; run Phase A first")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"missing {cache_dir}; run Phase A with cache")

    # Read-only: load into memory; never open Phase A paths for write.
    arrays = np.load(arrays_path)  # numpy memmap/read
    train_ok = np.asarray(arrays["train_ok"], dtype=bool)
    if "eval_mask" in arrays.files:
        eval_mask = np.asarray(arrays["eval_mask"], dtype=bool)
    elif "test_ok" in arrays.files:
        eval_mask = np.asarray(arrays["test_ok"], dtype=bool)
    else:
        y_xy = np.asarray(arrays["y"], dtype=float)
        eval_mask = (~train_ok) & np.isfinite(y_xy).all(1)

    y_xy = np.asarray(arrays["y"], dtype=float)
    decode_times = np.asarray(arrays["decode_times"], dtype=float)
    target_valid = (
        np.asarray(arrays["target_valid"], dtype=bool)
        if "target_valid" in arrays.files
        else np.isfinite(y_xy).all(1)
    )

    extras = _load_align_extras(session, decode_times)
    poly = _polygon_local(session)
    targets = build_secondary_target_table(
        x=y_xy[:, 0],
        y=y_xy[:, 1],
        times=decode_times,
        polygon_xy_local=poly,
        hd_deg=extras["hd_deg"],
        tracker_v=extras["tracker_v"],
        valid=target_valid,
    )

    from realtime.quadrant_n5_run import ridge_alpha_grid

    cfg = load_quadrant_n5_yaml()
    meta = lcache.read_meta(cache_dir) or {}
    latent_dims = list(meta.get("latent_dims") or cfg["latent_dims"])
    nested_fit_d = meta.get("nested_fit_d", cfg.get("nested_fit_d"))
    alphas = [float(a) for a in ridge_alpha_grid(cfg)]
    ks = [int(k) for k in cfg["decoders"]["knn_k"]]
    a13cfg = (cfg.get("phase3") or {}).get("a13") or {}
    n_shifts = int(a13cfg.get("n_shifts", 20))
    lo = float(a13cfg.get("shift_frac_lo", 0.25))
    hi = float(a13cfg.get("shift_frac_hi", 0.75))
    shift_fracs = list(np.linspace(lo, hi, n_shifts))
    pass_min = float(a13cfg.get("pass_min_cm", 0.0))

    final_dir = cache_dir / "final"
    available = sorted(
        p.stem for p in final_dir.glob("*.npz") if not p.name.endswith(".meta.npz")
    )
    # drop .meta confusion — stems are method names
    available = [m for m in available if not m.endswith(".meta")]
    if methods is None:
        methods = available
    else:
        methods = [m for m in methods if m in available]

    wall_bin, wall_meta = prepare_categorical_target(
        targets["wall_distance_bin"], train_ok, eval_mask,
    )
    move_st, move_meta = prepare_categorical_target(
        targets["movement_state"], train_ok, eval_mask,
    )

    target_specs = [
        ("distance_to_wall", targets["distance_to_wall"], "continuous"),
        ("speed", targets["speed"], "continuous"),
        ("acceleration", targets["acceleration"], "continuous"),
        ("head_direction", targets["head_direction_sincos"], "circular"),
        ("wall_distance_bin", wall_bin, "categorical"),
        ("movement_state", move_st, "categorical"),
        ("pc_v", targets["pc_v"], "continuous"),
    ]

    phase_b_dir = _phase_b_out(session)
    phase_b_dir.mkdir(parents=True, exist_ok=True)
    if phase_b_dir.resolve() != (PHASE_B_ROOT / session).resolve():
        raise RuntimeError(f"refusing unexpected Phase B path {phase_b_dir}")
    if "room_A" in phase_b_dir.resolve().parts:
        raise RuntimeError("Phase B must not write under room_A")

    phase_a_sha = (
        (meta or {}).get("git_sha")
        or json.loads((phase_a_dir / "source_summary.json").read_text()).get("git_sha")
    )
    results: dict[str, Any] = {
        "session": session,
        "bridge": bool(bridge),
        "git_sha": _git_sha(),
        "git_sha_phaseB": _git_sha(),
        "git_sha_phaseA_latents": phase_a_sha,
        "phase_a_dir": str(phase_a_dir.resolve()),
        "phase_b_dir": str(phase_b_dir.resolve()),
        "latent_cache_meta": {
            "git_sha": (meta or {}).get("git_sha"),
            "config_sha256": (meta or {}).get("config_sha256"),
            "n_units": (meta or {}).get("n_units"),
            "latent_dims": (meta or {}).get("latent_dims"),
            "extend_d": (meta or {}).get("extend_d"),
        },
        "ema_taus_s": list(EMA_TAUS_S),
        "hf_regions": sorted(HF_REGION_ALLOWLIST),
        "methods": methods,
        "class_merge": {
            "wall_distance_bin": wall_meta,
            "movement_state": move_meta,
        },
        "by_method": {},
    }

    pred_xy_by_method: dict[str, np.ndarray] = {}
    pred_path = contract / "predictions.npz"
    if pred_path.is_file():
        preds = np.load(pred_path)
        for m in methods:
            for key in (f"ridge_{m}", f"yhat_ridge_{m}", m):
                if key in preds.files:
                    arr = np.asarray(preds[key], dtype=float)
                    if arr.ndim == 2 and arr.shape[1] >= 2:
                        pred_xy_by_method[m] = arr[:, :2]
                    break

    for method in methods:
        method_out: dict[str, Any] = {}
        for tname, y_t, kind in target_specs:
            if y_t is None:
                method_out[tname] = {"skipped": True, "reason": "missing array"}
                continue
            if tname == "wall_distance_bin" and not wall_meta.get("evaluable"):
                method_out[tname] = {"skipped": True, "reason": wall_meta}
                continue
            if tname == "movement_state" and not move_meta.get("evaluable"):
                method_out[tname] = {"skipped": True, "reason": move_meta}
                continue
            if tname == "head_direction":
                y_use = y_t.copy()
                y_use[~targets["head_direction_valid"]] = np.nan
            else:
                y_use = y_t

            try:
                hp = select_hparams_from_fold_cache(
                    cache_dir, method, y_use, train_ok,
                    latent_dims=latent_dims,
                    nested_fit_d=(
                        None if method in ("raw", "raw_lag") else nested_fit_d
                    ),
                    alphas=alphas,
                    ks=ks,
                    kind=kind,
                )
                dec = decode_target_from_cache(
                    cache_dir, method, y_use, train_ok, eval_mask,
                    kind=kind, hparams=hp,
                    times=decode_times,
                    shift_fracs=shift_fracs,
                    pass_min=pass_min,
                    apply_ema=True,
                    ema_taus_s=EMA_TAUS_S,
                )
                slim = {
                    k: v for k, v in dec.items()
                    if k not in ("pred_ridge", "pred_knn", "eval_index")
                }
                for ek in ("ridge_ema", "knn_ema"):
                    if ek in slim:
                        slim[ek] = _slim_ema(slim[ek])
                if tname == "distance_to_wall" and method in pred_xy_by_method:
                    slim["beyond_position"] = beyond_position_control(
                        pred_xy_by_method[method],
                        targets["distance_to_wall"],
                        wall_bin,
                        poly,
                        dec["eval_index"],
                    )
                if tname == "speed":
                    method_out["_speed_pred_ridge"] = dec["pred_ridge"]
                    method_out["_speed_eval_index"] = dec["eval_index"]
                if tname == "movement_state" and "_speed_pred_ridge" in method_out:
                    slim["beyond_speed"] = beyond_speed_control(
                        method_out["_speed_pred_ridge"],
                        move_st,
                        method_out["_speed_eval_index"],
                    )
                if tname == "pc_v":
                    slim["sensitivity_only"] = True
                    slim["note"] = "tracker V; does not drive accel/movement_state"
                method_out[tname] = slim
            except Exception as exc:
                method_out[tname] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        method_out.pop("_speed_pred_ridge", None)
        method_out.pop("_speed_eval_index", None)
        results["by_method"][method] = method_out

    (phase_b_dir / "phaseB_results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    np.savez_compressed(
        phase_b_dir / "secondary_targets.npz",
        distance_to_wall=targets["distance_to_wall"],
        wall_distance_bin=wall_bin,
        speed=targets["speed"],
        acceleration=targets["acceleration"],
        movement_state=move_st,
        head_direction_sincos=targets["head_direction_sincos"],
        pc_v=targets["pc_v"] if targets["pc_v"] is not None else np.array([]),
        valid=targets["valid"],
        decode_times=decode_times,
    )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="First cohort session only.")
    parser.add_argument("--bridge", action="store_true",
                        help="Include v1-only bridge sessions.")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args(argv)

    require_clean_git_for_real_data()
    PHASE_B_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"PHASE_A_ROOT={PHASE_A_ROOT.resolve()}", flush=True)
    print(f"PHASE_B_ROOT={PHASE_B_ROOT.resolve()}", flush=True)
    print(f"EMA_TAUS_S={list(EMA_TAUS_S)}", flush=True)

    cohort = _load_v2_cohort(include_bridge=bool(args.bridge))
    selected = list(cohort["selected_sessions"])
    if args.session:
        selected = [s for s in selected if s["session"] == args.session]
        if not selected:
            selected = [{
                "session": args.session,
                "animal": args.session.split("_", 1)[0],
                "bridge": False,
            }]
    elif args.smoke:
        selected = selected[:1]

    t0 = time.perf_counter()
    results = []
    for sel in selected:
        session = sel["session"]
        bridge = bool(sel.get("bridge", False))
        print(f"Phase B start {session} bridge={bridge}", flush=True)
        try:
            rec = run_session_phase_b(session, bridge=bridge)
            ok = True
            err = None
        except Exception as exc:
            ok = False
            err = f"{type(exc).__name__}: {exc}"
            rec = {"session": session, "error": err}
        wall = time.perf_counter() - t0
        print(
            f"PROGRESS phaseB session={session} ok={ok} wall_s={wall:.1f}",
            flush=True,
        )
        results.append({
            "ok": ok,
            "session": session,
            "bridge": bridge,
            "error": err,
            "git_sha_phaseB": _git_sha(),
            "git_sha_phaseA_latents": (rec or {}).get("git_sha_phaseA_latents"),
        })

    out = PHASE_B_ROOT / "phaseB_results.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    n_ok = sum(1 for r in results if r["ok"])
    print(f"Phase B finished {n_ok}/{len(results)}", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
