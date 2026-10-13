"""Phase B: secondary-target decode on cached Phase A latents (Report 2 v2).

Reads ``outputs/real_quadrant/m3_v2/room_A/<session>/latent_cache/`` and
``figure_contract/arrays.npz``. Never writes under ``hippo-dq/`` or v1 ``m3/``.

Do not launch until Phase A finishes and this module has been reviewed.
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
from analysis.real_quadrant.run_m3_v2 import (
    OUT_ROOT,
    load_v2_cohort,
)
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

REPO_ROOT = Path(__file__).resolve().parents[2]
PHASE_B_DIRNAME = "phaseB"


def _session_out(session: str) -> Path:
    return OUT_ROOT / "room_A" / session


def _phase_b_out(session: str) -> Path:
    return _session_out(session) / PHASE_B_DIRNAME


def _load_align_extras(session: str, decode_times: np.ndarray) -> dict[str, Any]:
    """Load HD / V from postions_dataset aligned to decode timestamps."""
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    pos = pd.read_csv(
        root / session / "postions_dataset.csv",
        usecols=["timestamp", "X", "Y", "V", "HD", "valid", "room"],
    )
    # nearest-timestamp align (decode grid ⊂ positions timestamps in adapter)
    pos_t = pos["timestamp"].to_numpy(dtype=float)
    idx = np.searchsorted(pos_t, decode_times, side="left")
    idx = np.clip(idx, 0, len(pos_t) - 1)
    # prefer exact / left neighbour if closer
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


def run_session_phase_b(
    session: str,
    *,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    out_dir = _session_out(session)
    cache_dir = lcache.cache_root(out_dir)
    contract = out_dir / "figure_contract"
    arrays_path = contract / "arrays.npz"
    if not arrays_path.is_file():
        raise FileNotFoundError(f"missing {arrays_path}; run Phase A first")
    if not cache_dir.is_dir():
        raise FileNotFoundError(f"missing {cache_dir}; run Phase A with cache")

    arrays = np.load(arrays_path)
    train_ok = np.asarray(arrays["train_ok"], dtype=bool)
    # eval mask: prefer explicit; else ~train among finite y
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

    # available methods = final latent files
    final_dir = cache_dir / "final"
    available = sorted(p.stem for p in final_dir.glob("*.npz"))
    if methods is None:
        methods = available
    else:
        methods = [m for m in methods if m in available]

    # categorical prep
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
        ("pc_v", targets["pc_v"], "continuous"),  # sensitivity only
    ]

    phase_b_dir = _phase_b_out(session)
    phase_b_dir.mkdir(parents=True, exist_ok=True)
    # Dual provenance: Phase B code sha vs Phase A latent-cache sha.
    phase_a_sha = (
        (meta or {}).get("git_sha")
        or json.loads((out_dir / "source_summary.json").read_text()).get("git_sha")
    )
    results: dict[str, Any] = {
        "session": session,
        "git_sha": _git_sha(),  # Phase B code (this worktree HEAD)
        "git_sha_phaseB": _git_sha(),
        "git_sha_phaseA_latents": phase_a_sha,
        "latent_cache_meta": {
            "git_sha": (meta or {}).get("git_sha"),
            "config_sha256": (meta or {}).get("config_sha256"),
            "n_units": (meta or {}).get("n_units"),
            "latent_dims": (meta or {}).get("latent_dims"),
        },
        "hf_regions": sorted(HF_REGION_ALLOWLIST),
        "methods": methods,
        "class_merge": {
            "wall_distance_bin": wall_meta,
            "movement_state": move_meta,
        },
        "by_method": {},
    }

    # Need position predictions for beyond-position control — from primary
    # session_report / predictions if present
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
                # tighten masks via HD valid
                y_use = y_t.copy()
                bad = ~targets["head_direction_valid"]
                y_use[bad] = np.nan
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
                    shift_fracs=shift_fracs,
                    pass_min=pass_min,
                )
                # strip large arrays from JSON dump
                slim = {
                    k: v for k, v in dec.items()
                    if k not in ("pred_ridge", "pred_knn", "eval_index")
                }
                if tname == "distance_to_wall" and method in pred_xy_by_method:
                    slim["beyond_position"] = beyond_position_control(
                        pred_xy_by_method[method],
                        targets["distance_to_wall"],
                        wall_bin,
                        poly,
                        dec["eval_index"],
                    )
                if tname == "movement_state" and "speed" in method_out:
                    # use speed ridge preds if we kept them — re-decode light
                    sp = method_out.get("speed", {})
                if tname == "speed":
                    # stash for beyond-speed after movement_state
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
        # drop stashed preds
        method_out.pop("_speed_pred_ridge", None)
        method_out.pop("_speed_eval_index", None)
        results["by_method"][method] = method_out

    (phase_b_dir / "phaseB_results.json").write_text(
        json.dumps(results, indent=2, default=str) + "\n"
    )
    # also save target arrays for QC (no neural data)
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
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args(argv)

    require_clean_git_for_real_data()
    print(f"OUT_ROOT={OUT_ROOT.resolve()}", flush=True)
    print(f"PHASE_B under {OUT_ROOT / 'room_A'}/<session>/{PHASE_B_DIRNAME}", flush=True)

    cohort = load_v2_cohort(include_bridge=bool(args.bridge))
    selected = list(cohort["selected_sessions"])
    if args.session:
        selected = [s for s in selected if s["session"] == args.session]
    elif args.smoke:
        selected = selected[:1]

    t0 = time.perf_counter()
    results = []
    for sel in selected:
        session = sel["session"]
        print(f"Phase B start {session}", flush=True)
        try:
            rec = run_session_phase_b(session)
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
        results.append({"ok": ok, "session": session, "error": err})

    out = OUT_ROOT / "phaseB_results.json"
    out.write_text(json.dumps({"results": results}, indent=2) + "\n")
    n_ok = sum(1 for r in results if r["ok"])
    print(f"Phase B finished {n_ok}/{len(results)}", flush=True)
    return 0 if n_ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
