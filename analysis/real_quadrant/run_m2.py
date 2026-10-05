"""M2: all methods on rule-selected 2-room session, room A."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from analysis.real_quadrant.adapter import build_segment_bundle
from analysis.real_quadrant.session_select import (
    M1_RULE,
    M2_RULE,
    append_selection_manifest,
    m1_lex_first_session,
    select_median_units_2room_session,
)
from realtime.quadrant_n5 import load_quadrant_n5_yaml
from realtime.quadrant_n5_run import (
    _load_arrays_for_analyze_source,
    _make_rep,
    _sqrt_zscore_train,
    _valid_mask,
    analyze_source,
    causal_train_test_split,
    fit_transform_representation_timed,
    require_clean_git_for_real_data,
    segment_retained_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "outputs" / "real_quadrant" / "m2"
M2_METHODS = ("raw", "raw_lag", "pca", "dm", "lds", "isomap")
GPFA_BUDGET_S = 3600.0


def _probe_gpfa(cfg: dict, bundle: dict, streams: dict) -> dict:
    """Time one GPFA fit+transform at smallest latent d; project full grid."""
    X_counts, y, decode_times, _units_df, _unit_ids, load_meta = (
        _load_arrays_for_analyze_source(cfg, None, "real", bundle=bundle)
    )
    split = cfg["split"]
    train_mask, test_mask = causal_train_test_split(
        decode_times, float(split["train_frac"]), gap_s=float(split["gap_s"]),
    )
    retained = segment_retained_mask(
        decode_times,
        float(load_meta["segment_t0"]),
        float(load_meta["segment_t1"]),
    )
    target_valid = np.asarray(load_meta["target_valid"], dtype=bool)
    train_mask = train_mask & retained & target_valid
    test_mask = test_mask & retained & target_valid
    X, _ = _sqrt_zscore_train(X_counts, train_mask)
    n = len(decode_times)
    train_ok = train_mask.copy()
    eval_mask = test_mask.copy()
    for m in ("gpfa",):
        train_ok &= _valid_mask(m, n, cfg)
        eval_mask &= _valid_mask(m, n, cfg)

    dims = [int(d) for d in cfg["latent_dims"]]
    d_min = min(dims)
    n_folds = int(split["inner_cv_blocks"])
    # Non-nested refit_per_d: one fit per (fold, d) in CV + one per d in
    # score_d_curve + one final selected fit.
    n_projected_fits = n_folds * len(dims) + len(dims) + 1

    methods_seed = int(streams["methods"])
    model = _make_rep("gpfa", d_min, cfg, methods_seed, n_fit=int(train_ok.sum()))
    t0 = time.perf_counter()
    _Z, fit_s, transform_s = fit_transform_representation_timed(
        "gpfa", model, X, train_ok,
    )
    wall_one = time.perf_counter() - t0
    projected = float(n_projected_fits) * wall_one
    return {
        "d_min": d_min,
        "n_train": int(train_ok.sum()),
        "n_units": int(X.shape[1]),
        "fit_s": fit_s,
        "transform_s": transform_s,
        "wall_one_fit_transform_s": wall_one,
        "n_projected_fits": n_projected_fits,
        "projected_grid_s": projected,
        "budget_s": GPFA_BUDGET_S,
        "skip": projected > GPFA_BUDGET_S,
    }


def main() -> int:
    require_clean_git_for_real_data()
    data_root = Path(os.environ["HIPPO_DATA_ROOT"])
    selection = select_median_units_2room_session(data_root)
    append_selection_manifest({**selection, "milestone": "M2"})
    session = selection["selected_session"]
    m1_session = m1_lex_first_session(data_root)
    assert selection["m1_session"] == m1_session

    out_dir = OUT_ROOT / "room_A" / "all_methods"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "session_selection.json").write_text(
        json.dumps(
            {
                **selection,
                "m1_was_chosen_by": M1_RULE,
                "m2_chosen_by": M2_RULE,
                "m1_same_as_m2": bool(selection["same_as_m1_lex_first"]),
                "note": (
                    "M1 used lex-first 2rooms without the median-units rule; "
                    "M2 uses the median-units rule. Sessions differ."
                    if not selection["same_as_m1_lex_first"]
                    else "M1 lex-first coincides with median-units rule."
                ),
            },
            indent=2,
        )
        + "\n"
    )

    print(
        f"M2 session rule selected "
        f"n_units={selection['selected_n_units']} "
        f"(median={selection['median_units']}); "
        f"same_as_m1={selection['same_as_m1_lex_first']}",
        flush=True,
    )

    t_bundle = time.perf_counter()
    bundle = build_segment_bundle(session, room="A", data_root=data_root)
    print(f"bundle built in {time.perf_counter() - t_bundle:.1f}s", flush=True)

    cfg = dict(load_quadrant_n5_yaml())
    cfg["phase3"] = dict(cfg["phase3"], learning_curve_source="__skip__")
    streams = {
        "methods": 0,
        "data_seed": 0,
        "master_seed": 0,
        "seed_index": 0,
        "trajectory": 0,
        "neural": 0,
        "recording_noise": 0,
        "sorting_errors": 0,
    }

    gpfa_probe = _probe_gpfa(cfg, bundle, streams)
    (out_dir / "gpfa_timing_probe.json").write_text(
        json.dumps(gpfa_probe, indent=2) + "\n"
    )
    print(json.dumps({"gpfa_probe": gpfa_probe}, indent=2), flush=True)

    methods = list(M2_METHODS)
    if not gpfa_probe["skip"]:
        methods.append("gpfa")
    else:
        print(
            f"GPFA skipped for M2: projected {gpfa_probe['projected_grid_s']:.0f}s "
            f"> {GPFA_BUDGET_S:.0f}s budget "
            f"(one fit at d={gpfa_probe['d_min']}: "
            f"{gpfa_probe['wall_one_fit_transform_s']:.1f}s)",
            flush=True,
        )

    t0 = time.perf_counter()
    summary = analyze_source(
        cfg,
        None,
        "real",
        streams,
        0,
        bundle=bundle,
        output_dir=out_dir,
        method_keys=tuple(methods),
        save_figure_contract=True,
    )
    wall = time.perf_counter() - t0
    floor_med = float(summary["floor"]["median"])

    rows = []
    for m in summary["methods"]:
        ridge_med = float(m["ridge"]["median"])
        knn_med = float(m["knn"]["median"])
        a13 = m.get("a13") or {}
        rows.append({
            "method": m["method"],
            "selected_d": m.get("primary_d"),
            "ridge_median_cm": ridge_med,
            "normalized_error": ridge_med / floor_med if floor_med > 0 else None,
            "knn_median_cm": knn_med,
            "a13_status": a13.get("status"),
            "a13_ridge_pass": a13.get("ridge_pass"),
            "a13_knn_pass": a13.get("knn_pass"),
            "a13_ridge_delta_cm": a13.get("ridge_median_minus_floor"),
            "fit_s": m.get("fit_s"),
            "transform_s": m.get("transform_s"),
            "elapsed_s": m.get("elapsed_s"),
        })

    report = {
        "session_selection": {
            "same_as_m1": bool(selection["same_as_m1_lex_first"]),
            "selected_n_units": selection["selected_n_units"],
            "median_units": selection["median_units"],
            "n_eligible": selection["n_eligible"],
            "m1_rule": M1_RULE,
            "m2_rule": M2_RULE,
        },
        "n_units": int(summary["n_units"]),
        "n_train": int(summary["n_train"]),
        "n_eval": int(summary["n_eval"]),
        "floor_median_cm": floor_med,
        "integrity_match_fraction": float(
            bundle["meta"]["integrity_match_fraction"]
        ),
        "gpfa": gpfa_probe,
        "methods": rows,
        "wall_time_s": wall,
        "figure_contract_dir": summary.get("figure_contract_dir"),
        "methods_run": methods,
    }
    (out_dir / "m2_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
