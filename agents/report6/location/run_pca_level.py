#!/usr/bin/env python3
"""Amendment stop point 2: PCA-level results table (counts / pass-fail only).

Time-estimates N1 before the loop. Prefer N1=20 if N1=50 would exceed ~2 h;
complete 50 afterwards. No figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.report6.io_cohort import build_alias_map, iter_sessions  # noqa: E402
from agents.report6.location.pca_core import (  # noqa: E402
    evaluate_split,
    fit_pca_half,
    n1_shift_X,
    rebin_window_counts,
)
from agents.report6.location.split_summary import (  # noqa: E402
    load_cfg,
    purge_g_seconds,
    s0_masks,
    s1_masks,
    s2_masks,
    tau_ac_seconds,
)
from agents.report6.paths import DT_S, OUT_ROOT  # noqa: E402
from agents.report6.pipeline import save_json  # noqa: E402
from agents.report6.target_rule import choose_k  # noqa: E402

LOC_OUT = OUT_ROOT / "location"


def _split_masks(X, retained, cfg: dict[str, Any]) -> dict[str, dict]:
    dt = float(cfg.get("dt_s", DT_S))
    tau = tau_ac_seconds(X, retained, dt_s=dt)
    g = purge_g_seconds(
        tau,
        window_s=float(cfg.get("window_s", 0.25)),
        purge_cap_s=float(cfg.get("purge_cap_s", 5.0)),
    )
    n = X.shape[0]
    return {
        "S1": s1_masks(
            n, retained,
            block_s=float(cfg.get("block_s", 60.0)), g_s=g, dt_s=dt,
        ),
        "S2": s2_masks(n, retained, g_s=g, dt_s=dt),
        "S0": s0_masks(n, retained),
        "tau_ac_s": tau,
        "g_s": g,
    }


def _row_from_eval(ev: dict[str, Any], *, n1: dict[str, Any] | None) -> dict[str, Any]:
    tiles = ev.get("tiles") or {}
    coord = ev.get("coord") or {}
    e_all = coord.get("e_rigid_all") or {}
    e_b = coord.get("e_rigid_B") or {}
    n_stable = ev["n_stable"]
    ari = tiles.get("ari_all")
    frac = tiles.get("frac_shift_lt_radius")
    med_r = tiles.get("median_r_over_sigma_noise")

    n1_ns_p95 = None if n1 is None else n1.get("n_stable_p95")
    n1_ari_p95 = None if n1 is None else n1.get("ari_p95")

    pass_a = (
        n_stable >= 2
        and (n1_ns_p95 is None or n_stable > n1_ns_p95)
    )
    pass_b = (
        ari is not None
        and np.isfinite(ari)
        and (n1_ari_p95 is None or ari > n1_ari_p95)
    )
    pass_c = (
        frac is not None
        and np.isfinite(frac)
        and frac >= 0.5
        and med_r is not None
        and np.isfinite(med_r)
        and med_r >= 2.0
    )
    # Decision rule is for S1 + drop_u + primary d; callers set session_pass.
    return {
        "d": ev["d"],
        "drop_u": ev["drop_u"],
        "M": ev.get("M"),
        "n_stable": n_stable,
        "n_stable_n1_p95": n1_ns_p95,
        "e_rigid_median_all": e_all.get("median"),
        "e_rigid_p90_all": e_all.get("p90"),
        "e_rigid_median_B": e_b.get("median"),
        "e_rigid_p90_B": e_b.get("p90"),
        "e_affine_B_median": coord.get("e_affine_B_median"),
        "e_rigid_per_axis_frac": coord.get("e_rigid_per_axis_frac"),
        "sigma_noise": ev.get("sigma_noise"),
        "sigma_split": ev.get("sigma_split"),
        "ari": ari,
        "ari_n1_p95": n1_ari_p95,
        "frac_shift_lt_radius": frac,
        "median_r_over_sigma_noise": med_r,
        "pass_n_stable": bool(pass_a),
        "pass_ari": bool(pass_b) if ari is not None and np.isfinite(ari) else None,
        "pass_tiles": bool(pass_c) if frac is not None and np.isfinite(frac) else None,
    }


def _n1_one_repeat(
    X: np.ndarray,
    splits: dict[str, dict],
    retained: np.ndarray,
    *,
    d_list: list[int],
    times_s: np.ndarray,
    dt_s: float,
    seed: int,
    M_by_d: dict[int, int],
    n1_splits: tuple[str, ...] = ("S1", "S2", "S0"),
    n1_drop_u: tuple[bool, ...] = (False, True),
) -> dict[str, list[float]]:
    """One N1 shift: collect n_stable and ARI for each split × d × drop_u."""
    rng = np.random.default_rng(seed)
    Xs = n1_shift_X(X, rng)
    out: dict[str, list[float]] = {}
    for split_name in n1_splits:
        ma = splits[split_name]["mask_A"]
        mb = splits[split_name]["mask_B"]
        for drop_u in n1_drop_u:
            for d in d_list:
                key = f"{split_name}|d{d}|u{'out' if drop_u else 'in'}"
                try:
                    ev = evaluate_split(
                        Xs, ma, mb, retained,
                        d=d, drop_u=drop_u,
                        times_s=times_s, dt_s=dt_s, seed=seed,
                        M=M_by_d.get(d),
                        light=True,
                    )
                    out.setdefault(key + "|n_stable", []).append(float(ev["n_stable"]))
                    ari = (ev.get("tiles") or {}).get("ari_all")
                    if ari is not None and np.isfinite(ari):
                        out.setdefault(key + "|ari", []).append(float(ari))
                except Exception:
                    continue
    return out


def estimate_n1_seconds(
    X: np.ndarray,
    splits: dict[str, dict],
    retained: np.ndarray,
    *,
    d_list: list[int],
    times_s: np.ndarray,
    dt_s: float,
    M_by_d: dict[int, int],
    n_probe: int = 1,
    n1_splits: tuple[str, ...] = ("S1", "S2", "S0"),
    n1_drop_u: tuple[bool, ...] = (False, True),
) -> float:
    """Wall seconds for one N1 repeat on this session (probe)."""
    t0 = time.time()
    for i in range(n_probe):
        _n1_one_repeat(
            X, splits, retained,
            d_list=d_list, times_s=times_s, dt_s=dt_s,
            seed=9000 + i, M_by_d=M_by_d,
            n1_splits=n1_splits, n1_drop_u=n1_drop_u,
        )
    return (time.time() - t0) / max(1, n_probe)


def _d_list_for_session(s, cfg: dict[str, Any], X: np.ndarray) -> list[int]:
    primary_d = int(min(s.primary_d_pca, X.shape[1] // 2, X.shape[1]))
    grid = [int(d) for d in (cfg.get("d_grid") or [2, 3, 5, 10, 20])]
    return sorted({d for d in grid + [primary_d] if 1 <= d <= X.shape[1]})


def _prepare_X(s, cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, retained) at cfg window_s; NaN-leading bins cleared from retained."""
    X = np.asarray(s.X_counts, dtype=float)
    retained = np.asarray(s.retained, dtype=bool).copy()
    dt_s = float(cfg.get("dt_s", DT_S))
    w_src = 0.25  # count caches are Report 2 W = 250 ms
    w_dst = float(cfg.get("window_s", w_src))
    if abs(w_dst - w_src) > 1e-12:
        X = rebin_window_counts(X, w_src_s=w_src, w_dst_s=w_dst, dt_s=dt_s)
        bad = ~np.isfinite(X).all(axis=1)
        retained = retained & ~bad
        X = np.nan_to_num(X, nan=0.0)
    return X, retained


def run_session(
    s,
    cfg: dict[str, Any],
    *,
    n1_repeats: int,
    n1_splits: tuple[str, ...] = ("S1", "S2", "S0"),
    n1_drop_u: tuple[bool, ...] = (False, True),
) -> dict[str, Any]:
    if s.X_counts is None:
        return {"alias": s.alias, "arm": s.arm, "error": "no_X_counts"}
    X, retained = _prepare_X(s, cfg)
    times_s = np.asarray(s.decode_times, dtype=float)
    dt_s = float(cfg.get("dt_s", DT_S))
    primary_d = int(min(s.primary_d_pca, X.shape[1] // 2, X.shape[1]))
    d_list = _d_list_for_session(s, cfg, X)

    splits_meta = _split_masks(X, retained, cfg)
    splits = {k: splits_meta[k] for k in ("S1", "S2", "S0")}

    # Elbow M on S1 A half at each d
    M_by_d: dict[int, int] = {}
    for d in d_list:
        fa_probe = fit_pca_half(X, splits["S1"]["mask_A"], d=d, drop_u=True)
        M, _ = choose_k(fa_probe["C"][splits["S1"]["mask_A"]], seed=int(cfg.get("seed", 0)))
        M_by_d[d] = int(M)

    # Observed metrics
    observed: dict[str, Any] = {}
    for split_name in ("S1", "S2", "S0"):
        ma = splits[split_name]["mask_A"]
        mb = splits[split_name]["mask_B"]
        observed[split_name] = {}
        for drop_u in (False, True):
            u_key = "u_out" if drop_u else "u_in"
            observed[split_name][u_key] = {}
            for d in d_list:
                ev = evaluate_split(
                    X, ma, mb, retained,
                    d=d, drop_u=drop_u,
                    times_s=times_s, dt_s=dt_s,
                    seed=int(cfg.get("seed", 0)) + d,
                    M=M_by_d.get(d),
                )
                observed[split_name][u_key][f"d{d}"] = ev

    # N1 loop
    n1_acc: dict[str, list[float]] = {}
    t_n1 = time.time()
    for i in range(n1_repeats):
        part = _n1_one_repeat(
            X, splits, retained,
            d_list=d_list, times_s=times_s, dt_s=dt_s,
            seed=int(cfg.get("seed", 0)) + 1000 + i,
            M_by_d=M_by_d,
            n1_splits=n1_splits, n1_drop_u=n1_drop_u,
        )
        for k, vals in part.items():
            n1_acc.setdefault(k, []).extend(vals)
        if (i + 1) % 5 == 0 or i == 0:
            print(
                f"    N1 {i+1}/{n1_repeats} wall={time.time()-t_n1:.0f}s",
                flush=True,
            )

    def n1_stats(split: str, d: int, drop_u: bool) -> dict[str, float | None]:
        u = "out" if drop_u else "in"
        key = f"{split}|d{d}|u{u}"
        ns = n1_acc.get(key + "|n_stable") or []
        ar = n1_acc.get(key + "|ari") or []
        return {
            "n_stable_p95": float(np.percentile(ns, 95)) if ns else None,
            "ari_p95": float(np.percentile(ar, 95)) if ar else None,
            "n_repeats_n_stable": len(ns),
            "n_repeats_ari": len(ar),
        }

    table_rows = []
    session_pass = None
    for split_name in ("S1", "S2", "S0"):
        for drop_u in (False, True):
            u_key = "u_out" if drop_u else "u_in"
            for d in d_list:
                ev = observed[split_name][u_key][f"d{d}"]
                n1s = n1_stats(split_name, d, drop_u)
                row = _row_from_eval(ev, n1=n1s)
                row.update({
                    "alias": s.alias,
                    "arm": s.arm,
                    "split": split_name,
                    "primary_d": primary_d,
                    "is_primary_d": d == primary_d,
                    "n1_repeats": n1_repeats,
                    "window_s": float(cfg.get("window_s", 0.25)),
                    "angles_deg": ev.get("angles_deg"),
                })
                # session pass: S1, u_out, primary d (pre-registered conjunction)
                if (
                    split_name == "S1"
                    and drop_u
                    and d == primary_d
                ):
                    flags = [row["pass_n_stable"], row["pass_ari"], row["pass_tiles"]]
                    session_pass = all(bool(f) for f in flags if f is not None)
                    row["session_pass_rule"] = session_pass
                table_rows.append(row)

    return {
        "alias": s.alias,
        "arm": s.arm,
        "animal": s.animal_alias,
        "primary_d": primary_d,
        "M_by_d": M_by_d,
        "tau_ac_s": splits_meta["tau_ac_s"],
        "g_s": splits_meta["g_s"],
        "window_s": float(cfg.get("window_s", 0.25)),
        "d_list": d_list,
        "n1_repeats": n1_repeats,
        "session_pass_S1_u_out_primary": session_pass,
        "rows": table_rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n1", type=int, default=0, help="0 = auto from estimate")
    ap.add_argument("--estimate-only", action="store_true")
    ap.add_argument("--arms", default="real,sim")
    ap.add_argument("--alias", default="", help="optional single alias")
    ap.add_argument(
        "--window-s", type=float, default=None,
        help="override cfg window_s (0.25 source; 0.5 or 1.0 via rebin)",
    )
    ap.add_argument(
        "--d-grid", default="",
        help="comma d list; empty = cfg d_grid + primary",
    )
    ap.add_argument(
        "--n1-scope", default="full",
        choices=("full", "passrule"),
        help="full: all splits×u; passrule: S1×u_out only (faster floors)",
    )
    ap.add_argument(
        "--out-tag", default="",
        help="suffix for outputs (e.g. w1s, n1_50); empty = pca_level_table.json",
    )
    ap.add_argument("--observed-only", action="store_true", help="skip N1 loop")
    args = ap.parse_args()

    build_alias_map(force=False)
    cfg = load_cfg()
    if args.window_s is not None:
        cfg = dict(cfg)
        cfg["window_s"] = float(args.window_s)
    if args.d_grid.strip():
        cfg = dict(cfg)
        cfg["d_grid"] = [int(x) for x in args.d_grid.split(",") if x.strip()]
    LOC_OUT.mkdir(parents=True, exist_ok=True)

    if args.n1_scope == "passrule":
        n1_splits: tuple[str, ...] = ("S1",)
        n1_drop_u: tuple[bool, ...] = (True,)
    else:
        n1_splits = ("S1", "S2", "S0")
        n1_drop_u = (False, True)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    sessions = []
    for arm in arms:
        sessions.extend(list(iter_sessions(arm=arm)))
    if args.alias:
        sessions = [s for s in sessions if s.alias == args.alias]
        if not sessions:
            raise SystemExit(f"alias not found: {args.alias}")

    probe = next((s for s in sessions if s.arm == "real"), sessions[0])
    print(
        f"[estimate] probe {probe.alias} n_units={probe.n_units} "
        f"window_s={cfg.get('window_s', 0.25)} n1_scope={args.n1_scope}",
        flush=True,
    )
    X, retained = _prepare_X(probe, cfg)
    splits_meta = _split_masks(X, retained, cfg)
    splits = {k: splits_meta[k] for k in ("S1", "S2", "S0")}
    d_list = _d_list_for_session(probe, cfg, X)
    M_by_d = {}
    for d in d_list:
        fa = fit_pca_half(X, splits["S1"]["mask_A"], d=d, drop_u=True)
        M, _ = choose_k(fa["C"][splits["S1"]["mask_A"]], seed=0)
        M_by_d[d] = int(M)

    sec_per_n1 = 0.0 if args.observed_only else estimate_n1_seconds(
        X, splits, retained,
        d_list=d_list,
        times_s=np.asarray(probe.decode_times, dtype=float),
        dt_s=float(cfg.get("dt_s", DT_S)),
        M_by_d=M_by_d,
        n_probe=1,
        n1_splits=n1_splits, n1_drop_u=n1_drop_u,
    )
    n_sess = len(sessions)
    est_20 = sec_per_n1 * 20 * n_sess
    est_50 = sec_per_n1 * 50 * n_sess
    print(
        f"[estimate] d_list={d_list} sec/N1_repeat~{sec_per_n1:.1f}  "
        f"N1=20 cohort~{est_20/60:.1f} min  N1=50 cohort~{est_50/60:.1f} min  "
        f"n_sessions={n_sess} tau_ac={splits_meta['tau_ac_s']:.3f} "
        f"g_s={splits_meta['g_s']:.3f}",
        flush=True,
    )
    estimate = {
        "probe_alias": probe.alias,
        "sec_per_n1_repeat": sec_per_n1,
        "n_sessions": n_sess,
        "d_list_probe": d_list,
        "window_s": float(cfg.get("window_s", 0.25)),
        "tau_ac_s": splits_meta["tau_ac_s"],
        "g_s": splits_meta["g_s"],
        "est_minutes_n1_20": est_20 / 60.0,
        "est_minutes_n1_50": est_50 / 60.0,
    }
    tag = args.out_tag.strip()
    est_name = "n1_time_estimate.json" if not tag else f"n1_time_estimate_{tag}.json"
    save_json(estimate, LOC_OUT / est_name)
    if args.estimate_only:
        print(json.dumps(estimate, indent=2))
        return

    if args.observed_only:
        n1 = 0
        reason = "observed-only"
    elif args.n1 > 0:
        n1 = args.n1
        reason = "cli"
    elif est_50 > 2 * 3600:
        n1 = 20
        reason = "N1=50 would exceed 2h; first pass N1=20"
    else:
        n1 = 50
        reason = "N1=50 within 2h"
    print(f"[N1] using n1_repeats={n1} ({reason})", flush=True)

    sess_dir = LOC_OUT / ("pca_level" if not tag else f"pca_level_{tag}")
    sess_dir.mkdir(parents=True, exist_ok=True)
    table_name = "pca_level_table.json" if not tag else f"pca_level_table_{tag}.json"

    results = []
    for s in sessions:
        print(
            f"[pca_level] {s.alias} {s.arm} primary_d={s.primary_d_pca} "
            f"n1={n1} W={cfg.get('window_s', 0.25)}",
            flush=True,
        )
        t0 = time.time()
        r = run_session(
            s, cfg, n1_repeats=n1,
            n1_splits=n1_splits, n1_drop_u=n1_drop_u,
        )
        r["wall_s"] = time.time() - t0
        r["n1_reason"] = reason
        save_json(r, sess_dir / f"{s.alias}.json")
        results.append(r)
        print(
            f"  done wall_s={r['wall_s']:.0f} "
            f"pass={r.get('session_pass_S1_u_out_primary')} "
            f"tau_ac={r.get('tau_ac_s')} g_s={r.get('g_s')}",
            flush=True,
        )

    flat = []
    for r in results:
        flat.extend(r.get("rows") or [])
    save_json(
        {
            "n1_repeats": n1,
            "n1_reason": reason,
            "n1_scope": args.n1_scope,
            "window_s": float(cfg.get("window_s", 0.25)),
            "estimate": estimate,
            "n_sessions": len(results),
            "n_pass_S1_u_out_primary": sum(
                1 for r in results
                if r.get("session_pass_S1_u_out_primary") is True
            ),
            "sessions": [
                {
                    "alias": r["alias"],
                    "arm": r["arm"],
                    "primary_d": r.get("primary_d"),
                    "pass": r.get("session_pass_S1_u_out_primary"),
                    "tau_ac_s": r.get("tau_ac_s"),
                    "g_s": r.get("g_s"),
                }
                for r in results
            ],
            "rows": flat,
        },
        LOC_OUT / table_name,
    )
    print(
        f"[done] n1={n1} sessions={len(results)} table={table_name} "
        f"PASS S1/u_out/primary="
        f"{sum(1 for r in results if r.get('session_pass_S1_u_out_primary'))}/"
        f"{len(results)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
