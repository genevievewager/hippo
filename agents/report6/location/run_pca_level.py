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
    n1_shift_X,
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
) -> dict[str, list[float]]:
    """One N1 shift: collect n_stable and ARI for each split × d × drop_u."""
    rng = np.random.default_rng(seed)
    Xs = n1_shift_X(X, rng)
    out: dict[str, list[float]] = {}
    for split_name in ("S1", "S2", "S0"):
        ma = splits[split_name]["mask_A"]
        mb = splits[split_name]["mask_B"]
        for drop_u in (False, True):
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
) -> float:
    """Wall seconds for one N1 repeat on this session (probe)."""
    t0 = time.time()
    for i in range(n_probe):
        _n1_one_repeat(
            X, splits, retained,
            d_list=d_list, times_s=times_s, dt_s=dt_s,
            seed=9000 + i, M_by_d=M_by_d,
        )
    return (time.time() - t0) / max(1, n_probe)


def run_session(
    s,
    cfg: dict[str, Any],
    *,
    n1_repeats: int,
) -> dict[str, Any]:
    if s.X_counts is None:
        return {"alias": s.alias, "arm": s.arm, "error": "no_X_counts"}
    X = np.asarray(s.X_counts, dtype=float)
    retained = np.asarray(s.retained, dtype=bool)
    times_s = np.asarray(s.decode_times, dtype=float)
    dt_s = float(cfg.get("dt_s", DT_S))
    primary_d = int(min(s.primary_d_pca, X.shape[1] // 2, X.shape[1]))
    d_list = sorted({3, primary_d})
    d_list = [d for d in d_list if d >= 1]

    splits_meta = _split_masks(X, retained, cfg)
    splits = {k: splits_meta[k] for k in ("S1", "S2", "S0")}

    # Elbow M on S1 A half at d=3 (pilot target space) and at primary d
    M_by_d: dict[int, int] = {}
    for d in d_list:
        fa_probe = None
        from agents.report6.location.pca_core import fit_pca_half
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
                })
                # session pass: S1, u_out, primary d
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
    args = ap.parse_args()

    build_alias_map(force=False)
    cfg = load_cfg()
    LOC_OUT.mkdir(parents=True, exist_ok=True)

    arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    sessions = []
    for arm in arms:
        sessions.extend(list(iter_sessions(arm=arm)))
    if args.alias:
        sessions = [s for s in sessions if s.alias == args.alias]
        if not sessions:
            raise SystemExit(f"alias not found: {args.alias}")

    # Time estimate on first real session (or first available)
    probe = next((s for s in sessions if s.arm == "real"), sessions[0])
    print(f"[estimate] probe {probe.alias} n_units={probe.n_units}", flush=True)
    X = np.asarray(probe.X_counts, dtype=float)
    retained = np.asarray(probe.retained, dtype=bool)
    splits_meta = _split_masks(X, retained, cfg)
    splits = {k: splits_meta[k] for k in ("S1", "S2", "S0")}
    primary_d = int(min(probe.primary_d_pca, X.shape[1] // 2))
    d_list = sorted({3, primary_d})
    from agents.report6.location.pca_core import fit_pca_half
    M_by_d = {}
    for d in d_list:
        fa = fit_pca_half(X, splits["S1"]["mask_A"], d=d, drop_u=True)
        M, _ = choose_k(fa["C"][splits["S1"]["mask_A"]], seed=0)
        M_by_d[d] = int(M)

    sec_per_n1 = estimate_n1_seconds(
        X, splits, retained,
        d_list=d_list,
        times_s=np.asarray(probe.decode_times, dtype=float),
        dt_s=float(cfg.get("dt_s", DT_S)),
        M_by_d=M_by_d,
        n_probe=1,
    )
    n_sess = len(sessions)
    # observed evals ~ small vs N1; ignore
    est_20 = sec_per_n1 * 20 * n_sess
    est_50 = sec_per_n1 * 50 * n_sess
    print(
        f"[estimate] sec/N1_repeat~{sec_per_n1:.1f}  "
        f"N1=20 cohort~{est_20/60:.1f} min  N1=50 cohort~{est_50/60:.1f} min  "
        f"n_sessions={n_sess}",
        flush=True,
    )
    estimate = {
        "probe_alias": probe.alias,
        "sec_per_n1_repeat": sec_per_n1,
        "n_sessions": n_sess,
        "est_minutes_n1_20": est_20 / 60.0,
        "est_minutes_n1_50": est_50 / 60.0,
    }
    save_json(estimate, LOC_OUT / "n1_time_estimate.json")
    if args.estimate_only:
        print(json.dumps(estimate, indent=2))
        return

    if args.n1 > 0:
        n1 = args.n1
        reason = "cli"
    elif est_50 > 2 * 3600:
        n1 = 20
        reason = "N1=50 would exceed 2h; first pass N1=20"
    else:
        n1 = 50
        reason = "N1=50 within 2h"
    print(f"[N1] using n1_repeats={n1} ({reason})", flush=True)

    results = []
    for s in sessions:
        print(
            f"[pca_level] {s.alias} {s.arm} primary_d={s.primary_d_pca} n1={n1}",
            flush=True,
        )
        t0 = time.time()
        r = run_session(s, cfg, n1_repeats=n1)
        r["wall_s"] = time.time() - t0
        r["n1_reason"] = reason
        save_json(r, LOC_OUT / "pca_level" / f"{s.alias}.json")
        results.append(r)
        print(
            f"  done wall_s={r['wall_s']:.0f} "
            f"pass={r.get('session_pass_S1_u_out_primary')}",
            flush=True,
        )

    # flat table
    flat = []
    for r in results:
        flat.extend(r.get("rows") or [])
    save_json(
        {
            "n1_repeats": n1,
            "n1_reason": reason,
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
                }
                for r in results
            ],
            "rows": flat,
        },
        LOC_OUT / "pca_level_table.json",
    )
    print(
        f"[done] n1={n1} sessions={len(results)} "
        f"PASS S1/u_out/primary="
        f"{sum(1 for r in results if r.get('session_pass_S1_u_out_primary'))}/"
        f"{len(results)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
