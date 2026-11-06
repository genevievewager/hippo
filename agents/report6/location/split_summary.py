#!/usr/bin/env python3
"""Stop point 1: per-session split summary (tau_ac, purge, block/bin counts).

Counts and fractions only in printed output; no real IDs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.report6.io_cohort import SessionData, build_alias_map, iter_sessions  # noqa: E402
from agents.report6.paths import DT_S, OUT_ROOT  # noqa: E402
from agents.report6.pipeline import save_json  # noqa: E402

CFG_PATH = Path(__file__).resolve().parent / "config.yaml"
LOC_OUT = OUT_ROOT / "location"


def load_cfg() -> dict[str, Any]:
    return yaml.safe_load(CFG_PATH.read_text())


def tau_ac_seconds(
    X: np.ndarray,
    mask: np.ndarray,
    *,
    dt_s: float = DT_S,
    max_lag_s: float = 5.0,
) -> float:
    """Median over units of first lag where window-count ACF falls below 1/e."""
    max_lag = max(1, int(round(max_lag_s / dt_s)))
    m = np.asarray(mask, dtype=bool)
    taus: list[float] = []
    thr = 1.0 / np.e
    for u in range(X.shape[1]):
        x = np.asarray(X[m, u], dtype=float)
        if x.size < max_lag + 2:
            continue
        x = x - x.mean()
        var = float(np.dot(x, x))
        if var <= 0:
            continue
        found = False
        for lag in range(1, max_lag + 1):
            c = float(np.dot(x[:-lag], x[lag:])) / var
            if c < thr:
                taus.append(lag * dt_s)
                found = True
                break
        if not found:
            taus.append(max_lag_s)
    if not taus:
        return float("nan")
    return float(np.median(np.asarray(taus)))


def purge_g_seconds(tau_ac: float, *, window_s: float, purge_cap_s: float) -> float:
    """g = max(window, 2 tau_ac), capped."""
    if not np.isfinite(tau_ac):
        return float(min(purge_cap_s, window_s))
    return float(min(purge_cap_s, max(window_s, 2.0 * tau_ac)))


def s1_masks(
    n_bins: int,
    retained: np.ndarray,
    *,
    block_s: float,
    g_s: float,
    dt_s: float,
) -> dict[str, Any]:
    """Alternating 60 s blocks; purge bins within g of every block edge."""
    block = max(1, int(round(block_s / dt_s)))
    g_bins = int(np.ceil(g_s / dt_s))
    n_blocks = int(np.ceil(n_bins / block))
    assign = np.full(n_bins, -1, dtype=int)  # 0=A, 1=B, -1=purged/out
    retained = np.asarray(retained, dtype=bool)
    for b in range(n_blocks):
        lo = b * block
        hi = min(n_bins, (b + 1) * block)
        half = 0 if (b % 2 == 0) else 1  # odd blocks A per amendment: b=0 even->B?
        # Amendment: "odd blocks go to A, even blocks to B" (1-indexed odd = A).
        # 1-indexed block b+1: odd -> A. So b even (0,2,...) -> A.
        half = 0 if (b % 2 == 0) else 1
        assign[lo:hi] = half
        # purge near edges of this block
        left = lo
        right = hi - 1
        pl = max(lo, left)
        pr = min(hi, left + g_bins)
        assign[pl:pr] = -1
        pl = max(lo, right - g_bins + 1)
        pr = min(hi, right + 1)
        assign[pl:pr] = -1
    # also drop non-retained
    assign[~retained] = -1
    a = assign == 0
    b = assign == 1
    return {
        "n_blocks": n_blocks,
        "block_bins": block,
        "g_bins": g_bins,
        "n_bins_A": int(a.sum()),
        "n_bins_B": int(b.sum()),
        "n_bins_purged_or_out": int((assign < 0).sum()),
        "frac_A": float(a.sum() / max(1, retained.sum())),
        "frac_B": float(b.sum() / max(1, retained.sum())),
    }


def s2_masks(
    n_bins: int,
    retained: np.ndarray,
    *,
    g_s: float,
    dt_s: float,
) -> dict[str, Any]:
    """First half vs second half; purge at the single boundary."""
    retained = np.asarray(retained, dtype=bool)
    mid = n_bins // 2
    g_bins = int(np.ceil(g_s / dt_s))
    assign = np.full(n_bins, -1, dtype=int)
    assign[:mid] = 0
    assign[mid:] = 1
    lo = max(0, mid - g_bins)
    hi = min(n_bins, mid + g_bins)
    assign[lo:hi] = -1
    assign[~retained] = -1
    a = assign == 0
    b = assign == 1
    return {
        "g_bins": g_bins,
        "n_bins_A": int(a.sum()),
        "n_bins_B": int(b.sum()),
        "n_bins_purged_or_out": int((assign < 0).sum()),
        "frac_A": float(a.sum() / max(1, retained.sum())),
        "frac_B": float(b.sum() / max(1, retained.sum())),
    }


def summarize_session(s: SessionData, cfg: dict[str, Any]) -> dict[str, Any]:
    if s.X_counts is None:
        return {"alias": s.alias, "arm": s.arm, "error": "no_X_counts"}
    X = s.X_counts
    dt = float(cfg.get("dt_s", DT_S))
    tau = tau_ac_seconds(X, s.retained, dt_s=dt)
    g = purge_g_seconds(
        tau,
        window_s=float(cfg.get("window_s", 0.25)),
        purge_cap_s=float(cfg.get("purge_cap_s", 5.0)),
    )
    n = X.shape[0]
    s1 = s1_masks(
        n, s.retained,
        block_s=float(cfg.get("block_s", 60.0)),
        g_s=g,
        dt_s=dt,
    )
    s2 = s2_masks(n, s.retained, g_s=g, dt_s=dt)
    return {
        "alias": s.alias,
        "arm": s.arm,
        "n_units": int(s.n_units),
        "n_bins_total": int(n),
        "n_bins_retained": int(np.asarray(s.retained, dtype=bool).sum()),
        "tau_ac_s": tau,
        "g_s": g,
        "S1": s1,
        "S2": s2,
    }


def main() -> None:
    build_alias_map(force=False)
    cfg = load_cfg()
    LOC_OUT.mkdir(parents=True, exist_ok=True)
    # confirm gitignore before writing
    import subprocess
    chk = subprocess.run(
        ["git", "check-ignore", "-v", str(LOC_OUT / "split_summary.json")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if chk.returncode != 0:
        raise SystemExit(
            f"outputs/report6/location is not gitignored: {chk.stdout} {chk.stderr}"
        )

    # copy yaml into outputs (data-free)
    (LOC_OUT / "config.yaml").write_text(CFG_PATH.read_text())

    rows = []
    for s in list(iter_sessions(arm="real")) + list(iter_sessions(arm="sim")):
        r = summarize_session(s, cfg)
        rows.append(r)
        save_json(r, LOC_OUT / "splits" / f"{r['alias']}.json")

    save_json(rows, LOC_OUT / "split_summary.json")

    # chat-safe print: counts only
    print("location stop point 1: split summary", flush=True)
    print(f"yaml: agents/report6/location/config.yaml -> {LOC_OUT / 'config.yaml'}", flush=True)
    print(
        f"n_sessions={len(rows)} "
        f"(real={sum(1 for r in rows if r.get('arm')=='real')}, "
        f"sim={sum(1 for r in rows if r.get('arm')=='sim')})",
        flush=True,
    )
    for r in rows:
        if "error" in r:
            print(f"  {r['alias']} ERROR {r['error']}", flush=True)
            continue
        s1 = r["S1"]
        print(
            f"  {r['alias']} {r['arm']} "
            f"tau_ac_s={r['tau_ac_s']:.3f} g_s={r['g_s']:.3f} "
            f"S1_blocks={s1['n_blocks']} "
            f"S1_bins_A={s1['n_bins_A']} S1_bins_B={s1['n_bins_B']} "
            f"S2_bins_A={r['S2']['n_bins_A']} S2_bins_B={r['S2']['n_bins_B']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
