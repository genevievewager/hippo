"""Divergence ratio R = d_sim / d_lab (SPEC §5.1–5.2).

Bootstrap groups by recording_day when that column is present
(= animal_DDMMYYYY); otherwise falls back to animal|seed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Measurement / structural diagnostics — keep in tidy tables, exclude from R ranking.
# tracking_jitter_rest_cm → Q8 diagnostic only.
# frac_isi_below_1ms_gt → sim-only structural (generator lacks refractory); no lab GT.
RANK_EXCLUDE_METRICS = frozenset({
    "tracking_jitter_rest_cm",
    "frac_isi_below_1ms_gt",
})


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med))) * 1.4826


def _recording_day(session: str) -> str:
    parts = str(session).split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else str(session)


def _ensure_recording_day(df: pd.DataFrame) -> pd.DataFrame:
    if "recording_day" in df.columns:
        return df
    out = df.copy()
    out["recording_day"] = out["session"].astype(str).map(_recording_day)
    return out


def divergence_for_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    log_rates: bool = False,
    n_boot: int = 200,
    rng: np.random.Generator | None = None,
    sim_dataset: str | None = None,
) -> dict[str, Any]:
    """Compute R and bootstrap CI grouped by recording_day.

    Lab rows: dataset == 'lab'. Sim rows: dataset startswith 'sim'
    (or exact ``sim_dataset`` if given).
    """
    rng = rng or np.random.default_rng(0)
    sub = df[df["metric"] == metric].copy()
    if sub.empty:
        return {"metric": metric, "status": "empty"}
    sub = _ensure_recording_day(sub)

    vals = sub["value"].to_numpy(float)
    if log_rates or metric in {"firing_rate", "firing_rate_hz", "rate"}:
        vals = np.log1p(np.clip(vals, 0, None))
        sub = sub.assign(value=vals)

    lab = sub[sub["dataset"] == "lab"]
    if sim_dataset is not None:
        sim = sub[sub["dataset"] == sim_dataset]
    else:
        sim = sub[sub["dataset"].astype(str).str.startswith("sim")]
    if lab.empty or sim.empty:
        return {"metric": metric, "status": "missing_arm", "n_lab": len(lab), "n_sim": len(sim)}

    lab_sess = lab.groupby(["recording_day", "animal|seed", "session"], as_index=False)["value"].median()
    sim_sess = sim.groupby(["recording_day", "animal|seed", "session"], as_index=False)["value"].median()
    # Drop non-finite session medians (e.g. empty occupancy-matched masks).
    lab_sess = lab_sess[np.isfinite(lab_sess["value"].to_numpy(float))].reset_index(drop=True)
    sim_sess = sim_sess[np.isfinite(sim_sess["value"].to_numpy(float))].reset_index(drop=True)
    if lab_sess.empty or sim_sess.empty:
        return {
            "metric": metric,
            "status": "no_finite_sessions",
            "n_lab": int(len(lab_sess)),
            "n_sim": int(len(sim_sess)),
            "R": float("nan"),
        }

    scale = _mad(lab_sess["value"].to_numpy())
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0

    days_lab = sorted(lab_sess["recording_day"].astype(str).unique())
    days_sim = sorted(sim_sess["recording_day"].astype(str).unique())
    # for default arm (seed_*), recording_day may be seed_k — bootstrap those as groups
    groups_lab = days_lab
    groups_sim = days_sim if days_sim else sorted(sim_sess["animal|seed"].astype(str).unique())

    def d_lab_once(lab_df: pd.DataFrame) -> float:
        dists = []
        rows = lab_df.to_dict("records")
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if str(rows[i]["recording_day"]) == str(rows[j]["recording_day"]):
                    continue  # same-day pairs excluded (SPEC: prefer cross-day)
                dists.append(abs(float(rows[i]["value"]) - float(rows[j]["value"])) / scale)
        if not dists:
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    dists.append(abs(float(rows[i]["value"]) - float(rows[j]["value"])) / scale)
        return float(np.median(dists)) if dists else float("nan")

    def d_sim_once(sim_df: pd.DataFrame, lab_df: pd.DataFrame) -> float:
        dists = []
        for _, s in sim_df.iterrows():
            for _, l in lab_df.iterrows():
                dists.append(abs(float(s["value"]) - float(l["value"])) / scale)
        return float(np.median(dists)) if dists else float("nan")

    def d_simsim_once(sim_df: pd.DataFrame) -> float:
        rows = sim_df.to_dict("records")
        dists = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if str(rows[i]["recording_day"]) == str(rows[j]["recording_day"]):
                    continue
                dists.append(abs(float(rows[i]["value"]) - float(rows[j]["value"])) / scale)
        return float(np.median(dists)) if dists else float("nan")

    d_lab = d_lab_once(lab_sess)
    d_sim = d_sim_once(sim_sess, lab_sess)
    d_ss = d_simsim_once(sim_sess)
    R = float(d_sim / d_lab) if np.isfinite(d_lab) and d_lab > 0 else float("nan")
    direction = float(np.sign(sim_sess["value"].median() - lab_sess["value"].median()))

    boot = []
    for _ in range(int(n_boot)):
        g_lab = rng.choice(groups_lab, size=len(groups_lab), replace=True)
        g_sim = rng.choice(groups_sim, size=max(len(groups_sim), 1), replace=True)
        lab_b = lab_sess[lab_sess["recording_day"].astype(str).isin(g_lab)]
        # sim may use animal|seed for default arm
        if "recording_day" in sim_sess.columns and sim_sess["recording_day"].astype(str).isin(g_sim).any():
            sim_b = sim_sess[sim_sess["recording_day"].astype(str).isin(g_sim)]
        else:
            sim_b = sim_sess[sim_sess["animal|seed"].astype(str).isin(g_sim)]
        dl = d_lab_once(lab_b)
        ds = d_sim_once(sim_b, lab_b)
        if np.isfinite(dl) and dl > 0 and np.isfinite(ds):
            boot.append(ds / dl)
    boot_a = np.asarray(boot, dtype=float)
    lo = float(np.quantile(boot_a, 0.05)) if boot_a.size else float("nan")
    hi = float(np.quantile(boot_a, 0.95)) if boot_a.size else float("nan")

    return {
        "metric": metric,
        "status": "ok",
        "d_lab": d_lab,
        "d_sim": d_sim,
        "d_simsim": d_ss,
        "R": R,
        "R_lo90": lo,
        "R_hi90": hi,
        "direction": direction,
        "n_lab_sessions": int(len(lab_sess)),
        "n_sim_sessions": int(len(sim_sess)),
        "n_lab_days": len(groups_lab),
        "n_sim_groups": len(groups_sim),
        "bootstrap_group": "recording_day",
        "sim_dataset": sim_dataset,
        "flag_excludes_1": bool(np.isfinite(lo) and np.isfinite(hi) and (hi < 1 or lo > 1)),
        "too_regular": bool(np.isfinite(d_ss) and np.isfinite(d_lab) and d_ss < 0.5 * d_lab),
    }


def rank_divergences(
    rows: list[dict[str, Any]],
    *,
    exclude_metrics: frozenset[str] | None = None,
) -> pd.DataFrame:
    """Rank by R descending. ``exclude_metrics`` defaults to RANK_EXCLUDE_METRICS.

    Pass ``exclude_metrics=frozenset()`` to rank everything (including diagnostics).
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    excl = RANK_EXCLUDE_METRICS if exclude_metrics is None else exclude_metrics
    if excl and "metric" in df.columns:
        base = df["metric"].astype(str).str.split("@", n=1).str[0]
        df = df.loc[~base.isin(excl)].copy()
    return df.sort_values("R", ascending=False, na_position="last").reset_index(drop=True)
