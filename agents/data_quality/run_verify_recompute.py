#!/usr/bin/env python3
"""Pre-interpretation verification + revised suite score (no calibration)."""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.real_quadrant.adapter import _parse_spike_times
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.regions import add_mapped_lab_region, stratify_lab_units
from agents.data_quality.matching.session_discovery import eligible_room_a_sessions
from agents.data_quality.metrics.units import refractory_violation_rate
from agents.data_quality.metrics.trajectory import tracking_jitter_at_rest, POS_SMOOTH_SIGMA_S
from agents.data_quality.phase3.divergence import (
    RANK_EXCLUDE_METRICS,
    divergence_for_metric,
    rank_divergences,
)
from agents.data_quality.phase3.discriminator import run_discriminator
from agents.data_quality.run_metric_suite import (
    _animal,
    _run_suite_job,
    lab_session_rows,
    sim_session_rows,
)
from hippo_sim.config import SORTING_PARAMS, RECORDING_PARAMS


def _resources() -> dict[str, Any]:
    load = Path("/proc/loadavg").read_text().split()[:3]
    mem = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith(("MemTotal:", "MemAvailable:")):
            k, v, *_ = line.replace(":", "").split()
            mem[k] = int(v) / (1024 ** 2)
    # other users with >5% CPU
    import subprocess
    top = subprocess.check_output(
        ["ps", "-eo", "user,pcpu,pmem,comm", "--sort=-pcpu"], text=True
    ).splitlines()[1:40]
    heavy = []
    for line in top:
        parts = line.split(None, 3)
        if len(parts) >= 3 and float(parts[1]) >= 50:
            heavy.append({"user": parts[0], "pcpu": float(parts[1]), "cmd": parts[3] if len(parts) > 3 else ""})
    users = sorted({h["user"] for h in heavy})
    nproc = os.cpu_count() or 1
    # conservative n_jobs given load
    load1 = float(load[0])
    free_approx = max(0, nproc - int(load1) - 8)
    n_jobs = int(os.environ.get("DQ_N_JOBS") or max(4, min(12, free_approx // 2 or 4)))
    return {
        "loadavg": [float(x) for x in load],
        "nproc": nproc,
        "MemAvailable_GiB": mem.get("MemAvailable"),
        "MemTotal_GiB": mem.get("MemTotal"),
        "heavy_users": users,
        "n_heavy_procs": len(heavy),
        "recommended_n_jobs": n_jobs,
        "note": "other users (yaarit/stefano/eviatar) often hold load~50; do not use 48",
    }


def _isi_audit(spike_times: list[np.ndarray]) -> dict[str, float]:
    refr, min_isi, frac05 = [], [], []
    for st in spike_times:
        st = np.sort(np.asarray(st, float))
        d = np.diff(st)
        d = d[d > 0]
        if d.size == 0:
            continue
        refr.append(float(np.mean(d < 0.0015)))
        min_isi.append(float(np.min(d)))
        frac05.append(float(np.mean(d < 0.0005)))
    return {
        "refractory_violation_frac_median": float(np.nanmedian(refr)) if refr else float("nan"),
        "min_ISI_s_median": float(np.nanmedian(min_isi)) if min_isi else float("nan"),
        "min_ISI_s_p05": float(np.nanpercentile(min_isi, 5)) if min_isi else float("nan"),
        "frac_ISI_below_0.5ms_median": float(np.nanmedian(frac05)) if frac05 else float("nan"),
        "n_units": len(refr),
    }


def refractory_report() -> dict[str, Any]:
    """Item 1: GT vs sorted vs lab refractory; lab ISI censor check."""
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    eligible = eligible_room_a_sessions()[:8]  # sample for speed; full on matched later
    lab_audits = []
    for sess in eligible:
        try:
            bundle = build_segment_bundle_dq(sess, "A", require_ratemap_stability=False)
            meta = pd.read_csv(
                root / sess / "clusters_dataset.csv",
                usecols=lambda c: c in ("cell", "Region", "BClabel", "timestamp"),
            )
            if "BClabel" in meta.columns:
                meta = meta[meta["BClabel"].astype(str) != "NON-SOMA"].reset_index(drop=True)
            id_to = {int(c): i for i, c in enumerate(meta["cell"])}
            spikes, regions = [], []
            for uid in bundle["unit_ids"]:
                i = id_to[int(uid)]
                spikes.append(_parse_spike_times(meta.loc[i, "timestamp"]))
                regions.append(str(meta.loc[i, "Region"]))
            mask = stratify_lab_units(regions, set_name="primary")
            spikes = [spikes[i] for i in np.flatnonzero(mask)]
            a = _isi_audit(spikes)
            a["session"] = sess
            lab_audits.append(a)
        except Exception as e:
            lab_audits.append({"session": sess, "error": f"{type(e).__name__}: {e}"})

    def _pool(xs, key):
        vals = [x[key] for x in xs if key in x and np.isfinite(x.get(key, np.nan))]
        return float(np.median(vals)) if vals else float("nan")

    lab_summary = {
        k: _pool(lab_audits, k)
        for k in (
            "refractory_violation_frac_median",
            "min_ISI_s_median",
            "min_ISI_s_p05",
            "frac_ISI_below_0.5ms_median",
        )
    }
    # censor heuristic: almost no ISI < 0.5 ms and min ISI clustered near a hard floor
    lab_summary["likely_refractory_censor"] = bool(
        lab_summary["frac_ISI_below_0.5ms_median"] < 1e-4
        or (np.isfinite(lab_summary["min_ISI_s_median"]) and lab_summary["min_ISI_s_median"] >= 0.0005)
    )
    lab_summary["likely_duplicate_removal"] = bool(
        np.isfinite(lab_summary["min_ISI_s_median"]) and 0 < lab_summary["min_ISI_s_median"] < 0.001
        and lab_summary["frac_ISI_below_0.5ms_median"] < 0.001
    )

    # sim arms: sample manifested matched + default seed_0
    sim_arms = {"sim_default": [], "sim_matched": []}
    d0 = DATA_QUALITY_ROOT / "sim_inputs" / "seed_0" / "sim"
    if d0.exists():
        sim_arms["sim_default"].append(d0)
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    for e in man["sessions"][:8]:
        p = DATA_QUALITY_ROOT / "matched" / f"{e['session']}_A" / "sim"
        if p.exists():
            sim_arms["sim_matched"].append(p)

    arm_out = {}
    for arm, dirs in sim_arms.items():
        gt_a, so_a = [], []
        for d in dirs:
            units = add_mapped_lab_region(pd.read_csv(d / "units.csv"))
            units = units[units["lab_region"].isin(["MEC", "SUB"])]
            gt = pd.read_csv(d / "spikes_ground_truth.csv")
            so = pd.read_csv(d / "spikes_sorted.csv")
            gt_sp = [gt.loc[gt["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
            so_sp = [so.loc[so["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
            gt_a.append(_isi_audit(gt_sp))
            so_a.append(_isi_audit(so_sp))
        arm_out[arm] = {
            "ground_truth": {k: _pool(gt_a, k) for k in lab_summary if k.startswith(("refractory", "min", "frac"))},
            "sorted": {k: _pool(so_a, k) for k in lab_summary if k.startswith(("refractory", "min", "frac"))},
            "n_sessions_sampled": len(dirs),
        }

    return {
        "sorting_params_defaults": dict(SORTING_PARAMS),
        "recording_noise_std_uv": RECORDING_PARAMS.get("noise_std_uv"),
        "contamination_controls": {
            "miss_rate": SORTING_PARAMS["miss_rate"],
            "false_positive_rate": SORTING_PARAMS["false_positive_rate"],
            "contamination_rate": SORTING_PARAMS["contamination_rate"],
            "jitter_ms": SORTING_PARAMS["jitter_ms"],
            "merge_prob": SORTING_PARAMS["merge_prob"],
        },
        "lab": {"summary": lab_summary, "per_session_sample": lab_audits},
        "sim_arms": arm_out,
        "note": "refractory_violation = fraction of ISIs < 1.5 ms",
    }


def jitter_report() -> dict[str, Any]:
    """Lab tracking jitter at rest (raw positions)."""
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    vals = []
    for sess in eligible_room_a_sessions()[:12]:
        try:
            bundle = build_segment_bundle_dq(sess, "A", require_ratemap_stability=False)
            y = np.asarray(bundle["y"], float)
            t = np.asarray(bundle["decode_times"], float)
            v = np.asarray(bundle["target_valid"], bool)
            vals.append({"session": sess, "jitter_cm": tracking_jitter_at_rest(y[v], t[v])})
        except Exception as e:
            vals.append({"session": sess, "error": str(e)})
    js = [v["jitter_cm"] for v in vals if "jitter_cm" in v and np.isfinite(v["jitter_cm"])]
    return {
        "pos_smooth_sigma_s": POS_SMOOTH_SIGMA_S,
        "filter": "scipy.ndimage.gaussian_filter1d on x,y; σ=0.10 s (~2 samples @ 50 ms); identical for lab and sim",
        "lab_tracking_jitter_rest_cm_median": float(np.median(js)) if js else float("nan"),
        "lab_tracking_jitter_rest_cm_iqr": [
            float(np.percentile(js, 25)), float(np.percentile(js, 75))
        ] if js else None,
        "per_session_sample": vals,
    }


def thigmotaxis_report() -> dict[str, Any]:
    matched = DATA_QUALITY_ROOT / "matched"
    fits = set()
    for p in matched.glob("*_A/pilot_report.json"):
        fits.add(json.loads(p.read_text()).get("behavior_fit"))
    return {
        "fitted": False,
        "value_used": 0.8,
        "vs_default": "equal to Report 1 / SimConfig default 0.8",
        "pilot_behavior_fit_field": sorted(fits),
        "note": "matching did NOT fit thigmotaxis; run_matched_pilot hardcodes thigmotaxis=0.8 (behavior_fit pending). Do not fit yet.",
    }


def d_simsim_definition() -> dict[str, Any]:
    return {
        "definition": (
            "d_simsim is the median pairwise |z_i − z_j| across sim sessions from different "
            "recording_day groups, where z = (value − lab_median) is NOT used; instead each "
            "pair uses abs(value_i − value_j) / MAD_lab (same scale as d_lab and d_sim)."
        ),
        "is_ratio_to_d_lab": False,
        "units": "lab-MAD-scaled absolute distance (same as d_lab, d_sim)",
        "R": "R = d_sim / d_lab (ratio). d_simsim is NOT a ratio.",
        "value_gt_1_means": (
            "If d_simsim > 1, typical sim–sim differences exceed one lab MAD — "
            "sim sessions are more dispersed than the lab MAD scale. "
            "too_regular flag is the opposite: d_simsim < 0.5 × d_lab."
        ),
    }


def score_rows(df: pd.DataFrame) -> dict[str, Any]:
    out = {}
    for arm in ("sim_default", "sim_matched"):
        sub = df[(df["dataset"] == "lab") | (df["dataset"] == arm)].copy()
        metrics = sorted(sub["metric"].unique())
        # Prefer matched-N=20 rows when present for N-dependent metrics
        div = []
        for m in metrics:
            sm = sub[sub["metric"] == m]
            if "N" in sm.columns and sm["N"].notna().any():
                # score each N separately; keep N=20 as primary for ranking if available
                for N in sorted(sm["N"].dropna().unique()):
                    part = sm[sm["N"] == N]
                    # temporarily rewrite metric name for divergence grouping
                    part = part.assign(metric=f"{m}@N{int(N)}")
                    lab_part = sub[(sub["dataset"] == "lab") & (sub["metric"] == m) & (sub["N"] == N)]
                    if lab_part.empty:
                        continue
                    comb = pd.concat([lab_part.assign(metric=f"{m}@N{int(N)}"), part], ignore_index=True)
                    div.append(
                        divergence_for_metric(
                            comb, metric=f"{m}@N{int(N)}",
                            log_rates=(m in {"firing_rate", "firing_rate_hz"}),
                            sim_dataset=arm,
                        )
                    )
            else:
                div.append(
                    divergence_for_metric(
                        sub, metric=m, log_rates=(m in {"firing_rate", "firing_rate_hz"}), sim_dataset=arm,
                    )
                )
        # Ranking primary: one row per base metric; prefer @N20 when present.
        by_base: dict[str, list[dict]] = defaultdict(list)
        for r in div:
            by_base[str(r.get("metric", "")).split("@", 1)[0]].append(r)
        primary = []
        for base, sibs in by_base.items():
            n20 = next((x for x in sibs if str(x.get("metric", "")).endswith("@N20")), None)
            plain = next((x for x in sibs if "@" not in str(x.get("metric", ""))), None)
            primary.append(dict(n20 or plain or sibs[0]))
        ranked = rank_divergences(primary)
        disc = run_discriminator(sub)
        top = ranked.head(15).to_dict(orient="records") if len(ranked) else []
        for r in top:
            r["protocol"] = {
                "chunk_len_s": 300.0,
                "matched_N_preferred": 20,
                "matched_N_grid": [10, 20, 40, "max_common"],
                "n_subsamples": 20,
                "pos_smooth_sigma_s": POS_SMOOTH_SIGMA_S,
                "W_note": "population/decoding use W from grid / frozen 0.25; see metric name and W column in rows",
                "rank_exclude": sorted(RANK_EXCLUDE_METRICS),
            }
        out[arm] = {
            "top15": top,
            "all": ranked.to_dict(orient="records") if len(ranked) else [],
            "all_including_excluded": rank_divergences(div, exclude_metrics=frozenset()).to_dict(
                orient="records"
            )
            if div
            else [],
            "discriminator": disc,
            "n_metrics_scored": len(div),
        }
    return out


def run_suite(n_jobs: int) -> Path:
    out_dir = assert_dq_path(DATA_QUALITY_ROOT / "phase3_verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    eligible = eligible_room_a_sessions()
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    manifested = {e["session"] for e in man["sessions"]}
    jobs: list[tuple[str, dict]] = [("lab", {"session": s}) for s in eligible]
    for seed in range(5):
        d = DATA_QUALITY_ROOT / "sim_inputs" / f"seed_{seed}" / "sim"
        if (d / "units.csv").is_file():
            jobs.append(("sim", {
                "sim_dir": str(d), "dataset": "sim_default",
                "session": f"seed_{seed}", "seed_label": f"seed_{seed}",
            }))
    for sess in sorted(manifested):
        d = DATA_QUALITY_ROOT / "matched" / f"{sess}_A" / "sim"
        if (d / "units.csv").is_file():
            jobs.append(("sim", {
                "sim_dir": str(d), "dataset": "sim_matched",
                "session": sess, "seed_label": _animal(sess),
            }))
    print(json.dumps({"verify_suite": True, "n_jobs": n_jobs, "n_tasks": len(jobs)}), flush=True)
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        # only lab+sim metric jobs (no gate)
        futs = {ex.submit(_run_suite_job, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                part = fut.result()
                if part.get("kind") == "gate":
                    continue
                rows.extend(part["rows"])
                print(json.dumps({"done": {k: part[k] for k in part if k != "rows"}}), flush=True)
            except Exception as e:
                print(json.dumps({"fail": j, "error": f"{type(e).__name__}: {e}"}), flush=True)
    path = out_dir / "metric_rows_verify.jsonl"
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def proxy_fidelity_objective(n_jobs: int) -> dict[str, Any]:
    """Item 7: objective metrics only at matched N / 300 s (reuse existing proxies)."""
    obj_metrics = {
        # units
        "firing_rate_hz", "isi_cv", "isi_cv2", "burst_fraction", "refractory_violation_rate",
        "rate_drift", "fano_factor",
        # sorting
        "refractory_contam_proxy_median", "presence_ratio_median", "sorted_over_gt_spike_count_median",
        # spatial
        "spatial_info_median", "encoding_bits_per_spike_median", "sparsity_median",
        "coverage_bins_mean", "grid_score_median", "border_score_median",
        "hd_mean_vector_length_median", "speed_tuning_abs_r_median", "frac_with_field", "peak_rate_median",
    }
    proxy_root = DATA_QUALITY_ROOT / "calibration" / "proxy_fidelity"
    sessions = sorted([p.name for p in proxy_root.iterdir() if p.is_dir()]) if proxy_root.exists() else []
    rows = []
    for sess in sessions:
        sim_dir = proxy_root / sess / "sim"
        if not (sim_dir / "units.csv").is_file():
            continue
        part = sim_session_rows(str(sim_dir), dataset="sim_proxy", session=sess, seed_label=_animal(sess))
        rows.extend(part["rows"])
    # lab from verify file if present else phase3
    lab_rows = []
    for cand in (
        DATA_QUALITY_ROOT / "phase3_verify" / "metric_rows_verify.jsonl",
        DATA_QUALITY_ROOT / "phase3" / "metric_rows_full.jsonl",
    ):
        if cand.is_file():
            with cand.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("dataset") == "lab" and r.get("metric") in obj_metrics:
                        lab_rows.append(r)
            break
    matched_rows = []
    for cand in (
        DATA_QUALITY_ROOT / "phase3_verify" / "metric_rows_verify.jsonl",
        DATA_QUALITY_ROOT / "phase3" / "metric_rows_full.jsonl",
    ):
        if cand.is_file():
            with cand.open() as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("dataset") == "sim_matched" and r.get("metric") in obj_metrics:
                        matched_rows.append(r)
            break
    proxy_rows = [r for r in rows if r.get("metric") in obj_metrics]
    df_p = pd.DataFrame(lab_rows + proxy_rows)
    df_f = pd.DataFrame(lab_rows + matched_rows)
    # prefer N=20 when available
    comps = []
    n_agree = 0
    metrics = sorted(set(df_p["metric"].unique()) & set(df_f["metric"].unique()) & obj_metrics)
    for m in metrics:
        for N in (20, None):
            def _sub(df, dataset_prefix):
                s = df[df["metric"] == m]
                if N is not None and "N" in s.columns and s["N"].notna().any():
                    s = s[s["N"] == N]
                return s
            # build combined with sim_dataset labels
            sp = _sub(df_p, "proxy")
            sf = _sub(df_f, "full")
            if sp.empty or sf.empty:
                continue
            # rename datasets for divergence
            sp = sp.copy(); sf = sf.copy()
            # proxy rows already sim_proxy; full matched already sim_matched
            if (sp["dataset"] == "sim_proxy").sum() == 0:
                continue
            rp = divergence_for_metric(sp, metric=m, sim_dataset="sim_proxy", log_rates=(m == "firing_rate_hz"))
            rf = divergence_for_metric(sf, metric=m, sim_dataset="sim_matched", log_rates=(m == "firing_rate_hz"))

            def _f(x):
                try:
                    return float(x)
                except Exception:
                    return float("nan")
            Rp, Rf, lo, hi = _f(rp.get("R")), _f(rf.get("R")), _f(rf.get("R_lo90")), _f(rf.get("R_hi90"))
            agree = bool(np.isfinite(Rp) and np.isfinite(lo) and np.isfinite(hi) and lo <= Rp <= hi)
            n_agree += int(agree)
            comps.append({"metric": m, "N": N, "R_proxy": Rp, "R_full": Rf, "R_lo90": lo, "R_hi90": hi, "agree": agree})
            break  # one N per metric
    return {
        "objective_families": ["units", "sorting", "spatial"],
        "n_metrics": len(comps),
        "n_agree": n_agree,
        "frac_agree": float(n_agree / max(len(comps), 1)),
        "pass": bool(n_agree / max(len(comps), 1) >= 0.8),
        "comparisons": sorted(comps, key=lambda d: abs((d["R_proxy"] or 0) - (d["R_full"] or 0)), reverse=True),
        "n_proxy_sessions": len(sessions),
        "chunk_len_s": 300,
        "matched_N_preferred": 20,
    }


def time_fullsize_matched(n_jobs: int) -> dict[str, Any]:
    """Item 7: time one full-size matched session metric suite."""
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    sess = man["sessions"][0]["session"]
    d = DATA_QUALITY_ROOT / "matched" / f"{sess}_A" / "sim"
    t0 = time.perf_counter()
    part = sim_session_rows(str(d), dataset="sim_matched", session=sess, seed_label=_animal(sess))
    wall = time.perf_counter() - t0
    # budget from cheap calib
    morris, lhs, gp, val = 28, 60, 90, 18
    n_animals = 6
    # full-size: validation uses up to 3 sessions/animal; screening/LHS/GP if full-size would be:
    n_theta = morris + lhs + gp
    # if each θ needs 6 full-size gens (not done — proxy path preferred) vs validation only
    full_gen_est = float(json.loads((DATA_QUALITY_ROOT / "_phase0" / "pilot_longest_summary.json").read_text()).get("generation_wall_s", 1100))
    proxy_gen = float(json.loads((DATA_QUALITY_ROOT / "calibration" / "proxy_timing.json").read_text())["proxy_gen_wall_s"])
    # metric suite wall for one full matched session:
    metric_s = wall
    # calib cost comparison
    proxy_total_h = (n_theta * n_animals * proxy_gen) / n_jobs / 3600
    # full-size if used for all θ gens:
    full_all_h = (n_theta * n_animals * full_gen_est) / n_jobs / 3600
    # validation full-size only (current design):
    val_h = (val * full_gen_est) / n_jobs / 3600
    return {
        "session": sess,
        "metric_suite_wall_s": wall,
        "n_rows": part["n_rows"],
        "n_units": part["n_units"],
        "n_jobs_for_cost": n_jobs,
        "full_gen_wall_s_estimate": full_gen_est,
        "proxy_gen_wall_s": proxy_gen,
        "cost_comparison_h": {
            "proxy_screening_lhs_gp_h": proxy_total_h,
            "if_fullsize_all_theta_gens_h": full_all_h,
            "validation_fullsize_only_h": val_h,
            "cheap_design_total_proxy_plus_val_h": proxy_total_h + val_h,
        },
    }


def main() -> None:
    res = _resources()
    n_jobs = int(res["recommended_n_jobs"])
    print(json.dumps({"resources": res}, indent=2), flush=True)

    out = assert_dq_path(DATA_QUALITY_ROOT / "phase3_verify")
    out.mkdir(parents=True, exist_ok=True)

    refr = refractory_report()
    (out / "refractory_audit.json").write_text(json.dumps(refr, indent=2) + "\n")
    print(json.dumps({"refractory_done": True, "lab_min_ISI_med": refr["lab"]["summary"]["min_ISI_s_median"]}, indent=2), flush=True)

    jit = jitter_report()
    (out / "jitter_and_smooth.json").write_text(json.dumps(jit, indent=2) + "\n")
    thig = thigmotaxis_report()
    ddef = d_simsim_definition()

    # full-size timing (single session) before long suite
    full_t = time_fullsize_matched(n_jobs)
    (out / "fullsize_timing.json").write_text(json.dumps(full_t, indent=2) + "\n")
    print(json.dumps({"fullsize_timing": full_t}, indent=2), flush=True)

    rows_path = run_suite(n_jobs)
    # load + score
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    scores = score_rows(df)
    disc_default = scores["sim_default"]["discriminator"]
    disc_matched = scores["sim_matched"]["discriminator"]

    # proxy fidelity objective (may use prior proxies + new lab/matched rows)
    fid = proxy_fidelity_objective(n_jobs)
    (out / "proxy_fidelity_objective.json").write_text(json.dumps(fid, indent=2, default=str) + "\n")

    show = {
        "resources": res,
        "n_jobs_used": n_jobs,
        "refractory": {
            "sorting_defaults": refr["contamination_controls"],
            "lab": refr["lab"]["summary"],
            "sim_default": refr["sim_arms"].get("sim_default"),
            "sim_matched": refr["sim_arms"].get("sim_matched"),
        },
        "behaviour_filter": jit,
        "thigmotaxis": thig,
        "d_simsim": ddef,
        "discriminator_default": {
            "auc_logit": disc_default.get("auc_logit"),
            "auc_hgb": disc_default.get("auc_hgb"),
            "auc_group_shuffle_null": disc_default.get("auc_group_shuffle_null"),
            "ablation": disc_default.get("ablation"),
            "top_giveaways": (disc_default.get("importances") or [])[:8],
        },
        "discriminator_matched": {
            "auc_logit": disc_matched.get("auc_logit"),
            "auc_hgb": disc_matched.get("auc_hgb"),
            "auc_group_shuffle_null": disc_matched.get("auc_group_shuffle_null"),
            "ablation": disc_matched.get("ablation"),
            "top_giveaways": (disc_matched.get("importances") or [])[:8],
        },
        "top15_default": scores["sim_default"]["top15"],
        "top15_matched": scores["sim_matched"]["top15"],
        "proxy_fidelity_objective": {k: fid[k] for k in fid if k != "comparisons"},
        "proxy_fidelity_top_disagreements": fid["comparisons"][:10],
        "fullsize_cost": full_t,
        "protocol": {
            "chunk_len_s": 300,
            "matched_Ns": [10, 20, 40, "max_common"],
            "n_subsamples": 20,
            "pos_smooth_sigma_s": POS_SMOOTH_SIGMA_S,
            "reachable_bins": "lab boundary polygon; sim = equal-area square",
            "decoding": "Ridge on raw causal counts; W=0.25 frozen",
        },
    }
    (out / "VERIFY_STOP_AND_SHOW.json").write_text(json.dumps(show, indent=2, default=str) + "\n")
    print(json.dumps({"verify_complete": True, "n_rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
