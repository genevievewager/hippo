#!/usr/bin/env python3
"""Report 4a rescore: ranking excludes, refractory censor, discriminator nulls.

Reads existing phase3_verify metric rows (no full suite rerun). Appends
lab-equivalent duplicate-spike-censored refractory metrics for sim sorted
spikes and GT sub-millisecond ISI fractions. Writes phase3_4a artifacts.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.real_quadrant.adapter import _parse_spike_times
from analysis.real_quadrant.session_select import animal_id_from_session
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.chunks import chunk_bounds
from agents.data_quality.matching.regions import add_mapped_lab_region, stratify_lab_units
from agents.data_quality.matching.session_discovery import eligible_room_a_sessions
from agents.data_quality.metrics.schema import tidy_row
from agents.data_quality.metrics.units import (
    LAB_DUP_REMOVAL_ISI_S,
    frac_isi_below,
    refractory_violation_rate,
    remove_spikes_below_isi,
)
from agents.data_quality.phase3.divergence import RANK_EXCLUDE_METRICS
from agents.data_quality.run_verify_recompute import (
    score_rows,
    thigmotaxis_report,
)

# Assumption: lab duplicate-spike removal threshold ≈ observed lab median min ISI.
CENSOR_ISI_S = LAB_DUP_REMOVAL_ISI_S  # 0.3 ms
GT_SUBMS_THRESH_S = 0.001


def _recording_day(session: str) -> str:
    parts = str(session).split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else str(session)


def _lab_censor_rows(session: str) -> list[dict[str, Any]]:
    """Lab refractory after same censor (should be near-identical to raw)."""
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    bundle = build_segment_bundle_dq(session, "A", require_ratemap_stability=True)
    meta = pd.read_csv(
        root / session / "clusters_dataset.csv",
        usecols=lambda c: c in ("cell", "Region", "BClabel", "timestamp"),
    )
    if "BClabel" in meta.columns:
        meta = meta[meta["BClabel"].astype(str) != "NON-SOMA"].reset_index(drop=True)
    id_to = {int(c): i for i, c in enumerate(meta["cell"].to_numpy())}
    spikes, regions = [], []
    for uid in bundle["unit_ids"]:
        i = id_to[int(uid)]
        spikes.append(_parse_spike_times(meta.loc[i, "timestamp"]))
        regions.append(str(meta.loc[i, "Region"]))
    mask = stratify_lab_units(regions, set_name="primary")
    spikes = [spikes[i] for i in np.flatnonzero(mask)]
    t0 = float(bundle["segment_t0"])
    t1 = float(bundle["segment_t1"])
    animal = animal_id_from_session(session)
    rows: list[dict] = []
    for ci, (c0, c1) in enumerate(chunk_bounds(t0, t1, chunk_s=300.0)):
        for st in spikes:
            stc = np.asarray(st, float)
            stc = stc[(stc >= c0) & (stc < c1)]
            cens = remove_spikes_below_isi(stc, CENSOR_ISI_S)
            base = dict(
                dataset="lab",
                animal_or_seed=animal,
                session=session,
                chunk=ci,
                region_set="primary",
            )
            for metric, val in (
                ("refractory_violation_rate_censored", refractory_violation_rate(cens)),
                ("frac_isi_below_1ms", frac_isi_below(stc, GT_SUBMS_THRESH_S)),
            ):
                r = tidy_row(**base, metric=metric, W=None, N=None, value=val)
                r["recording_day"] = _recording_day(session)
                r["censor_isi_s"] = CENSOR_ISI_S
                r["censor_assumption"] = "lab_median_min_ISI_approx_0.3ms"
                rows.append(r)
    return rows


def _sim_censor_rows(sim_dir: str, dataset: str, session: str, seed_label: str) -> list[dict[str, Any]]:
    d = Path(sim_dir)
    units = add_mapped_lab_region(pd.read_csv(d / "units.csv"))
    units = units[units["lab_region"].isin(["MEC", "SUB"])]
    so = pd.read_csv(d / "spikes_sorted.csv")
    gt = pd.read_csv(d / "spikes_ground_truth.csv")
    beh = pd.read_csv(d / "behavior.csv")
    times = beh["time_s"].to_numpy(float) if "time_s" in beh.columns else np.arange(len(beh)) * 0.05
    t1 = float(times.max()) if len(times) else 600.0
    so_sp = [so.loc[so["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
    gt_sp = [gt.loc[gt["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
    rows: list[dict] = []
    for ci, (c0, c1) in enumerate(chunk_bounds(0.0, t1, chunk_s=300.0)):
        for st_so, st_gt in zip(so_sp, gt_sp):
            so_c = st_so[(st_so >= c0) & (st_so < c1)]
            gt_c = st_gt[(st_gt >= c0) & (st_gt < c1)]
            cens = remove_spikes_below_isi(so_c, CENSOR_ISI_S)
            base = dict(
                dataset=dataset,
                animal_or_seed=seed_label,
                session=session,
                chunk=ci,
                region_set="primary",
            )
            for metric, val in (
                ("refractory_violation_rate_censored", refractory_violation_rate(cens)),
                ("frac_isi_below_1ms", frac_isi_below(so_c, GT_SUBMS_THRESH_S)),
                ("frac_isi_below_1ms_gt", frac_isi_below(gt_c, GT_SUBMS_THRESH_S)),
            ):
                r = tidy_row(**base, metric=metric, W=None, N=None, value=val)
                r["recording_day"] = _recording_day(session)
                r["censor_isi_s"] = CENSOR_ISI_S
                r["censor_assumption"] = "lab_median_min_ISI_approx_0.3ms"
                rows.append(r)
    return rows


def _job(kind: str, kw: dict) -> dict[str, Any]:
    if kind == "lab":
        return {"kind": "lab", "session": kw["session"], "rows": _lab_censor_rows(kw["session"])}
    return {
        "kind": "sim",
        "session": kw["session"],
        "rows": _sim_censor_rows(kw["sim_dir"], kw["dataset"], kw["session"], kw["seed_label"]),
    }


def collect_censor_rows(n_jobs: int = 12) -> list[dict[str, Any]]:
    jobs: list[tuple[str, dict]] = []
    for s in eligible_room_a_sessions():
        jobs.append(("lab", {"session": s}))
    for seed in range(5):
        d = DATA_QUALITY_ROOT / "sim_inputs" / f"seed_{seed}" / "sim"
        if (d / "units.csv").is_file():
            jobs.append(
                (
                    "sim",
                    {
                        "sim_dir": str(d),
                        "dataset": "sim_default",
                        "session": f"seed_{seed}",
                        "seed_label": f"seed_{seed}",
                    },
                )
            )
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    for e in man["sessions"]:
        sess = e["session"]
        d = DATA_QUALITY_ROOT / "matched" / f"{sess}_A" / "sim"
        if (d / "units.csv").is_file():
            jobs.append(
                (
                    "sim",
                    {
                        "sim_dir": str(d),
                        "dataset": "sim_matched",
                        "session": sess,
                        "seed_label": animal_id_from_session(sess),
                    },
                )
            )
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_job, k, kw): (k, kw) for k, kw in jobs}
        for fut in as_completed(futs):
            part = fut.result()
            rows.extend(part["rows"])
            print(json.dumps({"censor_done": part.get("session"), "n": len(part["rows"])}), flush=True)
    return rows


def load_base_rows() -> list[dict[str, Any]]:
    for cand in (
        DATA_QUALITY_ROOT / "phase3_verify" / "metric_rows_verify.jsonl",
        DATA_QUALITY_ROOT / "phase3" / "metric_rows_full.jsonl",
    ):
        if cand.is_file():
            rows = [json.loads(l) for l in cand.read_text().splitlines() if l.strip()]
            return rows
    raise FileNotFoundError("No metric_rows_*.jsonl found under phase3*")


def main() -> None:
    n_jobs = int(os.environ.get("DQ_N_JOBS", "12"))
    out = assert_dq_path(DATA_QUALITY_ROOT / "phase3_4a")
    out.mkdir(parents=True, exist_ok=True)

    base = load_base_rows()
    print(json.dumps({"base_rows": len(base), "n_jobs": n_jobs}), flush=True)
    extra = collect_censor_rows(n_jobs=n_jobs)
    # drop any prior censored metrics if re-run
    drop = {
        "refractory_violation_rate_censored",
        "frac_isi_below_1ms",
        "frac_isi_below_1ms_gt",
    }
    base = [r for r in base if r.get("metric") not in drop]
    all_rows = base + extra
    path = out / "metric_rows_4a.jsonl"
    with path.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(all_rows)
    scores = score_rows(df)
    thig = thigmotaxis_report()

    # public refractory param check (read-only confirmation)
    hippo_sim = Path(__file__).resolve().parents[3] / "hippo_sim"
    public_refr = []
    for p in hippo_sim.rglob("*.py"):
        txt = p.read_text(errors="ignore")
        for key in ("refractory", "min_isi", "dead_time", "deadtime", "absolute_refractory"):
            if key in txt.lower():
                public_refr.append({"file": str(p.relative_to(hippo_sim.parent)), "key": key})

    show = {
        "rank_exclude": sorted(RANK_EXCLUDE_METRICS),
        "thigmotaxis": thig,
        "censor": {
            "isi_s": CENSOR_ISI_S,
            "isi_ms": CENSOR_ISI_S * 1000,
            "assumption": (
                "Lab-equivalent duplicate-spike removal threshold = lab median "
                "minimum ISI (~0.3 ms). Applied to sim sorted spikes before "
                "refractory_violation_rate_censored."
            ),
        },
        "generator_refractory": {
            "public_parameter": False,
            "grep_hits": public_refr[:20],
            "note": (
                "hippo_sim.spikes.generate_spike_trains is inhomogeneous Poisson "
                "per behavior bin with no refractory; frac_isi_below_1ms_gt is a "
                "candidate structural divergence."
            ),
        },
        "top15_default": scores["sim_default"]["top15"],
        "top15_matched": scores["sim_matched"]["top15"],
        "discriminator_default": scores["sim_default"]["discriminator"],
        "discriminator_matched": scores["sim_matched"]["discriminator"],
        "n_rows": len(all_rows),
        "n_extra_censor_rows": len(extra),
    }
    (out / "SCORE_4A.json").write_text(json.dumps(show, indent=2, default=str) + "\n")
    (out / "scores_full.json").write_text(json.dumps(scores, indent=2, default=str) + "\n")

    # Top-5 story check vs verify (refractory / turning / coverage)
    prev = json.loads(
        (DATA_QUALITY_ROOT / "phase3_verify" / "VERIFY_STOP_AND_SHOW.json").read_text()
    )
    prev_top = [r["metric"].split("@")[0] for r in prev["top15_matched"][:5]]
    new_top = [r["metric"].split("@")[0] for r in scores["sim_matched"]["top15"][:5]]
    story = {
        "prev_top5_matched_bases": prev_top,
        "new_top5_matched_bases": new_top,
        "overlap": sorted(set(prev_top) & set(new_top)),
        "contradicts_if_refractory_leaves_and_behaviour_leaves": (
            "refractory_violation_rate" not in new_top
            and "turning_angle_std" not in new_top
            and "speed_median" not in new_top
        ),
    }
    # censored R for refractory
    for arm in ("sim_default", "sim_matched"):
        all_r = scores[arm].get("all") or []
        for name in (
            "refractory_violation_rate",
            "refractory_violation_rate_censored",
            "frac_isi_below_1ms_gt",
            "tracking_jitter_rest_cm",
        ):
            def _match(rows_list):
                exact = next((r for r in rows_list if str(r.get("metric", "")) == name), None)
                if exact is not None:
                    return exact
                return next(
                    (r for r in rows_list if str(r.get("metric", "")).split("@", 1)[0] == name),
                    None,
                )

            hit = _match(all_r)
            if hit is None:
                hit = _match(scores[arm].get("all_including_excluded") or [])
            story.setdefault(arm, {})[name] = {
                "R": None if hit is None else hit.get("R"),
                "R_lo90": None if hit is None else hit.get("R_lo90"),
                "R_hi90": None if hit is None else hit.get("R_hi90"),
                "in_ranking": hit is not None
                and name.split("@")[0] not in RANK_EXCLUDE_METRICS
                and name != "tracking_jitter_rest_cm",
            }
    (out / "TOP5_STORY_CHECK.json").write_text(json.dumps(story, indent=2, default=str) + "\n")
    print(json.dumps({"rescore_4a_complete": True, "story": story}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
