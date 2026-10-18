#!/usr/bin/env python3
"""Full SPEC §4 metric suite: lab eligible + default/matched sim arms.

Primary region set {MEC, SUB}, ratemap gate on for lab. Scores R (recording_day
bootstrap) + d_simsim + discriminator per arm.
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.real_quadrant.adapter import (
    RATEMAP_MIN_FRAC_ABOVE,
    _parse_spike_times,
    split_half_rate_map_stability,
)
from analysis.real_quadrant.session_select import animal_id_from_session
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.chunks import chunk_bounds
from agents.data_quality.matching.regions import add_mapped_lab_region, stratify_lab_units
from agents.data_quality.matching.session_discovery import eligible_room_a_sessions
from agents.data_quality.metrics.behavior_metrics import behavior_metric_rows
from agents.data_quality.metrics.decoding import decoding_metric_rows, frozen_decode_config
from agents.data_quality.metrics.guards import seconds_guard
from agents.data_quality.metrics.population import population_metric_rows
from agents.data_quality.metrics.rhythm import rhythm_metric_rows
from agents.data_quality.metrics.sorting_realism import sorting_metric_rows
from agents.data_quality.metrics.spatial import spatial_metric_rows
from agents.data_quality.metrics.units import unit_metric_rows
from agents.data_quality.phase3.discriminator import run_discriminator
from agents.data_quality.phase3.divergence import divergence_for_metric, rank_divergences
from agents.data_quality.run_phase3 import sim_ratemap_gate


def _animal(name: str) -> str:
    return animal_id_from_session(name)


def _recording_day(session: str) -> str:
    parts = session.split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else session


MATCHED_NS = (10, 20, 40)
N_SUBSAMPLES = 20


def _family_rows(
    spike_times,
    *,
    dataset,
    animal_or_seed,
    session,
    chunk,
    region_set,
    t0,
    t1,
    times,
    y,
    valid,
    arena_hw,
    spike_times_gt=None,
    head_direction=None,
    boundary_poly=None,
) -> list[dict]:
    """One chunk of SPEC §3.3 metrics (300 s chunks; matched-N where required)."""
    from agents.data_quality.metrics.trajectory import (
        POS_SMOOTH_SIGMA_S,
        smooth_positions,
        speed_accel_from_pos,
    )

    rows: list[dict] = []
    chunk_len = float(t1 - t0)
    # shared position smoothing before speed (lab and sim)
    y_s = smooth_positions(y, times, sigma_s=POS_SMOOTH_SIGMA_S)
    speed_s, _ = speed_accel_from_pos(y_s, times)
    m = (times >= t0) & (times < t1)
    if valid is not None:
        m = m & np.asarray(valid, bool)
    rows.extend(
        unit_metric_rows(
            spike_times, dataset=dataset, animal_or_seed=animal_or_seed, session=session,
            chunk=chunk, region_set=region_set, t0=t0, t1=t1,
        )
    )
    rows.extend(
        sorting_metric_rows(
            spike_times, dataset=dataset, animal_or_seed=animal_or_seed, session=session,
            chunk=chunk, region_set=region_set, t0=t0, t1=t1, spike_times_gt=spike_times_gt,
        )
    )
    n_u = len(spike_times)
    matched = tuple(n for n in MATCHED_NS if n <= n_u)
    if n_u >= 5 and n_u not in matched:
        matched = matched + (n_u,)
    rows.extend(
        spatial_metric_rows(
            spike_times, y_s[m], times[m], dataset=dataset, animal_or_seed=animal_or_seed,
            session=session, chunk=chunk, region_set=region_set, t0=t0, t1=t1,
            speed=speed_s[m],
            head_direction=head_direction[m] if head_direction is not None else None,
            boundary_poly=boundary_poly,
            matched_Ns=matched,
            n_subsamples=N_SUBSAMPLES,
        )
    )
    rows.extend(
        rhythm_metric_rows(
            spike_times, dataset=dataset, animal_or_seed=animal_or_seed, session=session,
            chunk=chunk, region_set=region_set,
        )
    )
    rows.extend(
        population_metric_rows(
            spike_times, dataset=dataset, animal_or_seed=animal_or_seed, session=session,
            chunk=chunk, region_set=region_set, t0=t0, t1=t1,
            positions=y_s, position_times=times,
            matched_Ns=matched, n_subsamples=N_SUBSAMPLES,
        )
    )
    rows.extend(
        behavior_metric_rows(
            y[m], times[m], dataset=dataset, animal_or_seed=animal_or_seed,
            session=session, chunk=chunk, region_set=region_set, arena_half_width_cm=arena_hw,
            already_smoothed=False,
        )
    )
    frozen = frozen_decode_config()
    rows.extend(
        decoding_metric_rows(
            spike_times, y_s, times, dataset=dataset, animal_or_seed=animal_or_seed,
            session=session, chunk=chunk, region_set=region_set, valid=valid,
            Ns=matched or (max(5, n_u),),
            W_grid=(frozen["window_s"],),
            n_subsamples=N_SUBSAMPLES,
        )
    )
    for r in rows:
        r["recording_day"] = _recording_day(session)
        r["chunk_len_s"] = chunk_len
        r["pos_smooth_sigma_s"] = POS_SMOOTH_SIGMA_S
        if r.get("W") is None and r["metric"] in {
            "participation_ratio", "synchrony_frac", "noise_corr_mean",
            "mean_pairwise_corr", "decode_error_cm", "decode_floor_cm", "decode_error_over_floor",
        }:
            pass  # W already set by family
    return rows


def lab_session_rows(session: str) -> dict[str, Any]:
    t_wall0 = time.perf_counter()
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
    seconds_guard(spikes, name=f"lab:{session}")
    mask = stratify_lab_units(regions, set_name="primary")
    spikes = [spikes[i] for i in np.flatnonzero(mask)]
    regions = [regions[i] for i in np.flatnonzero(mask)]
    times = np.asarray(bundle["decode_times"], float)
    y = np.asarray(bundle["y"], float)
    valid = np.asarray(bundle["target_valid"], bool)
    arena_hw = 0.5 * float(
        bundle.get("arena_width_cm")
        or (bundle["arena_cm"][0] if "arena_cm" in bundle else 100.0)
    )
    # lab boundary polygon in room-local cm (same frame as bundle y)
    boundary_poly = None
    try:
        import yaml
        from agents.data_quality.metrics.boundary import boundary_polygon_xy
        cfg = yaml.safe_load((Path(os.environ["HIPPO_DATA_ROOT"]) / session / "config.yaml").read_text())
        boundary_poly = boundary_polygon_xy(cfg, "A")
    except Exception:
        boundary_poly = None
    t0 = float(bundle["segment_t0"])
    t1 = float(bundle["segment_t1"])
    rows: list[dict] = []
    for ci, (c0, c1) in enumerate(chunk_bounds(t0, t1, chunk_s=300.0)):
        rows.extend(
            _family_rows(
                spikes, dataset="lab", animal_or_seed=_animal(session), session=session,
                chunk=ci, region_set="primary", t0=c0, t1=c1, times=times, y=y,
                valid=valid, arena_hw=arena_hw, boundary_poly=boundary_poly,
            )
        )
    return {
        "session": session,
        "dataset": "lab",
        "n_units": len(spikes),
        "n_rows": len(rows),
        "wall_s": time.perf_counter() - t_wall0,
        "rows": rows,
    }


def sim_session_rows(sim_dir: str, *, dataset: str, session: str, seed_label: str) -> dict[str, Any]:
    t_wall0 = time.perf_counter()
    sim_dir_p = Path(sim_dir)
    units = add_mapped_lab_region(pd.read_csv(sim_dir_p / "units.csv"))
    # primary = MEC+SUB (fall back to all mapped units if filter empties)
    primary = units[units["lab_region"].isin(["MEC", "SUB"])].reset_index(drop=True)
    if len(primary) == 0:
        primary = units.reset_index(drop=True)
    units = primary
    sp = pd.read_csv(sim_dir_p / "spikes_sorted.csv")
    gt = pd.read_csv(sim_dir_p / "spikes_ground_truth.csv") if (sim_dir_p / "spikes_ground_truth.csv").is_file() else None
    spikes = [sp.loc[sp["unit_id"] == uid, "spike_time_s"].to_numpy(float) for uid in units["unit_id"]]
    spikes_gt = None
    if gt is not None:
        spikes_gt = [gt.loc[gt["unit_id"] == uid, "spike_time_s"].to_numpy(float) for uid in units["unit_id"]]
    seconds_guard(spikes, name=f"{dataset}:{session}")
    beh = pd.read_csv(sim_dir_p / "behavior.csv")
    times = beh["time_s"].to_numpy(float)
    y = np.column_stack([beh["x_cm"].to_numpy(float), beh["y_cm"].to_numpy(float)])
    valid = np.isfinite(y).all(1)
    hd = None
    for col in ("head_direction", "head_direction_rad"):
        if col in beh.columns:
            hd = beh[col].to_numpy(float)
            break
    arena_size = 100.0
    if (sim_dir_p.parent / "pilot_report.json").is_file():
        arena_size = float(json.loads((sim_dir_p.parent / "pilot_report.json").read_text()).get("arena_size_cm", 100.0))
    elif (sim_dir_p / "summary.json").is_file():
        arena_size = float(json.loads((sim_dir_p / "summary.json").read_text()).get("arena_size_cm", 100.0) or 100.0)
    arena_hw = 0.5 * arena_size
    from agents.data_quality.metrics.boundary import square_polygon
    boundary_poly = square_polygon(arena_hw)
    t1 = float(times.max()) if len(times) else 600.0
    rows: list[dict] = []
    for ci, (c0, c1) in enumerate(chunk_bounds(0.0, t1, chunk_s=300.0)):
        rows.extend(
            _family_rows(
                spikes, dataset=dataset, animal_or_seed=seed_label, session=session,
                chunk=ci, region_set="primary", t0=c0, t1=c1, times=times, y=y,
                valid=valid, arena_hw=arena_hw, spike_times_gt=spikes_gt,
                head_direction=hd, boundary_poly=boundary_poly,
            )
        )
    return {
        "session": session,
        "dataset": dataset,
        "n_units": len(spikes),
        "n_rows": len(rows),
        "wall_s": time.perf_counter() - t_wall0,
        "rows": rows,
    }


def _run_suite_job(job: tuple[str, dict]) -> dict[str, Any]:
    """Module-level worker (must be picklable for ProcessPoolExecutor)."""
    kind, kw = job
    if kind == "lab":
        return lab_session_rows(kw["session"])
    if kind == "gate":
        d = Path(kw["sim_dir"])
        return {"kind": "gate", "arm": kw["arm"], "label": kw["label"], **sim_ratemap_gate(d)}
    return sim_session_rows(**kw)


def _score_arm(df: pd.DataFrame, *, sim_dataset: str) -> dict[str, Any]:
    if df.empty or "dataset" not in df.columns:
        return {
            "sim_dataset": sim_dataset,
            "n_metrics": 0,
            "top15": [],
            "discriminator": {"status": "empty"},
            "all": [],
        }
    sub = df[(df["dataset"] == "lab") | (df["dataset"] == sim_dataset)].copy()
    metrics = sorted(sub["metric"].unique())
    div_rows = []
    for m in metrics:
        div_rows.append(
            divergence_for_metric(
                sub, metric=m, log_rates=(m in {"firing_rate", "firing_rate_hz"}), sim_dataset=sim_dataset,
            )
        )
    ranked = rank_divergences(div_rows)
    disc = run_discriminator(sub)
    top15 = ranked.head(15).to_dict(orient="records") if len(ranked) else []
    return {
        "sim_dataset": sim_dataset,
        "n_metrics": len(metrics),
        "top15": top15,
        "discriminator": disc,
        "all": ranked.to_dict(orient="records") if len(ranked) else [],
    }


def pilot_and_project(n_jobs: int) -> dict[str, Any]:
    """Time one lab + one matched session; project full suite wall time."""
    eligible = eligible_room_a_sessions()
    lab_sess = eligible[0]
    matched = sorted((DATA_QUALITY_ROOT / "matched").glob("*_A/sim"))
    matched = [d for d in matched if (d / "units.csv").is_file()]
    # prefer a manifested session
    man = DATA_QUALITY_ROOT / "matched" / "MANIFEST.json"
    if man.is_file():
        sessions = {e["session"] for e in json.loads(man.read_text())["sessions"]}
        matched = [d for d in matched if d.parent.name.replace("_A", "") in sessions] or matched
    sim_dir = matched[0]
    sim_sess = sim_dir.parent.name.replace("_A", "")

    lab = lab_session_rows(lab_sess)
    sim = sim_session_rows(
        str(sim_dir), dataset="sim_matched", session=sim_sess, seed_label=_animal(sim_sess),
    )
    n_lab = len(eligible)
    n_matched = len(matched)
    n_default = sum(
        1 for s in range(5) if (DATA_QUALITY_ROOT / "sim_inputs" / f"seed_{s}" / "sim" / "units.csv").is_file()
    )
    # project: lab dominate; matched similar to sim pilot; default like matched
    total_serial_s = lab["wall_s"] * n_lab + sim["wall_s"] * (n_matched + n_default)
    wall_h = total_serial_s / max(n_jobs, 1) / 3600.0
    out = {
        "pilot_lab_session": lab_sess,
        "pilot_lab_wall_s": lab["wall_s"],
        "pilot_lab_n_rows": lab["n_rows"],
        "pilot_lab_n_units": lab["n_units"],
        "pilot_sim_session": sim_sess,
        "pilot_sim_wall_s": sim["wall_s"],
        "pilot_sim_n_rows": sim["n_rows"],
        "pilot_sim_n_units": sim["n_units"],
        "n_lab": n_lab,
        "n_matched": n_matched,
        "n_default": n_default,
        "n_jobs": n_jobs,
        "projected_serial_s": total_serial_s,
        "projected_wall_h": wall_h,
        "frozen_decode": frozen_decode_config(),
    }
    path = assert_dq_path(DATA_QUALITY_ROOT / "phase3" / "suite_pilot_projection.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    return out


def main() -> None:
    mode = os.environ.get("DQ_SUITE_MODE", "full")  # pilot | full
    # n_jobs from memory: ~5 GiB/worker peak → leave 100 GiB headroom
    mem_gib = float(os.environ.get("DQ_MEM_AVAIL_GIB") or 0)
    if mem_gib <= 0:
        # read MemAvailable
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                mem_gib = int(line.split()[1]) / (1024 ** 2)
                break
    n_jobs = int(os.environ.get("DQ_N_JOBS") or max(1, min(48, int((mem_gib - 100) / 5))))
    out_dir = assert_dq_path(DATA_QUALITY_ROOT / "phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "pilot":
        proj = pilot_and_project(n_jobs)
        print(json.dumps({"suite_pilot": True, **proj}, indent=2))
        return

    proj_path = out_dir / "suite_pilot_projection.json"
    if proj_path.is_file() and os.environ.get("DQ_REPILOT", "0") != "1":
        proj = json.loads(proj_path.read_text())
        print(json.dumps({"suite_pilot_cached": True, **{k: proj[k] for k in proj if k != "frozen_decode"}}, indent=2), flush=True)
    else:
        proj = pilot_and_project(n_jobs)
        print(json.dumps({"suite_pilot": True, **{k: proj[k] for k in proj if k != "frozen_decode"}}, indent=2), flush=True)

    eligible = eligible_room_a_sessions()
    jobs: list[tuple[str, dict]] = []
    for s in eligible:
        jobs.append(("lab", {"session": s}))
    for seed in range(5):
        d = DATA_QUALITY_ROOT / "sim_inputs" / f"seed_{seed}" / "sim"
        if (d / "units.csv").is_file():
            jobs.append(
                ("sim", {
                    "sim_dir": str(d),
                    "dataset": "sim_default",
                    "session": f"seed_{seed}",
                    "seed_label": f"seed_{seed}",
                })
            )
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    manifested = {e["session"] for e in man["sessions"]}
    for sess in sorted(manifested):
        d = DATA_QUALITY_ROOT / "matched" / f"{sess}_A" / "sim"
        if (d / "units.csv").is_file():
            jobs.append(
                ("sim", {
                    "sim_dir": str(d),
                    "dataset": "sim_matched",
                    "session": sess,
                    "seed_label": _animal(sess),
                })
            )

    # include sim ratemap-gate jobs in the same pool (avoid serial bottleneck)
    for seed in range(5):
        d = DATA_QUALITY_ROOT / "sim_inputs" / f"seed_{seed}" / "sim"
        if d.exists():
            jobs.append(("gate", {"sim_dir": str(d), "arm": "default", "label": f"seed_{seed}"}))
    for sess in sorted(manifested):
        d = DATA_QUALITY_ROOT / "matched" / f"{sess}_A" / "sim"
        if d.exists():
            jobs.append(("gate", {"sim_dir": str(d), "arm": "matched", "label": sess}))

    rows: list[dict] = []
    timings: list[dict] = []
    print(json.dumps({"suite_collect": True, "n_jobs": n_jobs, "n_tasks": len(jobs)}), flush=True)

    gate: dict[str, Any] = {"default": [], "matched": []}
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_run_suite_job, j): j for j in jobs}
        for fut in as_completed(futs):
            j = futs[fut]
            try:
                part = fut.result()
                if part.get("kind") == "gate":
                    arm = part["arm"]
                    entry = {k: v for k, v in part.items() if k not in {"kind", "arm"}}
                    if arm == "default":
                        entry["seed"] = part["label"]
                    else:
                        entry["session"] = part["label"]
                    gate[arm].append(entry)
                    print(json.dumps({"gate_done": entry.get("seed") or entry.get("session"), "fail": entry.get("fail")}), flush=True)
                    continue
                rows.extend(part["rows"])
                timings.append({k: part[k] for k in part if k != "rows"})
                print(json.dumps({"done": timings[-1]}), flush=True)
            except Exception as e:
                print(json.dumps({"fail": j, "error": f"{type(e).__name__}: {e}"}), flush=True)

    gate["default_n_fail"] = sum(1 for r in gate["default"] if r.get("fail"))
    gate["matched_n_fail"] = sum(1 for r in gate["matched"] if r.get("fail"))
    (out_dir / "sim_ratemap_gate.json").write_text(json.dumps(gate, indent=2) + "\n")

    rows_path = out_dir / "metric_rows_full.jsonl"
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (out_dir / "suite_timings.json").write_text(json.dumps(timings, indent=2) + "\n")

    df = pd.DataFrame(rows)
    scores = {
        "default": _score_arm(df, sim_dataset="sim_default"),
        "matched": _score_arm(df, sim_dataset="sim_matched"),
        "pilot_projection": proj,
        "n_rows": len(rows),
        "RATEMAP_MIN_FRAC_ABOVE": RATEMAP_MIN_FRAC_ABOVE,
        "region_set": "primary={MEC,SUB}",
        "ratemap_gate": "on (lab require_ratemap_stability=True)",
    }
    (out_dir / "score_summary.json").write_text(json.dumps(scores, indent=2, default=str) + "\n")
    # stop-and-show extract
    show = {
        "pilot_projection": proj,
        "top15_default": scores["default"]["top15"],
        "top15_matched": scores["matched"]["top15"],
        "discriminator_default": {
            "auc_logit": scores["default"]["discriminator"].get("auc_logit"),
            "auc_hgb": scores["default"]["discriminator"].get("auc_hgb"),
            "top_giveaways": (scores["default"]["discriminator"].get("importances") or [])[:8],
        },
        "discriminator_matched": {
            "auc_logit": scores["matched"]["discriminator"].get("auc_logit"),
            "auc_hgb": scores["matched"]["discriminator"].get("auc_hgb"),
            "top_giveaways": (scores["matched"]["discriminator"].get("importances") or [])[:8],
        },
    }
    (out_dir / "STOP_AND_SHOW.json").write_text(json.dumps(show, indent=2, default=str) + "\n")
    print(json.dumps({"suite_complete": True, "n_rows": len(rows)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
