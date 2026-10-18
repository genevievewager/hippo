#!/usr/bin/env python3
"""Targeted recompute: revised spatial metrics + turning (speed>2); merge + rescore."""

from __future__ import annotations

import json
import os
import time
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
from agents.data_quality.metrics.behavior_metrics import behavior_metric_rows
from agents.data_quality.metrics.spatial import (
    MIN_OCC_S,
    SPEED_MIN_CM_S,
    SMOOTH_SIGMA_BINS,
    _rate_map,
    spatial_metric_rows,
)
from agents.data_quality.metrics.trajectory import (
    POS_SMOOTH_SIGMA_S,
    smooth_positions,
    speed_accel_from_pos,
)
from agents.data_quality.phase3.divergence import divergence_for_metric, rank_divergences
from agents.data_quality.run_verify_recompute import score_rows

def _is_replaced_metric(name: str) -> bool:
    """Drop prior spatial / turning rows (including scheme-tagged variants)."""
    base = str(name).split("@", 1)[0]
    stems = (
        "spatial_info_median", "encoding_bits_per_spike_median", "sparsity_median",
        "coverage_bins_mean", "place_field_size_median", "grid_score_median",
        "border_score_median", "frac_with_field", "peak_rate_median",
        "hd_mean_vector_length_median", "speed_tuning_abs_r_median",
        "frac_grid_pass", "frac_border_pass", "frac_hd_pass", "frac_speed_pass",
        "frac_grid_excess", "frac_border_excess", "frac_hd_excess", "frac_speed_excess",
        "frac_place_tuned", "frac_place_tuned_excess",
        "n_units_total", "n_units_eligible_spatial", "frac_units_eligible_spatial",
        "rate_map_consistency_rel_err", "turning_angle_std",
    )
    if base in stems:
        return True
    for s in stems:
        if base.startswith(s + "_"):
            return True
    return False


TURNING_METRICS = {"turning_angle_std"}


def _recording_day(session: str) -> str:
    parts = str(session).split("_")
    return f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else str(session)


def _lab_bundle(session: str):
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
    regions = [regions[i] for i in np.flatnonzero(mask)]
    times = np.asarray(bundle["decode_times"], float)
    y = np.asarray(bundle["y"], float)
    valid = np.asarray(bundle["target_valid"], bool)
    arena_hw = 0.5 * float(
        bundle.get("arena_width_cm")
        or (bundle["arena_cm"][0] if "arena_cm" in bundle else 100.0)
    )
    boundary_poly = None
    try:
        import yaml
        from agents.data_quality.metrics.boundary import boundary_polygon_xy
        cfg = yaml.safe_load((root / session / "config.yaml").read_text())
        boundary_poly = boundary_polygon_xy(cfg, "A")
    except Exception:
        boundary_poly = None
    return {
        "spikes": spikes,
        "regions": regions,
        "times": times,
        "y": y,
        "valid": valid,
        "t0": float(bundle["segment_t0"]),
        "t1": float(bundle["segment_t1"]),
        "arena_hw": arena_hw,
        "boundary_poly": boundary_poly,
        "hd": bundle.get("head_direction"),
    }


def _sim_bundle(sim_dir: Path):
    units = add_mapped_lab_region(pd.read_csv(sim_dir / "units.csv"))
    units = units[units["lab_region"].isin(["MEC", "SUB"])]
    so = pd.read_csv(sim_dir / "spikes_sorted.csv")
    beh = pd.read_csv(sim_dir / "behavior.csv")
    tcol = "time_s" if "time_s" in beh.columns else beh.columns[0]
    times = beh[tcol].to_numpy(float)
    if {"x_cm", "y_cm"} <= set(beh.columns):
        y = beh[["x_cm", "y_cm"]].to_numpy(float)
    else:
        y = beh.iloc[:, 1:3].to_numpy(float)
    spikes = [so.loc[so["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
    hd = beh["head_direction"].to_numpy(float) if "head_direction" in beh.columns else None
    arena_hw = 0.5 * float(np.ptp(y[:, 0])) if len(y) else 50.0
    return {
        "spikes": spikes,
        "regions": units["lab_region"].astype(str).tolist(),
        "times": times,
        "y": y,
        "valid": np.ones(len(times), bool),
        "t0": 0.0,
        "t1": float(times.max()) if len(times) else 600.0,
        "arena_hw": arena_hw,
        "boundary_poly": None,
        "hd": hd,
    }


def _visited_ok(y, times, valid, t0, t1, speed, edges_x, edges_y, speed_min=SPEED_MIN_CM_S):
    m = (times >= t0) & (times < t1) & valid
    if speed is not None:
        m &= np.asarray(speed, float) > float(speed_min)
    p = y[m]
    if len(p) < 10:
        return np.zeros((len(edges_x) - 1, len(edges_y) - 1), bool)
    dt = float(np.median(np.diff(times[m]))) if m.sum() > 1 else 0.05
    occ_thresh = max(1.0, MIN_OCC_S / max(dt, 1e-6))
    # train-only for occupancy match consistency with metrics
    n_train = int(0.8 * len(p))
    p_tr = p[:n_train]
    occ, _, _ = np.histogram2d(p_tr[:, 0], p_tr[:, 1], bins=[edges_x, edges_y])
    return occ >= occ_thresh


def _norm_region(r: str) -> str:
    rs = str(r).upper()
    if "MEC" in rs or "ENTO" in rs:
        return "MEC"
    if "SUB" in rs:
        return "SUB"
    return rs


def _elig_region_counts(spikes, regions, times, speed, valid, t0, t1) -> dict[str, int]:
    """Eligible unit counts per MEC/SUB in a chunk (train window, speed > 2)."""
    from agents.data_quality.metrics.spatial import unit_eligible, TRAIN_FRAC, SPEED_MIN_CM_S

    times = np.asarray(times, float)
    m = (times >= t0) & (times < t1) & np.asarray(valid, bool)
    if speed is not None:
        m &= np.asarray(speed, float) > SPEED_MIN_CM_S
    t = times[m]
    if t.size < 10:
        return {"MEC": 0, "SUB": 0}
    n_train = int(TRAIN_FRAC * t.size)
    t_lo, t_hi = float(t[0]), float(t[n_train - 1] if n_train else t[-1])
    counts = {"MEC": 0, "SUB": 0}
    for st, reg in zip(spikes, regions):
        if unit_eligible(st, t_lo, t_hi):
            rr = _norm_region(reg)
            if rr in counts:
                counts[rr] += 1
    return counts


def _path_fail_sessions() -> set[str]:
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    fails = set((man.get("path_containment_qc") or {}).get("fail_sessions") or [])
    for s in man.get("sessions") or []:
        pc = s.get("path_containment") or {}
        if pc.get("status") == "path_containment_fail" or pc.get("pass_qc") is False:
            fails.add(s["session"])
    return fails


def _chunk_rows(
    bundle,
    *,
    dataset,
    animal,
    session,
    shared_ok_by_scheme=None,
    edges_by_scheme=None,
    target_region_counts=None,
    use_path_visited_mask: bool = False,
) -> list[dict]:
    y = bundle["y"]
    times = bundle["times"]
    valid = bundle["valid"]
    y_s = smooth_positions(y, times, sigma_s=POS_SMOOTH_SIGMA_S)
    speed, _ = speed_accel_from_pos(y_s, times)
    hd = bundle["hd"]
    if hd is not None:
        hd = np.asarray(hd, float)
    regions = [_norm_region(r) for r in bundle.get("regions") or []]
    rows = []
    for ci, (c0, c1) in enumerate(chunk_bounds(bundle["t0"], bundle["t1"], chunk_s=300.0)):
        n_u = len(bundle["spikes"])
        matched = tuple(n for n in (10, 20, 40) if n <= n_u)
        if n_u >= 5 and n_u not in matched:
            matched = matched + (n_u,)
        # Per-chunk target counts: use provided (matched sim) or this bundle's own eligible (lab)
        tgt = target_region_counts
        if tgt is None and dataset == "lab" and regions:
            tgt = None  # lab uses all eligible; counts recorded for the paired sim
        spat = spatial_metric_rows(
            bundle["spikes"],
            y_s,
            times,
            dataset=dataset,
            animal_or_seed=animal,
            session=session,
            chunk=ci,
            region_set="primary",
            t0=c0,
            t1=c1,
            speed=speed,
            head_direction=hd,
            boundary_poly=None if use_path_visited_mask else bundle["boundary_poly"],
            matched_Ns=matched,
            n_subsamples=20,
            shared_ok_by_scheme=None,  # occ-matched dropped (grid-index across frames meaningless)
            edges_by_scheme=None,
            n_shuffles=100,
            do_shuffles=True,
            unit_regions=regions if regions else None,
            target_region_counts=tgt if dataset.startswith("sim") else None,
            use_path_visited_mask=use_path_visited_mask,
            rng=np.random.default_rng(abs(hash((session, ci, dataset))) % (2**31)),
        )
        m = (times >= c0) & (times < c1) & valid
        beh = behavior_metric_rows(
            y[m],
            times[m],
            dataset=dataset,
            animal_or_seed=animal,
            session=session,
            chunk=ci,
            region_set="primary",
            arena_half_width_cm=bundle["arena_hw"],
            already_smoothed=False,
        )
        # keep only turning from behavior (other behaviour rows unchanged in merge)
        beh = [r for r in beh if r["metric"] == "turning_angle_std"]
        for r in spat + beh:
            r["recording_day"] = _recording_day(session)
            r["pos_smooth_sigma_s"] = POS_SMOOTH_SIGMA_S
            r["spatial_revision"] = "4a_B"
            r["path_visited_mask"] = bool(use_path_visited_mask)
            r["spatial_params"] = {
                "primary": "grid20",
                "secondary": "bin3cm",
                "smooth_sigma_bins": SMOOTH_SIGMA_BINS,
                "min_occ_s": MIN_OCC_S,
                "speed_min_cm_s": SPEED_MIN_CM_S,
                "min_train_spikes": 100,
                "min_mean_rate_hz": 0.2,
                "n_shuffles": 100,
                "field_frac_of_peak": 0.2,
                "field_min_hz": 1.0,
                "field_min_bins": 4,
                "sparsity": "skaggs (sum p r)^2 / sum p r^2",
                "occmatched": "dropped",
            }
            rows.append(r)
    return rows


def _job(kind: str, kw: dict) -> dict[str, Any]:
    t0 = time.perf_counter()
    path_fails = _path_fail_sessions()
    if kind == "lab":
        b = _lab_bundle(kw["session"])
        rows = _chunk_rows(
            b, dataset="lab", animal=animal_id_from_session(kw["session"]), session=kw["session"],
            use_path_visited_mask=kw["session"] in path_fails,
        )
        return {"kind": "lab", "session": kw["session"], "rows": rows, "wall_s": time.perf_counter() - t0}
    if kind == "sim":
        b = _sim_bundle(Path(kw["sim_dir"]))
        rows = _chunk_rows(
            b, dataset=kw["dataset"], animal=kw["seed_label"], session=kw["session"],
        )
        return {"kind": "sim", "session": kw["session"], "dataset": kw["dataset"], "rows": rows, "wall_s": time.perf_counter() - t0}
    # matched pair (no occupancy-matched joint bins — coordinate frames differ)
    sess = kw["session"]
    lab = _lab_bundle(sess)
    sim = _sim_bundle(Path(kw["sim_dir"]))
    use_path = sess in path_fails

    def _prep(b):
        y_s = smooth_positions(b["y"], b["times"], sigma_s=POS_SMOOTH_SIGMA_S)
        spd, _ = speed_accel_from_pos(y_s, b["times"])
        return y_s, spd

    ly, lspd = _prep(lab)
    sy, sspd = _prep(sim)
    rows_lab, rows_sim = [], []
    sim_chunks = list(chunk_bounds(0.0, sim["t1"], chunk_s=300.0))
    for ci, (c0, c1) in enumerate(chunk_bounds(lab["t0"], lab["t1"], chunk_s=300.0)):
        sc0, sc1 = sim_chunks[min(ci, max(0, len(sim_chunks) - 1))]
        lab_regs = [_norm_region(r) for r in lab.get("regions") or []]
        lab_counts = _elig_region_counts(
            lab["spikes"], lab_regs, lab["times"], lspd, lab["valid"], c0, c1,
        )
        for b, y_s, spd, aa, bb, ds, animal in (
            (lab, ly, lspd, c0, c1, "lab", animal_id_from_session(sess)),
            (sim, sy, sspd, sc0, sc1, "sim_matched", animal_id_from_session(sess)),
        ):
            n_u = len(b["spikes"])
            matched = tuple(n for n in (10, 20, 40) if n <= n_u)
            if n_u >= 5 and n_u not in matched:
                matched = matched + (n_u,)
            regs = [_norm_region(r) for r in b.get("regions") or []]
            # Path mask only on lab when containment flagged; sim keeps its square.
            path_mask = bool(use_path and ds == "lab")
            spat = spatial_metric_rows(
                b["spikes"], y_s, b["times"], dataset=ds, animal_or_seed=animal, session=sess,
                chunk=ci, region_set="primary", t0=aa, t1=bb, speed=spd,
                head_direction=np.asarray(b["hd"], float) if b["hd"] is not None else None,
                boundary_poly=None if path_mask else b["boundary_poly"],
                matched_Ns=matched, n_subsamples=20,
                shared_ok_by_scheme=None, edges_by_scheme=None,
                n_shuffles=100, do_shuffles=True,
                unit_regions=regs if regs else None,
                target_region_counts=lab_counts if ds == "sim_matched" else None,
                use_path_visited_mask=path_mask,
                rng=np.random.default_rng(abs(hash((sess, ci, ds))) % (2**31)),
            )
            m = (b["times"] >= aa) & (b["times"] < bb) & b["valid"]
            beh = [
                r for r in behavior_metric_rows(
                    b["y"][m], b["times"][m], dataset=ds, animal_or_seed=animal, session=sess,
                    chunk=ci, region_set="primary", arena_half_width_cm=b["arena_hw"],
                )
                if r["metric"] == "turning_angle_std"
            ]
            dest = rows_lab if ds == "lab" else rows_sim
            for r in spat + beh:
                r["recording_day"] = _recording_day(sess)
                r["spatial_revision"] = "4a_B"
                r["path_visited_mask"] = path_mask
                dest.append(r)
    return {
        "kind": "pair",
        "session": sess,
        "rows_lab": rows_lab,
        "rows_sim": rows_sim,
        "wall_s": time.perf_counter() - t0,
        "path_visited_mask": use_path,
    }


def main() -> None:
    # Cap BLAS/OpenMP threads inside workers (avoid 100+ thread explosion).
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    n_jobs = int(os.environ.get("DQ_N_JOBS", "12"))
    out = assert_dq_path(DATA_QUALITY_ROOT / "phase3_4a")
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"main_start": True, "n_jobs": n_jobs}), flush=True)

    # Prefer previous spatialrev scores as "old" for this revision; else full suite.
    old_path = out / "scores_spatialrev.json"
    if not old_path.is_file():
        old_path = out / "scores_full.json"
    old_scores = json.loads(old_path.read_text())
    print(json.dumps({"old_scores_loaded": True, "from": str(old_path.name)}), flush=True)
    old_rows = []
    for arm in ("sim_default", "sim_matched"):
        for r in old_scores[arm].get("all_including_excluded") or old_scores[arm].get("all") or []:
            old_rows.append({"arm": arm.replace("sim_", ""), **{k: r.get(k) for k in (
                "metric", "R", "R_lo90", "R_hi90", "lab_median" if False else "d_lab", "d_sim"
            )}})
    # capture absolute medians from previous session metrics if available
    prev_div = DATA_QUALITY_ROOT / "reports" / "report4a_v2" / "data" / "data_fig3_divergence.csv"
    old_abs = pd.read_csv(prev_div) if prev_div.is_file() else None

    jobs: list[tuple[str, dict]] = []
    # Matched pairs cover all lab sessions in the MANIFEST; default sims separate.
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    manifested = {e["session"] for e in man["sessions"]}
    print(json.dumps({"manifest_n": len(manifested)}), flush=True)

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
            jobs.append(("pair", {"session": sess, "sim_dir": str(d)}))

    print(json.dumps({"recompute": True, "n_jobs": n_jobs, "n_tasks": len(jobs)}), flush=True)
    new_rows: list[dict] = []
    # Prefer spawn to avoid inheriting OpenMP thread pools from the parent.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as ex:
        futs = {ex.submit(_job, k, kw): (k, kw) for k, kw in jobs}
        for fut in as_completed(futs):
            k, kw = futs[fut]
            try:
                part = fut.result()
                if part["kind"] == "pair":
                    new_rows.extend(part["rows_lab"])
                    new_rows.extend(part["rows_sim"])
                    print(json.dumps({"pair_done": part["session"], "wall_s": round(part["wall_s"], 1),
                                      "n": len(part["rows_lab"]) + len(part["rows_sim"])}), flush=True)
                else:
                    new_rows.extend(part["rows"])
                    print(json.dumps({"done": part.get("session"), "dataset": part.get("dataset"),
                                      "wall_s": round(part["wall_s"], 1), "n": len(part["rows"])}), flush=True)
            except Exception as e:
                print(json.dumps({"fail": kw, "error": f"{type(e).__name__}: {e}"}), flush=True)

    # merge into base metric rows: drop old spatial/turning, keep rest
    base_path = out / "metric_rows_4a.jsonl"
    base = [json.loads(l) for l in base_path.read_text().splitlines() if l.strip()]
    kept = [r for r in base if not _is_replaced_metric(str(r.get("metric", "")))]
    merged = kept + new_rows
    path = out / "metric_rows_4a_spatialrev.jsonl"
    with path.open("w") as f:
        for r in merged:
            f.write(json.dumps(r, default=str) + "\n")

    df = pd.DataFrame(merged)
    scores = score_rows(df)
    (out / "scores_spatialrev.json").write_text(json.dumps(scores, indent=2, default=str) + "\n")

    # comparison table
    comps = []
    interest = [
        "place_field_size_median",
        "sparsity_median",
        "spatial_info_median",
        "grid_score_median", "border_score_median",
        "hd_mean_vector_length_median", "speed_tuning_abs_r_median",
        "frac_grid_pass", "frac_border_pass", "frac_hd_pass", "frac_speed_pass",
        "frac_grid_excess", "frac_border_excess", "frac_hd_excess", "frac_speed_excess",
        "frac_place_tuned", "frac_place_tuned_excess",
        "turning_angle_std", "peak_rate_median",
        "rate_map_consistency_rel_err",
    ]
    for arm in ("sim_default", "sim_matched"):
        old_list = old_scores[arm].get("all_including_excluded") or old_scores[arm].get("all") or []
        new_list = scores[arm].get("all_including_excluded") or scores[arm].get("all") or []

        def pick(rows, base, prefer_grid20=True):
            cands = [r for r in rows if str(r.get("metric", "")).split("@")[0] == base]
            if not cands:
                return None
            # prefer @N20 and region_set containing grid20 when present in metric string — metric may be base@N20
            n20 = [r for r in cands if str(r["metric"]).endswith("@N20")]
            return (n20 or cands)[0]

        for base in interest:
            o = pick(old_list, base)
            # for new, prefer grid20 region — encoded in region_set of rows, but divergence collapses by metric name only
            # so score_rows treats all region_sets of same metric together unless we filter before scoring.
            n = pick(new_list, base)
            comps.append({
                "arm": arm.replace("sim_", ""),
                "metric": base,
                "R_old": None if o is None else o.get("R"),
                "R_old_lo": None if o is None else o.get("R_lo90"),
                "R_old_hi": None if o is None else o.get("R_hi90"),
                "R_new": None if n is None else n.get("R"),
                "R_new_lo": None if n is None else n.get("R_lo90"),
                "R_new_hi": None if n is None else n.get("R_hi90"),
            })
    pd.DataFrame(comps).to_csv(out / "OLD_VS_NEW_R_spatial_turning.csv", index=False)

    # eligibility summary
    elig = df[df.metric == "n_units_eligible_spatial"].groupby("dataset")["value"].agg(["median", "sum", "count"])
    tot = df[df.metric == "n_units_total"].groupby("dataset")["value"].agg(["median", "sum"])
    summary = {
        "eligibility_median_per_chunk_session": elig.to_dict(),
        "n_units_total_median": tot.to_dict(),
        "n_new_rows": len(new_rows),
        "params": {
            "primary": "grid20",
            "smooth_sigma_bins": SMOOTH_SIGMA_BINS,
            "min_occ_s": MIN_OCC_S,
            "speed_min": SPEED_MIN_CM_S,
            "n_shuffles": 100,
        },
    }
    (out / "spatial_revision_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"recompute_complete": True, **summary}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
