#!/usr/bin/env python3
"""Phase 3: collect metric rows (lab all animals + default/matched sim) and score R.

Keeps n_jobs=2 while M3 runs. Primary lab eligibility = M3 gate on.
"""

from __future__ import annotations

import json
import os
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
from agents.data_quality.matching.regions import add_mapped_lab_region
from agents.data_quality.matching.session_discovery import (
    all_2room_sessions,
    eligibility_by_animal,
    eligible_room_a_sessions,
)
from agents.data_quality.metrics.guards import seconds_guard
from agents.data_quality.metrics.units import unit_metric_rows
from agents.data_quality.phase3.divergence import divergence_for_metric, rank_divergences
from agents.data_quality.phase3.discriminator import run_discriminator


def _animal(name: str) -> str:
    return animal_id_from_session(name)


def lab_eligibility(root: Path) -> dict[str, Any]:
    """Primary = M3 gate on (imported); secondary = all list_2room_sessions."""
    elig = eligibility_by_animal(root)
    all_sess = all_2room_sessions(root)
    primary = eligible_room_a_sessions(root)
    primary_fail = []
    for a in elig["animals"]:
        primary_fail.extend(a.get("excluded_sessions") or [])
    return {
        "primary_rule": "list_2room_sessions + _passes_room_a_exclusions (import)",
        "secondary_rule": "gate off — all list_2room_sessions",
        "n_2rooms": len(all_sess),
        "n_primary": len(primary),
        "n_secondary": len(all_sess),
        "primary_sessions": primary,
        "primary_fail": primary_fail,
        "by_animal": [
            {
                "animal": a["animal"],
                "total": a["total"],
                "eligible": a["eligible"],
                "excluded": a["excluded"],
                "exclude_reasons": a["exclude_reasons"],
            }
            for a in elig["animals"]
        ],
        "selection_affected_metrics": ["split_half_stability", "ratemap_stability"],
        "flag": "split-half / ratemap stability is selection-affected in primary comparison",
    }


def _sim_unit_spikes(sim_dir: Path, sorted_spikes: bool = True) -> tuple[pd.DataFrame, list[np.ndarray]]:
    units = add_mapped_lab_region(pd.read_csv(sim_dir / "units.csv"))
    sp = pd.read_csv(sim_dir / ("spikes_sorted.csv" if sorted_spikes else "spikes_ground_truth.csv"))
    spike_times = [
        sp.loc[sp["unit_id"] == uid, "spike_time_s"].to_numpy(float)
        for uid in units["unit_id"].to_numpy()
    ]
    return units, spike_times


def sim_ratemap_gate(sim_dir: Path, *, window_s: float = 0.25) -> dict[str, Any]:
    """Apply adapter split_half_rate_map_stability to a sim session (sorted spikes)."""
    from realtime.quadrant_n5_run import causal_train_test_split

    units, spike_times = _sim_unit_spikes(sim_dir, sorted_spikes=True)
    # behavior for positions
    beh = pd.read_csv(sim_dir / "behavior.csv")
    t = beh["time_s"].to_numpy(float)
    y = np.column_stack([beh["x_cm"].to_numpy(float), beh["y_cm"].to_numpy(float)])
    # count matrix on behavior grid
    X = np.zeros((len(t), len(spike_times)), dtype=float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.05
    for j, st in enumerate(spike_times):
        if st.size == 0:
            continue
        idx = np.clip(np.searchsorted(t, st) - 1, 0, len(t) - 1)
        np.add.at(X[:, j], idx, 1.0)
    X = X / max(dt, 1e-9)
    train_mask, _ = causal_train_test_split(t, 0.80, gap_s=max(window_s, 1.0))
    finite = np.isfinite(y).all(axis=1)
    mask = train_mask & finite
    stats = split_half_rate_map_stability(X, y, mask)
    frac = stats.get("frac_gt_thresh")
    frac_f = float(frac) if frac is not None and np.isfinite(frac) else 0.0
    fail = frac_f < float(RATEMAP_MIN_FRAC_ABOVE)
    return {
        "sim_dir": str(sim_dir),
        "frac_gt_thresh": frac_f,
        "fail": fail,
        **{k: stats.get(k) for k in ("n_scored_units", "n_stable_units", "median_corr")},
    }


def _lab_rows_for_session(session: str) -> list[dict]:
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    bundle = build_segment_bundle_dq(
        session, "A", require_ratemap_stability=False,  # secondary load; filter separately
    )
    meta = pd.read_csv(
        root / session / "clusters_dataset.csv",
        usecols=lambda c: c in ("cell", "Region", "BClabel", "timestamp"),
    )
    if "BClabel" in meta.columns:
        meta = meta[meta["BClabel"].astype(str) != "NON-SOMA"].reset_index(drop=True)
    else:
        meta = meta.reset_index(drop=True)
    id_to_row = {int(c): i for i, c in enumerate(meta["cell"].to_numpy())}
    spike_times, regions = [], []
    for uid in bundle["unit_ids"]:
        i = id_to_row[int(uid)]
        spike_times.append(_parse_spike_times(meta.loc[i, "timestamp"]))
        regions.append(str(meta.loc[i, "Region"]))
    seconds_guard(spike_times, name=f"lab:{session}")
    t0 = float(bundle["segment_t0"])
    t1 = float(bundle["segment_t1"])
    rows: list[dict] = []
    for ci, (c0, c1) in enumerate(chunk_bounds(t0, t1)):
        rows.extend(
            unit_metric_rows(
                spike_times,
                dataset="lab",
                animal_or_seed=_animal(session),
                session=session,
                chunk=ci,
                region_set="primary",
                t0=c0,
                t1=c1,
                regions=regions,
            )
        )
    return rows


def _sim_rows(sim_dir: Path, *, dataset: str, seed_label: str, session: str) -> list[dict]:
    units, spike_times = _sim_unit_spikes(sim_dir, sorted_spikes=True)
    seconds_guard(spike_times, name=f"{dataset}:{session}")
    # duration from behavior
    beh = pd.read_csv(sim_dir / "behavior.csv")
    t1 = float(beh["time_s"].max()) if len(beh) else 600.0
    rows: list[dict] = []
    for ci, (c0, c1) in enumerate(chunk_bounds(0.0, t1)):
        rows.extend(
            unit_metric_rows(
                spike_times,
                dataset=dataset,
                animal_or_seed=seed_label,
                session=session,
                chunk=ci,
                region_set="primary",
                t0=c0,
                t1=c1,
                regions=units["lab_region"].astype(str).tolist(),
            )
        )
    return rows


def main() -> None:
    n_jobs = int(os.environ.get("DQ_N_JOBS", "2"))
    root = Path(os.environ["HIPPO_DATA_ROOT"])
    out_dir = assert_dq_path(DATA_QUALITY_ROOT / "phase3")
    out_dir.mkdir(parents=True, exist_ok=True)

    elig = lab_eligibility(root)
    (out_dir / "lab_eligibility.json").write_text(json.dumps(elig, indent=2) + "\n")

    # Sim ratemap gate on default + matched arms
    sim_gate: dict[str, Any] = {"default": [], "matched": []}
    for seed in range(5):
        d = DATA_QUALITY_ROOT / "sim_inputs" / f"seed_{seed}" / "sim"
        if d.exists():
            sim_gate["default"].append({"seed": seed, **sim_ratemap_gate(d)})
    matched_root = DATA_QUALITY_ROOT / "matched"
    for d in sorted(matched_root.glob("*_A/sim")):
        if (d / "units.csv").exists():
            sim_gate["matched"].append({"session": d.parent.name, **sim_ratemap_gate(d)})
    sim_gate["default_n_fail"] = sum(1 for r in sim_gate["default"] if r.get("fail"))
    sim_gate["matched_n_fail"] = sum(1 for r in sim_gate["matched"] if r.get("fail"))
    (out_dir / "sim_ratemap_gate.json").write_text(json.dumps(sim_gate, indent=2) + "\n")

    # Collect lab rows for ALL 2rooms (secondary); tag primary eligibility
    primary_set = set(elig["primary_sessions"])
    lab_sessions = list(elig.get("secondary_sessions") or all_2room_sessions(root))
    # secondary_sessions may not be in elig — use all_2room
    lab_sessions = all_2room_sessions(root)
    rows: list[dict] = []
    print(json.dumps({"phase3_collect": True, "n_lab": len(lab_sessions), "n_jobs": n_jobs}), flush=True)

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futs = {ex.submit(_lab_rows_for_session, s): s for s in lab_sessions}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                part = fut.result()
                for r in part:
                    r["lab_eligibility"] = "primary" if s in primary_set else "secondary_only"
                rows.extend(part)
                print(json.dumps({"lab_done": s, "n_rows": len(part)}), flush=True)
            except Exception as e:
                print(json.dumps({"lab_fail": s, "error": f"{type(e).__name__}: {e}"}), flush=True)

    # Default sim arm (seeds 0–4)
    for seed in range(5):
        d = DATA_QUALITY_ROOT / "sim_inputs" / f"seed_{seed}" / "sim"
        if d.exists():
            rows.extend(_sim_rows(d, dataset="sim_default", seed_label=f"seed_{seed}", session=f"seed_{seed}"))

    # Matched arm
    for d in sorted(matched_root.glob("*_A/sim")):
        if (d / "units.csv").exists():
            sess = d.parent.name.replace("_A", "")
            rows.extend(
                _sim_rows(d, dataset="sim_matched", seed_label=_animal(sess), session=sess)
            )

    rows_path = out_dir / "metric_rows_units.jsonl"
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    df = pd.DataFrame(rows)
    # Primary comparison: lab primary-eligible only + sims
    if "lab_eligibility" in df.columns:
        df_primary = df[
            (df["dataset"].astype(str).str.startswith("sim"))
            | ((df["dataset"] == "lab") & (df["lab_eligibility"] == "primary"))
        ].copy()
    else:
        df_primary = df

    metrics = sorted(df_primary["metric"].unique())
    div_rows = []
    for m in metrics:
        div_rows.append(
            divergence_for_metric(
                df_primary,
                metric=m,
                log_rates=(m == "firing_rate"),
            )
        )
    ranked = rank_divergences(div_rows)
    ranked.to_json(out_dir / "divergence_units_primary.json", orient="records", indent=2)

    disc = run_discriminator(df_primary)
    (out_dir / "discriminator_primary.json").write_text(json.dumps(disc, indent=2) + "\n")

    summary = {
        "n_rows": len(rows),
        "n_lab_primary": elig["n_primary"],
        "n_lab_secondary": elig["n_secondary"],
        "sim_default_ratemap_fail": sim_gate["default_n_fail"],
        "sim_matched_ratemap_fail": sim_gate["matched_n_fail"],
        "top5_R": ranked.head(5).to_dict(orient="records") if len(ranked) else [],
        "discriminator_auc_logit": disc.get("auc_logit"),
        "selection_affected": elig["selection_affected_metrics"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"phase3_complete": True, **summary}, default=str), flush=True)


if __name__ == "__main__":
    main()
