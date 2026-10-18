#!/usr/bin/env python3
"""Build old vs new table after exact _spike_positions keep-mask recompute."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
SNAP = _REPO / "outputs/data_quality/reports/report4a_v3/_spike_keep_old"
OUT = _REPO / "outputs/data_quality/reports/report4a_v3"
PHASE = _REPO / "outputs/data_quality/phase3_4a"
BEH = OUT / "behavior_info"


def _pick_score(rows, base):
    cands = [r for r in rows if str(r.get("metric", "")).split("@")[0] == base]
    if not cands:
        return None
    n20 = [r for r in cands if str(r["metric"]).endswith("@N20")]
    return (n20 or cands)[0]


def main():
    old_fig3 = pd.read_csv(SNAP / "data_fig3_divergence.csv")
    old_spatial = old_fig3[old_fig3.family == "spatial"].set_index("metric_base")

    scores = json.loads((PHASE / "scores_spatialrev.json").read_text())
    new_list = scores["sim_matched"].get("all_including_excluded") or scores["sim_matched"].get("all")
    # score_rows stores lab in each arm via d_lab / lab fields — also read session metrics
    # Prefer absolute medians from scored rows if present
    rows = []
    interest = [
        "spatial_info_median", "sparsity_median", "place_field_size_median", "peak_rate_median",
        "grid_score_median", "border_score_median",
        "hd_mean_vector_length_median", "speed_tuning_abs_r_median",
        "frac_place_tuned", "frac_place_tuned_excess",
        "frac_border_pass", "frac_border_excess",
        "frac_hd_pass", "frac_hd_excess",
        "frac_speed_pass", "frac_speed_excess",
        "frac_grid_pass", "frac_grid_excess",
        "rate_map_consistency_rel_err",
    ]
    # Get lab/sim medians from metric_rows if available
    mpath = PHASE / "metric_rows_4a_spatialrev.jsonl"
    sess = []
    with mpath.open() as f:
        for line in f:
            if line.strip():
                sess.append(json.loads(line))
    sdf = pd.DataFrame(sess)
    # primary grid20 only
    if "region_set" in sdf.columns:
        sdf = sdf[sdf["region_set"].astype(str).str.contains("grid20") | ~sdf["region_set"].astype(str).str.contains(":")]

    def med(ds, metric):
        sub = sdf[(sdf.dataset == ds) & (sdf.metric.astype(str).str.split("@").str[0] == metric)]
        if sub.empty:
            return float("nan")
        return float(sub["value"].median())

    for m in interest:
        o = old_spatial.loc[m] if m in old_spatial.index else None
        n = _pick_score(new_list, m)
        lab_old = float(o["lab_median"]) if o is not None else float("nan")
        mat_old = float(o["sim_matched_median"]) if o is not None else float("nan")
        r_old = float(o["R_matched"]) if o is not None else float("nan")
        lab_new = med("lab", m)
        mat_new = med("sim_matched", m)
        r_new = float(n["R"]) if n is not None else float("nan")
        rows.append({
            "family": "spatial",
            "metric": m,
            "lab_old": lab_old, "lab_new": lab_new,
            "matched_old": mat_old, "matched_new": mat_new,
            "R_old": r_old, "R_new": r_new,
        })

    # encoding / bits
    old_enc = json.loads((SNAP / "encoding_diag_summary.json").read_text())["table"]
    new_enc = json.loads((BEH / "encoding_diag_summary.json").read_text())["table"]
    for key, label in (
        ("bits_end_med_place", "bits_per_spike_place"),
        ("bits_end_med_all", "bits_per_spike_all"),
        ("half_map_r_place", "half_map_r_place"),
        ("half_map_r_all", "half_map_r_all"),
        ("frac_place", "frac_place_cohort"),
        ("frac_above_null_p95_prior", "frac_above_null_p95"),
    ):
        lo, mo = old_enc["lab"].get(key), old_enc["sim_matched"].get(key)
        ln, mn = new_enc["lab"].get(key), new_enc["sim_matched"].get(key)
        rows.append({
            "family": "encoding",
            "metric": label,
            "lab_old": lo, "lab_new": ln,
            "matched_old": mo, "matched_new": mn,
            "R_old": float("nan"), "R_new": float("nan"),
        })

    # drift / sep curve
    old_d = json.loads((SNAP / "drift_noise_summary.json").read_text())
    new_d = json.loads((BEH / "drift_noise_summary.json").read_text())

    def drift_row(label, getter_old, getter_new):
        lo, mo = getter_old(old_d)
        ln, mn = getter_new(new_d)
        rows.append({
            "family": "drift",
            "metric": label,
            "lab_old": lo, "lab_new": ln,
            "matched_old": mo, "matched_new": mn,
            "R_old": float("nan"), "R_new": float("nan"),
        })

    drift_row(
        "r_first_last",
        lambda d: (d["by_dataset"]["lab"]["r_first_last"]["median"],
                   d["by_dataset"]["sim_matched"]["r_first_last"]["median"]),
        lambda d: (d["by_dataset"]["lab"]["r_first_last"]["median"],
                   d["by_dataset"]["sim_matched"]["r_first_last"]["median"]),
    )
    drift_row(
        "r_odd_even_min",
        lambda d: (d["by_dataset"]["lab"]["r_odd_even_min"]["median"],
                   d["by_dataset"]["sim_matched"]["r_odd_even_min"]["median"]),
        lambda d: (d["by_dataset"]["lab"]["r_odd_even_min"]["median"],
                   d["by_dataset"]["sim_matched"]["r_odd_even_min"]["median"]),
    )
    drift_row(
        "r_odd_even_2s",
        lambda d: (d["by_dataset"]["lab"]["r_odd_even_2s"]["median"],
                   d["by_dataset"]["sim_matched"]["r_odd_even_2s"]["median"]),
        lambda d: (d["by_dataset"]["lab"]["r_odd_even_2s"]["median"],
                   d["by_dataset"]["sim_matched"]["r_odd_even_2s"]["median"]),
    )
    drift_row(
        "sep_slope",
        lambda d: (d.get("lab_slope"), d.get("matched_slope")),
        lambda d: (d.get("lab_slope"), d.get("matched_slope")),
    )
    for sep in ("1", "2", "3", "4", "5"):
        drift_row(
            f"sep_r_{sep}",
            lambda d, s=sep: (
                d["sep_curve"]["lab"]["by_sep"][s]["median"],
                d["sep_curve"]["sim_matched"]["by_sep"][s]["median"],
            ),
            lambda d, s=sep: (
                d["sep_curve"]["lab"]["by_sep"][s]["median"],
                d["sep_curve"]["sim_matched"]["by_sep"][s]["median"],
            ),
        )

    out = pd.DataFrame(rows)
    out["lab_delta"] = out["lab_new"] - out["lab_old"]
    out["matched_delta"] = out["matched_new"] - out["matched_old"]
    out["R_delta"] = out["R_new"] - out["R_old"]
    path = OUT / "behavior_info" / "OLD_VS_NEW_spike_keep.csv"
    out.to_csv(path, index=False)
    # also copy to review
    rev = Path("/home/gmwager/projects/hippo/tmp_r4_review/v3/behavior_info")
    rev.mkdir(parents=True, exist_ok=True)
    out.to_csv(rev / "OLD_VS_NEW_spike_keep.csv", index=False)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
