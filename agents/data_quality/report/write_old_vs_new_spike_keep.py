#!/usr/bin/env python3
"""OLD_VS_NEW_spike_keep.csv for Report 4a v4 (pre- vs post-exact keep-mask)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
SNAP = _REPO / "outputs/data_quality/reports/report4a_v3/_spike_keep_old"
PHASE = _REPO / "outputs/data_quality/phase3_4a"
BEH = _REPO / "outputs/data_quality/reports/report4a_v3/behavior_info"
OUT = _REPO / "outputs/data_quality/reports/report4a_v4"

SPATIAL = [
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


def _pick(rows, base):
    cands = [r for r in rows if str(r.get("metric", "")).split("@")[0] == base]
    if not cands:
        return None
    n20 = [r for r in cands if str(r["metric"]).endswith("@N20")]
    return (n20 or cands)[0]


def _medians_from_rows(path: Path) -> dict:
    vals = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            m = str(r.get("metric", "")).split("@")[0]
            if m not in SPATIAL:
                continue
            rs = str(r.get("region_set", ""))
            if ":" in rs and "grid20" not in rs and "bin3cm" in rs:
                continue
            if "bin3cm" in rs:
                continue
            ds = r.get("dataset")
            if ds not in ("lab", "sim_matched", "sim_default"):
                continue
            v = r.get("value")
            if v is None or not np.isfinite(float(v)):
                continue
            vals[m][ds].append(float(v))
    out = {}
    for m in SPATIAL:
        out[m] = {
            ds: (float(np.median(vals[m][ds])) if vals[m][ds] else float("nan"))
            for ds in ("lab", "sim_matched", "sim_default")
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old_fig3 = pd.read_csv(SNAP / "data_fig3_divergence.csv")
    old_sp = old_fig3[old_fig3.family == "spatial"].set_index("metric_base")
    scores = json.loads((PHASE / "scores_spatialrev.json").read_text())
    new_med = _medians_from_rows(PHASE / "metric_rows_4a_spatialrev.jsonl")
    rows = []
    for m in SPATIAL:
        o = old_sp.loc[m] if m in old_sp.index else None
        rm = _pick(scores["sim_matched"].get("all_including_excluded") or scores["sim_matched"]["all"], m)
        rd = _pick(scores["sim_default"].get("all_including_excluded") or scores["sim_default"]["all"], m)
        med = new_med.get(m, {})
        rows.append({
            "family": "spatial",
            "metric": m,
            "lab_old": float(o["lab_median"]) if o is not None else float("nan"),
            "lab_new": med.get("lab", float("nan")),
            "matched_old": float(o["sim_matched_median"]) if o is not None else float("nan"),
            "matched_new": med.get("sim_matched", float("nan")),
            "default_old": float(o["sim_default_median"]) if o is not None and "sim_default_median" in o.index else float("nan"),
            "default_new": med.get("sim_default", float("nan")),
            "R_matched_old": float(o["R_matched"]) if o is not None else float("nan"),
            "R_matched_new": float(rm["R"]) if rm is not None else float("nan"),
            "R_default_old": float(o["R_default"]) if o is not None and pd.notna(o.get("R_default")) else float("nan"),
            "R_default_new": float(rd["R"]) if rd is not None else float("nan"),
        })

    # encoding + drift
    old_e = json.loads((SNAP / "encoding_diag_summary.json").read_text())["table"]
    new_e = json.loads((BEH / "encoding_diag_summary.json").read_text())["table"]
    for key, label in (
        ("bits_end_med_place", "bits_per_spike_place"),
        ("bits_end_med_all", "bits_per_spike_all"),
        ("half_map_r_place", "half_map_r_place"),
        ("half_map_r_all", "half_map_r_all"),
        ("frac_place", "frac_place_cohort"),
        ("frac_above_null_p95_prior", "frac_above_null_p95"),
    ):
        rows.append({
            "family": "encoding", "metric": label,
            "lab_old": old_e["lab"].get(key), "lab_new": new_e["lab"].get(key),
            "matched_old": old_e["sim_matched"].get(key), "matched_new": new_e["sim_matched"].get(key),
            "default_old": float("nan"), "default_new": float("nan"),
            "R_matched_old": float("nan"), "R_matched_new": float("nan"),
            "R_default_old": float("nan"), "R_default_new": float("nan"),
        })

    old_d = json.loads((SNAP / "drift_noise_summary.json").read_text())
    new_d = json.loads((BEH / "drift_noise_summary.json").read_text())

    def add_drift(label, ol, om, nl, nm):
        rows.append({
            "family": "drift", "metric": label,
            "lab_old": ol, "lab_new": nl, "matched_old": om, "matched_new": nm,
            "default_old": float("nan"), "default_new": float("nan"),
            "R_matched_old": float("nan"), "R_matched_new": float("nan"),
            "R_default_old": float("nan"), "R_default_new": float("nan"),
        })

    add_drift(
        "r_first_last",
        old_d["by_dataset"]["lab"]["r_first_last"]["median"],
        old_d["by_dataset"]["sim_matched"]["r_first_last"]["median"],
        new_d["by_dataset"]["lab"]["r_first_last"]["median"],
        new_d["by_dataset"]["sim_matched"]["r_first_last"]["median"],
    )
    add_drift(
        "r_odd_even_min",
        old_d["by_dataset"]["lab"]["r_odd_even_min"]["median"],
        old_d["by_dataset"]["sim_matched"]["r_odd_even_min"]["median"],
        new_d["by_dataset"]["lab"]["r_odd_even_min"]["median"],
        new_d["by_dataset"]["sim_matched"]["r_odd_even_min"]["median"],
    )
    add_drift(
        "r_odd_even_2s",
        old_d["by_dataset"]["lab"]["r_odd_even_2s"]["median"],
        old_d["by_dataset"]["sim_matched"]["r_odd_even_2s"]["median"],
        new_d["by_dataset"]["lab"]["r_odd_even_2s"]["median"],
        new_d["by_dataset"]["sim_matched"]["r_odd_even_2s"]["median"],
    )
    add_drift("sep_slope", old_d.get("lab_slope"), old_d.get("matched_slope"),
              new_d.get("lab_slope"), new_d.get("matched_slope"))
    for sep in ("1", "2", "3", "4", "5"):
        add_drift(
            f"sep_r_{sep}",
            old_d["sep_curve"]["lab"]["by_sep"][sep]["median"],
            old_d["sep_curve"]["sim_matched"]["by_sep"][sep]["median"],
            new_d["sep_curve"]["lab"]["by_sep"][sep]["median"],
            new_d["sep_curve"]["sim_matched"]["by_sep"][sep]["median"],
        )

    df = pd.DataFrame(rows)
    path = OUT / "OLD_VS_NEW_spike_keep.csv"
    df.to_csv(path, index=False)
    print(json.dumps({"wrote": str(path), "n_rows": len(df)}, indent=2))


if __name__ == "__main__":
    main()
