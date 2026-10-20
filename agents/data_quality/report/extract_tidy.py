#!/usr/bin/env python3
"""Build tidy CSVs for Report 4a figures from phase3_4a artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.phase3.divergence import RANK_EXCLUDE_METRICS

ANIMAL_CODES = {
    "1120": "A",
    "1130": "B",
    "1140": "C",
    "1150": "D",
    "1170": "E",
    "7394": "F",
}

FAMILY = {
    "firing_rate_hz": "units",
    "isi_cv": "units",
    "isi_cv2": "units",
    "burst_fraction": "units",
    "refractory_violation_rate": "units",
    "refractory_violation_rate_censored": "units",
    "frac_isi_below_1ms": "units",
    "frac_isi_below_1ms_gt": "units",
    "rate_drift": "units",
    "fano_factor": "units",
    "refractory_contam_proxy_median": "sorting",
    "refractory_contam_gt_median": "sorting",
    "presence_ratio_median": "sorting",
    "sorted_over_gt_spike_count_median": "sorting",
    "spatial_info_median": "spatial",
    "encoding_bits_per_spike_median": "spatial",
    "sparsity_median": "spatial",
    "place_field_size_median": "spatial",
    "frac_place_tuned": "spatial",
    "frac_place_tuned_excess": "spatial",
    "frac_grid_pass": "spatial",
    "frac_border_pass": "spatial",
    "frac_hd_pass": "spatial",
    "frac_speed_pass": "spatial",
    "peak_rate_median": "spatial",
    "grid_score_median": "spatial",
    "border_score_median": "spatial",
    "hd_mean_vector_length_median": "spatial",
    "speed_tuning_abs_r_median": "spatial",
    "theta_mod_index_median": "rhythm",
    "theta_freq_median": "rhythm",
    "noise_corr_mean": "population",
    "synchrony_frac": "population",
    "participation_ratio": "population",
    "mean_pairwise_corr": "population",
    "pca_var_frac_1": "population",
    "pca_var_frac_2": "population",
    "pca_var_frac_3": "population",
    "pca_var_frac_4": "population",
    "pca_var_frac_5": "population",
    "speed_mean": "behaviour",
    "speed_median": "behaviour",
    "accel_std": "behaviour",
    "occupancy_entropy": "behaviour",
    "wall_dist_mean_norm": "behaviour",
    "turning_angle_std": "behaviour",
    "dwell_time_mean_s": "behaviour",
    "tracking_jitter_rest_cm": "behaviour",
    "decode_error_cm": "decoding",
    "decode_floor_cm": "decoding",
    "decode_error_over_floor": "decoding",
}


def _base_metric(m: str) -> str:
    return str(m).split("@", 1)[0]


def _family(m: str) -> str:
    return FAMILY.get(_base_metric(m), "other")


def _anon_animal(animal: str, anon: bool) -> str:
    a = str(animal)
    if not anon:
        return a
    if a.startswith("seed_"):
        return a.replace("seed_", "S")
    return ANIMAL_CODES.get(a, a)


def load_scores(root: Path) -> dict:
    return json.loads((root / "SCORE_4A.json").read_text())


def _scores_full(src: Path) -> dict:
    """Prefer spatial-revision scores when present (4a_B)."""
    for name in ("scores_spatialrev.json", "scores_full.json"):
        p = src / name
        if p.is_file():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"no scores_*.json under {src}")


def write_tidy(out: Path, *, anon: bool = False) -> Path:
    out = assert_dq_path(out)
    out.mkdir(parents=True, exist_ok=True)
    src = DATA_QUALITY_ROOT / "phase3_4a"
    scores = load_scores(src)
    full = _scores_full(src)

    rows = []
    for arm_key, arm_label in (("sim_default", "default"), ("sim_matched", "matched")):
        block = full[arm_key]
        for r in block.get("all_including_excluded") or block.get("all") or []:
            m = str(r.get("metric", ""))
            base = _base_metric(m)
            rows.append(
                {
                    "arm": arm_label,
                    "metric": m,
                    "metric_base": base,
                    "family": _family(m),
                    "R": r.get("R"),
                    "R_lo90": r.get("R_lo90"),
                    "R_hi90": r.get("R_hi90"),
                    "d_lab": r.get("d_lab"),
                    "d_sim": r.get("d_sim"),
                    "d_simsim": r.get("d_simsim"),
                    "direction": r.get("direction"),
                    "flag_excludes_1": r.get("flag_excludes_1"),
                    "too_regular": r.get("too_regular"),
                    "n_lab_sessions": r.get("n_lab_sessions"),
                    "n_sim_sessions": r.get("n_sim_sessions"),
                    "in_ranking": base not in RANK_EXCLUDE_METRICS,
                }
            )
    div = pd.DataFrame(rows)
    div.to_csv(out / "data_divergence.csv", index=False)

    # session-level medians for unit / sorting / behaviour panels
    metric_path = src / "metric_rows_4a_spatialrev.jsonl"
    if not metric_path.is_file():
        metric_path = src / "metric_rows_4a.jsonl"
    want = {
        "firing_rate_hz",
        "isi_cv",
        "burst_fraction",
        "refractory_violation_rate",
        "refractory_violation_rate_censored",
        "frac_isi_below_1ms",
        "frac_isi_below_1ms_gt",
        "fano_factor",
        "rate_drift",
        "refractory_contam_proxy_median",
        "presence_ratio_median",
        "sorted_over_gt_spike_count_median",
        "speed_median",
        "turning_angle_std",
        "occupancy_entropy",
        "wall_dist_mean_norm",
        "dwell_time_mean_s",
        "tracking_jitter_rest_cm",
        "spatial_info_median",
        "sparsity_median",
        "place_field_size_median",
        "frac_place_tuned",
        "frac_border_pass",
        "frac_speed_pass",
        "frac_hd_pass",
        "frac_grid_pass",
        "peak_rate_median",
        "participation_ratio",
        "decode_error_over_floor",
        "theta_mod_index_median",
    }
    sess_rows = []
    with metric_path.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("metric") not in want:
                continue
            if r.get("metric") in {
                "fano_factor", "participation_ratio", "sparsity_median",
                "place_field_size_median", "decode_error_over_floor", "spatial_info_median",
            }:
                # prefer N=20 / W=0.25 when present
                if r.get("metric") == "fano_factor" and r.get("W") not in (None, 0.25):
                    continue
                if r.get("N") is not None and float(r["N"]) != 20.0 and r.get("metric") != "fano_factor":
                    # keep only N=20 for matched-N metrics when N present
                    try:
                        if abs(float(r["N"]) - 20.0) > 0.1:
                            continue
                    except Exception:
                        pass
            sess_rows.append(
                {
                    "dataset": r["dataset"],
                    "animal": _anon_animal(r.get("animal|seed", ""), anon),
                    "session": (
                        f"{_anon_animal(str(r.get('animal|seed', '')), anon)}_{str(r.get('session', '')).split('_', 1)[-1]}"
                        if anon and not str(r.get("session", "")).startswith("seed_")
                        else r.get("session")
                    ),
                    "chunk": r.get("chunk"),
                    "metric": r["metric"],
                    "W": r.get("W"),
                    "N": r.get("N"),
                    "value": r.get("value"),
                }
            )
    sess = pd.DataFrame(sess_rows)
    # collapse to session median
    gcols = ["dataset", "animal", "session", "metric"]
    sess_med = sess.groupby(gcols, as_index=False)["value"].median()
    sess_med.to_csv(out / "data_session_metrics.csv", index=False)

    # discriminator
    disc_rows = []
    for arm_key, arm_label in (("discriminator_default", "default"), ("discriminator_matched", "matched")):
        d = scores[arm_key]
        disc_rows.append(
            {
                "arm": arm_label,
                "auc_logit": d.get("auc_logit"),
                "auc_hgb": d.get("auc_hgb"),
                "auc_group_shuffle_null": d.get("auc_group_shuffle_null"),
                "auc_group_shuffle_null_stratified": d.get("auc_group_shuffle_null_stratified"),
                "n_rows": d.get("n_rows"),
            }
        )
    pd.DataFrame(disc_rows).to_csv(out / "data_discriminator.csv", index=False)

    imp_rows = []
    for arm_key, arm_label in (("discriminator_default", "default"), ("discriminator_matched", "matched")):
        for i, im in enumerate(scores[arm_key].get("importances") or []):
            imp_rows.append(
                {
                    "arm": arm_label,
                    "rank": i + 1,
                    "feature": im.get("feature"),
                    "auc_drop": im.get("auc_drop"),
                    "abs_coef": im.get("abs_coef"),
                }
            )
    pd.DataFrame(imp_rows).to_csv(out / "data_discriminator_importance.csv", index=False)

    abl_rows = []
    for arm_key, arm_label in (("discriminator_default", "default"), ("discriminator_matched", "matched")):
        for a in scores[arm_key].get("ablation") or []:
            abl_rows.append(
                {
                    "arm": arm_label,
                    "n_features": a.get("n_features"),
                    "auc": a.get("auc"),
                }
            )
    pd.DataFrame(abl_rows).to_csv(out / "data_discriminator_ablation.csv", index=False)

    # design / protocol meta
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    meta = {
        "n_lab_sessions": 47,
        "n_animals": 6,
        "animal_codes": ANIMAL_CODES if anon else {k: k for k in ANIMAL_CODES},
        "anon": anon,
        "n_default_seeds": 5,
        "n_matched_sessions": len(man.get("sessions") or []),
        "chunk_len_s": 300.0,
        "matched_N_preferred": 20,
        "matched_Ns": [10, 20, 40, "max"],
        "n_subsamples": 20,
        "pos_smooth_sigma_s": 0.1,
        "thigmotaxis": scores["thigmotaxis"]["value_used"],
        "thigmotaxis_fitted": scores["thigmotaxis"]["fitted"],
        "censor_isi_ms": scores["censor"]["isi_ms"],
        "censor_assumption": scores["censor"]["assumption"],
        "generator_has_public_refractory_param": scores["generator_refractory"]["public_parameter"],
        "rank_exclude": scores["rank_exclude"],
        "R_definition": "R = d_sim / d_lab; distances are lab-MAD-scaled absolute differences of session medians; bootstrap by recording_day",
        "region_set": "primary={MEC,SUB}",
        "decoding": "Ridge on raw causal counts; W=0.25 frozen",
    }
    (out / "data_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # findings numbers (from ranking only; never n_units_eligible / family=other dumps)
    findings = []
    ban_findings = {"n_units_eligible", "n_units"}
    for arm_label, arm_key in (("default", "sim_default"), ("matched", "sim_matched")):
        ranked = [
            r for r in (full[arm_key].get("all") or [])
            if _base_metric(r["metric"]) not in RANK_EXCLUDE_METRICS
            and _base_metric(r["metric"]) not in ban_findings
            and _family(r["metric"]) != "other"
        ]
        for i, r in enumerate(ranked[:15]):
            findings.append(
                {
                    "arm": arm_label,
                    "rank": i + 1,
                    "metric": r["metric"],
                    "family": _family(r["metric"]),
                    "R": r["R"],
                    "R_lo90": r["R_lo90"],
                    "R_hi90": r["R_hi90"],
                    "direction": r["direction"],
                    "d_simsim": r.get("d_simsim"),
                    "too_regular": r.get("too_regular"),
                }
            )
    pd.DataFrame(findings).to_csv(out / "data_findings.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--anon", action="store_true")
    args = ap.parse_args()
    path = write_tidy(Path(args.out), anon=args.anon)
    print(json.dumps({"tidy_written": str(path)}, indent=2))


if __name__ == "__main__":
    main()
