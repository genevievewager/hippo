#!/usr/bin/env python3
"""Extract tidy + example rate maps / trajectories for Report 4a rebuild (review cut)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.real_quadrant.adapter import _parse_spike_times
from agents.data_quality.io.adapter_wrap import build_segment_bundle_dq
from agents.data_quality.io.paths import DATA_QUALITY_ROOT, assert_dq_path
from agents.data_quality.matching.regions import add_mapped_lab_region, stratify_lab_units
from agents.data_quality.metrics.spatial import (
    MIN_OCC_S,
    MIN_MEAN_RATE_HZ,
    MIN_TRAIN_SPIKES,
    SMOOTH_SIGMA_BINS as SPATIAL_SMOOTH_SIGMA,
    SPEED_MIN_CM_S,
    TRAIN_FRAC,
    _rate_map,
    _spike_positions,
    smooth_rate_map,
    spatial_info_bits_per_spike,
    unit_eligible,
)
from agents.data_quality.metrics.units import _isi
from agents.data_quality.phase3.divergence import RANK_EXCLUDE_METRICS

EXAMPLE_SESSION = "1150_09102024_2rooms"  # fixed; both MEC and SUB present (Fig4)
GRID_N = 20
BIN_CM = 3.0  # secondary / occupancy display
OCC_THRESH_S = MIN_OCC_S
SMOOTH_SIGMA_BINS = SPATIAL_SMOOTH_SIGMA
SCORES_PATH = DATA_QUALITY_ROOT / "phase3_4a" / "scores_spatialrev.json"
METRIC_ROWS_PATH = DATA_QUALITY_ROOT / "phase3_4a" / "metric_rows_4a_spatialrev.jsonl"


PLAIN = {
    "refractory_violation_rate": "Refractory violations (ISI < 1.5 ms)",
    "turning_angle_std": "Turning-angle SD (speed > 2 cm/s)",
    "place_field_size_median": "Field size among place-tuned (field / visited)",
    "sparsity_median": "Skaggs sparsity (Σpr)²/Σpr²",
    "spatial_info_median": "Spatial information",
    "peak_rate_median": "Peak rate (Hz)",
    "frac_place_tuned": "Place-tuned (frac. SI > shuffle 95%)",
    "frac_place_tuned_excess": "Place-tuned (excess over 0.05 chance)",
    "frac_grid_pass": "Grid cells (frac. > shuffle 95%)",
    "frac_border_pass": "Border cells (frac. > shuffle 95%)",
    "frac_hd_pass": "HD cells (frac. > shuffle 95%)",
    "frac_speed_pass": "Speed cells (frac. > shuffle 95%)",
    "frac_grid_excess": "Grid cells (excess over 0.05 chance)",
    "frac_border_excess": "Border cells (excess over chance)",
    "frac_hd_excess": "HD cells (excess over chance)",
    "frac_speed_excess": "Speed cells (excess over chance)",
    "grid_score_median": "Grid score (raw median)",
    "border_score_median": "Border score (raw median)",
    "hd_mean_vector_length_median": "HD vector length (raw)",
    "speed_tuning_abs_r_median": "Speed tuning |r| (raw)",
    "participation_ratio": "Participation ratio",
    "speed_median": "Speed (median)",
    "decode_error_over_floor": "Decode error / chance floor",
    "wall_dist_mean_norm": "Wall distance (norm.)",
    "occupancy_entropy": "Occupancy entropy",
    "firing_rate_hz": "Firing rate",
    "isi_cv": "ISI CV",
    "burst_fraction": "Burst fraction",
    "fano_factor": "Fano factor (W=0.25 s)",
    "rate_drift": "Within-session rate drift",
    "mean_pairwise_corr": "Pairwise correlation",
    "synchrony_frac": "Synchrony fraction",
    "dwell_time_mean_s": "Dwell time",
    "noise_corr_mean": "Noise correlation",
    "theta_freq_median": "Theta frequency",
    "presence_ratio_median": "Presence ratio",
}

# One row per concept for Fig3 (prefer @N20 / W=0.25 forms in ranking).
PRIMARY_CONCEPTS = [
    "refractory_violation_rate",
    "turning_angle_std",
    "place_field_size_median",
    "sparsity_median",
    "spatial_info_median",
    "peak_rate_median",
    "frac_place_tuned",
    "frac_place_tuned_excess",
    "frac_border_excess",
    "frac_hd_excess",
    "frac_grid_excess",
    "frac_grid_pass",
    "frac_border_pass",
    "frac_hd_pass",
    # frac_speed_* dropped from primary tuned-cell list (R≈1.26 after keep-mask; not a primary request)
    "participation_ratio",
    "speed_median",
    "decode_error_over_floor",
    "wall_dist_mean_norm",
    "occupancy_entropy",
    "firing_rate_hz",
    "isi_cv",
    "fano_factor",
    "burst_fraction",
    "rate_drift",
    "mean_pairwise_corr",
    "dwell_time_mean_s",
]

FAMILY = {
    "refractory_violation_rate": "units",
    "firing_rate_hz": "units",
    "isi_cv": "units",
    "burst_fraction": "units",
    "fano_factor": "units",
    "rate_drift": "units",
    "place_field_size_median": "spatial",
    "sparsity_median": "spatial",
    "spatial_info_median": "spatial",
    "peak_rate_median": "spatial",
    "frac_place_tuned": "spatial",
    "frac_place_tuned_excess": "spatial",
    "frac_grid_pass": "spatial",
    "frac_border_pass": "spatial",
    "frac_hd_pass": "spatial",
    "frac_speed_pass": "spatial",
    "frac_grid_excess": "spatial",
    "frac_border_excess": "spatial",
    "frac_hd_excess": "spatial",
    "frac_speed_excess": "spatial",
    "grid_score_median": "spatial",
    "border_score_median": "spatial",
    "hd_mean_vector_length_median": "spatial",
    "speed_tuning_abs_r_median": "spatial",
    "participation_ratio": "population",
    "mean_pairwise_corr": "population",
    "noise_corr_mean": "population",
    "synchrony_frac": "population",
    "speed_median": "behaviour",
    "turning_angle_std": "behaviour",
    "wall_dist_mean_norm": "behaviour",
    "occupancy_entropy": "behaviour",
    "dwell_time_mean_s": "behaviour",
    "decode_error_over_floor": "decoding",
    "theta_freq_median": "rhythm",
    "theta_mod_index_median": "rhythm",
    "presence_ratio_median": "sorting",
}


def _base(m: str) -> str:
    return str(m).split("@", 1)[0]


def _pick_primary_row(rows: list[dict], base: str) -> dict | None:
    cands = [r for r in rows if _base(r["metric"]) == base]
    if not cands:
        return None
    n20 = [r for r in cands if str(r["metric"]).endswith("@N20")]
    plain = [r for r in cands if "@" not in str(r["metric"])]
    return (n20 or plain or cands)[0]


def session_medians(metric_path: Path) -> pd.DataFrame:
    rows = []
    with metric_path.open() as f:
        for line in f:
            r = json.loads(line)
            m = r["metric"]
            if m in RANK_EXCLUDE_METRICS or m.endswith("_gt"):
                continue
            base = _base(m)
            # Prefer N=20 when matched-N applies; keep plain / occmatched / frac_* rows.
            n_sensitive = {
                "coverage_bins_mean", "sparsity_median", "participation_ratio",
                "decode_error_over_floor", "spatial_info_median", "grid_score_median",
                "mean_pairwise_corr", "noise_corr_mean", "hd_mean_vector_length_median",
                "border_score_median",
            }
            if any(base == s or base.startswith(s + "_") for s in n_sensitive) and r.get("N") not in (None, 20, 20.0):
                try:
                    if abs(float(r["N"]) - 20) > 0.1:
                        continue
                except Exception:
                    continue
            # Prefer primary grid20 region_set when tagged
            rs = str(r.get("region_set") or "")
            if "bin3cm" in rs and base in {
                "coverage_bins_mean", "sparsity_median", "spatial_info_median",
                "grid_score_median", "border_score_median", "frac_with_field",
                "peak_rate_median", "encoding_bits_per_spike_median",
            }:
                continue
            if m == "fano_factor" and r.get("W") not in (None, 0.25, 0.25):
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    g = df.groupby(["dataset", "metric", "session"], as_index=False)["value"].median()
    return g


def write_divergence_both_arms(out: Path) -> pd.DataFrame:
    scores_path = SCORES_PATH if SCORES_PATH.is_file() else DATA_QUALITY_ROOT / "phase3_4a" / "scores_full.json"
    rows_path = METRIC_ROWS_PATH if METRIC_ROWS_PATH.is_file() else DATA_QUALITY_ROOT / "phase3_4a" / "metric_rows_4a.jsonl"
    full = json.loads(scores_path.read_text())
    sess = session_medians(rows_path)
    abs_med = sess.groupby(["dataset", "metric"])["value"].median().unstack("dataset")

    rows = []
    for base in PRIMARY_CONCEPTS:
        rd = _pick_primary_row(full["sim_default"]["all"], base)
        rm = _pick_primary_row(full["sim_matched"]["all"], base)
        if rd is None and rm is None:
            continue
        lab_med = float(abs_med.loc[base, "lab"]) if base in abs_med.index and "lab" in abs_med.columns else float("nan")
        def_med = float(abs_med.loc[base, "sim_default"]) if base in abs_med.index and "sim_default" in abs_med.columns else float("nan")
        mat_med = float(abs_med.loc[base, "sim_matched"]) if base in abs_med.index and "sim_matched" in abs_med.columns else float("nan")
        rows.append({
            "metric_base": base,
            "label": PLAIN.get(base, base),
            "family": FAMILY.get(base, "other"),
            "R_default": None if rd is None else rd.get("R"),
            "R_default_lo": None if rd is None else rd.get("R_lo90"),
            "R_default_hi": None if rd is None else rd.get("R_hi90"),
            "R_matched": None if rm is None else rm.get("R"),
            "R_matched_lo": None if rm is None else rm.get("R_lo90"),
            "R_matched_hi": None if rm is None else rm.get("R_hi90"),
            "lab_median": lab_med,
            "sim_default_median": def_med,
            "sim_matched_median": mat_med,
            "direction_matched": None if rm is None else rm.get("direction"),
            "too_regular_matched": None if rm is None else rm.get("too_regular"),
            "metric_matched": None if rm is None else rm.get("metric"),
            "metric_default": None if rd is None else rd.get("metric"),
        })
    df = pd.DataFrame(rows)
    # sort by matched R desc within family order
    fam_order = {f: i for i, f in enumerate(["units", "sorting", "spatial", "rhythm", "population", "behaviour", "decoding", "other"])}
    df["_fo"] = df["family"].map(fam_order).fillna(99)
    df = df.sort_values(["_fo", "R_matched"], ascending=[True, False]).drop(columns="_fo")
    df.to_csv(out / "data_fig3_divergence.csv", index=False)

    # R <= 1 list for story
    good = df[df["R_matched"].notna() & (df["R_matched"] <= 1.0)].copy()
    good.to_csv(out / "data_sim_gets_right.csv", index=False)
    return df


def _occ_thresh_frames(dt: float) -> float:
    return max(1.0, OCC_THRESH_S / max(dt, 1e-6))


def _maps_for_spikes(spikes, regions, pos, times, valid, speed, region_name: str):
    """Train-only 20×20 rate maps; eligible units; unvisited bins = NaN (grey)."""
    from agents.data_quality.metrics.trajectory import speed_accel_from_pos

    times = np.asarray(times, float)
    pos = np.asarray(pos, float)
    valid_m = np.asarray(valid, bool) if valid is not None else np.ones(len(times), bool)
    if speed is None:
        spd, _ = speed_accel_from_pos(pos, times)
    else:
        spd = np.asarray(speed, float)
    # Contiguous valid parent; keep = first TRAIN_FRAC of moving frames
    t_win, p_win = times[valid_m], pos[valid_m]
    spd_win = spd[valid_m]
    if len(t_win) < 10:
        return []
    moving = np.isfinite(spd_win) & (spd_win > SPEED_MIN_CM_S)
    moving_idx = np.flatnonzero(moving)
    if moving_idx.size < 10:
        return []
    n_train = int(TRAIN_FRAC * moving_idx.size)
    keep = np.zeros(len(t_win), dtype=bool)
    keep[moving_idx[:n_train]] = True
    t_tr, p_tr = t_win[keep], p_win[keep]
    dt = float(np.median(np.diff(t_tr))) if len(t_tr) > 1 else 0.05
    occ_fr = _occ_thresh_frames(dt)
    xmin, ymin = p_tr.min(0)
    xmax, ymax = p_tr.max(0)
    edges_x = np.linspace(xmin, xmax, GRID_N + 1)
    edges_y = np.linspace(ymin, ymax, GRID_N + 1)
    t_lo, t_hi = float(t_tr[0]), float(t_tr[-1])
    cand = []
    for st, reg in zip(spikes, regions):
        if str(reg) != region_name:
            continue
        if not unit_eligible(st, t_lo, t_hi):
            continue
        st_a = np.asarray(st, float)
        spike_pos = _spike_positions(st_a, t_win, p_win, keep=keep)
        rate, occ, ok = _rate_map(
            p_tr, spike_pos, edges_x, edges_y, dt=dt, occ_thresh_frames=occ_fr,
        )
        rate_s = smooth_rate_map(rate, ok, sigma=SMOOTH_SIGMA_BINS)
        rate_out = np.full_like(rate_s, np.nan)
        rate_out[ok] = rate_s[ok]
        si = spatial_info_bits_per_spike(rate_s, occ, ok)
        if not np.isfinite(si):
            continue
        cand.append({
            "region": region_name,
            "spat_info": float(si),
            "rate": rate_out,
            "ok": ok.astype(bool),
            "extent": [float(edges_x[0]), float(edges_x[-1]), float(edges_y[0]), float(edges_y[-1])],
            "n_spikes": int(len(spike_pos)),
            "peak_hz": float(np.nanmax(rate_out)) if np.isfinite(rate_out).any() else float("nan"),
            "eligible": True,
        })
    if len(cand) < 3:
        return cand
    sis = np.array([c["spat_info"] for c in cand])
    out = []
    for pct in (25, 50, 75):
        target = np.percentile(sis, pct)
        j = int(np.argmin(np.abs(sis - target)))
        c = dict(cand[j])
        c["percentile"] = pct
        sis[j] = 1e9
        out.append(c)
    return out


def extract_rate_maps_and_traj(out: Path, session: str = EXAMPLE_SESSION) -> None:
    from agents.data_quality.metrics.trajectory import (
        POS_SMOOTH_SIGMA_S,
        smooth_positions,
        speed_accel_from_pos,
    )

    root = Path(os.environ["HIPPO_DATA_ROOT"])
    # LAB
    bundle = build_segment_bundle_dq(session, "A", require_ratemap_stability=True)
    meta = pd.read_csv(root / session / "clusters_dataset.csv",
                       usecols=lambda c: c in ("cell", "Region", "BClabel", "timestamp"))
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
    reg_map = []
    for r in regions:
        rs = str(r).upper()
        if "MEC" in rs or "ENTO" in rs:
            reg_map.append("MEC")
        elif "SUB" in rs:
            reg_map.append("SUB")
        else:
            reg_map.append(rs)

    times = np.asarray(bundle["decode_times"], float)
    y = np.asarray(bundle["y"], float)
    valid = np.asarray(bundle["target_valid"], bool)
    y_s = smooth_positions(y, times, sigma_s=POS_SMOOTH_SIGMA_S)
    lspd, _ = speed_accel_from_pos(y_s, times)
    lab_maps = []
    for reg in ("MEC", "SUB"):
        for m in _maps_for_spikes(spikes, reg_map, y_s, times, valid, lspd, reg):
            m["dataset"] = "lab"
            m["session"] = session
            lab_maps.append(m)

    # SIM matched
    sim_dir = DATA_QUALITY_ROOT / "matched" / f"{session}_A" / "sim"
    units = add_mapped_lab_region(pd.read_csv(sim_dir / "units.csv"))
    units = units[units["lab_region"].isin(["MEC", "SUB"])]
    so = pd.read_csv(sim_dir / "spikes_sorted.csv")
    gt = pd.read_csv(sim_dir / "spikes_ground_truth.csv")
    beh = pd.read_csv(sim_dir / "behavior.csv")
    tcol = "time_s" if "time_s" in beh.columns else beh.columns[0]
    stimes = beh[tcol].to_numpy(float)
    sy = beh[["x_cm", "y_cm"]].to_numpy(float) if {"x_cm", "y_cm"} <= set(beh.columns) else beh.iloc[:, 1:3].to_numpy(float)
    svalid = np.ones(len(stimes), bool)
    sy_s = smooth_positions(sy, stimes, sigma_s=POS_SMOOTH_SIGMA_S)
    sspd, _ = speed_accel_from_pos(sy_s, stimes)
    sspikes = [so.loc[so["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
    sregs = units["lab_region"].astype(str).tolist()
    sim_maps = []
    for reg in ("MEC", "SUB"):
        for m in _maps_for_spikes(sspikes, sregs, sy_s, stimes, svalid, sspd, reg):
            m["dataset"] = "sim_matched"
            m["session"] = session
            sim_maps.append(m)

    payload = []
    for m in lab_maps + sim_maps:
        payload.append({
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items() if k not in ("rate", "ok")},
            "rate": np.asarray(m["rate"], float).tolist(),
            "ok": np.asarray(m["ok"], bool).tolist(),
        })
    (out / "data_rate_maps.json").write_text(json.dumps(payload) + "\n")

    def _traj(times_, pos_, valid_, t0_prefer=None, dur=60.0):
        times_ = np.asarray(times_, float)
        pos_ = np.asarray(pos_, float)
        valid_ = np.asarray(valid_, bool)
        t0 = float(times_[0] + 120.0) if t0_prefer is None else float(t0_prefer)
        m = valid_ & (times_ >= t0) & (times_ < t0 + dur)
        return {
            "t0": t0,
            "dur": dur,
            "x": pos_[m, 0].tolist(),
            "y": pos_[m, 1].tolist(),
            "t": times_[m].tolist(),
        }

    # Shared spatial scale for lab and sim (same axis limits)
    lx0, lx1 = float(y[valid, 0].min()), float(y[valid, 0].max())
    ly0, ly1 = float(y[valid, 1].min()), float(y[valid, 1].max())
    sx0, sx1 = float(sy[:, 0].min()), float(sy[:, 0].max())
    sy0, sy1 = float(sy[:, 1].min()), float(sy[:, 1].max())
    # centre both on their own bbox but share half-width span
    def _half(a0, a1, b0, b1):
        span = max(a1 - a0, b1 - b0, 1.0) * 0.55
        acx, acy = 0.5 * (a0 + a1), 0.5 * ((ly0 if a0 == lx0 else sy0) + (ly1 if a1 == lx1 else sy1))
        return span

    span = max(lx1 - lx0, ly1 - ly0, sx1 - sx0, sy1 - sy0, 1.0) * 0.55
    lcx, lcy = 0.5 * (lx0 + lx1), 0.5 * (ly0 + ly1)
    scx, scy = 0.5 * (sx0 + sx1), 0.5 * (sy0 + sy1)
    arena = {
        "lab": {"xlim": [lcx - span, lcx + span], "ylim": [lcy - span, lcy + span],
                "outline": [[lx0, ly0], [lx1, ly0], [lx1, ly1], [lx0, ly1], [lx0, ly0]]},
        "sim": {"xlim": [scx - span, scx + span], "ylim": [scy - span, scy + span],
                "outline": [[sx0, sy0], [sx1, sy0], [sx1, sy1], [sx0, sy1], [sx0, sy0]]},
        "shared_half_span_cm": span,
    }
    # Prefer config boundary polygon for lab when available
    try:
        import yaml
        from agents.data_quality.metrics.boundary import boundary_polygon_xy
        cfg = yaml.safe_load((root / session / "config.yaml").read_text())
        poly = boundary_polygon_xy(cfg, "A")
        if poly is not None and len(poly) >= 3:
            closed = np.vstack([poly, poly[:1]])
            arena["lab"]["outline"] = closed.tolist()
    except Exception:
        pass

    traj = {
        "session": session,
        "lab": _traj(times, y, valid),
        "sim_matched": _traj(stimes, sy, svalid),
        "arena": arena,
        "selection": f"fixed example session {session}; 60 s starting at segment_t0+120 s",
    }
    (out / "data_trajectories_60s.json").write_text(json.dumps(traj) + "\n")

    def _isi_pool(spike_list, n_units=40):
        isis = []
        for st in spike_list[:n_units]:
            d = _isi(st)
            if d.size:
                isis.append(d)
        if not isis:
            return []
        return np.concatenate(isis).tolist()

    gt_sp = [gt.loc[gt["unit_id"] == u, "spike_time_s"].to_numpy(float) for u in units["unit_id"]]
    isi_payload = {
        "lab": _isi_pool(spikes, 80),
        "sim_sorted": _isi_pool(sspikes, 80),
        "sim_gt": _isi_pool(gt_sp, 80),
        "refractory_s": 0.0015,
        "censor_s": 0.0003,
        "note": "ISIs pooled across units (subsample); log-x histograms in Fig4b",
    }
    (out / "data_isi_pool.json").write_text(json.dumps(isi_payload) + "\n")

    def _occ_map(pos, times, valid, xlim, ylim):
        m = np.asarray(valid, bool)
        pos = pos[m]
        dt = float(np.median(np.diff(times[m]))) if m.sum() > 1 else 0.05
        ex = np.linspace(xlim[0], xlim[1], GRID_N + 1)
        ey = np.linspace(ylim[0], ylim[1], GRID_N + 1)
        occ, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=[ex, ey])
        return {
            "occ_s": (occ * dt).tolist(),
            "extent": [float(ex[0]), float(ex[-1]), float(ey[0]), float(ey[-1])],
            "units": "s",
        }

    (out / "data_occupancy.json").write_text(json.dumps({
        "lab": _occ_map(y, times, valid, arena["lab"]["xlim"], arena["lab"]["ylim"]),
        "sim_matched": _occ_map(sy, stimes, svalid, arena["sim"]["xlim"], arena["sim"]["ylim"]),
        "session": session,
        "arena": arena,
    }) + "\n")

    def _speed_turn(pos, times, valid):
        m = np.asarray(valid, bool)
        pos_s = smooth_positions(pos, times, sigma_s=POS_SMOOTH_SIGMA_S)
        spd, _ = speed_accel_from_pos(pos_s, times)
        d = np.diff(pos_s[m], axis=0)
        ang = np.arctan2(d[:, 1], d[:, 0])
        turn = np.diff(np.unwrap(ang))
        # turning only while flanking samples move > 2 cm/s
        spd_m = spd[m]
        if turn.size and spd_m.size >= 3:
            moving = spd_m[1:-1] > SPEED_MIN_CM_S
            n = min(moving.shape[0], turn.shape[0])
            turn = turn[:n][moving[:n]]
        return {
            "speed_cm_s": spd[m][np.isfinite(spd[m])].tolist(),
            "turning_rad": turn[np.isfinite(turn)].tolist(),
        }

    example_st = {
        "lab": _speed_turn(y, times, valid),
        "sim_matched": _speed_turn(sy, stimes, svalid),
        "session": session,
    }

    # Cohort session CDFs (quantiles) for Fig5 e–f: thin per session
    def _quantiles(arr, qs=np.linspace(0, 1, 41)):
        a = np.asarray(arr, float)
        a = a[np.isfinite(a)]
        if a.size < 5:
            return None
        return np.quantile(a, qs).tolist()

    cohort = {"lab": [], "sim_matched": [], "example_session": session, "qs": np.linspace(0, 1, 41).tolist()}
    man = json.loads((DATA_QUALITY_ROOT / "matched" / "MANIFEST.json").read_text())
    for entry in man["sessions"]:
        sess = entry["session"]
        try:
            if sess == session:
                lab_st = example_st["lab"]
                sim_st = example_st["sim_matched"]
            else:
                # light: use session-metric values from tidy for thin-line approx via
                # per-session empirical CDF of precomputed speed/turning from matched folder
                bdir = DATA_QUALITY_ROOT / "matched" / f"{sess}_A" / "sim"
                if not (bdir / "behavior.csv").is_file():
                    continue
                # lab bundle is expensive; use metric rows for cohort lines of session medians instead
                continue
            for key, st in (("lab", lab_st), ("sim_matched", sim_st)):
                sq = _quantiles(st["speed_cm_s"])
                tq = _quantiles(np.abs(st["turning_rad"]))
                if sq is not None and tq is not None:
                    cohort[key].append({
                        "session": sess,
                        "speed_q": sq,
                        "turn_q": tq,
                        "example": sess == session,
                    })
        except Exception:
            continue

    # Fill cohort from session metric CDFs when raw per-session is too heavy:
    # store all session medians of speed_median / turning_angle_std for rain/CDF of scalars,
    # PLUS raw quantile curves for the example; for thin lines use tidy session values as
    # degenerate CDFs is wrong — instead load behavior for up to 12 lab + matched pairs.
    rows_path = METRIC_ROWS_PATH if METRIC_ROWS_PATH.is_file() else DATA_QUALITY_ROOT / "phase3_4a" / "metric_rows_4a.jsonl"
    # Build thin-line CDFs from a subsample of sessions' behavior files (matched arm has both)
    n_extra = 0
    for entry in man["sessions"]:
        if n_extra >= 20:
            break
        sess = entry["session"]
        if sess == session:
            continue
        try:
            # lab
            bun = build_segment_bundle_dq(sess, "A", require_ratemap_stability=True)
            lt = np.asarray(bun["decode_times"], float)
            ly = np.asarray(bun["y"], float)
            lv = np.asarray(bun["target_valid"], bool)
            lab_st = _speed_turn(ly, lt, lv)
            # sim
            bdir = DATA_QUALITY_ROOT / "matched" / f"{sess}_A" / "sim"
            beh2 = pd.read_csv(bdir / "behavior.csv")
            tc = "time_s" if "time_s" in beh2.columns else beh2.columns[0]
            st2 = beh2[tc].to_numpy(float)
            sy2 = beh2[["x_cm", "y_cm"]].to_numpy(float) if {"x_cm", "y_cm"} <= set(beh2.columns) else beh2.iloc[:, 1:3].to_numpy(float)
            sim_st = _speed_turn(sy2, st2, np.ones(len(st2), bool))
            for key, st in (("lab", lab_st), ("sim_matched", sim_st)):
                sq = _quantiles(st["speed_cm_s"])
                tq = _quantiles(np.abs(st["turning_rad"]))
                if sq is None or tq is None:
                    continue
                cohort[key].append({"session": sess, "speed_q": sq, "turn_q": tq, "example": False})
            n_extra += 1
        except Exception:
            continue

    example_st["cohort"] = cohort
    (out / "data_speed_turning_example.json").write_text(json.dumps(example_st) + "\n")


def write_meta(out: Path, div: pd.DataFrame) -> None:
    score = json.loads((DATA_QUALITY_ROOT / "phase3_4a" / "SCORE_4A.json").read_text())
    neu = json.loads((DATA_QUALITY_ROOT / "neutrality" / "SUMMARY.json").read_text())
    rows_path = METRIC_ROWS_PATH if METRIC_ROWS_PATH.is_file() else DATA_QUALITY_ROOT / "phase3_4a" / "metric_rows_4a.jsonl"
    sess = session_medians(rows_path)
    refr = sess[sess.metric == "refractory_violation_rate"].groupby("dataset")["value"].median()
    # eligibility counts
    elig = {}
    if rows_path.is_file():
        df = pd.DataFrame([json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()])
        if "n_units_eligible_spatial" in set(df.metric):
            g = df[df.metric == "n_units_eligible_spatial"].groupby("dataset")["value"]
            t = df[df.metric == "n_units_total"].groupby("dataset")["value"]
            for ds in g.mean().index:
                elig[ds] = {
                    "median_eligible": float(g.median().get(ds, np.nan)),
                    "median_total": float(t.median().get(ds, np.nan)),
                    "mean_eligible": float(g.mean().get(ds, np.nan)),
                }
    # decode absolute medians at N=20
    dec = {}
    for ds in ("lab", "sim_matched", "sim_default"):
        vals = sess[(sess.dataset == ds) & (sess.metric == "decode_error_over_floor")]["value"]
        dec[ds] = float(vals.median()) if len(vals) else float("nan")

    # SUB presence in lab
    sub_note = "SUB units are near-absent in most lab sessions; Fig 4 examples require a session with both MEC and SUB."
    meta = {
        "n_lab_sessions": 47,
        "n_animals": 6,
        "n_matched_sessions": 47,
        "n_default_seeds": 5,
        "thigmotaxis": 0.8,
        "thigmotaxis_fitted": False,
        "room": "A only",
        "censor_isi_ms": 0.3,
        "example_session": EXAMPLE_SESSION,
        "rate_map": {
            "primary": "grid20",
            "bin_cm_secondary": BIN_CM,
            "min_occupancy_s": OCC_THRESH_S,
            "smooth_sigma_bins": SMOOTH_SIGMA_BINS,
            "speed_min_cm_s": SPEED_MIN_CM_S,
            "min_train_spikes": MIN_TRAIN_SPIKES,
            "min_mean_rate_hz": MIN_MEAN_RATE_HZ,
            "train_frac": TRAIN_FRAC,
            "selection": (
                "eligible units only (≥100 train spikes, mean rate ≥0.2 Hz); "
                "per region (MEC, SUB) at 25th/50th/75th percentile of spatial information; "
                "unvisited bins grey"
            ),
        },
        "eligibility": elig,
        "decode_error_over_floor_median_N20": dec,
        "sub_caveat": sub_note,
        "refractory_pct": {
            "lab": float(refr.get("lab", np.nan) * 100),
            "sim_matched": float(refr.get("sim_matched", np.nan) * 100),
            "sim_default": float(refr.get("sim_default", np.nan) * 100),
        },
        "generator_has_public_refractory_param": False,
        "boost_neutrality": neu,
        "discriminator": {
            "default": {k: score["discriminator_default"].get(k) for k in (
                "auc_logit", "auc_group_shuffle_null", "auc_group_shuffle_null_stratified")},
            "matched": {k: score["discriminator_matched"].get(k) for k in (
                "auc_logit", "auc_group_shuffle_null", "auc_group_shuffle_null_stratified")},
        },
        "theta_mod_formula": (
            "TMI = (mean(ACG in 0.100–0.167 s) − mean(ACG in 0.05–0.5 s excluding that band)) "
            "/ mean(ACG baseline). Negative values mean the theta-period band is below the "
            "broadband baseline (often when short-lag bursting inflates the baseline)."
        ),
        "calibration_note": "Calibration (former Q13–Q15) deferred to Report 4b after proxy-fidelity FAIL.",
        "R_definition": "R = d_sim / d_lab; MAD-scaled absolute distances of session medians; 90% bootstrap by recording_day",
        "default_arm_intervals": False,
        "default_arm_note": "Default arm shown as point estimates only (5 seeds); 90% intervals are matched-arm recording_day bootstrap.",
        "spatial_revision": "4a_B",
        "sparsity_formula": "(Σ p_i r_i)² / Σ p_i r_i² (Skaggs; occupancy-normalised)",
        "place_field_definition": (
            "contiguous bins with rate ≥ max(20% of peak, 1 Hz), ≥4 bins; "
            "size = field/visited among place-tuned units (SI > shuffle 95%)"
        ),
        "occmatched": "dropped — grid-index intersection across lab polygon vs sim square is not meaningful",
        "path_containment": {
            "min": 0.95,
            "semantics": "flag only (Report 2); failures kept with path-visited bin mask",
        },
    }
    # Attach Q3 summary if present
    q3p = DATA_QUALITY_ROOT / "reports" / "report4a_v3" / "q3_path_containment.json"
    if q3p.is_file():
        meta["q3"] = json.loads(q3p.read_text())
    (out / "data_meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(DATA_QUALITY_ROOT / "reports" / "report4a_v3" / "data"),
    )
    ap.add_argument("--skip-heavy", action="store_true", help="Skip rate maps / traj / ISI pools")
    args = ap.parse_args()
    out = assert_dq_path(Path(args.out))
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"extract_start": True, "out": str(out)}), flush=True)
    div = write_divergence_both_arms(out)
    print(json.dumps({"fig3_rows": len(div)}), flush=True)
    if not args.skip_heavy:
        extract_rate_maps_and_traj(out)
        print(json.dumps({"assets": True}), flush=True)
    else:
        # Reuse v2 heavy assets when present
        v2 = DATA_QUALITY_ROOT / "reports" / "report4a_v2" / "data"
        for name in (
            "data_rate_maps.json",
            "data_trajectories_60s.json",
            "data_isi_pool.json",
            "data_occupancy.json",
            "data_speed_turning_example.json",
        ):
            src, dst = v2 / name, out / name
            if src.is_file() and not dst.is_file():
                dst.write_bytes(src.read_bytes())
        print(json.dumps({"assets": "reused_v2_heavy"}), flush=True)
    write_meta(out, div)
    rows_path = METRIC_ROWS_PATH if METRIC_ROWS_PATH.is_file() else DATA_QUALITY_ROOT / "phase3_4a" / "metric_rows_4a.jsonl"
    sess = session_medians(rows_path)
    sess.to_csv(out / "data_session_metrics.csv", index=False)
    # Absolute lab|matched table for A fixes
    abs_rows = []
    for m in (
        "sparsity_median",
        "place_field_size_median",
        "frac_place_tuned",
        "spatial_info_median",
        "peak_rate_median",
        "frac_border_pass",
        "frac_speed_pass",
        "frac_hd_pass",
        "frac_grid_pass",
    ):
        g = sess[sess.metric == m].groupby("dataset")["value"].median()
        abs_rows.append({
            "metric": m,
            "lab": float(g.get("lab", float("nan"))),
            "sim_matched": float(g.get("sim_matched", float("nan"))),
            "sim_default": float(g.get("sim_default", float("nan"))),
        })
    pd.DataFrame(abs_rows).to_csv(out / "data_spatial_abs_medians.csv", index=False)
    print(json.dumps({"done": True}), flush=True)


if __name__ == "__main__":
    main()
