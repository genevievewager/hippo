"""Assemble Report 2: story + figures with Report-1 page layout.

Layout (matches Report 1): figure at the TOP of each page, scaled to text
width (never beyond margins); legend BELOW. Every number from tidy tables.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Frame
from reportlab.pdfgen import canvas

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from agents.quadrant_n5.figures.build_story import (  # noqa: E402
    FONT, BOLD, BODY, LEG, H1, H2, SM, W, H, M,
)

SH = {
    "raw": "raw", "raw_smooth": "raw+EMA", "raw_lag": "raw+hist",
    "pca": "PCA", "pca_smooth": "PCA+EMA", "dm": "DM", "dm_smooth": "DM+EMA",
    "lds": "LDS", "lds_smooth": "LDS+EMA", "isomap": "Isomap",
    "gpfa_causal": "GPFA-c", "gpfa": "GPFA-off",
}


def _neg(v):
    return int((np.asarray(v) < 0).sum())


def _fmt_n(v, sd=None):
    """Normalized error: 3 decimals."""
    if sd is None:
        return f"{float(v):+.3f}"
    return f"{float(v):+.3f} +/- {float(sd):.3f}"


def _fmt_cm(v):
    """cm: 1 decimal."""
    return f"{float(v):.1f}"


def numbers(data_dir: str) -> dict:
    E = pd.read_csv(os.path.join(data_dir, "data_errors.csv"))
    F = pd.read_csv(os.path.join(data_dir, "data_floor.csv"))
    C = pd.read_csv(os.path.join(data_dir, "data_contrasts.csv"))
    meta = json.load(open(os.path.join(data_dir, "data_meta.json")))
    n_ani = int(meta.get("n_animals", E.seed.nunique()))
    n_sess = int(meta.get("n_sessions", 10))

    # Aggregation (stated once in methods): animal = mean of session medians;
    # cohort = mean over animals. data_errors.csv already stores animal means.
    def animal_mean(rep, col):
        sub = E[E.rep == rep]
        if sub.empty or col not in sub or not sub[col].notna().any():
            return float("nan"), 0.0
        return float(sub[col].mean()), float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0

    mean_rn, sd_rn, mean_cm, sd_cm = {}, {}, {}, {}
    mean_kn, sd_kn, mean_kcm, sd_kcm = {}, {}, {}, {}
    for rep in E.rep.unique():
        mean_rn[rep], sd_rn[rep] = animal_mean(rep, "ridge_norm")
        mean_cm[rep], sd_cm[rep] = animal_mean(rep, "ridge")
        if E[E.rep == rep]["knn_norm"].notna().any():
            mean_kn[rep], sd_kn[rep] = animal_mean(rep, "knn_norm")
            mean_kcm[rep], sd_kcm[rep] = animal_mean(rep, "knn")

    contrasts = {}
    for _, row in C[C.decoder == "ridge"].iterrows():
        key = row["contrast"]
        contrasts.setdefault(key, {
            "deltas": [], "label": row["label"], "grid": row["grid_mode"],
        })
        contrasts[key]["deltas"].append(float(row["delta"]))
    for key, block in contrasts.items():
        d = np.asarray(block["deltas"])
        block["mean"] = float(d.mean())
        block["sd"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        block["k"] = _neg(d)
        block["n"] = len(d)

    knn_c = {}
    for _, row in C[C.decoder == "knn"].iterrows():
        key = row["contrast"]
        knn_c.setdefault(key, {"deltas": [], "label": row["label"]})
        knn_c[key]["deltas"].append(float(row["delta"]))
    for key, block in knn_c.items():
        d = np.asarray(block["deltas"])
        block["mean"] = float(d.mean())
        block["sd"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        block["k"] = _neg(d)
        block["n"] = len(d)

    # kNN - Ridge per method (animal means) at Ridge-selected d (primary)
    knn_vs_ridge = {}
    for rep in E.rep.unique():
        if rep.endswith("_smooth"):
            continue
        sub = E[E.rep == rep]
        if not sub["knn"].notna().any():
            continue
        d_n = (sub["knn_norm"] - sub["ridge_norm"]).dropna().values
        d_cm = (sub["knn"] - sub["ridge"]).dropna().values
        if len(d_n) == 0:
            continue
        knn_vs_ridge[rep] = dict(
            mean_n=float(d_n.mean()), sd_n=float(d_n.std(ddof=1)) if len(d_n) > 1 else 0.0,
            k_n=int((d_n < 0).sum()), n=len(d_n),
            mean_cm=float(d_cm.mean()) if len(d_cm) else float("nan"),
        )

    # kNN at each method's kNN-best d (descriptive, chosen on test) from d_sweep
    knn_vs_ridge_best = {}
    ds_path = os.path.join(data_dir, "data_d_sweep.csv")
    if os.path.isfile(ds_path):
        DS = pd.read_csv(ds_path)
        if "grid" in DS.columns:
            DS = DS[DS.grid == "final"]
        for method in ("pca", "dm", "lds", "isomap", "gpfa"):
            sub = DS[DS.method == method]
            e_m = E[E.rep == method].set_index("seed")
            if sub.empty or e_m.empty or "knn_norm" not in e_m.columns:
                continue
            animal_knn = {}
            for seed, gseed in sub.groupby("seed"):
                sess_vals = []
                groups = (gseed.groupby("session") if "session" in gseed.columns
                          else [(None, gseed)])
                for _, g in groups:
                    g = g.dropna(subset=["knn_median", "knn_norm"])
                    if g.empty:
                        continue
                    best = g.sort_values(["knn_median", "d"]).iloc[0]
                    sess_vals.append(float(best["knn_norm"]))
                if sess_vals:
                    animal_knn[int(seed)] = float(np.mean(sess_vals))
            if not animal_knn:
                continue
            kb = pd.Series(animal_knn)
            ridge = e_m["ridge_norm"].reindex(kb.index)
            d_n = (kb - ridge).dropna().values
            if len(d_n) == 0:
                continue
            knn_vs_ridge_best[method] = dict(
                mean_n=float(d_n.mean()),
                sd_n=float(d_n.std(ddof=1)) if len(d_n) > 1 else 0.0,
                k_n=int((d_n < 0).sum()), n=len(d_n),
            )

    plateau = meta.get("plateau") or {}

    # Fig5c: Pearson r (coverage vs Ridge normalized) for legend
    coverage_r = {}
    if "coverage" in F.columns:
        for rep in ("raw", "pca", "lds"):
            sub = E[E.rep == rep].merge(
                F[["seed", "coverage"]], on="seed", how="inner",
            )
            if len(sub) >= 3 and sub["coverage"].nunique() > 1:
                coverage_r[rep] = float(np.corrcoef(
                    sub["coverage"].values, sub["ridge_norm"].values,
                )[0, 1])

    return dict(
        meta=meta, n_ani=n_ani, n_sess=n_sess,
        mean_rn=mean_rn, sd_rn=sd_rn, mean_cm=mean_cm, sd_cm=sd_cm,
        mean_kn=mean_kn, sd_kn=sd_kn, mean_kcm=mean_kcm, sd_kcm=sd_kcm,
        contrasts=contrasts, knn_c=knn_c,
        knn_vs_ridge=knn_vs_ridge, knn_vs_ridge_best=knn_vs_ridge_best,
        plateau=plateau, coverage_r=coverage_r,
        floor_lo=float(F.floor_median.min()), floor_hi=float(F.floor_median.max()),
        floor_mean=float(F.floor_median.mean()),
        ema_ridge_only=bool(meta.get("ema_ridge_only", True)),
        aggregation=(
            "All cohort summaries are the mean over animals of each animal's "
            "per-session median (normalized error = median Euclidean error / "
            "chance floor)."
        ),
    )


def _c(n, key):
    b = n["contrasts"].get(key) or {}
    if not b:
        return "n/a", "n/a"
    return (
        f"{_fmt_n(b['mean'], b['sd'])} ({b['k']}/{b['n']})",
        b.get("label", key),
    )


def story(n):
    mt = n["meta"]
    P = []
    P.append(Paragraph(
        "Temporal integration helps; static compression does not beat the full "
        "population: a paired N = 6 animal test of the representation quadrant "
        "on real hippocampal recordings",
        H1,
    ))
    cfg = (mt.get("config_sha256") or "")[:8]
    code = (mt.get("git_sha") or mt.get("report_code_sha") or "")[:8]
    fig_sha = mt.get("figure_set_sha256") or ""
    prov = f"Report 2 · real data · room A · config {cfg} · code {code}"
    if fig_sha:
        prov += f" · figure-set sha256 {fig_sha}"
    prov += " · real data · internal draft · not for distribution"
    P.append(Paragraph(prov, SM))

    P.append(Paragraph("The question", H2))
    P.append(Paragraph(
        f"Which aspects of population structure must a neural representation preserve for accurate "
        f"offline position decoding on real data? We compared the same four classes of representation "
        f"as in Report 1 (linear vs nonlinear, static vs dynamic), under one frozen causal pipeline: "
        f"spike times rebuilt into 250 ms causal count windows every 50 ms, contiguous train/test "
        f"split with purge, identical Ridge (primary) and kNN (sensitivity) readouts, and the same "
        f"inner-CV selection rule. Cohort: {n['n_ani']} animals, {n['n_sess']} room-A sessions "
        f"(2/animal where eligible). Chance floor = predict training-mean position; primary axis = "
        f"normalized error (error / floor). EMA smoothing is Ridge-only. {n['aggregation']}",
        BODY,
    ))

    P.append(Paragraph("How we got to the answer", H2))
    P.append(Paragraph(
        "Analyses were frozen after the latent-d plateau decision (PCA/LDS extended to "
        "{2,3,5,10,20,40,80}; 40 to 80 improvement &lt; 0.01 on both). DM, Isomap and offline GPFA "
        "remain on the d &lt;= 20 grid; dm_smooth - pca_smooth therefore uses matched grid20 rows "
        "for both arms. Offline GPFA is reference-only and excluded from planned contrasts. "
        "Fairness checks that require a simulator or realtime replay (A7, A9-A12) are marked n/a. "
        "Example session and units follow fixed rules in the tidy meta.",
        BODY,
    ))

    P.append(Paragraph(f"The answer at N = {n['n_ani']} animals", H2))
    t1, _ = _c(n, "raw_smooth - raw")
    t2, _ = _c(n, "pca_smooth - raw_smooth")
    t3, _ = _c(n, "dm_smooth - pca_smooth")
    t4, _ = _c(n, "lds_smooth - raw_smooth")
    t5, _ = _c(n, "gpfa_causal - raw_smooth")
    t6, _ = _c(n, "lds - raw_smooth")
    P.append(Paragraph(
        f"<b>1. Temporal integration helps.</b> Causal EMA on the full population beats unsmoothed "
        f"raw (raw_smooth - raw = {t1}, Ridge normalized; k/N = animals with first term lower). "
        f"Mean cm: raw {_fmt_cm(n['mean_cm'].get('raw', float('nan')))}, "
        f"raw+EMA {_fmt_cm(n['mean_cm'].get('raw_smooth', float('nan')))} "
        f"(floor {_fmt_cm(n['floor_lo'])}-{_fmt_cm(n['floor_hi'])} cm).",
        BODY,
    ))
    P.append(Paragraph(
        f"<b>2. Static compression does not beat the full population.</b> "
        f"PCA+EMA vs raw+EMA: {t2}. LDS vs raw+EMA: {t6}.",
        BODY,
    ))
    P.append(Paragraph(
        f"<b>3. DM does not beat PCA at matched d (grid20).</b> "
        f"dm_smooth - pca_smooth = {t3} (both arms locked to d &lt;= 20; "
        f"{n['contrasts'].get('dm_smooth - pca_smooth', {}).get('k', '?')}/"
        f"{n['contrasts'].get('dm_smooth - pca_smooth', {}).get('n', n['n_ani'])} animals with DM lower).",
        BODY,
    ))
    P.append(Paragraph(
        f"<b>4. LDS + EMA vs raw+EMA.</b> lds_smooth - raw_smooth = {t4}. "
        f"Causal GPFA vs raw+EMA: {t5}. Offline GPFA is excluded from contrasts.",
        BODY,
    ))
    # Point 5: readout — general kNN gain on real data (not LDS-specific)
    kvr = n.get("knn_vs_ridge") or {}
    kvrb = n.get("knn_vs_ridge_best") or {}
    bits_rs = []
    for rep in ("raw", "raw_lag", "pca", "dm", "lds", "isomap", "gpfa"):
        b = kvr.get(rep)
        if not b:
            continue
        bits_rs.append(f"{SH[rep]} {_fmt_n(b['mean_n'])} ({b['k_n']}/{b['n']})")
    bits_kb = []
    for rep in ("pca", "dm", "lds", "isomap", "gpfa"):
        b = kvrb.get(rep)
        if not b:
            continue
        bits_kb.append(f"{SH[rep]} {_fmt_n(b['mean_n'])} ({b['k_n']}/{b['n']})")
    P.append(Paragraph(
        f"<b>5. Readout: kNN is slightly better than Ridge for every method.</b> "
        f"At the pre-registered Ridge-selected d, kNN - Ridge (normalized): "
        + "; ".join(bits_rs)
        + f". At each method's kNN-best d (descriptive, chosen on test; "
        f"Fig. 3b/4a open diamonds): "
        + "; ".join(bits_kb)
        + ". Unlike Report 1, where the kNN gain was specific to the LDS state, the gain "
        f"on real data is general across representations. Primary contrasts stay at the "
        f"Ridge-selected d. Explanation deferred to the Report 1 revision.",
        BODY,
    ))

    P.append(Paragraph("In one sentence", H2))
    P.append(Paragraph(
        "<i>On these recordings, causal temporal integration (EMA / LDS) is the useful axis; "
        "static nonlinear geometry (DM) adds nothing at matched d; compressing below the "
        "full population does not help Ridge; and kNN is slightly better than Ridge for "
        "every method — a general gain, not the LDS-specific readout effect of Report 1.</i>",
        BODY,
    ))

    P.append(Paragraph("What this does not show", H2))
    plat = n["plateau"]
    pca_d = (plat.get("pca") or {}).get("improvement_40_to_80")
    lds_d = (plat.get("lds") or {}).get("improvement_40_to_80")
    P.append(Paragraph(
        f"Room A only (B and a deferred). No sorted-vs-GT, no realtime latency, no learning curves "
        f"(would need refits). Stability gate selects sessions with stable train maps. "
        f"Plateau stop: pca 40 to 80 Delta={_fmt_n(pca_d) if pca_d is not None else 'n/a'}, "
        f"lds Delta={_fmt_n(lds_d) if lds_d is not None else 'n/a'}. "
        f"N = {n['n_ani']} supports sign counts, not p-values. Nonlinear-dynamic cell still empty. "
        f"EMA is Ridge-only. RD9 region subsets would need new fits.",
        BODY,
    ))

    P.append(Paragraph("Next experiment", H2))
    P.append(Paragraph(
        "(i) Room B / a replications. (ii) Region-subset decoding (RD9). "
        "(iii) Report 1 revision with sim-vs-real normalized error and the readout gap. "
        "(iv) kNN with d chosen by kNN inner CV (not Ridge-selected d). "
        "(v) Causal EMA on kNN outputs (currently Ridge-only). "
        "(vi) One nonlinear-dynamic method when available. "
        "(vii) Larger N once effect sizes are locked.",
        BODY,
    ))
    return P


def legends(n):
    """Full Report-1-style legends with key numbers from tidy tables."""
    L = {}
    dm = n["contrasts"].get("dm_smooth - pca_smooth") or {}
    rs = n["contrasts"].get("raw_smooth - raw") or {}
    L["Fig1_design"] = (
        "Figure 1 | Question, pipeline and recordings.",
        f"<b>a</b>, Representation quadrant plus controls (raw, raw+hist, raw+EMA). "
        f"Nonlinear-dynamic cell empty. <b>b</b>, Pipeline retargeted to recordings: spike times "
        f"to causal counts [t-250 ms, t) to representation to decoder (no simulator / GT). "
        f"<b>c</b>, Units by region (pooled and per animal). <b>d</b>, Per-animal path with room "
        f"polygon (grey = train, black = test); cov = fraction of in-polygon 10 cm bins visited "
        f"by the test path. <b>e</b>, Train-only rate maps (min occupancy &gt;= 0.5 s, Gaussian "
        f"smooth, Hz); units by train spatial information among split-half-stable units. "
        f"N = {n['n_ani']} animals, {n['n_sess']} sessions. Example session by fixed rule in meta.",
    )
    L["Fig2_validity"] = (
        "Figure 2 | Fairness and causality on real data.",
        f"<b>a</b>, Fairness audit ({n['n_ani']} animals); simulator-only checks A7, A9-A12 shown n/a. "
        f"<b>b</b>, A13 time-shift null (Ridge filled, kNN open); fail line at -2 cm below floor. "
        f"<b>c</b>, Held-out test vs inner-CV Ridge error (selected d). "
        f"<b>d</b>, Features are causal: time diagram of centred Cell_* (+125 ms look-ahead) vs "
        f"causal [t-250, t), plus a 5 s example-unit count trace. EMA is Ridge-only.",
    )
    L["Fig3_quadrant_answer"] = (
        "Figure 3 | Quadrant answer (normalized error).",
        f"<b>a</b>, Ridge and <b>b</b>, kNN: per-animal normalized error per method (grey lines join "
        f"an animal; thick tick = animal mean). In <b>b</b>, filled markers = kNN at the "
        f"Ridge-selected d (primary); open diamonds = kNN at each method's kNN-best d from the "
        f"saved d sweep (descriptive, chosen on test; PCA/DM/LDS/Isomap/GPFA-off only). "
        f"Raw+EMA as control; each quadrant method with its +EMA variant. <b>c</b>, Planned "
        f"contrasts (Ridge filled, kNN open; still at Ridge-selected d) with k/{n['n_ani']}; "
        f"primary list includes dm_smooth - pca_smooth = {_fmt_n(dm.get('mean', float('nan')))} "
        f"({dm.get('k', '?')}/{dm.get('n', '?')}) at matched grid20. "
        f"Mean cm: raw {_fmt_cm(n['mean_cm'].get('raw', float('nan')))}, "
        f"raw+EMA {_fmt_cm(n['mean_cm'].get('raw_smooth', float('nan')))}, "
        f"chance {_fmt_cm(n['floor_mean'])}. {n['aggregation']} EMA Ridge-only.",
    )
    kvr = n.get("knn_vs_ridge") or {}
    kvrb = n.get("knn_vs_ridge_best") or {}
    lds_rs = kvr.get("lds") or {}
    lds_kb = kvrb.get("lds") or {}
    L["Fig4_mechanism"] = (
        "Figure 4 | Readout comparison (no ground truth).",
        f"<b>a</b>, kNN vs Ridge per method/animal (normalized); filled = kNN at Ridge-selected d, "
        f"open diamonds = kNN-best d (descriptive, chosen on test). Points below the diagonal: "
        f"kNN lower. LDS kNN - Ridge at Ridge-sel d = {_fmt_n(lds_rs.get('mean_n', float('nan')))} "
        f"({lds_rs.get('k_n', '?')}/{lds_rs.get('n', n['n_ani'])}); at kNN-best d = "
        f"{_fmt_n(lds_kb.get('mean_n', float('nan')))} "
        f"({lds_kb.get('k_n', '?')}/{lds_kb.get('n', n['n_ani'])}). "
        f"<b>b</b>, LDS - raw_lag by readout (Ridge-selected d). GT panel dropped (single spike "
        f"source). N = {n['n_ani']}.",
    )
    cov_r = n.get("coverage_r") or {}
    cov_bits = "; ".join(
        f"{SH[r]} r = {float(cov_r[r]):+.3f}" for r in ("raw", "pca", "lds") if r in cov_r
    )
    lds_m = n["mean_rn"].get("lds", float("nan"))
    gpfa_m = n["mean_rn"].get("gpfa", float("nan"))
    lds_minus_gpfa = (lds_m - gpfa_m) if np.isfinite(lds_m) and np.isfinite(gpfa_m) else float("nan")
    plat = n.get("plateau") or {}
    pca_plat = (plat.get("pca") or {}).get("improvement_40_to_80")
    lds_plat = (plat.get("lds") or {}).get("improvement_40_to_80")
    L["Fig5_deployability"] = (
        "Figure 5 | Causality cost, selected d, and coverage.",
        f"<b>a</b>, Cost of causality: Ridge and kNN normalized error for offline GPFA, "
        f"causal GPFA (Ridge only), and LDS; LDS - GPFA-off (Ridge) = "
        f"{_fmt_n(lds_minus_gpfa)}. <b>b</b>, Selected latent d per session (dot size = count); "
        f"shaded band = extended grid 40/80 (PCA, LDS). Plateau 40 to 80: PCA "
        f"Delta={_fmt_n(pca_plat) if pca_plat is not None else 'n/a'}, LDS "
        f"Delta={_fmt_n(lds_plat) if lds_plat is not None else 'n/a'}. "
        f"<b>c</b>, Ridge normalized error vs test-room coverage (fraction of in-polygon "
        f"10 cm bins visited by the test path); letters = animals. "
        f"{cov_bits}. Negative r: more room covered by the test path, lower error. "
        f"N = {n['n_ani']}. Latency / learning-curve panels omitted (need realtime / refits).",
    )
    L["Fig6_answer"] = (
        "Figure 6 | Answer cards.",
        f"Normalized error (Ridge and kNN; +EMA where defined) with four-point summary. "
        f"dm_smooth - pca_smooth at grid20 = {_fmt_n(dm.get('mean', float('nan')))} "
        f"({dm.get('k', '?')}/{dm.get('n', '?')}). "
        f"raw_smooth - raw = {_fmt_n(rs.get('mean', float('nan')))}. "
        f"N = {n['n_ani']}. Offline GPFA reference-only.",
    )
    L["Fig7_trajectories"] = (
        "Figure 7 | Example decoded trajectories.",
        f"Room polygon, equal aspect; example session by fixed rule. Same three best causal "
        f"methods in panels a-d (Ridge ranking; kNN row uses unsmoothed aliases). "
        f"GPFA offline excluded. Numbers: window median / chance (cm).",
    )
    L["FigS1_latent_d"] = (
        "Figure S1 | Latent-d sweep with extended grid.",
        f"<b>a</b>, Ridge and <b>b</b>, kNN: mean +/- s.e.m. normalized error vs latent d "
        f"(N = {n['n_ani']} animals); rings = selected d per session. "
        f"<b>c</b>, Plateau check 40 to 80 (PCA, LDS): PCA Delta="
        f"{_fmt_n(pca_plat) if pca_plat is not None else 'n/a'}, LDS Delta="
        f"{_fmt_n(lds_plat) if lds_plat is not None else 'n/a'} "
        f"(stop if |Delta| &lt; 0.010). PCA/LDS on final grid to 80; DM/Isomap/GPFA-off on "
        f"d &lt;= 20. EMA is Ridge-only (no kNN smooth curves).",
    )
    L["FigS2_a13_full"] = (
        "Figure S2 | Full A13 time-shift nulls.",
        f"<b>a</b>, Ridge and <b>b</b>, kNN: median error minus chance floor (cm) vs time shift "
        f"for each method; fail line at -2 cm. Real data only (no GT). "
        f"N = {n['n_ani']} animals. Chance floor mean {_fmt_cm(n['floor_mean'])} cm.",
    )
    L["FigS3_trajectories_ridge"] = (
        "Figure S3 | All sessions, Ridge.",
        f"Decoded trajectories for all {n['n_sess']} sessions (rows), best causal methods "
        f"(columns); animal+session labels (e.g. F·S9); room polygon; first 60 s of test. "
        f"Numbers: window median / chance (cm). N = {n['n_ani']}.",
    )
    L["FigS3_trajectories_knn"] = (
        "Figure S3 | All sessions, kNN.",
        f"Same layout as Ridge S3; kNN at Ridge-selected d; EMA methods omitted (Ridge-only). "
        f"All {n['n_sess']} sessions; N = {n['n_ani']}.",
    )
    L["FigS4_failure_modes"] = (
        "Figure S4 | Failure modes and spatial error maps.",
        f"<b>a</b>, Spatial error maps on the whole test block (not first 60 s); Ridge row "
        f"includes +EMA; kNN row uses unsmoothed methods (EMA is Ridge-only). "
        f"<b>b</b>, Centre-pull slopes (Ridge). <b>c</b>, Error CDF / chance (Ridge). "
        f"<b>d</b>, Jump rate (&gt; 20 cm / 50 ms step, Ridge). Example session only "
        f"(rooms differ). N = {n['n_ani']}.",
    )
    L["FigS5_phase8"] = (
        "Figure S5 | Latents coloured by position.",
        f"<b>Columns</b>: PCA, DM, LDS, GPFA-off (offline GPFA; causal GPFA latents not in "
        f"the saved figure_contract). <b>Rows</b>: colour = room-local x or y (cm), one "
        f"colormap and named colorbar per row. First two latent dimensions; example session "
        f"(fixed rule); eval samples. kNN-pressure panels dropped.",
    )
    return L


def _divider_page(title):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont(BOLD, 16)
    c.drawCentredString(W / 2, H / 2, title)
    c.setFont(FONT, 8)
    c.setFillColor(colors.HexColor("#8f8d87"))
    c.drawCentredString(W / 2, H / 2 - 18, "Report 2 · real data · internal draft")
    c.save()
    buf.seek(0)
    return PdfReader(buf).pages[0]


def build(figs_dir: str, out_pdf: str, data_dir: str):
    n = numbers(data_dir)
    h = hashlib.sha256()
    for name in sorted(os.listdir(figs_dir)):
        if name.startswith("Fig") and name.endswith((".pdf", ".png")):
            with open(os.path.join(figs_dir, name), "rb") as f:
                h.update(f.read())
    n["meta"]["figure_set_sha256"] = h.hexdigest()
    meta_path = os.path.join(data_dir, "data_meta.json")
    meta = json.load(open(meta_path))
    meta["figure_set_sha256"] = h.hexdigest()
    json.dump(meta, open(meta_path, "w"), indent=2)

    writer = PdfWriter()
    # story (may span pages)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    items = story(n)
    n_story = 0
    while items:
        f = Frame(M, M, W - 2 * M, H - 2 * M, showBoundary=0,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        f.addFromList(items, c)
        c.showPage()
        n_story += 1
        if not items:
            break
    c.save()
    buf.seek(0)
    for p in PdfReader(buf).pages:
        writer.add_page(p)

    L = legends(n)
    main = [
        "Fig1_design", "Fig2_validity", "Fig3_quadrant_answer", "Fig4_mechanism",
        "Fig5_deployability", "Fig6_answer", "Fig7_trajectories",
    ]
    supp = [
        "FigS1_latent_d", "FigS2_a13_full",
        "FigS3_trajectories_ridge", "FigS3_trajectories_knn",
        "FigS4_failure_modes", "FigS5_phase8",
    ]

    def _add_fig_page(name):
        fig_path = os.path.join(figs_dir, f"{name}.pdf")
        if name not in L or not os.path.isfile(fig_path):
            return False
        title, text = L[name]
        fig = PdfReader(fig_path).pages[0]
        fw, fh = float(fig.mediabox.width), float(fig.mediabox.height)
        # Scale to text width; cap height so legend fits below
        s = min((W - 2 * M) / fw, (H * 0.62) / fh)
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        top = H - M
        fig_bottom = top - fh * s
        # Legend BELOW the figure
        frame = Frame(
            M, M, W - 2 * M, fig_bottom - M - 7 * mm, showBoundary=0,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0,
        )
        frame.addFromList([Paragraph(f"<b>{title}</b> {text}", LEG)], c)
        c.setFont(FONT, 6.5)
        c.setFillColor(colors.HexColor("#8f8d87"))
        c.drawRightString(W - M, 8 * mm, "Report 2 · real data · internal draft")
        c.save()
        buf.seek(0)
        page = PdfReader(buf).pages[0]
        # Figure at TOP, centred
        page.merge_transformed_page(
            fig, Transformation().scale(s).translate((W - fw * s) / 2, fig_bottom),
        )
        writer.add_page(page)
        return True

    for name in main:
        _add_fig_page(name)
    writer.add_page(_divider_page("Supplementary"))
    for name in supp:
        _add_fig_page(name)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    with open(out_pdf, "wb") as fh:
        writer.write(fh)
    print("wrote", out_pdf, len(writer.pages), "pages")
    return out_pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.figs, a.out, a.data)
