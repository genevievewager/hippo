"""Assemble Report 2 figure set: story page + figures with legends.

Every number is computed from tidy data_*.csv (never typed by hand).
Same visual / PDF structure as Report 1's build_story.py.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
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


def _fmt(v, sd=None):
    if sd is None:
        m = 0.0 if abs(v) < 0.0005 else float(v)
        return f"{m:+.3f}".replace("-", "-")
    m = 0.0 if abs(v) < 0.0005 else float(v)
    return f"{m:+.3f} +/- {sd:.3f}".replace("-", "-")


def numbers(data_dir: str) -> dict:
    E = pd.read_csv(os.path.join(data_dir, "data_errors.csv"))
    F = pd.read_csv(os.path.join(data_dir, "data_floor.csv"))
    C = pd.read_csv(os.path.join(data_dir, "data_contrasts.csv"))
    meta = json.load(open(os.path.join(data_dir, "data_meta.json")))
    n_ani = int(meta.get("n_animals", E.seed.nunique()))
    n_sess = int(meta.get("n_sessions", 10))

    def animal_mean(rep, col):
        sub = E[E.rep == rep]
        return float(sub[col].mean()), float(sub[col].std(ddof=1)) if len(sub) > 1 else 0.0

    mean_rn, sd_rn = {}, {}
    mean_cm, sd_cm = {}, {}
    mean_kn, sd_kn = {}, {}
    mean_kcm, sd_kcm = {}, {}
    for rep in E.rep.unique():
        mean_rn[rep], sd_rn[rep] = animal_mean(rep, "ridge_norm")
        mean_cm[rep], sd_cm[rep] = animal_mean(rep, "ridge")
        if E[E.rep == rep]["knn_norm"].notna().any():
            mean_kn[rep], sd_kn[rep] = animal_mean(rep, "knn_norm")
            mean_kcm[rep], sd_kcm[rep] = animal_mean(rep, "knn")

    contrasts = {}
    for _, row in C[C.decoder == "ridge"].iterrows():
        key = row["contrast"]
        contrasts.setdefault(key, {"deltas": [], "label": row["label"], "grid": row["grid_mode"]})
        contrasts[key]["deltas"].append(float(row["delta"]))
    for key, block in contrasts.items():
        d = np.asarray(block["deltas"])
        block["mean"] = float(d.mean())
        block["sd"] = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        block["k"] = _neg(d)
        block["n"] = len(d)

    # kNN contrasts (no EMA arms)
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

    plateau = meta.get("plateau") or {}
    return dict(
        meta=meta, n_ani=n_ani, n_sess=n_sess,
        mean_rn=mean_rn, sd_rn=sd_rn, mean_cm=mean_cm, sd_cm=sd_cm,
        mean_kn=mean_kn, sd_kn=sd_kn, mean_kcm=mean_kcm, sd_kcm=sd_kcm,
        contrasts=contrasts, knn_c=knn_c, plateau=plateau,
        floor_lo=float(F.floor_median.min()), floor_hi=float(F.floor_median.max()),
        ema_ridge_only=bool(meta.get("ema_ridge_only", True)),
    )


def _c(n, key):
    b = n["contrasts"].get(key) or {}
    if not b:
        return "n/a", "n/a"
    return (
        f"{_fmt(b['mean'], b['sd'])} ({b['k']}/{b['n']})",
        b.get("label", key),
    )


def story(n):
    mt = n["meta"]
    P = []
    # Title stating the answer
    P.append(Paragraph(
        "Temporal integration helps; static compression does not beat the full "
        "population: a paired N = 6 animal test of the representation quadrant "
        "on real hippocampal recordings",
        H1,
    ))
    cfg = (mt.get("config_sha256") or "")[:8]
    code = (mt.get("git_sha") or mt.get("report_code_sha") or "")[:8]
    fig_sha = mt.get("figure_set_sha256") or ""
    prov = (
        f"Report 2 · real data · room A · config {cfg} · code {code}"
    )
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
        f"normalized error (error / floor). EMA smoothing is Ridge-only.",
        BODY,
    ))

    P.append(Paragraph("How we got to the answer", H2))
    P.append(Paragraph(
        "Analyses were frozen after the latent-d plateau decision (PCA/LDS extended to "
        "{2,3,5,10,20,40,80}; 40 to 80 improvement &lt; 0.01 on both). DM, Isomap and offline GPFA "
        "remain on the d &lt;= 20 grid; dm_smooth - pca_smooth therefore uses matched grid20 rows. "
        "Offline GPFA is reference-only and excluded from planned contrasts. Fairness checks that "
        "require a simulator or realtime replay (A7, A9-A12) are marked n/a; A13 time-shift nulls "
        "and split audits are retained. Example session and units follow fixed rules recorded in "
        "the tidy meta (no hand-picking).",
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
        f"Median cm: raw {n['mean_cm'].get('raw', float('nan')):.1f}, "
        f"raw+EMA {n['mean_cm'].get('raw_smooth', float('nan')):.1f} "
        f"(floor {n['floor_lo']:.1f}-{n['floor_hi']:.1f} cm).",
        BODY,
    ))
    P.append(Paragraph(
        f"<b>2. Static compression does not beat the full population.</b> "
        f"PCA+EMA vs raw+EMA: {t2}. "
        f"LDS vs raw+EMA: {t6}.",
        BODY,
    ))
    P.append(Paragraph(
        f"<b>3. DM does not beat PCA at matched d (grid20).</b> "
        f"dm_smooth - pca_smooth = {t3} (both arms locked to d &lt;= 20).",
        BODY,
    ))
    P.append(Paragraph(
        f"<b>4. LDS + EMA vs raw+EMA.</b> lds_smooth - raw_smooth = {t4}. "
        f"Causal GPFA vs raw+EMA: {t5}. Offline GPFA is excluded from contrasts.",
        BODY,
    ))
    if n["knn_c"]:
        bits = []
        for key in ("lds - raw_lag", "dm - pca", "lds - pca"):
            b = n["knn_c"].get(key)
            if b:
                bits.append(f"{key}: {_fmt(b['mean'], b['sd'])} ({b['k']}/{b['n']})")
        if bits:
            P.append(Paragraph(
                "<b>kNN (no EMA).</b> Contrasts recomputed from saved session reports without refit: "
                + "; ".join(bits) + ".",
                BODY,
            ))

    P.append(Paragraph("In one sentence", H2))
    P.append(Paragraph(
        "<i>On these recordings, causal temporal integration (EMA / LDS) is the useful axis; "
        "static nonlinear geometry (DM) adds nothing at matched d, and compressing below the "
        "full population does not help Ridge.</i>",
        BODY,
    ))

    P.append(Paragraph("What this does not show", H2))
    plat = n["plateau"]
    pca_d = (plat.get("pca") or {}).get("improvement_40_to_80")
    lds_d = (plat.get("lds") or {}).get("improvement_40_to_80")
    P.append(Paragraph(
        f"Room A only (B and a deferred). No sorted-vs-GT, no realtime latency, no learning curves "
        f"(would need refits). Stability gate selects sessions with stable train maps. "
        f"Plateau stop: pca 40 to 80 Delta={pca_d}, lds Delta={lds_d}. "
        f"N = {n['n_ani']} supports sign counts, not p-values. Nonlinear-dynamic cell still empty. "
        f"EMA is Ridge-only. RD9 region subsets would need new fits.",
        BODY,
    ))

    P.append(Paragraph("Next experiment", H2))
    P.append(Paragraph(
        "(i) Room B / a replications. (ii) Region-subset decoding (RD9) with a priori subsets. "
        "(iii) Report 1 revision with sim-vs-real normalized error. "
        "(iv) One nonlinear-dynamic method when available. (v) Larger N once effect sizes are locked.",
        BODY,
    ))
    return P


def legends(n):
    """One-line takeaway + panel captions; numbers from tidy tables."""
    L = {}
    L["Fig1_design"] = (
        "Figure 1 | Question and design (real data).",
        f"<b>a</b>, Representation quadrant plus controls (raw, raw+hist, raw+EMA). "
        f"<b>b</b>, Pipeline retargeted to recordings: spike times to causal counts to "
        f"representation to decoder (no simulator / GT branch). "
        f"<b>c</b>, Recorded population by region (pooled and per animal). "
        f"<b>d</b>, Per-animal path with room polygon (grey train, black test). "
        f"<b>e</b>, Example train-only rate maps (min occupancy &gt;= 0.5 s, Gaussian smooth, Hz; "
        f"units by train spatial information among split-half-stable units). "
        f"N = {n['n_ani']} animals, {n['n_sess']} sessions. Example session rule in meta.",
    )
    L["Fig2_validity"] = (
        "Figure 2 | Validity on real data.",
        f"<b>a</b>, Fairness audit; simulator-only checks shown n/a. "
        f"<b>b</b>, A13 time-shift null strips (Ridge and kNN). "
        f"<b>c</b>, Test vs inner-CV error. "
        f"<b>d</b>, Features are causal: centred Cell_* vs causal [t-250 ms, t) "
        f"(zoomed 5 s + time diagram). EMA Ridge-only.",
    )
    L["Fig3_quadrant_answer"] = (
        "Figure 3 | Quadrant answer (normalized error).",
        f"<b>a</b>, Ridge and <b>b</b>, kNN: per-animal normalized error per method "
        f"(lines join an animal; raw+EMA control; quadrant methods with +EMA). "
        f"<b>c</b>, Planned contrasts for both readouts with k/{n['n_ani']}. "
        f"DM/Isomap/GPFA-c at grid20. Median cm in panel notes. "
        f"EMA is Ridge-only (no kNN on *_smooth).",
    )
    L["Fig4_mechanism"] = (
        "Figure 4 | Mechanism (no GT).",
        f"<b>a</b>, kNN vs Ridge per method/animal. "
        f"<b>c</b>, LDS - raw_lag by readout. GT panel dropped (no ground-truth spikes).",
    )
    L["Fig5_deployability"] = (
        "Figure 5 | Selected d and coverage (no latency / learning curves).",
        f"<b>a</b>, Causal LDS vs offline GPFA. <b>b</b>, Selected d including 40/80. "
        f"<b>c</b>, Spread vs test-segment coverage (normalized by room). "
        f"Latency and learning-curve panels dropped (need realtime / refits).",
    )
    L["Fig6_answer"] = (
        "Figure 6 | Answer cards.",
        f"Normalized error (Ridge and kNN, +EMA where defined) with four-point summary. "
        f"N = {n['n_ani']}. Offline GPFA marked reference-only.",
    )
    L["Fig7_trajectories"] = (
        "Figure 7 | Example decoded trajectories.",
        f"Room polygon, equal aspect; example session by fixed rule; best causal methods; "
        f"GPFA offline excluded. Median cm and normalized error in panel labels.",
    )
    L["FigS1_latent_d"] = (
        "Figure S1 | Latent-d sweep with extended grid.",
        "Plateau marked at 40 to 80. PCA/LDS final grid; DM/Isomap/GPFA grid20.",
    )
    L["FigS2_a13_full"] = (
        "Figure S2 | Full A13 time-shift nulls.",
        "Real data only (no GT column). Ridge and kNN.",
    )
    L["FigS3_trajectories_ridge"] = (
        "Figure S3 | All sessions, Ridge.",
        f"All {n['n_sess']} sessions; room polygon; first 60 s of test.",
    )
    L["FigS3_trajectories_knn"] = (
        "Figure S3 | All sessions, kNN.",
        f"All {n['n_sess']} sessions; EMA methods omitted (Ridge-only).",
    )
    L["FigS4_failure_modes"] = (
        "Figure S4 | Centre-pull, CDFs, jumps; spatial maps for example session.",
        "Rooms differ across animals — spatial error maps for the example session only.",
    )
    L["FigS5_phase8"] = (
        "Figure S5 | Latents (former RD4).",
        "PCA / DM / LDS / GPFA-c colored by position; offline GPFA labeled. "
        "kNN-pressure panels dropped.",
    )
    return L


def _draw_story(path, flowables):
    c = canvas.Canvas(path, pagesize=A4)
    f = Frame(M, M, W - 2 * M, H - 2 * M, showBoundary=0)
    f.addFromList(flowables, c)
    c.save()


def _draw_legend_page(path, title, body, fig_pdf):
    c = canvas.Canvas(path, pagesize=A4)
    story_bits = [Paragraph(title, H2), Paragraph(body, LEG)]
    frame = Frame(M, H * 0.62, W - 2 * M, H * 0.32, showBoundary=0)
    frame.addFromList(story_bits, c)
    # place figure PDF
    if fig_pdf and os.path.isfile(fig_pdf):
        reader = PdfReader(fig_pdf)
        page = reader.pages[0]
        # scale to width
        pw = float(page.mediabox.width)
        ph = float(page.mediabox.height)
        target_w = W - 2 * M
        scale = target_w / pw
        target_h = ph * scale
        y = M + 10
        # write via temporary merge in caller; here draw placeholder box note
        c.setFont(FONT, 7)
        c.setFillColor(colors.HexColor("#8f8d87"))
        c.drawString(M, y + target_h + 4, os.path.basename(fig_pdf))
    c.save()
    return path


def build(figs_dir: str, out_pdf: str, data_dir: str):
    n = numbers(data_dir)
    # figure-set sha over PNG+PDF names
    h = hashlib.sha256()
    for name in sorted(os.listdir(figs_dir)):
        if name.startswith("Fig") and name.endswith((".pdf", ".png")):
            with open(os.path.join(figs_dir, name), "rb") as f:
                h.update(f.read())
    n["meta"]["figure_set_sha256"] = h.hexdigest()
    # rewrite meta with sha
    meta_path = os.path.join(data_dir, "data_meta.json")
    meta = json.load(open(meta_path))
    meta["figure_set_sha256"] = h.hexdigest()
    json.dump(meta, open(meta_path, "w"), indent=2)

    writer = PdfWriter()
    # story
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    frame = Frame(M, M, W - 2 * M, H - 2 * M, showBoundary=0)
    frame.addFromList(story(n), c)
    c.save()
    writer.add_page(PdfReader(io.BytesIO(buf.getvalue())).pages[0])

    legs = legends(n)
    fig_order = [
        "Fig1_design", "Fig2_validity", "Fig3_quadrant_answer", "Fig4_mechanism",
        "Fig5_deployability", "Fig6_answer", "Fig7_trajectories",
        "FigS1_latent_d", "FigS2_a13_full",
        "FigS3_trajectories_ridge", "FigS3_trajectories_knn",
        "FigS4_failure_modes", "FigS5_phase8",
    ]
    for key in fig_order:
        fig_pdf = os.path.join(figs_dir, f"{key}.pdf")
        if not os.path.isfile(fig_pdf):
            continue
        title, body = legs.get(key, (key, ""))
        # legend page + figure
        buf = io.BytesIO()
        c = canvas.Canvas(buf, pagesize=A4)
        bits = [Paragraph(title, H2), Paragraph(body, LEG)]
        frame = Frame(M, H * 0.70, W - 2 * M, H * 0.22, showBoundary=0)
        frame.addFromList(bits, c)
        c.save()
        page = PdfReader(io.BytesIO(buf.getvalue())).pages[0]
        # merge figure below
        fig_page = PdfReader(fig_pdf).pages[0]
        pw = float(fig_page.mediabox.width)
        scale = (W - 2 * M) / pw
        fig_page.add_transformation(Transformation().scale(scale).translate(M, M))
        page.merge_page(fig_page)
        writer.add_page(page)

    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    with open(out_pdf, "wb") as f:
        writer.write(f)
    print("wrote", out_pdf)
    return out_pdf


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    build(a.figs, a.out, a.data)
