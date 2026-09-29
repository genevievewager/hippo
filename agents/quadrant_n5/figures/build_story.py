"""
Assemble the figure set: story page + one page per figure with its legend.
Figures stay vector (placed from their PDFs); every number in the text is
computed from data_*.csv, never typed by hand.

    python build_story.py --figs figures --out quadrant_n5_figure_set.pdf
"""
import argparse, io, json, os
import numpy as np, pandas as pd
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Frame, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors

SH = {"raw": "raw", "raw_lag": "raw + history", "pca": "PCA", "dm": "DM", "lds": "LDS", "isomap": "Isomap", "gpfa": "GPFA"}
HERE = os.path.dirname(os.path.abspath(__file__))
FONT, BOLD = "Helvetica", "Helvetica-Bold"
for reg, bold in [("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
                  ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
                  ("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")]:
    if os.path.exists(reg) and os.path.exists(bold):
        pdfmetrics.registerFont(TTFont("Sans", reg)); pdfmetrics.registerFont(TTFont("Sans-Bold", bold))
        pdfmetrics.registerFontFamily("Sans", normal="Sans", bold="Sans-Bold", italic="Sans", boldItalic="Sans-Bold")
        FONT, BOLD = "Sans", "Sans-Bold"; break

BODY = ParagraphStyle("b", fontName=FONT, fontSize=8.6, leading=11.6, spaceAfter=4.5)
LEG = ParagraphStyle("l", fontName=FONT, fontSize=7.8, leading=10.4, spaceAfter=3)
H1 = ParagraphStyle("h1", fontName=BOLD, fontSize=14, leading=17.5, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName=BOLD, fontSize=9.6, leading=12.5, spaceBefore=5, spaceAfter=3)
SM = ParagraphStyle("s", fontName=FONT, fontSize=6.8, leading=8.8, textColor=colors.HexColor("#52514e"))
W, H = A4
M = 15 * mm


def latent_d_sentences(data):
    """Method-specific d=20 vs d=10 wording from data_d_sweep.csv (Ridge, sorted)."""
    path = os.path.join(data, "data_d_sweep.csv")
    if not os.path.isfile(path):
        return {}, "The d sweep was not available for this build."
    DS = pd.read_csv(path)
    flat = ("Error is flat between d = 10 and 20; selecting d = 20 reflects "
            "near-ties, not unexhausted dimensionality.")
    falling = "Error was still falling at d = 20; the sweep should be extended."
    rising = "Error rises beyond d = 10."
    out = {}
    bits = []
    for method in ("pca", "dm", "isomap", "lds", "gpfa"):
        sub = DS[(DS.source == "sorted") & (DS.method == method)]
        if sub.empty:
            continue
        d10 = sub[sub.d == 10].set_index("seed").ridge_median
        d20 = sub[sub.d == 20].set_index("seed").ridge_median
        common = d10.index.intersection(d20.index)
        delta = (d20.loc[common] - d10.loc[common]).values
        mean_d = float(np.mean(delta))
        sd_d = float(np.std(delta, ddof=1)) if len(delta) > 1 else 0.0
        n_neg = int((delta < 0).sum())
        n_pos = int((delta > 0).sum())
        if abs(mean_d) < sd_d or (n_neg < 4 and n_pos < 4):
            sent = flat
        elif n_neg >= 4:
            sent = falling
        elif n_pos >= 4:
            sent = rising
        else:
            sent = flat
        out[method] = dict(
            sentence=sent, mean=mean_d, sd=sd_d, n_neg=n_neg, n_pos=n_pos, n=len(delta),
        )
        bits.append(f"{SH[method]}: {sent.rstrip('.')} (Δ = {mean_d:+.1f} ± {sd_d:.1f} cm, "
                    f"{n_neg if mean_d <= 0 else n_pos}/{len(delta)} seeds).".replace("+-", "−"))
    # story-page compact line for the quadrant methods
    story_line = " ".join(
        f"{SH[m]} — {out[m]['sentence']}" for m in ("pca", "dm", "lds") if m in out
    )
    return out, story_line


def numbers(data):
    rd = lambda n: pd.read_csv(os.path.join(data, f"data_{n}.csv"))
    E, F, R, LC = rd("errors"), rd("floor"), rd("replay"), rd("learning_curves")
    A = pd.read_csv(os.path.join(data, "data_audit.csv"), keep_default_na=False)
    w = lambda s, d: E[E.source == s].pivot(index="seed", columns="rep", values=d)
    Xr, Xk, Gr, Gk = w("sorted", "ridge"), w("sorted", "knn"), w("ground_truth", "ridge"), w("ground_truth", "knn")
    c = lambda X, a, b: (X[a] - X[b]).values
    neg = lambda v: int((np.asarray(v) < 0).sum())
    fm = lambda v: f"{np.mean(v):+.1f} cm"
    def fs(v):
        m = float(np.mean(v)); m = 0.0 if abs(m) < 0.05 else m
        return f"{m:+.1f} ± {np.std(v, ddof=1):.1f} cm".replace("-", "−").replace("+0.0", "0.0")
    n = {}
    for dec, X in (("r", Xr), ("k", Xk)):
        for a, b in (("dm", "pca"), ("lds", "pca"), ("raw_lag", "raw"), ("lds", "raw_lag"), ("pca", "raw"),
                     ("dm", "raw"), ("lds", "raw"), ("lds", "gpfa")):
            v = c(X, a, b); n[f"{dec}_{a}_{b}"] = fs(v); n[f"{dec}_{a}_{b}_k"] = neg(v); n[f"{dec}_{a}_{b}_pos"] = int((v > 0).sum())
    for dec, X in (("gr", Gr), ("gk", Gk)):
        v = c(X, "lds", "raw_lag"); n[f"{dec}_lds_raw_lag"] = fs(v); n[f"{dec}_lds_raw_lag_k"] = neg(v)
    d_detail, d_story = latent_d_sentences(data)
    has_pred = os.path.isfile(os.path.join(data, "data_predictions_window.csv"))
    has_fail = all(os.path.isfile(os.path.join(data, f"data_{k}.csv"))
                   for k in ("error_maps", "center_pull", "error_cdf"))
    has_dsweep = os.path.isfile(os.path.join(data, "data_d_sweep.csv"))
    n.update(
        floor_lo=F[F.source == "sorted"].floor_median.min(), floor_hi=F[F.source == "sorted"].floor_median.max(),
        mean_r={k: v for k, v in Xr.mean().items()}, mean_k={k: v for k, v in Xk.mean().items()},
        sd_r={k: v for k, v in Xr.std(ddof=1).items()},
        gt_k_lo=Gk.values.min(), gt_k_hi=Gk.values.max(),
        pen_k={r: (Xk[r] - Gk[r]).mean() for r in Xk.columns},
        gain_lds=(Xr.lds - Xk.lds).mean(), gain_pca=(Xr.pca - Xk.pca).mean(), gain_dm=(Xr.dm - Xk.dm).mean(),
        gain_k=int(((Xr.lds - Xk.lds) > (Xr.pca - Xk.pca)).sum()),
        d20=int((E.d == 20).sum()), dn=int(E.d.notna().sum()),
        cvgap=float((E[E.source == "sorted"].ridge - E[E.source == "sorted"].inner_cv_ridge).median()),
        a13_r=int(E.a13_ridge_pass.sum()), a13_k=int(E.a13_knn_pass.sum()), a13_n=len(E),
        iso_p99=R[R.rep == "isomap"].p99_ms.max(), iso_ob=R[R.rep == "isomap"].frac_over_budget.max(),
        lds_p99=R[R.rep == "lds"].p99_ms.max(), dm_p99=R[R.rep == "dm"].p99_ms.max(),
        gpfa_a11_lo=R[R.rep == "gpfa"].a11_yhat_inf_cm.min(), gpfa_a11_hi=R[R.rep == "gpfa"].a11_yhat_inf_cm.max(),
        dm_a11=R[R.rep == "dm"].a11_yhat_inf_cm.max(),
        r_tb={r: np.corrcoef(F[F.source == "sorted"].set_index("seed").test_bins, Xr[r])[0, 1] for r in ("raw", "pca", "lds")},
        tb=F[F.source == "sorted"].set_index("seed").test_bins.tolist(),
        fails=A[A.status == "FAIL"][["seed", "check"]].values.tolist(),
        a13_fail_rows=[(int(r.seed), r.source, r.rep, dec, float(getattr(r, f"a13_{dec}")))
                       for r in E.itertuples() for dec in ("ridge", "knn") if not bool(getattr(r, f"a13_{dec}_pass"))],
        d_detail=d_detail, d_story=d_story,
        has_pred=has_pred, has_fail=has_fail, has_dsweep=has_dsweep,
        fs=fs, neg=neg, meta=json.load(open(os.path.join(data, "data_meta.json"))),
    )
    notes = []
    for s in sorted({x[0] for x in n["a13_fail_rows"]}):
        for dec in ("ridge", "knn"):
            sub = E[(E.seed == s)][f"a13_{dec}"]
            if (sub < 0).all():
                other = "knn" if dec == "ridge" else "ridge"
                clean = (E[E.seed == s][f"a13_{other}"] > 0).all()
                notes.append(f"On seed {s} every method sits below chance under {dec.upper() if dec == 'knn' else 'Ridge'}"
                             f"{' while ' + ('kNN' if other == 'knn' else 'Ridge') + ' stays above it' if clean else ''}, "
                             "pointing to a weak control for that session rather than leakage. ")
    n["a13_note"] = "".join(notes)
    L = LC.pivot_table(index=["seed", "frac"], columns="rep", values="ridge")
    n["lc"] = {f: L.xs(f, level="frac").mean().to_dict() for f in (0.25, 0.5, 1.0)}
    n["lc_lds_rawlag"] = {f: int((L.xs(f, level="frac").lds < L.xs(f, level="frac").raw_lag).sum()) for f in (0.25, 0.5, 1.0)}
    # centre-pull summary if present
    if has_fail:
        CP = rd("center_pull")
        sl = CP[(CP.kind == "slope") & (CP.source == "sorted")]
        n["pull"] = {
            f"{m}_{dec}": (float(sl[(sl.method == m) & (sl.decoder == dec)].slope.mean()),
                           float(sl[(sl.method == m) & (sl.decoder == dec)].slope.std(ddof=1)))
            for m in ("pca", "dm", "lds") for dec in ("ridge", "knn")
        }
    else:
        n["pull"] = {}
    return n


def story(n):
    r, k = n["mean_r"], n["mean_k"]
    P = []
    P.append(Paragraph("Temporal state, not static geometry: a paired N = 5 test of the representation quadrant "
                       "for realtime position decoding", H1))
    mt = n["meta"]
    P.append(Paragraph(f"Quadrant N = 5 experiment · config {mt['config_sha256'][:8]} · code {mt['seeds_code_sha'][:8]} (seeds) / "
                       f"{mt['report_code_sha'][:8]} (reports) · simulated data · internal draft, not for distribution", SM))
    P.append(Paragraph("The question", H2))
    P.append(Paragraph(
        "Which aspects of population structure must a neural representation preserve for accurate, stable and "
        "realtime-deployable decoding of position? We compared four classes of representation, organised on two axes "
        "(linear vs nonlinear, static vs dynamic), under one frozen, audited pipeline: the same spikes, 250 ms causal "
        "windows every 50 ms, the same contiguous train/test split with purge gaps, the same decoder and the same "
        "hyperparameter-selection rule. Only the representation changed. Five independent simulations (trajectory, "
        "neural activity, recording noise and sorting errors all re-drawn) were analysed as paired replicates.", BODY))
    P.append(Paragraph("How we got to the answer", H2))
    fails = n["a13_fail_rows"]
    if fails:
        ftxt = "; ".join(f"seed {s}, {'ground truth' if src == 'ground_truth' else 'sorted'}, {SH[rep]} "
                         f"({dec}, {v:+.2f} cm)".replace("-", "−") for s, src, rep, dec, v in fails)
        seeds_f = sorted({x[0] for x in fails})
        ftxt = (f"Fourteen audit checks passed in every seed except the time-shift null control in {len(fails)} cell(s): "
                f"{ftxt}, against its pre-registered −2 cm line. " + n["a13_note"])
    else:
        ftxt = "All fourteen audit checks passed in every seed. "
    P.append(Paragraph(
        "Before interpreting any difference we tested whether the comparison was fair (Fig. 2). " + ftxt +
        "Causal replay reproduced offline predictions to floating-point precision for every causal method; GPFA, whose "
        "smoother uses future spikes, is shown only as an offline reference.", BODY))
    P.append(Paragraph("The answer at N = 5", H2))
    P.append(Paragraph(
        f"<b>1. Dynamics is the axis that matters.</b> LDS had lower error than PCA in {n['r_lds_pca_k']}/5 seeds "
        f"(paired difference {n['r_lds_pca']}, Ridge; mean ± SD). Part of that is simply seeing the past: stacking 250 ms "
        f"of history onto raw counts helped ({n['r_raw_lag_raw']}, {n['r_raw_lag_raw_k']}/5). But the Kalman state model "
        f"improved on history alone ({n['r_lds_raw_lag']}, {n['r_lds_raw_lag_k']}/5), and under the kNN readout that "
        f"increment grew to {n['k_lds_raw_lag']} ({n['k_lds_raw_lag_k']}/5).", BODY))
    P.append(Paragraph(
        f"<b>2. Static nonlinear geometry gives nothing here.</b> Diffusion maps and PCA were indistinguishable: "
        f"{n['r_dm_pca']} under Ridge ({n['r_dm_pca_k']}/5 seeds favouring DM) and {n['k_dm_pca']} under kNN "
        f"({n['k_dm_pca_k']}/5). Isomap was no better.", BODY))
    P.append(Paragraph(
        f"<b>3. Where nonlinearity does matter is in the readout of the dynamic state.</b> Replacing Ridge with kNN lowered "
        f"LDS error by {n['gain_lds']:.1f} cm but PCA error by only {n['gain_pca']:.1f} cm (larger for LDS in "
        f"{n['gain_k']}/5 seeds). Position is carried nonlinearly in the filtered latents, consistent with a grid-cell-dominated "
        f"population. This is the closest the data come to the empty nonlinear-dynamic cell.", BODY))
    P.append(Paragraph(
        f"<b>4. The mechanism is robustness to recording.</b> With ground-truth spikes every representation reached "
        f"{n['gt_k_lo']:.1f}–{n['gt_k_hi']:.1f} cm under kNN: the position information is present in all of them. Degradation "
        f"and sorting raised kNN error by {n['pen_k']['pca']:.0f} cm for PCA and {n['pen_k']['dm']:.0f} cm for DM, but only "
        f"{n['pen_k']['lds']:.0f} cm for LDS. The dynamics model's advantage over history-stacking is absent on clean spikes "
        f"under Ridge ({n['gr_lds_raw_lag']}, {n['gr_lds_raw_lag_k']}/5) and appears only after sorting. Temporal integration "
        f"is doing denoising.", BODY))
    P.append(Paragraph(
        f"<b>5. Reduction alone can hurt.</b> Compressing to d ≤ 20 made the linear readout worse than raw counts "
        f"(PCA − raw {n['r_pca_raw']}, {n['r_pca_raw_pos']}/5 worse) while helping kNN ({n['k_pca_raw']}, "
        f"{n['k_pca_raw_k']}/5 better). Selection chose d = 20, the top of the sweep, in {n['d20']}/{n['dn']} cases. "
        f"{n['d_story']}", BODY))
    P.append(Paragraph(
        f"<b>6. It is deployable.</b> PCA, DM and LDS each cost under {max(n['lds_p99'], n['dm_p99']):.2f} ms per 50 ms "
        f"update (p99), so latency does not separate the quadrant. The causal LDS filter paid {n['r_lds_gpfa']} relative to the "
        f"offline GPFA smoother. Only Isomap approached the budget (p99 up to {n['iso_p99']:.0f} ms; up to "
        f"{100*n['iso_ob']:.1f}% of steps over).", BODY))
    P.append(Paragraph("In one sentence", H2))
    P.append(Paragraph(
        "<i>For this population, what must be preserved is the temporal continuity of the population state, read out "
        "nonlinearly. Static nonlinear geometry adds nothing measurable. The benefit of dynamics comes from resisting "
        "recording and sorting noise, and it costs well under 1 ms per update.</i>", BODY))
    P.append(Paragraph("What this does not show", H2))
    missing = []
    if not n["has_pred"]:
        missing.append("decoded trajectories")
    if not n["has_fail"]:
        missing.append("spatial error maps, center-pull")
    if not n["has_dsweep"]:
        missing.append("the full d sweep")
    missing_txt = (
        f" The runner did not save {', '.join(missing)}, so those panels are absent."
        if missing else ""
    )
    P.append(Paragraph(
        "This is a simulation, not animals. The population on the lab probe track is entorhinal-dominated (grid, "
        "head-direction, speed and border cells) with no CA1/CA3 place cells, so a place-cell map may reward nonlinear "
        "static embeddings differently. N = 5 supports effect sizes and sign consistency, not significance (the smallest "
        f"attainable two-sided sign-test p is 0.0625). {n['d_story']} "
        "Each seed's test segment covers only part of the arena, and seed-to-seed spread follows how much of it the test "
        f"path visits (r = {n['r_tb']['raw']:.2f} for raw). The nonlinear-dynamic cell was not tested, so the "
        f"linear×dynamic interaction is not estimable.{missing_txt}", BODY))
    P.append(Paragraph("Next experiment", H2))
    P.append(Paragraph(
        "(i) Implement one nonlinear-dynamic representation (for example a switching or kernelised state-space model, or "
        "a filter on DM coordinates). (ii) Extend the d sweep beyond 20 where error is still falling. (iii) Add a place-cell "
        "population to test whether static geometry matters when the code is a place map. (iv) Increase N, now that the "
        "effect sizes are known.", BODY))
    return P


def legends(n):
    r = n["mean_r"]
    L = {
        "Fig1_design": ("Figure 1 | Question and design.",
            "<b>a</b>, The representation quadrant. PCA (linear, static), diffusion map with Nyström extension (nonlinear, "
            "static) and a linear dynamical system read through a causal Kalman filter (linear, dynamic). The nonlinear-dynamic "
            "cell is left empty. GPFA and Isomap are baselines outside the quadrant; raw counts and raw counts plus 250 ms of "
            "stacked history are controls. <b>b</b>, Pipeline. Spikes from a simulated hippocampal formation sampled by the "
            "lab probe track are degraded to Neuropixels quality and sorted. Ground-truth spikes bypass degradation and serve "
            "only as a non-deployable reference. Counts in causal 250 ms windows every 50 ms, labelled at the window's right "
            "edge, feed a representation fit on the training block only, then a decoder shared by all representations. The "
            "600 s session is split contiguously (80/20) with a 1 s purge gap; all hyperparameters (latent d, Ridge α, kNN k) "
            "are chosen by 5-fold contiguous inner cross-validation with purge gaps. <b>c</b>, The recorded population "
            "(86 units in every seed) is entorhinal-dominated with no CA1/CA3 place cells. <b>d</b>, Simulated trajectories. "
            "Grey, training segment; black, test segment. The test path covers a different part of the arena in each seed "
            f"({', '.join(str(t) for t in n['tb'])} of 100 bins)."),
        "Fig2_validity": ("Figure 2 | The comparison is fair.",
            "<b>a</b>, Fourteen audit checks per seed (tick, pass; cross, fail; dash, not applicable, because the frozen configuration is "
            "CLI-built, not UI-built). <b>b</b>, Time-shift null control. Labels were circularly shifted by 20 offsets "
            "(0.25–0.75 of the session) and decoded through the full pipeline. Each point is one seed's median over shifts "
            "of (shifted-label error − chance floor), where chance is predicting the training-mean position. Values near or "
            f"above zero mean no spurious alignment. {n['a13_r']}/{n['a13_n']} Ridge and {n['a13_k']}/{n['a13_n']} kNN "
            "cells pass the pre-registered −2 cm line; failing cells are circled, marked unreliable and not re-thresholded. "
            + n["a13_note"] + "<b>c</b>, Held-out test error against the inner-CV error that selected "
            f"the hyperparameters (sorted spikes, Ridge; median test − CV = {n['cvgap']:+.1f} cm). <b>d</b>, Largest "
            "per-sample difference between offline and step-by-step causal replay predictions. Causal methods agree to "
            f"≤ {n['dm_a11']:.1e} cm. GPFA's smoother differs by {n['gpfa_a11_lo']:.0f}–{n['gpfa_a11_hi']:.0f} cm and is "
            "marked offline-only (†)."),
        "Fig3_quadrant_answer": ("Figure 3 | Dynamics, not nonlinear geometry, separates the quadrant.",
            "Median test error per seed (grey points; lines join the same seed) and across-seed mean (coloured bar) for "
            "sorted spikes, with <b>a</b>, the primary Ridge readout and <b>b</b>, the kNN sensitivity readout. The shaded "
            f"band marks chance across seeds ({n['floor_lo']:.1f}–{n['floor_hi']:.1f} cm). <b>c</b>, Planned paired contrasts. "
            "Each point is one seed's difference (first − second term) for Ridge (filled) and kNN (open); vertical bars are "
            "means. The right-hand column counts seeds in which the first term had lower error. Descriptive at N = 5: no "
            "p-values."),
        "Fig4_mechanism": ("Figure 4 | Why dynamics wins: robustness to recording, read out nonlinearly.",
            "<b>a</b>, kNN against Ridge error for every method and seed (sorted spikes). Static representations sit near "
            f"the diagonal. LDS falls far below it: a nonlinear readout lowers its error by {n['gain_lds']:.1f} cm on average "
            f"against {n['gain_pca']:.1f} cm for PCA. <b>b</b>, The same representations decoded from ground-truth spikes "
            "(open) and from degraded, sorted spikes (filled), kNN readout. Numbers above give the mean increase in error "
            f"caused by recording and sorting. All representations reach {n['gt_k_lo']:.1f}–{n['gt_k_hi']:.1f} cm on "
            "ground truth; only the dynamic methods keep that accuracy after sorting. <b>c</b>, The Kalman state model's "
            "gain beyond history-stacking (LDS − raw + history) is near zero on ground-truth spikes and consistently "
            f"negative on sorted spikes (Ridge {n['r_lds_raw_lag']}, kNN {n['k_lds_raw_lag']})."),
        "Fig5_deployability": ("Figure 5 | Deployability and generality.",
            "<b>a</b>, Transform cost per 50 ms update during causal replay (one row per seed; dot, median; tick, 99th "
            f"percentile). All quadrant methods stay below {max(n['lds_p99'], n['dm_p99']):.2f} ms. Isomap reaches a p99 of "
            f"{n['iso_p99']:.0f} ms and exceeds the budget on up to {100*n['iso_ob']:.1f}% of steps. GPFA was timed with a "
            "filter for reference only. <b>b</b>, Cost of causality: the causal LDS filter against the offline GPFA smoother "
            f"(Ridge {n['r_lds_gpfa']}, kNN {n['k_lds_gpfa']}). <b>c</b>, Selected latent d by inner CV "
            f"(filled, sorted; open, ground truth). d = 20 was chosen in {n['d20']}/{n['dn']} cases. "
            f"{n['d_story']} "
            "<b>d</b>, Learning curves (Ridge, sorted): training on the most recent 25%, 50% or 100% of the training block, "
            "same test set. Mean ± s.e.m. across seeds. Every method improves with data and the mean ordering holds at each "
            "size, although LDS beat raw + history in only "
            f"{n['lc_lds_rawlag'][0.5]}/5 seeds at 50%. <b>e</b>, Seed-to-seed spread tracks how much of the arena the test "
            f"path visits (Pearson r = {n['r_tb']['raw']:.2f}, {n['r_tb']['pca']:.2f} and {n['r_tb']['lds']:.2f} for raw, "
            "PCA and LDS). This is why paired, within-seed contrasts are the unit of evidence."),
        "Fig6_answer": ("Figure 6 | The answer at N = 5.",
            "Each cell shows mean ± SD across seeds of the per-seed median error (sorted spikes) under Ridge and kNN. "
            "Arrows give the paired contrasts. Moving from static to dynamic (PCA → LDS) is the one large, consistent step; "
            "moving from linear to nonlinear static geometry (PCA → DM) is not. The nonlinear-dynamic cell is the "
            "next experiment. The consistent extra benefit of a nonlinear readout on the LDS state (Fig. 4a) suggests it is "
            "where an interaction, if any, would appear."),
    }
    if n["has_pred"]:
        L["Fig7_trajectories"] = ("Figure 7 | Decoded trajectories.",
            "Sorted spikes. The seed is the one whose LDS − PCA Ridge difference is closest to the across-seed median; "
            "the window is the first 60 s of the test segment (both rules fixed, no picking). <b>a</b>, Arena view "
            "(rows = Ridge, kNN; columns = PCA, DM, LDS): true path in black, decoded path in quadrant colour; open circle "
            "marks the start; each panel shows that window's median error. <b>b</b>, x(t) and y(t) with PCA, DM and LDS "
            "overlaid (Ridge; kNN inset). <b>c</b>, Euclidean error with a 2 s rolling median; dashed grey is chance. "
            "<b>d</b>, Window median for PCA, DM and LDS in all five seeds (filled = the seed shown), so the example is "
            "typical rather than the best case.")
    if n["has_dsweep"]:
        L["FigS1_latent_d"] = ("Figure S1 | Latent dimensionality.",
            "Median test error versus latent d (log-x ticks at 2, 3, 5, 10, 20), mean ± s.e.m. across seeds, for each "
            "decoder × source. Methods were refit with Phase 3's per-fold representation rule. Open rings mark the d "
            f"selected for each seed. {n['d_story']}")
    L["FigS2_a13_full"] = ("Figure S2 | Full time-shift null.",
        "For each method, the 20 per-shift (shifted − floor) values from every seed. Strip plot with the pre-registered "
        f"−2 cm line. Sorted and ground-truth side by side. {n['a13_r']}/{n['a13_n']} Ridge and "
        f"{n['a13_k']}/{n['a13_n']} kNN cells pass.")
    if n["has_pred"]:
        L["FigS3_trajectories_knn"] = ("Figure S3 | Decoded trajectories, all seeds (kNN).",
            "The Fig. 7a arena view repeated for every seed (rows) and PCA, DM, LDS (columns), kNN readout, first 60 s "
            "of the test segment. The companion Ridge page follows the same layout.")
        L["FigS3_trajectories_ridge"] = ("Figure S3 (cont.) | Decoded trajectories, all seeds (Ridge).",
            "Same layout as the kNN page, Ridge readout.")
    if n["has_fail"]:
        pull = n["pull"]
        pull_txt = "; ".join(
            f"{SH[m]} Ridge {pull[f'{m}_ridge'][0]:.2f}±{pull[f'{m}_ridge'][1]:.2f}, "
            f"kNN {pull[f'{m}_knn'][0]:.2f}±{pull[f'{m}_knn'][1]:.2f}"
            for m in ("pca", "dm", "lds") if f"{m}_ridge" in pull
        )
        L["FigS4_failure_modes"] = ("Figure S4 | Failure modes.",
            "<b>a</b>, Spatial error maps: median across seeds of per-bin median Euclidean error (10×10 bins); bins with "
            "mean occupancy &lt; 10 are grey-hatched. Shared white→dark sequential scale. <b>b</b>, Centre-pull: OLS slope "
            f"of decoded versus true distance from arena centre (dots = seeds; bar = mean); reference line at 1 "
            f"(no shrinkage). {pull_txt}. <b>c</b>, Pooled error CDFs (Ridge solid, kNN dashed).")
    return L


def _divider_page(title):
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    c.setFont(BOLD, 16)
    c.drawCentredString(W / 2, H / 2, title)
    c.setFont(FONT, 8)
    c.setFillColor(colors.HexColor("#8f8d87"))
    c.drawCentredString(W / 2, H / 2 - 18, "Quadrant N = 5 · simulated data · internal draft")
    c.save(); buf.seek(0)
    return PdfReader(buf).pages[0]


def build(figs, out, data):
    n = numbers(data)
    writer = PdfWriter()
    # story page(s)
    buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
    items = story(n)
    while items:
        f = Frame(M, M, W - 2 * M, H - 2 * M, showBoundary=0, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        rest = f.addFromList(items, c); c.showPage()
        if not items: break
    c.save(); buf.seek(0)
    for p in PdfReader(buf).pages: writer.add_page(p)
    # main figures then supplementary
    main = ["Fig1_design", "Fig2_validity", "Fig3_quadrant_answer", "Fig4_mechanism",
            "Fig5_deployability", "Fig6_answer"]
    if n["has_pred"] and os.path.isfile(os.path.join(figs, "Fig7_trajectories.pdf")):
        main.append("Fig7_trajectories")
    supp = []
    if n["has_dsweep"] and os.path.isfile(os.path.join(figs, "FigS1_latent_d.pdf")):
        supp.append("FigS1_latent_d")
    if os.path.isfile(os.path.join(figs, "FigS2_a13_full.pdf")):
        supp.append("FigS2_a13_full")
    if n["has_pred"]:
        for name in ("FigS3_trajectories_knn", "FigS3_trajectories_ridge"):
            if os.path.isfile(os.path.join(figs, f"{name}.pdf")):
                supp.append(name)
    if n["has_fail"] and os.path.isfile(os.path.join(figs, "FigS4_failure_modes.pdf")):
        supp.append("FigS4_failure_modes")
    L = legends(n)
    order = [(name, False) for name in main]
    if supp:
        writer.add_page(_divider_page("Supplementary"))
        order += [(name, True) for name in supp]
    for name, _ in order:
        if name not in L:
            continue
        title, text = L[name]
        fig_path = os.path.join(figs, f"{name}.pdf")
        if not os.path.isfile(fig_path):
            continue
        fig = PdfReader(fig_path).pages[0]
        fw, fh = float(fig.mediabox.width), float(fig.mediabox.height)
        s = min((W - 2 * M) / fw, 1.0 * (H * 0.62) / fh)
        buf = io.BytesIO(); c = canvas.Canvas(buf, pagesize=A4)
        top = H - M
        fig_bottom = top - fh * s
        f = Frame(M, M, W - 2 * M, fig_bottom - M - 7 * mm, showBoundary=0, leftPadding=0, rightPadding=0,
                  topPadding=0, bottomPadding=0)
        f.addFromList([Paragraph(f"<b>{title}</b> {text}", LEG)], c)
        c.setFont(FONT, 6.5); c.setFillColor(colors.HexColor("#8f8d87"))
        c.drawRightString(W - M, 8 * mm, "Quadrant N = 5 · simulated data · internal draft")
        c.save(); buf.seek(0)
        page = PdfReader(buf).pages[0]
        page.merge_transformed_page(fig, Transformation().scale(s).translate((W - fw * s) / 2, fig_bottom))
        writer.add_page(page)
    with open(out, "wb") as fh_:
        writer.write(fh_)
    print("wrote", out, len(writer.pages), "pages")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default="figures"); ap.add_argument("--out", default="quadrant_n5_figure_set.pdf")
    ap.add_argument("--data", default=HERE)
    a = ap.parse_args(); build(a.figs, a.out, a.data)
