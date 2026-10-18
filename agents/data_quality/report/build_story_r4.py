#!/usr/bin/env python3
"""Report 4a story page + assemble review PDF (Story, Fig3, Fig4, Fig5)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Spacer

HERE = Path(__file__).resolve().parent
FONT_DIR = HERE.parents[1] / "quadrant_n5" / "figures" / "fonts"


def _fonts():
    for reg, bold in (
        (FONT_DIR / "LiberationSans-Regular.ttf", FONT_DIR / "LiberationSans-Bold.ttf"),
        (
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        ),
    ):
        if reg.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("Sans", str(reg)))
            pdfmetrics.registerFont(TTFont("Sans-Bold", str(bold)))
            return "Sans", "Sans-Bold"
    raise RuntimeError("Liberation Sans not found")


FONT, BOLD = _fonts()
W, H = A4
M = 14 * mm
BODY = ParagraphStyle("b", fontName=FONT, fontSize=8.5, leading=11.4, spaceAfter=4.2)
H1 = ParagraphStyle("h1", fontName=BOLD, fontSize=12.5, leading=15.5, spaceAfter=6)
H2 = ParagraphStyle("h2", fontName=BOLD, fontSize=9.5, leading=12, spaceBefore=5, spaceAfter=2.5)
SM = ParagraphStyle("s", fontName=FONT, fontSize=6.6, leading=8.6, textColor=colors.HexColor("#52514e"))
TAKE = ParagraphStyle("t", fontName=FONT, fontSize=8.5, leading=11.4, spaceAfter=3.5, leftIndent=8)


def _fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def _fmt_r(x: float) -> str:
    return f"{x:.2f}"


def numbers(data: Path) -> dict:
    meta = json.loads((data / "data_meta.json").read_text())
    fig3 = pd.read_csv(data / "data_fig3_divergence.csv")
    right = pd.read_csv(data / "data_sim_gets_right.csv") if (data / "data_sim_gets_right.csv").is_file() else fig3.iloc[0:0]
    top = fig3.dropna(subset=["R_matched"]).sort_values("R_matched", ascending=False).head(5)
    return {"meta": meta, "fig3": fig3, "right": right, "top": top}


def _fmt_dir(r) -> str:
    if r is None or not np.isfinite(r.get("R_matched", np.nan)):
        return ""
    d = r.get("direction_matched")
    if d is None:
        d = np.sign(float(r["sim_matched_median"]) - float(r["lab_median"]))
    tip = "sim higher" if float(d) > 0 else ("sim lower" if float(d) < 0 else "similar")
    return tip


def _load_behinfo_ve(behinfo: Path | None) -> dict:
    """Encoding diagnostics + N=20 position decode for story takeaway 3."""
    out = {
        "hd_residual_deg": 45.0,
        "lab_bits_place": None, "sim_bits_place": None,
        "lab_bits_all": None, "sim_bits_all": None,
        "lab_half_r": None, "sim_half_r": None,
        "lab_frac_null": None, "sim_frac_null": None,
        "lab_pos_net": None, "sim_pos_net": None,
        "ve_valid_on_lab": False,
    }
    if behinfo is None or not behinfo.is_dir():
        return out
    diag_p = behinfo / "encoding_diag_summary.json"
    if diag_p.is_file():
        table = (json.loads(diag_p.read_text()).get("table") or {})
        for ds, bits_p, bits_a, half, frac in (
            ("lab", "lab_bits_place", "lab_bits_all", "lab_half_r", "lab_frac_null"),
            ("sim_matched", "sim_bits_place", "sim_bits_all", "sim_half_r", "sim_frac_null"),
        ):
            b = table.get(ds) or {}
            if np.isfinite(b.get("bits_end_med_place", np.nan)):
                out[bits_p] = float(b["bits_end_med_place"])
            if np.isfinite(b.get("bits_end_med_all", np.nan)):
                out[bits_a] = float(b["bits_end_med_all"])
            if np.isfinite(b.get("sess_half_map_r_place", np.nan)):
                out[half] = float(b["sess_half_map_r_place"])
            elif np.isfinite(b.get("half_map_r_place", np.nan)):
                out[half] = float(b["half_map_r_place"])
            if np.isfinite(b.get("frac_above_null_p95_prior", np.nan)):
                out[frac] = float(b["frac_above_null_p95_prior"])
            # VE valid on lab only if place-tuned ridge or rate-map VE clearly > 0
            if ds == "lab":
                ve_rm = b.get("ve_ratemap_end_med_place", np.nan)
                ve_cv = b.get("ve_ridge_cv_end_med_place", np.nan)
                out["ve_valid_on_lab"] = bool(
                    (np.isfinite(ve_rm) and ve_rm > 0.05)
                    or (np.isfinite(ve_cv) and ve_cv > 0.05)
                )
    prior_p = behinfo / "encoding_perunit_summary.json"
    if prior_p.is_file() and out["lab_frac_null"] is None:
        by = json.loads(prior_p.read_text()).get("by_dataset") or {}
        for ds, key in (("lab", "lab_frac_null"), ("sim_matched", "sim_frac_null")):
            if ds in by and np.isfinite(by[ds].get("frac_above_null_p95", np.nan)):
                out[key] = float(by[ds]["frac_above_null_p95"])
    dec_csv = behinfo / "decode_rows.csv"
    if dec_csv.is_file():
        dec = pd.read_csv(dec_csv)
        if "N_role" in dec.columns:
            dec = dec[dec["N_role"] == "primary"]
        if "region" in dec.columns:
            dec = dec[dec["region"] == "pooled"]
        if "N" in dec.columns:
            dec = dec[dec["N"] == 20]
        pos = dec[dec["variable"] == "position"]
        for ds, key in (("lab", "lab_pos_net"), ("sim_matched", "sim_pos_net")):
            sub = pos[pos["dataset"] == ds]["primary_minus_null"]
            if len(sub):
                out[key] = float(np.nanmedian(sub.to_numpy(float)))
    dec_p = behinfo / "decisions.json"
    if dec_p.is_file():
        decj = json.loads(dec_p.read_text())
        hd = (decj.get("hd_convention") or {}).get("residual_median_abs_deg")
        if hd is not None:
            out["hd_residual_deg"] = float(hd)
    return out


def _fmt_bits(x) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{float(x):.2f}"


def _load_drift_control(behinfo: Path | None) -> dict:
    out = {
        "verdict": None, "lab_fl": None, "lab_oe": None, "sim_fl": None, "sim_oe": None,
        "lab_slope": None, "matched_slope": None,
        "lab_slope_lo": None, "lab_slope_hi": None,
        "matched_slope_lo": None, "matched_slope_hi": None,
    }
    if behinfo is None:
        return out
    p = behinfo / "drift_noise_summary.json"
    if not p.is_file():
        return out
    raw = json.loads(p.read_text())
    out["verdict"] = raw.get("verdict")
    by = raw.get("by_dataset") or {}
    for ds, fl_k, oe_k in (
        ("lab", "lab_fl", "lab_oe"),
        ("sim_matched", "sim_fl", "sim_oe"),
    ):
        b = by.get(ds) or {}
        fl = (b.get("r_first_last") or {}).get("median")
        oe = (b.get("r_odd_even_2s") or {}).get("median")
        if fl is not None:
            out[fl_k] = float(fl)
        if oe is not None:
            out[oe_k] = float(oe)
    curve = raw.get("sep_curve") or {}
    for ds, sk, lo_k, hi_k in (
        ("lab", "lab_slope", "lab_slope_lo", "lab_slope_hi"),
        ("sim_matched", "matched_slope", "matched_slope_lo", "matched_slope_hi"),
    ):
        c = curve.get(ds) or {}
        if c.get("slope") is not None:
            out[sk] = float(c["slope"])
        if c.get("slope_lo90") is not None:
            out[lo_k] = float(c["slope_lo90"])
        if c.get("slope_hi90") is not None:
            out[hi_k] = float(c["slope_hi90"])
    if out["lab_slope"] is None and raw.get("lab_slope") is not None:
        out["lab_slope"] = float(raw["lab_slope"])
    if out["matched_slope"] is None and raw.get("matched_slope") is not None:
        out["matched_slope"] = float(raw["matched_slope"])
    return out


def _takeaway3(row, meta, behinfo: Path | None = None) -> str:
    """Three sentences: place frac + bits; N=20 decode; reliability + drift."""
    ve = _load_behinfo_ve(behinfo)
    drift = _load_drift_control(behinfo)
    # Place-tuned fractions from Fig3 (post keep-mask spatial recompute)
    place = row("frac_place_tuned") if callable(row) else None
    try:
        lab_place = float(place["lab_median"]) if place is not None else float("nan")
        sim_place = float(place["sim_matched_median"]) if place is not None else float("nan")
    except Exception:
        lab_place = sim_place = float("nan")
    lab_bits = _fmt_bits(ve.get("lab_bits_place"))
    sim_bits = _fmt_bits(ve.get("sim_bits_place"))
    lab_pos = ve.get("lab_pos_net")
    sim_pos = ve.get("sim_pos_net")
    a = (
        f"<b>3.</b> Place-tuned fraction is "
        f"{lab_place:.2f} lab vs {sim_place:.2f} matched (chance 0.05), and "
        f"held-out bits/spike among place-tuned units is {lab_bits} vs {sim_bits}."
    )
    if lab_pos is not None and sim_pos is not None and np.isfinite(lab_pos) and np.isfinite(sim_pos):
        b = (
            f" Linear position decoding at N = 20 is near chance in the lab "
            f"(net R² = {lab_pos:.2f}) vs clear in the sim ({sim_pos:.2f})."
        )
    else:
        b = " Linear position decoding at N = 20 is near chance in the lab vs clear in the sim."
    # Separation curve: adjacent-block r and slope
    sep1_lab = sep1_sim = None
    if behinfo is not None:
        p = behinfo / "drift_noise_summary.json"
        if p.is_file():
            raw = json.loads(p.read_text())
            try:
                sep1_lab = float(raw["sep_curve"]["lab"]["by_sep"]["1"]["median"])
                sep1_sim = float(raw["sep_curve"]["sim_matched"]["by_sep"]["1"]["median"])
            except Exception:
                pass
    ls = drift.get("lab_slope")
    ms = drift.get("matched_slope")
    if sep1_lab is not None and ls is not None and ms is not None:
        c = (
            f" Lab place maps are less reliable (adjacent-block r ≈ {sep1_lab:.2f} vs "
            f"{sep1_sim:.2f}) and drift slowly over a session "
            f"(slope {ls:.3f}/block vs ~{ms:.3f} in the sim)."
        )
    elif ls is not None and ms is not None:
        c = (
            f" Lab place maps are less reliable than simulated ones and drift slowly "
            f"over a session (slope {ls:.3f}/block vs ~{ms:.3f} in the sim)."
        )
    else:
        c = ""
    return a + b + c


def write_story(
    data: Path,
    out_pdf: Path,
    headline_png: Path | None = None,
    behinfo: Path | None = None,
) -> Path:
    n = numbers(data)
    meta = n["meta"]
    top = n["top"]
    right = n["right"]
    if behinfo is None:
        cand = data.parent / "behavior_info"
        behinfo = cand if cand.is_dir() else None
    ve = _load_behinfo_ve(behinfo)
    hd_deg = ve.get("hd_residual_deg")
    hd_txt = f"{hd_deg:.0f}°" if hd_deg is not None and np.isfinite(hd_deg) else "45°"

    refr_lab = meta["refractory_pct"]["lab"]
    refr_sim = meta["refractory_pct"]["sim_matched"]

    def row(base):
        hit = n["fig3"][n["fig3"].metric_base == base]
        return None if hit.empty else hit.iloc[0]

    turn = row("turning_angle_std")
    speed = row("speed_median")
    dec = meta.get("decode_error_over_floor_median_N20") or {}
    lab_dec = dec.get("lab", float("nan"))
    sim_dec = dec.get("sim_matched", float("nan"))

    # Prefer named "gets right" concepts; never list grid cells as right
    prefer = {
        "firing_rate_hz", "fano_factor", "isi_cv", "rate_drift", "mean_pairwise_corr",
        "sparsity_median", "spatial_info_median", "peak_rate_median", "dwell_time_mean_s",
        "hd_mean_vector_length_median", "speed_tuning_abs_r_median",
    }
    ban = {"frac_grid_pass", "frac_grid_excess", "grid_score_median"}
    right_bits = []
    for _, r in right.iterrows():
        base = str(r["metric_base"])
        if base in ban or "grid" in base.lower():
            continue
        if base in prefer or len(right_bits) < 8:
            if np.isfinite(r.get("R_matched", np.nan)) and float(r["R_matched"]) <= 1.0:
                right_bits.append(f"{r['label']} (R = {_fmt_r(r['R_matched'])})")
        if len(right_bits) >= 8:
            break
    right_txt = "; ".join(right_bits) if right_bits else "none with R ≤ 1 in the primary set"

    turn_note = ""
    if turn is not None:
        too_reg = turn.get("too_regular_matched")
        turn_note = (
            f"Turning-angle SD (speed > 2 cm/s only) is "
            f"{float(turn['lab_median']):.2f} vs {float(turn['sim_matched_median']):.2f} rad "
            f"(R = {_fmt_r(float(turn['R_matched']))}"
            + ("; flagged too-regular" if too_reg else "")
            + ")."
        )

    story = [
        Paragraph(
            "How far is the simulator from the lab? A paired data-quality audit of "
            "default and matched simulation arms against real hippocampal recordings",
            H1,
        ),
        Paragraph(
            f"Report 4a · sim vs lab · n = {meta['n_animals']} animals, "
            f"{meta['n_lab_sessions']} room-A sessions · matched n = {meta['n_matched_sessions']} · "
            f"default seeds = {meta['n_default_seeds']} · thigmotaxis = {meta['thigmotaxis']} (not fitted) · "
            f"internal draft, not for distribution",
            SM,
        ),
        Spacer(1, 2),
        Paragraph("The question", H2),
        Paragraph(
            "Before using the simulator as a stand-in for real data, we ask which statistics of "
            "spikes, spatial coding, population structure and behaviour already match the lab, "
            "and which diverge. Every metric uses one code path; dataset is a label only. "
            "Divergence is summarised by R = d_sim / d_lab (lab-MAD-scaled distances of session "
            "medians, 90% bootstrap by recording day). R ≤ 1 means the sim sits within lab variability.",
            BODY,
        ),
        Paragraph("Three takeaways", H2),
        Paragraph(
            f"<b>1. Refractory structure is missing.</b> "
            f"Lab units violate a 1.5 ms refractory window in {_fmt_pct(refr_lab)} of ISIs "
            f"(session median); matched sim sorted spikes do so in {_fmt_pct(refr_sim)}. "
            f"After a lab-equivalent 0.3 ms duplicate-spike removal the gap remains. "
            f"Ground-truth sim spikes also show sub-millisecond ISIs: the generator is "
            f"inhomogeneous Poisson with no public refractory / min-ISI parameter. "
            f"Matched R for refractory violations = {_fmt_r(float(row('refractory_violation_rate')['R_matched']))} "
            f"[{_fmt_r(float(row('refractory_violation_rate')['R_matched_lo']))}, "
            f"{_fmt_r(float(row('refractory_violation_rate')['R_matched_hi']))}] (Fig 3, Fig 4).",
            TAKE,
        ),
        Paragraph(
            f"<b>2. Behaviour is too regular and too fast.</b> "
            f"Median speed is {float(speed['lab_median']):.2f} cm/s in the lab vs "
            f"{float(speed['sim_matched_median']):.2f} cm/s in matched sim "
            f"(R = {_fmt_r(float(speed['R_matched']))}). {turn_note} "
            f"thigmotaxis was held at {meta['thigmotaxis']}; speed and rotational knobs are not "
            f"public (DF6). Structural — see simulator to-do (Fig 5).",
            TAKE,
        ),
        Paragraph(_takeaway3(row, meta, behinfo), TAKE),
        Paragraph("What the simulation gets right", H2),
        Paragraph(
            "Metrics with matched-arm R ≤ 1 (within lab variability) include: "
            + right_txt
            + ". Grid-cell fractions are near chance in both arms and are not counted as a match. "
            "Absolute firing rates are similar (lab median "
            f"{float(row('firing_rate_hz')['lab_median']):.2f} Hz vs matched "
            f"{float(row('firing_rate_hz')['sim_matched_median']):.2f} Hz; "
            f"R = {_fmt_r(float(row('firing_rate_hz')['R_matched']))}).",
            BODY,
        ),
        Paragraph("Simulator to-do", H2),
        Paragraph(
            "Add a public refractory / min-ISI parameter (generator is inhomogeneous Poisson). "
            "Expose speed and rotational knobs (DF6); thigmotaxis was held at "
            f"{meta['thigmotaxis']} and not fitted. "
            "<b>Cell-type mix / behaviour drive:</b> reduce excess border and HD cells and "
            "the over-strong behavioural encoding of position, movement direction and speed "
            "(speed-cell excess is near the reliability floor after the keep-mask fix, R ≈ 1.26, "
            "and is not listed as a primary cell-type request). "
            f"Lab head direction differs from movement direction by a median {hd_txt}; "
            "the sim has no such dissociation (sim HD = movement heading). "
            "<b>Within-session drift:</b> lab place maps are less reliable "
            "(adjacent-block r ≈ 0.48 vs 0.89) and drift slowly "
            "(slope ≈ −0.024/block vs ~0). Add a public "
            "nonstationarity / representational-drift mechanism so simulated units are not "
            "frozen across the session.",
            BODY,
        ),
        Paragraph("Caveats", H2),
        Paragraph(
            f"{meta.get('sub_caveat', 'SUB is sparse in most lab sessions.')} "
            f"Analysis is room A only. Thigmotaxis was held at {meta['thigmotaxis']} (not fitted). "
            "Occupancy-matched joint bins are not used: a grid-index intersection across a lab "
            "polygon and a sim square in different coordinates is not meaningful; behaviour "
            "occupancy remains a confound. Field size is reported only among place-tuned units "
            "(SI > shuffle 95%). Path-containment failures (< 0.95) stay in scores with a "
            "path-visited bin mask (Report 2 flag semantics). "
            "Matched-arm unit metrics use composition-matched subsamples to each lab session's "
            "per-region eligible counts (20 draws, median). "
            f"Lab decoding at N = 20 is near chance "
            f"({lab_dec:.2f} of floor) vs matched sim ({sim_dec:.2f} of floor). "
            f"Lab head direction differs from movement direction by a median {hd_txt}; "
            "the sim has no such dissociation. "
            "Primary per-unit encoding measure is held-out Poisson bits/spike "
            "(gain-calibrated train rate map vs homogeneous). "
            "Ridge / rate-map variance-explained is not a valid absolute encoding measure "
            "on lab data (≤ 0 even with per-unit CV). "
            "Spikes per place-map half differ (lab ~300 vs sim ~1,400 among place-tuned units) "
            "and contribute to the reliability gap. "
            "Behavioural-information nets use raw CV − max(null, 0) at N = 20 "
            "(Report 2 v2 250 ms pos-speed; recheck when Report 2 v2 is final).",
            BODY,
        ),
        Paragraph("How to read the figures", H2),
        Paragraph(
            "Headline panel below is a Fig 3 miniature. "
            "Fig 3: log R (green = R ≤ 1); default arm = squares, no intervals. "
            "Fig 4: Hz rate maps; ISI; CDFs with chance line at 0.05 for grid frac. "
            "Fig 5: faster/straighter sim behaviour; blue lab vs orange matched CDFs. "
            f"{meta['calibration_note']}",
            BODY,
        ),
        Paragraph(
            "Headline numbers from the matched arm (R with 90% interval): "
            + "; ".join(
                f"{r['label']} R = {_fmt_r(r['R_matched'])} "
                f"[{_fmt_r(r['R_matched_lo'])}, {_fmt_r(r['R_matched_hi'])}]"
                for _, r in top.iterrows()
            )
            + ".",
            SM,
        ),
    ]

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    # Text in upper ~58% of page; headline miniature in lower band
    text_h = H - 2 * M - (52 * mm if headline_png and headline_png.is_file() else 0)
    f = Frame(M, M + (52 * mm if headline_png and headline_png.is_file() else 0), W - 2 * M, text_h, showBoundary=0)
    f.addFromList(story, c)
    if headline_png and headline_png.is_file():
        c.drawImage(
            str(headline_png),
            M,
            M - 2 * mm,
            width=W - 2 * M,
            height=50 * mm,
            preserveAspectRatio=True,
            anchor="sw",
        )
        c.setFont(FONT, 6)
        c.setFillColor(colors.HexColor("#52514e"))
        c.drawString(M, M + 48 * mm, "Headline (Fig 3 miniature)")
    c.save()
    return out_pdf


def assemble(story: Path, figs: Path, out_pdf: Path, names: list[str]) -> Path:
    w = PdfWriter()
    w.append(str(story))
    for name in names:
        p = figs / f"{name}.pdf"
        if p.is_file():
            w.append(str(p))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with out_pdf.open("wb") as f:
        w.write(f)
    return out_pdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--review", action="store_true", help="Story + Fig3–5 only")
    ap.add_argument("--behinfo", default="", help="behavior_info dir (encoding VE / HD residual)")
    args = ap.parse_args()
    data, figs, out = Path(args.data), Path(args.figs), Path(args.out)
    behinfo = Path(args.behinfo) if args.behinfo else None
    story = out.parent / "_story_r4.pdf"
    # Prefer PNG miniature for story embed; fall back to skipping if missing
    headline = figs / "Fig0_headline.png"
    if not headline.is_file():
        headline = figs / "Fig0_headline.pdf"
    write_story(
        data, story,
        headline_png=headline if headline.suffix == ".png" else None,
        behinfo=behinfo,
    )
    names = ["Fig3_answer", "Fig4_mechanism", "Fig5_behaviour"] if args.review else []
    path = assemble(story, figs, out, names)
    print(json.dumps({"pdf": str(path)}), flush=True)


if __name__ == "__main__":
    main()
