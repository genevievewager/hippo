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
from reportlab.platypus import Frame, Paragraph, Spacer, Table, TableStyle

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
# v6: ≤ 9 pt body; tight leading so the story fits one page with headline below
BODY = ParagraphStyle("b", fontName=FONT, fontSize=8.5, leading=10.2, spaceAfter=2.5)
H1 = ParagraphStyle("h1", fontName=BOLD, fontSize=11.0, leading=13.0, spaceAfter=3)
H2 = ParagraphStyle("h2", fontName=BOLD, fontSize=8.5, leading=10.0, spaceBefore=3, spaceAfter=1.5)
SM = ParagraphStyle("s", fontName=FONT, fontSize=6.5, leading=8.0, textColor=colors.HexColor("#52514e"), spaceAfter=2)
TAKE = ParagraphStyle("t", fontName=FONT, fontSize=8.5, leading=10.2, spaceAfter=2.5, leftIndent=6)
COL = ParagraphStyle("col", fontName=FONT, fontSize=7.5, leading=9.2, spaceAfter=1.5)


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

    rr = row("refractory_violation_rate")
    take1 = (
        f"<b>1. Refractory structure is missing.</b> "
        f"Lab units violate a 1.5 ms refractory window in {_fmt_pct(refr_lab)} of ISIs; "
        f"matched sim does so in {_fmt_pct(refr_sim)} (R = {_fmt_r(float(rr['R_matched']))}). "
        f"The gap survives a 0.3 ms lab-equivalent censor. "
        f"The generator is inhomogeneous Poisson with no public refractory / min-ISI knob."
    )
    take2 = (
        f"<b>2. Behaviour is too regular and too fast.</b> "
        f"Median speed is {float(speed['lab_median']):.2f} vs {float(speed['sim_matched_median']):.2f} cm/s "
        f"(R = {_fmt_r(float(speed['R_matched']))}). "
        + (
            f"Turning-angle SD is {float(turn['lab_median']):.2f} vs {float(turn['sim_matched_median']):.2f} rad "
            f"(R = {_fmt_r(float(turn['R_matched']))}). "
            if turn is not None else ""
        )
        + f"thigmotaxis was held at {meta['thigmotaxis']}; speed / rotational knobs are not public (DF6)."
    )

    methods_txt = (
        f"<b>Methods.</b> One code path; R = d_sim/d_lab with 90% day bootstrap. "
        f"Room A; thigmotaxis = {meta['thigmotaxis']} (held). Exact keep-mask. "
        f"Fields among place-tuned only. Decode N=20: lab {lab_dec:.2f} vs matched {sim_dec:.2f} of floor. "
        f"{meta.get('sub_caveat', 'SUB sparse.')}"
    )
    caveats_txt = (
        "<b>Caveats.</b> No occ-matched joint bins (lab polygon ≠ sim square). "
        "Ridge VE invalid as absolute encoding on lab. "
        "Spikes/half ~300 lab vs ~1,400 sim. Calibration deferred to Report 4b."
    )
    col_w = (W - 2 * M - 4 * mm) / 2
    two_col = Table(
        [[Paragraph(methods_txt, COL), Paragraph(caveats_txt, COL)]],
        colWidths=[col_w, col_w],
    )
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 3 * mm),
        ("RIGHTPADDING", (1, 0), (1, 0), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    story = [
        Paragraph(
            "How far is the simulator from the lab? Paired data-quality audit of "
            "default and matched arms against hippocampal recordings",
            H1,
        ),
        Paragraph(
            f"Report 4a · n = {meta['n_animals']} animals, {meta['n_lab_sessions']} room-A · "
            f"matched {meta['n_matched_sessions']} · default {meta['n_default_seeds']} · "
            f"thigmotaxis = {meta['thigmotaxis']} · internal draft",
            SM,
        ),
        Paragraph("The question", H2),
        Paragraph(
            "Which spike, spatial, population and behaviour statistics already match the lab? "
            "One code path; dataset is a label. R = d_sim/d_lab; R ≤ 1 = within lab variability.",
            BODY,
        ),
        Paragraph("Three takeaways", H2),
        Paragraph(take1, TAKE),
        Paragraph(take2, TAKE),
        Paragraph(_takeaway3(row, meta, behinfo), TAKE),
        Paragraph("What the simulation gets right", H2),
        Paragraph(
            "Matched R ≤ 1: " + right_txt
            + f". Firing rate {float(row('firing_rate_hz')['lab_median']):.2f} vs "
            f"{float(row('firing_rate_hz')['sim_matched_median']):.2f} Hz. Grid near chance — not a match.",
            BODY,
        ),
        Paragraph("Simulator to-do", H2),
        Paragraph(
            "<b>1</b> Refractory · <b>2</b> Speed/turning or external traj (DF6) · "
            f"<b>3</b> Independent HD (~{hd_txt}) · <b>4</b> Tuned-cell prevalence + bits/spike · "
            "<b>5</b> Within-session drift · <b>6</b> Capture cap (DF7) · <b>7</b> SimConfig.seed (DF5).",
            BODY,
        ),
        two_col,
        Paragraph(
            "Headline below: takeaways only. Fig 1 schematic · Fig 2 gates · Fig 3 forest · "
            "Fig 4 maps · Fig 5 behaviour · Fig 6 place/peak · Fig 7 behaviour info.",
            SM,
        ),
    ]

    hl_h = 48 * mm if headline_png and headline_png.is_file() else 0
    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    # Text frame sits directly above the headline — no blank gap
    text_bottom = M + hl_h + (2 * mm if hl_h else 0)
    text_h = H - M - text_bottom
    f = Frame(M, text_bottom, W - 2 * M, text_h, showBoundary=0,
              leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    remaining = f.addFromList(list(story), c)
    if remaining:
        # Overflow: shrink by dropping spacer content is already tight; warn via canvas note
        c.setFont(FONT, 6)
        c.setFillColor(colors.HexColor("#c83a39"))
        c.drawString(M, text_bottom + 1 * mm, f"[story overflow: {len(remaining)} blocks]")
    if headline_png and headline_png.is_file():
        c.drawImage(
            str(headline_png),
            M, M,
            width=W - 2 * M,
            height=hl_h,
            preserveAspectRatio=True,
            anchor="sw",
        )
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
