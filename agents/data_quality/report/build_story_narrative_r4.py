#!/usr/bin/env python3
"""Narrative story page for Report 4a v7 — Fig 3a (50 mm) top-right; text wraps then fills below."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Frame, Paragraph, Preformatted, Table, TableStyle

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
M = 9 * mm
BODY = ParagraphStyle("b", fontName=FONT, fontSize=8.8, leading=11.0, spaceAfter=2.5)
H1 = ParagraphStyle("h1", fontName=BOLD, fontSize=10.5, leading=12.5, spaceAfter=2.5)
H2 = ParagraphStyle("h2", fontName=BOLD, fontSize=8.8, leading=10.8, spaceBefore=2.5, spaceAfter=2.0)
SM = ParagraphStyle("s", fontName=FONT, fontSize=6.5, leading=8.0, textColor=colors.HexColor("#52514e"), spaceAfter=2.0)
TAKE = ParagraphStyle("t", fontName=FONT, fontSize=8.6, leading=10.8, spaceAfter=3.0, leftIndent=1)
COL = ParagraphStyle("col", fontName=FONT, fontSize=7.2, leading=9.0, spaceAfter=1.5)
PRE = ParagraphStyle("pre", fontName="Courier", fontSize=5.9, leading=8.0, spaceAfter=0.5)
PRE_H = ParagraphStyle("preh", fontName=BOLD, fontSize=7.5, leading=9.0, spaceBefore=1.0, spaceAfter=1.0)


def _fig3_row(fig3: pd.DataFrame, base: str):
    hit = fig3[fig3.metric_base == base]
    return None if hit.empty else hit.iloc[0]


def write_story_narrative(
    data: Path,
    out_pdf: Path,
    headline_png: Path | None = None,
    behinfo: Path | None = None,
    fill_png: Path | None = None,  # ignored — hybrid fill removed
) -> Path:
    del fill_png
    meta = json.loads((data / "data_meta.json").read_text())
    fig3 = pd.read_csv(data / "data_fig3_divergence.csv")
    if behinfo is None:
        cand = data.parent / "behavior_info"
        behinfo = cand if cand.is_dir() else None

    refr = _fig3_row(fig3, "refractory_violation_rate")
    speed = _fig3_row(fig3, "speed_median")
    turn = _fig3_row(fig3, "turning_angle_std")
    place = _fig3_row(fig3, "frac_place_tuned")
    peak = _fig3_row(fig3, "peak_rate_median")
    fano = _fig3_row(fig3, "fano_factor")
    isi_cv = _fig3_row(fig3, "isi_cv")
    rate_drift = _fig3_row(fig3, "rate_drift")
    spars = _fig3_row(fig3, "sparsity_median")
    field = _fig3_row(fig3, "place_field_size_median")

    lab_bits = sim_bits = None
    lab_pos = sim_pos = None
    sep1_lab = sep1_sim = None
    lab_slope = mat_slope = None
    if behinfo and (behinfo / "encoding_diag_summary.json").is_file():
        table = json.loads((behinfo / "encoding_diag_summary.json").read_text()).get("table") or {}
        lab_bits = (table.get("lab") or {}).get("bits_end_med_place")
        sim_bits = (table.get("sim_matched") or {}).get("bits_end_med_place")
    if behinfo and (behinfo / "decode_rows.csv").is_file():
        dec = pd.read_csv(behinfo / "decode_rows.csv")
        if "N_role" in dec.columns:
            dec = dec[dec.N_role == "primary"]
        if "region" in dec.columns:
            dec = dec[dec.region == "pooled"]
        if "N" in dec.columns:
            dec = dec[dec.N == 20]
        pos = dec[dec.variable == "position"]
        for ds in ("lab", "sim_matched"):
            sub = pos[pos.dataset == ds]["primary_minus_null"]
            if len(sub):
                if ds == "lab":
                    lab_pos = float(np.nanmedian(sub))
                else:
                    sim_pos = float(np.nanmedian(sub))
    if behinfo and (behinfo / "drift_noise_summary.json").is_file():
        raw = json.loads((behinfo / "drift_noise_summary.json").read_text())
        try:
            sep1_lab = float(raw["sep_curve"]["lab"]["by_sep"]["1"]["median"])
            sep1_sim = float(raw["sep_curve"]["sim_matched"]["by_sep"]["1"]["median"])
            lab_slope = float(raw["sep_curve"]["lab"]["slope"])
            mat_slope = float(raw["sep_curve"]["sim_matched"]["slope"])
        except Exception:
            pass

    refr_lab = meta["refractory_pct"]["lab"]
    refr_sim = meta["refractory_pct"]["sim_matched"]
    bits_l = f"{lab_bits:.2f}" if lab_bits is not None else "—"
    bits_s = f"{sim_bits:.2f}" if sim_bits is not None else "—"
    pos_l = f"{lab_pos:.2f}" if lab_pos is not None else "—"
    pos_s = f"{sim_pos:.2f}" if sim_pos is not None else "—"

    def _abs(row, fmt=".3g"):
        if row is None:
            return "—", "—", "—"
        return (
            f"{float(row['lab_median']):{fmt}}",
            f"{float(row['sim_matched_median']):{fmt}}",
            f"{float(row['R_matched']):.2f}",
        )

    take1 = (
        f"<b>1. Sim populations carry far more behavioural information (Fig 3).</b> "
        f"bits_ratemap_end among place-tuned units: {bits_l} lab vs {bits_s} matched (Fig 2d). "
        f"Position info at N = 20 is near chance in lab (net {pos_l}) vs clear in matched ({pos_s}; Fig 3a). "
        f"Lab population VE ≤ 0 — not a valid absolute; prefer bits/spike. "
        f"Population decode error/floor at N = 20 is higher in lab than matched (Fig 3e–f); "
        f"matched climbs with N while lab stays near chance until larger N (Fig 3b)."
    )
    take2 = (
        f"<b>2. Sim units are tuned more often, more strongly, and never drift (Fig 2).</b> "
        f"Place-tuned fraction {_abs(place)[0]} lab vs {_abs(place)[1]} matched "
        f"(chance 0.05; Fig 2b; R = {_abs(place)[2]}). "
        f"Adjacent-block map r ≈ {sep1_lab:.2f} vs {sep1_sim:.2f}"
        + (f"; drift slope {lab_slope:.3f}/block vs ~{mat_slope:.3f}" if lab_slope is not None else "")
        + " (Fig 2f–g). "
        f"Peak rate {_abs(peak)[0]} vs {_abs(peak)[1]} Hz; sparsity {_abs(spars)[0]} vs {_abs(spars)[1]}; "
        f"field size {_abs(field)[0]} vs {_abs(field)[1]} (Fig 4)."
    )
    take3 = (
        f"<b>3. No refractory period; behaviour too fast and too straight (Fig 1).</b> "
        f"Lab units violate a 1.5 ms window in {refr_lab:.2f}% of ISIs; matched sim {refr_sim:.2f}% "
        f"(R = {float(refr['R_matched']):.1f}; Fig 1c). "
        f"Median speed {float(speed['lab_median']):.2f} vs {float(speed['sim_matched_median']):.2f} cm/s; "
        f"turning SD {float(turn['lab_median']):.2f} vs {float(turn['sim_matched_median']):.2f} rad "
        f"(Fig 1e–f). "
        f"Fano {_abs(fano)[0]} vs {_abs(fano)[1]}; ISI CV {_abs(isi_cv)[0]} vs {_abs(isi_cv)[1]}; "
        f"rate drift {_abs(rate_drift)[0]} vs {_abs(rate_drift)[1]} (Fig 4 paired strips)."
    )
    methods = (
        f"<b>Methods.</b> One code path; dataset is a label. "
        f"R = d_sim/d_lab with 90% day bootstrap (Fig 5a only). "
        f"Room A; thigmotaxis = {meta['thigmotaxis']} (held). Exact keep-mask. "
        f"<b>bits_ratemap_end</b> = held-out Poisson log-likelihood ratio of the train rate map "
        f"(gain-calibrated on train) vs a homogeneous train rate λ0, per spike: "
        f"ΔLL = Σ[n log(λ/λ0) − (λ−λ0)Δt]; bits/spike = ΔLL/(Nspk ln 2); "
        f"reported as the median among place-tuned units (bits_end_med_place = 0.07 | 0.43). "
        f"Decode info = primary − max(null, 0) at N = 20 unless noted; N-ladder "
        f"{{10, 20, 40, 80, 160, max}} when the ladder table is present. "
        f"Numbers frozen at v5 tidy (+ ladder table when present)."
    )
    caveats = (
        "<b>Caveats.</b> No occ-matched joint bins (lab polygon ≠ sim square). "
        "Lab population VE ≤ 0 — not a valid absolute measure; cite bits/spike and "
        "decode info above null instead. "
        "Fig 2a rate maps = stored SI p25/p50/p75 (6/arm), not population p10–p90. "
        "Matched arm holds thigmotaxis and keep-mask; residual divergence is therefore "
        "not explained by those two levers alone. "
        "Calibration deferred to Report 4b. "
        "Default arm (n = 5 seeds) is shown for orientation; primary claims use lab vs matched."
    )

    has_hl = bool(headline_png and headline_png.is_file())
    hl_w = 105 * mm if has_hl else 0
    hl_h = 50 * mm if has_hl else 0
    gap = 2.0 * mm
    title_band = 22 * mm

    col_w = (W - 2 * M - 3 * mm) / 2
    two_col = Table(
        [[Paragraph(methods, COL), Paragraph(caveats, COL)]],
        colWidths=[col_w, col_w],
    )
    two_col.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (0, 0), 2.5 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

    # Beside image: question + takeaways start (wrap)
    side_flow = [
        Paragraph("The question", H2),
        Paragraph(
            "Which spike, spatial, population and behaviour statistics already match the lab, "
            "and which diverge? Absolute values lead (Figs 1–4); divergence ratio R is Fig 5 only.",
            BODY,
        ),
        Paragraph("Three takeaways", H2),
        Paragraph(take1, TAKE),
    ]

    # Frozen numbers cited in takeaways/methods (text only — no bar hybrid).
    # Pad each Courier line with middle-dots so dilated ink spans the full
    # content width (a narrow table left the right third blank).
    def _wide(s: str, width: int = 156) -> str:
        s = s.rstrip()
        if len(s) >= width:
            return s[:width]
        return s + " " + ("·" * (width - len(s) - 1))

    abs_lines = [
        _wide(f"{'metric':<36} {'lab':>8} {'matched':>8} {'R':>6}"),
        _wide("-" * 62),
    ]
    for _, r in fig3.dropna(subset=["R_matched"]).sort_values("R_matched", ascending=False).iterrows():
        lab = str(r["label"])[:36]
        abs_lines.append(_wide(
            f"{lab:<36} {float(r.lab_median):>8.3g} {float(r.sim_matched_median):>8.3g} {float(r.R_matched):>6.2f}"
        ))
    abs_lines.append(_wide(
        f"{'bits_ratemap_end (place-tuned)':<36} "
        f"{float(lab_bits) if lab_bits is not None else float('nan'):>8.2f} "
        f"{float(sim_bits) if sim_bits is not None else float('nan'):>8.2f} {'—':>6}"
    ))
    if lab_pos is not None and sim_pos is not None:
        abs_lines.append(_wide(
            f"{'position info N=20 (above null)':<36} {lab_pos:>8.2f} {sim_pos:>8.2f} {'—':>6}"
        ))
    if behinfo and (behinfo / "encoding_diag_summary.json").is_file():
        et = json.loads((behinfo / "encoding_diag_summary.json").read_text()).get("table") or {}
        abs_lines.append(_wide("-" * 62))
        for key, label in (
            ("ve_ratemap_end_med_place", "VE ratemap end"),
            ("ve_ridge_cv_end_med_place", "VE ridge CV end"),
            ("frac_place", "frac place (enc)"),
            ("bits_end_med_place", "bits_end_med_place"),
            ("bits_end_med_all", "bits_end_med_all"),
            ("half_map_r_place", "half-map r (place)"),
            ("half_map_r_all", "half-map r (all)"),
            ("alpha_cv_med", "ridge alpha CV med"),
            ("n_units", "n units (enc)"),
            ("n_sessions", "n sessions (enc)"),
        ):
            lv = (et.get("lab") or {}).get(key)
            mv = (et.get("sim_matched") or {}).get(key)
            if lv is None and mv is None:
                continue
            abs_lines.append(_wide(
                f"{label:<36} "
                f"{float(lv) if lv is not None else float('nan'):>8.3g} "
                f"{float(mv) if mv is not None else float('nan'):>8.3g} {'—':>6}"
            ))
    if behinfo and (behinfo / "decode_rows.csv").is_file():
        dec = pd.read_csv(behinfo / "decode_rows.csv")
        if "N_role" in dec.columns:
            dec = dec[dec.N_role == "primary"]
        if "region" in dec.columns:
            dec = dec[dec.region == "pooled"]
        abs_lines.append(_wide("-" * 62))
        for var in (
            "position", "wall_distance_bin2", "movement_direction",
            "speed", "movement_state", "head_direction",
        ):
            vals = {}
            for ds in ("lab", "sim_matched"):
                sub = dec[(dec.dataset == ds) & (dec.variable == var)]
                if "N" in sub.columns:
                    sub = sub[sub.N == 20]
                col = "primary_minus_null"
                if col not in sub.columns or sub.empty:
                    vals[ds] = float("nan")
                else:
                    vals[ds] = float(np.nanmedian(sub[col]))
            abs_lines.append(_wide(
                f"{('dec '+var)[:36]:<36} {vals['lab']:>8.2f} {vals['sim_matched']:>8.2f} {'—':>6}"
            ))
    abs_block = "\n".join(abs_lines)

    bot_flow = [
        Paragraph(take2, TAKE),
        Paragraph(take3, TAKE),
        Paragraph(
            "What the simulator gets right: paired strips in Fig 4 "
            "(firing rate, Fano, ISI CV, rate drift, pairwise correlation, sparsity, field size, peak rate).",
            BODY,
        ),
        two_col,
        Paragraph("Frozen absolutes cited above (lab | matched | R) — text; R forest is Fig 5a", PRE_H),
        Preformatted(abs_block, PRE),
    ]
    title_flow = [
        Paragraph(
            "How far is the simulator from the lab? A paired data-quality audit of "
            "default and matched arms against hippocampal recordings",
            H1,
        ),
        Paragraph(
            f"Report 4a · n = {meta['n_animals']} animals, {meta['n_lab_sessions']} room-A · "
            f"matched {meta['n_matched_sessions']} · default {meta['n_default_seeds']} · "
            f"thigmotaxis = {meta['thigmotaxis']} · internal draft · numbers frozen (v5 tidy)",
            SM,
        ),
    ]

    c = canvas.Canvas(str(out_pdf), pagesize=A4)
    img_y = H - M - title_band - hl_h
    side_w = (W - 2 * M - hl_w - gap) if has_hl else (W - 2 * M)
    bot_h = img_y - M - gap

    f_title = Frame(M, H - M - title_band, W - 2 * M, title_band, showBoundary=0,
                    leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f_side = Frame(M, img_y, side_w, hl_h, showBoundary=0,
                   leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    f_bot = Frame(M, M, W - 2 * M, bot_h, showBoundary=0,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    rem = f_title.addFromList(list(title_flow), c) or []
    rem = f_side.addFromList(list(rem) + list(side_flow), c) or []
    rem = f_bot.addFromList(list(rem) + list(bot_flow), c) or []
    if rem:
        c.setFont(FONT, 6)
        c.setFillColor(colors.HexColor("#c83a39"))
        c.drawString(M, M + 1 * mm, f"[story overflow: {len(rem)} blocks]")

    if has_hl:
        c.drawImage(
            str(headline_png), W - M - hl_w, img_y,
            width=hl_w, height=hl_h, preserveAspectRatio=True, anchor="sw",
        )
    c.save()
    return out_pdf
