"""Story, limitations, and methods text pages for Report 2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis.real_quadrant.report import data as D
from analysis.real_quadrant.report.style import MUTED, W_FULL, save_fig


def _text_page(out_dir: Path, name: str, title: str, lines: list[str]) -> Path:
    fig = plt.figure(figsize=(W_FULL, W_FULL * 1.15))
    fig.text(0.06, 0.95, title, fontsize=11, fontweight="bold", va="top")
    y = 0.90
    for line in lines:
        weight = "bold" if line.startswith("•") or line.startswith("#") else "normal"
        size = 7.2 if not line.startswith("  ") else 6.5
        fig.text(0.06, y, line, fontsize=size, fontweight=weight, va="top",
                 color="#0b0b0b" if weight == "bold" else MUTED)
        y -= 0.028 if line else 0.018
        if y < 0.05:
            break
    return save_fig(fig, out_dir, name)


def page_story(out_dir: Path) -> Path:
    r2 = D.load_report2_contrasts()
    final = {c["contrast"]: c for c in r2["primary_contrasts_final"]}
    g20 = {c["contrast"]: c for c in r2["primary_contrasts_grid20"]}
    # dm contrast uses grid20
    dm = g20["dm_smooth - pca_smooth"]
    lines = [
        "Report 2 — five-point answer (real data, room A, frozen final grid)",
        "",
        "1. Temporal integration helps.",
        f"   raw_smooth − raw: mean={final['raw_smooth - raw']['mean']:+.3f}, "
        f"{final['raw_smooth - raw']['n_a_better']}/{final['raw_smooth - raw']['n_b_better']} "
        f"(N={final['raw_smooth - raw']['n_animals']}).",
        "   Causal EMA on the full population beats unsmoothed raw on every animal.",
        "   Cite: Figure RD7; supporting CDFs in RD6.",
        "",
        "2. Static compression does not beat the full population.",
        f"   pca_smooth − raw_smooth: mean={final['pca_smooth - raw_smooth']['mean']:+.3f}, "
        f"{final['pca_smooth - raw_smooth']['n_a_better']}/{final['pca_smooth - raw_smooth']['n_b_better']}.",
        "   PCA+EMA loses to raw+EMA on most animals.",
        "   Cite: Figure RD7; latents RD4; predictions RD5.",
        "",
        "3. DM does not beat PCA at matched d (grid20).",
        f"   dm_smooth − pca_smooth (grid20 both arms): mean={dm['mean']:+.3f}, "
        f"{dm['n_a_better']}/{dm['n_b_better']}.",
        "   DM was not extended past d=20; comparison locked to grid20.",
        "   Cite: Figure RD7 (*).",
        "",
        "4. LDS + smoothing ties raw_smooth.",
        f"   lds_smooth − raw_smooth: mean={final['lds_smooth - raw_smooth']['mean']:+.3f}, "
        f"{final['lds_smooth - raw_smooth']['n_a_better']}/{final['lds_smooth - raw_smooth']['n_b_better']} "
        "(sign flipped vs grid20).",
        "   Cite: Figure RD7 [flip]; lag convention on session reports / methods page.",
        "",
        "5. Causal GPFA loses to raw_smooth.",
        f"   gpfa_causal − raw_smooth: mean={final['gpfa_causal - raw_smooth']['mean']:+.3f}, "
        f"{final['gpfa_causal - raw_smooth']['n_a_better']}/{final['gpfa_causal - raw_smooth']['n_b_better']}.",
        "   Offline GPFA smoother is reference-only (excluded from contrasts).",
        "   Cite: Figure RD7; offline mark on RD4–RD6.",
        "",
        "RD8 (sim vs real) lives in the Report 1 revision. RD9 (region subsets)",
        "is future work — would require new fits.",
    ]
    return _text_page(out_dir, "story", "Story", lines)


def page_limitations(out_dir: Path) -> Path:
    cohort = D.load_cohort()
    ext = D.extent_exclusion_count(cohort)
    singles = [a["animal"] for a in cohort["animals"] if a.get("n_selected", 0) < 2]
    plateau = D.load_report2_contrasts().get("plateau_check") or {}
    lines = [
        "Limitations (Report 2)",
        "",
        f"• Extent exclusions: {ext} sessions removed for path-extent failures",
        "  before cohort selection (recorded in cohort_manifest).",
        f"• Single-session animals: {', '.join(singles) or 'none'} "
        "(fewer than 2 eligible).",
        "• Room A only; first-exposure / primary room — B and a not reported here.",
        "• DM, Isomap, and offline GPFA were not extended past d=20;",
        "  dm_smooth − pca_smooth therefore uses grid20 rows for both arms.",
        "• PCA/LDS selection sits at d=80 on ~half the sessions; median error",
        "  40→80 improvement < 0.01 (plateau stop). "
        f"pca Δ={plateau.get('pca', {}).get('improvement_40_to_80')}, "
        f"lds Δ={plateau.get('lds', {}).get('improvement_40_to_80')}.",
        "• Isomap failed on 1 session (disconnected neighbor graph); baseline only.",
        "• No new methods, controls, or reruns after the plateau freeze (PLAN).",
        "• RD9 region-subset decoding deferred (would need new fitting).",
    ]
    return _text_page(out_dir, "limitations", "Limitations", lines)


def page_methods(out_dir: Path) -> Path:
    lines = [
        "Methods (Report 2)",
        "",
        "• Features: causal spike counts in [t − 250 ms, t) rebuilt from spike",
        "  times (never Cell_* as features). Cell_* is a centred 250 ms window",
        "  (±125 ms) — see RD2 panels c–d for the lookahead.",
        "• Segment: room A; trim first 60 s and last 10 s; LDS/raw_lag warm-start",
        "  from segment start; fit/eval on retained ∩ valid.",
        "• Exclusions (pre-registered): ≥30 units after NON-SOMA filter;",
        "  ≥0.7 valid-target fraction; path extent; train split-half rate-map",
        "  stability. Cohort: 2 sessions/animal near median units (ties → lex).",
        "• Metric: normalized error = median Euclidean error / chance floor",
        "  (train-mean position). Contrasts: per-animal means, sign counts, no p.",
        "• Model selection: blocked inner CV on train with purge gap; primary d",
        "  = lowest inner-CV median Ridge error (ties → smaller d).",
        "• Lag: train-based sync XC (positive = prediction behind truth) plus",
        "  eval plateau range (lags within 0.01 of peak corr). No eval point",
        "  estimates in reports.",
        "• GPFA offline smoother = reference only; excluded from contrasts and",
        "  from “best method” panels. gpfa_causal is the causal filter arm.",
        "• Analysis frozen after plateau decision; figures from saved artifacts.",
    ]
    return _text_page(out_dir, "methods", "Methods", lines)
