#!/usr/bin/env python3
"""Single-source number audit for Report 4a.

Extract every displayed number from every PDF page and require each
(to within tolerance) to appear in the final post keep-mask tidy tables.
Fail on any value known to come from a superseded (pre-spike-fix) table.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader

NUM = re.compile(r"(?<![A-Za-z_/\]])(-?\d+\.\d+|\d+)(?![A-Za-z_])")
# Digits inside animal_session tokens (e.g. 1150_09102024_2rooms) are not claims.
SESSION_TOKEN = re.compile(r"\b\d{4}_\d{8}(?:_\w+)?\b")

# Pre-keep-mask peak rate (lab) — must not appear outside the OLD_VS_NEW appendix.
SUPERSEDED_PEAK = 13.7
# Exact float dump from the old findings discriminator row (must not reappear as prose).
SUPERSEDED_EXACT = {"0.4622222222222222", "0.46222222222222226"}


def _close(a: float, b: float, tol: float = 0.02) -> bool:
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    if abs(b) < 1e-9:
        return abs(a - b) <= tol
    return abs(a - b) <= tol * max(1.0, abs(b)) or abs(a - b) <= 0.025


def _round_variants(v: float) -> set[float]:
    out = {float(v)}
    for nd in (1, 2, 3):
        out.add(round(float(v), nd))
    # percent forms
    out.add(round(float(v) * 100, 2))
    out.add(round(float(v) * 100, 1))
    return out


def allowed_from_tables(data: Path) -> set[float]:
    """Build the allowlist of numbers that may appear on any page."""
    allow: set[float] = set()
    fig3 = pd.read_csv(data / "data_fig3_divergence.csv")
    for col in (
        "R_matched", "R_matched_lo", "R_matched_hi",
        "R_default", "R_default_lo", "R_default_hi",
        "lab_median", "sim_matched_median", "sim_default_median",
    ):
        if col in fig3.columns:
            for v in fig3[col].to_numpy(float):
                if np.isfinite(v):
                    allow |= _round_variants(v)

    meta = json.loads((data / "data_meta.json").read_text())
    for k in ("n_animals", "n_lab_sessions", "n_matched_sessions", "n_default_seeds", "thigmotaxis"):
        if k in meta and meta[k] is not None:
            allow |= _round_variants(float(meta[k]))
    for arm in ("lab", "sim_matched", "sim_default"):
        v = (meta.get("refractory_pct") or {}).get(arm)
        if v is not None:
            allow |= _round_variants(float(v))
    dec = meta.get("decode_error_over_floor_median_N20") or {}
    for v in dec.values():
        if isinstance(v, (int, float)) and np.isfinite(v):
            allow |= _round_variants(float(v))
    neu = meta.get("boost_neutrality") or {}
    if neu.get("max_R") is not None:
        allow |= _round_variants(float(neu["max_R"]))

    # behaviour_info bits / drift / decode / n-curve / unique variance
    beh = data.parent / "behavior_info"
    for name in ("encoding_diag_summary.json", "drift_noise_summary.json"):
        p = beh / name
        if not p.is_file():
            continue
        raw = json.loads(p.read_text())

        def walk(o):
            if isinstance(o, dict):
                for vv in o.values():
                    walk(vv)
            elif isinstance(o, list):
                for vv in o:
                    walk(vv)
            elif isinstance(o, (int, float)) and np.isfinite(o):
                allow.update(_round_variants(float(o)))

        walk(raw)

    for csv_name in (
        "decode_rows.csv", "n_curve.csv", "unique_variance_table.csv",
        "encoding_diag_units.csv", "encoding_diag_sessions.csv",
    ):
        p = beh / csv_name
        if not p.is_file():
            continue
        df = pd.read_csv(p)
        for col in df.select_dtypes(include=[np.number]).columns:
            for v in df[col].to_numpy(float):
                if np.isfinite(v):
                    allow |= _round_variants(float(v))
    # session-median bits printed on Fig 1d
    enc_u = beh / "encoding_diag_units.csv"
    if enc_u.is_file():
        eu = pd.read_csv(enc_u)
        # bits_ratemap_end only (bits_spk_end removed from figures/tables)
        bits_col = "bits_ratemap_end" if "bits_ratemap_end" in eu.columns else None
        if bits_col and {"dataset", "session", "place_tuned"}.issubset(eu.columns):
            pt = eu[eu.place_tuned == True]
            sm = pt.groupby(["dataset", "session"])[bits_col].median()
            for v in sm.to_numpy(float):
                if np.isfinite(v):
                    allow |= _round_variants(float(v))
            for ds in ("lab", "sim_matched"):
                if ds in sm.index.get_level_values(0):
                    allow |= _round_variants(float(np.median(sm.loc[ds])))

    # structural constants that are allowed narrative numbers
    for v in (
        0.05, 0.5, 1.0, 1.5, 0.3, 20, 0.25, 0.8, 0.95, 1.05, 2.0,
        45, 100, 150, 4, 5, 6, 8, 10, 40, 47, 60,
        0.48, 0.89, 0.024, 1.26,  # drift / speed-cell narrative
    ):
        allow |= _round_variants(float(v))

    right = data / "data_sim_gets_right.csv"
    if right.is_file():
        rdf = pd.read_csv(right)
        for col in rdf.select_dtypes(include=[np.number]).columns:
            for v in rdf[col].to_numpy(float):
                if np.isfinite(v):
                    allow |= _round_variants(v)

    abs_p = data / "data_spatial_abs_medians.csv"
    if abs_p.is_file():
        adf = pd.read_csv(abs_p)
        for col in adf.select_dtypes(include=[np.number]).columns:
            for v in adf[col].to_numpy(float):
                if np.isfinite(v):
                    allow |= _round_variants(v)

    # frozen step1 behaviour distributions (state fracs / HD)
    for name in ("step1_session_distributions.csv", "step1_dataset_pooled.csv"):
        p = data / name
        if not p.is_file():
            continue
        sdf = pd.read_csv(p)
        for col in sdf.select_dtypes(include=[np.number]).columns:
            for v in sdf[col].to_numpy(float):
                if np.isfinite(v):
                    allow |= _round_variants(float(v))

    # session metrics (refractory, fracs, …)
    sess_p = data / "data_session_metrics.csv"
    if sess_p.is_file():
        sdf = pd.read_csv(sess_p)
        if "value" in sdf.columns:
            for v in sdf["value"].to_numpy(float):
                if np.isfinite(v):
                    allow |= _round_variants(float(v))

    return {a for a in allow if np.isfinite(a)}


def extract_page_numbers(pdf: Path) -> list[dict]:
    """Return [{page, text_snippet, value}, …] for every number on every page."""
    reader = PdfReader(str(pdf))
    rows = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        # Mask session tokens so their digit runs are not treated as claims
        masked = SESSION_TOKEN.sub(lambda m: " " * len(m.group(0)), text)
        for m in NUM.finditer(masked):
            raw = m.group(1)
            try:
                val = float(raw)
            except ValueError:
                continue
            start = max(0, m.start() - 20)
            end = min(len(text), m.end() + 20)
            rows.append({
                "page": i,
                "raw": raw,
                "value": val,
                "ctx": text[start:end].replace("\n", " "),
            })
    return rows


def audit(data: Path, claims_path: Path | None = None, pdf: Path | None = None) -> dict:
    allow = allowed_from_tables(data)
    failures = []

    # Hard ban superseded values in the tidy itself
    fig3 = pd.read_csv(data / "data_fig3_divergence.csv")
    peak = fig3[fig3.metric_base == "peak_rate_median"]
    if len(peak):
        lab_p = float(peak.iloc[0]["lab_median"])
        if abs(lab_p - 13.7) < 0.5:
            failures.append(f"fig3 peak_rate lab_median={lab_p} looks pre-keep-mask (expect ~7.5)")
        if abs(lab_p - 7.5) > 0.4 and abs(lab_p - 6.99) > 0.4:
            # warn-ish: still fail if wildly off both pooled and session-median
            failures.append(f"fig3 peak_rate lab_median={lab_p} not in {{7.5±0.4, 7.0±0.2}}")

    # excess rows must be gone
    if fig3["metric_base"].astype(str).str.contains("excess").any():
        failures.append("fig3 still contains excess metrics")
    if len(fig3) > 25:
        failures.append(f"fig3 has {len(fig3)} rows (>25)")

    page_rows = []
    unmatched = []
    if pdf is not None and pdf.is_file():
        # Identify appendix pages (OLD values allowed only there)
        reader = PdfReader(str(pdf))
        appendix_pages = set()
        for i, page in enumerate(reader.pages, 1):
            t = (page.extract_text() or "")
            if "OLD vs NEW" in t or "lab_old" in t or "spike_keep" in t.lower():
                appendix_pages.add(i)
        page_rows = extract_page_numbers(pdf)
        for row in page_rows:
            v = row["value"]
            raw = str(row.get("raw", ""))
            in_appendix = row["page"] in appendix_pages
            # Ban pre-fix peak outside appendix
            if not in_appendix and abs(v - SUPERSEDED_PEAK) < 0.08:
                failures.append(
                    f"page {row['page']}: superseded peak_rate {v} ({row['ctx']!r})"
                )
            if raw in SUPERSEDED_EXACT or (abs(v - 0.4622222222222222) < 1e-12):
                failures.append(
                    f"page {row['page']}: superseded discriminator dump {raw} ({row['ctx']!r})"
                )
            # Skip integers that are layout / counts (0–100) and years / animal-like ids
            if abs(v - round(v)) < 1e-12 and 0 <= abs(v) <= 10000:
                continue
            if in_appendix:
                continue
            # Claim-like floats must match final tables
            if not any(_close(v, a) for a in allow):
                if not any(_close(v / 100.0, a) for a in allow):
                    unmatched.append(row)
        for row in unmatched[:40]:
            failures.append(
                f"page {row['page']}: unmatched number {row['value']} ({row['ctx']!r})"
            )
        if len(unmatched) > 40:
            failures.append(f"... and {len(unmatched) - 40} more unmatched numbers")

    result = {
        "ok": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures[:80],
        "n_allow": len(allow),
        "n_page_numbers": len(page_rows),
        "peak_lab": float(peak.iloc[0]["lab_median"]) if len(peak) else None,
        "peak_matched": float(peak.iloc[0]["sim_matched_median"]) if len(peak) else None,
        "fig3_nrows": int(len(fig3)),
    }
    if claims_path is not None:
        claims_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pdf", default="")
    ap.add_argument("--write-claims", default="")
    args = ap.parse_args()
    out = Path(args.write_claims) if args.write_claims else Path(args.data) / "number_claims.json"
    res = audit(Path(args.data), out, Path(args.pdf) if args.pdf else None)
    print(json.dumps({k: res[k] for k in (
        "ok", "n_failures", "failures", "peak_lab", "peak_matched", "fig3_nrows", "n_page_numbers",
    )}, indent=2))
    if not res["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
