#!/usr/bin/env python3
"""Metric → panel registry for Report 4a v7. Duplicate audit fails on any remapping."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Each metric key appears in exactly one main-figure panel.
# Supplements may cite but not re-plot (enforced when S pages exist).
REGISTRY: dict[str, str] = {
    # Fig 1 — spikes + behaviour
    "raster_1s": "Fig1a",
    "isi_density": "Fig1b",
    "refractory_violation_rate": "Fig1c",
    "trajectory_60s": "Fig1d",
    "speed_median": "Fig1e",
    "turning_angle_std": "Fig1f",
    "movement_state_fractions": "Fig1g",
    "hd_minus_move_deg": "Fig1h",
    # Fig 2 — tuning + drift
    "rate_map_gallery": "Fig2a",
    "frac_place_tuned": "Fig2b",
    "frac_border_pass": "Fig2b",
    "frac_hd_pass": "Fig2b",
    "frac_speed_pass": "Fig2b",
    "frac_grid_pass": "Fig2b",
    "spatial_info_place_tuned": "Fig2c",
    "bits_ratemap_end_place": "Fig2d",
    "drift_sep_curve": "Fig2f",
    "drift_adjacent_r": "Fig2g",
    # Fig 3 — information (reserved; not built in this preview)
    "info_per_variable_N20": "Fig3a",
    "position_info_vs_N": "Fig3b",
    "mec_vs_sub_info": "Fig3c",
    "population_ve": "Fig3d",
    "decode_err_floor_vs_N": "Fig3e",
    "decode_err_floor_N20": "Fig3f",
    "reweighted_sensitivity": "Fig3g",
    # Fig 4 — what's right
    "firing_rate_hz": "Fig4",
    "fano_factor": "Fig4",
    "isi_cv": "Fig4",
    "rate_drift": "Fig4",
    "mean_pairwise_corr": "Fig4",
    "sparsity_median": "Fig4",
    "place_field_size_median": "Fig4",
    "peak_rate_median": "Fig4",
    # Fig 5 — R only
    "R_forest": "Fig5a",
    "sim_todo": "Fig5b",
}

# Text tokens on a page that imply a metric is being *plotted* (not just cited).
PLOT_HINTS: dict[str, list[str]] = {
    "Fig1a": [r"\braster\b", r"1 s"],
    "Fig1b": [r"ISI density", r"refractory.*shaded"],
    "Fig1c": [r"Refractory violation frac", r"frac ISI < 1\.5"],
    "Fig1d": [r"60 s", r"trajectory", r"path"],
    "Fig1e": [r"speed.*CDF", r"Speed CDF"],
    "Fig1f": [r"turning.*CDF", r"Turning-angle CDF"],
    "Fig1g": [r"still/slow/fast"],
    "Fig1h": [r"HD.minus.move", r"\|HD.−.move\|"],
    "Fig2a": [r"rate map", r"Place-tuned rate"],
    "Fig2b": [r"Fraction tuned", r"frac.*tuned"],
    "Fig2c": [r"Spatial information \(place-tuned", r"SI \(bits\)"],
    "Fig2d": [r"bits_ratemap_end among place", r"Held-out bits/spike \(place-tuned"],
    "Fig2f": [r"block separation", r"Place-map r vs"],
    "Fig2g": [r"Adjacent-block reliability"],
}


def expected_panel(metric: str) -> str | None:
    return REGISTRY.get(metric)


def audit_duplicates(page_texts: dict[str, str]) -> dict:
    """page_texts: {'Fig1': text, 'Fig2': text, ...}. Fail if a panel's plot hints appear on another page."""
    failures = []
    # Invert: for each panel, which pages should own it
    panel_owner = {}
    for metric, panel in REGISTRY.items():
        fig = panel[:4]  # Fig1 / Fig2 / …
        panel_owner.setdefault(panel, fig)

    for panel, owner_fig in panel_owner.items():
        hints = PLOT_HINTS.get(panel) or []
        if not hints:
            continue
        for page_name, text in page_texts.items():
            if not text:
                continue
            # Story may cite; only fail if a *figure* page other than owner matches
            if not page_name.startswith("Fig"):
                continue
            page_fig = page_name[:4] if page_name.startswith("Fig") else page_name
            if page_fig == owner_fig:
                continue
            for pat in hints:
                if re.search(pat, text, re.IGNORECASE):
                    failures.append({
                        "panel": panel,
                        "owner": owner_fig,
                        "found_on": page_name,
                        "pattern": pat,
                    })
                    break
    return {
        "ok": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures[:40],
        "n_registry": len(REGISTRY),
    }


def write_registry(path: Path) -> None:
    path.write_text(json.dumps({"registry": REGISTRY, "plot_hints": PLOT_HINTS}, indent=2) + "\n")
