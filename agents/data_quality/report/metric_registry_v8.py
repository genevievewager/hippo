#!/usr/bin/env python3
"""Metric → panel registry for Report 4a v8 (information leads)."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Each metric key appears in exactly one main-figure panel.
REGISTRY: dict[str, str] = {
    # Fig 1 — information (was v7 Fig 3)
    "info_per_variable_N20": "Fig1a",
    "position_info_vs_N": "Fig1b",
    "mec_vs_sub_info": "Fig1c",
    "population_ve": "Fig1d",
    "decode_err_floor_vs_N": "Fig1e",
    "decode_err_floor_N20": "Fig1f",
    "reweighted_sensitivity": "Fig1g",
    # Fig 2 — tuning + drift (e/f were f/g in v7)
    "rate_map_gallery": "Fig2a",
    "frac_place_tuned": "Fig2b",
    "frac_border_pass": "Fig2b",
    "frac_hd_pass": "Fig2b",
    "frac_speed_pass": "Fig2b",
    "frac_grid_pass": "Fig2b",
    "spatial_info_place_tuned": "Fig2c",
    "bits_ratemap_end_place": "Fig2d",
    "drift_sep_curve": "Fig2e",
    "drift_adjacent_r": "Fig2f",
    # Fig 3 — spikes + behaviour (was v7 Fig 1)
    "raster_1s": "Fig3a",
    "isi_density": "Fig3b",
    "refractory_violation_rate": "Fig3c",
    "trajectory_60s": "Fig3d",
    "speed_median": "Fig3e",
    "turning_angle_std": "Fig3f",
    "movement_state_fractions": "Fig3g",
    "hd_minus_move_deg": "Fig3h",
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

PLOT_HINTS: dict[str, list[str]] = {
    "Fig1a": [r"Information per variable", r"Info above null"],
    "Fig1b": [r"Position info vs N", r"lab = 0 at N"],
    "Fig1c": [r"MEC vs SUB"],
    "Fig1d": [r"Population variance explained", r"Variance share"],
    "Fig1e": [r"Decode error/floor vs N"],
    "Fig1f": [r"Error/floor at N = 20"],
    "Fig1g": [r"Reweighted test"],
    "Fig2a": [r"rate map", r"Place-tuned rate"],
    "Fig2b": [r"Fraction tuned"],
    "Fig2c": [r"Spatial information \(place-tuned", r"SI \(bits\)"],
    "Fig2d": [r"Held-out bits/spike \(place-tuned"],
    "Fig2e": [r"block separation", r"Place-map r vs"],
    "Fig2f": [r"Adjacent-block reliability"],
    "Fig3a": [r"\braster\b", r"1 s"],
    "Fig3b": [r"ISI density", r"refractory.*shaded"],
    "Fig3c": [r"Refractory violation frac", r"frac ISI < 1\.5"],
    "Fig3d": [r"60 s", r"paths with arena"],
    "Fig3e": [r"Speed CDF"],
    "Fig3f": [r"turning CDF", r"Moving-only turning"],
    "Fig3g": [r"Movement-state fractions", r"still / slow / fast"],
    "Fig3h": [r"HD − move", r"\|HD − move\|"],
}


def expected_panel(metric: str) -> str | None:
    return REGISTRY.get(metric)


def audit_duplicates(page_texts: dict[str, str]) -> dict:
    """Fail if a panel's plot hints appear on another figure page."""
    failures = []
    panel_owner = {}
    for _metric, panel in REGISTRY.items():
        panel_owner.setdefault(panel, panel[:4])  # Fig1a → Fig1

    for panel, owner_fig in panel_owner.items():
        hints = PLOT_HINTS.get(panel) or []
        if not hints:
            continue
        for page_name, text in page_texts.items():
            if not text or not page_name.startswith("Fig"):
                continue
            page_fig = page_name[:4]
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
