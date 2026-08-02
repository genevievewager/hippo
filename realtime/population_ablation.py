"""Hippocampal region / layer / cell-type population ablations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

POPULATION_MODES = (
    "all_units",
    "CA1_only",
    "CA2_only",
    "CA3_only",
    "DG_only",
    "CA1_CA3",
    "layer_specific",
    "cell_type_specific",
)


def unit_ids_for_population(
    units_df: pd.DataFrame,
    unit_ids: np.ndarray,
    mode: str,
    *,
    layer: str | None = None,
    cell_type: str | None = None,
) -> np.ndarray:
    """Return unit_ids kept for a population ablation mode."""
    unit_ids = np.asarray(unit_ids, dtype=int)
    if mode == "all_units":
        return unit_ids

    df = units_df.copy()
    if "unit_id" not in df.columns:
        raise ValueError("units_df must contain unit_id")
    df = df[df["unit_id"].isin(set(unit_ids.tolist()))]

    region_col = "region" if "region" in df.columns else None
    if region_col is None and "region_canonical" in df.columns:
        region_col = "region_canonical"

    if mode.endswith("_only") and region_col is not None:
        region = mode.replace("_only", "")
        keep = df[df[region_col].astype(str).str.upper() == region.upper()]["unit_id"]
        return np.array([u for u in unit_ids if u in set(keep.tolist())], dtype=int)

    if mode == "CA1_CA3" and region_col is not None:
        keep = df[df[region_col].astype(str).str.upper().isin({"CA1", "CA3"})]["unit_id"]
        return np.array([u for u in unit_ids if u in set(keep.tolist())], dtype=int)

    if mode == "layer_specific":
        if layer is None or "layer" not in df.columns:
            return np.asarray([], dtype=int)
        keep = df[df["layer"].astype(str) == str(layer)]["unit_id"]
        return np.array([u for u in unit_ids if u in set(keep.tolist())], dtype=int)

    if mode == "cell_type_specific":
        if cell_type is None or "cell_type" not in df.columns:
            return np.asarray([], dtype=int)
        keep = df[df["cell_type"].astype(str) == str(cell_type)]["unit_id"]
        return np.array([u for u in unit_ids if u in set(keep.tolist())], dtype=int)

    return np.asarray([], dtype=int)


def expand_population_jobs(units_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Expand population ablation jobs including layer/cell-type specifics."""
    jobs: list[dict[str, Any]] = [
        {"population_mode": "all_units", "population_label": "all_units"},
        {"population_mode": "CA1_only", "population_label": "CA1_only"},
        {"population_mode": "CA2_only", "population_label": "CA2_only"},
        {"population_mode": "CA3_only", "population_label": "CA3_only"},
        {"population_mode": "DG_only", "population_label": "DG_only"},
        {"population_mode": "CA1_CA3", "population_label": "CA1_CA3"},
    ]
    if "layer" in units_df.columns:
        for layer in sorted(units_df["layer"].astype(str).unique()):
            jobs.append({
                "population_mode": "layer_specific",
                "population_label": f"layer:{layer}",
                "layer": layer,
            })
    if "cell_type" in units_df.columns:
        for ct in sorted(units_df["cell_type"].astype(str).unique()):
            jobs.append({
                "population_mode": "cell_type_specific",
                "population_label": f"cell_type:{ct}",
                "cell_type": ct,
            })
    return jobs
