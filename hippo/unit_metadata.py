"""Unit metadata schema for hierarchical hippocampal populations.

Phase 1 normalizes columns present in current ``units.csv`` and reserves
optional anatomical / projection / mixed-selectivity fields for later phases.
Missing optional columns are filled with NaN or defaults and flagged in
``metadata_status``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Columns always expected from the current simulator.
REQUIRED_UNIT_COLUMNS = (
    "unit_id",
    "region",
    "cell_type",
)

# Present in current pipeline; treated as required when available.
CORE_UNIT_COLUMNS = (
    "unit_id",
    "cell_type",
    "region",
    "layer",
    "channel",
    "depth_um",
    "place_x_cm",
    "place_y_cm",
    "hd_pref_rad",
    "rate_model",
    "ratinabox_class",
)

# Phase 2+ reserved fields (optional until simulation emits them).
OPTIONAL_UNIT_COLUMNS = (
    "subfield",
    "cell_class",  # excitatory / inhibitory
    "deep_superficial_coordinate",
    "deep_superficial_group",
    "proximodistal_coordinate",
    "proximodistal_group",
    "dorsoventral_coordinate",
    "dorsoventral_group",
    "projection_target",
    "baseline_rate_hz",
    "burst_index",
    "waveform_width_ms",
    "theta_phase_preference_rad",
    "theta_modulation_strength",
    "ripple_participation_probability",
    # Continuous mixed-selectivity strengths (Phase 2)
    "spatial_tuning_strength",
    "head_direction_tuning_strength",
    "speed_tuning_strength",
    "boundary_tuning_strength",
    "reward_tuning_strength",
    "context_tuning_strength",
    "time_tuning_strength",
    "memory_identity_tuning_strength",
)

METADATA_STATUS = {
    "known_in_simulation": "column exists in current units.csv / simulation",
    "reserved_phase2": "schema reserved; filled with NaN until simulation emits it",
    "not_experimentally_identifiable": "ground-truth simulation only; not recoverable from sorted spikes",
}


def _derive_cell_class(cell_type: str) -> str:
    ct = str(cell_type).lower()
    if "inh" in ct or "inter" in ct or "pv" in ct or "som" in ct:
        return "inhibitory"
    return "excitatory"


def _derive_deep_superficial_from_layer(layer: Any) -> tuple[float, str]:
    """Map CA1 stratum labels to a provisional deep/superficial coordinate.

    This is a Phase-1 convenience mapping from existing layer names, not a claim
    that oriens/radiatum equal biological deep/superficial pyramids.
    """
    if layer is None or (isinstance(layer, float) and np.isnan(layer)):
        return float("nan"), "unknown"
    name = str(layer).lower()
    if name in ("oriens", "deep"):
        return 0.0, "deep"
    if name in ("radiatum", "superficial", "lm", "lacunosum"):
        return 1.0, "superficial"
    if name in ("pyramidal", "pyr", "intermediate"):
        return 0.5, "intermediate"
    return float("nan"), "unknown"


def normalize_unit_metadata(units_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of units_df with required columns validated and optional
    schema columns present (NaN if unavailable).
    """
    if units_df is None or units_df.empty:
        raise ValueError("units_df is empty")
    df = units_df.copy()
    missing = [c for c in REQUIRED_UNIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"units_df missing required columns: {missing}")

    if "subfield" not in df.columns:
        df["subfield"] = df["region"]
    if "cell_class" not in df.columns:
        df["cell_class"] = df["cell_type"].map(_derive_cell_class)

    if "deep_superficial_coordinate" not in df.columns:
        if "layer" in df.columns:
            coords = df["layer"].map(lambda x: _derive_deep_superficial_from_layer(x)[0])
            groups = df["layer"].map(lambda x: _derive_deep_superficial_from_layer(x)[1])
            df["deep_superficial_coordinate"] = coords
            df["deep_superficial_group"] = groups
        else:
            df["deep_superficial_coordinate"] = np.nan
            df["deep_superficial_group"] = "unknown"

    for col in (
        "proximodistal_coordinate",
        "dorsoventral_coordinate",
        "baseline_rate_hz",
        "burst_index",
        "waveform_width_ms",
        "theta_modulation_strength",
        "ripple_participation_probability",
        "spatial_tuning_strength",
        "head_direction_tuning_strength",
        "speed_tuning_strength",
        "boundary_tuning_strength",
    ):
        if col not in df.columns:
            df[col] = np.nan

    for col in (
        "proximodistal_group",
        "dorsoventral_group",
        "projection_target",
    ):
        if col not in df.columns:
            df[col] = "unknown"

    # Normalize continuous anatomical coordinates into [0, 1] when present
    for col in (
        "deep_superficial_coordinate",
        "proximodistal_coordinate",
        "dorsoventral_coordinate",
    ):
        vals = pd.to_numeric(df[col], errors="coerce")
        finite = vals.dropna()
        if len(finite) and finite.min() >= 0 and finite.max() <= 1:
            df[col] = vals
        elif len(finite) and finite.max() > finite.min():
            df[col] = (vals - finite.min()) / (finite.max() - finite.min())

    df["metadata_schema_version"] = "phase1"
    return df


def metadata_availability_table(units_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize which schema columns are populated."""
    df = normalize_unit_metadata(units_df)
    rows = []
    for col in CORE_UNIT_COLUMNS + OPTIONAL_UNIT_COLUMNS:
        present = col in units_df.columns
        if present:
            n_valid = int(units_df[col].notna().sum()) if col in units_df.columns else 0
            status = METADATA_STATUS["known_in_simulation"]
        else:
            n_valid = int(df[col].notna().sum()) if col in df.columns else 0
            status = METADATA_STATUS["reserved_phase2"]
        rows.append({
            "column": col,
            "in_raw_units_csv": present,
            "n_non_null_after_normalize": n_valid,
            "status": status,
        })
    return pd.DataFrame(rows)
