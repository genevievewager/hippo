"""Hippocampal-system allowlist for simulation, decoding, and manifolds.

Only RatInABox-modeled hippocampal / MEC-afferent populations are analysis-
eligible by default. Probe bands outside that system (e.g. visual cortex) may
appear on trajectory figures but must not enter decoding or manifold features
unless explicitly opted in.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

# Cell types emitted by hippo_sim / RatInABox hippocampal populations.
ALLOWED_CELL_TYPES: frozenset[str] = frozenset({
    "CA1_pyr",
    "INT_CA1",
    "INT_CA2",
    "INT_CA3",
    "INT_DG",
    "INT_SUB",
    "interneuron",  # legacy alias → INT_CA1
    "CA1_int",  # legacy alias
    "CA2_pyr",
    "CA3_pyr",
    "DG_granule",
    "Sub_bvc",
    "MEC_grid",
    "MEC_hd",
    "MEC_speed",
})

# Canonical region labels used in decoder / viz after alias normalization.
CANONICAL_REGIONS: frozenset[str] = frozenset({
    "CA1",
    "CA2",
    "CA3",
    "DG",
    "Subiculum",
    "MEC",
})

# Lab trajectory / NTE-style names → canonical analysis regions.
REGION_ALIASES: dict[str, str] = {
    "ca1": "CA1",
    "ca2": "CA2",
    "ca3": "CA3",
    "dg": "DG",
    "dentate_gyrus": "DG",
    "dentate gyrus": "DG",
    "subiculum": "Subiculum",
    "sub": "Subiculum",
    "hpf_pros_transition": "Subiculum",
    "hpf_pros": "Subiculum",
    "mec": "MEC",
    "entorhinal_cortex": "MEC",
    "ent": "MEC",
    "deep_entorhinal_hata": "MEC",
    "hata_ent": "MEC",
    "hata": "MEC",
    # Explicitly non-analysis (kept for figures only).
    "visual_cortex": "visual_cortex",
    "vis": "visual_cortex",
}

NON_HIPPOCAMPAL_REGIONS: frozenset[str] = frozenset({
    "visual_cortex",
    "VIS",
    "VISp",
    "VISl",
    "VISpor",
    "VISli",
})

# Placeholder / optional types that must never enter analysis by default.
EXCLUDED_CELL_TYPES: frozenset[str] = frozenset({
    "visual_units_optional",
    "HATA_optional",
    "unknown",
    "",
})


def canonicalize_region(region: Any) -> str:
    """Map free-text / lab region labels to a canonical analysis name."""
    if region is None or (isinstance(region, float) and np.isnan(region)):
        return "unknown"
    raw = str(region).strip()
    if not raw:
        return "unknown"
    key = raw.lower().replace("-", "_").replace(" ", "_")
    if key in REGION_ALIASES:
        return REGION_ALIASES[key]
    # Already canonical?
    for canon in CANONICAL_REGIONS:
        if raw == canon or key == canon.lower():
            return canon
    return raw


def is_allowed_cell_type(cell_type: Any) -> bool:
    if cell_type is None or (isinstance(cell_type, float) and np.isnan(cell_type)):
        return False
    ct = str(cell_type).strip()
    if ct in EXCLUDED_CELL_TYPES:
        return False
    return ct in ALLOWED_CELL_TYPES


def is_hippocampal_region(region: Any) -> bool:
    """True if region is (or aliases to) an analysis-eligible hippocampal region."""
    canon = canonicalize_region(region)
    if canon in NON_HIPPOCAMPAL_REGIONS or canon.lower() in {r.lower() for r in NON_HIPPOCAMPAL_REGIONS}:
        return False
    if "visual" in str(region).lower() or canon.lower().startswith("vis"):
        return False
    return canon in CANONICAL_REGIONS


def unit_include_in_decoder(
    *,
    cell_type: Any,
    region: Any = None,
    include_flag: Any = None,
    include_non_hippocampal: bool = False,
) -> bool:
    """Decide whether a unit may enter decoding / manifold features."""
    if include_non_hippocampal:
        if include_flag is not None and not _as_bool(include_flag, True):
            # Explicit False still respected unless cell type is allowlisted
            # and caller forced include_non_hippocampal for contamination studies.
            pass
        return is_allowed_cell_type(cell_type) or (
            include_flag is not None and _as_bool(include_flag, False)
        )

    if include_flag is not None and not _as_bool(include_flag, True):
        return False
    if not is_allowed_cell_type(cell_type):
        return False
    if region is not None and not is_hippocampal_region(region):
        return False
    return True


def _as_bool(val: Any, default: bool = True) -> bool:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() not in {"false", "0", "no", "n"}


def annotate_units_for_analysis(
    units_df: pd.DataFrame,
    *,
    include_non_hippocampal: bool = False,
) -> pd.DataFrame:
    """Add/refresh ``include_in_decoder``, ``region_canonical``, ``subfield``."""
    if units_df is None or units_df.empty:
        return units_df
    df = units_df.copy()
    if "region" in df.columns:
        df["region_canonical"] = df["region"].map(canonicalize_region)
        if "subfield" not in df.columns or df["subfield"].isna().all():
            df["subfield"] = df["region_canonical"]
        else:
            # Prefer canonical when subfield duplicates raw region.
            df["subfield"] = [
                canonicalize_region(s) if canonicalize_region(s) in CANONICAL_REGIONS
                else canonicalize_region(r)
                for s, r in zip(df["subfield"], df["region"])
            ]
    else:
        df["region_canonical"] = "unknown"

    flags = []
    for _, row in df.iterrows():
        flags.append(
            unit_include_in_decoder(
                cell_type=row.get("cell_type"),
                region=row.get("region"),
                include_flag=row.get("include_in_decoder") if "include_in_decoder" in df.columns else None,
                include_non_hippocampal=include_non_hippocampal,
            )
        )
    # When not forcing non-HPC, recompute strictly from allowlist (ignore stale True).
    if not include_non_hippocampal:
        flags = [
            unit_include_in_decoder(
                cell_type=row.get("cell_type"),
                region=row.get("region"),
                include_flag=None,
                include_non_hippocampal=False,
            )
            for _, row in df.iterrows()
        ]
    df["include_in_decoder"] = flags
    return df


def filter_units_for_analysis(
    units_df: pd.DataFrame,
    *,
    include_non_hippocampal: bool = False,
) -> pd.DataFrame:
    """Return only analysis-eligible units (RiaB hippocampal system)."""
    annotated = annotate_units_for_analysis(
        units_df, include_non_hippocampal=include_non_hippocampal,
    )
    if annotated is None or annotated.empty:
        return annotated
    return annotated.loc[annotated["include_in_decoder"].astype(bool)].copy()


def filter_unit_ids_for_analysis(
    units_df: pd.DataFrame,
    unit_ids: Iterable[int] | None = None,
    *,
    include_non_hippocampal: bool = False,
) -> list[int]:
    """Filter a unit_id list to analysis-eligible units."""
    eligible = filter_units_for_analysis(
        units_df, include_non_hippocampal=include_non_hippocampal,
    )
    allowed = set(eligible["unit_id"].astype(int).tolist()) if len(eligible) else set()
    if unit_ids is None:
        return sorted(allowed)
    return [int(u) for u in unit_ids if int(u) in allowed]


# Preferred display / report order for canonical analysis regions.
ANALYSIS_REGION_ORDER: tuple[str, ...] = (
    "CA1", "CA2", "CA3", "DG", "Subiculum", "MEC",
)

CIRCUIT_PROFILES: dict[str, dict[str, Any]] = {
    "trisynaptic_full": {
        "description": "Classic MEC→DG→CA3→CA1 (+ MEC→CA1, local INT→home)",
        "feedforward": {},
    },
    "sub_ent_dominant": {
        "description": (
            "Trajectory emphasizes Subiculum / MEC(entorhinal); "
            "boost MEC→SUB coupling and de-emphasize missing CA targets"
        ),
        "feedforward": {
            "w_mec_to_sub": 0.25,
            "w_mec_to_dg": 0.15,
            "w_dg_to_ca3": 0.10,
            "w_ca3_to_ca1": 0.05,
            "w_mec_to_ca1": 0.10,
            "w_mec_to_ca2": 0.05,
            "w_ca3_to_ca2": 0.05,
        },
    },
    "mixed_hpc": {
        "description": "Multiple hippocampal fields present; use balanced trisynaptic weights",
        "feedforward": {
            "w_mec_to_sub": 0.15,
        },
    },
}


def present_canonical_regions(units_df: pd.DataFrame | None) -> set[str]:
    if units_df is None or units_df.empty or "region" not in units_df.columns:
        return set()
    df = annotate_units_for_analysis(units_df)
    df = df.loc[df["include_in_decoder"].astype(bool)]
    return {canonicalize_region(r) for r in df["region_canonical"].tolist()} & set(CANONICAL_REGIONS)


def infer_circuit_profile(units_df: pd.DataFrame | None = None, *, regions: set[str] | None = None) -> str:
    """Infer feedforward / partition profile from captured regions."""
    regs = set(regions) if regions is not None else present_canonical_regions(units_df)
    ca_fields = regs & {"CA1", "CA2", "CA3"}
    sub_ent = regs & {"Subiculum", "MEC"}
    if not regs:
        return "trisynaptic_full"
    if sub_ent and not ca_fields:
        return "sub_ent_dominant"
    if sub_ent and ca_fields and len(ca_fields) <= 1:
        return "mixed_hpc"
    if ca_fields >= {"CA1", "CA3"} or len(ca_fields) >= 2:
        return "trisynaptic_full"
    if sub_ent:
        return "sub_ent_dominant"
    return "trisynaptic_full"


def recommended_partitions(profile: str | None = None, units_df: pd.DataFrame | None = None) -> list[str]:
    """Default partition set for manifold / BCI compare stages."""
    profile = profile or infer_circuit_profile(units_df)
    regs = present_canonical_regions(units_df) if units_df is not None else set()
    parts = ["all_units", "subfield", "layer", "cell_class", "cell_type"]
    if "CA1" in regs:
        parts.append("ca1_deep_superficial")
    parts.append("deep_superficial")  # any region with laminar labels
    if profile == "sub_ent_dominant":
        # Prefer cell_type / subfield; skip CA1-only if absent (already gated).
        parts = [p for p in parts if p != "ca1_deep_superficial"]
    # Deduplicate, preserve order.
    return list(dict.fromkeys(parts))


def region_order_for_units(units_df: pd.DataFrame | None) -> list[str]:
    """Canonical region order filtered to regions actually present."""
    present = present_canonical_regions(units_df) if units_df is not None else set(ANALYSIS_REGION_ORDER)
    ordered = [r for r in ANALYSIS_REGION_ORDER if r in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def geometry_summary(units_df: pd.DataFrame | None) -> dict[str, Any]:
    """Compact metadata for reports / summary.json."""
    regs = present_canonical_regions(units_df)
    profile = infer_circuit_profile(regions=regs)
    return {
        "circuit_profile": profile,
        "circuit_profile_description": CIRCUIT_PROFILES[profile]["description"],
        "present_regions": sorted(regs),
        "region_order": region_order_for_units(units_df),
        "recommended_partitions": recommended_partitions(profile, units_df),
        "ca1_centric": "CA1" in regs and len(regs & {"Subiculum", "MEC"}) == 0,
        "sub_ent_dominant": profile == "sub_ent_dominant",
    }
