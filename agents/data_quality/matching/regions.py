"""Region map and composition-matched subsampling (never use include_in_decoder)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

# Spec §3.1 / Phase-1 decisions
LAB_PRIMARY = frozenset({"MEC", "SUB"})
LAB_SECONDARY = frozenset({"MEC", "PARA", "SUB"})

# Sim probe band → lab label
SIM_REGION_TO_LAB: dict[str, str] = {
    "entorhinal_transition": "MEC",
    "entorhinal_cortex_layer6a": "MEC",
    "entorhinal_cortex_layer5": "MEC",
    "medial_entorhinal_layer3": "MEC",
    "medial_entorhinal_layer2": "MEC",
    "medial_entorhinal_layer1": "MEC",
    "hippocampal_formation_transition": "SUB",
}


def map_sim_region(region: str) -> str:
    """Map a simulator anatomy region to a lab Region label (MEC/SUB/OTHER)."""
    return SIM_REGION_TO_LAB.get(str(region), "OTHER")


def add_mapped_lab_region(units_df: pd.DataFrame, *, region_col: str = "region") -> pd.DataFrame:
    out = units_df.copy()
    out["lab_region"] = out[region_col].map(map_sim_region)
    return out


def stratify_lab_units(
    regions: Iterable[str],
    *,
    set_name: str = "primary",
) -> np.ndarray:
    """Boolean mask for lab units in a named region set."""
    regions = np.asarray(list(regions), dtype=str)
    if set_name == "primary":
        keep = LAB_PRIMARY
    elif set_name == "secondary":
        keep = LAB_SECONDARY
    elif set_name == "mec_only":
        keep = frozenset({"MEC"})
    elif set_name == "all":
        return np.ones(len(regions), dtype=bool)
    else:
        raise ValueError(set_name)
    return np.isin(regions, list(keep))


def composition_matched_subsample(
    sim_lab_regions: np.ndarray,
    target_counts: dict[str, int],
    *,
    rng: np.random.Generator,
    n_subsamples: int = 20,
) -> list[np.ndarray]:
    """Subsample sim unit indices to match per-region counts (MEC/SUB).

    Raises if any region has fewer available units than required.
    """
    sim_lab_regions = np.asarray(sim_lab_regions, dtype=str)
    out: list[np.ndarray] = []
    for _ in range(int(n_subsamples)):
        chosen: list[int] = []
        for reg, need in target_counts.items():
            need = int(need)
            if need <= 0:
                continue
            pool = np.flatnonzero(sim_lab_regions == reg)
            if pool.size < need:
                raise RuntimeError(
                    f"sim yield {reg}={pool.size} < required {need}; "
                    "raise unit_density_per_mm and regenerate"
                )
            chosen.extend(rng.choice(pool, size=need, replace=False).tolist())
        out.append(np.asarray(sorted(chosen), dtype=int))
    return out
