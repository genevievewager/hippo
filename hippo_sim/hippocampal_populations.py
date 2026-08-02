"""Maximally hippocampal RatInABox population table.

This module is the source of truth for the ``ratinabox_neurons`` backend:
counts, RatInABox class, anatomical cell type, and which post-hoc dynamics
each group receives.

Region mapping rationale
------------------------
- Place / phase-precessing place → CA1 / CA2 / CA3 / DG (hippocampus proper)
- Grid / HD / speed → MEC (entorhinal afferents), not CA1/CA2/DG fillers
- Boundary vector → subiculum (boundary-rich), not CA3
- Local interneurons → one synthetic INT pool per principal region
  (INT_CA1 / INT_CA2 / INT_CA3 / INT_DG / INT_SUB), matching RiaB's
  one-group-per-role pattern and regional organization

Dynamics flags
--------------
- ``theta``: multiplicative cos(θ) envelope (skipped when RiaB already
  implements phase precession)
- ``ripple``: additive sharp-wave-ripple bursts
- ``sparsity``: DG-like soft threshold on normalized rate
- ``recurrent``: CA3-like population-mean recurrent boost
- ``speed_gain``: mild multiplicative speed modulation

Trisynaptic / EC feedforward (MEC→DG→CA3→CA1 plus local INT→home) is
applied after these flags, in ``hippo_sim.feedforward``.
"""

from __future__ import annotations

from typing import Any

# Default counts for the maximally hippocampal config.
# Keys are looked up in ``SimConfig.ratinabox_params`` (with these defaults).
HIPPOCAMPAL_RIA_B_POPULATIONS: list[dict[str, Any]] = [
    {
        "name": "CA1_place_pp",
        "n_key": "n_ca1_place_cells",
        "n_default": 60,
        "ratinabox_class": "PhasePrecessingPlaceCells",
        "cell_type": "CA1_pyr",
        "riab_params": {
            "widths": 0.12,
            "description": "gaussian",
            "theta_freq": 8.0,
            "precess_fraction": 0.5,
            "kappa": 2.0,
        },
        "dynamics": {
            "theta": False,  # built into phase precession
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": True,
        },
        "notes": "CA1 pyramidal place fields with theta phase precession",
    },
    {
        "name": "CA3_place",
        "n_key": "n_ca3_place_cells",
        "n_default": 40,
        "ratinabox_class": "PlaceCells",
        "cell_type": "CA3_pyr",
        "riab_params": {
            "widths": 0.18,
            "description": "gaussian",
        },
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": True,
            "speed_gain": False,
        },
        "notes": "CA3 place fields + recurrent population drive + SWR",
    },
    {
        "name": "DG_place",
        "n_key": "n_dg_place_cells",
        "n_default": 50,
        "ratinabox_class": "PlaceCells",
        "cell_type": "DG_granule",
        "riab_params": {
            "widths": 0.08,
            "description": "gaussian_threshold",
        },
        "dynamics": {
            "theta": True,
            "ripple": False,
            "sparsity": True,
            "recurrent": False,
            "speed_gain": True,
        },
        "notes": "Sparse DG granule place fields (pattern separation)",
    },
    {
        "name": "CA2_place",
        "n_key": "n_ca2_place_cells",
        "n_default": 20,
        "ratinabox_class": "PlaceCells",
        "cell_type": "CA2_pyr",
        "riab_params": {
            "widths": 0.14,
            "description": "gaussian",
        },
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": "CA2 place fields (weaker ripple participation)",
    },
    {
        "name": "MEC_grid",
        "n_key": "n_mec_grid_cells",
        "n_default": 40,
        "ratinabox_class": "GridCells",
        "cell_type": "MEC_grid",
        "riab_params": {
            "gridscale_distribution": "modules",
            "gridscale": (0.3, 0.5, 0.8),
        },
        "dynamics": {
            "theta": True,
            "ripple": False,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": True,
        },
        "notes": "MEC grid cells as entorhinal afferent drive",
    },
    {
        "name": "MEC_hd",
        "n_key": "n_mec_hd_cells",
        "n_default": 30,
        "ratinabox_class": "HeadDirectionCells",
        "cell_type": "MEC_hd",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": False,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": "MEC / para-subicular head-direction afferents",
    },
    {
        "name": "Sub_bvc",
        "n_key": "n_sub_bvc_cells",
        "n_default": 30,
        "ratinabox_class": "BoundaryVectorCells",
        "cell_type": "Sub_bvc",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": False,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": "Subiculum boundary-vector cells",
    },
    {
        "name": "MEC_speed",
        "n_key": "n_mec_speed_cells",
        "n_default": 20,
        "ratinabox_class": "SpeedCells_fallback",
        "cell_type": "MEC_speed",
        "riab_params": {},
        "dynamics": {
            "theta": False,
            "ripple": False,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": "MEC speed cells (RiaB SpeedCell is n=1; use population fallback)",
    },
    # Local per-region interneurons (RiaB-style: one synthetic group per home).
    # Defaults sum to 15 to preserve the prior global INT budget.
    {
        "name": "INT_CA1",
        "n_key": "n_int_ca1",
        "n_default": 6,
        "ratinabox_class": "Interneurons_synthetic",
        "cell_type": "INT_CA1",
        "home_node": "CA1",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": (
            "CA1-local interneurons: high tonic + theta/ripple; "
            "anti-correlated with CA1 pyr; Stage-C feedback onto CA1"
        ),
    },
    {
        "name": "INT_CA3",
        "n_key": "n_int_ca3",
        "n_default": 3,
        "ratinabox_class": "Interneurons_synthetic",
        "cell_type": "INT_CA3",
        "home_node": "CA3",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": (
            "CA3-local interneurons: anti-correlated with CA3 pyr; "
            "Stage-C feedback onto CA3"
        ),
    },
    {
        "name": "INT_DG",
        "n_key": "n_int_dg",
        "n_default": 3,
        "ratinabox_class": "Interneurons_synthetic",
        "cell_type": "INT_DG",
        "home_node": "DG",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": (
            "DG-local interneurons: anti-correlated with DG granule; "
            "Stage-C feedback onto DG"
        ),
    },
    {
        "name": "INT_CA2",
        "n_key": "n_int_ca2",
        "n_default": 1,
        "ratinabox_class": "Interneurons_synthetic",
        "cell_type": "INT_CA2",
        "home_node": "CA2",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": (
            "CA2-local interneurons: anti-correlated with CA2 pyr; "
            "Stage-C feedback onto CA2"
        ),
    },
    {
        "name": "INT_SUB",
        "n_key": "n_int_sub",
        "n_default": 2,
        "ratinabox_class": "Interneurons_synthetic",
        "cell_type": "INT_SUB",
        "home_node": "SUB",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": (
            "Subiculum-local interneurons: anti-correlated with Sub BVC; "
            "Stage-C feedback onto SUB"
        ),
    },
]

# Legacy count keys that still feed the CA1-local INT pool.
_INT_CA1_LEGACY_N_KEYS = ("n_interneurons", "n_ca1_interneurons")


def population_count(spec: dict[str, Any], ratinabox_params: dict) -> int:
    """Resolve population size from config params with table default."""
    if spec["n_key"] in ratinabox_params:
        return int(ratinabox_params[spec["n_key"]])
    # Legacy global INT counts map onto the CA1-local pool.
    if spec["n_key"] == "n_int_ca1":
        for legacy in _INT_CA1_LEGACY_N_KEYS:
            if legacy in ratinabox_params:
                return int(ratinabox_params[legacy])
    return int(spec["n_default"])


def population_table_summary(ratinabox_params: dict | None = None) -> list[dict[str, Any]]:
    """Human-readable summary rows for docs / metadata."""
    rp = ratinabox_params or {}
    rows = []
    for spec in HIPPOCAMPAL_RIA_B_POPULATIONS:
        dyn = spec["dynamics"]
        active = [k for k, v in dyn.items() if v]
        rows.append({
            "name": spec["name"],
            "n": population_count(spec, rp),
            "ratinabox_class": spec["ratinabox_class"],
            "cell_type": spec["cell_type"],
            "dynamics": ",".join(active) if active else "none",
            "notes": spec.get("notes", ""),
        })
    return rows
