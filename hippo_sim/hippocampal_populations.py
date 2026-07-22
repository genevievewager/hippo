"""Maximally hippocampal RatInABox population table.

This module is the source of truth for the ``ratinabox_neurons`` backend:
counts, RatInABox class, anatomical cell type, and which post-hoc dynamics
each group receives.

Region mapping rationale
------------------------
- Place / phase-precessing place → CA1 / CA2 / CA3 / DG (hippocampus proper)
- Grid / HD / speed → MEC (entorhinal afferents), not CA1/CA2/DG fillers
- Boundary vector → subiculum (boundary-rich), not CA3
- CA1 oriens interneurons → synthetic inhibitory population

Dynamics flags
--------------
- ``theta``: multiplicative cos(θ) envelope (skipped when RiaB already
  implements phase precession)
- ``ripple``: additive sharp-wave-ripple bursts
- ``sparsity``: DG-like soft threshold on normalized rate
- ``recurrent``: CA3-like population-mean recurrent boost
- ``speed_gain``: mild multiplicative speed modulation

Trisynaptic / EC feedforward (MEC→DG→CA3→CA1, INT→CA1) is applied after
these flags, in ``hippo_sim.feedforward``.
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
    {
        "name": "CA1_int",
        "n_key": "n_ca1_interneurons",
        "n_default": 15,
        "ratinabox_class": "CA1_Interneurons_synthetic",
        "cell_type": "CA1_int",
        "riab_params": {},
        "dynamics": {
            "theta": True,
            "ripple": True,
            "sparsity": False,
            "recurrent": False,
            "speed_gain": False,
        },
        "notes": "CA1 oriens interneurons: high tonic rate, anti-correlated with CA1 pyr",
    },
]


def population_count(spec: dict[str, Any], ratinabox_params: dict) -> int:
    """Resolve population size from config params with table default."""
    return int(ratinabox_params.get(spec["n_key"], spec["n_default"]))


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
