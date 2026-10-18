"""Boost SimConfig.ratinabox_params so mapped MEC/SUB yields can meet targets.

``apply_cell_capture_to_ratinabox_params`` (lab format) caps each ``n_*`` at the
baseline RatInABox size, so raising ``unit_density_per_mm`` alone cannot grow
populations past ~90–120 MEC units. This wrapper raises the ``n_*`` knobs
after ``apply_trajectory_to_config`` (public SimConfig fields; no monkeypatch).
"""

from __future__ import annotations

import math
from typing import Any

from hippo_sim.config import SimConfig

MEC_KEYS = ("n_mec_grid_cells", "n_mec_hd_cells", "n_mec_speed_cells")
SUB_KEYS = ("n_sub_bvc_cells", "n_int_ca1")  # HPF band hosts Sub_bvc + INT_CA1


def boost_ratinabox_for_required(
    config: SimConfig,
    required: dict[str, int],
    *,
    safety: float = 1.3,
) -> dict[str, Any]:
    """In-place: ensure sum of MEC/SUB population knobs ≥ safety × required."""
    rp = dict(config.ratinabox_params)
    report: dict[str, Any] = {"before": {}, "after": {}, "targets": {}}
    for keys, lab in ((MEC_KEYS, "MEC"), (SUB_KEYS, "SUB")):
        need = int(math.ceil(max(0, int(required.get(lab, 0))) * float(safety)))
        report["targets"][lab] = need
        cur = [int(rp.get(k, 0)) for k in keys]
        report["before"][lab] = dict(zip(keys, cur))
        total = sum(cur)
        if need <= 0:
            report["after"][lab] = dict(zip(keys, cur))
            continue
        if total <= 0:
            # equal split
            base = max(1, need // len(keys))
            for i, k in enumerate(keys):
                rp[k] = base + (1 if i < (need - base * len(keys)) else 0)
        elif total < need:
            scale = need / total
            new_vals = [max(1, int(math.ceil(v * scale))) for v in cur]
            # fix rounding shortfall
            while sum(new_vals) < need:
                new_vals[int(max(range(len(new_vals)), key=lambda i: new_vals[i]))] += 1
            for k, v in zip(keys, new_vals):
                rp[k] = int(v)
        report["after"][lab] = {k: int(rp[k]) for k in keys}
    # sync legacy aliases used elsewhere
    aliases = {
        "n_mec_hd_cells": "n_head_direction_cells",
        "n_sub_bvc_cells": "n_boundary_vector_cells",
        "n_mec_speed_cells": "n_speed_cells",
        "n_mec_grid_cells": "n_grid_cells",
        "n_int_ca1": "n_interneurons",
    }
    for src, alias in aliases.items():
        if src in rp:
            rp[alias] = int(rp[src])
    config.ratinabox_params = rp
    return report
