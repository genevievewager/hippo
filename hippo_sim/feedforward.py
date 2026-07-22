"""Trisynaptic / entorhinal feedforward coupling for RatInABox group rates.

After each population has local RiaB tuning + hippocampal overlays, region
population means drive downstream targets along a classic sketch of the
hippocampal loop (plus direct EC→CA1 and CA1 interneuron feedback):

    MEC ──► DG ──► CA3 ──► CA1
     │              ▲       ▲
     │              │       │
     └──────────────┴───────┘
                    ▲
              CA1_int (inhibition)

All weights are tunable hypotheses in ``FEEDFORWARD_PARAMS`` / ``ratinabox_params``.
CA3 autoassociative recurrence is applied earlier in
``apply_hippocampal_dynamics`` and is *not* duplicated here.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np


# Default synaptic / gain weights (Hz of drive per normalized upstream mean).
FEEDFORWARD_PARAMS: dict[str, Any] = {
    "enabled": True,
    "w_mec_to_dg": 0.20,
    "w_dg_to_ca3": 0.25,
    "w_ca3_to_ca1": 0.20,
    "w_mec_to_ca1": 0.15,
    "w_int_to_ca1": 0.30,
    "w_mec_to_ca2": 0.10,
    "w_ca3_to_ca2": 0.10,
    "normalize_upstream": True,  # scale by max mean so weights are dimensionless-ish
}


# Group name → circuit node used for mean pooling / targeting.
GROUP_TO_NODE: dict[str, str] = {
    "MEC_grid": "MEC",
    "MEC_hd": "MEC",
    "MEC_speed": "MEC",
    "DG_place": "DG",
    "CA3_place": "CA3",
    "CA1_place_pp": "CA1",
    "CA2_place": "CA2",
    "CA1_int": "INT",
    "Sub_bvc": "SUB",
}


class _RateGroup(Protocol):
    name: str
    rates_hz: np.ndarray


def resolve_feedforward_params(ratinabox_params: dict | None = None) -> dict[str, Any]:
    """Merge defaults with optional overrides from ``ratinabox_params['feedforward']``."""
    params: dict[str, Any] = dict(FEEDFORWARD_PARAMS)
    if not ratinabox_params:
        return params
    if "apply_feedforward" in ratinabox_params:
        params["enabled"] = bool(ratinabox_params["apply_feedforward"])
    overrides = ratinabox_params.get("feedforward")
    if isinstance(overrides, dict):
        params.update(overrides)
    return params


def _node_mean(groups: list[_RateGroup], node: str, n_steps: int) -> np.ndarray:
    blocks = [g.rates_hz for g in groups if GROUP_TO_NODE.get(g.name) == node]
    if not blocks:
        return np.zeros(n_steps, dtype=float)
    stacked = np.vstack(blocks)
    return stacked.mean(axis=0)


def _normalize_trace(x: np.ndarray) -> np.ndarray:
    peak = float(np.max(x)) if x.size else 0.0
    if peak <= 1e-12:
        return np.zeros_like(x, dtype=float)
    return x / peak


def _add_drive(groups: list[_RateGroup], node: str, drive: np.ndarray) -> None:
    """Add a length-T drive vector to every unit in groups belonging to ``node``."""
    if drive.ndim != 1:
        raise ValueError(f"drive must be shape (T,), got {drive.shape}")
    for g in groups:
        if GROUP_TO_NODE.get(g.name) != node:
            continue
        g.rates_hz = np.maximum(g.rates_hz + drive[None, :], 0.0)


def apply_trisynaptic_feedforward(
    groups: list[_RateGroup],
    *,
    ratinabox_params: dict | None = None,
) -> dict[str, Any]:
    """
    Apply directed region→region rate coupling in causal anatomical order.

    Order
    -----
    1. Pool MEC mean (afferents already finalized).
    2. DG ← + w_MEC→DG · MEC
    3. CA3 ← + w_DG→CA3 · DG   (CA3 recurrent already in local dynamics)
    4. CA2 ← + w_MEC→CA2 · MEC + w_CA3→CA2 · CA3
    5. CA1 ← + w_CA3→CA1 · CA3 + w_MEC→CA1 · MEC
    6. Recompute INT mean (caller should have built INT from pre- or post-step CA1;
       this pass applies INT→CA1 inhibition using the INT rates present in ``groups``).
    7. CA1 ← − w_INT→CA1 · INT

    Returns a metadata dict of weights and whether the pass ran.
    """
    params = resolve_feedforward_params(ratinabox_params)
    meta: dict[str, Any] = {
        "apply_feedforward": bool(params.get("enabled", True)),
        "feedforward_weights": {
            k: float(v) for k, v in params.items() if k.startswith("w_")
        },
        "normalize_upstream": bool(params.get("normalize_upstream", True)),
    }
    if not params.get("enabled", True) or not groups:
        meta["applied"] = False
        return meta

    n_steps = int(groups[0].rates_hz.shape[1])
    norm = bool(params.get("normalize_upstream", True))

    def _src(node: str) -> np.ndarray:
        m = _node_mean(groups, node, n_steps)
        return _normalize_trace(m) if norm else m

    # 1–2: MEC → DG
    mec = _src("MEC")
    _add_drive(groups, "DG", float(params["w_mec_to_dg"]) * mec)

    # 3: DG → CA3 (recurrent already applied locally)
    dg = _src("DG")
    _add_drive(groups, "CA3", float(params["w_dg_to_ca3"]) * dg)

    # 4: MEC + CA3 → CA2
    ca3 = _src("CA3")
    _add_drive(
        groups,
        "CA2",
        float(params["w_mec_to_ca2"]) * mec + float(params["w_ca3_to_ca2"]) * ca3,
    )

    # 5: CA3 + MEC → CA1
    _add_drive(
        groups,
        "CA1",
        float(params["w_ca3_to_ca1"]) * ca3 + float(params["w_mec_to_ca1"]) * mec,
    )

    # INT→CA1 is applied separately after interneurons are constructed
    # (see ``apply_int_to_ca1_inhibition``).
    meta["applied"] = True
    meta["node_peak_means_hz"] = {
        node: float(_node_mean(groups, node, n_steps).max())
        for node in ("MEC", "DG", "CA3", "CA2", "CA1", "SUB")
        if any(GROUP_TO_NODE.get(g.name) == node for g in groups)
    }
    return meta


def apply_int_to_ca1_inhibition(
    groups: list[_RateGroup],
    *,
    ratinabox_params: dict | None = None,
) -> dict[str, Any]:
    """Subtract normalized INT population drive from CA1 pyramidal rates."""
    params = resolve_feedforward_params(ratinabox_params)
    meta: dict[str, Any] = {
        "w_int_to_ca1": float(params["w_int_to_ca1"]),
        "applied": False,
    }
    if not params.get("enabled", True):
        return meta
    if not any(GROUP_TO_NODE.get(g.name) == "INT" for g in groups):
        return meta
    if not any(GROUP_TO_NODE.get(g.name) == "CA1" for g in groups):
        return meta

    n_steps = int(groups[0].rates_hz.shape[1])
    inten = _node_mean(groups, "INT", n_steps)
    if params.get("normalize_upstream", True):
        inten = _normalize_trace(inten)
    _add_drive(groups, "CA1", -float(params["w_int_to_ca1"]) * inten)
    meta["applied"] = True
    return meta
