"""Trisynaptic / entorhinal feedforward coupling for RatInABox group rates.

After each population has local RiaB tuning + hippocampal overlays, region
population means drive downstream targets. The default sketch is the classic
hippocampal loop with **local** per-region interneuron feedback:

    MEC ──► DG ──► CA3 ──► CA1
     │       ▲      ▲       ▲
     │      INT_DG INT_CA3 INT_CA1
     └──────┴──────┴───────┘
            (+ INT_CA2 / INT_SUB locally)

Trajectory-gated capture may omit CA fields. In that case edges whose targets
are absent are skipped, and a ``sub_ent_dominant`` profile can strengthen
MEC→SUB coupling for Subiculum/entorhinal-heavy insertions.

All weights are tunable in ``FEEDFORWARD_PARAMS`` / ``ratinabox_params``.
"""

from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from hippo.anatomy.hippocampal_system import CIRCUIT_PROFILES, infer_circuit_profile

# Default synaptic / gain weights (Hz of drive per normalized upstream mean).
FEEDFORWARD_PARAMS: dict[str, Any] = {
    "enabled": True,
    "profile": "auto",  # auto | trisynaptic_full | sub_ent_dominant | mixed_hpc
    "w_mec_to_dg": 0.20,
    "w_dg_to_ca3": 0.25,
    "w_ca3_to_ca1": 0.20,
    "w_mec_to_ca1": 0.15,
    "w_int_to_ca1": 0.30,  # local INT_CA1 → CA1 (legacy key name kept)
    "w_int_to_ca3": 0.25,
    "w_int_to_dg": 0.25,
    "w_int_to_ca2": 0.20,
    "w_int_to_sub": 0.20,
    "w_mec_to_ca2": 0.10,
    "w_ca3_to_ca2": 0.10,
    "w_mec_to_sub": 0.0,  # used by sub_ent / mixed profiles
    "normalize_upstream": True,
}

# Local INT node → (home principal node, weight key).
LOCAL_INT_EDGES: tuple[tuple[str, str, str], ...] = (
    ("INT_CA1", "CA1", "w_int_to_ca1"),
    ("INT_CA3", "CA3", "w_int_to_ca3"),
    ("INT_DG", "DG", "w_int_to_dg"),
    ("INT_CA2", "CA2", "w_int_to_ca2"),
    ("INT_SUB", "SUB", "w_int_to_sub"),
)

# Group name → circuit node used for mean pooling / targeting.
GROUP_TO_NODE: dict[str, str] = {
    "MEC_grid": "MEC",
    "MEC_hd": "MEC",
    "MEC_speed": "MEC",
    "DG_place": "DG",
    "CA3_place": "CA3",
    "CA1_place_pp": "CA1",
    "CA2_place": "CA2",
    "INT_CA1": "INT_CA1",
    "INT_CA3": "INT_CA3",
    "INT_DG": "INT_DG",
    "INT_CA2": "INT_CA2",
    "INT_SUB": "INT_SUB",
    # Legacy aliases → CA1-local INT pool.
    "interneuron": "INT_CA1",
    "CA1_int": "INT_CA1",
    "Sub_bvc": "SUB",
}


class _RateGroup(Protocol):
    name: str
    rates_hz: np.ndarray


def _present_nodes(groups: list[_RateGroup]) -> set[str]:
    return {GROUP_TO_NODE[g.name] for g in groups if g.name in GROUP_TO_NODE}


def _profile_from_groups(groups: list[_RateGroup]) -> str:
    nodes = _present_nodes(groups)
    # Map circuit nodes → canonical region labels for profile inference.
    region_like = set()
    mapping = {
        "CA1": "CA1", "CA2": "CA2", "CA3": "CA3",
        "DG": "DG", "SUB": "Subiculum", "MEC": "MEC",
        "INT_CA1": "CA1", "INT_CA2": "CA2", "INT_CA3": "CA3",
        "INT_DG": "DG", "INT_SUB": "Subiculum",
    }
    for node in nodes:
        if node in mapping:
            region_like.add(mapping[node])
    return infer_circuit_profile(regions=region_like)


def resolve_feedforward_params(
    ratinabox_params: dict | None = None,
    *,
    groups: list[_RateGroup] | None = None,
) -> dict[str, Any]:
    """Merge defaults, profile overrides, and ``ratinabox_params['feedforward']``.

    Order: ``FEEDFORWARD_PARAMS`` → circuit profile weights → explicit
    ``ratinabox_params['feedforward']`` overrides (so callers can force weights).
    """
    user_ff: dict[str, Any] = {}
    enabled = True
    requested_profile = "auto"
    if ratinabox_params:
        if "apply_feedforward" in ratinabox_params:
            enabled = bool(ratinabox_params["apply_feedforward"])
        overrides = ratinabox_params.get("feedforward")
        if isinstance(overrides, dict):
            user_ff = dict(overrides)
            requested_profile = str(user_ff.pop("profile", "auto"))

    profile = requested_profile
    if profile == "auto":
        profile = _profile_from_groups(groups) if groups else "trisynaptic_full"

    merged: dict[str, Any] = dict(FEEDFORWARD_PARAMS)
    profile_ff = (CIRCUIT_PROFILES.get(profile) or {}).get("feedforward") or {}
    merged.update(profile_ff)
    merged.update(user_ff)
    merged["enabled"] = enabled
    merged["profile"] = requested_profile
    merged["resolved_profile"] = profile
    return merged


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

    Edges whose target populations were not captured (trajectory gating) are
    skipped. Profile ``sub_ent_dominant`` additionally drives SUB from MEC.
    """
    params = resolve_feedforward_params(ratinabox_params, groups=groups)
    meta: dict[str, Any] = {
        "apply_feedforward": bool(params.get("enabled", True)),
        "feedforward_profile": params.get("resolved_profile"),
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
    present_nodes = _present_nodes(groups)

    def _src(node: str) -> np.ndarray:
        m = _node_mean(groups, node, n_steps)
        return _normalize_trace(m) if norm else m

    mec = _src("MEC")
    if "DG" in present_nodes:
        _add_drive(groups, "DG", float(params.get("w_mec_to_dg", 0.0)) * mec)

    dg = _src("DG")
    if "CA3" in present_nodes:
        _add_drive(groups, "CA3", float(params.get("w_dg_to_ca3", 0.0)) * dg)

    ca3 = _src("CA3")
    if "CA2" in present_nodes:
        _add_drive(
            groups,
            "CA2",
            float(params.get("w_mec_to_ca2", 0.0)) * mec
            + float(params.get("w_ca3_to_ca2", 0.0)) * ca3,
        )

    if "CA1" in present_nodes:
        _add_drive(
            groups,
            "CA1",
            float(params.get("w_ca3_to_ca1", 0.0)) * ca3
            + float(params.get("w_mec_to_ca1", 0.0)) * mec,
        )

    # Trajectory-aware: MEC → Subiculum (important for SUB/ENT insertions).
    if "SUB" in present_nodes and float(params.get("w_mec_to_sub", 0.0)) != 0.0:
        _add_drive(groups, "SUB", float(params["w_mec_to_sub"]) * mec)

    meta["applied"] = True
    meta["present_nodes"] = sorted(present_nodes)
    meta["node_peak_means_hz"] = {
        node: float(_node_mean(groups, node, n_steps).max())
        for node in ("MEC", "DG", "CA3", "CA2", "CA1", "SUB")
        if any(GROUP_TO_NODE.get(g.name) == node for g in groups)
    }
    return meta


def apply_local_int_inhibition(
    groups: list[_RateGroup],
    *,
    ratinabox_params: dict | None = None,
) -> dict[str, Any]:
    """Subtract each local INT population drive from its home principal rates."""
    params = resolve_feedforward_params(ratinabox_params, groups=groups)
    edges_meta: dict[str, Any] = {}
    meta: dict[str, Any] = {
        "applied": False,
        "feedforward_profile": params.get("resolved_profile"),
        "edges": edges_meta,
    }
    if not params.get("enabled", True) or not groups:
        return meta

    n_steps = int(groups[0].rates_hz.shape[1])
    norm = bool(params.get("normalize_upstream", True))
    any_applied = False

    for int_node, home_node, weight_key in LOCAL_INT_EDGES:
        w = float(params.get(weight_key, 0.0))
        edge = {
            "int_node": int_node,
            "home_node": home_node,
            "weight_key": weight_key,
            "weight": w,
            "applied": False,
        }
        edges_meta[f"{int_node}_to_{home_node}"] = edge
        if w == 0.0:
            continue
        if not any(GROUP_TO_NODE.get(g.name) == home_node for g in groups):
            continue
        if not any(GROUP_TO_NODE.get(g.name) == int_node for g in groups):
            continue

        inten = _node_mean(groups, int_node, n_steps)
        if norm:
            inten = _normalize_trace(inten)
        _add_drive(groups, home_node, -w * inten)
        edge["applied"] = True
        any_applied = True

    meta["applied"] = any_applied
    # Backward-compat mirror of the CA1 edge weight.
    ca1_edge = edges_meta.get("INT_CA1_to_CA1") or {}
    meta["w_int_to_ca1"] = float(ca1_edge.get("weight", params.get("w_int_to_ca1", 0.0)))
    return meta


def apply_int_to_ca1_inhibition(
    groups: list[_RateGroup],
    *,
    ratinabox_params: dict | None = None,
) -> dict[str, Any]:
    """Legacy alias for ``apply_local_int_inhibition`` (all local INT edges)."""
    return apply_local_int_inhibition(groups, ratinabox_params=ratinabox_params)
