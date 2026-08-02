"""Publication figure: trisynaptic / EC feedforward circuit graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch

from hippo_sim.feedforward import FEEDFORWARD_PARAMS, LOCAL_INT_EDGES
from visualization.constants import (
    CELL_TYPE_TO_CIRCUIT_NODE,
    FEEDFORWARD_DIAGRAM_NODE_ORDER,
    FIGURE_DPI,
    REGION_TO_CIRCUIT_NODE,
    circuit_node_colors,
)
from visualization.load_outputs import SimulationOutputs
from visualization.publication_style import (
    apply_publication_theme,
    panel_label,
    save_pub_figure,
)

apply_publication_theme()

# Diagram order: principals first, then one aggregated INT node.
_DIAGRAM_NODE_ORDER = FEEDFORWARD_DIAGRAM_NODE_ORDER

_INT_POOL_NODES = frozenset({
    "INT_CA1", "INT_CA2", "INT_CA3", "INT_DG", "INT_SUB", "INT",
})

# Synapse key → (source, target, kind)
_EDGES: list[tuple[str, str, str, str]] = [
    ("w_mec_to_dg", "MEC", "DG", "excitatory"),
    ("w_dg_to_ca3", "DG", "CA3", "excitatory"),
    ("w_ca3_to_ca1", "CA3", "CA1", "excitatory"),
    ("w_mec_to_ca1", "MEC", "CA1", "excitatory"),
    ("w_mec_to_ca2", "MEC", "CA2", "excitatory"),
    ("w_ca3_to_ca2", "CA3", "CA2", "excitatory"),
    ("w_mec_to_sub", "MEC", "SUB", "excitatory"),
    *[
        (weight_key, "INT", home, "inhibitory")
        for _, home, weight_key in LOCAL_INT_EDGES
    ],
]

# Canvas matches figure aspect so equal-aspect circles fill the page.
_CANVAS_W = 1.72
_CANVAS_H = 1.0

# Principals on two rows; aggregated INT sits below center.
_NODE_XY: dict[str, tuple[float, float]] = {
    "MEC": (0.28, 0.82),
    "DG": (0.86, 0.82),
    "CA3": (1.44, 0.82),
    "SUB": (0.28, 0.48),
    "CA2": (0.86, 0.48),
    "CA1": (1.44, 0.48),
    "INT": (0.86, 0.18),
}

_NODE_ROLE: dict[str, str] = {
    "MEC": "EC afferent",
    "DG": "Trisynaptic relay",
    "CA3": "Associative",
    "CA2": "CA3–MEC",
    "CA1": "Principal out",
    "INT": "Local inh.",
    "SUB": "Subicular/BVC",
}

# Compact class labels for inside-node text.
_CLASS_SHORT: dict[str, str] = {
    "MEC_grid": "grid",
    "MEC_hd": "hd",
    "MEC_speed": "speed",
    "DG_granule": "granule",
    "CA3_pyr": "pyr",
    "CA2_pyr": "pyr",
    "CA1_pyr": "pyr",
    "INT_CA1": "int",
    "INT_CA2": "int",
    "INT_CA3": "int",
    "INT_DG": "int",
    "INT_SUB": "int",
    "interneuron": "int",
    "CA1_int": "int",
    "Sub_bvc": "bvc",
}

_NODE_RADIUS = 0.100
_INT_RADIUS = 0.092

# Per-edge arc routing (arc3 rad). Optional label_t ∈ (0, 1) slides the weight along the arc.
_EDGE_STYLE: dict[tuple[str, str], dict[str, float]] = {
    ("MEC", "DG"): {"rad": 0.0},
    ("DG", "CA3"): {"rad": 0.0},
    ("MEC", "CA1"): {"rad": -0.52},
    ("MEC", "CA2"): {"rad": -0.08},
    ("CA3", "CA2"): {"rad": -0.12},
    ("CA3", "CA1"): {"rad": 0.08},
    ("MEC", "SUB"): {"rad": 0.0},
    ("INT", "DG"): {"rad": 0.58},
    ("INT", "CA2"): {"rad": -0.22},
    ("INT", "CA1"): {"rad": 0.16},
    ("INT", "CA3"): {"rad": 0.14},
    ("INT", "SUB"): {"rad": -0.18},
}


def _edge_rad(src: str, tgt: str) -> float:
    style = _EDGE_STYLE.get((src, tgt), {})
    return float(style.get("rad", 0.0))


def _edge_label_t(src: str, tgt: str) -> float:
    style = _EDGE_STYLE.get((src, tgt), {})
    return float(style.get("label_t", 0.5))


def _arc3_point(
    x0: float, y0: float, x1: float, y1: float, rad: float, t: float,
) -> tuple[float, float]:
    """Point on matplotlib ``arc3`` curve (matches FancyArrowPatch routing)."""
    t = max(0.05, min(0.95, t))
    if abs(rad) < 1e-12:
        return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    cx = 0.5 * (x0 + x1) + (y1 - y0) * rad
    cy = 0.5 * (y0 + y1) - (x1 - x0) * rad
    omt = 1.0 - t
    x = omt * omt * x0 + 2.0 * omt * t * cx + t * t * x1
    y = omt * omt * y0 + 2.0 * omt * t * cy + t * t * y1
    return x, y


def _edge_label_xy(
    src: str, tgt: str, x0: float, y0: float, x1: float, y1: float,
) -> tuple[float, float]:
    return _arc3_point(
        x0, y0, x1, y1, _edge_rad(src, tgt), _edge_label_t(src, tgt),
    )


def _diagram_node(node: str) -> str:
    if node in _INT_POOL_NODES or node.startswith("INT_"):
        return "INT"
    return node


def _circuit_node_for_unit(row: pd.Series) -> str:
    ct = str(row.get("cell_type", ""))
    if ct in CELL_TYPE_TO_CIRCUIT_NODE:
        return CELL_TYPE_TO_CIRCUIT_NODE[ct]
    region = str(row.get("region", ""))
    return REGION_TO_CIRCUIT_NODE.get(region, "OTHER")


def _int_homes_with_pools(units: pd.DataFrame) -> set[str]:
    """Home principals that have a captured local INT pool in this session."""
    present_int = set(units["circuit_node"].astype(str))
    return {
        home
        for int_node, home, _ in LOCAL_INT_EDGES
        if int_node in present_int
    }


def _load_feedforward_meta(data: SimulationOutputs) -> dict[str, Any]:
    path = Path(data.input_dir) / "neural_backend_metadata.json"
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            blob = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    ff = blob.get("feedforward")
    return ff if isinstance(ff, dict) else {}


def _node_summary(data: SimulationOutputs) -> pd.DataFrame:
    units = data.units.copy()
    units["circuit_node"] = units.apply(_circuit_node_for_unit, axis=1)
    units["diagram_node"] = units["circuit_node"].map(_diagram_node)
    units["mean_rate_hz"] = units["unit_id"].map(data.unit_mean_rates_gt).fillna(0.0)
    rows = []
    for node in _DIAGRAM_NODE_ORDER:
        mask = units["diagram_node"] == node
        if not mask.any():
            continue
        classes = sorted(units.loc[mask, "cell_type"].astype(str).unique())
        short = sorted({_CLASS_SHORT.get(c, c) for c in classes})
        rows.append({
            "node": node,
            "n_units": int(mask.sum()),
            "mean_rate_hz": float(units.loc[mask, "mean_rate_hz"].mean()),
            "cell_classes": ", ".join(classes),
            "cell_classes_short": ", ".join(short),
            "role": _NODE_ROLE.get(node, ""),
        })
    return pd.DataFrame(rows)


def _weights_from_meta(ff: dict[str, Any]) -> dict[str, float]:
    weights = dict(FEEDFORWARD_PARAMS)
    stored = ff.get("feedforward_weights")
    if isinstance(stored, dict):
        for k, v in stored.items():
            try:
                weights[k] = float(v)
            except (TypeError, ValueError):
                continue
    for key in ("local_int_inhibition", "int_to_ca1"):
        int_meta = ff.get(key)
        if not isinstance(int_meta, dict):
            continue
        if "w_int_to_ca1" in int_meta:
            try:
                weights["w_int_to_ca1"] = float(int_meta["w_int_to_ca1"])
            except (TypeError, ValueError):
                pass
        edges = int_meta.get("edges")
        if isinstance(edges, dict):
            for edge in edges.values():
                if not isinstance(edge, dict):
                    continue
                wk = edge.get("weight_key")
                if wk and "weight" in edge:
                    try:
                        weights[str(wk)] = float(edge["weight"])
                    except (TypeError, ValueError):
                        continue
    return weights


def _text_color_for(face: tuple) -> str:
    """Light text on dark fills, dark text on light fills."""
    r, g, b = face[:3]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else "0.12"


def _edge_color_for(face: tuple, *, factor: float = 0.62) -> tuple[float, float, float, float]:
    r, g, b = face[:3]
    a = face[3] if len(face) > 3 else 1.0
    return (max(r * factor, 0.0), max(g * factor, 0.0), max(b * factor, 0.0), a)


def _draw_graph(
    ax,
    *,
    present: set[str],
    summary: pd.DataFrame,
    weights: dict[str, float],
    colors: dict[str, tuple],
    int_build_homes: set[str],
) -> None:
    ax.set_xlim(0.0, _CANVAS_W)
    ax.set_ylim(0.0, _CANVAS_H)
    ax.set_aspect("equal")
    ax.axis("off")

    info = {str(r["node"]): r for _, r in summary.iterrows()}
    int_homes = int_build_homes & present

    # Edges first (under nodes).
    for key, src, tgt, kind in _EDGES:
        if src not in present or tgt not in present:
            continue
        if kind == "inhibitory" and tgt not in int_homes:
            continue
        if src not in _NODE_XY or tgt not in _NODE_XY:
            continue
        w = float(weights.get(key, 0.0))
        if abs(w) <= 1e-12:
            continue
        x0, y0 = _NODE_XY[src]
        x1, y1 = _NODE_XY[tgt]
        rad = _edge_rad(src, tgt)

        color = "#2c3e50" if kind == "excitatory" else "#c0392b"
        lw = 1.4 + 3.2 * min(abs(w) / 0.35, 1.0)
        ls = "-" if kind == "excitatory" else (0, (4, 2.5))
        shrink = 44 if src == "INT" or tgt == "INT" else 48
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="->",
            mutation_scale=16,
            lw=lw,
            color=color,
            linestyle=ls,
            alpha=0.85,
            shrinkA=shrink,
            shrinkB=shrink,
            zorder=1,
        )
        ax.add_patch(arrow)

        mx, my = _edge_label_xy(src, tgt, x0, y0, x1, y1)
        ax.text(
            mx, my, f"{w:.2f}",
            fontsize=9.5 if src == "INT" else 11,
            ha="center", va="center",
            color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.04", fc="white", ec="none", alpha=1.0),
            zorder=2,
        )

    for node in _DIAGRAM_NODE_ORDER:
        if node not in present:
            continue
        if node not in _NODE_XY:
            continue
        x, y = _NODE_XY[node]
        row = info[node]
        face = colors.get(node, (0.6, 0.6, 0.6, 1.0))
        is_int = node == "INT"
        r_node = _INT_RADIUS if is_int else _NODE_RADIUS
        edge = _edge_color_for(face)
        tc = _text_color_for(face)

        circ = Circle(
            (x, y), r_node,
            facecolor=face,
            edgecolor=edge,
            linewidth=2.0 if is_int else 1.7,
            zorder=3,
            alpha=0.97,
        )
        ax.add_patch(circ)

        ax.text(
            x, y + (0.038 if is_int else 0.042), node,
            ha="center", va="center",
            fontsize=12 if is_int else 15, fontweight="bold", color=tc, zorder=4,
        )
        ax.text(
            x, y + 0.004,
            f"n={int(row['n_units'])} · {row['mean_rate_hz']:.1f}",
            ha="center", va="center",
            fontsize=8.5 if is_int else 10, color=tc, zorder=4, alpha=0.95,
        )
        if not is_int:
            ax.text(
                x, y - 0.028, str(row["cell_classes_short"]),
                ha="center", va="center",
                fontsize=9.5, color=tc, zorder=4, alpha=0.95,
            )
            ax.text(
                x, y - 0.052, str(row["role"]),
                ha="center", va="center",
                fontsize=9, style="italic", color=tc, zorder=4, alpha=0.90,
            )
        else:
            ax.text(
                x, y - 0.028, str(row["cell_classes_short"]),
                ha="center", va="center",
                fontsize=8.5, color=tc, zorder=4, alpha=0.95,
            )
            ax.text(
                x, y - 0.052, str(row["role"]),
                ha="center", va="center",
                fontsize=8, style="italic", color=tc, zorder=4, alpha=0.90,
            )

    ax.plot([], [], color="#2c3e50", lw=2.2, label="Excitatory drive (weight)")
    ax.plot([], [], color="#c0392b", lw=2.2, ls="--", label="Inhibitory (INT→home)")
    ax.legend(
        loc="lower center", fontsize=11, frameon=False, ncol=2,
        bbox_to_anchor=(0.42, -0.01),
    )


def plot_fig_circuit_feedforward(
    data: SimulationOutputs, output_dir: Path,
) -> Path:
    """Full-page feedforward graph with node stats inside each node."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ff = _load_feedforward_meta(data)
    weights = _weights_from_meta(ff)
    summary = _node_summary(data)
    present = set(summary["node"].astype(str)) if not summary.empty else set()
    colors = circuit_node_colors(_DIAGRAM_NODE_ORDER)

    units = data.units.copy()
    units["circuit_node"] = units.apply(_circuit_node_for_unit, axis=1)
    int_build_homes = _int_homes_with_pools(units)

    fig = plt.figure(figsize=(15.5, 9.0))
    # Axes aspect ≈ figure aspect so equal-aspect circles fill the page.
    ax = fig.add_axes([0.02, 0.07, 0.96, 0.85])
    _draw_graph(
        ax,
        present=present,
        summary=summary,
        weights=weights,
        colors=colors,
        int_build_homes=int_build_homes,
    )
    panel_label(ax, "A", x=-0.01, y=1.03)

    return save_pub_figure(
        fig, output_dir / "fig_circuit_feedforward.png", dpi=FIGURE_DPI,
        pad_inches=0.25,
        adjust=False,
    )


def generate_circuit_feedforward_plots(
    data: SimulationOutputs, output_dir: Path,
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [plot_fig_circuit_feedforward(data, output_dir)]
