"""Shared layout helpers so publication panels avoid text/legend overlap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def panel_label(ax: Axes, label: str, *, x: float = -0.02, y: float = 1.04) -> None:
    """Bold panel letter above the axes, clear of data."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="bottom", ha="right",
        clip_on=False,
    )


def legend_outside(
    ax: Axes,
    *,
    loc: str = "upper left",
    bbox: tuple[float, float] = (1.02, 1.0),
    ncol: int = 1,
    fontsize: float = 7,
    title: str | None = None,
    frameon: bool = False,
    **kwargs,
):
    """Place legend outside the axes so it cannot cover data."""
    return ax.legend(
        loc=loc,
        bbox_to_anchor=bbox,
        borderaxespad=0.0,
        frameon=frameon,
        fontsize=fontsize,
        ncol=ncol,
        title=title,
        **kwargs,
    )


def legend_below(
    ax: Axes,
    *,
    ncol: int = 2,
    fontsize: float = 7,
    frameon: bool = False,
    y: float = -0.22,
    **kwargs,
):
    """Place legend centered below the axes."""
    return ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0.0,
        frameon=frameon,
        fontsize=fontsize,
        ncol=ncol,
        **kwargs,
    )


def figure_legend_below(
    fig: Figure,
    handles,
    labels,
    *,
    ncol: int = 4,
    fontsize: float = 7,
    y: float = -0.02,
):
    """Figure-level legend below all panels (avoids per-axis data overlap)."""
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        borderaxespad=0.0,
    )


def save_pub_figure(
    fig: Figure,
    path: Path,
    *,
    dpi: int,
    rect: tuple[float, float, float, float] = (0.08, 0.10, 0.98, 0.90),
) -> Path:
    """Save with reserved margins for labels/legends outside axes.

    Avoid ``tight_layout`` here: it frequently pulls ``bbox_to_anchor`` legends
    back on top of data.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(
        left=rect[0],
        bottom=rect[1],
        right=rect[2],
        top=min(rect[3], 0.90),
        wspace=0.45,
        hspace=0.45,
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    return path


def expand_xlim_for_labels(ax: Axes, values, *, pad_frac: float = 0.35) -> None:
    """Widen x-limits so right-side bar annotations stay inside the axes."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return
    lo = min(0.0, min(vals))
    hi = max(vals)
    span = max(hi - lo, 1e-6)
    ax.set_xlim(lo, hi + pad_frac * span)
