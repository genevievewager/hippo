"""Shared layout helpers so publication panels avoid text/legend overlap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

_OPEN_AXES_ENABLED = False

# Publication-readable defaults (were too small / too close previously).
LEGEND_FONTSIZE = 11
LEGEND_TITLE_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 12
TICK_LABEL_FONTSIZE = 9
PANEL_LABEL_FONTSIZE = 13


def strip_box_frames(fig: Figure) -> None:
    """Keep left/bottom axes only; drop closed boxes and colorbar outlines."""
    for ax in fig.axes:
        spines = ax.spines
        if ax.get_label() == "<colorbar>" or "outline" in spines:
            for spine in spines.values():
                spine.set_visible(False)
            continue
        if "top" in spines:
            spines["top"].set_visible(False)
        if "right" in spines:
            spines["right"].set_visible(False)


def enable_open_axes() -> None:
    """Apply open-axes defaults for all subsequent figures in this process."""
    global _OPEN_AXES_ENABLED
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.labelsize"] = AXIS_LABEL_FONTSIZE
    plt.rcParams["xtick.labelsize"] = TICK_LABEL_FONTSIZE
    plt.rcParams["ytick.labelsize"] = TICK_LABEL_FONTSIZE
    plt.rcParams["legend.fontsize"] = LEGEND_FONTSIZE
    if _OPEN_AXES_ENABLED:
        return

    _orig_savefig = Figure.savefig

    def _savefig(self, *args, **kwargs):
        strip_box_frames(self)
        return _orig_savefig(self, *args, **kwargs)

    Figure.savefig = _savefig  # type: ignore[method-assign]
    _OPEN_AXES_ENABLED = True


def apply_publication_theme(*, font_scale: float = 1.35) -> None:
    """Seaborn paper ticks theme without closed axis boxes."""
    import seaborn as sns

    sns.set_theme(style="ticks", context="paper", font_scale=font_scale)
    enable_open_axes()


def style_axes(ax: Axes, *, labelsize: float | None = None, ticksize: float | None = None) -> None:
    """Ensure axis labels / ticks are large enough to read in the PDF."""
    labelsize = AXIS_LABEL_FONTSIZE if labelsize is None else labelsize
    ticksize = TICK_LABEL_FONTSIZE if ticksize is None else ticksize
    ax.xaxis.label.set_size(labelsize)
    ax.yaxis.label.set_size(labelsize)
    ax.tick_params(axis="both", labelsize=ticksize)
    title = ax.get_title()
    if title:
        ax.title.set_size(max(labelsize, 10))


def style_figure_axes(fig: Figure) -> None:
    """Apply ``style_axes`` to all non-colorbar axes on a figure."""
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>" or "outline" in ax.spines:
            continue
        style_axes(ax)


def panel_label(
    ax: Axes,
    label: str,
    *,
    x: float = -0.02,
    y: float = 1.06,
    ha: str = "right",
    va: str = "bottom",
) -> None:
    """Bold panel letter (default: above the axes, clear of data)."""
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=PANEL_LABEL_FONTSIZE, fontweight="bold", va=va, ha=ha,
        clip_on=False,
    )


def legend_outside(
    ax: Axes,
    *,
    loc: str = "upper left",
    bbox: tuple[float, float] = (1.04, 1.0),
    ncol: int = 1,
    fontsize: float | None = None,
    title: str | None = None,
    frameon: bool = False,
    **kwargs,
):
    """Place legend outside the axes so it cannot cover data."""
    fontsize = LEGEND_FONTSIZE if fontsize is None else fontsize
    leg = ax.legend(
        loc=loc,
        bbox_to_anchor=bbox,
        borderaxespad=0.4,
        frameon=frameon,
        fontsize=fontsize,
        ncol=ncol,
        title=title,
        **kwargs,
    )
    if title and leg is not None and leg.get_title() is not None:
        leg.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)
    return leg


def legend_below(
    ax: Axes,
    *,
    ncol: int = 2,
    fontsize: float | None = None,
    frameon: bool = False,
    y: float = -0.28,
    **kwargs,
):
    """Place legend centered below the axes (clear of x tick labels)."""
    fontsize = LEGEND_FONTSIZE if fontsize is None else fontsize
    title = kwargs.pop("title", None)
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, y),
        borderaxespad=0.2,
        frameon=frameon,
        fontsize=fontsize,
        ncol=ncol,
        title=title,
        **kwargs,
    )
    if title and leg is not None and leg.get_title() is not None:
        leg.get_title().set_fontsize(LEGEND_TITLE_FONTSIZE)
    return leg


def figure_legend_below(
    fig: Figure,
    handles,
    labels,
    *,
    ncol: int = 4,
    fontsize: float | None = None,
    title_fontsize: float | None = None,
    y: float = 0.02,
    title: str | None = None,
    markerscale: float = 1.4,
):
    """Figure-level legend below all panels (avoids per-axis data overlap)."""
    fontsize = LEGEND_FONTSIZE if fontsize is None else fontsize
    if title_fontsize is None:
        title_fontsize = LEGEND_TITLE_FONTSIZE if title else None
    return fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        fontsize=fontsize,
        frameon=False,
        borderaxespad=0.5,
        title=title,
        title_fontsize=title_fontsize,
        markerscale=markerscale,
        handlelength=2.2,
        columnspacing=1.2,
        labelspacing=0.45,
    )


def clear_axes_legends(fig: Figure) -> None:
    """Remove any axes-level legends (use before a shared figure legend)."""
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()


def save_pub_figure(
    fig: Figure,
    path: Path,
    *,
    dpi: int,
    rect: tuple[float, float, float, float] = (0.10, 0.16, 0.96, 0.92),
    hspace: float = 0.55,
    wspace: float = 0.45,
    pad_inches: float = 0.55,
    adjust: bool = True,
) -> Path:
    """Save with reserved margins for labels/legends outside axes.

    Prefer placing legends in dedicated subplot axes (or figure-level
    legends in the bottom margin). Do not call ``tight_layout`` — it pulls
    outside legends onto data. ``bbox_inches='tight'`` only crops empty
    canvas so PDF pages are not dominated by whitespace.

    Set ``adjust=False`` when the figure already set margins via GridSpec
    (e.g. nested grids + ``make_axes_locatable`` colorbars); re-running
    ``subplots_adjust`` would undo those positions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    enable_open_axes()
    style_figure_axes(fig)
    strip_box_frames(fig)
    if adjust:
        fig.subplots_adjust(
            left=rect[0],
            bottom=rect[1],
            right=rect[2],
            top=min(rect[3], 0.94),
            wspace=wspace,
            hspace=hspace,
        )
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
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
