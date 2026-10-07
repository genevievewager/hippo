"""Publication-style helpers aligned with agents/quadrant_n5/figures."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

MM = 1 / 25.4
W_FULL = 180 * MM
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8f8d87", "#e4e3df"
FAILC = "#c83a39"

COL = {
    "raw": "#52514e",
    "raw_lag": "#a3a19b",
    "raw_smooth": "#3a3936",
    "pca": "#2a78d6",
    "pca_smooth": "#2a78d6",
    "dm": "#eb6834",
    "dm_smooth": "#eb6834",
    "lds": "#1baf7a",
    "lds_smooth": "#1baf7a",
    "isomap": "#c4c2bb",
    "gpfa": "#9a9890",
    "gpfa_causal": "#0e7a54",
}
SHORT = {
    "raw": "Raw",
    "raw_lag": "Raw+hist",
    "raw_smooth": "Raw+EMA",
    "pca": "PCA",
    "pca_smooth": "PCA+EMA",
    "dm": "DM",
    "dm_smooth": "DM+EMA",
    "lds": "LDS",
    "lds_smooth": "LDS+EMA",
    "isomap": "Isomap",
    "gpfa": "GPFA†",
    "gpfa_causal": "GPFA-c",
}
GPFA_OFFLINE_NOTE = "† GPFA = offline non-causal smoother (reference only; excluded from contrasts)."


def register_font() -> str:
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        str(Path(__file__).resolve().parents[3]
            / "agents/quadrant_n5/figures/fonts/LiberationSans-Regular.ttf"),
    ]
    for reg in candidates:
        bold = reg.replace("Regular", "Bold")
        if os.path.isfile(reg) and os.path.isfile(bold):
            font_manager.fontManager.addfont(reg)
            font_manager.fontManager.addfont(bold)
            name = font_manager.FontProperties(fname=reg).get_name()
            mpl.rcParams["font.family"] = "sans-serif"
            mpl.rcParams["font.sans-serif"] = [name, "Liberation Sans", "Arial"]
            return name
    mpl.rcParams["font.family"] = "sans-serif"
    return "DejaVu Sans"


def apply_rc() -> None:
    register_font()
    mpl.rcParams.update({
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.3,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#3a3936", "xtick.color": "#3a3936", "ytick.color": "#3a3936",
        "axes.labelcolor": INK, "text.color": INK,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 120, "lines.linewidth": 1.0,
    })


def head(ax, letter: str, title: str = "", *, x: float = 0.0, y: float = 1.05) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold",
            va="bottom", ha="left", clip_on=False)
    if title:
        ax.text(x + 0.08, y + 0.002, title, transform=ax.transAxes,
                fontsize=7, fontweight="bold", va="bottom", ha="left", clip_on=False)


def save_fig(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png
