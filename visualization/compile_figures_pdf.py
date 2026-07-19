"""Compile experiment figures/ PNGs into a single sectioned PDF report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from visualization.figure_captions import caption_for, title_for

SECTION_TITLES: dict[str, str] = {
    ".": "Simulation visualizations",
    "realtime_decoding": "Realtime closed-loop decoding",
    "realtime_decoding/comparison": "Realtime decoding — ground truth vs sorted comparison",
    "realtime_decoding/ground_truth": "Realtime decoding — ground-truth spikes",
    "realtime_decoding/sorted": "Realtime decoding — Neuropixels (sorted)",
    "decoder_comparison": "Decoder comparison — source summary",
    "decoder_comparison/ground_truth": "Decoder comparison — ground-truth spikes",
    "decoder_comparison/sorted": "Decoder comparison — Neuropixels (sorted)",
    "temporal_decoding": "Temporal manifold decoding",
}

# Preferred section order; any other relative dirs are appended alphabetically.
SECTION_ORDER = [
    ".",
    "decoder_comparison",
    "decoder_comparison/ground_truth",
    "decoder_comparison/sorted",
    "realtime_decoding",
    "realtime_decoding/comparison",
    "realtime_decoding/ground_truth",
    "realtime_decoding/sorted",
    "temporal_decoding",
]


def _section_key(figures_dir: Path, png_path: Path) -> str:
    """Map a PNG path to a section key, collapsing known nested run folders."""
    rel = png_path.parent.relative_to(figures_dir)
    if str(rel) == ".":
        return "."
    key = rel.as_posix()
    # Prefer the longest known section prefix (e.g. realtime_decoding/sorted/...).
    known = sorted(
        (k for k in SECTION_TITLES if k != "."),
        key=len,
        reverse=True,
    )
    for prefix in known:
        if key == prefix or key.startswith(prefix + "/"):
            return prefix
    return key


def _title_for_section(section_key: str) -> str:
    if section_key in SECTION_TITLES:
        return SECTION_TITLES[section_key]
    return section_key.replace("/", " — ").replace("_", " ").title()


def _ordered_sections(section_keys: set[str]) -> list[str]:
    ordered = [key for key in SECTION_ORDER if key in section_keys]
    extras = sorted(key for key in section_keys if key not in SECTION_ORDER)
    return ordered + extras


def collect_pngs_by_section(figures_dir: Path) -> dict[str, list[Path]]:
    """Group PNG files under figures_dir by logical section."""
    figures_dir = Path(figures_dir)
    groups: dict[str, list[Path]] = {}
    for png in sorted(figures_dir.rglob("*.png")):
        key = _section_key(figures_dir, png)
        groups.setdefault(key, []).append(png)
    for paths in groups.values():
        paths.sort(key=lambda p: p.name.lower())
    return groups


def _add_title_page(pdf: PdfPages, title: str, subtitle: str | None = None) -> None:
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(
        0.5, 0.58, title,
        ha="center", va="center", fontsize=22, fontweight="bold",
        wrap=True,
    )
    if subtitle:
        ax.text(
            0.5, 0.42, subtitle,
            ha="center", va="center", fontsize=12, color="#444444",
        )
    pdf.savefig(fig)
    plt.close(fig)


def _wrap_caption(text: str, width: int = 108) -> str:
    """Simple word wrap for caption text (keeps paragraphs intact)."""
    words = text.split()
    if not words:
        return text
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        trial = " ".join(current + [word])
        if current and len(trial) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def _add_image_page(
    pdf: PdfPages,
    image_path: Path,
    *,
    figure_number: int,
    figures_dir: Path,
) -> None:
    img = plt.imread(image_path)
    caption = caption_for(
        image_path,
        figure_number=figure_number,
        figures_dir=figures_dir,
    )
    caption_wrapped = _wrap_caption(caption)

    # Landscape letter page: image above, caption below (article style).
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    ax_img = fig.add_axes([0.05, 0.22, 0.90, 0.70])
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(title_for(image_path), fontsize=11, pad=6)

    ax_cap = fig.add_axes([0.07, 0.04, 0.86, 0.16])
    ax_cap.axis("off")
    ax_cap.text(
        0.0,
        1.0,
        caption_wrapped,
        ha="left",
        va="top",
        fontsize=9,
        linespacing=1.35,
        wrap=True,
        family="serif",
    )

    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def compile_figures_pdf(
    figures_dir: Path,
    output_pdf: Path | None = None,
) -> Path:
    """
    Write a PDF under figures_dir with one title page per section, then its PNGs.

    Each figure page includes a numbered research-style caption beneath the
    image. Default output: figures_dir / output.pdf
    """
    figures_dir = Path(figures_dir)
    if not figures_dir.is_dir():
        raise FileNotFoundError(f"Figures directory not found: {figures_dir}")

    groups = collect_pngs_by_section(figures_dir)
    if not groups:
        raise FileNotFoundError(f"No PNG files found under {figures_dir}")

    output_pdf = Path(output_pdf) if output_pdf is not None else figures_dir / "output.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    sections = _ordered_sections(set(groups))
    figure_number = 0
    with PdfPages(output_pdf) as pdf:
        _add_title_page(
            pdf,
            "Experiment figure report",
            subtitle=str(figures_dir.resolve()),
        )
        for section_key in sections:
            paths = groups[section_key]
            if not paths:
                continue
            _add_title_page(
                pdf,
                _title_for_section(section_key),
                subtitle=f"{len(paths)} figure(s) · {section_key}",
            )
            for png_path in paths:
                figure_number += 1
                _add_image_page(
                    pdf,
                    png_path,
                    figure_number=figure_number,
                    figures_dir=figures_dir,
                )

    return output_pdf
