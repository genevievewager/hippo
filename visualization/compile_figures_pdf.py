"""Compile experiment figures/ PNGs into a single sectioned PDF report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SECTION_TITLES: dict[str, str] = {
    ".": "Simulation visualizations",
    "realtime_decoding": "Realtime closed-loop decoding",
    "realtime_decoding/comparison": "Realtime decoding — ground truth vs sorted comparison",
    "realtime_decoding/ground_truth": "Realtime decoding — ground-truth spikes",
    "realtime_decoding/sorted": "Realtime decoding — Neuropixels (sorted)",
    "decoder_comparison": "Decoder comparison — source summary",
    "decoder_comparison/ground_truth": "Decoder comparison — ground-truth spikes",
    "decoder_comparison/sorted": "Decoder comparison — Neuropixels (sorted)",
}

# Preferred section order; any other relative dirs are appended alphabetically.
SECTION_ORDER = [
    ".",
    "realtime_decoding",
    "realtime_decoding/comparison",
    "realtime_decoding/ground_truth",
    "realtime_decoding/sorted",
    "decoder_comparison",
    "decoder_comparison/ground_truth",
    "decoder_comparison/sorted",
]


def _section_key(figures_dir: Path, png_path: Path) -> str:
    rel = png_path.parent.relative_to(figures_dir)
    key = "." if str(rel) == "." else rel.as_posix()
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
    """Group PNG files under figures_dir by relative parent folder."""
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


def _add_image_page(pdf: PdfPages, image_path: Path) -> None:
    img = plt.imread(image_path)
    # Landscape letter-ish page; image scaled to fit while preserving aspect.
    fig = plt.figure(figsize=(11, 8.5))
    ax = fig.add_axes([0.04, 0.06, 0.92, 0.88])
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(image_path.stem.replace("_", " "), fontsize=10, pad=8)
    pdf.savefig(fig, dpi=150)
    plt.close(fig)


def compile_figures_pdf(
    figures_dir: Path,
    output_pdf: Path | None = None,
) -> Path:
    """
    Write a PDF under figures_dir with one title page per section, then its PNGs.

    Default output: figures_dir / output.pdf
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
                _add_image_page(pdf, png_path)

    return output_pdf
