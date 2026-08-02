"""Compile experiment figures/ PNGs into a single sectioned PDF report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from visualization.figure_captions import caption_for

SECTION_TITLES: dict[str, str] = {
    "trajectory": "Probe trajectory — Neuropixels insertion anatomy",
    "behavior": "Behavior — locomotor covariates and neural drivers",
    "features": "Behavior — locomotor covariates and neural drivers",
    "neural": "Neural activity — rates, rasters, and example units",
    "sorting": "Sorting vs ground truth",
    "decoder_comparison": "Decoding and manifolds",
    "decoder_comparison/ground_truth": "Decoder comparison — ground-truth spikes (legacy)",
    "decoder_comparison/sorted": "Decoder comparison — Neuropixels sorted (legacy)",
    "realtime_decoding": "Closed-loop realtime decoding",
    "realtime_decoding/comparison": "Realtime decoding — ground truth vs sorted comparison",
    "realtime_decoding/ground_truth": "Realtime decoding — ground-truth spikes (legacy)",
    "realtime_decoding/sorted": "Realtime decoding — Neuropixels sorted (legacy)",
    "deployment_decoder_selection": "Deployment decoder selection",
    "latency": "Causal-update latency budget",
    "temporal_decoding": "Temporal manifold decoding (W × L)",
}

# Preferred section order: simulation → decoding → manifolds → realtime → latency.
SECTION_ORDER = [
    "trajectory",
    "behavior",
    "neural",
    "sorting",
    "decoder_comparison",
    "decoder_comparison/ground_truth",
    "decoder_comparison/sorted",
    "realtime_decoding",
    "realtime_decoding/comparison",
    "realtime_decoding/ground_truth",
    "realtime_decoding/sorted",
    "deployment_decoder_selection",
    "latency",
    "temporal_decoding",
]

# Within decoder_comparison, keep manifold vs spikes onepager after the
# window-selection story and feature×W / decoder×W / decoder-family pages.
_DECODER_COMPARISON_TRAILING = (
    "fig_decoder_comparison_grid",
    "fig_window_selection_story",
    "fig_feature_x_window",
    "fig_decoder_x_window",
    "fig_continuous_decoders_feature_x_window",
    "fig_categorical_decoders_feature_x_window",
    "fig_manifold_vs_spikes_onepager",
)

# Keep closed-loop primary page before suite auxiliary targets.
_REALTIME_TRAILING = (
    "fig_closed_loop",
    "fig_closed_loop_suite",
)

# Aggregate benchmark overview before per-update distributions.
_LATENCY_TRAILING = (
    "fig_latency",
    "fig_latency_realtime",
)


def _section_key(figures_dir: Path, png_path: Path) -> str:
    """Map a PNG path to a section key, collapsing known nested run folders."""
    rel = png_path.parent.relative_to(figures_dir)
    if str(rel) == ".":
        # Legacy flat layout; keep as its own catch-all section.
        return "misc"
    key = rel.as_posix()
    # Prefer the longest known section prefix (e.g. realtime_decoding/sorted/...).
    known = sorted(SECTION_TITLES, key=len, reverse=True)
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
        # Skip legacy alias that duplicates fig_latent_geometry_position.
        if png.stem == "fig_latent_geometry":
            sibling = png.with_name("fig_latent_geometry_position.png")
            if sibling.exists():
                continue
        # Obsolete composite; panels already live in behavior/neural/sorting/trajectory.
        if png.stem == "simulation_report_summary":
            continue
        # Retired: superseded by fig_feature_x_window and fig_decoder_x_window.
        if png.stem in (
            "fig_manifold_decoding",
            "fig_manifold_decoder_window_threeway",
            "fig_deployable_decoder_x_window_heatmaps",
            "fig_deployable_winner_onepager",
            "deployable_winner_onepager",
        ):
            continue
        key = _section_key(figures_dir, png)
        # Behavior overview/covariates and neural drivers are one topic.
        if key == "features":
            key = "behavior"
        groups.setdefault(key, []).append(png)
    for key, paths in groups.items():
        if key == "decoder_comparison":
            trailing = {stem: i for i, stem in enumerate(_DECODER_COMPARISON_TRAILING)}

            def _decoder_sort_key(p: Path, _trailing=trailing) -> tuple:
                stem = p.stem
                if stem.startswith("fig_latent_geometry_"):
                    return (0.4, stem, p.name.lower())
                if stem.startswith("fig_decoder_geometry_"):
                    return (0.5, stem, p.name.lower())
                if stem in _trailing:
                    return (1, _trailing[stem], p.name.lower())
                return (0, 0, p.name.lower())

            paths.sort(key=_decoder_sort_key)
        elif key == "realtime_decoding":
            trailing = {stem: i for i, stem in enumerate(_REALTIME_TRAILING)}

            def _realtime_sort_key(p: Path, _trailing=trailing) -> tuple:
                stem = p.stem
                if stem in _trailing:
                    return (1, _trailing[stem], p.name.lower())
                return (0, 0, p.name.lower())

            paths.sort(key=_realtime_sort_key)
        elif key == "latency":
            trailing = {stem: i for i, stem in enumerate(_LATENCY_TRAILING)}

            def _latency_sort_key(p: Path, _trailing=trailing) -> tuple:
                stem = p.stem
                if stem in _trailing:
                    return (1, _trailing[stem], p.name.lower())
                return (0, 0, p.name.lower())

            paths.sort(key=_latency_sort_key)
        elif key == "behavior":
            # Spatial+covariates first, then neural drivers.
            order = {
                "fig_behavior_dynamics": 0,
                "fig_behavior_overview": 0,
                "fig_neural_drivers": 1,
            }
            paths.sort(key=lambda p: (order.get(p.stem, 50), p.name.lower()))
        elif key == "neural":
            def _neural_sort_key(p: Path) -> tuple:
                stem = p.stem
                # Spikes → tuning → feedforward → structure → mean-rate last.
                order = {
                    "fig_spikes_on_trajectory_by_class": 0,
                    "fig_population_tuning": 1,
                    "fig_circuit_feedforward": 2,
                    "fig_population_structure": 3,
                    "fig_spike_raster_summary": 3,
                    "fig_population_activity": 4,
                    "fig_cell_class_population": 4,
                    "fig_circuit_population": 4,
                }
                if stem in order:
                    return (order[stem], stem.lower())
                if stem.startswith("population_tuning_"):
                    return (1, stem.lower())
                if stem.startswith("fig_"):
                    return (5, stem.lower())
                return (6, stem.lower())

            paths.sort(key=_neural_sort_key)
        else:
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
    # No page title — panel letters live in the PNG; the caption is the only text label.
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")

    # Favor the figure: dense multi-panels need the page; caption stays readable.
    ax_img = fig.add_axes([0.04, 0.16, 0.92, 0.80])
    ax_img.imshow(img)
    ax_img.axis("off")

    ax_cap = fig.add_axes([0.06, 0.02, 0.88, 0.13])
    ax_cap.axis("off")
    ax_cap.text(
        0.5,
        1.0,
        caption_wrapped,
        ha="center",
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
