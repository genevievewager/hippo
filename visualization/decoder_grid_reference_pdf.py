"""Generate a transportable PDF reference for the decoder comparison grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from realtime.adaptive_windows import COARSE_DECODE_WINDOWS, WINDOW_CANDIDATE_POOL
from realtime.decoder_comparison import DEFAULT_DECODE_WINDOWS
from realtime.decoder_models import (
    TARGET_FAMILY,
    categorical_model_names,
    continuous_model_names,
)
from realtime.manifold_features import (
    MANIFOLDS_FEATURE_MODES,
    MANIFOLDS_ISOMAP_N_NEIGHBORS,
    MANIFOLDS_N_COMPONENTS,
    OFFLINE_ONLY_FEATURE_MODES,
)
from realtime.search_space import compose_feature_mode, expand_fe_jobs


PAGE_SIZE = (8.5, 11.0)  # US Letter, portrait
MARGIN = 0.72
BODY_FONTSIZE = 9
HEADING_FONTSIZE = 13
TITLE_FONTSIZE = 20
LINES_PER_PAGE = 52
CHARS_PER_LINE = 92


def _wrap_block(text: str, width: int = CHARS_PER_LINE) -> list[str]:
    out: list[str] = []
    for raw in text.strip().splitlines():
        line = raw.rstrip()
        if not line:
            out.append("")
            continue
        if line.startswith("  ") or line.startswith("|") or line.startswith("-"):
            out.extend(wrap(line, width=width, subsequent_indent=2) or [""])
        else:
            out.extend(wrap(line, width=width) or [""])
    return out


def _fe_job_lines() -> list[str]:
    jobs = expand_fe_jobs(
        feature_modes=MANIFOLDS_FEATURE_MODES,
        manifold_n_components=MANIFOLDS_N_COMPONENTS,
        isomap_n_neighbors=MANIFOLDS_ISOMAP_N_NEIGHBORS,
    )
    lines = [f"Encoder jobs per window W: {len(jobs)}", ""]
    for feature_type, embedding, k, nn in jobs:
        mode = compose_feature_mode(feature_type, embedding)
        if k is None and nn is None:
            lines.append(f"  • {mode}")
        elif nn is None:
            lines.append(f"  • {mode}  (k={k})")
        else:
            lines.append(f"  • {mode}  (k={k}, nn={nn})")
    return lines


def _decoder_lines() -> list[str]:
    lines = ["Decoders tried per target (quick / manifolds profile):", ""]
    for target in sorted(TARGET_FAMILY):
        family = TARGET_FAMILY[target]
        if family == "continuous":
            names = continuous_model_names("quick", target)
        else:
            names = categorical_model_names("quick", target)
        lines.append(f"  {target} ({family}):")
        lines.append(f"    {', '.join(names)}")
    return lines


def _winner_lines(experiment_dir: Path) -> list[str]:
    path = experiment_dir / "models" / "best_realtime_decoders.json"
    if not path.exists():
        return ["(No best_realtime_decoders.json found for this experiment.)"]
    data = json.loads(path.read_text())
    lines = [
        f"Experiment: {data.get('run_id', experiment_dir.name)}",
        f"Selection policy: {data.get('selection_policy', 'shortest_near_optimal')}",
        f"Spike source: {data.get('spike_source', 'sorted')}",
        "",
        "Target          W (s)   Feature mode              Decoder",
        "--------------  ------  ------------------------  -------------------------------",
    ]
    for target in sorted((data.get("targets") or {}).keys()):
        row = data["targets"][target]
        w = row.get("selected_causal_window_s", "")
        feat = row.get("selected_feature_mode", "")
        dec = row.get("selected_decoder", "")
        k = row.get("manifold_n_components")
        if k is not None:
            feat = f"{feat} (k={int(k)})"
        lines.append(f"{target:<14}  {w:<6}  {feat:<24}  {dec}")
    return lines


def build_reference_lines(experiment_dir: Path) -> list[str]:
    """Return flat line list for the full reference document."""
    coarse = ", ".join(f"{w:g}" for w in COARSE_DECODE_WINDOWS)
    standard = ", ".join(f"{w:g}" for w in WINDOW_CANDIDATE_POOL)
    full = ", ".join(f"{w:g}" for w in DEFAULT_DECODE_WINDOWS)
    offline = ", ".join(sorted(OFFLINE_ONLY_FEATURE_MODES))
    modes = ", ".join(MANIFOLDS_FEATURE_MODES)
    k_vals = ", ".join(str(k) for k in MANIFOLDS_N_COMPONENTS)
    nn_vals = ", ".join(str(n) for n in MANIFOLDS_ISOMAP_N_NEIGHBORS)

    sections: list[str] = [
        "HIPPO — DECODER COMPARISON GRID REFERENCE",
        "",
        "Project summary",
        "---------------",
        "Hippo compares causal spike representations → neural manifold encoders → decoders",
        "to predict rodent behaviors from sorted Neuropixels-like spikes. A bounded grid search",
        "scores ~1,500 configurations (this run), then picks one deployable model per behavior.",
        "",
        "Pipeline (one CSV row = one configuration)",
        "  spike counts x_t(W)  →  encoder E  →  latent z_t  →  decoder D  →  ŷ_t  →  metric",
        "",
        "Key concepts",
        "  • Causal: features use spikes in [t−W, t) only; no future leakage",
        "  • Frozen: encoder E and decoder D fit on train; test/realtime is transform/predict only",
        "  • Single W: same causal window for encoder and decoder in each configuration",
        "  • Bounded grid: nested loops, not random Monte Carlo trials",
        "",
        "CAUSAL INTEGRATION WINDOWS (W)",
        "--------------------------------",
        f"Used in {experiment_dir.name} (profile: manifolds):  {coarse} seconds",
        f"Standard refine pool:                               {standard} seconds",
        f"Full grid (full profile):                           {full} seconds",
        "",
        "Selection rule: shortest_near_optimal — shortest W within 5% of best held-out score.",
        "",
        "MANIFOLDS / FEATURE REPRESENTATIONS",
        "-----------------------------------",
        "Two-stage representation: feature type F → embedding/manifold E → latent z_t",
        "",
        f"Feature modes in manifolds profile ({len(MANIFOLDS_FEATURE_MODES)} names, "
        f"{len(expand_fe_jobs(feature_modes=MANIFOLDS_FEATURE_MODES, manifold_n_components=MANIFOLDS_N_COMPONENTS, isomap_n_neighbors=MANIFOLDS_ISOMAP_N_NEIGHBORS))} encoder jobs per W):",
        f"  {modes}",
        "",
        "  counts                  Raw causal spike counts per unit (no embedding)",
        "  global_pca              PCA on all units together",
        "  region_pca              Separate PCA per brain region",
        "  layer_pca               Separate PCA per cortical layer",
        f"  global_isomap           Classic Isomap (offline only; not auto-deployed)",
        "  global_isomap_distilled Isomap teacher → MLP student (realtime-eligible if fast enough)",
        "",
        f"Manifold dimensions k searched:     {{{k_vals}}}",
        f"Isomap neighbors nn searched:       {{{nn_vals}}}",
        f"Offline-only feature modes:           {offline}",
        "",
        "Full manifold zoo in codebase (not all run in every experiment):",
        "  counts, rates, global_pca, region_pca, layer_pca, cell_type_pca, rate_model_pca,",
        "  pls, bayesian_place_tuning, global_isomap, global_isomap_distilled",
        "",
    ]
    sections.extend(_fe_job_lines())
    sections.extend([
        "",
        "DECODERS",
        "--------",
        "Decoders map latent z_t to behavioral predictions ŷ_t.",
        "",
        "Used in this experiment (7 unique decoder families):",
        "  ridge, pca_ridge, random_forest_regressor, bayesian_place_decoder,",
        "  logistic_regression, random_forest_classifier, bayesian_place_decoder_derived_context",
        "",
    ])
    sections.extend(_decoder_lines())
    sections.extend([
        "",
        "Full decoder zoo (full profile — not used in manifolds quick run):",
        "  Continuous: ridge, elastic_net, pca_ridge, pls_regression, random_forest_regressor,",
        "    hist_gradient_boosting_regressor, knn_regressor, rbf_svr, mlp_regressor,",
        "    bayesian_place_decoder, bayesian_place_decoder_smoothed, state_space_or_kalman_optional",
        "  Categorical: logistic_regression, linear_svm_classifier, random_forest_classifier,",
        "    hist_gradient_boosting_classifier, knn_classifier, rbf_svc, mlp_classifier,",
        "    bayesian_place_decoder_derived_context",
        "",
        "BEHAVIORAL TARGETS (8)",
        "----------------------",
        "  position            continuous (x, y)     2D arena location",
        "  speed               continuous              running speed",
        "  acceleration        continuous              speed change",
        "  head_direction      continuous              circular angle (sin/cos)",
        "  distance_to_wall    continuous              cm to nearest wall",
        "  spatial_context     categorical             corner / edge / center",
        "  movement_state      categorical             still / walk / run",
        "  wall_distance_bin   categorical             near_wall / middle / center",
        "",
        "GRID SEARCH STRUCTURE",
        "---------------------",
        "FOR each causal window W:",
        "  FOR each (feature, manifold) encoder job:",
        "    Fit encoder E once on train spikes → z_t",
        "    FOR each behavioral target T (8):",
        "      FOR each decoder D allowed for T:",
        "        Fit D on train z → score held-out test → write 1 CSV row",
        "",
        "Per target: pick best primary metric, then shortest_near_optimal W",
        "Output: models/best_realtime_decoders.json",
        "",
        f"Scale ({experiment_dir.name}): 4 windows × 15 encoder jobs × ~25 decoder-target pairs ≈ 1,500 rows",
        "Spike source for deployment: sorted (Neuropixels/Kilosort-like); ground-truth spikes are oracle only.",
        "",
        "DEPLOYABLE WINNERS",
        "------------------",
    ])
    sections.extend(_winner_lines(experiment_dir))
    sections.extend([
        "",
        "Suggested figures for audience presentations",
        "  1. Nested loop: W → encoder job → target × decoder → CSV row",
        "  2. Pipeline strip: x_t(W) → E → z_t → D → ŷ → metric",
        "  3. Heatmaps: feature×W and decoder×W (best collapsed per cell)",
        "  4. Sankey: 1,500 configs → 8 deployable winners",
        "",
        f"Generated from hippo experiment directory: {experiment_dir}",
    ])

    wrapped: list[str] = []
    for block in sections:
        if block.startswith("HIPPO") or block.isupper() and block.endswith("---") is False and " " not in block.strip():
            wrapped.extend(_wrap_block(block))
        else:
            wrapped.extend(_wrap_block(block))
    return wrapped


def _render_pages(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    chunk: list[str] = []
    for line in lines:
        chunk.append(line)
        if len(chunk) >= LINES_PER_PAGE:
            pages.append(chunk)
            chunk = []
    if chunk:
        pages.append(chunk)
    return pages or [[]]


def _draw_page(pdf: PdfPages, page_lines: list[str], *, page_num: int, total: int) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    if page_num == 1:
        ax.text(
            MARGIN / PAGE_SIZE[0], 1 - MARGIN / PAGE_SIZE[1],
            "Hippo Decoder Comparison Grid",
            fontsize=TITLE_FONTSIZE, fontweight="bold", va="top", ha="left",
        )
        ax.text(
            MARGIN / PAGE_SIZE[0], 1 - (MARGIN + 28) / PAGE_SIZE[1],
            "Reference for decoders, manifolds, windows, and grid search",
            fontsize=11, color="#444444", va="top", ha="left",
        )
        y_start = 1 - (MARGIN + 52) / PAGE_SIZE[1]
    else:
        y_start = 1 - MARGIN / PAGE_SIZE[1]

    body = "\n".join(page_lines)
    ax.text(
        MARGIN / PAGE_SIZE[0], y_start,
        body,
        fontsize=BODY_FONTSIZE, va="top", ha="left",
        family="sans-serif",
        linespacing=1.25,
    )
    ax.text(
        0.5, MARGIN / PAGE_SIZE[1] / 2,
        f"Page {page_num} of {total}",
        ha="center", va="center", fontsize=8, color="#666666",
    )
    pdf.savefig(fig)
    plt.close(fig)


def write_decoder_grid_reference_pdf(
    experiment_dir: Path,
    output_pdf: Path | None = None,
) -> Path:
    """Write a multi-page PDF reference document for an experiment."""
    experiment_dir = Path(experiment_dir)
    if output_pdf is None:
        output_pdf = experiment_dir / "decoder_comparison_grid_reference.pdf"
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    lines = build_reference_lines(experiment_dir)
    pages = _render_pages(lines)
    with PdfPages(output_pdf) as pdf:
        total = len(pages)
        for i, page_lines in enumerate(pages, start=1):
            _draw_page(pdf, page_lines, page_num=i, total=total)
    return output_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("outputs/ratinabox_001"),
        help="Experiment output directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path (default: <experiment-dir>/decoder_comparison_grid_reference.pdf)",
    )
    args = parser.parse_args()
    path = write_decoder_grid_reference_pdf(args.experiment_dir, args.output)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
