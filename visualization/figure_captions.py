"""Research-article-style captions for experiment figure PNGs.

Captions are resolved from the PNG stem (and optional path context such as
``ground_truth`` vs ``sorted``). Unknown stems fall back to a readable
description derived from the filename.
"""

from __future__ import annotations

import re
from pathlib import Path

# Exact stem → caption body (without "Figure N." prefix).
CAPTIONS: dict[str, str] = {
    # ---- Behavior / simulation ----
    "behavior_trajectory_xy": (
        "Open-field locomotor trajectory in the simulated arena. Green and red "
        "markers indicate start and end positions; dashed lines mark arena "
        "boundaries and any configured walls, barriers, or reward zones."
    ),
    "behavior_trajectory_time_colored": (
        "Same open-field trajectory with path segments colored by elapsed time, "
        "showing how the animal's spatial coverage evolves over the session."
    ),
    "behavior_speed_over_time": (
        "Instantaneous locomotor speed as a function of time during the "
        "simulated navigation session."
    ),
    "behavior_head_direction_over_time": (
        "Head-direction angle over time, used as a proprioceptive driver of "
        "head-direction-tuned hippocampal rate equations."
    ),
    "behavior_occupancy_heatmap": (
        "Spatial occupancy heatmap of the open-field session. Color indicates "
        "dwell time per spatial bin and reveals thigmotaxis or preferential "
        "exploration of arena regions."
    ),
    "behavior_speed_map": (
        "Average locomotor speed mapped onto arena coordinates. Values are "
        "computed within spatial bins occupied during the session."
    ),
    "behavior_features_over_time": (
        "Time series of behavioral feature channels (position-derived speed, "
        "acceleration, head direction, and related covariates) that drive the "
        "hippocampal rate model."
    ),
    "behavior_feature_distributions": (
        "Marginal distributions of behavioral feature channels across the "
        "session, summarizing the range of locomotor and postural conditions "
        "sampled by the simulated animal."
    ),
    "neural_driver_features_over_time": (
        "Population-mean neural driver features over time, stratified by cell "
        "class. Drivers include place, head-direction, speed, boundary, and "
        "oscillatory (theta/ripple) components used by the rate equations."
    ),
    "neural_driver_features_by_cell_class": (
        "Distribution of neural driver feature values by hippocampal cell "
        "class, illustrating class-specific dependence on spatial and "
        "proprioceptive covariates."
    ),
    "simulation_report_summary": (
        "Composite summary of the hippocampal Neuropixels simulation: "
        "trajectory and occupancy (A–B), speed (C), population rate (D), "
        "ground-truth spike raster subset (E), mean rate by cell class (F), "
        "and probe geometry through hippocampal regions (G)."
    ),
    # ---- Neural / sorting ----
    "rate_equation_population_rates_over_time": (
        "Population-averaged firing rates by cell class under the ground-truth "
        "rate equations, showing how CA1–CA3 pyramidal and DG granule "
        "populations modulate during navigation."
    ),
    "rate_equation_cell_class_rate_distributions": (
        "Distributions of mean firing rates across units within each cell "
        "class for the ground-truth rate-equation model."
    ),
    "ground_truth_spike_counts_by_cell_class": (
        "Total ground-truth spike counts aggregated by hippocampal cell class "
        "over the full session."
    ),
    "ground_truth_mean_rate_by_cell_class": (
        "Mean firing rate by cell class computed from ground-truth spike times."
    ),
    "ground_truth_rate_distribution_by_cell_class": (
        "Per-unit firing-rate distributions for each cell class under "
        "ground-truth spikes."
    ),
    "ground_truth_population_activity_by_cell_class": (
        "Binned population activity (spike counts) over time for each cell "
        "class, derived from ground-truth spike times."
    ),
    "ground_truth_spike_raster_all_cell_classes": (
        "Ground-truth spike raster spanning all recorded cell classes. Each "
        "row is a unit; ticks mark spike times across the session."
    ),
    "ground_truth_spike_raster_by_rate_equation": (
        "Ground-truth spike raster ordered by rate-equation cell class, "
        "highlighting differences in temporal structure across hippocampal "
        "populations."
    ),
    "sorted_vs_ground_truth_spike_counts": (
        "Comparison of total spike counts between ground-truth and "
        "Kilosort-like sorted extractions, stratified by cell class. "
        "Deviations quantify misses, contamination, and sorting losses."
    ),
    "sorted_vs_ground_truth_population_activity": (
        "Side-by-side population activity traces for sorted versus "
        "ground-truth spikes, illustrating how Neuropixels-like recording "
        "degradation and re-extraction alter population dynamics."
    ),
    "sorting_loss_by_cell_class": (
        "Relative sorting loss by cell class (sorted minus ground-truth spike "
        "counts, normalized). Negative values indicate apparent contamination "
        "or over-counting after sorting."
    ),
    "probe_region_geometry": (
        "Simulated Neuropixels 1.0 single-shank probe geometry through "
        "hippocampal anatomy. Shaded bands mark regional extents along the "
        "probe depth axis and associated channel mapping."
    ),
    "unit_depth_by_cell_class": (
        "Assigned probe depth of simulated units by cell class, reflecting "
        "anatomical placement relative to the Neuropixels shank."
    ),
    "unit_count_by_region_and_cell_class": (
        "Unit counts cross-tabulated by hippocampal region and cell class for "
        "the simulated population."
    ),
    # ---- Decoder comparison (source summary) ----
    "ground_truth_vs_sorted_best_position_error": (
        "Best offline position decoding error for ground-truth spikes versus "
        "Neuropixels-sorted spikes. Lower values indicate more accurate "
        "2-D position reconstruction."
    ),
    "ground_truth_vs_sorted_best_context_accuracy": (
        "Best offline spatial-context classification accuracy for ground-truth "
        "versus sorted spike sources, quantifying the impact of recording "
        "degradation on discrete context decoding."
    ),
    "manifold_feature_performance_by_target": (
        "Decoding performance across behavioral targets when population "
        "observations are represented as low-dimensional manifold features "
        "(e.g., global or region PCA) rather than raw spike counts."
    ),
    "counts_vs_manifold_decoding_summary": (
        "Summary comparison of spike-count features versus manifold "
        "embeddings for decoding behavioral targets. Bars report the "
        "selected metric for each feature mode."
    ),
    "region_manifold_decoding_by_target": (
        "Region-wise PCA manifold decoding performance by behavioral target, "
        "showing how region-restricted latent features support different "
        "latent variables."
    ),
    "layer_manifold_decoding_by_target": (
        "Layer-wise PCA manifold decoding performance by behavioral target."
    ),
    "manifold_explained_variance_by_region": (
        "Cumulative explained variance of region-level PCA manifolds fit to "
        "population spike-count features."
    ),
    "manifold_explained_variance_by_layer": (
        "Cumulative explained variance of layer-level PCA manifolds fit to "
        "population spike-count features."
    ),
    "global_pca_manifold_trajectory_position": (
        "Trajectory of the population state in the first two global PCA "
        "manifold coordinates. Axes are latent neural dimensions, not "
        "physical arena coordinates; color typically encodes time or "
        "decoded position context."
    ),
    "region_pca_manifold_trajectory_position": (
        "Population-state trajectory in the first two region-PCA manifold "
        "coordinates. Axes are latent neural dimensions rather than "
        "physical position."
    ),
    # ---- Per-source decoder comparison ----
    "best_decoder_by_target": (
        "Best offline decoder configuration selected for each behavioral "
        "target, summarizing peak validation performance across models, "
        "causal windows, and feature modes."
    ),
    "recommended_window_by_target": (
        "Recommended causal integration window (W) per behavioral target "
        "under the configured selection policy (e.g., shortest near-optimal "
        "versus best accuracy)."
    ),
    "position_error_vs_window": (
        "Offline 2-D position decoding error as a function of causal window "
        "length W. Each curve corresponds to a decoder model / feature mode "
        "evaluated during hyperparameter search."
    ),
    "head_direction_error_vs_window": (
        "Head-direction decoding error versus causal window length W across "
        "candidate decoder configurations."
    ),
    "speed_r2_vs_window": (
        "Coefficient of determination (R²) for speed decoding versus causal "
        "window length W."
    ),
    "acceleration_r2_vs_window": (
        "Coefficient of determination (R²) for acceleration decoding versus "
        "causal window length W."
    ),
    "distance_to_wall_r2_vs_window": (
        "Coefficient of determination (R²) for distance-to-wall decoding "
        "versus causal window length W."
    ),
    "spatial_context_accuracy_vs_window": (
        "Spatial-context classification accuracy versus causal window length "
        "W across candidate models and feature modes."
    ),
    "movement_state_accuracy_vs_window": (
        "Movement-state classification accuracy versus causal window length W."
    ),
    "wall_distance_bin_accuracy_vs_window": (
        "Wall-distance bin classification accuracy versus causal window "
        "length W."
    ),
    "best_position_true_vs_predicted": (
        "True versus predicted 2-D position for the best position decoder on "
        "held-out test data. Points near the identity line (or overlapping "
        "true/predicted trajectories) indicate accurate reconstruction."
    ),
    "best_head_direction_true_vs_predicted_over_time": (
        "True and predicted head direction over time for the best "
        "head-direction decoder on the test partition."
    ),
    "best_speed_true_vs_predicted_over_time": (
        "True and predicted locomotor speed over time for the best speed "
        "decoder on the test partition."
    ),
    "best_distance_to_wall_true_vs_predicted_over_time": (
        "True and predicted distance to the nearest wall over time for the "
        "best distance-to-wall decoder."
    ),
    "best_spatial_context_confusion_matrix": (
        "Confusion matrix for the best spatial-context classifier on held-out "
        "test samples. Diagonal entries indicate correct class assignments."
    ),
    "best_movement_state_confusion_matrix": (
        "Confusion matrix for the best movement-state classifier on held-out "
        "test samples."
    ),
    "best_wall_distance_bin_confusion_matrix": (
        "Confusion matrix for the best wall-distance-bin classifier on "
        "held-out test samples."
    ),
    # ---- Realtime / closed-loop ----
    "position_decoding_error_over_time": (
        "Causal real-time position decoding error over the session. Error is "
        "the Euclidean distance between true and decoded (x, y) at each "
        "update using only past spike history within the selected window W."
    ),
    "true_vs_decoded_position_time_colored": (
        "True and causally decoded positions in the arena, colored by time. "
        "Circles mark ground-truth location; crosses mark decoder estimates "
        "from online population spike counts."
    ),
    "true_vs_decoded_position": (
        "Side-by-side comparison of true versus causally decoded positions "
        "for ground-truth and sorted spike sources."
    ),
    "spatial_context_confusion_matrix": (
        "Confusion matrix of real-time spatial-context predictions against "
        "ground truth during closed-loop replay."
    ),
    "movement_state_confusion_matrix": (
        "Confusion matrix of real-time movement-state predictions against "
        "ground truth during closed-loop replay."
    ),
    "closed_loop_events_over_time": (
        "Closed-loop trigger events overlaid on real-time position decoding "
        "error. Green and red markers indicate correct and incorrect "
        "triggers under the configured closed-loop policy."
    ),
    "metrics_summary": (
        "Summary of real-time decoding metrics comparing ground-truth and "
        "Neuropixels-sorted spike sources across regression and "
        "classification targets."
    ),
    # ---- Temporal decoding ----
    "latency_distribution": (
        "Distribution of inference latency across temporal decoder model "
        "classes evaluated in the W×L temporal-manifold comparison."
    ),
    "timing_overview": (
        "Overview of timing roles in the temporal decoding pipeline, "
        "including update interval and causal window / sequence-length "
        "constraints."
    ),
}

# Regex patterns for parameterized stems → caption template with {match} groups.
_PATTERN_CAPTIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^example_units_(.+)$"),
        (
            "Example single-unit activity for {0} cells. Panels show "
            "driver/rate traces, spike trains, and instantaneous firing rates "
            "for representative units of this class."
        ),
    ),
    (
        re.compile(r"^ground_truth_spike_raster_(.+)$"),
        (
            "Ground-truth spike raster restricted to {0} units. Each row is a "
            "unit; ticks mark spike times across the session."
        ),
    ),
    (
        re.compile(r"^heatmap_(.+)_(.+)_(.+)$"),
        (
            "Validation performance heatmap for target {0} under "
            "representation {1} and model class {2}, sweeping temporal "
            "hyperparameters (window W and sequence length L where applicable)."
        ),
    ),
    (
        re.compile(r"^model_comparison_(.+)$"),
        (
            "Best validation score by temporal model class for target {0} in "
            "the W×L comparison."
        ),
    ),
    (
        re.compile(r"^(.+)_trajectory_position$"),
        (
            "Latent neural-state trajectory for manifold feature mode {0}. "
            "Axes are low-dimensional embedding coordinates, not physical "
            "arena position."
        ),
    ),
]

_SOURCE_NOTES = {
    "ground_truth": (
        " Results shown for ground-truth spike times (oracle spike source)."
    ),
    "sorted": (
        " Results shown for Neuropixels-like recording with Kilosort-like "
        "spike re-extraction."
    ),
}

_CELL_CLASS_LABELS = {
    "CA1_pyr": "CA1 pyramidal",
    "CA2_pyr": "CA2 pyramidal",
    "CA3_pyr": "CA3 pyramidal",
    "DG_granule": "DG granule",
    "all_cell_classes": "all cell classes",
    "by_rate_equation": "rate-equation cell classes",
}


def _humanize_stem(stem: str) -> str:
    return stem.replace("_", " ").strip()


def _humanize_token(token: str) -> str:
    if token in _CELL_CLASS_LABELS:
        return _CELL_CLASS_LABELS[token]
    return token.replace("_", " ").strip()


def _source_note(image_path: Path, figures_dir: Path | None) -> str:
    parts = image_path.parts
    for key, note in _SOURCE_NOTES.items():
        if key in parts:
            return note
    if figures_dir is not None:
        try:
            rel_parts = image_path.relative_to(figures_dir).parts
        except ValueError:
            rel_parts = ()
        for key, note in _SOURCE_NOTES.items():
            if key in rel_parts:
                return note
    return ""


def _caption_body(stem: str) -> str:
    if stem in CAPTIONS:
        return CAPTIONS[stem]

    for pattern, template in _PATTERN_CAPTIONS:
        match = pattern.match(stem)
        if match:
            groups = tuple(_humanize_token(g) for g in match.groups())
            try:
                return template.format(*groups)
            except (IndexError, KeyError):
                return template

    return (
        f"{_humanize_stem(stem).capitalize()}. "
        "Visualization generated from the experiment outputs for this run."
    )


def caption_for(
    image_path: Path,
    *,
    figure_number: int,
    figures_dir: Path | None = None,
) -> str:
    """
    Return a numbered research-style caption for ``image_path``.

    Parameters
    ----------
    image_path
        Path to the PNG being placed in the PDF.
    figure_number
        1-based figure index in the compiled report.
    figures_dir
        Optional figures root used to infer ground-truth vs sorted context.
    """
    image_path = Path(image_path)
    body = _caption_body(image_path.stem)
    note = _source_note(image_path, figures_dir)
    # Avoid duplicating source notes already implied by caption text.
    if note and "ground-truth" not in body.lower() and "sorted" not in body.lower():
        # Still add note when path context is informative (decoder/realtime).
        rel = ""
        if figures_dir is not None:
            try:
                rel = image_path.relative_to(figures_dir).as_posix()
            except ValueError:
                rel = ""
        if any(
            token in rel
            for token in (
                "decoder_comparison/",
                "realtime_decoding/",
                "temporal_decoding/",
            )
        ):
            body = body.rstrip(".") + "." + note
    return f"Figure {figure_number}. {body}"


def title_for(image_path: Path) -> str:
    """Short page title derived from the PNG stem."""
    return _humanize_stem(Path(image_path).stem).title()
