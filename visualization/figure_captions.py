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
    "fig_behavior_overview": (
        "Open-field locomotor behavior. "
        "(A) Trajectory with start (green) and end (red). "
        "(B) Same trajectory colored by elapsed time. "
        "(C) Spatial occupancy (dwell time per bin). "
        "(D) Mean locomotor speed mapped onto arena coordinates."
    ),
    "fig_behavior_dynamics": (
        "Locomotor and postural dynamics over the session. "
        "(A) Instantaneous speed. "
        "(B) Head-direction angle. "
        "(C) Distance to the nearest wall. "
        "(D) Z-scored distributions of speed, head direction, wall distance, and acceleration."
    ),
    "fig_behavior_features": (
        "Behavioral covariates that drive RatInABox populations and decoding "
        "targets over the session, including position, speed, heading, wall "
        "distance, acceleration, and oscillatory channels when present."
    ),
    "fig_neural_drivers": (
        "Neural driver features by cell class. "
        "(A–C) Population-mean place, head-direction, and speed drive over time. "
        "(D) Session-mean absolute driver strength (cell class × driver)."
    ),
    # Legacy single-panel behavior stems (older PDFs)
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
        "Head-direction angle over time, used as a proprioceptive input to "
        "RatInABox head-direction and place populations."
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
        "RatInABox neural populations and hippocampal overlays."
    ),
    "behavior_feature_distributions": (
        "Marginal distributions of behavioral feature channels across the "
        "session, summarizing the range of locomotor and postural conditions "
        "sampled by the simulated animal."
    ),
    "neural_driver_features_over_time": (
        "Population-mean neural driver features over time, stratified by cell "
        "class. Drivers include place, head-direction, speed, boundary, and "
        "oscillatory (theta/ripple) components used by RatInABox overlays."
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
    "fig_circuit_population": (
        "Hippocampal circuit population activity. "
        "(A) Mean rates for circuit nodes overlaid (MEC, DG, CA3, CA2, CA1, INT, SUB). "
        "(B) Stacked traces emphasizing trisynaptic / entorhinal flow. "
        "(C) Session-mean rate by circuit node. "
        "(D) Population size versus mean rate by node."
    ),
    "fig_cell_class_population": (
        "Cell-class population activity. "
        "(A) Mean rate traces by cell class. "
        "(B) Mean rate traces by anatomical region. "
        "(C) Violin distributions of per-unit mean rates by cell class."
    ),
    "fig_population_structure": (
        "Population structure. "
        "(A) Unit-count crosstab of region × cell class. "
        "(B) Mean-rate heatmap of region × cell class. "
        "(C) Ground-truth rate heatmap with units ordered by circuit node."
    ),
    "fig_spike_raster_summary": (
        "Spike raster and sorting yield. "
        "(A) Compressed ground-truth raster ordered by circuit node. "
        "(B) Ground-truth versus sorted total spike counts by cell class."
    ),
    # ---- Decoding / manifolds / realtime ----
    "fig_decoding_performance": (
        "Causal decoding performance on sorted spikes. "
        "(A) Best primary metric versus causal window W for continuous targets. "
        "(B) Balanced accuracy versus W for categorical targets. "
        "(C) Best decoder and metric value by target. "
        "(D) Selected / recommended realtime window by target."
    ),
    "fig_manifold_decoding": (
        "Manifold versus spike-count decoding. "
        "(A) Feature-mode × target heatmap of normalized best scores "
        "(counts, PCA, region PCA, and Isomap when present). "
        "(B) Signed manifold − counts performance difference by target. "
        "(C) Best region-PCA score by target. "
        "(D) Mean explained variance by anatomical group."
    ),
    "fig_latent_geometry": (
        "Latent neural geometry across embedding modes (sorted / deployable). "
        "Each panel is one feature mode at the causal window (and k / n_neighbors) "
        "that best decoded the colored behavioral variable on held-out data. "
        "Position pages encode arena (x, y) as hue and brightness; other pages "
        "color by the named continuous or categorical variable. "
        "See also fig_latent_geometry_<feature> for the full suite."
    ),
    "fig_isomap_diagnostics": (
        "Isomap geometry diagnostics. "
        "(A) Trustworthiness versus n_neighbors. "
        "(B) Graph connectivity / largest-component fraction. "
        "(C) Residual variance versus latent dimension. "
        "(D) Geodesic-distance correlation and/or knn-overlap / continuity. "
        "Empty panels indicate that global_isomap was not included in the comparison."
    ),
    "fig_isomap_story": (
        "Isomap decoding and distillation. "
        "(A) Normalized best scores for counts / PCA / Isomap by target. "
        "(B) Decoder family × representation matrix. "
        "(C) Classic offline Isomap versus distilled encoder latency (50 ms budget). "
        "(D) Distilled versus classic accuracy when both feature modes exist. "
        "Classic Isomap is offline-only."
    ),
    "fig_closed_loop": (
        "Closed-loop realtime decoding. "
        "(A) True versus decoded position. "
        "(B) Position error over time with closed-loop trigger markers. "
        "(C) Spatial-context confusion matrix (row-normalized). "
        "(D) Trigger reliability (correct versus incorrect counts)."
    ),
    "fig_deployment": (
        "Deployment decoder selection on sorted spikes. "
        "(A) Winner summary: decoder, feature mode, and selected window per target. "
        "(B–C) Decoder × causal-window heatmaps for key deployment targets."
    ),
    "fig_latency": (
        "Causal-update latency budget. "
        "(A) Feature-transform latencies. "
        "(B) Realtime decode-stage latencies. "
        "(C) Classic Isomap teacher versus distilled encoder. "
        "(D) Top overall latency contributors. "
        "Dashed line marks the update budget (typically 50 ms); green bars are "
        "realtime-compatible."
    ),
    "fig_temporal_wl": (
        "Temporal manifold decoding. "
        "Heatmaps of validation metric over integration window W versus latent "
        "history frames L for representative target / representation / model slices."
    ),
    # Legacy stems kept for older figure directories / PDFs
    "population_activity_by_cell_class": (
        "Mean population firing rate over time for each hippocampal / afferent "
        "cell class (legacy single-panel figure)."
    ),
    "population_activity_by_region": (
        "Mean population firing rate over time stratified by anatomical region "
        "(legacy single-panel figure)."
    ),
    "population_activity_by_rate_model": (
        "Mean population firing rate for each RatInABox population group "
        "(legacy single-panel figure)."
    ),
    "circuit_population_activity": (
        "Stacked circuit-node population rates (legacy single-panel figure)."
    ),
    "circuit_population_activity_overlay": (
        "Circuit-node mean rates overlaid (legacy single-panel figure)."
    ),
    "population_rate_heatmap": (
        "Per-unit ground-truth rate heatmap (legacy single-panel figure)."
    ),
    "mean_rate_by_cell_class": (
        "Session-mean firing rate by cell class (legacy single-panel figure)."
    ),
    "mean_rate_by_circuit_node": (
        "Session-mean firing rate by circuit node (legacy single-panel figure)."
    ),
    "population_rates_over_time": (
        "Population-averaged firing rates by cell class (legacy)."
    ),
    "cell_class_rate_distributions": (
        "Mean firing-rate distributions by cell class (legacy)."
    ),
    "rate_equation_population_rates_over_time": (
        "Legacy population-rate figure stem."
    ),
    "rate_equation_cell_class_rate_distributions": (
        "Legacy cell-class rate distribution figure stem."
    ),
    "ground_truth_spike_counts_by_cell_class": (
        "Total ground-truth spike counts by cell class (legacy)."
    ),
    "ground_truth_mean_rate_by_cell_class": (
        "Mean firing rate by cell class (legacy)."
    ),
    "ground_truth_rate_distribution_by_cell_class": (
        "Per-unit rate distributions by cell class (legacy)."
    ),
    "ground_truth_population_activity_by_cell_class": (
        "Population activity by cell class (legacy)."
    ),
    "ground_truth_spike_raster_all_cell_classes": (
        "Ground-truth spike raster by cell class (legacy)."
    ),
    "ground_truth_spike_raster_by_rate_model": (
        "Ground-truth spike raster by rate model (legacy)."
    ),
    "ground_truth_spike_raster_by_rate_equation": (
        "Ground-truth spike raster by rate model (legacy)."
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
        "Signed performance difference between the best manifold feature mode "
        "and the best spike-count decoder for each target (positive = manifold "
        "helps). Companion detail is in manifold_vs_spikes_onepager."
    ),
    "manifold_vs_spikes_onepager": (
        "One-pager comparing spike-count features versus manifold embeddings for "
        "sorted / Neuropixels decoding. Top table lists best counts setup, best "
        "manifold setup, ±5% verdict, and the deployable registry selection per "
        "target. Lower panels are decoder × feature heatmaps collapsed over "
        "windows (best W annotated; greener is better). A dashed line separates "
        "spike features (counts/rates) from manifold modes; gold outline marks "
        "the deployable selection; hatching marks offline-only features."
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
    "isomap_vs_pca_decoder_position": (
        "Position decoding error for counts, global PCA, and global Isomap "
        "features, broken down by decoder. Lower is better. Panels are "
        "stratified by spike source so ground-truth and sorted results are "
        "not mixed."
    ),
    "isomap_vs_pca_decoder_position_sorted": (
        "Position decoding error (sorted spikes) for counts vs PCA vs Isomap "
        "across decoders. Lower mean position error is better."
    ),
    "isomap_position_best_by_representation": (
        "Best position decoder for each representation (counts, PCA, Isomap) "
        "on sorted spikes. Highlights whether nonlinear Isomap geometry "
        "improves allocentric position decoding relative to linear PCA and "
        "raw counts."
    ),
    "isomap_position_best_by_representation_sorted": (
        "Best position decoder per representation on sorted spikes. "
        "Isomap should appear as the lowest bar when nonlinear geometry "
        "improves held-out position decoding."
    ),
    "isomap_vs_pca_best_by_target": (
        "Best counts / PCA / Isomap score for each behavioral target "
        "(errors are negated so higher bars are better). Stars mark the "
        "winning representation per target."
    ),
    "isomap_vs_pca_best_by_target_sorted": (
        "Best counts / PCA / Isomap score by target on sorted spikes. "
        "Stars mark the winning representation; errors are negated so "
        "higher is better."
    ),
    "isomap_trustworthiness_vs_neighbors_sorted": (
        "Isomap neighborhood trustworthiness as a function of n_neighbors "
        "on sorted-spike training embeddings."
    ),
    "isomap_connectivity_vs_neighbors_sorted": (
        "Fraction of training samples in the largest Isomap neighbor-graph "
        "component versus n_neighbors."
    ),
    "isomap_residual_variance_vs_dim_sorted": (
        "Isomap residual variance (1 − R² between geodesic and embedding "
        "distances) versus latent dimension."
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
    # ---- Latency profiling ----
    "latency_everything": (
        "Causal update latency for every measured stage: feature transforms "
        "(counts / PCA / Isomap / distilled Isomap), closed-loop decode stages "
        "(spike binning, feature map, each decoder head, trigger policy), and "
        "classic Isomap teacher versus parametric distilled student. The dashed "
        "line marks the 20 Hz update budget (50 ms). Green bars are "
        "realtime-compatible; red bars are offline-only."
    ),
    "feature_transform_latency": (
        "Per-update wall-clock cost of each neural feature front-end applied to "
        "a single causal spike-count vector. Distilled Isomap is the "
        "streaming-friendly parametric approximation of classic Isomap."
    ),
    "isomap_teacher_vs_distilled_latency": (
        "Direct latency comparison of classic offline Isomap (teacher) versus "
        "the parametric distilled encoder used for realtime deployment."
    ),
    "realtime_stage_latency": (
        "Breakdown of closed-loop realtime latency by stage on sorted spikes: "
        "spike binning, feature transform, position / speed / context / "
        "movement decoders, optional primary override, and closed-loop policy. "
        "The dashed line is the 50 ms update budget."
    ),
    "deployment_model_summary": (
        "Text summary of the deployable realtime decoder registry: per-target "
        "selected decoder, causal window, and feature mode from sorted spikes only."
    ),
    "deployable_winner_onepager": (
        "One-pager for sorted / Neuropixels deployable selection. Top table lists "
        "the registry winner per target (decoder, feature, W, metric). Lower panels "
        "are decoder × feature heatmaps collapsed over windows (best W annotated in "
        "each cell; greener is better). Gold outline marks the selected cell; "
        "hatching marks offline-only features such as classic Isomap."
    ),
    "deployable_decoder_x_window_heatmaps": (
        "Deployable decoder × causal-window heatmaps on sorted spikes. Each cell "
        "shows the best realtime-compatible feature mode at that window and the "
        "metric value; gold outline marks the selected (decoder, W) pair."
    ),
}

# Regex patterns for parameterized stems → caption template with {match} groups.
_PATTERN_CAPTIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^fig_latent_geometry_(.+)$"),
        (
            "Latent neural geometry colored by {0} (sorted / deployable spikes). "
            "Each panel is one embedding mode (counts, PCA variants, Isomap, …) "
            "at the causal window and hyperparameters that best decoded this "
            "behavioral variable on held-out data. Axes are the leading two "
            "latent coordinates (z₁, z₂). For position pages, arena coordinates "
            "are encoded as hue (x) and brightness (y)."
        ),
    ),
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
    "CA1_int": "CA1 interneuron",
    "CA2_pyr": "CA2 pyramidal",
    "CA3_pyr": "CA3 pyramidal",
    "DG_granule": "DG granule",
    "Sub_bvc": "subiculum BVC",
    "MEC_grid": "MEC grid",
    "MEC_hd": "MEC head direction",
    "MEC_speed": "MEC speed",
    "all_cell_classes": "all cell classes",
    "by_rate_model": "rate-model cell classes",
    "by_rate_equation": "rate-model cell classes",
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
