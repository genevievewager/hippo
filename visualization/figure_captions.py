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
        "Open-field locomotor behavior (absorbed into fig_behavior_dynamics)."
    ),
    "fig_behavior_dynamics": (
        "Open-field locomotor behavior and session covariates. "
        "(A) Trajectory colored by elapsed time, with start (green) and end (red). "
        "(B) Mean locomotor speed mapped onto arena coordinates. "
        "(C) Spatial occupancy (dwell time per bin). "
        "(D–I) Behavioral covariates over time: x, y, speed, head direction, "
        "wall distance, and acceleration (when present)."
    ),
    "fig_behavior_features": (
        "Behavioral covariates (absorbed into fig_behavior_dynamics)."
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
        "Legacy composite report; panels absorbed into fig_behavior_dynamics, "
        "fig_population_activity / fig_population_structure, "
        "fig_sorting_summary, and fig_probe_trajectory."
    ),
    # ---- Probe trajectory / insertion anatomy ----
    "probe_trajectory_regions": (
        "Neuropixels probe path through brain regions by depth. "
        "Colored bands mark crossed structures; channel markers indicate "
        "site assignment along the shank. Visual cortex may appear but is "
        "excluded from hippocampal decoding by default."
    ),
    "probe_areas_nte_style": (
        "Neuropixels Trajectory Explorer–style probe-areas strip summarizing "
        "region identity along the recorded shank."
    ),
    "probe_trajectory_3d": (
        "Approximate 3D probe trajectory in coordinate space (not a full "
        "Allen CCF mesh unless NTE endpoints are imported)."
    ),
    "channel_region_map": (
        "Channel index mapped onto anatomical regions along the probe depth, "
        "showing which sites fall in each crossed band."
    ),
    "unit_count_by_region": (
        "Number of simulated units assigned to each anatomical region crossed "
        "by the active probe trajectory."
    ),
    "unit_count_by_cell_type": (
        "Number of simulated units by RatInABox cell class after trajectory-"
        "informed capture."
    ),
    "unit_count_by_imported_region": (
        "Number of simulated units by imported anatomy region (legacy alias)."
    ),
    # ---- Neural / sorting ----
    "fig_circuit_feedforward": (
        "Trisynaptic / entorhinal feedforward circuit. "
        "Directed graph of circuit nodes used as Stage-C population drives. "
        "Each node shows name, captured unit count and session-mean rate, "
        "pooled cell classes, and circuit role. "
        "INT is one inhibitory node aggregating all local INT_* pools; "
        "dashed fan-out edges carry region-specific inhibition weights from "
        "neural_backend_metadata (excitatory solid). "
        "Only edges with nonzero weights are drawn. "
        "Circuit node ≠ region or cell class: classes sharing a node are "
        "pooled for feedforward, not claimed to share receptive-field type."
    ),
    "fig_population_activity": (
        "Cell-class, circuit-node, and regional population activity. "
        "(A) Mean rate traces by cell class (per-unit mean). "
        "(B) Mean rate traces by circuit node (per-unit mean; "
        "INT_* → local inhibitory pools, Subiculum → SUB). "
        "(C) Mean rate traces by anatomical region (per-unit mean; "
        "all cell classes in that region pooled). "
        "Unit counts are shown in fig_probe_trajectory."
    ),
    "fig_circuit_population": (
        "Legacy stem; content now lives in fig_population_activity panel B."
    ),
    "fig_cell_class_population": (
        "Legacy stem; content now lives in fig_population_activity panel A."
    ),
    "fig_population_structure": (
        "Ground-truth spike raster by circuit node. "
        "Units are ordered MEC → DG → CA3 → CA2 → CA1 → INT_* → SUB "
        "(present nodes only; local INT satellites follow their home), "
        "then by cell class and descending mean rate. "
        "Crimson lines mark node boundaries; left labels name each node. "
        "When many units are present, each node is downsampled to keep the "
        "raster readable. "
        "Region × cell-class unit counts live in fig_probe_trajectory panel D. "
        "Ground-truth versus sorted spike counts live in fig_sorting_summary."
    ),
    "fig_spike_raster_summary": (
        "Legacy stem; content now lives in fig_population_structure."
    ),
    "fig_spikes_on_trajectory_by_class": (
        "Spikes on trajectory by cell class (3×3, one panel per class). "
        "Each occupied class shows a median-rate unit’s spikes on the "
        "open-field path. Empty classes show ‘not in session’."
    ),
    "fig_population_tuning": (
        "Population tuning by cell class (3×3, one panel per class). "
        "MEC_grid shows a 3×3 mosaic of example spatial rate maps; other "
        "classes are units × feature rate heatmaps (rows sorted by preferred "
        "feature). MEC_hd: head direction; MEC_speed: speed; Sub_bvc: wall "
        "distance; place classes: distance to field center; INT_*: theta "
        "phase. Empty classes show ‘not in session’."
    ),
    "fig_probe_trajectory": (
        "Neuropixels insertion anatomy for the active trajectory. "
        "(A) Region bands along probe depth. "
        "(B) Channel index mapped to regions. "
        "(C) Unit positions in probe-local 3D coordinates (depth along the "
        "shank vs small lateral offsets), colored by cell class. "
        "(D) Annotated heatmap of simulated unit counts by region × cell "
        "class — the cells picked up along the probe trajectory. Local "
        "INT_* pools appear as a single interneuron class (counts by region)."
    ),
    "fig_sorting_summary": (
        "Sorting yield summary. "
        "(A) Ground-truth versus sorted spike counts by cell class. "
        "(B) Sorting loss (1 − sorted/GT)."
    ),
    "fig_latent_geometry_features": (
        "Compact overview of latent embeddings for non-position behavioral "
        "features (optional 2-page summary mode). Prefer the per-feature "
        "fig_latent_geometry_<feature> pages, which show every embedding mode."
    ),
    # ---- Decoding / manifolds / realtime ----
    "fig_decoding_performance": (
        "Causal decoding performance on sorted spikes. "
        "(A) Best primary metric versus causal window W for continuous targets. "
        "(B) Balanced accuracy versus W for categorical targets. "
        "(C) Best decoder and metric value by target. "
        "(D) Selected / recommended realtime window by target."
    ),
    "fig_feature_x_window": (
        "Causal decoding tables on sorted spikes: feature × window. "
        "One panel per target in a 4×2 grid; feature modes are rows and "
        "causal windows W are columns. Each cell shows the metric and best "
        "decoder at that (feature, W); hatching marks offline-only features, "
        "and gold outlines the selected (feature, W)."
    ),
    "fig_decoder_x_window": (
        "Causal decoding tables on sorted spikes: decoder × window "
        "(realtime-compatible features only). One panel per target in a 4×2 "
        "grid; decoders are rows and causal windows W are columns. Each cell "
        "shows the metric and best realtime-compatible feature at that "
        "(decoder, W); gold outlines the selected (decoder, W)."
    ),
    "fig_decoder_comparison_grid": (
        "Schematic of the bounded decoder comparison grid on sorted spikes. "
        "(A) Nested search: for each causal window W, for each (feature F, "
        "embedding E) job fit a frozen encoder once, then for each target T and "
        "decoder D fit and score one held-out row (one CSV entry). "
        "(B) Single-row pipeline: counts x_t(W) → encoder E → latent z_t → "
        "decoder D → prediction and primary metric. (C) Scale of this experiment "
        "(windows × feature modes × target×decoder combinations)."
    ),
    "fig_window_selection_story": (
        "How causal windows are chosen and how decoding is scored on sorted "
        "spikes. (A) Primary held-out metric per behavioral target. "
        "(B) Causal pipeline: spike counts in [t−W, t) → frozen encoder E → "
        "frozen decoder D → ŷ, with default shortest_near_optimal window policy "
        "(shortest W within 5% of best metric). (C–D) Example continuous and "
        "categorical targets: score vs W for counts / global_pca / region_pca "
        "(best decoder at each W); yellow band = near-optimal; red dashed = "
        "best-accuracy W; gold = selected W. (E) Best vs selected W for all "
        "targets. (F) At selected W, best score by representation; gold star = "
        "deployable feature mode."
    ),
    "fig_continuous_decoders_feature_x_window": (
        "Continuous-target decoder suite on sorted spikes. One section per "
        "continuous decoder; each panel is feature × causal window W for one "
        "target. Hatch = offline-only features; gold = deployable selection "
        "when that decoder is chosen."
    ),
    "fig_categorical_decoders_feature_x_window": (
        "Categorical-target decoder suite on sorted spikes. One section per "
        "categorical decoder; each panel is feature × causal window W for one "
        "target. Hatch = offline-only features; gold = deployable selection "
        "when that decoder is chosen."
    ),
    "fig_manifold_decoding": (
        "Retired combined page — feature × window and decoder × window now "
        "appear as separate figures (fig_feature_x_window, fig_decoder_x_window)."
    ),
    "fig_latent_geometry": (
        "Latent neural geometry across embedding modes (sorted / deployable). "
        "Each page is one recovered behavioral variable; each panel is one "
        "feature mode at the causal window (and k / n_neighbors) that best "
        "decoded that variable on held-out data. Position pages encode arena "
        "(x, y) as hue and brightness; other pages color by the named continuous "
        "or categorical variable. "
        "See fig_latent_geometry_<feature> for the full suite."
    ),
    "fig_decoder_geometry": (
        "Decoder comparison on a shared neural manifold (sorted / deployable). "
        "Each page is one behavioral variable; all panels use the same deployable "
        "encoder while dot color encodes true behavior in z₁ vs z₂. Each panel "
        "adds a decoder-specific prediction overlay. "
        "See fig_decoder_geometry_<feature> for the full suite."
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
        "Closed-loop realtime decoding with continuous position as the primary "
        "target. "
        "(A) True versus decoded position, both colored by elapsed time "
        "(circles = true, crosses = decoded) so matched timepoints and "
        "large spatial mismatches are visible. "
        "(B) Position error over time with closed-loop trigger markers "
        "(position-in-zone policy). "
        "(C) Spatial-context confusion from the auxiliary context head "
        "(row-normalized). "
        "(D) Trigger reliability (correct versus incorrect counts)."
    ),
    "fig_closed_loop_suite": (
        "Realtime auxiliary suite targets decoded on every causal update "
        "(same run as fig_closed_loop). "
        "(A) True versus decoded speed over time. "
        "(B) Movement-state confusion matrix (row-normalized). "
        "(C) Head direction: decoded versus true when available; otherwise true "
        "heading only (decoded HD requires --closed-loop-target head_direction). "
        "(D) Summary metrics for position error, context accuracy, movement "
        "accuracy, and speed R² from realtime_metrics.json."
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
    "fig_pipeline_stage_timing": (
        "Full-run pipeline stage timing (wall clock). "
        "Shows how long simulation, decoder comparison axes "
        "(W × FeatureSet × Manifold × Decoder), closed-loop replay, "
        "and visualization take for a complete experiment. "
        "This is orchestration cost, not per-update causal latency."
    ),
    "fig_latency_realtime": (
        "Per-update closed-loop latency on sorted spikes. "
        "(A) Distribution of total update latency with mean, p95, and budget "
        "compliance. "
        "(B) Total latency versus session time. "
        "(C) Per-stage latency box plots across profiled updates. "
        "(D) Median stage contributions stacked horizontally (sequential pipeline, "
        "not parallel). Dashed line marks the update budget."
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
        "Legacy stem; probe geometry now lives in fig_probe_trajectory panels A–B."
    ),
    "unit_depth_by_cell_class": (
        "Legacy stem; unit depth by cell class now lives in fig_probe_trajectory panel C."
    ),
    "unit_count_by_region_and_cell_class": (
        "Legacy stem; region × cell-class counts now live in "
        "fig_probe_trajectory panel D (annotated heatmap)."
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
    "fig_manifold_vs_spikes_onepager": (
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
    "fig_deployable_winner_onepager": (
        "One-pager for sorted / Neuropixels deployable selection. Top table lists "
        "the registry winner per target (decoder, feature, W, metric). Lower panels "
        "are decoder × feature heatmaps collapsed over windows (best W annotated in "
        "each cell; greener is better). Gold outline marks the selected cell; "
        "hatching marks offline-only features such as classic Isomap."
    ),
    "deployable_decoder_x_window_heatmaps": (
        "Retired standalone page — decoder × window heatmaps now appear in "
        "fig_decoder_x_window."
    ),
    "fig_deployable_decoder_x_window_heatmaps": (
        "Retired standalone page — decoder × window heatmaps now appear in "
        "fig_decoder_x_window."
    ),
}

# Regex patterns for parameterized stems → caption template with {match} groups.
_PATTERN_CAPTIONS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"^fig_decoder_geometry_(.+)$"),
        (
            "Decoder comparison on a shared neural manifold, colored by true {0} "
            "(sorted / deployable spikes). All panels use the same deployable encoder "
            "(feature mode and causal window W from the registry). Dot color encodes "
            "true behavior in latent space (z₁ vs z₂), matching fig_latent_geometry_* "
            "style. Each decoder panel stacks the shared manifold (top) and a "
            "true-vs-predicted panel (bottom): trajectory for position, trace for "
            "continuous targets, or class strips for categorical targets."
        ),
    ),
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
        re.compile(r"^population_tuning_(.+)$"),
        (
            "Population primary-feature tuning for {0} cells. "
            "Place/grid classes: rate-map mosaic (top units by mean rate) and "
            "field-center scatter for all units. One-dimensional classes: all "
            "unit tuning curves with population mean, plus a preference "
            "heatmap (or amplitude histogram for speed cells)."
        ),
    ),
    (
        re.compile(r"^example_units_(.+)$"),
        (
            "Legacy example-unit page for {0} cells (replaced by "
            "population_tuning_* and fig_spikes_on_trajectory_by_class)."
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
    "INT_CA1": "CA1 interneuron",
    "INT_CA2": "CA2 interneuron",
    "INT_CA3": "CA3 interneuron",
    "INT_DG": "DG interneuron",
    "INT_SUB": "Sub interneuron",
    "interneuron": "CA1 interneuron",  # legacy
    "CA1_int": "CA1 interneuron",  # legacy
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
