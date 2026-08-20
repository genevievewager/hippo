"""Typed analysis artifact model for the Streamlit visualization browser."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


# Categories aligned to real figure subdirs / stems in this repo.
CATEGORY_BEHAVIOR = "Behavior"
CATEGORY_PROBE = "Probe / Anatomy"
CATEGORY_NEURAL = "Neural Activity"
CATEGORY_FEATURES = "Features"
CATEGORY_SPIKE_QUALITY = "Spike Quality"
CATEGORY_MANIFOLDS = "Manifolds"
CATEGORY_DECODING = "Decoding"
CATEGORY_DECODER_COMPARISON = "Decoder Comparison"
CATEGORY_REALTIME = "Realtime Replay"
CATEGORY_DEPLOYMENT = "Deployment"
CATEGORY_PERFORMANCE = "Performance / Runtime"
CATEGORY_OTHER = "Other"

ALL_CATEGORIES: tuple[str, ...] = (
    CATEGORY_BEHAVIOR,
    CATEGORY_PROBE,
    CATEGORY_NEURAL,
    CATEGORY_FEATURES,
    CATEGORY_SPIKE_QUALITY,
    CATEGORY_MANIFOLDS,
    CATEGORY_DECODING,
    CATEGORY_DECODER_COMPARISON,
    CATEGORY_REALTIME,
    CATEGORY_DEPLOYMENT,
    CATEGORY_PERFORMANCE,
    CATEGORY_OTHER,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".webp"}
HTML_SUFFIXES = {".html", ".htm"}
PDF_SUFFIXES = {".pdf"}
VISUAL_SUFFIXES = IMAGE_SUFFIXES | HTML_SUFFIXES | PDF_SUFFIXES

# Subdirectory under figures/ → default category
SUBDIR_CATEGORY: dict[str, str] = {
    "trajectory": CATEGORY_PROBE,
    "behavior": CATEGORY_BEHAVIOR,
    "features": CATEGORY_FEATURES,
    "neural": CATEGORY_NEURAL,
    "sorting": CATEGORY_SPIKE_QUALITY,
    "manifolds": CATEGORY_MANIFOLDS,
    "dynamic": CATEGORY_MANIFOLDS,
    "decoder_comparison": CATEGORY_DECODING,
    "realtime_decoding": CATEGORY_REALTIME,
    "deployment_decoder_selection": CATEGORY_DEPLOYMENT,
    "latency": CATEGORY_PERFORMANCE,
    "temporal_decoding": CATEGORY_DECODING,
    "report": CATEGORY_OTHER,
}

# Stem prefixes that live under decoder_comparison/ but are manifold figures.
# fig_manifold_vs_spikes_onepager is a decoding comparison (see STEM_CATEGORY_OVERRIDES).
# fig_decoder_geometry_* is omitted from the UI (not clean enough).
MANIFOLD_STEM_PREFIXES = (
    "fig_latent_geometry",
    "fig_isomap",
    "latent_geometry",
    "isomap_",
    "global_pca_",
    "region_pca_",
)

# Per-stem category when the figures/ subdirectory would otherwise mis-file them.
STEM_CATEGORY_OVERRIDES: dict[str, str] = {
    "fig_neural_drivers": CATEGORY_BEHAVIOR,
    "fig_manifold_vs_spikes_onepager": CATEGORY_DECODING,
}

# Hidden from every UI gallery (PDF Figs 21–28).
UI_HIDDEN_STEM_PREFIXES: tuple[str, ...] = (
    "fig_decoder_geometry",
)

# Publication PDF order for included figures (skips decoder-geometry 21–28).
PDF_UI_STEM_ORDER: tuple[str, ...] = (
    "fig_probe_trajectory",
    "fig_behavior_dynamics",
    "fig_neural_drivers",
    "fig_spikes_on_trajectory_by_class",
    "fig_population_tuning",
    "fig_circuit_feedforward",
    "fig_population_structure",
    "fig_population_activity",
    "fig_sorting_summary",
    "fig_decoding_performance",
    "fig_isomap_diagnostics",
    "fig_isomap_story",
    "fig_latent_geometry_acceleration",
    "fig_latent_geometry_distance_to_wall",
    "fig_latent_geometry_head_direction",
    "fig_latent_geometry_movement_state",
    "fig_latent_geometry_position",
    "fig_latent_geometry_spatial_context",
    "fig_latent_geometry_speed",
    "fig_latent_geometry_wall_distance_bin",
    "fig_decoder_comparison_grid",
    "fig_window_selection_story",
    "fig_feature_x_window",
    "fig_decoder_x_window",
    "fig_continuous_decoders_feature_x_window",
    "fig_categorical_decoders_feature_x_window",
    "fig_manifold_vs_spikes_onepager",
    "fig_closed_loop",
    "fig_closed_loop_suite",
    "fig_deployment",
    "fig_latency",
    "fig_latency_realtime",
)

# Stems promoted as primary/overview figures (order = priority)
PRIMARY_STEMS: tuple[str, ...] = (
    "fig_behavior_dynamics",
    "fig_population_activity",
    "fig_sorting_summary",
    "fig_latent_geometry_position",
    "fig_decoding_performance",
    "fig_decoder_comparison_grid",
    "fig_closed_loop",
    "fig_manifold_vs_spikes_onepager",
    "fig_feature_x_window",
    "fig_probe_trajectory",
    "fig_neural_drivers",
    "fig_population_structure",
    "fig_spikes_on_trajectory_by_class",
    "fig_isomap_diagnostics",
    "fig_latency",
)

READABLE_TITLES: dict[str, str] = {
    "fig_behavior_dynamics": "Behavior dynamics — trajectory, speed, occupancy",
    "fig_neural_drivers": "Neural driver features by cell class",
    "fig_probe_trajectory": "Probe trajectory and region capture",
    "fig_circuit_feedforward": "Feedforward circuit diagram",
    "fig_population_activity": "Population activity",
    "fig_population_structure": "Population structure",
    "fig_population_tuning": "Population tuning",
    "fig_spikes_on_trajectory_by_class": "Spikes on trajectory by cell class",
    "fig_sorting_summary": "Sorting quality summary (GT vs sorted)",
    "fig_decoding_performance": "Decoding performance summary",
    "fig_decoder_comparison_grid": "Decoder comparison grid",
    "fig_feature_x_window": "Feature × decode-window performance",
    "fig_decoder_x_window": "Decoder × decode-window performance",
    "fig_continuous_decoders_feature_x_window": "Continuous targets — feature × window",
    "fig_categorical_decoders_feature_x_window": "Categorical targets — feature × window",
    "fig_window_selection_story": "Window selection story",
    "fig_manifold_vs_spikes_onepager": "Manifold vs spike-count one-pager",
    "fig_isomap_diagnostics": "Isomap diagnostics",
    "fig_isomap_story": "Isomap story",
    "fig_closed_loop": "Closed-loop realtime replay",
    "fig_closed_loop_suite": "Closed-loop suite",
    "fig_deployment": "Deployment decoder selection",
    "fig_latency": "Decoder latency",
    "fig_latency_realtime": "Realtime latency profile",
    "fig_pipeline_stage_timing": "Pipeline stage timing",
    "fig_feature_set_performance": "Feature-set performance",
    "fig_feature_set_manifold_heatmap": "Feature-set × manifold heatmap",
    "fig_feature_dim_cost_tradeoff": "Feature dimension vs compute cost",
}


@dataclass
class AnalysisArtifact:
    """One discoverable analysis visualization or related artifact."""

    path: Path
    artifact_type: str  # image | html | pdf | data
    category: str
    title: str
    description: str | None = None
    run_id: str | None = None
    target: str | None = None
    feature_set: str | None = None
    manifold: str | None = None
    decoder: str | None = None
    decode_window: float | None = None
    degradation_level: float | None = None
    created_at: datetime | None = None
    relative_path: str | None = None
    experiment_dir: Path | None = None
    source: str = "filesystem"  # manifest | filesystem
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        d["experiment_dir"] = str(self.experiment_dir) if self.experiment_dir else None
        d["created_at"] = self.created_at.isoformat() if self.created_at else None
        return d

    @property
    def stem(self) -> str:
        return self.path.stem

    @property
    def is_image(self) -> bool:
        return self.path.suffix.lower() in IMAGE_SUFFIXES

    @property
    def is_html(self) -> bool:
        return self.path.suffix.lower() in HTML_SUFFIXES

    @property
    def is_pdf(self) -> bool:
        return self.path.suffix.lower() in PDF_SUFFIXES
