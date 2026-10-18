"""Phase 1 matching: regions, chunks, composition-matched subsamples, derived configs."""

from agents.data_quality.matching.regions import (
    LAB_PRIMARY,
    LAB_SECONDARY,
    map_sim_region,
    stratify_lab_units,
    composition_matched_subsample,
)
from agents.data_quality.matching.derived_config import (
    CONDITION_KEYS,
    lab_segment_match_spec,
    build_matched_capture_yaml,
    equal_area_arena_side_cm,
)
from agents.data_quality.matching.chunks import chunk_bounds

__all__ = [
    "LAB_PRIMARY",
    "LAB_SECONDARY",
    "map_sim_region",
    "stratify_lab_units",
    "composition_matched_subsample",
    "CONDITION_KEYS",
    "lab_segment_match_spec",
    "build_matched_capture_yaml",
    "equal_area_arena_side_cm",
    "chunk_bounds",
]
