"""Real-data adapter for the shared quadrant_n5 analyze_source pipeline."""

from analysis.real_quadrant.adapter import (
    ExtentError,
    IntegrityError,
    assert_path_extent_matches_boundary,
    build_segment_bundle,
    causal_count_matrix,
    centre_window_match_fraction,
)

__all__ = [
    "ExtentError",
    "IntegrityError",
    "assert_path_extent_matches_boundary",
    "build_segment_bundle",
    "causal_count_matrix",
    "centre_window_match_fraction",
]
