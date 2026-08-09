"""Named neural feature-set configurations.

A *feature set* describes which causal neural variables are extracted from the
spike stream.  A *manifold / embedding* is a separate transform applied after
feature extraction (identity / PCA / Isomap / …).

Hypothesis under test: population relationships and dynamics may provide a more
stable representation than individual firing rates under Neuropixels degradation.
"""

from __future__ import annotations

from typing import Any

# All feature family identifiers.
ALL_FEATURE_FAMILIES: tuple[str, ...] = (
    "counts",
    "count_dynamics",
    "population_statistics",
    "regional_statistics",
    "within_region_coactivity",
    "cross_region_coactivity",
    "lagged_coupling",
)

# Named feature sets → ordered unique families (counts always first when present).
FEATURE_SET_DEFINITIONS: dict[str, tuple[str, ...]] = {
    "counts": ("counts",),
    "counts_dynamics": ("counts", "count_dynamics"),
    "counts_population": ("counts", "population_statistics"),
    "counts_regional": ("counts", "regional_statistics"),
    "counts_coactivity": ("counts", "within_region_coactivity"),
    "counts_regional_coactivity": (
        "counts",
        "regional_statistics",
        "within_region_coactivity",
        "cross_region_coactivity",
    ),
    "full_population_state": (
        "counts",
        "count_dynamics",
        "population_statistics",
        "regional_statistics",
        "within_region_coactivity",
        "cross_region_coactivity",
        "lagged_coupling",
    ),
}

ALL_FEATURE_SETS: tuple[str, ...] = tuple(FEATURE_SET_DEFINITIONS.keys())

# Default exploration grid for CLI / comparison.
DEFAULT_FEATURE_SETS: tuple[str, ...] = (
    "counts",
    "counts_dynamics",
    "counts_regional",
    "counts_regional_coactivity",
    "full_population_state",
)

QUICK_FEATURE_SETS: tuple[str, ...] = (
    "counts",
    "counts_dynamics",
    "counts_regional",
)

# Ablations: remove one family from full_population_state at a time.
ABLATION_REMOVE_FAMILIES: tuple[str, ...] = (
    "count_dynamics",
    "population_statistics",
    "regional_statistics",
    "within_region_coactivity",
    "cross_region_coactivity",
    "lagged_coupling",
)

# Families that need internal coactivity bins.
COACTIVITY_FAMILIES = frozenset({
    "within_region_coactivity",
    "cross_region_coactivity",
    "lagged_coupling",
})

# Manifolds that assume one feature column per unit (groupwise PCA).
UNIT_ALIGNED_EMBEDDINGS = frozenset({
    "region_pca",
    "layer_pca",
    "cell_type_pca",
    "rate_model_pca",
})


def resolve_feature_sets(
    feature_sets: tuple[str, ...] | list[str] | None,
    *,
    max_models: str = "quick",
) -> tuple[str, ...]:
    if feature_sets:
        unknown = [s for s in feature_sets if s not in FEATURE_SET_DEFINITIONS]
        if unknown:
            raise ValueError(
                f"Unknown feature set(s): {unknown}. "
                f"Expected one of {ALL_FEATURE_SETS}"
            )
        return tuple(dict.fromkeys(feature_sets))
    if max_models == "full":
        return ALL_FEATURE_SETS
    return ("counts",)


def families_for_feature_set(feature_set: str) -> tuple[str, ...]:
    if feature_set not in FEATURE_SET_DEFINITIONS:
        raise ValueError(f"Unknown feature set: {feature_set!r}")
    return FEATURE_SET_DEFINITIONS[feature_set]


def ablation_feature_set_name(removed_family: str) -> str:
    return f"full_minus_{removed_family}"


def build_ablation_definitions() -> dict[str, tuple[str, ...]]:
    """Return named ablations of ``full_population_state``."""
    base = list(FEATURE_SET_DEFINITIONS["full_population_state"])
    out: dict[str, tuple[str, ...]] = {}
    for fam in ABLATION_REMOVE_FAMILIES:
        if fam not in base:
            continue
        remaining = tuple(f for f in base if f != fam)
        # Always keep counts as the anchor representation.
        if "counts" not in remaining:
            remaining = ("counts",) + remaining
        out[ablation_feature_set_name(fam)] = remaining
    return out


def feature_set_requires_coactivity_bins(feature_set: str) -> bool:
    fams = set(families_for_feature_set(feature_set))
    return bool(fams & COACTIVITY_FAMILIES)


def feature_set_is_unit_aligned(feature_set: str) -> bool:
    """True when the feature vector is still one column per unit (counts only)."""
    return families_for_feature_set(feature_set) == ("counts",)


def embedding_compatible_with_feature_set(
    embedding_type: str,
    feature_set: str,
) -> bool:
    """Groupwise PCA requires unit-aligned count features."""
    if embedding_type in UNIT_ALIGNED_EMBEDDINGS:
        return feature_set_is_unit_aligned(feature_set)
    return True


def default_extractor_config(
    feature_set: str,
    *,
    decode_window: float = 0.250,
    update_dt: float = 0.050,
    coactivity_bin_dt: float | None = None,
    include_count_derivative: bool = False,
    allow_full_pairwise: bool = False,
    lagged_coupling_lags: tuple[int, ...] = (1, 2),
) -> dict[str, Any]:
    """Build a NeuralFeatureExtractor kwargs dict for a named feature set."""
    fams = families_for_feature_set(feature_set)
    bin_dt = coactivity_bin_dt
    if bin_dt is None:
        # Prefer ~10 bins within W, floored at a fine but practical resolution.
        bin_dt = max(0.010, float(decode_window) / 10.0)
    return {
        "feature_set": feature_set,
        "families": fams,
        "decode_window": float(decode_window),
        "update_dt": float(update_dt),
        "coactivity_bin_dt": float(bin_dt),
        "include_count_derivative": bool(include_count_derivative),
        "allow_full_pairwise": bool(allow_full_pairwise),
        "lagged_coupling_lags": tuple(int(x) for x in lagged_coupling_lags),
    }
