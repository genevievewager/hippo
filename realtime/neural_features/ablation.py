"""Feature-family ablation helpers for richer population-state representations."""

from __future__ import annotations

from typing import Any

from realtime.neural_features.feature_sets import (
    ABLATION_REMOVE_FAMILIES,
    FEATURE_SET_DEFINITIONS,
    ablation_feature_set_name,
    build_ablation_definitions,
)


def ablation_feature_sets(
    base: str = "full_population_state",
) -> dict[str, tuple[str, ...]]:
    """Named ablations: base plus base-minus-each-family."""
    if base != "full_population_state":
        # Generic: remove each non-count family from the named base.
        fams = list(FEATURE_SET_DEFINITIONS.get(base, ()))
        out: dict[str, tuple[str, ...]] = {base: tuple(fams)}
        for fam in ABLATION_REMOVE_FAMILIES:
            if fam not in fams:
                continue
            remaining = tuple(f for f in fams if f != fam)
            if "counts" not in remaining:
                remaining = ("counts",) + remaining
            out[ablation_feature_set_name(fam)] = remaining
        return out
    out = {"full_population_state": FEATURE_SET_DEFINITIONS["full_population_state"]}
    out.update(build_ablation_definitions())
    return out


def describe_ablation(feature_set: str) -> dict[str, Any]:
    defs = ablation_feature_sets()
    fams = defs.get(feature_set)
    removed = None
    if feature_set.startswith("full_minus_"):
        removed = feature_set[len("full_minus_"):]
    return {
        "feature_set": feature_set,
        "families": list(fams) if fams else None,
        "removed_family": removed,
        "is_ablation": removed is not None,
    }
