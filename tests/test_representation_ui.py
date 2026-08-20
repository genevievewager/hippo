"""UI helper tests for representation capability badges."""

from __future__ import annotations

from ui.services.representations import (
    format_representation_label,
    representation_capabilities,
    REPRESENTATION_QUADRANTS,
    REALTIME_QUADRANT_DEFAULTS,
)


def test_static_and_dynamic_badges():
    pca = representation_capabilities("global_pca")
    assert pca["representation_family"] == "static"
    assert pca["supports_realtime"] is True
    assert "REALTIME" in pca["badge"]

    isomap = representation_capabilities("global_isomap")
    assert isomap["supports_realtime"] is False
    assert "OFFLINE" in isomap["badge"]

    lds = representation_capabilities("global_lds")
    assert lds["representation_family"] == "dynamic"
    assert lds["supports_realtime"] is True

    gpfa = representation_capabilities("gpfa")
    assert gpfa["supports_realtime"] is False
    assert "OFFLINE" in format_representation_label("gpfa")

    diff = representation_capabilities("diffusion_nystrom")
    assert diff["representation_family"] == "static"
    assert diff["supports_realtime"] is True
    assert "REALTIME" in format_representation_label("diffusion_nystrom")


def test_comparison_targets_subset():
    from realtime.decoder_comparison import (
        CATEGORICAL_TARGETS,
        CONTINUOUS_TARGETS,
        ComparisonRunConfig,
        comparison_targets,
    )

    cfg = ComparisonRunConfig(
        input_dir=".",
        output_dir=".",
        targets=CONTINUOUS_TARGETS,
    )
    assert comparison_targets(cfg) == CONTINUOUS_TARGETS
    cfg2 = ComparisonRunConfig(
        input_dir=".",
        output_dir=".",
        targets=CATEGORICAL_TARGETS,
    )
    assert comparison_targets(cfg2) == CATEGORICAL_TARGETS
    cfg3 = ComparisonRunConfig(input_dir=".", output_dir=".")
    assert "position" in comparison_targets(cfg3)
    assert "spatial_context" in comparison_targets(cfg3)
    assert "layer_pca" not in REPRESENTATION_QUADRANTS["static_linear"]
    assert REPRESENTATION_QUADRANTS["static_linear"] == (
        "counts", "global_pca", "region_pca",
    )
    assert "diffusion_nystrom" in REPRESENTATION_QUADRANTS["static_nonlinear"]
    assert "global_lds" in REPRESENTATION_QUADRANTS["dynamic_linear"]
    assert "gpfa" in REPRESENTATION_QUADRANTS["dynamic_linear"]
    assert REPRESENTATION_QUADRANTS["dynamic_nonlinear"] == ()
    assert REALTIME_QUADRANT_DEFAULTS["static_nonlinear"] == "diffusion_nystrom"
    assert REALTIME_QUADRANT_DEFAULTS["dynamic_nonlinear"] is None
    public = [m for methods in REPRESENTATION_QUADRANTS.values() for m in methods]
    assert "layer_pca" not in public


def test_comparison_targets_subset():
    from realtime.decoder_comparison import (
        CATEGORICAL_TARGETS,
        CONTINUOUS_TARGETS,
        ComparisonRunConfig,
        comparison_targets,
    )

    cfg = ComparisonRunConfig(
        input_dir=".",
        output_dir=".",
        targets=CONTINUOUS_TARGETS,
    )
    assert comparison_targets(cfg) == CONTINUOUS_TARGETS
    cfg2 = ComparisonRunConfig(
        input_dir=".",
        output_dir=".",
        targets=CATEGORICAL_TARGETS,
    )
    assert comparison_targets(cfg2) == CATEGORICAL_TARGETS
    cfg3 = ComparisonRunConfig(input_dir=".", output_dir=".")
    assert "position" in comparison_targets(cfg3)
    assert "spatial_context" in comparison_targets(cfg3)
