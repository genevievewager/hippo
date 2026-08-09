"""UI helper tests for representation capability badges."""

from __future__ import annotations

from ui.services.representations import (
    format_representation_label,
    representation_capabilities,
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
