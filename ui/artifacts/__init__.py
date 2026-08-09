"""Public exports for the UI artifact browser.

Keep imports light: pages should import concrete submodules
(``ui.artifacts.models``, ``ui.artifacts.rendering``) rather than forcing
the full discovery / Streamlit rendering stack at package import time.
"""

from ui.artifacts.models import ALL_CATEGORIES, AnalysisArtifact

__all__ = [
    "ALL_CATEGORIES",
    "AnalysisArtifact",
]


def __getattr__(name: str):
    if name in {
        "discover_artifacts",
        "filter_artifacts",
        "primary_artifacts",
        "artifacts_by_category",
    }:
        from ui.artifacts import discovery as _discovery

        return getattr(_discovery, name)
    if name in {
        "load_artifacts",
        "render_artifact_gallery",
        "render_category_sections",
        "render_overview_strip",
        "render_tabbed_gallery",
    }:
        from ui.artifacts import rendering as _rendering

        return getattr(_rendering, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
