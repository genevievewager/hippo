"""Reusable gallery widgets (thin wrappers around ui.artifacts.rendering)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui.artifacts.models import AnalysisArtifact
from ui.artifacts.rendering import (
    load_artifacts,
    render_artifact_gallery,
    render_category_sections,
    render_overview_strip,
    render_tabbed_gallery,
)


def gallery_for_experiment(
    experiment_dir: Path | None,
    *,
    categories: list[str] | None = None,
    title: str | None = None,
    key: str = "exp_gallery",
    empty_message: str | None = None,
    columns: int = 2,
    show_filters: bool = False,
) -> list[AnalysisArtifact]:
    """Discover + render figures for an experiment (no scientific recomputation)."""
    if experiment_dir is None or not Path(experiment_dir).exists():
        st.info(empty_message or "Select an experiment to browse visualizations.")
        return []
    arts = load_artifacts(Path(experiment_dir))
    if categories:
        from ui.artifacts.discovery import filter_artifacts
        arts = filter_artifacts(arts, categories=categories)
    return render_artifact_gallery(
        arts,
        title=title,
        empty_message=empty_message,
        columns=columns,
        key=key,
        show_filters=show_filters,
    )


__all__ = [
    "gallery_for_experiment",
    "load_artifacts",
    "render_artifact_gallery",
    "render_category_sections",
    "render_overview_strip",
    "render_tabbed_gallery",
]
