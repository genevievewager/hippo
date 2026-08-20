"""Streamlit rendering helpers for analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from ui.artifacts.discovery import (
    artifacts_by_category,
    discover_artifacts,
    experiment_mtime_token,
    filter_artifacts,
    primary_artifacts,
)
from ui.artifacts.models import ALL_CATEGORIES, AnalysisArtifact

# Large enough for a full publication / maximal manifold×feature×window suite.
DEFAULT_GALLERY_PAGE_SIZE = 200


@st.cache_data(show_spinner=False)
def cached_discover_artifacts(experiment_dir: str, mtime_token: str) -> list[dict]:
    """Cached discovery — returns JSON-serializable dicts for cache safety."""
    arts = discover_artifacts(Path(experiment_dir))
    return [a.to_dict() for a in arts]


def load_artifacts(experiment_dir: Path) -> list[AnalysisArtifact]:
    """Load artifacts for an experiment, using cache when the tree is unchanged."""
    exp = Path(experiment_dir)
    token = experiment_mtime_token(exp)
    raw = cached_discover_artifacts(str(exp.resolve()), token)
    return [_from_dict(d) for d in raw]


def _from_dict(d: dict) -> AnalysisArtifact:
    created = d.get("created_at")
    from datetime import datetime

    created_dt = None
    if created:
        try:
            created_dt = datetime.fromisoformat(created)
        except ValueError:
            created_dt = None
    return AnalysisArtifact(
        path=Path(d["path"]),
        artifact_type=d.get("artifact_type", "image"),
        category=d.get("category", "Other"),
        title=d.get("title") or Path(d["path"]).stem,
        description=d.get("description"),
        run_id=d.get("run_id"),
        target=d.get("target"),
        feature_set=d.get("feature_set"),
        manifold=d.get("manifold"),
        decoder=d.get("decoder"),
        decode_window=d.get("decode_window"),
        degradation_level=d.get("degradation_level"),
        created_at=created_dt,
        relative_path=d.get("relative_path"),
        experiment_dir=Path(d["experiment_dir"]) if d.get("experiment_dir") else None,
        source=d.get("source", "filesystem"),
        extras=d.get("extras") or {},
    )


def render_artifact_card(
    artifact: AnalysisArtifact,
    *,
    key_prefix: str = "",
    show_pdf_caption: bool = False,
    figure_number: int | None = None,
) -> None:
    """Compact card: title, short meta, image (optional PDF-style caption)."""
    st.markdown(f"**{artifact.title}**")
    meta_bits = []
    if artifact.target:
        meta_bits.append(f"{artifact.target}")
    if artifact.feature_set:
        meta_bits.append(f"{artifact.feature_set}")
    if artifact.manifold:
        meta_bits.append(f"{artifact.manifold}")
    if artifact.decoder:
        meta_bits.append(f"{artifact.decoder}")
    if artifact.decode_window is not None:
        meta_bits.append(f"W={artifact.decode_window}s")
    if meta_bits:
        st.caption(" · ".join(meta_bits))

    if not artifact.path.exists():
        st.warning(f"Missing file: `{artifact.path}`")
        return

    if artifact.is_image:
        try:
            st.image(str(artifact.path), width="stretch")
        except TypeError:
            st.image(str(artifact.path), use_column_width=True)
        if show_pdf_caption:
            try:
                from visualization.figure_captions import caption_for
                from ui.artifacts.models import PDF_UI_STEM_ORDER

                figs_root = None
                if artifact.experiment_dir is not None:
                    figs_root = Path(artifact.experiment_dir) / "figures"
                if figure_number is not None:
                    n = int(figure_number)
                else:
                    try:
                        n = PDF_UI_STEM_ORDER.index(artifact.stem) + 1
                    except ValueError:
                        n = 1
                st.caption(
                    caption_for(artifact.path, figure_number=n, figures_dir=figs_root)
                )
            except Exception:
                pass
    elif artifact.is_html:
        try:
            html = artifact.path.read_text(encoding="utf-8", errors="replace")
            components.html(html, height=480, scrolling=True)
        except OSError as exc:
            st.error(f"Could not load HTML artifact: {exc}")
    elif artifact.is_pdf:
        st.caption(f"PDF: `{artifact.path.name}`")
        _download(artifact, key_prefix)
    else:
        st.caption(f"Unsupported artifact type: {artifact.artifact_type}")


def _download(artifact: AnalysisArtifact, key_prefix: str) -> None:
    try:
        data = artifact.path.read_bytes()
    except OSError:
        return
    safe = abs(hash(f"{key_prefix}:{artifact.path}"))
    st.download_button(
        label=f"Download {artifact.path.name}",
        data=data,
        file_name=artifact.path.name,
        key=f"dl_{safe}",
    )


def render_artifact_gallery(
    artifacts: list[AnalysisArtifact],
    *,
    title: str | None = None,
    empty_message: str | None = None,
    columns: int = 2,
    page_size: int = DEFAULT_GALLERY_PAGE_SIZE,
    key: str = "gallery",
    show_filters: bool = False,
    show_pdf_captions: bool = False,
) -> list[AnalysisArtifact]:
    """Dense scientific gallery — defaults to showing a full-run figure set."""
    if title:
        st.subheader(title)

    visible = list(artifacts)
    if show_filters and artifacts:
        visible = _filter_controls(visible, key=key)

    if not visible:
        st.info(
            empty_message
            or "No visualization has been generated for this section yet."
        )
        return []

    total = len(visible)
    if total > page_size:
        n_pages = (total + page_size - 1) // page_size
        page = st.number_input(
            "Gallery page",
            min_value=1,
            max_value=n_pages,
            value=1,
            key=f"{key}_page",
        )
        start = (int(page) - 1) * page_size
        page_arts = visible[start: start + page_size]
        st.caption(f"Showing {start + 1}–{start + len(page_arts)} of {total}")
    else:
        page_arts = visible
        start = 0
        if total > 4:
            st.caption(f"{total} figure(s)")

    from ui.artifacts.models import PDF_UI_STEM_ORDER

    pdf_rank = {s: i for i, s in enumerate(PDF_UI_STEM_ORDER)}

    cols_n = max(1, min(columns, 3))
    for i in range(0, len(page_arts), cols_n):
        row = page_arts[i: i + cols_n]
        cols = st.columns(len(row))
        for j, (col, art) in enumerate(zip(cols, row)):
            with col:
                fallback = start + i + j + 1
                fig_n = None
                if show_pdf_captions:
                    fig_n = (
                        pdf_rank[art.stem] + 1
                        if art.stem in pdf_rank
                        else fallback
                    )
                render_artifact_card(
                    art,
                    key_prefix=f"{key}_{i}_{art.stem}",
                    show_pdf_caption=show_pdf_captions,
                    figure_number=fig_n,
                )
    return visible


def render_category_sections(
    artifacts: list[AnalysisArtifact],
    *,
    categories: list[str] | None = None,
    empty_hint: str | None = None,
    columns: int = 2,
    key: str = "cat",
) -> None:
    """Render artifacts grouped by category as successive sections."""
    grouped = artifacts_by_category(artifacts)
    order = categories or list(ALL_CATEGORIES)
    shown = False
    for cat in order:
        arts = [a for a in grouped.get(cat, []) if a.is_image or a.is_html or a.is_pdf]
        if not arts:
            continue
        shown = True
        st.markdown(f"### {cat}")
        render_artifact_gallery(
            arts,
            columns=columns,
            key=f"{key}_{cat}",
            page_size=DEFAULT_GALLERY_PAGE_SIZE,
        )
    if not shown:
        st.info(
            empty_hint
            or "No analysis visualizations found for this experiment yet."
        )


def render_overview_strip(
    artifacts: list[AnalysisArtifact],
    *,
    limit: int = 4,
    key: str = "overview",
) -> None:
    """Compact representative figures for Experiment Setup."""
    primaries = primary_artifacts(artifacts, limit=limit)
    if not primaries:
        st.caption("No overview figures yet.")
        return
    st.markdown("### Experiment overview")
    render_artifact_gallery(
        primaries,
        columns=min(2, len(primaries)),
        key=key,
        page_size=limit,
    )


def render_tabbed_gallery(
    artifacts: list[AnalysisArtifact],
    tab_categories: dict[str, list[str]],
    *,
    key: str = "tabs",
    columns: int = 2,
    page_size: int = DEFAULT_GALLERY_PAGE_SIZE,
    show_pdf_captions: bool = False,
) -> None:
    """Tabs mapped to category lists, e.g. Behavior → [Behavior, Probe]."""
    labels = list(tab_categories.keys())
    if not labels:
        return
    tabs = st.tabs(labels)
    for tab, label in zip(tabs, labels):
        cats = tab_categories[label]
        subset = filter_artifacts(artifacts, categories=cats)
        with tab:
            if not subset:
                st.info(f"No {label.lower()} visualization has been generated for this run.")
            else:
                render_artifact_gallery(
                    subset,
                    columns=columns,
                    page_size=page_size,
                    key=f"{key}_{label}",
                    empty_message=f"No {label.lower()} figures found.",
                    show_pdf_captions=show_pdf_captions,
                )


def _filter_controls(
    artifacts: list[AnalysisArtifact],
    *,
    key: str,
) -> list[AnalysisArtifact]:
    c1, c2, c3, c4 = st.columns(4)

    def _opts(attr: str) -> list[str]:
        vals = sorted({getattr(a, attr) for a in artifacts if getattr(a, attr)})
        return [str(v) for v in vals]

    with c1:
        cats = st.multiselect("Category", sorted({a.category for a in artifacts}), key=f"{key}_f_cat")
    with c2:
        targets = st.multiselect("Target", _opts("target"), key=f"{key}_f_tgt")
    with c3:
        manifolds = st.multiselect("Manifold", _opts("manifold"), key=f"{key}_f_man")
    with c4:
        feature_sets = st.multiselect("Feature set", _opts("feature_set"), key=f"{key}_f_fs")

    return filter_artifacts(
        artifacts,
        categories=cats or None,
        targets=targets or None,
        manifolds=manifolds or None,
        feature_sets=feature_sets or None,
    )
