"""Page: Manifold Explorer — saved figures + on-demand analysis runs."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui.artifacts.models import CATEGORY_MANIFOLDS
from ui.artifacts.rendering import load_artifacts, render_artifact_gallery
from ui.components.controls import dataset_selector, spike_source_selector
from ui.components.plots import (
    explained_variance_bars,
    latent_trajectory_2d,
    latent_trajectory_3d,
)
from ui.services.comparison import (
    UI_FEATURE_SET_OPTIONS,
    format_decode_window,
    valid_feature_manifold_pairs,
)
from ui.services.manifold_analysis import ManifoldAnalysisRequest, run_manifold_analysis
from ui.services.manifolds import ui_manifold_choices
from ui import state

logger = logging.getLogger(__name__)


def render(outputs_root: Path) -> None:
    st.header("Manifold Explorer")
    st.caption(
        "Browse PDF-style latent geometry (colored by behavioral variables) "
        "or **Run Manifold Analysis** over feature sets × manifolds × windows. "
        "Widget changes never start computation."
    )

    dataset = dataset_selector(outputs_root, key="man_dataset")
    spike_source = spike_source_selector(dataset, key="man_spike_source")
    if dataset is None:
        return

    tab_figs, tab_run, tab_live = st.tabs([
        "Saved figures", "Run Manifold Analysis", "Quick interactive fit",
    ])

    with tab_figs:
        arts = load_artifacts(dataset)
        from ui.artifacts.discovery import filter_artifacts

        man_arts = filter_artifacts(arts, categories=[CATEGORY_MANIFOLDS])
        extras = [
            a for a in arts
            if a.stem.startswith((
                "fig_latent_geometry", "fig_isomap", "fig_manifold", "fig_decoder_geometry",
            )) and a not in man_arts
        ]
        show = man_arts + extras

        def _geometry_rank(a) -> tuple:
            # Prefer PDF-style geometry pages under figures/decoder_comparison/
            is_geo = a.stem.startswith("fig_latent_geometry_")
            path_s = str(a.path).replace("\\", "/")
            in_pub = "/figures/decoder_comparison/" in path_s
            # Stable COLOR_FEATURES order
            from visualization.publication_isomap_plots import COLOR_FEATURES
            order = {k: i for i, (k, _) in enumerate(COLOR_FEATURES)}
            feat = a.target or ""
            for k in order:
                if a.stem == f"fig_latent_geometry_{k}" or a.stem.startswith(
                    f"fig_latent_geometry_{k}__"
                ):
                    feat = k
                    break
            return (
                0 if is_geo else 1,
                0 if in_pub else 1,
                order.get(feat, 99),
                a.title.lower(),
                str(a.path),
            )

        # One card per geometry stem (drop manifolds/ run copies when pub exists)
        by_stem: dict[str, object] = {}
        for a in sorted(show, key=_geometry_rank):
            if a.stem.startswith("fig_latent_geometry_"):
                if a.stem not in by_stem:
                    by_stem[a.stem] = a
            else:
                by_stem.setdefault(f"__other__{a.path}", a)
        show = sorted(by_stem.values(), key=_geometry_rank)

        if not show:
            st.info("No manifold figures yet. Use **Run Manifold Analysis**.")
            from ui.components.viz_actions import render_generate_viz_panel

            if render_generate_viz_panel(
                dataset,
                key="man_viz",
                compact=True,
                default_simulation=False,
            ):
                st.rerun()
        else:
            geo = [a for a in show if a.stem.startswith("fig_latent_geometry_")]
            other = [a for a in show if a not in geo]
            if geo:
                st.subheader("Latent geometry (manifolds × behavioral color)")
                st.caption(
                    "Each figure is a 2×3 panel grid of embedding modes, "
                    "colored by one behavioral variable."
                )
                render_artifact_gallery(geo, key="man_geo_gallery", columns=2)
            if other:
                with st.expander(f"Other manifold figures ({len(other)})", expanded=not geo):
                    render_artifact_gallery(other, key="man_gallery", columns=2)

    with tab_run:
        st.markdown(f"**Active dataset:** `{dataset.name}`")
        feature_sets = st.multiselect(
            "Feature sets",
            options=list(UI_FEATURE_SET_OPTIONS),
            default=["counts"],
            key="man_run_fs",
            help="Groupwise PCA (region/layer) requires the `counts` feature set.",
        )
        manifolds = st.multiselect(
            "Compare manifolds",
            options=ui_manifold_choices(),
            default=[
                "counts",
                "global_pca",
                "region_pca",
                "layer_pca",
                "global_isomap",
                "global_isomap_distilled",
            ],
            key="man_run_mans",
            help="Six modes fill a 2×3 latent-geometry page (one page per behavioral color).",
        )
        windows = st.multiselect(
            "Compare windows",
            options=[0.025, 0.050, 0.100, 0.250, 0.500, 1.000],
            default=[0.100, 0.250, 0.500],
            format_func=format_decode_window,
            key="man_run_wins",
        )
        n_components = st.select_slider(
            "Components", options=[2, 3, 5, 8, 10], value=3, key="man_run_k",
        )
        st.caption(
            "Figures use the same layout as the PDF latent-geometry suite "
            "(`fig_latent_geometry_<behavior>.png`): one page per behavioral "
            "variable, panels = selected manifolds. When a decoder comparison "
            "exists, each panel uses that mode's **optimal decode window**; "
            "otherwise panels use ~250 ms. Extra feature sets beyond `counts` "
            "get filenames `fig_latent_geometry_<behavior>__<feature_set>.png`."
        )

        pairs = valid_feature_manifold_pairs(feature_sets, manifolds)
        compatible_manifolds = [m for m in manifolds if any(mm == m for _, mm in pairs)]
        from realtime.neural_features import embedding_compatible_with_feature_set
        from realtime.search_space import resolve_manifold_alias
        from visualization.publication_isomap_plots import COLOR_FEATURES

        n_color_pages = len(COLOR_FEATURES)
        group_count = 0
        for fs in feature_sets:
            mans_ok = [
                m for m in manifolds
                if embedding_compatible_with_feature_set(resolve_manifold_alias(m), fs)
            ]
            if mans_ok:
                group_count += len(windows)

        st.write({
            "feature×manifold pairs": len(pairs),
            "manifold fits": len(pairs) * max(len(windows), 1),
            "geometry page groups (feature×window)": group_count,
            "behavioral color pages per group": n_color_pages,
            "planned geometry figures": group_count * n_color_pages,
            "valid manifolds": compatible_manifolds,
        })
        if feature_sets and manifolds and len(compatible_manifolds) < len(manifolds):
            st.warning(
                "Some manifolds were excluded as incompatible with the selected "
                "feature sets (e.g. region_pca / layer_pca require `counts`)."
            )

        if st.button(
            "Run Manifold Analysis",
            type="primary",
            disabled=not pairs or not windows,
        ):
            state.request_action(state.KEY_MANIFOLD_COMPUTE_REQUESTED)

        if state.consume_action(state.KEY_MANIFOLD_COMPUTE_REQUESTED):
            req = ManifoldAnalysisRequest(
                input_dir=dataset,
                feature_sets=tuple(feature_sets),
                manifolds=tuple(m for m in manifolds if m in set(compatible_manifolds)),
                decode_windows=tuple(float(w) for w in windows),
                n_components=int(n_components),
                spike_source=spike_source,
            )

            progress = st.progress(0, text="Starting…")

            def _cb(msg, step, n):
                progress.progress(min(step / max(n, 1), 1.0), text=f"[{step}/{n}] {msg}")

            try:
                with st.status("Running manifold analysis…", expanded=True) as status:
                    meta = run_manifold_analysis(req, progress_callback=_cb)
                    status.update(label="Manifold analysis complete", state="complete")
                state.set_active_analysis_run(meta["run_id"])
                st.success(f"MANIFOLD ANALYSIS COMPLETE — run `{meta['run_id']}`")
                st.write({
                    "n_jobs_run": meta["n_jobs_run"],
                    "n_geometry_pages": meta.get("n_geometry_pages"),
                    "n_figures": len(meta.get("figures") or []),
                    "output": str(dataset / "manifolds" / meta["run_id"]),
                })
                st.cache_data.clear()
                st.info(
                    "Open the **Saved figures** tab to browse the regenerated "
                    "`fig_latent_geometry_*` pages (2×3 manifolds × behavioral color)."
                )
            except Exception as exc:
                logger.exception("Manifold analysis failed")
                st.error(f"Failed: {exc}")

    with tab_live:
        st.caption(
            "Single fit for interactive Plotly inspection. Prefer **Run Manifold Analysis** "
            "for PDF-style multi-panel figures colored by each behavioral variable."
        )
        live_fs = st.selectbox(
            "Feature set",
            options=list(UI_FEATURE_SET_OPTIONS),
            index=0,
            key="man_live_fs",
        )
        manifold = st.selectbox("Manifold", ui_manifold_choices(), index=1, key="man_live_m")
        window = st.select_slider(
            "Window",
            options=[0.025, 0.050, 0.100, 0.250, 0.500, 1.000],
            value=0.250,
            format_func=format_decode_window,
            key="man_live_w",
        )
        live_color = st.selectbox(
            "Preview color",
            options=["position", "speed", "head_direction", "time"],
            key="man_live_color",
        )
        if st.button("Compute quick fit", key="man_live_btn"):
            from ui.services.manifolds import compute_manifold_diagnostics
            with st.spinner("Fitting…"):
                try:
                    diag = compute_manifold_diagnostics(
                        dataset,
                        manifold,
                        feature_set=live_fs,
                        spike_source=spike_source,
                        decode_window=float(window),
                        n_components=3,
                    )
                    if diag.explained_variance_ratio:
                        st.plotly_chart(
                            explained_variance_bars(diag.explained_variance_ratio),
                            use_container_width=True,
                        )
                    beh = diag.behavior
                    if live_color == "time":
                        color, label = diag.times, "time"
                    elif live_color == "position" and {"x", "y"}.issubset(beh.columns):
                        color, label = beh["x"].to_numpy(), "x (position proxy)"
                    elif live_color in beh.columns:
                        color, label = beh[live_color].to_numpy(), live_color
                    else:
                        color, label = diag.times, "time"
                    st.plotly_chart(
                        latent_trajectory_2d(diag.latent, color=color, color_label=label),
                        use_container_width=True,
                    )
                    if diag.latent.shape[1] >= 3:
                        st.plotly_chart(
                            latent_trajectory_3d(diag.latent, color=color, color_label=label),
                            use_container_width=True,
                        )
                except Exception as exc:
                    st.error(str(exc))
