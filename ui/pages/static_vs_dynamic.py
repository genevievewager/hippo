"""Page: Static vs Dynamic representation comparison."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.controls import (
    dataset_selector,
    render_architecture_diagram,
    spike_source_selector,
)
from ui.services.representations import (
    UI_DYNAMIC_LATENT_OPTIONS,
    UI_STATIC_MANIFOLD_OPTIONS,
    format_representation_label,
    representation_capabilities,
)


def _load_metrics(dataset: Path) -> pd.DataFrame | None:
    candidates = list(dataset.rglob("decoder_comparison_metrics.csv"))
    if not candidates:
        return None
    frames = []
    for p in candidates:
        try:
            frames.append(pd.read_csv(p))
        except Exception:
            continue
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def render(outputs_root: Path) -> None:
    st.header("Static vs Dynamic Comparison")
    st.caption(
        "Compare the same decoder and target under a static manifold versus a "
        "dynamic latent-state representation. Uses saved decoder_comparison metrics."
    )

    dataset = dataset_selector(outputs_root, key="svsd_dataset")
    spike_source_selector(dataset, key="svsd_spike")
    if dataset is None:
        return

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Static representation")
        render_architecture_diagram("static")
        static_rep = st.selectbox(
            "Static",
            options=list(UI_STATIC_MANIFOLD_OPTIONS),
            index=1,
            format_func=format_representation_label,
            key="svsd_static",
        )
    with col_b:
        st.subheader("Dynamic representation")
        render_architecture_diagram("dynamic")
        dynamic_rep = st.selectbox(
            "Dynamic",
            options=list(UI_DYNAMIC_LATENT_OPTIONS),
            index=0,
            format_func=format_representation_label,
            key="svsd_dyn",
        )

    decoder = st.selectbox(
        "Decoder",
        options=[
            "ridge",
            "random_forest_regressor",
            "logistic",
            "random_forest_classifier",
            "pca_ridge",
        ],
        index=1,
        key="svsd_decoder",
    )
    target = st.selectbox(
        "Target",
        options=["position", "speed", "head_direction", "spatial_context", "movement_state"],
        index=0,
        key="svsd_target",
    )

    caps_s = representation_capabilities(static_rep)
    caps_d = representation_capabilities(dynamic_rep)
    st.info(
        f"**{static_rep}** → `{caps_s['badge']}` · "
        f"**{dynamic_rep}** → `{caps_d['badge']}`"
    )

    df = _load_metrics(dataset)
    if df is None or df.empty:
        st.warning("No decoder_comparison_metrics.csv found. Run a benchmark first.")
        return

    # Normalize embedding column names
    emb_col = "embedding_type" if "embedding_type" in df.columns else "feature_mode"
    if emb_col not in df.columns:
        st.error("Metrics table missing embedding_type / feature_mode columns.")
        return

    from realtime.search_space import resolve_manifold_alias

    static_key = resolve_manifold_alias(static_rep)
    dyn_key = resolve_manifold_alias(dynamic_rep)

    mask = df["decoder_name"].astype(str) == decoder
    if "target_name" in df.columns:
        mask &= df["target_name"].astype(str) == target
    sub = df.loc[mask].copy()
    sub["_emb"] = sub[emb_col].astype(str).map(resolve_manifold_alias)

    static_rows = sub[sub["_emb"] == static_key]
    dyn_rows = sub[sub["_emb"] == dyn_key]

    if static_rows.empty and dyn_rows.empty:
        st.warning("No matching rows for this decoder/target/representation pair.")
        return

    metric_candidates = [
        "mean_position_error_cm",
        "median_position_error_cm",
        "p95_position_error_cm",
        "r2",
        "balanced_accuracy",
        "mean_circular_error_deg",
        "inference_latency_ms",
        "mean_total_update_ms",
        "latent_dimension",
        "manifold_n_components",
        "decode_window_s",
    ]
    show_cols = [c for c in metric_candidates if c in sub.columns]
    show_cols = [emb_col, "decoder_name", "target_name", "representation_family", "causal_status", *show_cols]
    show_cols = [c for c in show_cols if c in sub.columns]

    st.subheader("Summary table")
    combined = pd.concat([static_rows, dyn_rows], ignore_index=True)
    st.dataframe(combined[show_cols] if show_cols else combined, use_container_width=True)

    # Plots
    st.subheader("Performance vs decode window")
    if "decode_window_s" in combined.columns and "mean_position_error_cm" in combined.columns and target == "position":
        import plotly.express as px

        fig = px.line(
            combined,
            x="decode_window_s",
            y="mean_position_error_cm",
            color="_emb",
            markers=True,
            title="Position error vs decode window",
        )
        st.plotly_chart(fig, use_container_width=True)
    elif "decode_window_s" in combined.columns and "r2" in combined.columns:
        import plotly.express as px

        fig = px.line(
            combined,
            x="decode_window_s",
            y="r2",
            color="_emb",
            markers=True,
            title="R² vs decode window",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "manifold_n_components" in combined.columns:
        ycol = "mean_position_error_cm" if "mean_position_error_cm" in combined.columns else (
            "r2" if "r2" in combined.columns else None
        )
        if ycol:
            import plotly.express as px

            fig2 = px.line(
                combined.dropna(subset=["manifold_n_components"]),
                x="manifold_n_components",
                y=ycol,
                color="_emb",
                markers=True,
                title=f"{ycol} vs latent dimensionality",
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Dynamic figures gallery
    dyn_fig_dir = dataset / "figures" / "dynamic" / dyn_key
    if dyn_fig_dir.exists():
        st.subheader("Dynamic latent figures")
        pngs = sorted(dyn_fig_dir.glob("*.png"))
        cols = st.columns(2)
        for i, p in enumerate(pngs):
            cols[i % 2].image(str(p), caption=p.stem, use_container_width=True)

    # Behavioral association inspection
    assoc_paths = list((dataset / "decoder_comparison").rglob(f"behavioral_association_*{dyn_key}*"))
    assoc_paths += list((dataset / "decoder_comparison").rglob("behavioral_association_*.csv"))
    assoc_paths = [p for p in assoc_paths if dyn_key in str(p)]
    if assoc_paths:
        st.subheader("Behavioral association (latent dimensions)")
        st.caption("Association scores — not biological latent meanings.")
        assoc = pd.read_csv(assoc_paths[0])
        st.dataframe(assoc, use_container_width=True)
