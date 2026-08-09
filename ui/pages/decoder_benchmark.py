"""Page: Decoder Benchmark — launch comparison via shared backend."""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

import streamlit as st

from ui.components.controls import (
    dataset_selector,
    decode_window_multiselect,
    feature_set_multiselect,
    manifold_multiselect,
    render_architecture_diagram,
    representation_family_selector,
    spike_source_selector,
)
from ui.components.run_status import render_run_metadata, render_status
from ui.services.comparison import (
    UIBenchmarkSelection,
    estimate_workload,
    selection_to_dict,
    validate_benchmark_selection,
    run_benchmark,
)
from ui.services.representations import (
    UI_DYNAMIC_LATENT_OPTIONS,
    UI_STATIC_MANIFOLD_OPTIONS,
    format_representation_label,
)
from ui.services.registry import (
    create_pending_metadata,
    save_run_metadata,
)
from ui import state

logger = logging.getLogger(__name__)


def render(outputs_root: Path) -> None:
    st.header("Decoder Benchmark")
    st.caption(
        "Configure the W × FeatureSet × Manifold × Decoder grid, then click "
        "**Run Benchmark**. Changing checkboxes alone never starts computation."
    )

    dataset = dataset_selector(outputs_root, key="bench_dataset")
    if dataset is None:
        return
    st.markdown(f"**Input dataset:** `{dataset.name}`")
    spike_source = spike_source_selector(dataset, key="bench_spike_source")

    # Show existing decoding figures for this experiment (no recompute)
    from ui.artifacts.models import CATEGORY_DECODER_COMPARISON, CATEGORY_DECODING
    from ui.artifacts.rendering import load_artifacts, render_artifact_gallery
    from ui.artifacts.discovery import filter_artifacts

    existing_figs = filter_artifacts(
        load_artifacts(dataset),
        categories=[CATEGORY_DECODING, CATEGORY_DECODER_COMPARISON],
    )
    if existing_figs:
        st.markdown(f"### Decoding figures ({len(existing_figs)})")
        render_artifact_gallery(
            existing_figs,
            columns=2,
            key="bench_existing_figs",
        )

    mode = st.radio(
        "Analysis mode",
        options=["Quick Exploration", "Full Benchmark"],
        horizontal=True,
        key="bench_mode",
        help="Quick Exploration uses a lean grid for newly generated datasets.",
    )

    family = representation_family_selector(key="bench_rep_family")
    render_architecture_diagram(family)

    st.subheader("Feature sets")
    if mode == "Quick Exploration":
        feature_sets = feature_set_multiselect(
            key="bench_feature_sets",
            defaults=("counts", "counts_dynamics"),
        )
    else:
        feature_sets = feature_set_multiselect(key="bench_feature_sets")

    if family == "static":
        st.subheader("Static manifolds")
        if mode == "Quick Exploration":
            manifolds = manifold_multiselect(
                key="bench_manifolds_static",
                options=UI_STATIC_MANIFOLD_OPTIONS,
                defaults=("counts", "global_pca", "region_pca"),
            )
        else:
            manifolds = manifold_multiselect(
                key="bench_manifolds_static",
                options=UI_STATIC_MANIFOLD_OPTIONS,
            )
    else:
        st.subheader("Dynamic latent states")
        manifolds = manifold_multiselect(
            key="bench_manifolds_dyn",
            options=UI_DYNAMIC_LATENT_OPTIONS,
            defaults=("global_lds",),
        )
        st.caption(
            "Global LDS is realtime/causal. GPFA is an offline/acausal comparison method."
        )

    st.subheader("Decode windows")
    if mode == "Quick Exploration":
        decode_windows = decode_window_multiselect(
            key="bench_windows",
            defaults=[0.100, 0.250],
        )
    else:
        decode_windows = decode_window_multiselect(key="bench_windows")

    st.subheader("Additional options")
    c1, c2, c3 = st.columns(3)
    with c1:
        run_feature_ablation = st.checkbox("Run feature ablation", value=False, key="bench_ablation")
        compare_sources = st.checkbox(
            "Compare GT vs sorted (sorting robustness)",
            value=False,
            key="bench_compare_sources",
            help=(
                "Dataset-level recording degradation is set at simulation time. "
                "This option runs GT vs sorted comparison for robustness analysis."
            ),
        )
        include_controls = st.checkbox("Include controls", value=False, key="bench_controls")
    with c2:
        max_models = st.selectbox(
            "Model zoo",
            options=["quick", "full"],
            index=0 if mode == "Quick Exploration" else 0,
            key="bench_max_models",
        )
        n_components = st.multiselect(
            "Latent dimensions" if family == "dynamic" else "Manifold components",
            options=[2, 3, 5, 8, 10, 20],
            default=[3, 5] if family == "dynamic" else [3],
            key="bench_k",
        )
        update_dt = st.number_input(
            "Update dt (s)",
            value=0.025 if family == "dynamic" else 0.050,
            min_value=0.001,
            step=0.025,
            key="bench_dt",
        )
    with c3:
        region_ablation = st.checkbox("Region ablation", value=False, key="bench_region_abl")
        layer_ablation = st.checkbox("Layer ablation", value=False, key="bench_layer_abl")
        population_ablation = st.checkbox("Population ablation", value=False, key="bench_pop_abl")
        no_trigger_search = st.checkbox("Disable trigger search", value=mode == "Quick Exploration", key="bench_no_trig")

    with st.expander("Advanced Settings", expanded=False):
        st.caption("Reserved for method-specific hyperparameters (EM iters, GPFA τ, …).")
        st.write({
            "representation_family": family,
            "selected_representations": [format_representation_label(m) for m in manifolds],
        })

    workload = estimate_workload(
        feature_sets=feature_sets,
        manifolds=manifolds,
        decode_windows=decode_windows,
        max_models=max_models,
        compare_sources=compare_sources,
        run_feature_ablation=run_feature_ablation,
        n_components=n_components or [3],
    )
    st.info(
        f"**Planned configurations (approx.): {workload['planned_configurations']:,}**  \n"
        f"{workload['n_valid_feature_manifold_pairs']} valid feature×manifold pairs "
        f"(excluded {workload['n_invalid_pairs_excluded']} invalid) · "
        f"{workload['n_windows']} windows · ~{workload['n_decoders_approx']} decoders"
        + (f" · ×{workload['n_spike_sources']} spike sources" if compare_sources else "")
    )

    default_out = (
        "decoder_comparison/dynamic" if family == "dynamic" else "decoder_comparison/sorted"
    )
    out_name = st.text_input(
        "Output subdirectory",
        value=default_out,
        help="Relative to the selected dataset, matching CLI convention.",
        key="bench_out_rel",
    )
    output_dir = dataset / out_name

    with st.expander("Resolved configuration preview", expanded=False):
        preview = {
            "input": str(dataset),
            "output": str(output_dir),
            "spike_source": spike_source,
            "feature_sets": feature_sets,
            "manifolds": manifolds,
            "decode_windows": decode_windows,
            "run_feature_ablation": run_feature_ablation,
            "compare_sources": compare_sources,
            "max_models": max_models,
            "manifold_n_components": n_components or [3],
        }
        st.json(preview)

    render_status(
        st.session_state.get(state.KEY_BENCHMARK_STATUS, "idle"),
        error=st.session_state.get(state.KEY_BENCHMARK_ERROR),
    )

    run_clicked = st.button("Run Benchmark", type="primary")
    if run_clicked:
        state.request_action(state.KEY_BENCHMARK_REQUESTED)

    if not state.consume_action(state.KEY_BENCHMARK_REQUESTED):
        return

    sel = UIBenchmarkSelection(
        input_dir=dataset,
        output_dir=output_dir,
        spike_source=spike_source,
        feature_sets=tuple(feature_sets),
        manifolds=tuple(manifolds),
        decode_windows=tuple(decode_windows),
        update_dt=float(update_dt),
        max_models=max_models,
        run_feature_ablation=run_feature_ablation,
        compare_sources=compare_sources,
        manifold_n_components=tuple(n_components or [3]),
        include_controls=include_controls,
        region_ablation=region_ablation,
        layer_ablation=layer_ablation,
        population_ablation=population_ablation,
        no_trigger_search=no_trigger_search,
    )
    errors = validate_benchmark_selection(sel)
    if errors:
        st.session_state[state.KEY_BENCHMARK_STATUS] = "failed"
        st.session_state[state.KEY_BENCHMARK_ERROR] = "; ".join(errors)
        for err in errors:
            st.error(err)
        return

    meta = create_pending_metadata(
        input_dataset=dataset,
        output_directory=output_dir,
        feature_sets=feature_sets,
        manifolds=manifolds,
        decode_windows=decode_windows,
        feature_ablation=run_feature_ablation,
        compare_sources=compare_sources,
        spike_source=spike_source,
        configuration=selection_to_dict(sel),
    )
    meta.status = "running"
    save_run_metadata(meta, output_dir)
    st.session_state[state.KEY_BENCHMARK_STATUS] = "running"
    st.session_state[state.KEY_BENCHMARK_RUN_ID] = meta.run_id
    st.session_state[state.KEY_LAST_COMPARISON_DIR] = str(output_dir)
    render_run_metadata(meta)

    progress = st.progress(0, text="Starting decoder comparison…")
    try:
        progress.progress(10, text="Calling run_decoder_comparison…")
        result = run_benchmark(sel)
        progress.progress(100, text="Complete")
        meta.status = "completed"
        save_run_metadata(meta, output_dir)
        st.session_state[state.KEY_BENCHMARK_STATUS] = "completed"
        st.session_state[state.KEY_BENCHMARK_ERROR] = None
        st.success(f"Wrote results → `{output_dir}`")
        if hasattr(result, "shape"):
            st.write(f"Metrics rows: {result.shape[0]}")
        st.info("Open the **Results** page to inspect the leaderboard.")
    except Exception as exc:
        logger.exception("Benchmark failed")
        meta.status = "failed"
        meta.error = str(exc)
        save_run_metadata(meta, output_dir)
        st.session_state[state.KEY_BENCHMARK_STATUS] = "failed"
        st.session_state[state.KEY_BENCHMARK_ERROR] = str(exc)
        st.error(f"Benchmark failed: {exc}")
        with st.expander("Traceback"):
            st.code(traceback.format_exc())
