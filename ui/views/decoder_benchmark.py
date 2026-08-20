"""Page: Decoder Benchmark — launch comparison via shared backend."""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from ui.components.controls import (
    active_spike_source,
    feature_set_multiselect,
    gated_decode_window_selector,
    require_active_dataset,
)
from ui.components.run_status import render_job_autofresh, render_workload_estimate
from ui.jobs import get_slot_job, submit_job
from realtime.decoder_comparison import CATEGORICAL_TARGETS, CONTINUOUS_TARGETS
from ui.services.comparison import (
    DECODER_TRANSFORM_TIME_SHARE,
    UIBenchmarkSelection,
    UI_CATEGORICAL_DECODER_OPTIONS,
    UI_CONTINUOUS_DECODER_OPTIONS,
    UI_DEFAULT_CATEGORICAL_DECODERS,
    UI_DEFAULT_CONTINUOUS_DECODERS,
    apply_remaining_work_to_workload,
    estimate_workload,
    inventory_transform_reuse,
    selection_to_dict,
    validate_benchmark_selection,
    run_benchmark,
)
from ui.services.representations import (
    QUADRANT_ORDER,
    REPRESENTATION_QUADRANT_LABELS,
    REPRESENTATION_QUADRANTS,
    format_representation_label,
)
from ui.services.registry import (
    create_pending_metadata,
    save_run_metadata,
)
from ui import state

logger = logging.getLogger(__name__)

_JOB_SLOT = "decoder:benchmark"
_PENDING_KEY = "bench_pending_request"

_CONTINUOUS_TABLE_FIGURES: tuple[tuple[str, str], ...] = (
    (
        "fig_continuous_decoders_feature_x_window",
        "Continuous decoders · feature × window",
    ),
)
_CATEGORICAL_TABLE_FIGURES: tuple[tuple[str, str], ...] = (
    (
        "fig_categorical_decoders_feature_x_window",
        "Categorical decoders · feature × window",
    ),
)


def _regenerate_decoding_tables(experiment_dir: Path) -> tuple[list[Path], list[str]]:
    """Rewrite the publication feature/decoder × window table PNGs."""
    from visualization.publication_decoding_plots import (
        plot_fig_categorical_decoders_feature_x_window,
        plot_fig_continuous_decoders_feature_x_window,
        plot_fig_decoder_x_window,
        plot_fig_feature_x_window,
    )

    written: list[Path] = []
    errors: list[str] = []
    for fn in (
        plot_fig_feature_x_window,
        plot_fig_decoder_x_window,
        plot_fig_continuous_decoders_feature_x_window,
        plot_fig_categorical_decoders_feature_x_window,
    ):
        try:
            path = fn(experiment_dir)
            if path is not None:
                written.append(Path(path))
            else:
                errors.append(f"{fn.__name__} returned None")
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s failed: %s", fn.__name__, exc)
            errors.append(f"{fn.__name__}: {exc}")
    return written, errors


def _render_decoding_tables(
    dataset: Path,
    *,
    figures: tuple[tuple[str, str], ...],
) -> None:
    """Show family-specific feature×window heatmap tables."""
    st.subheader("Decoding tables")
    st.caption(
        "Publication heatmaps under `figures/decoder_comparison/`. "
        "Gold = **best combo on disk for this dataset** (decoder × feature × W) "
        "across all comparison runs, not the deployable registry. "
        "A later one-decoder run updates the pool and can move the box."
    )
    fig_dir = dataset / "figures" / "decoder_comparison"
    missing: list[str] = []
    for stem, title in figures:
        path = fig_dir / f"{stem}.png"
        st.markdown(f"**{title}**")
        if path.exists():
            mtime = path.stat().st_mtime_ns
            st.image(str(path), width="stretch")
            st.caption(f"`{path.name}` · mtime `{mtime}`")
        else:
            missing.append(path.name)
            st.caption(f"Missing `{path.relative_to(dataset)}`")
    if missing:
        st.info(
            "Missing table figures regenerate automatically after a successful "
            f"**Run Benchmark**, or from the figure pipeline. Missing: "
            + ", ".join(missing)
        )


def render(outputs_root: Path) -> None:
    st.header("Decoder Benchmark")
    st.caption(
        "Background job (same paradigm as Feature Construction / Latent Representations). "
        "Continuous and discrete targets run as separate jobs. "
        "Progress shows window · feature · manifold · decoder · target with "
        "last-step and elapsed timing."
    )

    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return
    spike_source = active_spike_source(dataset, readonly=True)

    st.subheader("Feature sets")
    feature_sets = feature_set_multiselect(
        key="bench_feature_sets",
        defaults=("counts", "counts_dynamics"),
    )

    st.subheader("Representations")
    st.caption(
        "Grouped by the four representation classes. "
        "Global LDS is realtime/causal; GPFA and classic Isomap are offline."
    )
    manifolds: list[str] = []
    for qid in QUADRANT_ORDER:
        methods = list(REPRESENTATION_QUADRANTS[qid])
        label = REPRESENTATION_QUADRANT_LABELS[qid]
        if not methods:
            st.caption(f"**{label}** — not implemented.")
            continue
        defaults = methods if qid == "static_linear" else []
        picked = st.multiselect(
            label,
            options=methods,
            default=defaults,
            format_func=format_representation_label,
            key=f"bench_mans_{qid}",
        )
        manifolds.extend(picked)

    st.subheader("Decode windows")
    from realtime.transform_cache import list_cached_decode_windows

    cached_windows = list_cached_decode_windows(
        dataset, spike_source=spike_source, feature_sets=feature_sets,
    )
    decode_windows = gated_decode_window_selector(
        cached_windows,
        key="bench_windows",
        defaults=[0.100, 0.250],
        label="Decode windows",
    )

    max_models = "quick"
    run_feature_ablation = False
    compare_sources = False
    include_controls = False
    region_ablation = False
    layer_ablation = False
    population_ablation = False
    no_trigger_search = True
    n_components = [3]
    update_dt = 0.050

    out_name = st.text_input(
        "Output subdirectory",
        value="decoder_comparison/sorted",
        help="Relative to the selected dataset, matching CLI convention.",
        key="bench_out_rel",
    )
    output_dir = dataset / out_name

    reuse_transforms = st.checkbox(
        "Reuse saved manifold / feature transforms",
        value=True,
        key="bench_reuse_transforms",
        help=(
            "Load checkpoints from Latent Representations / prior benchmarks under "
            "decoder_comparison/*/models/manifold_transforms (and feature_transforms). "
            "Uncheck only if you need a forced refit."
        ),
    )
    inv_sel = UIBenchmarkSelection(
        input_dir=dataset,
        output_dir=output_dir,
        spike_source=spike_source,
        feature_sets=tuple(feature_sets),
        manifolds=tuple(manifolds),
        decode_windows=tuple(decode_windows),
        manifold_n_components=tuple(n_components),
        reuse_transforms=bool(reuse_transforms),
    )
    inv = inventory_transform_reuse(inv_sel)
    if inv.get("reuse_enabled"):
        st.info(
            f"**Transform reuse:** {inv['n_reusable']}/{inv['n_requested']} manifold "
            f"checkpoint(s) on disk"
            + (
                f" · {inv['n_feature_reusable']}/{inv['n_feature_requested']} "
                f"feature transform(s)"
                if inv.get("n_feature_requested")
                else ""
            )
            + (
                f" · {inv['n_missing']} still to fit"
                if inv.get("n_missing")
                else " · none missing"
            )
            + ". Source: Latent Representations / prior comparison runs."
        )

    slot = _JOB_SLOT
    existing = get_slot_job(slot)
    busy = existing is not None and existing.is_active

    if existing is not None:
        frozen = None
        if existing.meta:
            raw_est = existing.meta.get("estimated_runtime_s")
            try:
                frozen = float(raw_est) if raw_est is not None else None
            except (TypeError, ValueError):
                frozen = None
        job = render_job_autofresh(slot=slot, estimated_runtime_s=frozen)
        if job is not None and job.status == "completed" and job.result is not None:
            result = job.result
            meta = (job.meta or {}).get("run_meta") or {}
            run_id = meta.get("run_id") or job.job_id
            state.set_active_analysis_run(str(run_id))
            st.session_state[state.KEY_BENCHMARK_STATUS] = "completed"
            st.session_state[state.KEY_BENCHMARK_ERROR] = None
            st.session_state[state.KEY_LAST_COMPARISON_DIR] = str(output_dir)
            st.success(f"BENCHMARK COMPLETE — `{output_dir}`")
            n_rows = result.get("n_metrics_rows") if isinstance(result, dict) else None
            if n_rows is not None:
                st.write(f"Metrics rows: {n_rows}")
            n_tables = result.get("n_table_figures") if isinstance(result, dict) else None
            if n_tables:
                st.caption(
                    f"Updated {n_tables} decoding table figure(s) "
                    "under figures/decoder_comparison/."
                )
            table_errors = result.get("table_figure_errors") if isinstance(result, dict) else None
            if table_errors:
                st.warning(
                    "Decoding table figure regen issues: "
                    + "; ".join(str(e) for e in table_errors)
                )
        elif job is not None and job.status == "failed":
            st.session_state[state.KEY_BENCHMARK_STATUS] = "failed"
            st.session_state[state.KEY_BENCHMARK_ERROR] = job.error

    if "bench_family_tab" not in st.session_state:
        st.session_state["bench_family_tab"] = "continuous"
    family_tab = st.radio(
        "Target family",
        options=("continuous", "discrete"),
        format_func=lambda k: k.title(),
        horizontal=True,
        key="bench_family_tab",
    )
    if family_tab == "continuous":
        _render_family_tab(
            dataset=dataset,
            family="continuous",
            targets=CONTINUOUS_TARGETS,
            decoder_options=UI_CONTINUOUS_DECODER_OPTIONS,
            decoder_defaults=UI_DEFAULT_CONTINUOUS_DECODERS,
            table_figures=_CONTINUOUS_TABLE_FIGURES,
            feature_sets=feature_sets,
            manifolds=manifolds,
            decode_windows=decode_windows,
            busy=busy,
            max_models=max_models,
            compare_sources=compare_sources,
            run_feature_ablation=run_feature_ablation,
            n_components=n_components,
            transform_inventory=inv,
        )
    else:
        _render_family_tab(
            dataset=dataset,
            family="discrete",
            targets=CATEGORICAL_TARGETS,
            decoder_options=UI_CATEGORICAL_DECODER_OPTIONS,
            decoder_defaults=UI_DEFAULT_CATEGORICAL_DECODERS,
            table_figures=_CATEGORICAL_TABLE_FIGURES,
            feature_sets=feature_sets,
            manifolds=manifolds,
            decode_windows=decode_windows,
            busy=busy,
            max_models=max_models,
            compare_sources=compare_sources,
            run_feature_ablation=run_feature_ablation,
            n_components=n_components,
            transform_inventory=inv,
        )

    _submit_pending_benchmark(
        dataset=dataset,
        output_dir=output_dir,
        spike_source=spike_source,
        feature_sets=feature_sets,
        manifolds=manifolds,
        decode_windows=decode_windows,
        update_dt=update_dt,
        max_models=max_models,
        run_feature_ablation=run_feature_ablation,
        compare_sources=compare_sources,
        n_components=n_components,
        include_controls=include_controls,
        region_ablation=region_ablation,
        layer_ablation=layer_ablation,
        population_ablation=population_ablation,
        no_trigger_search=no_trigger_search,
        reuse_transforms=reuse_transforms,
        slot=slot,
    )

    st.divider()
    from ui.views.decoder_diagnostics import render_decoding_diagnostics

    render_decoding_diagnostics(dataset, spike_source)

    st.divider()
    _render_diffusion_deployability(dataset)


def _render_family_tab(
    *,
    dataset: Path,
    family: str,
    targets: tuple[str, ...],
    decoder_options: tuple[str, ...],
    decoder_defaults: tuple[str, ...],
    table_figures: tuple[tuple[str, str], ...],
    feature_sets: list[str],
    manifolds: list[str],
    decode_windows: list[float],
    busy: bool,
    max_models: str,
    compare_sources: bool,
    run_feature_ablation: bool,
    n_components: list[int],
    transform_inventory: dict | None = None,
) -> None:
    st.caption("Targets: `" + "`, `".join(targets) + "`.")
    decoder_names = st.multiselect(
        "Decoders",
        options=list(decoder_options),
        default=[d for d in decoder_defaults if d in decoder_options],
        key=f"bench_decoders_{family}",
    )
    if not decoder_names:
        st.caption("Select at least one decoder.")
    workload = estimate_workload(
        feature_sets=feature_sets,
        manifolds=manifolds,
        decode_windows=decode_windows,
        max_models=max_models,
        compare_sources=compare_sources,
        run_feature_ablation=run_feature_ablation,
        n_components=n_components,
        n_decoders_hint=len(decoder_names) if decoder_names else None,
        n_targets_hint=len(targets),
    )
    inv = transform_inventory or {}
    n_requested = int(inv.get("n_requested") or 0)
    if inv.get("reuse_enabled") and n_requested > 0:
        n_reusable = int(inv.get("n_reusable") or 0)
        n_missing = int(inv.get("n_missing") or 0)
        workload = apply_remaining_work_to_workload(
            workload,
            n_wanted=n_requested,
            n_covered=n_reusable,
            n_to_compute=n_missing,
            fully_covered=False,
            hide_when_complete=False,
            scale_floor=1.0 - DECODER_TRANSFORM_TIME_SHARE,
            scale_planned=False,
            remaining_detail=(
                f"{n_missing} transform(s) to fit · {n_reusable} reusable"
                if n_reusable
                else None
            ),
        )
    render_workload_estimate(workload)
    run_clicked = st.button(
        f"Run {family.title()} Benchmark",
        type="primary",
        disabled=not decoder_names or not decode_windows or not manifolds or busy,
        key=f"bench_run_{family}",
        help="Runs in the background — you can switch pages while it works.",
    )
    if run_clicked:
        st.session_state["bench_family_tab"] = family
        state.request_action(state.KEY_BENCHMARK_REQUESTED)
        st.session_state[_PENDING_KEY] = {
            "family": family,
            "decoder_names": tuple(decoder_names),
            "targets": tuple(targets),
            "estimated_runtime_s": float(workload.get("estimated_runtime_s") or 0.0),
        }
    _render_decoding_tables(dataset, figures=table_figures)


def _submit_pending_benchmark(
    *,
    dataset: Path,
    output_dir: Path,
    spike_source: str,
    feature_sets: list[str],
    manifolds: list[str],
    decode_windows: list[float],
    update_dt: float,
    max_models: str,
    run_feature_ablation: bool,
    compare_sources: bool,
    n_components: list[int],
    include_controls: bool,
    region_ablation: bool,
    layer_ablation: bool,
    population_ablation: bool,
    no_trigger_search: bool,
    reuse_transforms: bool,
    slot: str,
) -> None:
    if not state.consume_action(state.KEY_BENCHMARK_REQUESTED):
        return
    pending = st.session_state.pop(_PENDING_KEY, None) or {}
    decoder_names = tuple(pending.get("decoder_names") or ())
    targets = tuple(pending.get("targets") or ())
    family = pending.get("family") or "decoder"
    if family in ("continuous", "discrete"):
        st.session_state["bench_family_tab"] = family
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
        manifold_n_components=tuple(n_components),
        include_controls=include_controls,
        region_ablation=region_ablation,
        layer_ablation=layer_ablation,
        population_ablation=population_ablation,
        no_trigger_search=no_trigger_search,
        reuse_transforms=bool(reuse_transforms),
        decoder_names=decoder_names,
        targets=targets,
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

    def _job_fn(*, progress_callback=None):
        try:
            result = run_benchmark(sel, progress_callback=progress_callback)
            if progress_callback is not None:
                progress_callback(
                    "Writing feature/decoder × window tables…", 1, 1,
                )
            table_paths, table_errors = _regenerate_decoding_tables(Path(sel.input_dir))
            meta.status = "completed"
            save_run_metadata(meta, output_dir)
            n_rows = (
                int(getattr(result, "shape", [0])[0])
                if hasattr(result, "shape")
                else None
            )
            return {
                "run_id": meta.run_id,
                "output_dir": str(output_dir),
                "n_metrics_rows": n_rows,
                "n_table_figures": len(table_paths),
                "table_figures": [str(p) for p in table_paths],
                "table_figure_errors": table_errors,
            }
        except Exception:
            meta.status = "failed"
            save_run_metadata(meta, output_dir)
            raise

    submit_job(
        kind="decoder_benchmark",
        label=f"{family.title()} decoder benchmark",
        fn=_job_fn,
        slot=slot,
        pass_progress=True,
        meta={
            "run_meta": {"run_id": meta.run_id, "output_dir": str(output_dir)},
            "estimated_runtime_s": pending.get("estimated_runtime_s"),
        },
    )
    st.rerun()


def _render_diffusion_deployability(dataset: Path) -> None:
    """Show diffusion_nystrom latency / landmark tradeoff when artifacts exist."""
    import pandas as pd

    metrics_paths = list(dataset.glob("decoder_comparison/**/decoder_comparison_metrics.csv"))
    bench_paths = list(dataset.glob("**/diffusion_landmark_benchmark.csv"))
    diag_pngs = list(dataset.glob("**/manifold_transforms/**/diagnostics/*.png"))
    if not metrics_paths and not bench_paths and not diag_pngs:
        return

    st.subheader("Diffusion Maps + Nyström")
    st.caption(
        "Offline eigendecomposition on landmarks; online path is a query-to-landmark "
        "kernel plus Nyström projection. Decode window length is not compute latency."
    )
    if metrics_paths:
        df = pd.read_csv(metrics_paths[0])
        mode_col = "embedding_type" if "embedding_type" in df.columns else "feature_type"
        sub = df[df[mode_col].astype(str) == "diffusion_nystrom"] if mode_col in df.columns else df.iloc[0:0]
        if not sub.empty:
            row = sub.iloc[0]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Landmarks", int(row["n_landmarks"]) if pd.notna(row.get("n_landmarks")) else "—")
            c2.metric(
                "Diffusion dims",
                int(row["manifold_n_components"]) if pd.notna(row.get("manifold_n_components")) else "—",
            )
            c3.metric("Landmark method", str(row.get("landmark_method") or "—"))
            c4.metric("local-scale k", str(row.get("local_scale_k") or "—"))
            lat_cols = st.columns(4)
            p99_emb = row.get("embedding_transform_ms")
            p99_tot = row.get("total_compute_ms")
            miss = row.get("deadline_miss_pct")
            qual = row.get("realtime_qualified", row.get("passes_realtime_gate"))
            lat_cols[0].metric("Embedding latency (ms)", f"{float(p99_emb):.3f}" if pd.notna(p99_emb) else "—")
            lat_cols[1].metric("Total compute (ms)", f"{float(p99_tot):.3f}" if pd.notna(p99_tot) else "—")
            lat_cols[2].metric("Deadline miss %", f"{float(miss):.2f}" if pd.notna(miss) else "—")
            lat_cols[3].metric("Realtime qualified", "yes" if bool(qual) else "no")

    if bench_paths:
        bdf = pd.read_csv(bench_paths[0])
        st.markdown("**Landmark count vs decoding vs latency**")
        st.dataframe(bdf, width="stretch", hide_index=True)
        png = bench_paths[0].with_name("diffusion_landmark_tradeoff.png")
        if png.exists():
            st.image(str(png), width="stretch")

    if diag_pngs:
        with st.expander("Diffusion diagnostics", expanded=False):
            for p in sorted(diag_pngs)[:8]:
                st.caption(str(p.relative_to(dataset)))
                st.image(str(p), width="stretch")
