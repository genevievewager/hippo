"""Page: Realtime Replay — recorded/simulated closed-loop inspection."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.components.controls import (
    active_spike_source,
    gated_decode_window_selector,
    metric_row,
    require_active_dataset,
)
from ui.components.plots import (
    error_over_time,
    speed_true_vs_pred,
)
from ui.components.run_status import render_job_autofresh
from ui.jobs import get_slot_job, submit_job
from ui.services.realtime import (
    CANONICAL_N_COMPONENTS,
    find_realtime_runs,
    list_cached_n_components,
    list_replay_ready_windows,
    list_ridge_diag_ready_windows,
    load_quadrant_ridge_sidecar,
    load_quadrant_sidecar,
    load_replay_artifacts,
    run_quadrant_comparison,
    run_quadrant_ridge_diagnostics,
)
from ui import state

logger = logging.getLogger(__name__)

_REPLAY_SECTIONS = ("quadrant", "ridge", "inspect")
_REPLAY_SECTION_LABELS = {
    "quadrant": "Quadrant comparison",
    "ridge": "Quadrant Comparison on Ridge",
    "inspect": "Inspect saved replay CSV",
}


def _replay_section_selector() -> str:
    if "rt_replay_section" not in st.session_state:
        st.session_state["rt_replay_section"] = "quadrant"
    return st.radio(
        "Section",
        options=_REPLAY_SECTIONS,
        format_func=lambda k: _REPLAY_SECTION_LABELS[k],
        horizontal=True,
        key="rt_replay_section",
    )


def _true_vs_decoded_time_colored(decoded: pd.DataFrame) -> go.Figure:
    """True (circles) and decoded (crosses) positions, both colored by time."""
    needed = {"true_x", "true_y", "decoded_x", "decoded_y"}
    if not needed.issubset(decoded.columns):
        return go.Figure().update_layout(title="Position columns missing in replay CSV")
    tcol = "time" if "time" in decoded.columns else (
        "decode_time" if "decode_time" in decoded.columns else None
    )
    fig = go.Figure()
    if tcol is not None:
        tvals = pd.to_numeric(decoded[tcol], errors="coerce")
        fig.add_trace(go.Scatter(
            x=decoded["true_x"], y=decoded["true_y"],
            mode="markers", name="True",
            marker=dict(
                size=6, color=tvals, colorscale="Viridis",
                colorbar=dict(title="Time (s)"),
            ),
        ))
        fig.add_trace(go.Scatter(
            x=decoded["decoded_x"], y=decoded["decoded_y"],
            mode="markers", name="Decoded",
            marker=dict(
                size=7, symbol="x", color=tvals, colorscale="Viridis",
                showscale=False,
            ),
        ))
    else:
        fig.add_trace(go.Scatter(
            x=decoded["true_x"], y=decoded["true_y"],
            mode="markers", name="True", marker=dict(size=6),
        ))
        fig.add_trace(go.Scatter(
            x=decoded["decoded_x"], y=decoded["decoded_y"],
            mode="markers", name="Decoded",
            marker=dict(size=7, symbol="x"),
        ))
    fig.update_layout(
        title="True vs decoded position colored by time",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis_title="x (cm)",
        yaxis_title="y (cm)",
    )
    return fig


def _gated_k_selector(cached_ks: list[int], *, key: str, default: int = 3) -> int | None:
    """Single k among values cached for all three quadrant embeddings."""
    st.markdown("**Components (k)**")
    cached = [int(k) for k in cached_ks]
    if not cached:
        st.caption(
            "Run **Latent Representations** first (counts × global_pca / "
            "diffusion_nystrom / global_lds × this W × k)."
        )
        return None
    missing = [str(k) for k in CANONICAL_N_COMPONENTS if k not in cached]
    if missing:
        st.caption("Not yet generated: k=" + ", ".join(missing))
    default_k = default if default in cached else cached[0]
    radio_key = f"{key}_pick"
    if radio_key not in st.session_state or st.session_state[radio_key] not in cached:
        st.session_state[radio_key] = default_k
    pick = st.radio(
        "Use k",
        options=cached,
        key=radio_key,
        horizontal=True,
        help="Enabled only when all three realtime classes have this k cached.",
    )
    return int(pick)


def render(outputs_root: Path) -> None:
    st.header("Realtime Replay")
    st.caption(
        "Closed-loop replay of the three realtime-capable representation classes. "
        "Embeddings and decoders are loaded from Latent Representations / Decoder "
        "Benchmark caches. Not coupled to live Open Ephys."
    )

    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return
    spike_source = active_spike_source(dataset, readonly=True)

    slot = "realtime:quadrant"
    existing_job = get_slot_job(slot)
    if existing_job is not None:
        job = render_job_autofresh(slot=slot)
        if job is not None and job.status == "completed":
            st.success("Quadrant comparison complete.")
        elif job is not None and job.status == "failed":
            st.error(f"Quadrant comparison failed: {job.error or 'unknown error'}")

    existing = find_realtime_runs(dataset)
    section = _replay_section_selector()

    if section == "quadrant":
        _render_quadrant_run(dataset, spike_source, slot=slot)
    elif section == "ridge":
        _render_ridge_quadrant(dataset, spike_source)
    else:
        if not existing:
            st.info("No `decoded_realtime.csv` found under this experiment yet.")
        else:
            labels = [str(p.relative_to(dataset)) for p in existing]
            idx = st.selectbox(
                "Replay run",
                options=list(range(len(existing))),
                format_func=lambda i: labels[i],
                key="rt_pick",
            )
            try:
                arts = load_replay_artifacts(existing[idx])
            except Exception as exc:
                logger.exception("Failed loading replay")
                st.error(str(exc))
                return
            _render_replay(arts)


def _render_replay(arts) -> None:
    decoded = arts.decoded
    metrics = arts.metrics or {}

    # Metric cards
    cards = {}
    if "position_error_cm" in decoded.columns:
        cards["Mean pos err (cm)"] = f"{float(decoded['position_error_cm'].mean()):.2f}"
    for key in ("mean_position_error_cm", "r2", "balanced_accuracy", "mean_predict_ms", "update_dt_s"):
        if key in metrics:
            val = metrics[key]
            cards[key] = f"{val:.3f}" if isinstance(val, float) else val
    if "decode_window_s" in decoded.columns:
        cards["Decode window"] = f"{float(decoded['decode_window_s'].iloc[0])*1000:.0f} ms"
    if "update_dt_s" in decoded.columns:
        cards["Update dt"] = f"{float(decoded['update_dt_s'].iloc[0])*1000:.0f} ms"
    metric_row(cards)

    # Scrubber
    n = len(decoded)
    idx = st.slider("Replay index", min_value=0, max_value=max(n - 1, 0), value=0, key="rt_idx")
    row = decoded.iloc[idx]
    tcol = "time" if "time" in decoded.columns else "decode_time"
    st.markdown(
        f"**t = {row.get(tcol, '—')} s** · "
        f"spikes in window = `{row.get('n_spikes_in_window', '—')}` · "
        f"active units = `{row.get('n_active_units_in_window', '—')}`"
    )
    if "position_error_cm" in decoded.columns:
        st.markdown(f"Position error: **{row['position_error_cm']:.2f} cm**")
    if {"true_x", "true_y", "decoded_x", "decoded_y"}.issubset(decoded.columns):
        st.markdown(
            f"True (x,y)=`({row['true_x']:.1f}, {row['true_y']:.1f})` · "
            f"Decoded=`({row['decoded_x']:.1f}, {row['decoded_y']:.1f})`"
        )

    t1, t2, t3, t4 = st.tabs(["Trajectory", "Error", "Speed", "Config / triggers"])
    with t1:
        st.plotly_chart(
            _true_vs_decoded_time_colored(decoded.iloc[: idx + 1]),
            width="stretch",
        )
    with t2:
        st.plotly_chart(error_over_time(decoded), width="stretch")
    with t3:
        st.plotly_chart(speed_true_vs_pred(decoded), width="stretch")
    with t4:
        if arts.selected_config:
            st.json(arts.selected_config)
        if arts.closed_loop is not None and not arts.closed_loop.empty:
            st.dataframe(arts.closed_loop.head(100), width="stretch", hide_index=True)
        if metrics:
            with st.expander("realtime_metrics.json"):
                st.json(metrics)

    # Optional mini-raster of nearby spikes is expensive; skip auto-load.
    st.caption("Neural rasters for live scrubbing can be heavy — use Neural Simulation for full rasters.")


def _render_quadrant_figures(dataset: Path) -> None:
    sidecar = load_quadrant_sidecar(dataset)
    fig_dir = dataset / "figures" / "realtime_decoding"
    stab = fig_dir / "fig_quadrant_stability.png"
    beh = fig_dir / "fig_quadrant_behavior.png"
    st.subheader("Quadrant comparison figures")
    if sidecar:
        k = sidecar.get("n_components")
        k_bit = f"k={k} · " if k is not None else ""
        d_bits = []
        for qid, label in (
            ("static_linear", "PCA"),
            ("static_nonlinear", "Nyström"),
            ("dynamic_linear", "LDS"),
        ):
            run = (sidecar.get("runs") or {}).get(qid) or {}
            src = run.get("decoder_source")
            if src == "cached":
                d_bits.append(f"{label} D=cached")
            elif src:
                d_bits.append(f"{label} D=Ridge-on-frozen-E")
        d_note = " · ".join(d_bits)
        st.caption(
            f"Target `{sidecar.get('target')}` · "
            f"W = {float(sidecar.get('decode_window_s') or 0)*1000:.0f} ms · "
            f"{k_bit}"
            "static linear = global_pca · static nonlinear = diffusion_nystrom · "
            "dynamic linear = global_lds · dynamic nonlinear = not implemented"
            + (f" · {d_note}" if d_note else "")
            + ". Position panels: true circles and decoded crosses, both colored by time."
        )
    else:
        st.caption(
            "Stability and time-colored true vs decoded / latency / accuracy "
            "appear here after a quadrant run (updated after each class)."
        )
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Stability (six metrics)**")
        if stab.exists():
            st.image(str(stab), width="stretch")
        else:
            st.caption("No `fig_quadrant_stability.png` yet.")
    with cols[1]:
        st.markdown("**Behavior · time-colored true vs decoded**")
        if beh.exists():
            st.image(str(beh), width="stretch")
        else:
            st.caption("No `fig_quadrant_behavior.png` yet.")


def _render_quadrant_run(dataset: Path, spike_source: str, *, slot: str) -> None:
    st.markdown(
        "Replay **one** behavioral target at **one** generated window across the "
        "three realtime-capable quadrants (`global_pca`, `diffusion_nystrom`, "
        "`global_lds`). Reuses Latent Representations embeddings; Decoder "
        "Benchmark heads when present. Dynamic nonlinear is left empty."
    )
    from realtime.decoder_comparison import ALL_TARGETS

    target = st.selectbox(
        "Behavior",
        options=list(ALL_TARGETS),
        index=0,
        key="rt_quad_target",
    )
    cached_windows = list_replay_ready_windows(dataset, spike_source)
    window_sel = gated_decode_window_selector(
        cached_windows,
        key="rt_quad_w",
        defaults=[0.250],
        label="Decode window",
        multiple=False,
        disabled_help=(
            "Run Decoder Benchmark for this window and the three realtime "
            "representations first."
        ),
    )
    decode_window = window_sel[0] if window_sel else None
    cached_ks: list[int] = []
    if decode_window is not None:
        cached_ks = list_cached_n_components(dataset, spike_source, float(decode_window))
    n_components = _gated_k_selector(cached_ks, key="rt_quad_k", default=3)
    update_dt = st.number_input(
        "Update dt (s)", value=0.050, min_value=0.001, step=0.025, key="rt_quad_dt",
        help="LDS internal step; independent of the observation window W.",
    )
    existing = get_slot_job(slot)
    busy = existing is not None and existing.is_active
    can_run = decode_window is not None and n_components is not None and not busy
    if st.button(
        "Run quadrant comparison",
        type="primary",
        disabled=not can_run,
        key="rt_quad_run",
    ):
        state.request_action(state.KEY_QUADRANT_REPLAY_REQUESTED)
        st.session_state["rt_replay_section"] = "quadrant"
        st.session_state["rt_quad_pending"] = {
            "target": target,
            "decode_window": decode_window,
            "update_dt": float(update_dt),
            "n_components": int(n_components) if n_components is not None else 3,
        }

    _render_quadrant_figures(dataset)

    if not state.consume_action(state.KEY_QUADRANT_REPLAY_REQUESTED):
        return
    pending = st.session_state.pop("rt_quad_pending", None) or {}
    w = pending.get("decode_window")
    if w is None:
        st.error("Generate a decode window on Feature Construction first.")
        return
    k = pending.get("n_components")
    if k is None:
        st.error("Run Latent Representations so a cached k exists for this window.")
        return

    def _job_fn(*, progress_callback=None):
        return run_quadrant_comparison(
            input_dir=dataset,
            spike_source=spike_source,
            target=str(pending.get("target") or "position"),
            decode_window=float(w),
            update_dt=float(pending.get("update_dt") or 0.050),
            manifold_n_components=int(k),
            progress_callback=progress_callback,
        )

    submit_job(
        kind="quadrant_replay",
        label="Quadrant realtime comparison",
        fn=_job_fn,
        slot=slot,
        pass_progress=True,
    )
    st.rerun()


def _render_ridge_quadrant(dataset: Path, spike_source: str) -> None:
    st.subheader("Quadrant Comparison on Ridge")
    st.markdown(
        "Ridge-only diagnostic comparing **counts** (baseline) with static linear "
        "(global PCA, region PCA), static nonlinear (diffusion + Nyström), and "
        "dynamic linear (LDS). Dynamic nonlinear is reserved for a future method."
    )
    st.markdown(
        "Ridge minimizes MSE with an L2 penalty on weights. When latent coordinates "
        "compress place information, coefficients shrink and predictions regress "
        "toward the training mean of `(x, y)` — often the arena center under "
        "uniform occupancy. **Calibration slope** `< 1` on an axis (or radially) "
        "indicates spatial shrinkage; **radial slope** `< 1` indicates "
        "center-pull. Smoother latents can still decode poorly if variance in "
        "behaviorally relevant directions is lost."
    )

    from ui.services.realtime import RIDGE_DIAG_EMBEDDINGS

    slot = "realtime:quadrant_ridge"
    existing_job = get_slot_job(slot)
    if existing_job is not None:
        job = render_job_autofresh(slot=slot)
        if job is not None and job.status == "completed":
            st.success("Ridge quadrant diagnostics complete.")
        elif job is not None and job.status == "failed":
            st.error(f"Ridge quadrant diagnostics failed: {job.error or 'unknown error'}")

    cached_windows = list_ridge_diag_ready_windows(
        dataset, spike_source, n_components=3,
    )
    window_sel = gated_decode_window_selector(
        cached_windows,
        key="rt_ridge_w",
        defaults=[0.250],
        label="Decode window",
        multiple=False,
        disabled_help=(
            "Generate this window on Feature Construction and fit all latent "
            "representations (global_pca, region_pca, diffusion_nystrom, global_lds) "
            "on Latent Representations first."
        ),
    )
    decode_window = window_sel[0] if window_sel else None
    cached_ks: list[int] = []
    if decode_window is not None:
        cached_ks = list_cached_n_components(
            dataset, spike_source, float(decode_window), embeddings=RIDGE_DIAG_EMBEDDINGS,
        )
    n_components = _gated_k_selector(cached_ks, key="rt_ridge_k", default=3)
    if decode_window is not None and n_components is not None:
        ready_at_k = list_ridge_diag_ready_windows(
            dataset, spike_source, n_components=int(n_components),
        )
        if float(decode_window) not in ready_at_k:
            st.warning(
                f"Window {decode_window*1000:.0f} ms is not ready for k={n_components} "
                "across all ridge-diagnostic embeddings."
            )
    update_dt = st.number_input(
        "Update dt (s)", value=0.050, min_value=0.001, step=0.025, key="rt_ridge_dt",
    )
    st.caption("Decoder: **ridge** (linear) · position target · train/test split 70/30.")

    existing = get_slot_job(slot)
    busy = existing is not None and existing.is_active
    ready_at_k = (
        list_ridge_diag_ready_windows(
            dataset, spike_source, n_components=int(n_components),
        )
        if n_components is not None
        else []
    )
    can_run = (
        decode_window is not None
        and n_components is not None
        and not busy
        and float(decode_window) in ready_at_k
    )
    if st.button(
        "Run ridge quadrant diagnostics",
        type="primary",
        disabled=not can_run,
        key="rt_ridge_run",
    ):
        state.request_action(state.KEY_QUADRANT_RIDGE_REQUESTED)
        st.session_state["rt_replay_section"] = "ridge"
        st.session_state["rt_ridge_pending"] = {
            "decode_window": decode_window,
            "update_dt": float(update_dt),
            "n_components": int(n_components) if n_components is not None else 3,
        }

    _render_ridge_figures(dataset)

    if not state.consume_action(state.KEY_QUADRANT_RIDGE_REQUESTED):
        return
    pending = st.session_state.pop("rt_ridge_pending", None) or {}
    w = pending.get("decode_window")
    if w is None:
        st.error("Generate a decode window on Feature Construction first.")
        return
    k = pending.get("n_components")
    if k is None:
        st.error("Run Latent Representations for all ridge-diagnostic embeddings at this W.")
        return

    def _job_fn(*, progress_callback=None):
        return run_quadrant_ridge_diagnostics(
            input_dir=dataset,
            spike_source=spike_source,
            decode_window=float(w),
            update_dt=float(pending.get("update_dt") or 0.050),
            n_components=int(k),
            progress_callback=progress_callback,
        )

    submit_job(
        kind="quadrant_ridge",
        label="Ridge quadrant center-bias diagnostics",
        fn=_job_fn,
        slot=slot,
        pass_progress=True,
    )
    st.rerun()


def _render_ridge_figures(dataset: Path) -> None:
    from visualization.figure_captions import caption_for

    sidecar = load_quadrant_ridge_sidecar(dataset)
    fig_dir = dataset / "figures" / "realtime_decoding" / "quadrant_ridge"
    traj = fig_dir / "fig_quadrant_ridge_trajectory.png"
    shrink = fig_dir / "fig_quadrant_ridge_shrinkage.png"

    if sidecar:
        st.markdown("**Interpretation**")
        st.markdown(sidecar.get("interpretation", ""))
        methods = sidecar.get("methods") or {}
        bits = []
        for mid, meta in methods.items():
            if not meta.get("available"):
                continue
            m = meta.get("metrics") or {}
            bits.append(
                f"{meta.get('label', mid)}: err={m.get('mean_position_error_cm', '—'):.2f} cm, "
                f"radial slope={m.get('slope_r', '—'):.3f}"
                if isinstance(m.get("mean_position_error_cm"), (int, float))
                else f"{meta.get('label', mid)}"
            )
        if bits:
            st.caption(" · ".join(bits))

    for n, (path, stem, title) in enumerate(
        (
            (traj, "fig_quadrant_ridge_trajectory", "Figure 1 — Trajectory and center bias"),
            (shrink, "fig_quadrant_ridge_shrinkage", "Figure 2 — Regularization and compression"),
        ),
        start=1,
    ):
        st.markdown(f"**{title}**")
        if path.exists():
            st.image(str(path), width="stretch")
            try:
                st.caption(caption_for(path, figure_number=n, figures_dir=dataset / "figures"))
            except Exception:
                st.caption(path.name)
            pdf = path.with_suffix(".pdf")
            if pdf.exists():
                st.caption(f"PDF: `{pdf.relative_to(dataset)}`")
        else:
            st.caption(f"No `{path.name}` yet — run ridge quadrant diagnostics.")

    if sidecar:
        out_dir = sidecar.get("output_dir")
        tables = sidecar.get("tables") or {}
        with st.expander("Detailed diagnostics (tables)"):
            for label, p in tables.items():
                pth = Path(p)
                if pth.exists():
                    st.markdown(f"**{label}**")
                    st.dataframe(pd.read_csv(pth), width="stretch", hide_index=True)
            if out_dir:
                st.caption(f"Artifacts: `{out_dir}`")

