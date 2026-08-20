"""Page: Live Deployment — single-target causal inference substrate."""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from ui.components.controls import require_active_dataset
from ui.services import live_deployment as svc

logger = logging.getLogger(__name__)

_SS_DECODER = "live_dep_decoder"
_SS_BUNDLE = "live_dep_bundle"
_SS_CONFIG = "live_dep_config"
_SS_SESSION = "live_dep_session"
_SS_LAST_UI = "live_dep_last_ui"
_SS_RUNNING = "live_dep_running"


def render(outputs_root: Path) -> None:
    st.header("Live Deployment")
    st.caption(
        "Select one behavioral target → load the best **deployable** F×E×D×W×C "
        "bundle → run causal inference on Replay or Live Open Ephys spikes. "
        "Neural updates and UI refresh are separate."
    )

    dataset = require_active_dataset(outputs_root)
    if dataset is None:
        return

    # ------------------------------------------------------------------ 1. Target
    st.subheader("1. Behavioral target")
    targets = svc.available_targets(dataset)
    target = st.selectbox(
        "Behavioral variable",
        options=targets,
        index=targets.index("position") if "position" in targets else 0,
        key="live_target",
    )
    metric_name, direction = svc.primary_metric_info(target)
    if metric_name:
        st.caption(f"Selection metric: `{metric_name}` ({direction}-is-better)")

    cfg = None
    cfg_err = None
    try:
        cfg = svc.select_best_config(dataset, target)
        st.session_state[_SS_CONFIG] = cfg
    except Exception as exc:  # noqa: BLE001
        cfg_err = str(exc)
        st.warning(
            f"Could not resolve best deployable configuration: {exc}. "
            "Run Decoder Benchmark (sorted spikes) and deployment selection first."
        )

    # ------------------------------------------------------------------ 2. Config
    st.subheader("2. Selected deployment configuration")
    if cfg is not None:
        st.markdown("**Best deployable configuration**")
        for line in cfg.summary_lines():
            st.text(line)
        with st.expander("Full C / extras", expanded=False):
            st.json(svc.config_display_dict(cfg))
        if not cfg.deployable:
            st.error("Selected configuration is marked non-deployable.")
        else:
            st.success("Labeled **best deployable configuration** (sorted spikes).")
    elif cfg_err:
        st.code(cfg_err)

    existing = svc.list_existing_bundles(dataset)
    if existing:
        st.caption(f"{len(existing)} existing bundle(s) under `deployment_bundles/`.")

    # ------------------------------------------------------------------ 3. Acquisition
    st.subheader("3. Acquisition / input status")
    source = st.radio(
        "Input source",
        options=["replay", "open_ephys"],
        format_func=lambda s: "Replay" if s == "replay" else "Live Open Ephys",
        horizontal=True,
        key="live_input_source",
    )
    oe_endpoint = None
    if source == "open_ephys":
        oe_endpoint = st.text_input(
            "Open Ephys endpoint (stub)",
            value="",
            placeholder="TODO: laboratory ZMQ / SpikeInterface endpoint",
            key="live_oe_endpoint",
        )
        st.info(
            "Open Ephys streaming is an isolated stub. Use **Replay** to validate "
            "the full live pathway until the lab connector is wired."
        )

    decoder = st.session_state.get(_SS_DECODER)
    if decoder is not None:
        status = decoder.status_dict()
        units = status.get("units") or {}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Expected units", units.get("expected", "—"))
        c2.metric("Mapped units", units.get("mapped", "—"))
        c3.metric("Missing units", units.get("missing", "—"))
        c4.metric("Unexpected units", units.get("unexpected", "—"))
        src = status.get("source") or "—"
        connected = status["state"] not in ("DISCONNECTED", "ERROR")
        st.write(
            f"Acquisition: {'connected' if connected else 'disconnected'} · source `{src}`"
        )
    else:
        st.caption("Connect and load a model to see unit-mapping status.")

    # ------------------------------------------------------------------ 4. Controls
    st.subheader("4. Decoder / runtime controls")
    banner = None
    if decoder is not None:
        banner = svc.mode_banner(decoder)
    if banner:
        st.warning(banner)

    pipeline_override = st.checkbox(
        "Pipeline Test Mode override (allow start with incomplete unit mapping)",
        value=False,
        key="live_pipeline_override",
        help=(
            "For engineering tests only. Does not create scientifically valid "
            "unit correspondence between simulation-trained models and live labs."
        ),
    )

    ui_hz = st.slider(
        "UI refresh rate (Hz)",
        min_value=1.0,
        max_value=10.0,
        value=5.0,
        step=1.0,
        key="live_ui_hz",
        help="Neural inference cadence comes from the bundle (≈ 25–50 ms), not this slider.",
    )
    steps_per_refresh = st.number_input(
        "Inference steps per UI refresh",
        min_value=1,
        max_value=80,
        value=8,
        key="live_steps_per_ui",
    )
    max_replay_s = st.number_input(
        "Replay batch duration per refresh (s)",
        min_value=0.1,
        max_value=30.0,
        value=2.0,
        step=0.5,
        key="live_replay_batch_s",
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        do_connect = st.button("Connect", key="live_btn_connect")
    with b2:
        do_load = st.button(
            "Load Best Model",
            key="live_btn_load",
            disabled=cfg is None,
        )
    with b3:
        can_start = False
        if decoder is not None:
            decoder.set_pipeline_test_override(pipeline_override)
            can_start = decoder.can_start() and not st.session_state.get(_SS_RUNNING)
        do_start = st.button(
            "Start Live Decoding",
            key="live_btn_start",
            disabled=not can_start,
        )
    with b4:
        do_stop = st.button("Stop", key="live_btn_stop")

    if do_load and cfg is not None:
        try:
            with st.spinner("Packing / loading deployment bundle…"):
                dec, bundle_path = svc.build_live_decoder_from_experiment(
                    dataset, target, force_rebuild=False,
                )
            st.session_state[_SS_DECODER] = dec
            st.session_state[_SS_BUNDLE] = str(bundle_path)
            st.session_state[_SS_RUNNING] = False
            st.success(f"Loaded bundle `{bundle_path.name}`")
            decoder = dec
        except Exception as exc:  # noqa: BLE001
            logger.exception("Load best model failed")
            st.error(str(exc))

    if do_connect:
        decoder = st.session_state.get(_SS_DECODER)
        if decoder is None:
            st.error("Load Best Model before Connect.")
        else:
            try:
                stream = svc.make_stream(
                    source, dataset, endpoint=oe_endpoint or None,
                )
                decoder.connect(stream)
                decoder.set_pipeline_test_override(pipeline_override)
                st.session_state[_SS_DECODER] = decoder
                st.success(f"Connected: {stream.source_name}")
            except NotImplementedError as exc:
                st.error(str(exc))
            except Exception as exc:  # noqa: BLE001
                logger.exception("Connect failed")
                st.error(str(exc))

    if do_start:
        decoder = st.session_state.get(_SS_DECODER)
        if decoder is None:
            st.error("No decoder loaded.")
        else:
            try:
                decoder.set_pipeline_test_override(pipeline_override)
                session = svc.create_session_logger(dataset)
                decoder.start(session_logger=session)
                st.session_state[_SS_SESSION] = str(session.session_dir)
                st.session_state[_SS_RUNNING] = True
                st.session_state[_SS_LAST_UI] = time.time()
                st.success(f"Running · logging → `{session.session_dir.name}`")
            except Exception as exc:  # noqa: BLE001
                logger.exception("Start failed")
                st.error(str(exc))

    if do_stop:
        decoder = st.session_state.get(_SS_DECODER)
        if decoder is not None:
            decoder.stop()
        st.session_state[_SS_RUNNING] = False
        st.info("Stopped.")

    decoder = st.session_state.get(_SS_DECODER)
    if decoder is not None:
        status = decoder.status_dict()
        st.markdown("##### Status")
        cols = st.columns(3)
        with cols[0]:
            st.markdown("**Acquisition**")
            st.write(f"Source: `{status.get('source')}`")
            st.write(f"Spike rate (est.): {status.get('spike_rate_hz', 0):.1f} Hz")
            u = status.get("units") or {}
            st.write(
                f"Units expected/mapped/missing/unexpected: "
                f"{u.get('expected')}/{u.get('mapped')}/{u.get('missing')}/{u.get('unexpected')}"
            )
        with cols[1]:
            st.markdown("**Decoder**")
            st.write(f"Target: `{status.get('target')}`")
            st.write(f"Update interval: {1000 * float(status.get('update_dt_s', 0)):.0f} ms")
            st.write(f"Window: {1000 * float(status.get('decode_window_s', 0)):.0f} ms")
            st.write(f"Status: **{status.get('state')}**")
            st.write(f"Mode: `{status.get('mode')}`")
            st.write(f"Model: `{status.get('model_id')}`")
        with cols[2]:
            st.markdown("**Performance**")
            inf = status.get("inference_latency_ms")
            loop = status.get("mean_loop_latency_ms")
            st.write(
                f"Inference latency: {inf:.2f} ms"
                if inf is not None
                else "Inference latency: —"
            )
            st.write(
                f"Mean loop latency: {loop:.2f} ms"
                if loop is not None
                else "Mean loop latency: —"
            )
            st.write(f"Samples processed: {status.get('samples_processed', 0):,}")
            st.write(f"Dropped / late updates: {status.get('dropped_updates', 0):,}")
            budget = float(status.get("update_dt_s") or 0.025) * 1000.0
            if loop is not None and loop > budget:
                st.error(
                    f"Realtime overrun: mean loop {loop:.1f} ms > budget {budget:.1f} ms"
                )

        session_path = st.session_state.get(_SS_SESSION)
        if session_path:
            st.caption(f"Session log: `{session_path}`")

    # Drive inference when running (neural cadence decoupled from UI Hz).
    if st.session_state.get(_SS_RUNNING) and decoder is not None:
        try:
            from realtime.live.spike_stream import ReplaySpikeStream

            n_steps = int(steps_per_refresh)
            max_by_time = max(1, int(float(max_replay_s) / max(decoder.update_dt_s, 1e-3)))
            n_steps = min(n_steps, max_by_time)
            for _ in range(n_steps):
                if decoder.state.value != "RUNNING":
                    break
                if isinstance(decoder._stream, ReplaySpikeStream):
                    if (
                        decoder._t_cursor is not None
                        and decoder._t_cursor > decoder._stream.t_end
                    ):
                        decoder.stop()
                        st.session_state[_SS_RUNNING] = False
                        st.info("Replay finished.")
                        break
                decoder.step()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Live step failed")
            st.error(str(exc))
            st.session_state[_SS_RUNNING] = False

        st.session_state[_SS_LAST_UI] = time.time()
        time.sleep(max(0.05, 1.0 / float(ui_hz)))
        st.rerun()

    # ------------------------------------------------------------------ 5. Visualization
    st.subheader("5. Live prediction visualization")
    decoder = st.session_state.get(_SS_DECODER)
    if decoder is None or not decoder.history:
        st.caption("Start decoding to see live predictions for the selected target.")
        return

    df = svc.history_to_frame(decoder)
    latest = decoder.latest_prediction
    _render_target_viz(decoder.target, df, latest)


def _render_target_viz(target: str, df: pd.DataFrame, latest) -> None:
    tail_n = min(len(df), 400)
    tail = df.tail(tail_n)

    if target == "position":
        if latest is not None:
            pred = latest.prediction if isinstance(latest.prediction, dict) else {}
            x = pred.get("x", latest.flags.get("decoded_x"))
            y = pred.get("y", latest.flags.get("decoded_y"))
            st.markdown("**Current decoded position**")
            c1, c2 = st.columns(2)
            c1.metric("X (cm)", f"{float(x):.1f}" if x is not None else "—")
            c2.metric("Y (cm)", f"{float(y):.1f}" if y is not None else "—")
        if {"decoded_x", "decoded_y"}.issubset(tail.columns) or {"x", "y"}.issubset(
            tail.columns
        ):
            plot = tail.copy()
            if "decoded_x" not in plot.columns and "x" in plot.columns:
                plot = plot.rename(columns={"x": "decoded_x", "y": "decoded_y"})
            st.scatter_chart(plot, x="decoded_x", y="decoded_y")
        return

    if target == "speed":
        if latest is not None:
            val = latest.flags.get("decoded_speed", latest.prediction)
            st.metric(
                "Decoded speed", f"{float(val):.3f}" if val is not None else "—"
            )
        col = "decoded_speed" if "decoded_speed" in tail.columns else None
        if col:
            st.line_chart(tail.set_index("time")[[col]])
        return

    if target == "acceleration":
        if latest is not None:
            val = latest.flags.get("decoded_acceleration", latest.prediction)
            st.metric(
                "Decoded acceleration",
                f"{float(val):.3f}" if val is not None else "—",
            )
        col = "decoded_acceleration" if "decoded_acceleration" in tail.columns else None
        if col:
            st.line_chart(tail.set_index("time")[[col]])
        return

    if target == "head_direction":
        deg = None
        if latest is not None:
            deg = latest.flags.get("decoded_head_direction_deg", latest.prediction)
            st.metric(
                "Head direction (deg)",
                f"{float(deg):.1f}" if deg is not None else "—",
            )
        if deg is not None:
            rad = math.radians(float(deg))
            st.caption(f"Heading vector ≈ ({math.cos(rad):.2f}, {math.sin(rad):.2f})")
        col = (
            "decoded_head_direction_deg"
            if "decoded_head_direction_deg" in tail.columns
            else None
        )
        if col:
            st.line_chart(tail.set_index("time")[[col]])
        return

    if latest is not None:
        cls = latest.flags.get("predicted_class", latest.prediction)
        st.metric("Predicted class", str(cls))
        probs = latest.flags.get("class_probabilities")
        if isinstance(probs, dict) and probs:
            st.bar_chart(pd.Series(probs, name="probability"))
    pred_col = None
    for c in ("predicted_class", "prediction", f"decoded_{target}"):
        if c in tail.columns:
            pred_col = c
            break
    if pred_col:
        hist = tail[["time", pred_col]].copy()
        codes, uniques = pd.factorize(hist[pred_col].astype(str))
        hist["class_code"] = codes
        st.line_chart(hist.set_index("time")[["class_code"]])
        st.caption(
            "Class codes: " + ", ".join(f"{i}={u}" for i, u in enumerate(uniques))
        )
