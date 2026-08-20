"""Decoding Diagnostics panels for Decoder Benchmark (Plotly + export).

Imported only by ``ui.views.decoder_benchmark``. Traces are offline held-out
test predictions, never realtime replay CSVs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from realtime.decoding_diagnostics_prep import (
    arena_limits_from_trace,
    circular_error_by_angle,
    class_labels_from_meta,
    compression_slope,
    confusion_from_trace,
    downsample_for_display,
    header_metrics_from_row,
    magnitude_bins,
    metric_delta,
    predicted_class_probability,
    radial_shrinkage,
    recall_by_true_class,
    sparse_link_indices,
    spatial_error_map,
    wrap_head_direction_series,
)
from ui.services.decoding_diagnostics import (
    LEGACY_TRACE_MESSAGE,
    TARGET_DISPLAY_NAMES,
    any_prediction_traces,
    available_targets,
    best_config_row,
    config_label,
    counts_baseline_row,
    load_comparison_metrics,
    load_config_pair,
    load_prediction_trace,
    row_by_config_id,
    same_decoder_other_embedding,
    same_embedding_other_decoder,
    target_configs,
)
from ui.services.representations import (
    REPRESENTATION_QUADRANT_LABELS,
    REPRESENTATION_QUADRANTS,
)
from visualization.publication_style import PRED_B_COLOR, PRED_COLOR, TRUE_COLOR

_SPLIT_NOTE = "Offline held-out test — not realtime causal replay."
_LEADER_COLS = (
    "config_id",
    "embedding_type",
    "decoder_name",
    "feature_set",
    "decode_window_s",
    "manifold_n_components",
    "spike_source",
    "primary_metric",
)


def _plotly_layout(fig: go.Figure, *, height: int = 520, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height,
        title=title,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=56, r=48, t=64, b=48),
        template="plotly_white",
    )
    return fig


def _quadrant_label(embedding: str) -> str:
    for qid, names in REPRESENTATION_QUADRANTS.items():
        if embedding in names:
            return REPRESENTATION_QUADRANT_LABELS.get(qid, qid)
    if embedding in ("counts", "identity"):
        return REPRESENTATION_QUADRANT_LABELS.get("static_linear", "Static linear")
    return ""


def _metric_pills(row: pd.Series, target: str, *, label: str) -> None:
    values = header_metrics_from_row(row, target)
    cols = st.columns(max(len(values), 1))
    for col, (name, val) in zip(cols, values.items()):
        pretty = name.replace("_", " ")
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            col.metric(f"{label} · {pretty}", "—")
        else:
            col.metric(f"{label} · {pretty}", f"{val:.3f}")


def _leaderboard_view(sub: pd.DataFrame, target: str) -> pd.DataFrame:
    cols = [c for c in _LEADER_COLS if c in sub.columns]
    metric_key = None
    from realtime.decoder_comparison import PRIMARY_METRIC

    if target in PRIMARY_METRIC:
        metric_key = PRIMARY_METRIC[target][0]
        if metric_key in sub.columns and metric_key not in cols:
            cols.append(metric_key)
    extra = [c for c in ("r2", "mae", "rmse", "balanced_accuracy", "macro_f1", "accuracy") if c in sub.columns and c not in cols]
    view = sub[cols + extra].copy()
    if "embedding_type" in view.columns:
        view.insert(
            1,
            "representation_class",
            view["embedding_type"].astype(str).map(_quadrant_label),
        )
    return view


def _select_config_id_from_table(view: pd.DataFrame, key: str) -> str | None:
    if view.empty or "config_id" not in view.columns:
        return None
    labels = [f"{config_label(r)}  [{r['config_id'][-10:]}]" for _, r in view.iterrows()]
    ids = view["config_id"].astype(str).tolist()
    choice = st.selectbox("Configuration", options=list(range(len(ids))), format_func=lambda i: labels[i], key=key)
    return ids[int(choice)]


def _position_figure(frame: pd.DataFrame, frame_b: pd.DataFrame | None, aligned: bool) -> go.Figure:
    disp = downsample_for_display(frame)
    x0, x1, y0, y1 = arena_limits_from_trace(frame if frame_b is None else pd.concat([frame, frame_b], ignore_index=True))
    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 3}, None, None],
            [{"type": "heatmap"}, {"type": "xy"}, {"type": "heatmap"}],
        ],
        subplot_titles=(
            "A  True trajectory", "B  Predicted trajectory", "C  True → predicted (sparse)",
            "D  Position error through time",
            "E  Spatial error", "F  Radial shrinkage", "Δ local error" if frame_b is not None else "",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.08,
    )
    t = disp["time"]
    fig.add_trace(go.Scatter(
        x=disp["true_x"], y=disp["true_y"], mode="markers",
        marker=dict(size=5, color=t, colorscale="Viridis", symbol="circle", colorbar=dict(title="Time (s)", x=0.32, len=0.28, y=0.88)),
        name="True",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=disp["pred_x"], y=disp["pred_y"], mode="markers",
        marker=dict(size=5, color=t, colorscale="Viridis", symbol="x"),
        name="Predicted", showlegend=False,
    ), row=1, col=2)
    idx = sparse_link_indices(len(disp))
    for i in idx:
        fig.add_trace(go.Scatter(
            x=[disp["true_x"].iloc[i], disp["pred_x"].iloc[i]],
            y=[disp["true_y"].iloc[i], disp["pred_y"].iloc[i]],
            mode="lines", line=dict(color="rgba(80,80,80,0.25)", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=disp["true_x"].iloc[idx], y=disp["true_y"].iloc[idx], mode="markers",
        marker=dict(size=6, color=TRUE_COLOR, symbol="circle"), name="True", showlegend=False,
    ), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=disp["pred_x"].iloc[idx], y=disp["pred_y"].iloc[idx], mode="markers",
        marker=dict(size=6, color=PRED_COLOR, symbol="x"), name="Predicted", showlegend=False,
    ), row=1, col=3)

    fig.add_trace(go.Scatter(
        x=frame["time"], y=frame["error_cm"], mode="lines",
        line=dict(color=PRED_COLOR, width=1.2), name="error_cm",
    ), row=2, col=1)
    med = float(np.nanmedian(frame["error_cm"]))
    p90 = float(np.nanpercentile(frame["error_cm"], 90))
    fig.add_hline(y=med, line_dash="dash", line_color="#333", annotation_text=f"median {med:.1f}", row=2, col=1)
    fig.add_hline(y=p90, line_dash="dot", line_color="#666", annotation_text=f"P90 {p90:.1f}", row=2, col=1)

    sm = spatial_error_map(frame, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
    fig.add_trace(go.Heatmap(
        z=sm["mean_error"], x=sm["x_edges"], y=sm["y_edges"],
        colorscale="YlOrRd", colorbar=dict(title="cm", x=0.30, len=0.28, y=0.16),
        hoverongaps=False,
    ), row=3, col=1)

    rad = radial_shrinkage(frame, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
    fig.add_trace(go.Scatter(
        x=rad["r_true"], y=rad["r_pred"], mode="markers",
        marker=dict(size=4, color=TRUE_COLOR, opacity=0.35), name="radius", showlegend=False,
    ), row=3, col=2)
    hi = float(np.nanmax([np.nanmax(rad["r_true"]), np.nanmax(rad["r_pred"]), 1.0]))
    fig.add_trace(go.Scatter(
        x=[0, hi], y=[0, hi], mode="lines", line=dict(color="#888", dash="dash"),
        name="identity", showlegend=False,
    ), row=3, col=2)

    if frame_b is not None and aligned:
        tmp = frame.copy()
        tmp["error_cm"] = np.asarray(frame["error_cm"], dtype=float) - np.asarray(frame_b["error_cm"], dtype=float)
        dm = spatial_error_map(tmp, x_min=x0, x_max=x1, y_min=y0, y_max=y1)
        vmax = np.nanmax(np.abs(dm["mean_error"]))
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        fig.add_trace(go.Heatmap(
            z=dm["mean_error"], x=dm["x_edges"], y=dm["y_edges"],
            colorscale="RdBu_r", zmin=-vmax, zmax=vmax,
            colorbar=dict(title="Δ cm", x=1.0, len=0.28, y=0.16),
        ), row=3, col=3)

    for r, c in ((1, 1), (1, 2), (1, 3), (3, 1), (3, 3)):
        fig.update_xaxes(range=[x0, x1], scaleanchor=f"y{'' if (r, c) == (1, 1) else ''}", scaleratio=1, row=r, col=c)
        fig.update_yaxes(range=[y0, y1], row=r, col=c)
    fig.update_xaxes(title_text="Time (s)", row=2, col=1)
    fig.update_yaxes(title_text="Error (cm)", row=2, col=1)
    fig.update_xaxes(title_text="True radius (cm)", row=3, col=2)
    fig.update_yaxes(title_text="Predicted radius (cm)", row=3, col=2)
    return _plotly_layout(fig, height=980, title="Position diagnostics")


def _scalar_figure(frame: pd.DataFrame, ylabel: str, frame_b: pd.DataFrame | None, aligned: bool) -> go.Figure:
    disp = downsample_for_display(frame)
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "A  True and predicted", "B  Residual (pred − true)",
            "C  Predicted vs actual", "D  Residual vs actual",
            "E  Bias by true magnitude", "F  |error| difference vs true" if frame_b is not None else "",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Scatter(x=disp["time"], y=disp["true"], name="True", line=dict(color=TRUE_COLOR, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp["time"], y=disp["pred"], name="Predicted", line=dict(color=PRED_COLOR, width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp["time"], y=disp["residual"], name="residual", line=dict(color=PRED_COLOR, width=1), showlegend=False), row=1, col=2)
    fig.add_hline(y=0, line_color="#444", row=1, col=2)
    t_all, p_all = frame["true"], frame["pred"]
    fig.add_trace(go.Scatter(x=t_all, y=p_all, mode="markers", marker=dict(size=4, color=PRED_COLOR, opacity=0.35), showlegend=False), row=2, col=1)
    lo = float(np.nanmin([t_all.min(), p_all.min()]))
    hi = float(np.nanmax([t_all.max(), p_all.max()]))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="#888", dash="dash"), name="identity", showlegend=False), row=2, col=1)
    slope = compression_slope(t_all, p_all)
    fig.add_trace(go.Scatter(x=t_all, y=frame["residual"], mode="markers", marker=dict(size=4, color=PRED_COLOR, opacity=0.35), showlegend=False), row=2, col=2)
    fig.add_hline(y=0, line_color="#444", row=2, col=2)
    bins = magnitude_bins(t_all, p_all)
    if not bins.empty:
        fig.add_trace(go.Scatter(x=bins["mean_true"], y=bins["mean_pred"], mode="lines+markers", line=dict(color=PRED_COLOR), name="bin mean pred", showlegend=False), row=3, col=1)
        fig.add_trace(go.Scatter(x=bins["mean_true"], y=bins["mean_true"], mode="lines", line=dict(color=TRUE_COLOR, dash="dash"), showlegend=False), row=3, col=1)
    if frame_b is not None and aligned:
        ea = np.abs(np.asarray(frame["residual"], dtype=float))
        eb = np.abs(np.asarray(frame_b["residual"], dtype=float))
        fig.add_trace(go.Scatter(x=frame["true"], y=ea - eb, mode="markers", marker=dict(size=4, color=PRED_B_COLOR, opacity=0.4), showlegend=False), row=3, col=2)
        fig.add_hline(y=0, line_color="#444", row=3, col=2)
    fig.update_yaxes(title_text=ylabel, row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_xaxes(title_text="True", row=2, col=1)
    fig.update_yaxes(title_text="Predicted", row=2, col=1)
    fig.update_annotations(font_size=12)
    fig.add_annotation(
        text=f"pred ~ true slope = {slope['slope']:.2f} (1 = no compression)",
        xref="paper", yref="paper", x=0.25, y=0.52, showarrow=False, font=dict(size=11),
    )
    return _plotly_layout(fig, height=900, title="Scalar diagnostics")


def _hd_figure(frame: pd.DataFrame, frame_b: pd.DataFrame | None, aligned: bool) -> go.Figure:
    disp = downsample_for_display(frame)
    tw, pw = wrap_head_direction_series(disp["true_deg"], disp["pred_deg"])
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "A  True vs predicted (unwrapped to true)",
            "B  Circular error (shortest arc)",
            "C  Predicted vs true (mod 360°)",
            "D  Error by true angle",
        ),
        vertical_spacing=0.14, horizontal_spacing=0.10,
    )
    fig.add_trace(go.Scatter(x=disp["time"], y=tw, name="True", line=dict(color=TRUE_COLOR, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp["time"], y=pw, name="Predicted", line=dict(color=PRED_COLOR, width=2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=disp["time"], y=disp["circular_error_deg"], name="circ. err", line=dict(color=PRED_COLOR, width=1), showlegend=False), row=1, col=2)
    if frame_b is not None and aligned:
        diff = np.asarray(frame["circular_error_deg"], dtype=float) - np.asarray(frame_b["circular_error_deg"], dtype=float)
        fig.add_trace(go.Scatter(x=frame["time"], y=diff, name="A−B", line=dict(color=PRED_B_COLOR, width=1)), row=1, col=2)
    fig.add_trace(go.Scatter(
        x=np.asarray(frame["true_deg"]) % 360.0, y=np.asarray(frame["pred_deg"]) % 360.0,
        mode="markers", marker=dict(size=4, color=PRED_COLOR, opacity=0.35), showlegend=False,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, 360], y=[0, 360], mode="lines", line=dict(color="#888", dash="dash"), showlegend=False), row=2, col=1)
    by_ang = circular_error_by_angle(frame["true_deg"], frame["circular_error_deg"])
    if not by_ang.empty:
        fig.add_trace(go.Bar(x=by_ang["bin_center_deg"], y=by_ang["mean_circular_error_deg"], marker_color=PRED_COLOR, showlegend=False), row=2, col=2)
    fig.update_yaxes(range=[0, 180], row=1, col=2)
    fig.update_xaxes(range=[0, 360], row=2, col=1)
    fig.update_yaxes(range=[0, 360], row=2, col=1)
    return _plotly_layout(fig, height=780, title="Head-direction diagnostics")


def _categorical_figure(
    frame: pd.DataFrame,
    class_labels: list[str],
    frame_b: pd.DataFrame | None,
    aligned: bool,
) -> go.Figure:
    labels = class_labels or sorted(set(frame["true"].astype(str)) | set(frame["pred"].astype(str)))
    lab_to_i = {lab: i for i, lab in enumerate(labels)}
    t_idx = [lab_to_i.get(str(v), -1) for v in frame["true"]]
    p_idx = [lab_to_i.get(str(v), -1) for v in frame["pred"]]
    times = np.asarray(frame["time"], dtype=float)
    correct = frame["true"].astype(str).to_numpy() == frame["pred"].astype(str).to_numpy()
    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "heatmap"}, {"type": "heatmap"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "A  True labels", "B  Predicted labels",
            "C  Incorrect (1)", "D  Confusion (this configuration)",
            "E  P(predicted class)", "F  Recall by true class",
        ),
        vertical_spacing=0.10, horizontal_spacing=0.10,
    )
    zmax = max(len(labels) - 1, 1)
    fig.add_trace(go.Heatmap(z=[t_idx], x=times, y=["true"], colorscale="Turbo", zmin=0, zmax=zmax, showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=[p_idx], x=times, y=["pred"], colorscale="Turbo", zmin=0, zmax=zmax, showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=[(~correct).astype(float)], x=times, y=["err"], colorscale="Greys", zmin=0, zmax=1, showscale=False), row=2, col=1)
    mat, labs = confusion_from_trace(frame, class_labels=labels)
    fig.add_trace(go.Heatmap(z=mat, x=labs, y=labs, colorscale="Blues", colorbar=dict(title="count", len=0.28, y=0.5)), row=2, col=2)
    proba = predicted_class_probability(frame, labels)
    if proba is not None:
        fig.add_trace(go.Scatter(x=times, y=proba, line=dict(color=PRED_COLOR, width=1.2), name="P(pred)", showlegend=False), row=3, col=1)
        fig.update_yaxes(range=[0, 1.05], row=3, col=1)
    rec = recall_by_true_class(frame, class_labels=labels)
    fig.add_trace(go.Bar(x=rec["class"], y=rec["recall"], marker_color=PRED_COLOR, showlegend=False), row=3, col=2)
    fig.update_yaxes(range=[0, 1.05], row=3, col=2)
    if frame_b is not None and aligned:
        ok_a = frame["true"].astype(str) == frame["pred"].astype(str)
        ok_b = frame_b["true"].astype(str) == frame_b["pred"].astype(str)
        only_a = ok_a & ~ok_b
        fig.add_trace(go.Scatter(
            x=times[only_a.to_numpy()], y=np.ones(int(only_a.sum())) * 0.9,
            mode="markers", marker=dict(symbol="line-ns", color=PRED_COLOR, size=8),
            name="A only correct",
        ), row=2, col=1)
    return _plotly_layout(fig, height=900, title="Categorical diagnostics")


def render_decoding_diagnostics(dataset: Path, spike_source: str) -> None:
    st.header("Decoding Diagnostics")
    st.caption(_SPLIT_NOTE)

    metrics = load_comparison_metrics(dataset, spike_source=spike_source)
    if metrics.empty:
        st.info("No decoder comparison metrics on disk yet. Run a benchmark first.")
        return

    traces_exist = any_prediction_traces(dataset, spike_source=spike_source)
    if not traces_exist:
        st.warning(LEGACY_TRACE_MESSAGE)

    targets = available_targets(metrics)
    if not targets:
        st.info("No scored targets in the comparison table.")
        return
    display = [TARGET_DISPLAY_NAMES.get(t, t) for t in targets]
    choice = st.selectbox(
        "Behavioral target",
        options=list(range(len(targets))),
        format_func=lambda i: display[i],
        key="diag_target",
    )
    target = targets[int(choice)]
    sub = target_configs(metrics, target)
    if sub.empty:
        st.info(f"No successful evaluations for `{target}`.")
        return

    view = _leaderboard_view(sub, target)
    st.subheader("Configurations")
    st.caption("Select a leaderboard row, then optionally compare against a second configuration.")
    try:
        event = st.dataframe(
            view,
            width="stretch",
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="diag_table",
        )
        rows = getattr(getattr(event, "selection", None), "rows", None) or []
        selected_idx = int(rows[0]) if rows else None
    except TypeError:
        st.dataframe(view, width="stretch", hide_index=True)
        selected_idx = None
    if selected_idx is None:
        cid_a = _select_config_id_from_table(view, "diag_cfg_a")
    else:
        cid_a = str(view.iloc[selected_idx]["config_id"])
        st.caption(f"Selected: `{config_label(sub[sub['config_id'].astype(str) == cid_a].iloc[0])}`")

    row_a = row_by_config_id(sub, cid_a) if cid_a else None
    if row_a is None:
        return

    compare_mode = st.radio(
        "Compare against",
        options=("none", "best", "same_decoder", "same_embedding", "counts", "manual"),
        format_func=lambda k: {
            "none": "One configuration",
            "best": "Best on this target",
            "same_decoder": "Same decoder, different representation",
            "same_embedding": "Same representation, different decoder",
            "counts": "Counts / identity baseline",
            "manual": "Manually selected row",
        }[k],
        horizontal=True,
        key="diag_compare_mode",
    )

    row_b = None
    cid_b = None
    if compare_mode == "best":
        row_b = best_config_row(sub, target)
    elif compare_mode == "same_decoder":
        alts = same_decoder_other_embedding(sub, row_a)
        if alts.empty:
            st.caption("No other representation with this decoder, feature set, and window.")
        else:
            cid_b = _select_config_id_from_table(_leaderboard_view(alts, target), "diag_cfg_b_sd")
            row_b = row_by_config_id(alts, cid_b) if cid_b else None
    elif compare_mode == "same_embedding":
        alts = same_embedding_other_decoder(sub, row_a)
        if alts.empty:
            st.caption("No other decoder with this representation, feature set, and window.")
        else:
            cid_b = _select_config_id_from_table(_leaderboard_view(alts, target), "diag_cfg_b_se")
            row_b = row_by_config_id(alts, cid_b) if cid_b else None
    elif compare_mode == "counts":
        row_b = counts_baseline_row(
            sub, target=target,
            decoder_name=str(row_a.get("decoder_name")),
            decode_window_s=float(row_a["decode_window_s"]) if "decode_window_s" in row_a else None,
        )
        if row_b is None:
            st.caption("No counts/identity baseline at this target.")
    elif compare_mode == "manual":
        others = sub[sub["config_id"].astype(str) != str(cid_a)]
        if others.empty:
            st.caption("No second row available.")
        else:
            cid_b = _select_config_id_from_table(_leaderboard_view(others, target), "diag_cfg_b_man")
            row_b = row_by_config_id(others, cid_b) if cid_b else None

    if row_b is not None and str(row_b.get("config_id")) == str(cid_a):
        row_b = None

    _metric_pills(row_a, target, label="A")
    if row_b is not None:
        _metric_pills(row_b, target, label="B")
        delta = metric_delta(header_metrics_from_row(row_a, target), header_metrics_from_row(row_b, target))
        cols = st.columns(max(len(delta), 1))
        for col, (name, val) in zip(cols, delta.items()):
            col.metric(f"Δ {name.replace('_', ' ')}", "—" if val is None else f"{val:.3f}")

    if not traces_exist:
        return

    spike = str(row_a.get("spike_source") or spike_source)
    bundle_a = load_prediction_trace(dataset, str(cid_a), metrics_row=row_a, spike_source=spike)
    if bundle_a is None:
        st.warning(LEGACY_TRACE_MESSAGE)
        return

    bundle_b = None
    alignment = None
    if row_b is not None:
        cid_b = str(row_b["config_id"])
        bundle_a, bundle_b, alignment = load_config_pair(
            dataset, str(cid_a), cid_b, row_a=row_a, row_b=row_b, spike_source=spike,
        )
        if bundle_b is None:
            st.caption("Second configuration has no prediction trace.")
        elif alignment is not None and not alignment.aligned:
            st.info(alignment.message)

    aligned = bool(alignment.aligned) if alignment is not None else False
    frame_b_for_diff = bundle_b.frame if bundle_b is not None and aligned else None

    family = bundle_a.family
    if family == "position":
        st.plotly_chart(_position_figure(bundle_a.frame, frame_b_for_diff, aligned), width="stretch")
    elif family == "head_direction":
        st.plotly_chart(_hd_figure(bundle_a.frame, frame_b_for_diff, aligned), width="stretch")
    elif family == "categorical":
        labels = class_labels_from_meta(bundle_a.meta)
        st.plotly_chart(
            _categorical_figure(bundle_a.frame, labels, frame_b_for_diff, aligned),
            width="stretch",
        )
    else:
        st.plotly_chart(
            _scalar_figure(bundle_a.frame, target.replace("_", " "), frame_b_for_diff, aligned),
            width="stretch",
        )

    c1, c2 = st.columns(2)
    with c1:
        export_diag = st.button("Export publication diagnostics PNG", key="diag_export")
    with c2:
        export_vs = st.button(
            "Export “where decoding succeeds and fails” PNG",
            key="diag_export_vs",
            disabled=bundle_b is None,
        )
    if export_diag or export_vs:
        from visualization.publication_decoding_diagnostics import export_diagnostics_figure

        path = export_diagnostics_figure(
            experiment_dir=dataset,
            target=target,
            frame_a=bundle_a.frame,
            meta_a=bundle_a.meta,
            frame_b=bundle_b.frame if bundle_b is not None else None,
            meta_b=bundle_b.meta if bundle_b is not None else None,
            label_a=config_label(row_a),
            label_b=config_label(row_b) if row_b is not None else "B",
            where_succeeds_fails=bool(export_vs and bundle_b is not None),
        )
        st.success(f"Wrote `{path}`")
        st.image(str(path), width="stretch")
