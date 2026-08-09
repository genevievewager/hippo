"""Plotly figure builders for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def trajectory_xy(behavior: pd.DataFrame, *, color: str | None = None) -> go.Figure:
    df = behavior.copy()
    if "time" not in df.columns and df.columns[0].lower().startswith("t"):
        df = df.rename(columns={df.columns[0]: "time"})
    color = color or ("time" if "time" in df.columns else None)
    fig = px.scatter(
        df, x="x", y="y", color=color, color_continuous_scale="Viridis",
        title="Animal trajectory",
    )
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=50, b=40))
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def speed_over_time(behavior: pd.DataFrame) -> go.Figure:
    df = behavior.copy()
    tcol = "time" if "time" in df.columns else df.columns[0]
    speed_col = "speed" if "speed" in df.columns else None
    if speed_col is None:
        return go.Figure().update_layout(title="Speed column not available")
    fig = px.line(df, x=tcol, y=speed_col, title="Speed over time")
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def neural_raster(
    spikes: pd.DataFrame,
    *,
    units_df: pd.DataFrame | None = None,
    max_spikes: int = 8000,
    title: str = "Neural raster",
) -> go.Figure:
    df = spikes.copy()
    if len(df) > max_spikes:
        df = df.sample(max_spikes, random_state=0).sort_values("time")
    color = None
    if units_df is not None and "unit_id" in units_df.columns:
        region_col = next(
            (c for c in ("region", "subfield", "anatomy_region") if c in units_df.columns),
            None,
        )
        if region_col:
            merged = df.merge(
                units_df[["unit_id", region_col]].drop_duplicates("unit_id"),
                on="unit_id",
                how="left",
            )
            color = region_col
            df = merged
    fig = px.scatter(
        df, x="time", y="unit_id", color=color,
        title=title, opacity=0.65,
    )
    fig.update_traces(marker=dict(size=3))
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def feature_traces(traces: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    t = traces["time_s"] if "time_s" in traces.columns else traces.index
    for col in traces.columns:
        if col == "time_s":
            continue
        fig.add_trace(go.Scatter(x=t, y=traces[col], mode="lines", name=col))
    fig.update_layout(
        title="Example feature traces (highest variance)",
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h"),
    )
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
        title="Feature correlation (subset)",
    )
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def explained_variance_bars(ratios: list[float], *, title: str = "Explained variance") -> go.Figure:
    fig = px.bar(
        x=[f"PC{i+1}" for i in range(len(ratios))],
        y=ratios,
        labels={"x": "Component", "y": "Explained variance ratio"},
        title=title,
    )
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def latent_trajectory_2d(
    latent: np.ndarray,
    *,
    color: np.ndarray | pd.Series | None = None,
    color_label: str = "value",
    title: str = "2D latent trajectory",
) -> go.Figure:
    if latent.ndim != 2 or latent.shape[1] < 2:
        return go.Figure().update_layout(title="Need ≥2 latent dimensions")
    df = pd.DataFrame({"z1": latent[:, 0], "z2": latent[:, 1]})
    if color is not None:
        df[color_label] = np.asarray(color)
        fig = px.scatter(df, x="z1", y="z2", color=color_label, title=title,
                         color_continuous_scale="Viridis")
    else:
        fig = px.scatter(df, x="z1", y="z2", title=title)
    fig.update_traces(marker=dict(size=4))
    fig.update_layout(height=420, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def latent_trajectory_3d(
    latent: np.ndarray,
    *,
    color: np.ndarray | pd.Series | None = None,
    color_label: str = "value",
    title: str = "3D latent trajectory",
) -> go.Figure:
    if latent.ndim != 2 or latent.shape[1] < 3:
        return go.Figure().update_layout(title="Need ≥3 latent dimensions")
    marker: dict[str, Any] = {"size": 3}
    if color is not None:
        marker["color"] = np.asarray(color)
        marker["colorscale"] = "Viridis"
        marker["colorbar"] = {"title": color_label}
    fig = go.Figure(data=[go.Scatter3d(
        x=latent[:, 0], y=latent[:, 1], z=latent[:, 2],
        mode="markers", marker=marker,
    )])
    fig.update_layout(title=title, height=480, margin=dict(l=0, r=0, t=50, b=0))
    return fig


def metric_by_category(
    df: pd.DataFrame,
    *,
    category: str,
    metric: str,
    title: str | None = None,
) -> go.Figure:
    if category not in df.columns or metric not in df.columns:
        return go.Figure().update_layout(title="Missing columns for plot")
    plot_df = df.dropna(subset=[metric])
    fig = px.box(
        plot_df, x=category, y=metric,
        title=title or f"{metric} by {category}",
        points="all",
    )
    fig.update_layout(height=380, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def degradation_or_source_curve(
    df: pd.DataFrame,
    *,
    metric: str,
    group_col: str = "spike_source",
    title: str = "Decoding vs spike source",
) -> go.Figure:
    if group_col not in df.columns or metric not in df.columns:
        return go.Figure().update_layout(
            title="Sorting-robustness curve unavailable (need GT vs sorted metrics)",
        )
    agg = df.groupby(group_col, dropna=False)[metric].mean().reset_index()
    fig = px.bar(agg, x=group_col, y=metric, title=title)
    fig.update_layout(height=340, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def true_vs_decoded_position(decoded: pd.DataFrame) -> go.Figure:
    needed = {"true_x", "true_y", "decoded_x", "decoded_y"}
    if not needed.issubset(decoded.columns):
        return go.Figure().update_layout(title="Position columns missing in replay CSV")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=decoded["true_x"], y=decoded["true_y"],
        mode="lines", name="True", line=dict(width=2),
    ))
    fig.add_trace(go.Scatter(
        x=decoded["decoded_x"], y=decoded["decoded_y"],
        mode="lines", name="Decoded", line=dict(width=2, dash="dash"),
    ))
    fig.update_layout(
        title="True vs decoded trajectory",
        height=420,
        margin=dict(l=40, r=20, t=50, b=40),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig


def error_over_time(decoded: pd.DataFrame) -> go.Figure:
    tcol = "time" if "time" in decoded.columns else "decode_time"
    if "position_error_cm" not in decoded.columns or tcol not in decoded.columns:
        return go.Figure().update_layout(title="Position error time series unavailable")
    fig = px.line(decoded, x=tcol, y="position_error_cm", title="Position error over time")
    fig.update_layout(height=320, margin=dict(l=40, r=20, t=50, b=40))
    return fig


def speed_true_vs_pred(decoded: pd.DataFrame) -> go.Figure:
    cols = [c for c in ("true_speed", "decoded_speed", "speed", "speed_pred") if c in decoded.columns]
    tcol = "time" if "time" in decoded.columns else ("decode_time" if "decode_time" in decoded.columns else None)
    if tcol is None or len(cols) < 2:
        # Try common naming
        true_c = next((c for c in decoded.columns if "speed" in c and "true" in c), None)
        pred_c = next((c for c in decoded.columns if "speed" in c and "decod" in c), None)
        if true_c and pred_c and tcol:
            fig = make_subplots(specs=[[{"secondary_y": False}]])
            fig.add_trace(go.Scatter(x=decoded[tcol], y=decoded[true_c], name="True speed"))
            fig.add_trace(go.Scatter(x=decoded[tcol], y=decoded[pred_c], name="Decoded speed"))
            fig.update_layout(title="Speed: true vs predicted", height=320)
            return fig
        return go.Figure().update_layout(title="Speed columns not found")
    fig = go.Figure()
    for c in cols[:2]:
        fig.add_trace(go.Scatter(x=decoded[tcol], y=decoded[c], name=c, mode="lines"))
    fig.update_layout(title="Speed: true vs predicted", height=320,
                      margin=dict(l=40, r=20, t=50, b=40))
    return fig


def confusion_from_metrics_row(row: pd.Series) -> go.Figure | None:
    cm = row.get("confusion_matrix")
    labels = row.get("class_labels")
    if cm is None or (isinstance(cm, float) and np.isnan(cm)):
        return None
    if isinstance(cm, str):
        import ast
        try:
            cm = ast.literal_eval(cm)
        except (ValueError, SyntaxError):
            return None
    if isinstance(labels, str):
        import ast
        try:
            labels = ast.literal_eval(labels)
        except (ValueError, SyntaxError):
            labels = None
    mat = np.asarray(cm)
    fig = px.imshow(
        mat, x=labels, y=labels, text_auto=True,
        color_continuous_scale="Blues",
        title=f"Confusion — {row.get('target_name', '')}",
    )
    fig.update_layout(height=360, margin=dict(l=40, r=20, t=50, b=40))
    return fig
