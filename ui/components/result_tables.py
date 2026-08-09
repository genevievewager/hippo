"""Result tables / leaderboards."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.services.results import add_score_column, highlight_best_mask, leaderboard


def render_leaderboard(metrics: pd.DataFrame, *, highlight_best: bool = True) -> pd.DataFrame:
    """Show a sortable leaderboard; returns the displayed frame."""
    if metrics is None or metrics.empty:
        st.warning("No metrics rows to display.")
        return pd.DataFrame()

    board = leaderboard(metrics)
    if highlight_best and "target_name" in metrics.columns:
        full = add_score_column(metrics)
        best = highlight_best_mask(full)
        # Align by joining on available identity columns if board is derived
        board = board.copy()
        board.insert(0, "best", False)
        # Mark rows that match best primary score per target
        if "target_name" in board.columns and "score" in board.columns:
            for target, g in board.groupby("target_name"):
                direction = "lower"
                from realtime.decoder_comparison import PRIMARY_METRIC
                if target in PRIMARY_METRIC:
                    direction = PRIMARY_METRIC[target][1]
                if g["score"].isna().all():
                    continue
                idx = g["score"].idxmin() if direction == "lower" else g["score"].idxmax()
                board.loc[idx, "best"] = True

    st.dataframe(board, use_container_width=True, hide_index=True)
    n_best = int(board["best"].sum()) if "best" in board.columns else 0
    st.caption(f"{len(board)} rows · {n_best} best-per-target highlighted.")
    return board


def render_filters(metrics: pd.DataFrame) -> dict:
    """Sidebar-style filter widgets returning selected values."""
    filters: dict = {}
    c1, c2, c3, c4 = st.columns(4)

    def _opts(col: str) -> list:
        if col not in metrics.columns:
            return []
        return sorted({v for v in metrics[col].dropna().unique().tolist()})

    with c1:
        filters["targets"] = st.multiselect("Target", _opts("target_name"))
        filters["feature_sets"] = st.multiselect("Feature set", _opts("feature_set"))
    with c2:
        man_col = "embedding_type" if "embedding_type" in metrics.columns else "manifold"
        filters["manifolds"] = st.multiselect("Manifold", _opts(man_col))
        filters["decoders"] = st.multiselect("Decoder", _opts("decoder_name"))
    with c3:
        wins = _opts("decode_window_s")
        filters["decode_windows"] = st.multiselect(
            "Decode window (s)",
            [float(w) for w in wins],
            format_func=lambda w: f"{float(w)*1000:.0f} ms" if float(w) < 1 else "1 s",
        )
        filters["spike_sources"] = st.multiselect("Spike source", _opts("spike_source"))
    with c4:
        st.caption("Filters apply to the leaderboard below.")
    return filters
