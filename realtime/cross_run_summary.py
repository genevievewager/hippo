"""Cross-run / cross-seed decoder generalization summaries."""

from __future__ import annotations

from typing import Any

import pandas as pd


def build_cross_run_decoder_summary(
    metrics_by_run: dict[str, pd.DataFrame],
    *,
    pass_gate_col: str = "passes_realtime_gate",
) -> pd.DataFrame:
    """
    Aggregate candidate metrics across runs.

    ``metrics_by_run`` maps run_id → metrics dataframe for that run.
    """
    from realtime.decoder_comparison import PRIMARY_METRIC

    if not metrics_by_run:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for run_id, df in metrics_by_run.items():
        if df is None or df.empty:
            continue
        part = df.copy()
        part["run_id"] = str(run_id)
        frames.append(part)
    if not frames:
        return pd.DataFrame()

    all_df = pd.concat(frames, ignore_index=True)
    if "target_name" in all_df.columns:
        all_df = all_df[all_df["target_name"].notna()].copy()
    if all_df.empty:
        return pd.DataFrame()

    key_cols = [
        c for c in (
            "target_name",
            "feature_type",
            "embedding_type",
            "decoder_name",
            "decode_window_s",
            "manifold_n_components",
            "trigger_rule",
            "spike_source",
        )
        if c in all_df.columns
    ]

    rows: list[dict[str, Any]] = []
    for keys, group in all_df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(key_cols, keys))
        target = str(key_map.get("target_name"))
        if target not in PRIMARY_METRIC:
            continue
        metric_name, direction = PRIMARY_METRIC[target]
        if metric_name not in group.columns:
            continue
        vals = group[metric_name].astype(float).dropna()
        if vals.empty:
            continue
        if pass_gate_col in group.columns:
            frac_pass = float(group[pass_gate_col].fillna(False).astype(bool).mean())
        else:
            frac_pass = float("nan")
        worst = float(vals.max() if direction == "lower" else vals.min())
        rows.append({
            **key_map,
            "primary_metric": metric_name,
            "mean_metric": float(vals.mean()),
            "std_metric": float(vals.std(ddof=0)),
            "worst_run_metric": worst,
            "n_runs": int(vals.shape[0]),
            "fraction_runs_passing_gate": frac_pass,
        })
    return pd.DataFrame(rows)
