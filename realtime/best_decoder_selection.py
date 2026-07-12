"""Select best decoder/window from decoder comparison outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_best_decoder_table(
    comparison_dir: Path,
    spike_source: str,
) -> tuple[pd.DataFrame, Path]:
    """Load best_decoder_by_target.csv for a spike source."""
    comparison_dir = Path(comparison_dir)
    candidates = [
        comparison_dir / spike_source / "best_decoder_by_target.csv",
        comparison_dir / "best_decoder_by_target.csv",
    ]
    for path in candidates:
        if path.exists():
            df = pd.read_csv(path)
            if "spike_source" in df.columns:
                filtered = df[df["spike_source"] == spike_source]
                if not filtered.empty:
                    return filtered.reset_index(drop=True), path
            return df.reset_index(drop=True), path
    raise FileNotFoundError(
        f"Could not find best_decoder_by_target.csv under {comparison_dir} "
        f"(looked for {spike_source}/best_decoder_by_target.csv and best_decoder_by_target.csv)"
    )


def select_best_decoder_row(
    comparison_dir: Path,
    spike_source: str,
    closed_loop_target: str,
    selection_policy: str = "shortest_near_optimal",
) -> tuple[dict[str, Any], Path]:
    """Return the selected best-decoder row and source file path."""
    df, path = resolve_best_decoder_table(comparison_dir, spike_source)
    target_df = df[df["target_name"] == closed_loop_target]
    if target_df.empty:
        available = sorted(df["target_name"].unique().tolist())
        raise ValueError(
            f"Target {closed_loop_target!r} not found in {path}. "
            f"Available targets: {available}"
        )
    row = target_df.iloc[0].to_dict()

    if selection_policy == "best_accuracy":
        selected_window = float(row["best_decode_window_s"])
    elif selection_policy == "shortest_near_optimal":
        selected_window = float(row["recommended_realtime_window_s"])
    else:
        raise ValueError(
            f"Unknown selection_policy {selection_policy!r}; "
            "use 'best_accuracy' or 'shortest_near_optimal'"
        )

    decoder_config = row.get("decoder_config_json", "{}")
    if isinstance(decoder_config, str):
        try:
            decoder_config = json.loads(decoder_config)
        except json.JSONDecodeError:
            decoder_config = {}

    selected = {
        **row,
        "selection_policy": selection_policy,
        "selected_decoder_name": row["best_decoder_name"],
        "selected_decode_window_s": selected_window,
        "selected_feature_type": row.get("best_feature_type", "counts"),
        "decoder_config": decoder_config,
        "from_file": str(path),
    }
    return selected, path
