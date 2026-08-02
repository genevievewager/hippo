"""Shared simulation data loading for decoder computation scripts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hippo.anatomy.hippocampal_system import (
    annotate_units_for_analysis,
    filter_unit_ids_for_analysis,
)
from realtime.spike_binner import _resolve_spike_columns


def load_simulation_data(
    input_dir: Path,
    spike_source: str,
    *,
    include_non_hippocampal: bool = False,
) -> dict:
    """Load behavior, units, spikes, and summary from a simulation output directory.

    By default only RatInABox hippocampal-system units
    (``include_in_decoder`` / allowlisted cell types) enter ``unit_ids`` used for
    decoding and manifold features. Non-hippocampal probe contaminants are
    dropped unless ``include_non_hippocampal=True``.
    """
    input_dir = Path(input_dir)
    required = ["behavior.csv", "units.csv", "summary.json"]
    for fname in required:
        if not (input_dir / fname).exists():
            raise FileNotFoundError(f"Required file not found: {input_dir / fname}")

    if spike_source == "sorted":
        spike_file = input_dir / "spikes_sorted.csv"
    elif spike_source == "ground_truth":
        spike_file = input_dir / "spikes_ground_truth.csv"
    else:
        raise ValueError(
            f"spike_source must be 'sorted' or 'ground_truth', got {spike_source!r}"
        )
    if not spike_file.exists():
        raise FileNotFoundError(f"Spike file not found: {spike_file}")

    behavior_df = pd.read_csv(input_dir / "behavior.csv")
    units_raw = pd.read_csv(input_dir / "units.csv")
    units_df = annotate_units_for_analysis(
        units_raw, include_non_hippocampal=include_non_hippocampal,
    )
    spikes_df = pd.read_csv(spike_file)
    with open(input_dir / "summary.json") as f:
        summary = json.load(f)

    time_col, _ = _resolve_spike_columns(spikes_df)
    spikes_df = spikes_df.rename(columns={time_col: "time"})
    # count_spikes_in_window uses searchsorted and requires sorted spike times.
    spikes_df = spikes_df.sort_values("time", kind="mergesort").reset_index(drop=True)

    session_duration = summary.get("session_duration_s")
    if session_duration is None:
        session_duration = float(max(
            behavior_df.iloc[:, 0].max(),
            spikes_df["time"].max(),
        ))

    all_ids = sorted(units_df["unit_id"].unique().tolist())
    unit_ids = filter_unit_ids_for_analysis(
        units_df, all_ids, include_non_hippocampal=include_non_hippocampal,
    )
    # Restrict spikes to analysis units so contamination cannot leak in.
    spikes_df = spikes_df[spikes_df["unit_id"].isin(unit_ids)].copy()

    return {
        "behavior_df": behavior_df,
        "units_df": units_df,
        "units_df_all": units_raw,
        "spikes_df": spikes_df,
        "summary": summary,
        "session_duration": float(session_duration),
        "unit_ids": unit_ids,
        "n_units_excluded": int(len(all_ids) - len(unit_ids)),
        "include_non_hippocampal": include_non_hippocampal,
        "spike_source": spike_source,
    }


def make_decode_times(
    session_duration: float,
    decode_window: float,
    update_dt: float,
    *,
    behavior_times: np.ndarray | None = None,
) -> np.ndarray:
    """
    Decoder update timestamps.

    When ``behavior_times`` is provided (preferred), use the original behavioral
    / video frame timestamps at or after ``decode_window`` so every prediction
    corresponds to one behavioral frame. Otherwise fall back to a fixed grid
    with spacing ``update_dt``.
    """
    if update_dt <= 0 or decode_window <= 0:
        raise ValueError("update_dt and decode_window must be positive")
    if behavior_times is not None:
        t = np.asarray(behavior_times, dtype=float)
        decode_times = t[t >= float(decode_window) - 1e-12]
        if len(decode_times) == 0:
            raise ValueError(
                f"No behavioral timestamps >= decode_window ({decode_window})"
            )
        return decode_times

    t_start = decode_window
    t_end = session_duration
    if t_start >= t_end:
        raise ValueError(
            f"decode_window ({decode_window}) must be less than session duration ({t_end})"
        )
    return np.arange(t_start, t_end + 1e-9, update_dt)
