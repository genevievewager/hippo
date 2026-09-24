"""Shared simulation data loading for decoder computation scripts."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hippo.anatomy.hippocampal_system import (
    CANONICAL_REGIONS,
    NON_HIPPOCAMPAL_REGIONS,
    annotate_units_for_analysis,
    filter_unit_ids_for_analysis,
    is_allowed_cell_type,
)
from realtime.pipeline_invariants import assert_times_in_seconds
from realtime.spike_binner import _resolve_spike_columns
from realtime.timing import extract_behavior_times


def load_simulation_data(
    input_dir: Path,
    spike_source: str,
    *,
    include_non_hippocampal: bool = False,
    include_regions: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Load behavior, units, spikes, and summary from a simulation output directory.

    By default only RatInABox hippocampal-system units
    (``include_in_decoder`` / allowlisted cell types) enter ``unit_ids`` used for
    decoding and manifold features. Non-hippocampal probe contaminants are
    dropped unless ``include_non_hippocampal=True``.

    ``include_regions`` is an explicit raw-region allowlist (quadrant_n5).
    When set, a unit enters if its ``region`` is in the list and its cell
    type is allowlisted. Default pipeline behaviour is unchanged when this
    argument is omitted.
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
    # Gate before deriving session length from the times themselves — otherwise
    # sample indices would set a huge session and pass their own check.
    assert_times_in_seconds(
        spikes_df["time"].to_numpy(),
        session_length_s=session_duration,
        context="spike times",
    )
    try:
        behavior_times = extract_behavior_times(behavior_df)
    except ValueError:
        behavior_times = None
    if behavior_times is not None:
        assert_times_in_seconds(
            behavior_times,
            session_length_s=session_duration,
            context="behavior times",
        )
    if session_duration is None:
        session_duration = float(max(
            behavior_df.iloc[:, 0].max(),
            spikes_df["time"].max(),
        ))

    all_ids = sorted(units_df["unit_id"].unique().tolist())
    if include_regions is not None:
        allowed_regions = {str(r) for r in include_regions}
        unit_ids = sorted({
            int(row["unit_id"])
            for _, row in units_df.iterrows()
            if str(row.get("region")) in allowed_regions
            and is_allowed_cell_type(row.get("cell_type"))
        })
    else:
        unit_ids = filter_unit_ids_for_analysis(
            units_df, all_ids, include_non_hippocampal=include_non_hippocampal,
        )
    if include_regions is None and all_ids and unit_ids and len(unit_ids) < len(all_ids):
        # Partial exclusion is legitimate (visual cortex on the way in), but an
        # unrecognised label is not: it drops real units and looks identical to
        # a deliberate exclusion. Name the labels so the drop is a decision.
        dropped = units_df[~units_df["unit_id"].isin(unit_ids)]
        unrecognised = sorted(
            {
                str(r)
                for r in dropped.get("region_canonical", pd.Series(dtype=str))
                if str(r) not in NON_HIPPOCAMPAL_REGIONS
                and str(r) not in CANONICAL_REGIONS
                and str(r) != "unknown"
            }
        )
        if unrecognised:
            warnings.warn(
                f"{len(dropped)} of {len(all_ids)} units were excluded from analysis "
                f"under region labels that canonicalize_region does not recognise: "
                f"{unrecognised}. These are being dropped as non-hippocampal by "
                "default. If they are on the probe track and should be decoded, add "
                "them to hippo.anatomy.hippocampal_system.REGION_ALIASES.",
                UserWarning,
                stacklevel=2,
            )

    if all_ids and not unit_ids:
        # Excluding every unit is never a valid analysis result. Returning an
        # empty frame here reads downstream as a science finding — all-zero
        # features, a decoder trained on nothing, accuracy at chance — instead
        # of as the load failure it is.
        observed = sorted(
            {str(r) for r in units_df.get("region_canonical", pd.Series(dtype=str))}
        ) or ["<no region column>"]
        raise ValueError(
            f"All {len(all_ids)} units were excluded from analysis, leaving nothing "
            f"to decode.\n"
            f"  region_canonical values seen: {observed}\n"
            f"  units.csv columns: {list(units_raw.columns)}\n"
            "Usual causes: the units table carries no region / cell-type columns "
            "(a bare Kilosort or Phy export), or its region labels are not "
            "recognised by hippo.anatomy.hippocampal_system.canonicalize_region. "
            "Add the labels to REGION_ALIASES, supply a region mapping at ingest, "
            "or pass include_non_hippocampal=True if the exclusion is intended."
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
